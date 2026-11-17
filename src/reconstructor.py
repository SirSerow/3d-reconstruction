import logging
from pathlib import Path

from PIL import Image
import pycolmap


class Reconstructor:
    """Reconstruct an object from folders of background-removed camera PNGs.

    Each immediate subfolder represents one physical camera with fixed focus,
    zoom, image dimensions, and crop. PNGs must contain an alpha channel.
    """

    def __init__(
        self,
        output_dir=".output/reconstruction",
        camera_model="SIMPLE_RADIAL",
        device="cpu",
        mask_threshold=128,
    ):
        if not 1 <= mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 1 and 255")

        self.output_dir = Path(output_dir).resolve()
        self.camera_model = camera_model
        self.device = pycolmap.Device(device)
        self.mask_threshold = mask_threshold
        self.logger = logging.getLogger(__name__)


    def reconstruct_from_folder(self, input_folder, dense=False):
        """Return output paths after sparse reconstruction and optional meshing.

        The output directory must be new. Existing runs are never overwritten.
        Dense stereo requires a GPU-enabled PyCOLMAP build and a supported GPU.
        """

        input_path = Path(input_folder).resolve()

        if not input_path.is_dir():
            raise FileNotFoundError(f"Camera image folder does not exist: {input_path}")

        
        if dense and not pycolmap.has_cuda:
            raise RuntimeError(
                "Dense reconstruction requires a GPU-enabled PyCOLMAP build. "
                "The standard Windows pip wheel supports sparse reconstruction; "
                "omit --dense or install a GPU-enabled build."
            )

        files = sorted(
            path
            for camera in input_path.iterdir()
            if camera.is_dir()
            for path in camera.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )

        if not files:
            raise ValueError(f"No PNG images found in camera subfolders of {input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=False)

        images = self.output_dir / "images"
        masks = self.output_dir / "masks"

        self._prepare_images(input_path, files, images, masks)

        database = self.output_dir / "database.db"
        sparse = self.output_dir / "sparse"

        sparse.mkdir()

        reader = pycolmap.ImageReaderOptions()

        reader.camera_model = self.camera_model

        reader.mask_path = str(masks)

        self.logger.info("Extracting object features from %d images", len(files))

        pycolmap.extract_features(
            database_path=database,
            image_path=images,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            reader_options=reader,
            device=self.device,
        )

        self.logger.info("Matching features across all camera folders")

        pycolmap.match_exhaustive(database_path=database, device=self.device)

        self.logger.info("Estimating camera parameters and sparse object geometry")

        models = pycolmap.incremental_mapping(
            database_path=database,
            image_path=images,
            output_path=sparse,
        )

        if not models:
            raise RuntimeError(
                "COLMAP could not reconstruct the object. "
                "Check surface texture, image overlap, sharpness, and masks. "
                f"Prepared images and the database are preserved in {self.output_dir}."
            )

        best = max(models.values(), key=lambda model: model.num_reg_images())
        
        if len(models) > 1:
            self.logger.warning(
                "COLMAP produced %d disconnected models. Exporting the largest; "
                "all components remain in %s", len(models), sparse
            )

        if best.num_reg_images() < len(files):
            self.logger.warning(
                "Only %d of %d images registered in the selected model; "
                "inspect camera coverage before using the result",
                best.num_reg_images(), len(files),
            )

        model_path = self.output_dir / "best_model"
        model_path.mkdir()
        best.write(model_path)
        sparse_cloud = self.output_dir / "sparse_object.ply"
        best.export_PLY(sparse_cloud)
        self.logger.info("%s", best.summary())

        result = {
            "model_path": model_path,
            "sparse_cloud_path": sparse_cloud,
            "dense_cloud_path": None,
            "mesh_path": None,
        }

        if dense:
            result.update(self._reconstruct_dense(images, model_path))

        return result


    def _prepare_images(self, input_path, files, images, masks):
        camera_sizes = {}
        for path in files:
            relative = path.relative_to(input_path)
            image_path = images / relative
            # COLMAP appends .png to the entire image filename, including .png.
            mask_path = masks / relative.parent / (relative.name + ".png")

            with Image.open(path) as image:
                if "A" not in image.getbands():
                    raise ValueError(f"PNG must contain a foreground alpha channel: {path}")
                expected_size = camera_sizes.setdefault(relative.parent, image.size)
                if image.size != expected_size:
                    raise ValueError(
                        f"Images from {relative.parent} must have the same dimensions: "
                        f"expected {expected_size}, got {image.size} in {path}"
                    )

                rgba = image.convert("RGBA")
                mask = rgba.getchannel("A").point(
                    lambda value: 255 if value >= self.mask_threshold else 0
                )
                if mask.getbbox() is None:
                    raise ValueError(f"Foreground mask is empty: {path}")
                rgb = Image.new("RGB", rgba.size, "black")
                rgb.paste(rgba.convert("RGB"), (0, 0), mask)

                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(image_path)
                mask.save(mask_path)


    def _reconstruct_dense(self, images, model_path):
        dense = self.output_dir / "dense"
        self.logger.info("Undistorting images and estimating dense depth maps")
        pycolmap.undistort_images(
            output_path=dense, input_path=model_path, image_path=images,
        )
        pycolmap.patch_match_stereo(workspace_path=dense)

        dense_cloud = dense / "fused.ply"
        pycolmap.stereo_fusion(
            output_path=dense_cloud, workspace_path=dense, output_type="ply",
        )
        mesh = dense / "mesh.ply"
        self.logger.info("Reconstructing triangle mesh")
        pycolmap.poisson_meshing(input_path=dense_cloud, output_path=mesh)
        return {"dense_cloud_path": dense_cloud, "mesh_path": mesh}


if __name__ == "__main__":
    folder_path = ".input"
    output_folder = ".output/reconstruction"
    reconstructor = Reconstructor(output_dir=output_folder, device="cuda")
    result = reconstructor.reconstruct_from_folder(folder_path, True)

    print(result)
