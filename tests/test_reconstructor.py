from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pycolmap

from src.reconstructor import Reconstructor


class ReconstructorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "background_removed"
        self.source.mkdir()
        self.output = self.root / "reconstruction"

    def make_image(self, camera="cam_0_0", filename="0.png", size=(4, 4)):
        path = self.source / camera / filename
        path.parent.mkdir(exist_ok=True)
        image = Image.new("RGBA", size, (90, 120, 150, 255))
        image.putpixel((0, 0), (90, 120, 150, 0))
        image.putpixel((1, 0), (90, 120, 150, 127))
        image.putpixel((2, 0), (90, 120, 150, 128))
        image.save(path)
        return path

    def test_empty_input_does_not_create_output(self):
        with self.assertRaisesRegex(ValueError, "No PNG images"):
            Reconstructor(self.output).reconstruct_from_folder(self.source)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.make_image()
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            Reconstructor(self.output).reconstruct_from_folder(self.source)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_missing_alpha_is_rejected(self):
        path = self.make_image()
        Image.new("RGB", (4, 4)).save(path)
        with self.assertRaisesRegex(ValueError, "alpha channel"):
            Reconstructor(self.output).reconstruct_from_folder(self.source)

    def test_empty_foreground_is_rejected(self):
        path = self.make_image()
        Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(path)
        with self.assertRaisesRegex(ValueError, "mask is empty"):
            Reconstructor(self.output).reconstruct_from_folder(self.source)

    def test_changing_camera_dimensions_is_rejected(self):
        self.make_image()
        self.make_image(filename="1.png", size=(8, 4))
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            Reconstructor(self.output).reconstruct_from_folder(self.source)

    def test_dense_on_cpu_build_fails_before_creating_output(self):
        self.make_image()
        with patch.object(pycolmap, "has_cuda", False):
            with self.assertRaisesRegex(RuntimeError, "GPU-enabled"):
                Reconstructor(self.output).reconstruct_from_folder(self.source, dense=True)
        self.assertFalse(self.output.exists())

    def test_five_camera_masks_and_rgb_preserve_alignment_on_solver_failure(self):
        camera_names = ("cam_0_0", "cam_0_1", "cam_1_0", "cam_1_1", "cam_2_0")
        originals = {}
        for camera in camera_names:
            for frame in ("0.png", "1.png"):
                path = self.make_image(camera, frame)
                originals[path] = path.read_bytes()

        # Isolate the image preparation contract from SfM's texture requirements.
        with (
            patch.object(pycolmap, "extract_features"),
            patch.object(pycolmap, "match_exhaustive"),
            patch.object(pycolmap, "incremental_mapping", return_value={}),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not reconstruct"):
                Reconstructor(self.output).reconstruct_from_folder(self.source)

        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)
            relative = path.relative_to(self.source)
            mask_path = self.output / "masks" / relative.parent / (relative.name + ".png")
            with Image.open(mask_path) as mask:
                self.assertEqual(mask.size, (4, 4))
                self.assertEqual(mask.mode, "L")
                self.assertEqual([mask.getpixel((x, 0)) for x in range(4)], [0, 0, 255, 255])
            with Image.open(self.output / "images" / relative) as rgb:
                self.assertEqual(rgb.size, (4, 4))
                self.assertEqual(rgb.mode, "RGB")
                self.assertEqual(rgb.getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(rgb.getpixel((2, 0)), (90, 120, 150))


if __name__ == "__main__":
    unittest.main()
