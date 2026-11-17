# Turntable 3D Reconstruction

This repository is for reconstructing a physical object from a full turn of a five-camera capture rig. The planned pipeline produces a 3D point cloud and a triangle mesh for geometry, then uses the posed photos and a colored point cloud to train a Gaussian splat for novel-view rendering.

The repository includes image splitting, background removal, and PyCOLMAP reconstruction classes, along with the reconstruction plan below. A local sample dataset is available under `.input/sample_1/`. The `.input/` directory is ignored by Git.

## Running the pipeline

Install the dependencies from `requirements_cpu.txt` for CPU processing. For an NVIDIA GPU machine, use x86_64 Linux or WSL2 with Python 3.11 or 3.12 and a working NVIDIA driver; confirm the GPU is visible with `nvidia-smi` inside that environment. Create a fresh virtual environment and install with `python -m pip install -r requirements_gpu.txt`. Install only one requirements file per environment: the CPU and CUDA PyCOLMAP packages both provide `import pycolmap`, and the CPU and GPU ONNX Runtime packages also share an import namespace.

The GPU requirements select `pycolmap-cuda12==4.2.1` and the CUDA 12-compatible `onnxruntime-gpu[cuda,cudnn]==1.23.2` for GPU reconstruction and background removal. The ONNX Runtime extras install its CUDA and cuDNN libraries. The official PyCOLMAP CUDA wheels support Linux; native Windows requires a CUDA source build for GPU reconstruction. [PyCOLMAP CUDA package](https://pypi.org/project/pycolmap-cuda12/) · [ONNX Runtime CUDA libraries](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls)

Run `src/main.py` with `--input` pointing to a composite-image or video folder and `--output` pointing to the output root. Each run creates a timestamped folder with `split`, `background_removed`, and `reconstruction` subfolders.

`Reconstructor` in `src/reconstructor.py` accepts the background-removed PNG folders through `reconstruct_from_folder`. Each camera needs its own subfolder, such as `cam_0_0`, containing images with a foreground alpha channel and consistent dimensions. It generates RGB images and binary masks, shares lens parameters per camera folder, matches across all cameras, and estimates camera parameters and poses without prior calibration. Keep focus, zoom, and crop fixed within each camera's images. Use a new reconstruction output directory for every attempt.

The default output is `reconstruction/sparse_object.ply`, with the estimated cameras and poses in `reconstruction/best_model`. Separate reconstruction components are retained under `reconstruction/sparse`; warnings identify disconnected models or images missing from the selected model. Inspect the sparse result before using it downstream.

Add `--dense` to also create `reconstruction/dense/fused.ply` and `reconstruction/dense/mesh.ply`. This requires the GPU environment described above. Add `--device cuda` to use the GPU for PyCOLMAP feature extraction and matching too; its default is `cpu`, and `auto` selects an available device. Feature masks do not automatically filter dense geometry, so inspect the dense cloud for outliers before accepting the mesh. [PyCOLMAP installation and GPU support](https://pypi.org/project/pycolmap/)

## Capture setup and input format

A rotating platform carries the object while five Logitech C920 webcams stay fixed. Four cameras view the platform from the sides of a square enclosure, and one looks down from above. Each JPEG is a composite of the five views captured at the same nominal turntable angle. The tile order stays fixed across the sequence.

| Row | Left | Right |
| --- | --- | --- |
| Top | Side camera 1 | Side camera 2 |
| Middle | Side camera 3 | Side camera 4 |
| Bottom | Overhead camera | Unused tile, may contain a label |

The checked sample composite is 1600 × 1800 pixels, with 800 × 600 pixel tiles. The dark platform is background. Its scratches, seams, and shadows can still appear in the photos.

## Reconstruction plan

### 1. Extract views and foreground masks

- Split each composite into five images and discard the unused tile. Retain the camera ID, capture ID, and any recorded timestamp or angle so the five views can be grouped again.
- Start with a few views from each camera at different turntable angles. Open each image in an editor with selection and mask tools. Select the object using an automatic foreground selection or by tracing its outline manually, then refine the selection at high zoom.
- Create a binary mask from the selection: white for the object and black for everything else. Exclude the platform, scratches, walls, cast shadows, and reflections on the platform. Keep dark or shaded surfaces that belong to the object; selecting by brightness alone can remove these by mistake.
- If empty-platform reference images are available, use the differences from each object image to help identify the foreground. Keep the camera position, exposure, lighting, and framing consistent, and compare matching turntable angles when the platform has visible markings. Review the result because shadows and lighting changes can also appear as differences.
- Refine the mask around thin features, recesses, and the contact edge. Keep narrow protrusions and mark visible background through holes or gaps as black. Remove isolated white patches in the background and fill accidental black holes on the object's surface. Avoid smoothing or shrinking the outline enough to erase real details.
- Save a separate, lossless PNG mask for every extracted view, using only black and white pixels. Keep each mask paired with its source image. Apply the same crop, resize, and undistortion to both so their dimensions and pixel positions match; use nearest-neighbor resampling for masks to preserve binary values.
- Overlay the masks on the photos to check alignment and silhouettes across all five cameras and several angles before processing the full sequence. Correct individual frames where the outline changes incorrectly, then use the masks during feature extraction to keep matches on the object. [COLMAP masking](https://colmap.github.io/faq.html#mask-image-regions)

### 2. Estimate camera poses relative to the object

From the object's perspective, the five cameras move around it as the platform turns. Each tile needs a camera pose in this object-fixed coordinate system.

#### Existing captures without calibration or recorded angles

Use masked structure from motion in COLMAP to estimate poses from recognizable details on the object. Recorded turntable angles are not required for this route.

1. **Prepare the image groups.** Put the extracted views in one project, with a separate folder for each physical camera and consistent capture IDs. Include sharp images around the full turn. Supply the foreground masks during feature extraction so the stationary enclosure and rotating platform markings do not drive the reconstruction.
2. **Assign camera parameters.** Share one set of intrinsics, meaning focal lengths, principal point, and lens distortion, across images from each physical camera. Keep five separate sets even though the webcams are the same model. For uncalibrated, distorted images, start with a simple model such as `SIMPLE_RADIAL`; use a suitable calibrated model if lens parameters are already available. [COLMAP camera settings](https://colmap.github.io/faq.html#share-intrinsics)
3. **Extract and match features.** For the first small validation set, use exhaustive matching to find overlap both around the turn and between cameras. If matching a larger set selectively, include neighboring angles, the start/end of the full turn, and overlapping views from different cameras. Matching only within each camera folder can leave separate reconstructions.
4. **Reconstruct and inspect.** Run sparse reconstruction and bundle adjustment, which jointly refine the poses and sparse object points. Check that the images form one connected model, the side-camera poses follow plausible orbits, and the overhead images register to the same object. Inspect matches and mask boundaries if poses jump, points form duplicate surfaces, or cameras remain unregistered. [COLMAP reconstruction workflow](https://colmap.github.io/tutorial.html)
5. **Set scale and save.** SfM has arbitrary scale: use a known object dimension or a measured camera separation matched to reconstructed camera centers from the same capture group if metric output is needed. Export the camera intrinsics, per-image poses, and sparse cloud together. Smooth, shiny, or repetitive objects may not provide enough reliable matches; adding masks alone cannot resolve that. Use the calibrated route for new captures when this happens.

#### Calibrated rig with an ArUco or ChArUco board

Calibrate three things separately: each camera's **intrinsics**, the five cameras' fixed **extrinsics** (positions and orientations in one rig frame), and the **turntable axis and rotation angles**. Lens calibration alone does not determine the rig geometry. The following is a proposed procedure for this five-camera layout.

1. **Prepare a measured target.** Use an ArUco GridBoard, or preferably a ChArUco board, which combines ArUco IDs with chessboard corners for more accurate calibration. Print at actual size on matte paper and mount it flat on a rigid backing. Measure the printed dimensions rather than trusting the printer scale. Save the dictionary, marker IDs, board layout, marker lengths, and marker gaps or chessboard square lengths in one consistent unit, such as millimeters. Make the markers large enough to remain clearly readable in the 800 × 600 camera tiles. [OpenCV board selection and calibration](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)
2. **Calibrate each camera's lens.** Keep the cameras mounted, set the capture resolution, and lock focus and zoom. Move the board through the object volume, varying its distance and tilt and covering the image center and edges. As a starting capture plan, collect roughly 20–30 sharp, varied views per camera; a stack of nearly identical front-facing views is not sufficient. Detect the board corners and fit intrinsics and distortion separately for each camera with OpenCV. Inspect per-image reprojection errors and corner overlays, then check several unused board images. Reprojection error measures the pixel distance between detected corners and their predicted locations. [OpenCV camera calibration](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)
3. **Connect the cameras with shared target observations.** Stop the turntable and hold the board still for each grouped capture. Place and tilt it so neighboring cameras can observe it together, repeating at several positions until every camera, including the overhead one, is connected through shared observations. A single flat board usually cannot face all four opposing side cameras at once. If overlap is insufficient, use a rigid target with markers on several faces and measured corner positions in one target coordinate system. Separate boards with unknown relative positions do not establish that connection. [OpenCV ArUco boards and 3D layouts](https://docs.opencv.org/4.x/db/da9/tutorial_aruco_board_detection.html)
4. **Solve the fixed rig geometry.** With intrinsics fixed, estimate the board-to-camera pose from each observation using OpenCV's board corner correspondences and pose solver. Choose one camera as the rig reference. For each shared board placement, combine one board-to-camera pose with the inverse of the other to obtain the relative camera transform. Connect those transforms across the rig, then jointly refine all fixed camera poses and board placements against the detected corners. Do not treat board poses from different placements as the same rig frame. Save one rig-to-camera transform per camera and validate it on additional shared board captures. [OpenCV board pose estimation](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
5. **Measure the turntable axis.** Fasten the target rigidly to the platform and capture a full rotation at many stopped positions, keeping it visible to at least one calibrated camera. From each board pose and the fixed camera transform, express the same board corners in rig coordinates. Track corners away from the axis, fit their circular paths in 3D, and fit a common axis through the circle centers with its direction perpendicular to the circle planes. Choose an origin on that axis and record its height relative to the platform. This measures axis position and tilt instead of assuming the platform center or overhead optical axis is exact. Large departures from circular paths indicate target movement, platform wobble, or calibration errors.
6. **Establish zero and record angles.** Define a repeatable zero position and rotation direction. Use encoder readings or verified motor steps, or estimate rotation from the rigid target's changing pose. For object captures, retain visible platform markers outside the object mask if they are needed to measure angles; use them for angle estimation and exclude them from object feature matching. Stop and settle before each capture group so the five cameras observe the same angle. A separate calibration sweep does not recover unknown angles in old, unmarked captures.
7. **Generate object-relative poses.** Define the object frame as attached to the platform at zero angle. For each capture, first transform object points into the rig frame using that capture's rotation about the measured axis, then transform them into each camera with its fixed extrinsics. Include translation around the measured axis; rotating orientations alone is insufficient. Export these object-to-camera transforms for COLMAP, which stores world-to-camera poses. Invert them for a consumer that expects camera-to-world poses. [COLMAP pose conventions](https://colmap.github.io/format.html#images-txt)
8. **Validate and pass the poses downstream.** Reproject measured target corners in held-out captures from all five cameras and check several rotation angles. Save intrinsics, distortion, image dimensions, fixed extrinsics, axis, zero convention, and per-capture angles. Keep calibration matched to the actual crop and resize. If images are undistorted, use the resulting intrinsics and remove the original distortion from their camera model. Convert OpenCV intrinsics to COLMAP's pixel convention and compatible distortion model when importing. Build a COLMAP model with the known poses, extract and match masked features, and triangulate a seed cloud without re-estimating the calibrated poses. [OpenCV calibration import](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools) · [COLMAP workflow for known poses](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)

Repeat the affected calibration when camera mounts, focus, zoom, or capture geometry change. Calibration taken later applies to existing captures only if the relevant geometry and camera settings are unchanged.

### 3. Build a point cloud and triangle mesh

- **Textured surfaces.** Triangulate sparse points in COLMAP, estimate depth maps with PatchMatch stereo, and fuse them into a dense cloud such as `fused.ply`. Use the aligned masks during fusion or filter the cloud against the masks. Clean the cloud and build a triangle mesh with Poisson or Delaunay reconstruction. [COLMAP dense reconstruction](https://colmap.github.io/tutorial.html)
- **Low-texture surfaces.** Build a visual hull by carving voxels that project outside reliable object silhouettes. Extract the surface with Marching Cubes and sample it to create a point cloud. A visual hull fills in recesses that never affect a silhouette. [Visual hull paper](https://ieeexplore.ieee.org/document/273735)

The overhead camera covers the top. If it is centered on the rotation axis, successive overhead frames add little stereo baseline. The platform hides the contact surface. To measure the underside, capture a second pass with the object flipped or supported differently and align the passes using overlapping surfaces.

### 4. Train and export a Gaussian splat

- Train Nerfstudio Splatfacto with the same posed images and a clean, colored seed cloud. Use reliable sparse COLMAP points, a downsampled dense cloud, or colored samples from the visual hull. Keep the cloud and poses in the same coordinate system. A native Nerfstudio dataset can reference the seed cloud through `ply_file_path`. [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) · [Nerfstudio point cloud loader](https://docs.nerf.studio/_modules/nerfstudio/data/dataparsers/nerfstudio_dataparser.html)
- For the first run, composite the object onto pure black, render against black, and include the background in the image loss so stray splats outside the silhouette receive a penalty. Export a Gaussian `.ply` for rendering and a separate triangle mesh for workflows that need an explicit surface. [Splatfacto training loss](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py) · [Splatfacto export](https://docs.nerf.studio/nerfology/methods/splat.html)

## First validation run

1. Select roughly 36–72 sharp composites spread around the full turn. Check mask overlays from all five cameras.
2. Reproject the initial cloud or mesh into several views. If the outline drifts with rotation or disagrees across cameras, check the tile mapping, camera pose conventions, and turntable axis before training.
3. Hold out several whole capture groups from splat training. Compare renders from their poses against the held-out photos, paying particular attention to silhouettes, recesses, and fine details.
