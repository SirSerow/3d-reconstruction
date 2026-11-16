# Turntable 3D Reconstruction

This repository is for reconstructing a physical object from a full turn of a five-camera capture rig. The planned pipeline produces a 3D point cloud and a triangle mesh for geometry, then uses the posed photos and a colored point cloud to train a Gaussian splat for novel-view rendering.

The repository currently contains the reconstruction plan. A local sample dataset is available under `.input/sample_1/`. The `.input/` directory is ignored by Git.

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
- Create a binary object mask for each view. Remove the platform, scratches, walls, and shadows. If empty-platform reference images are available, compare matching angles when the platform has visible markings. Inspect dark object surfaces, thin features, and the contact edge before processing the full sequence.
- Apply the same crop, resize, and undistortion to the image and its mask. Use the mask during feature extraction to keep the matches on the object. [COLMAP masking](https://colmap.github.io/faq.html#mask-image-regions)

### 2. Estimate camera poses relative to the object

From the object's perspective, the five cameras move around it as the platform turns. Each tile needs a camera pose in this object-fixed coordinate system.

- **Existing captures.** If calibration and angles are unavailable, run masked structure from motion in COLMAP. Share intrinsics across images from each physical camera. Inspect the side-camera orbits and check that the overhead poses align with the same reconstruction. Use a known dimension if physical scale is required. [COLMAP tutorial](https://colmap.github.io/tutorial.html)
- **Calibrated rig.** Use a measured ChArUco target to calibrate intrinsics and lens distortion for each C920 at the capture settings. Determine the five fixed camera poses and the turntable axis in one rig coordinate system. Compose each fixed rig-to-camera transform with the object-to-rig rotation about that axis. Use recorded angles or recover them from visible turntable markers when possible. This route is useful when smooth or repetitive surfaces make feature matching unreliable. [OpenCV ChArUco calibration](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html) · [COLMAP workflow for known poses](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)

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
