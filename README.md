# PointCloudTools
Some useful tools to processing point cloud data

## Registration
- We refer to the global registration [code](http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html) from open3d tutorial
- This is a coarse-to-fine algorithm that register two point clouds without initial alignment.
- With this registration algorithm, we reconstruct **East playground** of Tsinghua University using four LIDAR (left image) and localize another two LIDAR in this scene (right image).
<div align="center">
<img src="doc/playground-1.png" alt="test" height="300" align="middle" />
<img src="doc/playground-2.png" alt="test" height="300" align="middle" />
</div>


## Move Detect
- We used lidar to collect point cloud data for about 60 seconds at the entrance of the Tsinghua **Central Main Building**, where the lidar was moved three times. 
- We divide the data into several sub-point clouds at an interval of 20 frames. 
- We detect the moving points (red points in the figure) between two adjacent sub-point clouds, and calculate the similarity score.
<div align="center">
<img src="doc/move_detect_demo.GIF" alt="test" width="600" align="middle" />
</div>

## Pointcloud to Image
- This helps to generate an image from a point cloud with a specific perspective.
- The principle of this algorithm is to project the color points onto an imaging plane perpendicular to the x axis.
- And can mask out the pixels that are not projected on the image.
<div align="center">
<img src="doc/pc2image_demo_pc.png" alt="test" height="300" align="middle" />
<img src="doc/pc2image_demo_image.jpg" alt="test" height="300" align="middle" />
</div>