import sys
sys.path.append(".")
import cv2
import open3d as o3d
from math import pi
from tools.pc2image import pointcloud_to_image

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("data/liujiao1/leica6SE.pts")
    
    image, intrinsics = pointcloud_to_image(pcd, 800, 800, pi/2, pi/2, 0, 0, -pi/3, kernel_size=3)
    print('Camera Intrinsics: \n', intrinsics)
    cv2.imwrite("leica_projection.jpg", image)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()