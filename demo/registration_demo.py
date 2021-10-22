import sys
sys.path.append(".")
from preprocess.read_data import read_point_cloud
from registration.registration import register_two_pointclouds


if __name__ == '__main__':
    source = read_point_cloud(r"C:\Users\win10\Desktop\data\20210923\DCNW.ply")
    target = read_point_cloud(r"C:\Users\win10\Desktop\data\20210923\DCE.ply")
    result = register_two_pointclouds(source, target)
    print(result)