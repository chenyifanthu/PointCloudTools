# PointCloudTools.preprocess

## read_leica_data
能够利用python快速读取Leica三维激光扫描仪得到的点云数据，并转换成`open3d.geometry.PointCloud`格式。

1、在`python=3.6`环境下安装相关库

    pip install open3d==0.13.0
    conda install xerces-c
    pip install pye57

2、ctrl+左键单击代码中的`pye57.E57`，进入`pye57>e57.py`的库文件，找到line19的`SUPPORTED_POINT_FIELDS`字典，在最后添加几个key-value对。

    SUPPORTED_POINT_FIELDS = {

        ........

        "sphericalRange": "d",
        "sphericalAzimuth": "d",
        "sphericalElevation": "d",
        "sphericalInvalidState": "b"
    }

