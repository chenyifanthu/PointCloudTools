# examples/python/visualization/interactive_visualization.py

import numpy as np
import copy
import open3d as o3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def registeration_manual(source, target, voxel_size):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source_down)
    picked_id_target = pick_points(target_down)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target
    
    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source_down, target_down,
                                            o3d.utility.Vector2iVector(corr))
    
    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size*0.4, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")
    


if __name__ == "__main__":
    # demo_crop_geometry()
    leica = o3d.io.read_point_cloud('./data/volleyball/leica.pts')
    livox = o3d.io.read_point_cloud('./data/volleyball/livox_1.pcd')
    registeration_manual(leica, livox, 0.05)