import laspy 
import numpy as np 
import open3d as o3d
from scipy.spatial import KDTree


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def icp(source, target, initial_tranformation,max_correspondence_distance, estimation_type):

   
    if estimation_type == 'point2point':
        registration_icp =o3d.pipelines.registration.registration_icp(source, target,max_correspondence_distance,initial_tranformation,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint())
    if estimation_type == 'point2plane':
        # Estimate normals for point-to-plane ICP
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        registration_icp =o3d.pipelines.registration.registration_icp(source,target, max_correspondence_distance,initial_tranformation,
                                                                    o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print("Inlier Fitness: ", registration_icp.fitness)
    print("Inlier RMSE: ", registration_icp.inlier_rmse)
    print(registration_icp.transformation)

    return registration_icp 


def voxelization(pcd,N, voxel_size):

    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center())
    
    if  pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))

    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=voxel_size)

    return voxel_grid

def shift_evalutaion(original, registered): 
    ### Let's compare the coordinates of the registered shifted point cloud in stable parts to the original:
    #### https://github.com/3dgeo-heidelberg/py4dgeo/blob/main/demo/registration_standard_ICP.ipynb

        # Build kd-tree from 3D coordinates
    tree_orig = KDTree(original)

    # Query indices of nearest neighbors of registered coordinates
    nn_dists = tree_orig.query(registered, k=1)

    # Obtain distances as first element in tuple returned by query above
    distances = nn_dists[0]

    print(f"Mean dist: {np.mean(distances):.3f} m")
    print(f"Median dist: {np.median(distances):.3f} m")
    print(f"Std.dev. of dists: {np.std(distances):.3f} m")