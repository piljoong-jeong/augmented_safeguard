import copy
import os
import sys


import numpy as np
import open3d as o3d

# read image
# point cloud => src
# noise
# apply transformation => dst

SEED = 0
np.random.seed(SEED)

def pointcloud_from_rgbd(dir_color, dir_depth):

    color = o3d.io.read_image(dir_color)
    depth = o3d.io.read_image(dir_depth)

    rgbd = o3d.geometry.PointCloud.create_from_color_and_depth(
        color, depth, 
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsics(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))


    return pcd


def add_noise_to_depth(o3d_depth: o3d.geometry.Image, sigma: float):

    noise = np.random.normal(0, sigma, np.asarray(o3d_depth).shape)
    np_depth_with_noise = copy.deepcopy(np.asarray(o3d_depth)) + noise
    depth_with_noise = o3d.geometry.Image()

    return NotImplementedError

def add_noise_to_point(o3d_pcd: o3d.geometry.PointCloud, sigma: float):

    
    noise = np.random.normal(0, sigma, np.asarray(o3d_pcd.points).shape)
    points_with_noise = np.asarray(o3d_pcd.points) + noise
    pc_with_noise = copy.deepcopy(o3d_pcd)
    pc_with_noise.points = o3d.utility.Vector3dVector(points_with_noise)

    return