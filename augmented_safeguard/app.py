"""
### app.py

maintain entrypoints
"""

import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd
import seaborn

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import augmented_safeguard as asfgd
from augmented_safeguard.dataset import DatasetType, DatasetManager


def run_default():
    """
    ### run_default


    """

    dir_dataset = "/mnt/d/"
    type_dataset = DatasetType.RIO10
    name_train_sequence = "seq01_01"
    name_test_sequence = "seq01_02"

    dataset_manager = DatasetManager(DatasetType.RIO10, dir_dataset,  name_train_sequence, name_test_sequence)

    intrinsics_train, intrinsics_test = dataset_manager.read_intrinsics() # NOTE: for scene01, identical
    dataset_train, dataset_test = dataset_manager.read_data(is_NeuralRouting_normalized=True)
    IDX_TRAIN = 277
    IDX_TEST = 817

    pose_train = np.loadtxt(dataset_train.poses[IDX_TRAIN])
    pose_test = np.loadtxt(dataset_test.poses[IDX_TEST])
    print(f"train #{IDX_TRAIN}: \n{pose_train}")
    print(f"test  #{IDX_TEST}: \n{pose_test}")

    # TODO: point cloud fuse & visualize them to see if they're aligned
    pcd_train = dataset_manager.pointcloud_from_rgbd(dataset_train.colors[IDX_TRAIN], dataset_train.depths[IDX_TRAIN], intrinsics_train)
    pcd_test = dataset_manager.pointcloud_from_rgbd(dataset_test.colors[IDX_TEST], dataset_test.depths[IDX_TEST], intrinsics_train)
    pcd_train = pcd_train.transform(pose_train)
    pcd_test = pcd_test.transform(pose_test)
    o3d.io.write_point_cloud(f"train_{IDX_TRAIN}_NR.ply", pcd_train)
    o3d.io.write_point_cloud(f"test_{IDX_TEST}_NR.ply", pcd_test)



    return