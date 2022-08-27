import copy
import os
import sys
from enum import Enum, auto

import numpy as np
import open3d as o3d

# read image
# point cloud => src
# noise
# apply transformation => dst

SEED = 0
np.random.seed(SEED)

class DatasetType(Enum):
    ICCV2017 = "ICCV2017"
    RIO10 = "RIO10"

class DatasetManager():
    def __init__(self, 
        type_dataset: DatasetType, 
        dir_dataset: str,
        name_train_sequence: str, 
        name_test_sequence: str = "") -> None:
        """
        ### __init__

        TODO: use `self.dir_train_dataset` for now to experiment recursive SVD
        """
        self.type_dataset = type_dataset
        self.name_train_sequence = name_train_sequence
        self.name_test_sequence = name_test_sequence if name_test_sequence != "" else name_train_sequence


        if DatasetType.RIO10 == type_dataset:

            self.dir_dataset = os.path.join(dir_dataset, type_dataset.value)
            
            name_train_scene = name_train_sequence[3:5] 
            dir_train_sequence = os.path.join(f"scene{name_train_scene}/seq{name_train_scene}", name_train_sequence)
            
            name_test_scene = name_test_sequence[3:5]
            dir_test_sequence = os.path.join(f"scene{name_test_scene}/seq{name_test_scene}", name_test_sequence)
            
            self.dir_train_dataset = os.path.join(self.dir_dataset, dir_train_sequence)
            self.dir_test_dataset = os.path.join(self.dir_dataset, dir_test_sequence)
            
            print(f"{self.dir_train_dataset=}")
            print(f"{self.dir_test_dataset=}")

        return




    def read_data(self, 
        /, 
        is_NeuralRouting_normalized: bool = True,  
        num_frames: int = -1):


        mask_color = "color.jpg"
        mask_depth = "rendered.depth.png"
        mask_pose = "pose.txt" if not is_NeuralRouting_normalized else "pose.rnd.txt"

        
        list_colors = [
            os.path.join(self.dir_train_dataset, filename)
            for filename in os.listdir(self.dir_train_dataset)
            if mask_color in filename
        ]
        # print("\n".join(list_colors[:num_frames]))
        list_depths = [
            os.path.join(self.dir_train_dataset, filename)
            for filename in os.listdir(self.dir_train_dataset)
            if mask_depth in filename
        ]
        # print("\n".join(list_depths[:num_frames]))
        list_poses = [
            os.path.join(self.dir_train_dataset, filename)
            for filename in os.listdir(self.dir_train_dataset)
            if mask_pose in filename
        ]
        # print("\n".join(list_poses[:num_frames]))

        

        dict_dataset = {
            "colors": list_colors[:num_frames], 
            "depths": list_depths[:num_frames], 
            "poses": list_poses[:num_frames], 
        }

        print(f"[DEBUG] total {num_frames} dataset(s) read! access with keys: {dict_dataset.keys()}")

        # returns dict
        return dict_dataset
    


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

if __name__ == "__main__":

    dir_dataset = "/mnt/d/"
    type_dataset = DatasetType.RIO10
    name_train_sequence = "seq01_01"
    name_test_sequence = name_train_sequence

    dataset_manager = DatasetManager(DatasetType.RIO10, dir_dataset,  name_train_sequence, name_test_sequence)

    # read data
    dataset = dataset_manager.read_data(num_frames=1)

    

