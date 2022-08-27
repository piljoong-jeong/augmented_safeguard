import copy
import os
import sys

from enum import Enum, auto

import numpy as np
import open3d as o3d
import yaml

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
        """
        ### read_data

        reads total `num_frames` images
        """

        mask_color = "color.jpg"
        mask_depth = "rendered.depth.png"
        mask_pose = "pose.txt" if not is_NeuralRouting_normalized else "pose.rnd.txt"

        # read
        def __read_with_mask(mask):
            return [
                os.path.join(self.dir_train_dataset, filename)
                for filename in os.listdir(self.dir_train_dataset)
                if mask in filename
            ]
        list_colors = __read_with_mask(mask_color)
        list_depths = __read_with_mask(mask_depth)
        list_poses = __read_with_mask(mask_pose)

        if len(list_colors) < num_frames or num_frames == -1:
            num_frames = len(list_colors)

        dict_dataset = {
            "colors": list_colors[:num_frames], 
            "depths": list_depths[:num_frames], 
            "poses": list_poses[:num_frames], 
        }

        print(f"[DEBUG] total {num_frames} dataset(s) read! access with keys: {dict_dataset.keys()}")

        # returns dict
        return dict_dataset
    
    def read_intrinsics(self):
        """
        ### read_intrinsics

        returns intrinsics for train & test dataset
        """

        if self.type_dataset == DatasetType.RIO10:
            filename_intrinsic = "camera.yaml"

            # intrinsics for train dataset
            with open(os.path.join(self.dir_train_dataset, filename_intrinsic), "r") as f:
                data = yaml.load(f, yaml.FullLoader)["camera_intrinsics"]
                intrinsics_train = o3d.camera.PinholeCameraIntrinsic(
                    width=data["width"], 
                    height=data["height"], 
                    fx=data["model"][0], 
                    fy=data["model"][1], 
                    cx=data["model"][2], 
                    cy=data["model"][3]
                )

            # intrinsics for test dataset
            with open(os.path.join(self.dir_test_dataset, filename_intrinsic), "r") as f:
                data = yaml.load(f, yaml.FullLoader)["camera_intrinsics"]
                intrinsics_test = o3d.camera.PinholeCameraIntrinsic(
                    width=data["width"], 
                    height=data["height"], 
                    fx=data["model"][0], 
                    fy=data["model"][1], 
                    cx=data["model"][2], 
                    cy=data["model"][3]
                )
        
        elif self.type_dataset == DatasetType.ICCV2017:
            intrinsics_train = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            intrinsics_test = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)


        return (intrinsics_train, intrinsics_test)


    def pointcloud_from_rgbd(self, dir_color: str, dir_depth: str, intrinsics: o3d.camera.PinholeCameraIntrinsic):
        """
        ### pointcloud_from_rgbd

        generates local point cloud using color, depth and intrinsics
        """

        color = o3d.io.read_image(dir_color)
        depth = o3d.io.read_image(dir_depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, 
            convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

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

    # read camera intrinsics
    intrinsics, _ = dataset_manager.read_intrinsics()
    print(intrinsics)

    # read data
    dataset = dataset_manager.read_data(num_frames=1)

    # generate point cloud
    pcd_local = dataset_manager.pointcloud_from_rgbd(
        dataset["colors"][0],
        dataset["depths"][0],
        intrinsics 
    )

    # DEBUG: visualize
    o3d.visualization.draw_geometries([pcd_local])