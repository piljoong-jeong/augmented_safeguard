import copy
import os
import sys

from enum import Enum, auto

import numpy as np
import open3d as o3d
import yaml

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import augmented_safeguard as asfgd

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

    # NOTE: read camera intrinsics
    intrinsics, _ = dataset_manager.read_intrinsics()
    print(intrinsics)

    # NOTE: read data
    dataset = dataset_manager.read_data(num_frames=1)

    # NOTE: generate local point cloud
    pcd_local = dataset_manager.pointcloud_from_rgbd(
        dataset["colors"][0],
        dataset["depths"][0],
        intrinsics 
    )

    # DEBUG: visualize
    # o3d.visualization.draw_geometries([pcd_local]) # OK

    # NOTE: generate global point cloud
    tf = np.loadtxt(dataset["poses"][0])
    
    # NOTE: 1. apply transformation directly
    pcd_global = copy.deepcopy(pcd_local)
    pcd_global = pcd_global.transform(tf)

    # NOTE: 2. decompose R & t
    R = tf[:3, :3]
    t = tf[:3, 3]
    pcd_global2 = copy.deepcopy(pcd_local)
    pcd_global2.points = o3d.utility.Vector3dVector((R @ np.asarray(pcd_global2.points).T).T + np.tile(t.reshape((1, 3)), (np.asarray(pcd_global2.points).shape[0], 1)))

    # NOTE: o3d.geometry.PointCloud.transform() equivalents to affine transformation
    assert (np.asarray(pcd_global.points) == np.asarray(pcd_global2.points)).all()
    del pcd_global2

    # o3d.visualization.draw_geometries([pcd_local, pcd_global]) # OK


    n_kabsch = int(np.asarray(pcd_local.points).shape[0] - np.asarray(pcd_local.points).shape[0]%3)
    print(f"{n_kabsch=}")
    
    # ordering
    P = np.asarray(pcd_local.points)
    Q = np.asarray(pcd_global.points)
    P = asfgd.utility.uniform_sample(P, n_kabsch//3)
    Q = asfgd.utility.uniform_sample(Q, n_kabsch//3)

    # TODO: run Kabsch for each points
    P = asfgd.utility.blockshaped(P[:n_kabsch], 3, 3)
    Q = asfgd.utility.blockshaped(Q[:n_kabsch], 3, 3)

    list_angular_errors = []
    list_euler_angles_x = []
    list_euler_angles_y = []
    list_euler_angles_z = []

    list_angular_errors_cum = []

    sum_angular_errors = 0.0
    for idx, (P_, Q_) in enumerate(zip(P, Q)):
        # print(f"P = \n{P_}")
        
        P_ = np.asmatrix(P_)
        Q_ = np.asmatrix(Q_)

        _, R_, t_ = asfgd.transformations.rigid_transform_3D(P_, Q_, False)

        P_kabsch = (R_ @ Q_.T) + np.tile(t_, (1, 3))
        P_kabsch = P_kabsch.T
        # print(f"P_Kabsch = \n{P_kabsch}")


        # print(f"tf = \n{tf}")
        tf_kabsch = asfgd.transformations.pose_from_rot_and_trans(R_, t_.reshape(1, 3))
        # print(f"tf_kabsch = \n{tf_kabsch}")

        err_rot = asfgd.metric.error_angular(R, R_)
        # print(f"[DEBUG] err_rot = \n{err_rot}")

        euler_normalized = (euler:=asfgd.transformations.euler_from_rotmat(R_)) / np.linalg.norm(euler)
        # print(euler_normalized)
        
        list_angular_errors.append(err_rot)
        list_euler_angles_x.append(euler_normalized[0])
        list_euler_angles_y.append(euler_normalized[1])
        list_euler_angles_z.append(euler_normalized[2])

        sum_angular_errors += err_rot
        list_angular_errors_cum.append(sum_angular_errors / (idx+1))

        # if idx == 1000:
        #     break

    import pandas as pd
    df = pd.DataFrame()
    df["correspondence index"] = [i for i in range(len(list_angular_errors))]
    df["angular error"] = list_angular_errors
    df["angular error (cum)"] = list_angular_errors_cum
    

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme()
    sns.set(rc = {'figure.figsize':(20,8)})

    # angular error plot
    fig, ax = plt.subplots()
    # sns.scatterplot(data=df, x="correspondence index", y="angular error", ax=ax)
    sns.lineplot(data=df, x="correspondence index", y="angular error (cum)", ax=ax)
    plt.savefig("uniform_angular_error_all_cum.png")
    plt.show()
    plt.clf()
    exit()

    # sphere plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    df["euler_x"] = list_euler_angles_x
    df["euler_y"] = list_euler_angles_y
    df["euler_z"] = list_euler_angles_z
    x=df["euler_x"]
    y=df["euler_y"]
    z=df["euler_z"]
    ax.scatter(x, y, z)
    plt.savefig("uniform_sphere_euler_angle_all.png")
    plt.show()
    plt.clf()

    # histogram
    sns.histplot(data=df, x="angular error")
    plt.savefig("uniform_angular_error_hist.png")
    plt.clf()

    # KDE
    fig, ax = plt.subplots()
    kde = sns.kdeplot(data=df, x="angular error", ax=ax)
    lines = kde.get_lines()
    for line in lines:
        x, y = line.get_data()
        print(x[np.argmax(y)])
        ax.axvline(x[np.argmax(y)], ls="--", color="black")
    plt.savefig("uniform_angular_error_kde.png")
    plt.show()
    plt.clf()

    


