"""
### app.py

maintain entrypoints
"""

import copy
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

    # ---------- SHARED ----------
    dir_dataset = "/mnt/d/"
    type_dataset = DatasetType.RIO10
    name_train_sequence = "seq01_01"
    name_test_sequence = "seq01_02"

    dataset_manager = DatasetManager(DatasetType.RIO10, dir_dataset,  name_train_sequence, name_test_sequence)

    intrinsics_train, intrinsics_test = dataset_manager.read_intrinsics() # NOTE: for scene01, identical
    dataset_train, dataset_test = dataset_manager.read_data(is_NeuralRouting_normalized=True)

    IDX_TRAIN = 277
    IDX_TEST = 817

    pcd_train = dataset_manager.pointcloud_from_rgbd(dataset_train.colors[IDX_TRAIN], dataset_train.depths[IDX_TRAIN], intrinsics_train)
    pcd_test = dataset_manager.pointcloud_from_rgbd(dataset_test.colors[IDX_TEST], dataset_test.depths[IDX_TEST], intrinsics_train)
    # ---------- SHARED ----------


    pose_train = np.loadtxt(dataset_train.poses[IDX_TRAIN])
    pose_test = np.loadtxt(dataset_test.poses[IDX_TEST])
    print(f"train #{IDX_TRAIN}: \n{pose_train}")
    print(f"test  #{IDX_TEST}: \n{pose_test}")

    # TODO: point cloud fuse & visualize them to see if they're aligned
    pcd_train = pcd_train.transform(pose_train)
    pcd_test = pcd_test.transform(pose_test)
    o3d.io.write_point_cloud(f"train_{IDX_TRAIN}_NR.ply", pcd_train)
    o3d.io.write_point_cloud(f"test_{IDX_TEST}_NR.ply", pcd_test)



    return

def run_incremental_SVD():

    print(f"[DEBUG] run_incremental_SVD() started...")

    # ---------- SHARED ----------
    dir_dataset = "/mnt/d/"
    type_dataset = DatasetType.RIO10
    name_train_sequence = "seq01_01"
    name_test_sequence = "seq01_02"

    dataset_manager = DatasetManager(DatasetType.RIO10, dir_dataset,  name_train_sequence, name_test_sequence)

    intrinsics_train, intrinsics_test = dataset_manager.read_intrinsics() # NOTE: for scene01, identical
    dataset_train, dataset_test = dataset_manager.read_data(is_NeuralRouting_normalized=True)

    IDX_TRAIN = 277
    IDX_TEST = 817

    pcd_train = dataset_manager.pointcloud_from_rgbd(dataset_train.colors[IDX_TRAIN], dataset_train.depths[IDX_TRAIN], intrinsics_train)
    pcd_test = dataset_manager.pointcloud_from_rgbd(dataset_test.colors[IDX_TEST], dataset_test.depths[IDX_TEST], intrinsics_train)
    # ---------- SHARED ----------

    IDX_GT_FRAME = 1000
    SRC = np.asarray(pcd_train.points)
    DST = np.asarray(copy.deepcopy(pcd_train).transform(
        tf := np.loadtxt(dataset_train.poses[IDX_GT_FRAME]
    )).points)

    (R, t) = tf[:3, :3], tf[:3, 3]
    
    # NOTE: kabsch
    n_kabsch = int(SRC.shape[0] - SRC.shape[0]%3)
    print(f"{n_kabsch=}")


    SRC = asfgd.utility.uniform_stride_ordering(SRC, n_kabsch//3)
    DST = asfgd.utility.uniform_stride_ordering(DST, n_kabsch//3)

    SRC = asfgd.utility.blockshaped(SRC[:n_kabsch], 3, 3)
    DST = asfgd.utility.blockshaped(DST[:n_kabsch], 3, 3)
    
    print(f"----- R GT =\n{R}")
    
    # ---------- NOTE: initial SVD
    A = SRC[0]
    B = DST[0]
    # 1. normalize
    Am = np.mean(A, axis=0)
    # print(f"[DEBUG] {Am=}")
    # print(f"[DEBUG] mean matrix = \n{np.tile(Am, (SRC_.shape[0], 1))}")
    Am = np.tile(Am, (A.shape[0], 1))
    Ac = A - Am
    Bm = np.tile(np.mean(B, axis=0), (B.shape[0], 1))
    Bc = B - Bm

    M = np.matmul( np.transpose(Ac), Bc) # H = P^T * Q
    print(f"[DEBUG] M=Bc*Ac = \n{M}")

    U, S, Vt = np.linalg.svd(M)
    print(f"[DEBUG] U = \n{U}")
    print(f"[DEBUG] S = \n{S}")
    print(f"[DEBUG] Vt = \n{Vt}")

    R_init = np.matmul(Vt.T , U.T)
    sign = 1
    # special reflection case
    if np.linalg.det(R_init) < 0:
        # print("[DEBUG] Reflection detected")
        Vt[2, :] *= -1
        sign = -1
        R_init =np.matmul(Vt.T , U.T)
    
    print(f"----- R GT =\n{R}")
    print(f"----- R init =\n{R_init}")

    err_rot = asfgd.metric.error_angular(R, R_init)
    print(f"[DEBUG] {err_rot=}")

    # ---------- NOTE: update SVD
    
    for idx, (SRC_, DST_) in enumerate(zip(SRC, DST)):

        if 0 == idx: continue

        # 1. normalize
        Am = np.tile(np.mean(SRC_, axis=0), (SRC_.shape[0], 1))
        Ac = SRC_ - Am
        Bm = np.tile(np.mean(DST_, axis=0), (DST_.shape[0], 1))
        Bc = DST_ - Bm

        # FIXME: 22-09-30 test with c=1 !!!
        C = np.matmul(np.transpose(Ac), Bc) # H = P^T * Q
        print(f"[DEBUG] C=Ac^T*Bc = \n{C}")

        L = np.matmul(U.T, C)
        H = C - np.matmul(U, L)
        J, K = np.linalg.qr(H)

        # NOTE: middle matrix
        Q = np.zeros(shape=(A.shape[0]+SRC_.shape[0], A.shape[1]+SRC_.shape[1]), dtype = A.dtype)
        Q[:3, :3] = np.diag(S)
        Q[:3, 3:] = L
        Q[3:, 3:] = K
        np.set_printoptions(precision=4)
        print(f"[DEBUG] Q = \n{Q}")
        print(Q.shape)



        U_, S_, Vt_ = np.linalg.svd(Q)
        # print(f"[DEBUG] U_ = \n{U_}")
        # print(f"[DEBUG] S_ = \n{S_}")
        # print(f"[DEBUG] Vt_ = \n{Vt_}")

        # NOTE: update U
        UJ = np.hstack((U, J))
        print(f"[DEBUG] UJ = \n{UJ}")
        print(UJ.shape)
        U__ = np.matmul(UJ, U_)
        print(f"[DEBUG] U'' = \n{U__}")

        # NOTE: update S
        S__ = np.diag(S_)

        # NOTE: update V
        VI = np.eye(A.shape[0]+SRC_.shape[0])
        VI[:3, :3] = Vt.T
        V__ = np.matmul(VI, Vt_.T)
        print(f"[DEBUG] V'' = \n{V__}")

        print(f"{U__.shape=}")
        print(f"{S__.shape=}")
        print(f"{V__.shape=}")

        U__S__Vt__ = np.matmul(np.matmul(U__, S__), V__.T)
        print(f"[DEBUG] U''S''V''^T = \n{U__S__Vt__}")

        print(f"[DEBUG] M = \n{M}")
        print(f"[DEBUG] C = \n{C}")
        

        exit()

    return