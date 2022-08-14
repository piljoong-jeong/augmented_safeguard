import os
import sys
sys.path.append("..")

import numpy as np
import open3d as o3d

import augmented_safeguard as asfn


def main():

    dir_root = os.path.abspath("")

    # load src
    path_src = os.path.join(dir_root, "scripts/notebooks/src.ply")
    src = o3d.io.read_point_cloud(path_src)

    # load noisy dst
    sigma = 0.005
    path_dst = os.path.join(dir_root, f"scripts/notebooks/dst_noise_{sigma}.ply")
    dst = o3d.io.read_point_cloud(path_dst)

    # train RLS
    A = np.matrix([[10.0,10.0,10.0],
                [20.0,10.0,10.0],
                [20.0,10.0,15.0],])

    B = np.matrix([[18.8106,17.6222,12.8169],
                [28.6581,19.3591,12.8173],
                [28.9554, 17.6748, 17.5159],])
    solver = asfn.solvers.RecursiveLeastSquares(A, B)

    return

if __name__ == "__main__":
    main()