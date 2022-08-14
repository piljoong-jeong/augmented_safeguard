import os
import sys

import numpy as np

def measure_metric(pose_GT: np.ndarray, pose_input: np.ndarray):

    

    return NotImplementedError


if __name__ == "__main__":
    # load GT
    dir_root = os.path.abspath("")
    print(dir_root)

    path_pose_GT = os.path.join(dir_root, "scripts/notebooks/reg_p2l.txt")
    pose_GT = np.loadtxt(path_pose_GT)
    print(pose_GT)