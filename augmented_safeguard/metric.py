import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

def measure_metric(pose_GT: np.ndarray, pose_input: np.ndarray):

    

    return NotImplementedError


def error_angular(R_gt: np.ndarray, R_est: np.ndarray):
    """
    ### error_angular

    measure angular error between `R_gt` and `R_est` using Quaternion
    """

    q_gt = Rotation.from_matrix(R_gt).as_quat()
    q_est = Rotation.from_matrix(R_est).as_quat()

    d1 = math.fabs(q_gt.dot(q_est))
    d2 = min(1.0, max(-1.0, d1))

    return 2 * math.acos(d2) * 180 / math.pi
    

if __name__ == "__main__":
    # load GT
    dir_root = os.path.abspath("")
    print(dir_root)

    path_pose_GT = os.path.join(dir_root, "scripts/notebooks/reg_p2l.txt")
    pose_GT = np.loadtxt(path_pose_GT)
    print(pose_GT)