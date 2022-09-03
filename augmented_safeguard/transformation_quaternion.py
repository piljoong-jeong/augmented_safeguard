"""
### transformation_quaternion

finds optimal rotation using quaternion
"""

import numpy as np
from scipy.spatial.transform import Rotation
import quaternion

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def cross_to_mat(v: np.ndarray):
    # cross product to 3*3 skew symmetric matrix
    return np.cross(v, np.identity(v.shape[0]) * -1)

def quat_to_rot_mult(q: quaternion):
    # quaternion to 4*4 rotation matrix

    qvec = quaternion.as_float_array(q)[1:]
    Rq = np.zeros((4, 4))
    Rq[1:4, 0] = qvec
    Rq[0, 1:4] = -qvec
    Rq[1:, 1:] = -cross_to_mat(qvec)

    return Rq

def rigid_transform_3D(A, B, scale):

    # print(f"[DEBUG] A=\n{A}")
    # print(f"[DEBUG] B=\n{B}")

    A = np.asarray(A)
    B = np.asarray(B)

    N = np.zeros((4, 4))

    # TODO: for each point i, transform into imaginary quaternion rotation matrix
    for i in range(3):
        Ai = A[i]
        qA = np.quaternion(0, Ai[0], Ai[1], Ai[2])
        Bi = B[i]
        qB = np.quaternion(0, Bi[0], Bi[1], Bi[2])

        RqA = quat_to_rot_mult(qA)
        RqB = quat_to_rot_mult(qB)
        RqB[1:, 1:] *= -1 # NOTE: inverse skew symmetric

        # print(f"RqA = {Ai} -> \n{RqA}")
        # print(f"RqB = {Bi} -> \n{RqB}") # OK

        # TODO: caculate N_i A_i^T \cdot B_i
        N_i = RqA.T @ RqB
        N += N_i

        print(f"N_{i} = \n{N_i}")
    
    print(f"N = \n{N}")


    exit()

    return