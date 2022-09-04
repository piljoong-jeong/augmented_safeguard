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

    assert len(A) == len(B)
    n = A.shape[0]

    # mean of each point cloud
    Am = np.mean(A, axis=0)
    Bm = np.mean(B, axis=0)

    # centered point clouds
    Ac = A - np.tile(Am, (n, 1))
    Bc = B - np.tile(Bm, (n, 1))

    N = np.zeros((4, 4))

    # TODO: for each point i, transform into imaginary quaternion rotation matrix
    for i in range(3):
        Ai = Ac[i]
        qA = np.quaternion(0, Ai[0], Ai[1], Ai[2])
        Bi = Bc[i]
        qB = np.quaternion(0, Bi[0], Bi[1], Bi[2])

        RqA = quat_to_rot_mult(qA)
        RqB = quat_to_rot_mult(qB)
        RqB[1:, 1:] *= -1 # NOTE: inverse skew symmetric

        # print(f"RqA = {Ai} -> \n{RqA}")
        # print(f"RqB = {Bi} -> \n{RqB}") # OK

        # TODO: caculate N_i A_i^T \cdot B_i
        N_i = RqA.T @ RqB
        N += N_i

        # print(f"N_{i} = \n{N_i}")
    
    # print(f"N = \n{N}")

    U, S, V_t = np.linalg.svd(N)
    # print(S)
    best_eigenvalue = S[0]
    homogeneous_system = (N - best_eigenvalue * np.eye(N.shape[0]))
    # print(f"homogeneous_system = \n{homogeneous_system}")

    is_solve_using_svd = False
    if is_solve_using_svd:
        U2, S2, V2_t = np.linalg.svd(homogeneous_system)
        print(f"{S2=}")
        print(V2_t)

        q_result = V2_t[:, np.argmin(S2)] # NOTE: point to least singular value index
        print(f"{q_result=}")


    is_solve_using_cofactor = True
    if is_solve_using_cofactor:
        cofactor = np.linalg.inv(homogeneous_system).T * np.linalg.det(homogeneous_system)
        # print(f"cofactor = \n{cofactor}")
        # for i in range(4):
        #     print(np.linalg.norm(cofactor[i, :]))
        q_result = cofactor[np.argmax(np.linalg.norm(cofactor, axis=1))]
        # print(q_result)


    
    R_result = Rotation.from_quat(q_result).as_matrix()
    # print(f"R_result = \n{R_result}")
        
    # FIXME: shape should be (3, 1)
    t = (-R_result.dot(Bm.T) + Am.T)[..., None]
    # print(f"{t.shape=}")
    # exit()


    return None, R_result, t