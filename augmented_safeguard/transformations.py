

import numpy as np
from scipy.spatial.transform import Rotation

# https://gist.github.com/oshea00/dfb7d657feca009bf4d095d4cb8ea4be
def rigid_transform_3D(A, B, scale):

    assert len(A) == len(B)
    N = A.shape[0]

    # mean of each point cloud
    Am = np.mean(A, axis=0)
    Bm = np.mean(B, axis=0)

    # centered point clouds
    Ac = A - np.tile(Am, (N, 1))
    Bc = B - np.tile(Bm, (N, 1))

    H = np.transpose(Bc) * Ac # NOTE: Kabsch; H = P^T \cdot Q
    if scale: H /= N

    """
    Based on rotation formula, optimal rotation R is

    R = sqrt(H^T \cdot H) \cdot H^-1

    since directly solving this is complicated and hard to handle edge cases (e.g., singular H), use SVD
    """

    U, S, Vt = np.linalg.svd(H) # NOTE: Kabsch; H = USV^T
    
    R = Vt.T * U.T # NOTE: Kabsch; R = V \cdot U^T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("[DEBUG] Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T
    
    if scale:
        Avar = np.var(A, axis=0).sum()
        c = 1 / (1 / Avar * np.sum(S)) # scale singular value
        t = -R * (Bm.T * c) + Am.T
        
    else:
        c = 1
        t = -R.dot(Bm.T) + Am.T

    return c, R, t

def pose_from_rot_and_trans(R: np.ndarray, t: np.ndarray):

    assert len(R.shape) == 2
    assert R.shape[0] == 3
    assert R.shape[0] == R.shape[1]

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def euler_from_rotmat(R:np.ndarray):
    return Rotation.from_matrix(R).as_euler("zyx", degrees=True)

if __name__ == "__main__":
    # dataset generation
    # points
    A = np.matrix([[10.0,10.0,10.0],
                [20.0,10.0,10.0],
                [20.0,10.0,15.0],])

    B = np.matrix([[18.8106,17.6222,12.8169],
                [28.6581,19.3591,12.8173],
                [28.9554, 17.6748, 17.5159],])
    n = B.shape[0]

    # transformations
    T_src = np.matrix([[0.9848, 0.1737,0.0000,-11.5865],
                    [-0.1632,0.9254,0.3420, -7.621],
                    [0.0594,-0.3369,0.9400,2.7752],
                    [0.0000, 0.0000,0.0000,1.0000]])
    T_dst = np.matrix([[0.9848, 0.1737,0.0000,-11.5859],
                    [-0.1632,0.9254,0.3420, -7.621],
                    [0.0594,-0.3369,0.9400,2.7755],
                    [0.0000, 0.0000,0.0000,1.0000]])

    s, R, t = rigid_transform_3D(A, B, scaling:=False)

    # validation
    A2 = (R * B.T) + np.tile(t, (1, n))
    A2 = A2.T
    print(A)
    print(A2)