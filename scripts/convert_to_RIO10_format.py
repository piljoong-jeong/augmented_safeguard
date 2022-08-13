import os 
import shutil
import sys

import numpy as np 
import scipy.spatial.transform

def main():

    dir_result = "/mnt/d/results"
    name_experiment = "rio10_scene01"
    name_test_seq = "seq01_02"
    name_pose_refinement_method = "RANSAC" # or ICP

    dir_base = os.path.join(dir_result, name_experiment, name_test_seq, name_pose_refinement_method,)

    name_reloc_back_transformed = "20220813T151020_leaf=0"

    dir_pose_bt = os.path.join(dir_base, name_reloc_back_transformed)
    print(dir_pose_bt)

    # extract reloc back transformed poses
    list_filename_poses_bt = [
        os.path.join(dir_pose_bt, filename_pose_bt)
        for filename_pose_bt in os.listdir(dir_pose_bt)
        if "reloc.bm" in filename_pose_bt
    ]

    with open("neuralrouting_ransac.txt", "w") as f:
        for idx, filename_pose in enumerate(list_filename_poses_bt):
            pose = np.loadtxt(filename_pose)
            # print(pose)
            R = pose[:3, :3]
            t = pose[:3, 3].astype(np.float32)
            q = scipy.spatial.transform.Rotation.from_matrix(R).as_quat().astype(np.float32)
            i = idx*10 # NOTE: NeuralRouting evaluated with step=10!

            s = f"{name_test_seq}/frame-{i:06d}"
            s += f" {str(q[0])} {str(q[1])} {str(q[2])} {str(q[3])}"
            s += f" {str(t[0])} {str(t[1])} {str(t[2])}\n"

            f.write(s)


    return

if __name__ == "__main__":
    main()