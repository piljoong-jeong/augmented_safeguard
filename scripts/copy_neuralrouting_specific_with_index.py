"""
copy_NeuralRouting_specific_with_index

This script copies poses from a regression tree with given index to a target directory, and renames them as they are the one with best inference confidence among all regression trees.

---

it is found that `neuralrouting/ransac_v0.2` returns 4 poses, which each corresponds to evaluation for each regression tree.

They mentioned that the final pose is selected from a tree which returns best probability, but the function seems missing.

Nevertheless, we wanted to measure performance on evaluation script provided by RIO10

"""

import os 
import shutil
import sys


def main():

    dir_result = "/mnt/d/results"
    name_experiment = "rio10_scene01"
    name_test_seq = "seq01_02"
    name_pose_refinement_method = "RANSAC" # or ICP

    dir_base = os.path.join(dir_result, name_experiment, name_test_seq, name_pose_refinement_method,)

    name_neuralrouting_result_dir = "20220813T151020"

    dir_pose_neuralrouting = os.path.join(dir_base, name_neuralrouting_result_dir)

    # set src to copy with target index
    idx_leaf = 0 # NOTE: set this 0~3
    list_src_filename_poses = [
        os.path.join(dir_pose_neuralrouting, filename_src_pose) 
        for filename_src_pose in os.listdir(dir_pose_neuralrouting)
        if idx_leaf == int(filename_src_pose.split("-")[2][0])
    ] 

    # set dst, and make if not exist
    dir_pose_copied = os.path.join(dir_base, f"{name_neuralrouting_result_dir}_leaf={idx_leaf}")
    if not os.path.exists(dir_pose_copied):
        os.makedirs(dir_pose_copied, exist_ok=True)

    # rename
    list_dst_filename_poses = [
        os.path.join(
            dir_pose_copied, 
            "-".join([(tokens:=os.path.basename(filename_src_pose).split("-"))[0], tokens[1]]) + tokens[2][1:]
        )
        for filename_src_pose in list_src_filename_poses
    ]

    # copy
    _ = [shutil.copy2(src, dst) 
    for src, dst 
    in zip(list_src_filename_poses, list_dst_filename_poses)]; del _


    return

if __name__ == "__main__":
    main()