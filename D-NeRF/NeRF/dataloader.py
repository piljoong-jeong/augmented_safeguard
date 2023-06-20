import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from pose import pose_spherical

# NOTE: implement Blender data loader
def load_blender_data(basedir, half_res=False, testskip=1):

    """
    NOTE: if D-NeRF extended, marked as # as a suffix
    """

    splits = ["train", "val", "test"]
    metas = {}

    # NOTE: load poses
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = [] #

    counts = [0] # NOTE: start of `train` index
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        times = [] # 
        # if s == "train" or testskip == 0: # NOTE: if `train`, use all frames
        #     skip = 1
        # else:
        #     skip = testskip
        skip = testskip # # NOTE: `time` should be re-normalized based on this

        # for frame in meta["frames"][::skip]:
        #     fname = os.path.join(basedir, frame["file_path"] + ".png")
        #     imgs.append(imageio.imread(fname))
        #     poses.append(np.array(frame["transform_matrix"]))
        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transformation_matrix']))
            times.append(
                (cur_time := frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip]) - 1))
            ) #

        assert times[0] == 0, "Time must start at 0" # 

        imgs = (np.array(imgs) / 255.).astype(np.float32) # NOTE: keep all 4 channels
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32) #
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times) #

        counts.append(counts[-1] + imgs.shape[0]) # NOTE: end of each split index
 
    # NOTE: index for each split
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0) #

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal_length = 0.5*W / np.tan(camera_angle_x/2.) # NOTE: see handwritten note

    # NOTE: poses for inference/test
    # render_poses = torch.stack(
    #     [
    #         pose_spherical(theta=angle, phi=-30.0, radius=4.0) 
    #         for angle
    #         in np.linspace(-180, 180, 160+1)[:-1]
    #     ], dim=0
    # )
    path_transforms_render = os.path.join(basedir, "transforms_render.json")
    if os.path.exists(path_transforms_render):
        
        with open(path_transforms_render, "r") as fp:
            meta = json.load(fp)
        
        render_poses = []
        for frame in meta["frames"]:
            render_poses.append(np.array(frame["transform_matrix"]))
        render_poses = np.array(render_poses).astype(np.float32)

    else:
        render_poses = torch.stack(
            [
                pose_spherical(theta=angle, phi=-30.0, radius=4.0) 
                for angle
                in np.linspace(-180, 180, 160+1)[:-1]
            ], dim=0
        )
    render_times = torch.linspace(0.0, 1.0, render_poses.shape[0])

    if True is half_res:
        H = H//2
        W = W//2

        # NOTE: IMPORTANT: change focal length!!!
        focal_length = focal_length/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for idx, image in enumerate(imgs):
            imgs_half_res[idx] = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, render_times, [H, W, focal_length], i_split