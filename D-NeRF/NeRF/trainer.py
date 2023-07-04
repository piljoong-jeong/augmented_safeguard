import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import NeRF.dataloader

def __save_args_and_config(basedir, expname, args):

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write(f"{arg} = {attr}\n")
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    return

def __render_only(
        
)


def train(args):

    # ---------------------------------------------
    if args.dataset_type == "llff":
        # TODO: implement
        raise NotImplementedError(f"[ERROR] for D-NeRF, llff data is not required!")
    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = NeRF.dataloader.load_blender_data(args.datadir, args.half_res, args.testskip)

        i_train, i_val, i_test = i_split

        near = 2.0
        far = 6.0

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]
    # ---------------------------------------------

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([
        [focal, 0, 0.5 * W], 
        [0, focal, 0.5 * H], 
        [0, 0, 1]
    ])


    basedir = args.basedir
    expname = args.expname
    __save_args_and_config(basedir, expname, args)




    
    if args.render_test:
        render_poses = torch.Tensor(np.array(poses[i_test])).to(device="cuda") # TODO: fix this

    if args.render_only:
        
    