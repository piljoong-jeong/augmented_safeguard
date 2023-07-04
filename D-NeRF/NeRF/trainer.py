import imageio.v2 as imageio
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import NeRF.dataloader
import NeRF.model
import NeRF.ops
import NeRF.rendering


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

@torch.no_grad
def __test_render(
        
        basedir, 
        expname, 
        start,

        render_poses, 
        hwf, 
        K, 

        chunk, 
        render_factor,
        is_render_test,

        gt_images, 

        **render_test_kwargs
):
    """
    Writes a rendered video for testing purpose
    """

    test_save_dir = os.path.join(basedir, expname, f"render_only_{'test' if is_render_test else 'path'}_{start:06d}")
    os.makedirs(test_save_dir, exist_ok=True)

    rgbs, _ = NeRF.rendering.render_to_path(
        render_poses, 
        hwf, 
        K, 
        chunk, 
        render_test_kwargs, 
        gt_imgs=gt_images,
        savedir=test_save_dir, 
        render_factor=render_factor
    )
    imageio.mimwrite(os.path.join(test_save_dir, "video.mp4"), NeRF.ops.to_8b(rgbs), fps=30, quality=10)
    print(f"[DEBUG] Done test rendering at {test_save_dir}")

    return


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

    if args.render_test:
        render_poses = np.array(poses[i_test])

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


    render_train_kwargs, render_test_kwargs, start, grad_vars, optimizer = NeRF.model.create_NeRF(args)
    global_step = start
    render_test_kwargs.update(
        (dict_bounds := {
            "near": near, 
            "far": far, 
        })
    )

    _device = "cuda" # TODO: fix
    render_poses = torch.Tensor(render_poses).to(_device)

    if args.render_only:
        with torch.no_grad():
            gt_images = images[i_test] if args.render_test else None
        return __test_render(
            basedir, 
            expname, 
            start, 
            render_poses, 
            hwf, 
            K, 
            args.chunk,
            args.render_factor, 
            args.render_test, 
            gt_images=gt_images, 
            render_test_kwargs=render_test_kwargs
        )
    