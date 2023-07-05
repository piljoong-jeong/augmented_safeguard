import imageio.v2 as imageio
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

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


def shuffle_rays(H, W, K, images, poses, i_train):

    print(f"[DEBUG] get rays")
    # NOTE: dim: [N(#images, dim=0), ro + rd, H, W, 3]
    # NOTE: `get_rays` returns [ro(1D), H, W, 3] & [rd(1D), H, W, 3]
    with torch.no_grad():
        rays = torch.stack([NeRF.rendering.get_rays(H, W, K, p) for p in poses[:, :3, :4]], dim=0).detach().cpu().numpy()


    print(f"[DEBUG] Done. concats")
    # NOTE: dim: [N, ro + rd + rgb, H, W, 3]
    # NOTE: assort per-pixel info: ray origin + ray direction + rgb pixel value
    rays_rgb = np.concatenate([rays, images[:, None]], axis=1)

    # NOTE: dim: [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])

    # NOTE: train images only
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)

    # NOTE: dim: [i_train*H*W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]).astype(np.float32)

    print(f"[DEBUG] shuffle rays")
    np.random.shuffle(rays_rgb)

    print(f"[DEBUG] done.")
    return rays_rgb

# TODO: to be clear, this is not only for training; test rendering is included - rename it
def train(args):

    # ---------------------------------------------
    if args.dataset_type == "llff":
        # TODO: implement
        raise NotImplementedError(f"[ERROR] for D-NeRF, llff data is not required!")
    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = NeRF.dataloader.load_blender_data(args.datadir, args.half_res, args.testskip)

        i_train, i_val, i_test, near, far, images = NeRF.dataloader.post_load_blender_data(i_split, images, args.white_bkgd)
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
    render_train_kwargs.update(dict_bounds)

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
    
    # ----------------------------------------------

    N_rand = args.N_rand
    
    if (use_shuffled_batching := not args.no_batching):
        rays_rgb = shuffle_rays(H, W, K, images, poses, i_train)
        i_batch = 0

    if use_shuffled_batching:
        images = torch.Tensor(images).to(_device)
        rays_rgb = torch.Tensor(rays_rgb).to(_device)
    poses = torch.Tensor(poses).to(_device)

    # NOTE: train iteration
    N_iters = 200000+1 # NOTE: add to arguments
    print(f"[DEBUG] begin training; views: {i_train=} {i_test=} {i_val=}")

    psnrs = []
    iternums = []
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        if use_shuffled_batching:
            batch = rays_rgb[i_batch : i_batch + N_rand]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print(f"[DEBUG] shuffle data after an epoch!")
                rays_rgb = rays_rgb[
                    (idx_rand := torch.randperm(rays_rgb.shape[0]))
                ]
                i_batch = 0

        



    



