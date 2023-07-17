import imageio.v2 as imageio
import os
import time
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import NeRF.dataloader
import NeRF.loss
import NeRF.metric
import NeRF.model
import NeRF.ops
import NeRF.rendering


def save_args_and_config(basedir, expname, args):

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

@torch.no_grad()
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
        render_factor=render_factor, 
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

def initialize_NeRF(fn_create_NeRF: Callable, near: float, far: float, args):
    """
    ### trainer.initialize_NeRF

    Returns NeRF states with updated near & far bound, as well as its continuing global step
    """
    
    render_train_kwargs, render_test_kwargs, start, grad_vars, optimizer = fn_create_NeRF(args)
    global_step = start
    dict_bounds = {
        "near": near, 
        "far": far, 
    }
    render_train_kwargs.update(dict_bounds)
    render_test_kwargs.update(dict_bounds)

    return render_train_kwargs, render_test_kwargs, global_step, grad_vars, optimizer

# TODO: to be clear, this is not only for training; test rendering is included - rename it
def train(args):

    # ---------------------------------------------
    if args.dataset_type == "llff":
        # TODO: implement (used in vanilla NeRF)
        raise NotImplementedError
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
    save_args_and_config(basedir, expname, args)

    render_train_kwargs, render_test_kwargs, global_step, grad_vars, optimizer = initialize_NeRF(NeRF.model.create_NeRF, near, far, args)
    

    _device = "cuda" # TODO: fix
    render_poses = torch.Tensor(render_poses).to(_device)

    if args.render_only:
        with torch.no_grad():
            gt_images = images[i_test] if args.render_test else None
        return __test_render(
            basedir, 
            expname, 
            global_step, 
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
    start = global_step + 1
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

        # NOTE: if not shuffled batching, generate ray batch per image
        else:
            img_i = np.random.choice(i_train)
            target = torch.Tensor(images[img_i]).to(_device)
            pose = poses[img_i, :3, :4]

            # TODO: meaning??
            if N_rand is not None:
                
                rays_o, rays_d = NeRF.rendering.get_rays(H, W, K, torch.Tensor(pose))

                # NOTE: for fast training
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)

                    h_start = H//2 - dH
                    h_end = H//2 + dH - 1
                    h_steps = 2*dH 

                    w_start = W//2 - dW
                    w_end = W//2 + dW - 1
                    w_steps = 2*dW                   

                    if i == start:
                        print(f"[INFO ] center cropping of size {2*dH} x {2*dW}is enabled until iter {args.precrop_iters}")

                # NOTE: ray batch from all pixels
                else:
                    
                    h_start = 0
                    h_end = H - 1
                    h_steps = H

                    w_start = 0
                    w_end = W - 1
                    w_steps = W
                
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(h_start, h_end, h_steps), 
                        torch.linspace(w_start, w_end, w_steps)
                    ), dim=-1
                )

            coords = torch.reshape(coords, [-1, 2])
            select_coords = coords[
                np.random.choice(coords.shape[0], size=[N_rand], replace=False)
            ].long()

            batch_rays = torch.stack([
                rays_o[select_coords[:, 0], select_coords[:, 1]],
                rays_d[select_coords[:, 0], select_coords[:, 1]] 
            ], dim=0)

            # NOTE: GT pixels
            target_s = target[select_coords[:, 0], select_coords[:, 1]]
        # NOTE: end of shuffled batch ray generation

        # NOTE: core optimization
        rgb, disp, acc, extras = NeRF.rendering.render(
            H, W, K, 
            chunk=args.chunk, 
            rays=batch_rays, 
            verbose=i < 10, 
            retraw=True, 
            **render_train_kwargs
        )

        # NOTE: loss calculation & backpropagation
        optimizer.zero_grad()
        img_loss_fine = NeRF.loss.image_to_MSE(
            output=rgb, 
            target=target_s
        )
        loss = img_loss_fine
        if "rgb0" in extras:
            img_loss_coarse = NeRF.loss.image_to_MSE(
                output=extras["rgb0"], 
                target=target_s
            )
            loss = loss + img_loss_coarse
        loss.backward()
        optimizer.step()

        # NOTE: PSNR for statistics
        psnr_fine = NeRF.metric.loss_to_PSNR(img_loss_fine)
        psnr_coarse = NeRF.metric.loss_to_PSNR(img_loss_coarse)

        # NOTE: learning rate update (sec. 5.3)
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        # NOTE: logging
        dt = time.time() - time0
        # NOTE: save checkpoints TODO: rename `args.i_weights`
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, f"{i:06d}.tar")
            torch.save(
                {
                    "global_step": global_step, 
                    "network_fn_state_dict": render_train_kwargs["network_fn"].state_dict(), 
                    "network_fine_state_dict": render_train_kwargs["network_fine"].state_dict(), 
                    "optimizer_state_dict": optimizer.state_dict(),
                }, f=path
            )

        # NOTE: save video
        if i % args.i_video == 0 and i > 0:
            __test_render(
                basedir, 
                expname, 
                start, 
                render_poses, 
                hwf, 
                K, 
                args.chunk,
                args.render_factor, 
                args.render_test, 
                gt_images=images, 
                render_test_kwargs=render_test_kwargs
            )

        if i % args.i_print == 0:
            tqdm.write(f"[INFO ] train; iter={i}, loss={loss.item()}, psnr={psnr_fine.item()}")

            # NOTE: render for plot
            with torch.no_grad():
                rgb, depth, acc, _ = NeRF.rendering.render(
                    H, W, K, 
                    c2w=(testpose := poses[len(poses)//2])[:3, :4], 
                    **render_test_kwargs
                )

            # NOTE: plotting
            psnrs.append(psnr_fine.detach().cpu().numpy())
            iternums.append(i)

            fig = plt.figure(figsize=(10, 4))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(rgb.cpu())
            ax1.set_title(f"Iteration: {i}")
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(iternums, psnrs)
            ax2.set_title(f"PSNR")
            fig.show() # TODO: check if this should be turned on; why not headless?

            # NOTE: save plot
            dir_iter_plot = os.path.join(basedir, expname, "iter_plots")
            os.makedirs(dir_iter_plot, exist_ok=True)
            fig.savefig(
                os.path.join(dir_iter_plot, f"iter={i}_PSNR={psnr_fine.item():.6f}.png")
            )

            fig.clear()
            plt.close(fig)

        global_step += 1

    return