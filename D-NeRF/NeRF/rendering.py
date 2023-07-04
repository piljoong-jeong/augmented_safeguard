import imageio.v2 as imageio # TODO: migrate to v3
import os
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import NeRF.sampler

def decompose_ray_batch(ray_batch, is_time_included: bool):

    
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    # NOTE: `bounds` contain `frame_time` if D-NeRF
    time_included = 1 if is_time_included else 0
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > (8+time_included) else None
    bounds = torch.reshape(ray_batch[..., 6:(8+time_included)], [-1, 1, (2+time_included)])
    near, far = bounds[..., 0], bounds[..., 1]
    frame_time = bounds[..., 2] if is_time_included else None
    
    return rays_o, rays_d, viewdirs, near, far, frame_time

# TODO: move to sampler.py
def sample_z(near, far, N_samples, lindisp):
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    if not lindisp:
        z_vals = (near * (1.0-t_vals)) + (far * (t_vals))
    else:
        z_vals = 1.0 / ( ( 1.0 / (near*(1.0-t_vals)) ) + ( 1.0 / (far*(t_vals))) )  # [-1, 1]
    
    return z_vals

# TODO: move to sampler.py
def add_noise_z(z_vals, perturb, pytest=False):
    if perturb <= 0.0:
        return z_vals
    
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., :1], mids], dim=-1)
    t_rand = torch.rand(z_vals.shape) # NOTE: rand before expand would cause identical rand?

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        t_rand = np.random.rand(*list(z_vals.shape))
        t_rand = torch.Tensor(t_rand)

    z_vals = lower + (upper - lower) * t_rand
    return z_vals

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    
    

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], 
        dim=-1
    )

    # NOTE: rotate `dists` w.r.t. direction
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # NOTE: color
    rgb = torch.sigmoid(raw[..., :3]) # NOTE: sigmoid for guaranteeing positive value

    # NOTE: add noise to prevent overfitting
    noise = torch.randn(raw[..., 3].shape) * raw_noise_std if raw_noise_std > 0.0 else 0.0
    # NOTE: eq. 3
    raw2alpha = lambda raw, dists, activation_fn=F.relu: 1.0 - torch.exp(-activation_fn(raw) * dists)
    alpha = raw2alpha(raw[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(
        torch.cat([
            torch.ones(alpha.shape[0], 1), 
            1.0 - alpha + 1e-10
        ], dim=-1), 
        dim=-1
    )[:, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1) # NOTE: the more farther, the more darker; unintuitive! hence we use `disp_map`
    disp_map = 1.0 / torch.max(
        1e-10*torch.ones_like(depth_map), depth_map/torch.sum(weights, dim=-1)
    ) # NOTE: inv. normalized `depth_map`
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(
        ray_batch, 
        network_fn, 
        network_query_fn, 
        N_samples, 
        retraw=False, 
        lindisp=False, 
        perturb=0.0, 
        N_importance=0, 
        network_fine=None, 
        white_bkgd=False, 
        raw_noise_std=0.0, 
        verbose=False, 
        pytest=False,
):
    
    rays_o, rays_d, viewdirs, near, far, _ = decompose_ray_batch(ray_batch, is_time_included=False)

    N_rays = ray_batch.shape[0]

    # NOTE: get z-values
    z_vals = sample_z(near, far, N_samples, lindisp)
    z_vals = z_vals.expand([N_rays, N_samples])
    z_vals = add_noise_z(z_vals, perturb) # NOTE: to ensure CG recording

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    raw = network_query_fn(pts, viewdirs, network_fn)

    ret = {}
    if retraw: ret["raw"] = raw

    def __stability_check():
        if False: # TODO: add DEBUG
            return
        for key, tensor in ret.items():
            if torch.isnan(tensor).any():
                print(f"[ERROR] Numerical error! {key=} contains NaN.")
            if torch.isinf(tensor).any():
                print(f"[ERROR] Numerical error! {key=} contains INF.")
    
    # NOTE: coarse network
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )
    ret["rgb_map"] = rgb_map
    ret["disp_map"] = disp_map
    ret["acc_map"] = acc_map
    __stability_check()
    if N_importance <= 0:
        return ret

    # NOTE: fine network
    rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    

    # NOTE: importance sampling
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_importance_samples = NeRF.sampler.sample_pdf(
        z_vals_mid, 
        weights[..., 1:-1], 
        N_importance, 
        det=(perturb == 0.0), 
        pytest=pytest
    )
    z_importance_samples = z_importance_samples.detach()

    # NOTE: aggregate importance samples
    z_vals, __IDX_NO_NEED = torch.sort(
        torch.cat([z_vals, z_importance_samples], dim=-1),
        dim=-1
    )
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # NOTE: run fine network
    run_fn = network_fn if network_fine is None else network_fine
    raw = network_query_fn(pts, viewdirs, run_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, 
        z_vals, 
        rays_d, 
        raw_noise_std, 
        white_bkgd, 
        pytest=pytest
    )

    ret["rgb_map"] = rgb_map
    ret["disp_map"] = disp_map
    ret["acc_map"] = acc_map
    ret["rgb0"] = rgb_map_0
    ret["disp0"] = disp_map_0
    ret["acc0"] = acc_map_0
    ret["z_std"] = torch.std(z_importance_samples, dim=-1, unbiased=False) # NOTE: dim: [N_rays]
    __stability_check()

    return ret

def batchify_rays(rays_flat, chunk, fn_render_rays, **kwargs):

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        ret = fn_render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret

def get_rays(
        H, 
        W, 
        K, 
        c2w,
):
    
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W), 
        torch.linspace(0, H-1, H)
    )
    i = i.t()
    j = j.t()

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    dirs = torch.stack(
        [
            (i-cx)/fx, 
            -(j-cy)/fy, # NOTE: image coord -> NDC coord
            -torch.ones_like(i) # NOTE: lookat: -Z direction
        ], dim=-1
    )

    # NOTE: convert NDC `dirs` to world coord
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], dim=-1) # NOTE: is identical to: c2w @ dirs | c2w.dot(dirs)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    ### ndc_rays


    """

    # NOTE: shift ray origins to near plane (z=-n) (following last paragraph in Appendix. C)
    t_n = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t_n[..., None] * rays_d

    # NOTE: equation (25)
    o0 = (-1. * focal / (0.5*W)) * (rays_o[..., 0] / rays_o[..., 2])
    o1 = (-1. * focal / (0.5*H)) * (rays_o[..., 1] / rays_o[..., 2])
    o2 = (1. + 2.*near / rays_o[..., 2])

    # NOTE: equation (26)
    d0 = (-1. * focal / (0.5*W)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = (-1. * focal / (0.5*H)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near * (1. / rays_o[..., 2])

    rays_o = torch.stack([o0, o1, o2], dim=-1)
    rays_d = torch.stack([d0, d1, d2], dim=-1)
    return rays_o, rays_d


def prepare_rays(
        H, 
        W, 
        K, 
        rays=None, 
        c2w=None, 
        ndc=True, 
        near=0.0, 
        far=1.0, 
        use_viewdirs=False, 
        c2w_staticcam=None, 
):
    """
    ### rendering.prepare_rays

    Returns transformed ray origin & direction, and its corresponding near & far bounds, as well as the original shape of ray tensor (used to reshape flattened ray batches) 
    
    """
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays
    rays_original_shape = rays_d.shape

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    if ndc:
        rays_o, rays_d = ndc_rays(
            H, 
            W, 
            K[0][0], 
            1.0, 
            rays_o, 
            rays_d
        )

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    return rays_o, rays_d, near, far, viewdirs, rays_original_shape


def images_from_rendering(
        rays: torch.Tensor, 
        chunk: int, 
        fn_render_rays: Callable, 
        rays_original_shape: torch.Size, 
        maps_to_extract: list = ["rgb_map", "disp_map", "acc_map"],
        **kwargs
):
    
    all_ret = batchify_rays(rays, chunk, fn_render_rays, **kwargs)
    for k in all_ret:
        k_shape = list(rays_original_shape[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_shape)
    
    ret_list = [all_ret[k] for k in maps_to_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in maps_to_extract}

    return ret_list, ret_dict


def render(
        H, 
        W, 
        K, 
        chunk=1024*32, 
        rays=None, 
        c2w=None, 
        ndc=True, 
        near=0.0, 
        far=1.0, 
        use_viewdirs=False,
        c2w_staticcam=None,
        **kwargs
):
    
    rays_o, rays_d, near, far, viewdirs, rays_original_shape = prepare_rays(H, W, K, rays, c2w, ndc, near, far, use_viewdirs, c2w_staticcam)

    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], dim=-1)

    ret_list, ret_dict = images_from_rendering(rays, chunk, render_rays, rays_original_shape)

    return ret_list + [ret_dict]

def __to_8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def render_to_path(
        render_poses, 
        hwf, 
        K, 
        chunk, 
        render_kwargs, 
        gt_imgs=None, 
        savedir=None, 
        render_factor=0
):
    # TODO: remove dependency `K`; we already have `hwf`!

    H, W, focal = hwf
    if render_factor > 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor # NOTE: float

    rgbs, disps = [], []

    t = time.time()
    for idx, c2w in enumerate(tqdm(render_poses)):
        print(f"{idx=} took {time.time()-t}s")
        t = time.time()

        rgb, disp, acc, _ = render(
            H, W, K, 
            chunk=chunk, 
            c2w=c2w[:3, :4], 
            **render_kwargs
        )
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if idx == 0: print(f"{rgb.shape=} & {disp.shape=}")

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = __to_8b(rgbs[-1])
            filename = os.path.join(savedir, f"{idx:03d}.png")
            imageio.imwrite(filename, rgb8)

    return np.stack(rgbs, axis=0), np.stack(disps, axis=0)