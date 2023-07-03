import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_pdf(
        bins, 
        weights, 
        N_samples, 
        det=False, # NOTE: linear sampling
        pytest=False,
):
    
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    # NOTE: sampling is done using inverse pdf
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([
        torch.zeros_like(cdf[..., :1]), cdf
    ], dim=-1)

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # NOTE: pytest; overwrite `u` with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # NOTE: inverse cdf
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, side="right")

    # NOTE: clamping indices
    below = torch.max(torch.zeros_like(indices-1), indices-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(indices), indices)

    indices_grid = torch.stack([below, above], dim=-1)
    matched_shape = [indices_grid.shape[0], indices_grid.shape[1], cdf.shape[-1]] # NOTE: dim: [B, N_samples, len(bins)]
    cdf_grid = torch.gather(
        cdf.unsqueeze(dim=1).expand(matched_shape), 
        dim=2, index=indices_grid
    )
    bins_grid = torch.gather(
        bins.unsqueeze(dim=1).expand(matched_shape), 
        dim=2, index=indices_grid
    )

    # NOTE: calculate importance
    denom = cdf_grid[..., 1] - cdf_grid[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_grid[..., 0]) / denom # NOTE: `t` is world coordinate of `u`
    samples = bins_grid[..., 0] + t * (bins_grid[..., 1] - bins_grid[..., 0])

    return samples