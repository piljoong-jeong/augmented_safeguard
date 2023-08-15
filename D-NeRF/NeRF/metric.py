import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def loss_to_PSNR(loss: torch.Tensor):

    return (
        -10.0 * torch.log(loss) / 
                torch.log(torch.Tensor([10.0]))
    )

class MSE:
    def __call__(self, pred, gt):
        return torch.mean((pred - gt) ** 2)
    
class PSNR:
    def __call__(self, pred, gt):
        return 10 * torch.log10(1 / MSE(pred, gt))
    
class SSIM:
    def __call__(self, pred, gt, w_size=11, size_average=True, full=False):
        """
        
        Arguments:
        - pred: [B, C, H, W]
        - gt: [B, C, H, W]
        - w_size: Patch window size; widely chosen: https://en.wikipedia.org/wiki/Structural_similarity#Application_of_the_formula
        - size_average: 
        - full: 

        """

        # NOTE: set boundary values
        max_val = 255 if torch.max(pred) > 128 else 1
        min_val = -1 if torch.min(pred) < -0.5 else 0
        L = max_val - min_val # NOTE: dynamic range of pixels
        c1 = ((k1 := 0.01) * L) ** 2
        c2 = ((k2 := 0.03) * L) ** 2

        # NOTE: define weight of each pixel in the window using Gaussian 
        _, channel, height, width = pred.size()
        window = self.create_window(w_size, channel).to(pred.device)

        NO_PAD = 0
        # NOTE: compute: means
        mean_pred = F.conv2d(pred, window, padding=NO_PAD, groups=channel)
        mean_gt = F.conv2d(gt, window, padding=NO_PAD, groups=channel)
        mean_pred_sq = mean_pred.pow(2)
        mean_gt_sq = mean_gt.pow(2)
        mean_pred_gt = mean_pred * mean_gt

        # NOTE: compute: variances
        sigma_pred_sq = F.conv2d(pred * pred, window, padding=NO_PAD, groups=channel) - mean_pred_sq
        sigma_gt_sq = F.conv2d(gt * gt, window, padding=NO_PAD, groups=channel) - mean_gt_sq
        cross_correlation = F.conv2d(pred * gt, window, padding=NO_PAD, groups=channel) - mean_pred_gt

        # NOTE: compute: SSIM components
        dividend1 = (2*mean_pred_gt + c1)
        dividend2 = (2*cross_correlation + c2)
        denom1 = (mean_pred_sq + mean_gt_sq + c1)
        denom2 = (sigma_pred_sq + sigma_gt_sq + c2)

        ssim_map = (
            (dividend1 * dividend2) / 
            (denom1 * denom2)
        )
        
        if True is size_average:
            ret = ssim_map.mean()
        else:
            # NOTE: per-pixel SSIM?
            ret = ssim_map.mean(1).mean(1).mean(1)

        result = (
            ret, (contrast_sensitivity := torch.mean(dividend2 / denom2))
        ) if full else ret
        return result
    
    def create_window(self, w_size, channel=1):
        # TODO: why `sigma=1.5`?
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        
        # NOTE: Create 2D Gaussian distribution via auto-correlation
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def gaussian(self, w_size, sigma):
        """
        ### SSIM.gaussian

        Creates 1D Gaussian distribution
        """

        gaussian = torch.Tensor(
            [
                math.exp(
                    -(x - w_size//2) ** 2 /
                    float(2*sigma ** 2)
                )
                for x in range(w_size)
            ]
        )

        return gaussian / gaussian.sum()