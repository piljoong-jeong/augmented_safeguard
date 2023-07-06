import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def loss_to_PSNR(loss: torch.Tensor):

    return (
        -10.0 * torch.log(loss) / 
                torch.log(torch.Tensor([10.0]))
    )
