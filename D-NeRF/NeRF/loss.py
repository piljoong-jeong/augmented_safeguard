import torch
import torch.nn as nn
import torch.nn.functional as F

def image_to_MSE(output: torch.Tensor, target: torch.Tensor):
    assert output.shape == target.shape, f"[ERROR] {output.shape=} is not matched with {target.shape=}!"
    
    return torch.mean((output-target) ** 2)

