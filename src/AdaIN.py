"""
adain.py

This file contains the implementation of the Adaptive Instance Normalization (AdaIN) function.

Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""
from __future__ import annotations
import torch

T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T
# used only for shared attention
def concat_first(feat: T, dim=2, scale=1.) -> T: # takes K or V as input with shape = (batch, heads, tokens, dim_head)
    feat_style = expand_first(feat, scale=scale) # creates K/V of the two reference imgs (1th and middle-batch) of shape -> as above
    # feat_style holds the K/V of the first image repeated for the first half of the batch,
    # and of the middle img repeated for the second half of the batch
    return torch.cat((feat, feat_style), dim=dim) # concatenate the real K/V along the "tokens" dimensions, so that Q_target pays attention to both target and reference(s) Keys and Values tokens

# when this takes mean and std -> input shape: (batch, heads, 1, channels), see below
# when it takes K or V -> input shape: (batch, heads, tokens, dim_head), as used in "concat_first" above

def expand_first(feat: T, scale=1.,) -> T:
    b = feat.shape[0] # Extract batch size
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1) # shape: (2, 1, heads, 1, channels), stack the mean (or std) of first and middle images in the batch
    if scale == 1:
        # repeat the mean or std batch/2 times (since we are considering 2 stats, from first and middle img in the batch)
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:]) # shape: (2, batch//2, heads, 1, channels)
    else: # apply a scaling factor to the mean/std corresponding to all images except for the reference ones
        # the ref. imgs. will be normalized using THEIR unscaled stats thus they remain the SAME.
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1) # shape: (2, batch//2, heads, 1, channels) like "expand"
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape) # reshape so that first half of batch has assigned the mean/std of the first img, second half of the middle image

def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:  # computes mean and std along number of tokens dimension
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std # output shape: (batch, heads, 1, channels)

def adain(feat: T) -> T: # Input shape: (Batch, Heads, #Tokens, Channels), #Tokens is number of "pixels" in the feature map, channels = dim_head = token_dim
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std  # normalize the feature map
    feat = feat * feat_style_std + feat_style_mean  # scale and shift the feature map (reparameterization with reference stats)
    return feat