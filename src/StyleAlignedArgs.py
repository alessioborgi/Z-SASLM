"""
StyleAlignedArgs.py

This file contains the implementation of the StyleAlignedArgs Class having all the necessary parameters.

Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 1, 2024
"""


from __future__ import annotations
import torch
from dataclasses import dataclass

T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T

@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    # Indicates whether to share group normalization across the model.
    share_layer_norm: bool = True
    # Indicates whether to share layer normalization across the model.
    share_attention: bool = True
    # Indicates whether to share attention mechanisms across the model.
    adain_queries: bool = True
    # Indicates whether to apply AdaIN (Adaptive Instance Normalization) to the queries.
    adain_keys: bool = True
    # Indicates whether to apply AdaIN to the keys.
    adain_values: bool = False
    # Indicates whether to apply AdaIN to the values.
    only_self_level: float = 0.0
    # Scale factor for the shared attn.
    style_alignment_score_scale: float = 1.0
    # Shift factor for the shared attn.
    style_alignment_score_shift: float = 0.0
