"""
HarrisCorner.py

This file contains the implementation of the Harris Corner Edge Detector over an original image.
Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""

from __future__ import annotations
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T


def get_edge_map(image: Image, feature_processor: DPTImageProcessor, depth_estimator: DPTForDepthEstimation) -> Image:

    # 1) Preprocess Image: Convert the input image into a tensor suitable for processing by the depth estimation model.
    processed_img = feature_processor(images=image, return_tensors="pt").pixel_values.to("cuda")

    # 2) Estimate Depth Map: Use the depth estimation model to predict the depth map for the input image.
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(processed_img).predicted_depth

    # 3) Interpolate Depth Map: Resize the depth map to the desired dimensions using bicubic interpolation.
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )

    # 4) Normalize Depth Map: Normalize the depth map values to a range between 0 and 1.
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)

    # 5) Convert Depth Map to RGB Image: Convert the normalized depth map into an RGB image format for visualization.
    depth_rgb_image = torch.cat([depth_map_normalized] * 3, dim=1)
    depth_rgb_image = depth_rgb_image.permute(0, 2, 3, 1).cpu().numpy()[0]
    depth_rgb_image = (depth_rgb_image * 255.0).clip(0, 255).astype(np.uint8)
    depth_rgb_image = Image.fromarray(depth_rgb_image)

    # 6) Convert to Grayscale and Apply Canny Edge Detection
    depth_gray = cv2.cvtColor(np.array(depth_rgb_image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(depth_gray, 100, 200)  # Adjust thresholds as needed

    # Convert edges back to PIL Image format
    edge_map = Image.fromarray(edges)

    return edge_map
