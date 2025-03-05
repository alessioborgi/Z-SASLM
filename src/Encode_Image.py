"""
Encode_Image.py

This file contains the implementation of the Image Encoding function.
Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""


from __future__ import annotations
import torch
import numpy as np
from src.Handler import Handler
from diffusers import StableDiffusionXLPipeline
from src.StyleAlignedArgs import StyleAlignedArgs

# 1) Normal Painting
# These are some parameters you can Adjust to Control StyleAlignment to Reference Image.
style_alignment_score_shift_normal = np.log(2)  # higher value induces higher fidelity, set 0 for no shift
style_alignment_score_scale_normal = 1.0  # higher value induces higher, set 1 for no rescale

# 2) Very Famous Paintings
style_alignment_score_shift_famous = np.log(1)
style_alignment_score_scale_famous = 0.5

normal_sa_args = StyleAlignedArgs(
    share_group_norm=True,
    share_layer_norm=True,
    share_attention=True,
    adain_queries=True,
    adain_keys=True,
    adain_values=False,
    style_alignment_score_shift=style_alignment_score_shift_normal,
    style_alignment_score_scale=style_alignment_score_scale_normal)


famous_sa_args = StyleAlignedArgs(
    share_group_norm=True,
    share_layer_norm=True,
    share_attention=True,
    adain_queries=True,
    adain_keys=True,
    adain_values=False,
    style_alignment_score_shift=style_alignment_score_shift_famous,
    style_alignment_score_scale=style_alignment_score_scale_famous)



############ SINGLE-STYLE REFERENCE #################################

def image_encoding(model: StableDiffusionXLPipeline, image: np.ndarray) -> T:

    # 1) Set VAE to Float32: Ensure the VAE operates in float32 precision for encoding.
    model.vae.to(dtype=torch.float32)

    # 2) Convert Image to PyTorch Tensor: Convert the input image from a numpy array to a PyTorch tensor and normalize pixel values to [0, 1].
    scaled_image = torch.from_numpy(image).float() / 255.

    # 3) Normalize and Prepare Image: Scale pixel values to the range [-1, 1], rearrange dimensions, and add batch dimension.
    permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

    # 4) Encode Image Using VAE: Use the VAE to encode the image into the latent space.
    latent_img = model.vae.encode(permuted_image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor

    # 5) Reset VAE to Float16: Optionally reset the VAE to float16 precision.
    model.vae.to(dtype=torch.float16)

    # 6) Return Latent Representation: Return the encoded latent representation of the image.
    return latent_img


############ MULTI-STYLE REFERENCE: LINEAR WEIGHTED AVERAGE #################################
def images_encoding(model, images: list[np.ndarray], blending_weights: list[float], normal_famous_scaling: list[str], handler: Handler):
    """
    Encode a list of images using the VAE model and blend their latent representations
    according to the given blending_weights.

    Args:
    - model: The StableDiffusionXLPipeline model.
    - images: A list of numpy arrays, each representing an image.
    - blending_weights: A list of floats representing the blending weights for each image.
              The blending_weights should sum to 1.

    Returns:
    - blended_latent_img: The blended latent representation.
    """

    # Ensure the blending_weights sum to 1.
    assert len(images) == len(blending_weights), "The number of images and blending_weights must match."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."
    assert len(normal_famous_scaling) == len(images), "The number of scaling classifications must match the number of images."

    # Set VAE to Float32 for encoding.
    model.vae.to(dtype=torch.float32)

    # Initialize blended latent representation as None.
    blended_latent_img = None

    for img, weight, scaling_type in zip(images, blending_weights, normal_famous_scaling):
        
        # Set VAE to Float32 for encoding.
        model.vae.to(dtype=torch.float32)
        # Check if the weight is greater than 0.
        if weight > 0.0:

            # Apply the style arguments dynamically
            if scaling_type == "n":
                handler.register(normal_sa_args)
            elif scaling_type == "f":
                handler.register(famous_sa_args)
            else:
                raise ValueError(f"Invalid scaling type: {scaling_type}")

            # Convert image to PyTorch tensor and normalize pixel values to [0, 1].
            scaled_image = torch.from_numpy(img).float() / 255.

            # Normalize and prepare image.
            permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

            # Encode image using VAE.
            latent_img = model.vae.encode(permuted_image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor

            # Blend the latent representation based on the weight.
            if blended_latent_img is None:
                blended_latent_img = latent_img * weight
            else:
                blended_latent_img += latent_img * weight

    # Reset VAE to Float16 if necessary.
    model.vae.to(dtype=torch.float16)

    # Return the blended latent representation.
    return blended_latent_img







############ MULTI-STYLE REFERENCE: WEIGHTED SLERP (SPHERICAL LINEAR INTERPOLATION) ###

def weighted_slerp(weight, v0, v1):
    """Spherical linear interpolation with a weight factor."""
    v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
    dot_product = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0 - weight) * omega) / sin_omega) * v0 + (torch.sin(weight * omega) / sin_omega) * v1


def images_encoding_slerp(model, images: list[np.ndarray], blending_weights: list[float], normal_famous_scaling: list[str], handler: Handler):
    """
    Encode a list of images using the VAE model and blend their latent representations
    using Weighted Spherical Interpolation (slerp) according to the given blending_weights.

    Args:
    - model: The StableDiffusionXLPipeline model.
    - images: A list of numpy arrays, each representing an image.
    - blending_weights: A list of floats representing the blending weights for each image.
                        The blending_weights should sum to 1.
    - sa_args_list: A list of StyleAlignedArgs for style alignment.
    - normal_famous_scaling: A list of classifications ("n" for normal, "f" for famous) for each image.

    Returns:
    - blended_latent_img: The blended latent representation.
    """

    # Ensure the blending_weights sum to 1.
    assert len(images) == len(blending_weights), "The number of images and blending_weights must match."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."
    assert len(normal_famous_scaling) == len(images), "The number of scaling classifications must match the number of images."

    # Initialize variables to store valid latents and corresponding weights
    valid_latents = []
    valid_weights = []

    # Iterate over images and weights
    for idx, (img, weight, scaling_type) in enumerate(zip(images, blending_weights, normal_famous_scaling)):
        
        model.vae.to(dtype=torch.float32)
        if weight > 0.0:

            # Apply the style arguments dynamically
            if scaling_type == "n":
                handler.register(normal_sa_args)
            elif scaling_type == "f":
                handler.register(famous_sa_args)
            else:
                raise ValueError(f"Invalid scaling type: {scaling_type}")

            # Convert image to PyTorch tensor and normalize pixel values to [0, 1].
            scaled_image = torch.from_numpy(img).float() / 255.

            # Normalize and prepare image.
            permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

            # Encode image using VAE.
            latent_img = model.vae.encode(permuted_image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor

            # Store valid latent and weight
            valid_latents.append(latent_img)
            valid_weights.append(weight)

    # Convert valid_weights to tensor and normalize them
    valid_weights = torch.tensor(valid_weights, device=model.vae.device, dtype=torch.float32)
    valid_weights = valid_weights / valid_weights.sum()

    # Perform SLERP only if there are valid latents
    if len(valid_latents) == 1:
        # If there's only one valid latent, no need to interpolate, just return it scaled by the weight
        blended_latent_img = valid_latents[0] * valid_weights[0]
    else:
        # Perform SLERP for multiple valid latents
        blended_latent_img = valid_latents[0] * valid_weights[0]
        for i in range(1, len(valid_latents)):
            blended_latent_img = weighted_slerp(valid_weights[i], blended_latent_img, valid_latents[i])

    # Reset VAE to Float16 if necessary.
    model.vae.to(dtype=torch.float16)

    # Return the blended latent representation.
    return blended_latent_img



# ############ MULTI-STYLE REFERENCE: (EBI) WEIGHTED EUCLIDEAN BARYCENTER INTERPOLATION(FRECHET MEAN IN R^n) ###

def euclidean_barycenter(valid_latents, weights):
    """
    Compute the weighted Euclidean barycenter of latent vectors.
    
    Args:

        valid_latents: List of torch tensors of latent representations.
                       Each tensor has shape (1, C, H, W).
        weights: 1D tensor of blending weights (summing to 1) for each latent.
    Returns:
        The blended latent representation, reshaped to the original latent shape.
    """
    # Flatten each latent into a vector
    X = torch.stack([latent.view(-1) for latent in valid_latents], dim=0)  # shape: (n, d)

    # Weighted sum in Euclidean space
    w = weights.view(-1, 1)  # shape: (n, 1)
    m = (w * X).sum(dim=0, keepdim=True)  # shape: (1, d)

    # Reshape the barycenter to the original latent shape
    original_shape = valid_latents[0].shape
    return m.view(original_shape)


def images_encoding_ebi(model, images: list[np.ndarray], blending_weights: list[float],
                        normal_famous_scaling: list[str], handler):
    """
    Encode a list of images using the VAE model and blend their latent representations
    using a Euclidean Barycenter in R^n (instead of the spherical approach).

    Args:
        model: The StableDiffusionXLPipeline model.
        images: A list of numpy arrays, each representing an image.
        blending_weights: A list of floats representing the blending weights for each image.
                          These should sum to 1.
        normal_famous_scaling: A list of classifications ("n" for normal, "f" for famous) for each image.
        handler: An instance for handling style arguments (assumed to have a `register` method).

    Returns:
        blended_latent_img: The blended latent representation.
    """
    # Check that inputs are consistent.
    assert len(images) == len(blending_weights), "Mismatch between number of images and blending_weights."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."
    assert len(normal_famous_scaling) == len(images), "Mismatch between scaling classifications and images."

    valid_latents = []
    valid_weights = []

    # Process each image
    for img, weight, scaling_type in zip(images, blending_weights, normal_famous_scaling):
        model.vae.to(dtype=torch.float32)
        if weight > 0.0:
            # Dynamically apply the style arguments based on scaling type
            if scaling_type == "n":
                handler.register(normal_sa_args)
            elif scaling_type == "f":
                handler.register(famous_sa_args)
            else:
                raise ValueError(f"Invalid scaling type: {scaling_type}")

            # Convert image to a PyTorch tensor and normalize pixel values
            scaled_image = torch.from_numpy(img).float() / 255.
            permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

            # Encode image using the VAE and scale accordingly
            latent_output = model.vae.encode(permuted_image.to(model.vae.device))
            latent_img = latent_output['latent_dist'].mean * model.vae.config.scaling_factor

            valid_latents.append(latent_img)
            valid_weights.append(weight)

    if not valid_latents:
        raise ValueError("No valid latent representations obtained.")

    # Convert weights to a tensor and normalize them
    valid_weights_tensor = torch.tensor(valid_weights, device=model.vae.device, dtype=torch.float32)
    valid_weights_tensor = valid_weights_tensor / valid_weights_tensor.sum()

    # Compute the Euclidean barycenter of the latent representations
    blended_latent_img = euclidean_barycenter(valid_latents, valid_weights_tensor)

    # Optionally, reset VAE to float16 precision
    model.vae.to(dtype=torch.float16)

    return blended_latent_img


# ############ MULTI-STYLE REFERENCE: (REBI)  WEIGHTED ROBUST EUCLIDEAN BARYCENTER INTERPOLATION(FRECHET MEAN IN R^n) ###


def robust_euclidean_barycenter(valid_latents, weights, delta=1.0, num_iterations=10):
    """
    Compute a robust weighted Euclidean barycenter of latent vectors using an iterative reweighting scheme.
    
    Args:
        valid_latents: List of torch tensors of latent representations.
                       Each tensor has shape (1, C, H, W).
        weights: 1D tensor of blending weights (summing to 1) for each latent.
        delta: Hyperparameter controlling the threshold for robust weighting.
        num_iterations: Maximum number of iterations for the reweighting procedure.
    
    Returns:
        The robust blended latent representation, reshaped to the original latent shape.
    """
    # Flatten each latent into a vector.
    X = torch.stack([latent.view(-1) for latent in valid_latents], dim=0)  # shape: (n, d)
    # Initialize with the original weights.
    combined_weights = weights.clone()
    combined_weights = combined_weights / combined_weights.sum()
    # Initial barycenter.
    w = combined_weights.view(-1, 1)
    m = (w * X).sum(dim=0, keepdim=True)  # shape: (1, d)
    
    for _ in range(num_iterations):
        # Compute Euclidean distances of each latent from current barycenter.
        distances = torch.norm(X - m, dim=1)  # shape: (n,)
        # Compute robust reweighting factors: if distance < delta, use 1; else delta/distance.
        robust_factors = torch.where(distances < delta,
                                     torch.ones_like(distances),
                                     delta / (distances + 1e-8))
        # Combine original weights with robust factors.
        new_weights = combined_weights * robust_factors
        new_weights = new_weights / new_weights.sum()
        w_new = new_weights.view(-1, 1)
        m_new = (w_new * X).sum(dim=0, keepdim=True)
        # Check for convergence.
        if torch.norm(m_new - m) < 1e-6:
            m = m_new
            break
        m = m_new
        combined_weights = new_weights

    original_shape = valid_latents[0].shape
    return m.view(original_shape)


def images_encoding_rebi(model: StableDiffusionXLPipeline,
                        images: list[np.ndarray],
                        blending_weights: list[float],
                        normal_famous_scaling: list[str],
                        handler: Handler):
    """
    Encode a list of images using the VAE model and blend their latent representations
    using a robust Euclidean barycenter in R^n.
    
    Args:
        model: The StableDiffusionXLPipeline model.
        images: A list of numpy arrays, each representing an image.
        blending_weights: A list of floats representing the blending weights for each image.
                          These should sum to 1.
        normal_famous_scaling: A list of classifications ("n" for normal, "f" for famous) for each image.
        handler: An instance for handling style arguments (assumed to have a `register` method).
    
    Returns:
        blended_latent_img: The robust blended latent representation.
    """
    # Validate input lengths.
    assert len(images) == len(blending_weights), "Mismatch between number of images and blending_weights."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."
    assert len(normal_famous_scaling) == len(images), "Mismatch between scaling classifications and images."

    valid_latents = []
    valid_weights = []

    # Process each image.
    for img, weight, scaling_type in zip(images, blending_weights, normal_famous_scaling):
        model.vae.to(dtype=torch.float32)
        if weight > 0.0:
            # Apply style arguments based on scaling type.
            if scaling_type == "n":
                handler.register(normal_sa_args)
            elif scaling_type == "f":
                handler.register(famous_sa_args)
            else:
                raise ValueError(f"Invalid scaling type: {scaling_type}")

            # Convert image to a PyTorch tensor and normalize pixel values.
            scaled_image = torch.from_numpy(img).float() / 255.
            permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            # Encode image using the VAE and scale accordingly.
            latent_output = model.vae.encode(permuted_image.to(model.vae.device))
            latent_img = latent_output['latent_dist'].mean * model.vae.config.scaling_factor

            valid_latents.append(latent_img)
            valid_weights.append(weight)

    if not valid_latents:
        raise ValueError("No valid latent representations obtained.")

    # Convert weights to a tensor and normalize.
    valid_weights_tensor = torch.tensor(valid_weights, device=model.vae.device, dtype=torch.float32)
    valid_weights_tensor = valid_weights_tensor / valid_weights_tensor.sum()

    # Compute the robust Euclidean barycenter of the latent representations.
    blended_latent_img = robust_euclidean_barycenter(valid_latents, valid_weights_tensor)

    # Optionally, reset VAE to float16 precision.
    model.vae.to(dtype=torch.float16)

    return blended_latent_img




# ############ MULTI-STYLE REFERENCE: (REBIGRAM) ITERATIVE REWEIGHTED ROBUST EUCLIDEAN BARYCENTER WITH SUBSEQUENT STYLE REFINEMENT VIA GRAM MATRIX (FRECHET MEAN IN R^n) ###
###############################################################################
# Robust Euclidean Barycenter with Iterative Reweighting
###############################################################################
def robust_euclidean_barycenter(valid_latents, weights, delta=1.0, num_iterations=10):
    """
    Compute a robust weighted Euclidean barycenter of latent vectors using an iterative
    reweighting scheme.
    
    Args:
        valid_latents: List of torch tensors of latent representations, each of shape (1, C, H, W).
        weights: 1D tensor of blending weights (summing to 1).
        delta: Hyperparameter controlling the threshold for robust weighting.
        num_iterations: Number of iterations for reweighting.
    
    Returns:
        The robust blended latent representation, reshaped to the original latent shape.
    """
    # Flatten each latent into a vector.
    X = torch.stack([latent.view(-1) for latent in valid_latents], dim=0)  # (n, d)
    combined_weights = weights.clone() / weights.sum()
    m = (combined_weights.view(-1, 1) * X).sum(dim=0, keepdim=True)  # initial barycenter

    for _ in range(num_iterations):
        distances = torch.norm(X - m, dim=1)  # Euclidean distances
        robust_factors = torch.where(distances < delta,
                                     torch.ones_like(distances),
                                     delta / (distances + 1e-8))
        new_weights = combined_weights * robust_factors
        new_weights = new_weights / new_weights.sum()
        m_new = (new_weights.view(-1, 1) * X).sum(dim=0, keepdim=True)
        if torch.norm(m_new - m) < 1e-6:
            m = m_new
            break
        m = m_new
        combined_weights = new_weights

    original_shape = valid_latents[0].shape
    return m.view(original_shape)

###############################################################################
# Gram Matrix Computation for Style Representation
###############################################################################
def compute_gram_matrix(feature):
    """
    Compute the Gram matrix for a feature map.
    
    Args:
        feature: A tensor of shape (1, C, H, W).
    
    Returns:
        Gram matrix of shape (C, C).
    """
    b, c, h, w = feature.shape
    feature = feature.view(c, h * w)
    gram = torch.mm(feature, feature.t()) / (c * h * w)
    return gram

###############################################################################
# Style Refinement via Gradient Descent
###############################################################################
def style_refine(latent, target_gram, num_steps=100, lr=0.01):
    """
    Refine the latent representation to better match the target style (via Gram matrix).
    
    Args:
        latent: The initial blended latent tensor, shape (1, C, H, W).
        target_gram: The target Gram matrix.
        num_steps: Number of optimization steps.
        lr: Learning rate.
    
    Returns:
        The refined latent representation.
    """
    # Ensure latent is float32 and a leaf tensor requiring gradients.
    latent_refined = latent.clone().detach().float()
    latent_refined.requires_grad_(True)
    
    optimizer = torch.optim.Adam([latent_refined], lr=lr)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        current_gram = compute_gram_matrix(latent_refined)
        loss = torch.nn.functional.mse_loss(current_gram, target_gram)
        loss.backward()
        optimizer.step()
    return latent_refined.detach()

###############################################################################
# Advanced Multi-Style Reference Blending (REBIGRAM)
###############################################################################
def images_encoding_rebigram(model: StableDiffusionXLPipeline,
                             images: list[np.ndarray],
                             blending_weights: list[float],
                             normal_famous_scaling: list[str],
                             handler: Handler):
    """
    Encode a list of images using the VAE model and blend their latent representations
    using a robust Euclidean barycenter with subsequent style refinement via Gram matrix matching.
    
    Args:
        model: The StableDiffusionXLPipeline model.
        images: List of numpy arrays, each representing an image.
        blending_weights: List of floats (should sum to 1).
        normal_famous_scaling: List of classifications ("n" for normal, "f" for famous).
        handler: An instance for handling style arguments.
    
    Returns:
        The advanced blended latent representation.
    """
    assert len(images) == len(blending_weights), "Mismatch between images and blending_weights."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."
    assert len(normal_famous_scaling) == len(images), "Mismatch between scaling classifications and images."

    valid_latents = []
    valid_weights = []
    gram_matrices = []

    # Process each image.
    for img, weight, scaling_type in zip(images, blending_weights, normal_famous_scaling):
        model.vae.to(dtype=torch.float32)
        if weight > 0.0:
            if scaling_type == "n":
                handler.register(normal_sa_args)
            elif scaling_type == "f":
                handler.register(famous_sa_args)
            else:
                raise ValueError(f"Invalid scaling type: {scaling_type}")
            
            scaled_image = torch.from_numpy(img).float() / 255.
            permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            latent_output = model.vae.encode(permuted_image.to(model.vae.device))
            latent_img = latent_output['latent_dist'].mean * model.vae.config.scaling_factor

            valid_latents.append(latent_img)
            valid_weights.append(weight)
            gram_matrices.append(compute_gram_matrix(latent_img))
    
    if not valid_latents:
        raise ValueError("No valid latent representations obtained.")

    valid_weights_tensor = torch.tensor(valid_weights, device=model.vae.device, dtype=torch.float32)
    valid_weights_tensor = valid_weights_tensor / valid_weights_tensor.sum()
    
    # Step 1: Compute robust global barycenter.
    blended_latent = robust_euclidean_barycenter(valid_latents, valid_weights_tensor)
    
    # Step 2: Blend the style information (Gram matrices) as a weighted sum.
    blended_gram = sum(w * g for w, g in zip(valid_weights, gram_matrices))
    blended_gram = blended_gram / sum(valid_weights)
    
    # Step 3: Refine the global barycenter latent to better match the blended style.
    with torch.enable_grad():
        refined_latent = style_refine(blended_latent, blended_gram, num_steps=100, lr=0.01)
    
    model.vae.to(dtype=torch.float16)
    return refined_latent



# ############ MULTI-STYLE REFERENCE: (REBIGRAM++) ITERATIVE REWEIGHTED ROBUST EUCLIDEAN BARYCENTER WITH SUBSEQUENT STYLE REFINEMENT VIA GRAM MATRIX (FRECHET MEAN IN R^n) ###
# -----------------------------------------------------------------------------
# Robust Euclidean Barycenter with Iterative Reweighting (REBIGRAM)
# -----------------------------------------------------------------------------
def robust_euclidean_barycenter(valid_latents, weights, delta=1.0, num_iterations=10):
    """
    Compute a robust weighted Euclidean barycenter of latent vectors using an iterative reweighting scheme.
    
    Args:
        valid_latents: List of torch tensors of latent representations, each with shape (1, C, H, W).
        weights: 1D tensor of blending weights (summing to 1).
        delta: Hyperparameter controlling the threshold for robust weighting.
        num_iterations: Number of iterations for reweighting.
    
    Returns:
        The robust blended latent representation, reshaped to the original latent shape.
    """
    X = torch.stack([latent.view(-1) for latent in valid_latents], dim=0)  # (n, d)
    combined_weights = weights.clone() / weights.sum()
    m = (combined_weights.view(-1, 1) * X).sum(dim=0, keepdim=True)
    
    for _ in range(num_iterations):
        distances = torch.norm(X - m, dim=1)
        robust_factors = torch.where(distances < delta,
                                     torch.ones_like(distances),
                                     delta / (distances + 1e-8))
        new_weights = combined_weights * robust_factors
        new_weights = new_weights / new_weights.sum()
        m_new = (new_weights.view(-1, 1) * X).sum(dim=0, keepdim=True)
        if torch.norm(m_new - m) < 1e-6:
            m = m_new
            break
        m = m_new
        combined_weights = new_weights

    original_shape = valid_latents[0].shape
    return m.view(original_shape)

# -----------------------------------------------------------------------------
# Gram Matrix Computation for Style Representation
# -----------------------------------------------------------------------------
def compute_gram_matrix(feature):
    """
    Compute the Gram matrix for a feature map.
    
    Args:
        feature: Tensor of shape (1, C, H, W).
    
    Returns:
        Gram matrix of shape (C, C).
    """
    b, c, h, w = feature.shape
    feature = feature.view(c, h * w)
    gram = torch.mm(feature, feature.t()) / (c * h * w)
    return gram

# -----------------------------------------------------------------------------
# Advanced Feature Extractor using CLIP (default extractor on CPU)
# -----------------------------------------------------------------------------
# def extract_features(decoded_image):
#     """
#     Extract content features from a decoded image using a pretrained CLIP model,
#     running the CLIP encoder entirely on CPU to reduce GPU memory usage.
    
#     Args:
#         decoded_image: A torch.Tensor of shape (B, C, H, W) with pixel values in [0, 1],
#                        or a DecoderOutput with a 'sample' attribute.
    
#     Returns:
#         The CLIP image features (on CPU).
#     """
#     import clip
#     # Ensure decoded_image is a tensor.
#     if not isinstance(decoded_image, torch.Tensor):
#         if hasattr(decoded_image, 'sample'):
#             decoded_image = decoded_image.sample
#         elif isinstance(decoded_image, dict) and 'sample' in decoded_image:
#             decoded_image = decoded_image['sample']
#         else:
#             raise ValueError("Decoded image is not a tensor and cannot be processed.")
    
#     device_cpu = torch.device("cpu")
#     global clip_model
#     if 'clip_model' not in globals():
#         clip_model, _ = clip.load("ViT-B/32", device=device_cpu, jit=False)
#         clip_model.eval()
#     # Move decoded image to CPU.
#     decoded_cpu = decoded_image.to(device_cpu)
#     resized = torch.nn.functional.interpolate(decoded_cpu, size=(224, 224), mode='bilinear', align_corners=False)
#     mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device_cpu).view(1, 3, 1, 1)
#     std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device_cpu).view(1, 3, 1, 1)
#     normed = (resized - mean) / std
#     with torch.no_grad():
#         features = clip_model.encode_image(normed)
#     return features  # remains on CPU

def extract_features(decoded_image):
    """
    Extract content features from a decoded image using a pretrained CLIP model,
    running the CLIP encoder entirely on CPU to reduce GPU memory usage.
    
    Args:
        decoded_image: A torch.Tensor of shape (B, C, H, W) with pixel values in [0, 1],
                       or a DecoderOutput with a 'sample' attribute.
    
    Returns:
        The CLIP image features (on CPU).
    """
    import clip
    # Ensure decoded_image is a tensor.
    if not isinstance(decoded_image, torch.Tensor):
        if hasattr(decoded_image, 'sample'):
            decoded_image = decoded_image.sample
        elif isinstance(decoded_image, dict) and 'sample' in decoded_image:
            decoded_image = decoded_image['sample']
        else:
            raise ValueError("Decoded image is not a tensor and cannot be processed.")
    
    device_cpu = torch.device("cpu")
    global clip_model
    if 'clip_model' not in globals():
        clip_model, _ = clip.load("ViT-B/32", device=device_cpu, jit=False)
        clip_model.eval()
    # Move decoded image to CPU and cast to float32.
    decoded_cpu = decoded_image.to(device_cpu).float()
    resized = torch.nn.functional.interpolate(decoded_cpu, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device_cpu).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device_cpu).view(1, 3, 1, 1)
    normed = (resized - mean) / std
    with torch.no_grad():
        features = clip_model.encode_image(normed)
    return features


# -----------------------------------------------------------------------------
# Advanced Style Refinement via Gradient Descent (REBIGRAM++)
# -----------------------------------------------------------------------------
# def style_refine_advanced(latent, target_gram, target_content, extract_features, model, 
#                             num_steps=100, lr=0.01, tv_weight=0.001, content_weight=0.1):
#     """
#     Refine the latent representation to better match the target style and content,
#     by performing refinement on CPU using a temporary copy of the VAE.
    
#     The loss is a weighted sum of:
#       - Style loss (MSE between the latent's Gram matrix and target Gram matrix)
#       - Content loss (MSE between extracted features of the decoded latent and target content features)
#       - Total Variation (TV) loss for spatial smoothness.
    
#     Args:
#         latent: The initial blended latent tensor, shape (1, C, H, W), on the original device.
#         target_gram: The target Gram matrix for style.
#         target_content: The target content features.
#         extract_features: A function that extracts content features from a decoded image.
#         model: The StableDiffusionXLPipeline model.
#         num_steps: Number of optimization steps.
#         lr: Learning rate.
#         tv_weight: Weight for TV loss.
#         content_weight: Weight for content loss.
    
#     Returns:
#         The refined latent representation on the original device.
#     """
#     # Store the original device (likely cuda:0).
#     original_device = model.vae.device
#     # Create a temporary copy of the VAE on CPU for the refinement.
#     vae_cpu = model.vae.to("cpu")
    
#     # Move the latent to CPU for refinement.
#     latent_refined = latent.clone().detach().float().to("cpu")
#     latent_refined.requires_grad_(True)
    
#     optimizer = torch.optim.Adam([latent_refined], lr=lr)
    
#     def total_variation_loss(img):
#         return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
#                torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    
#     for step in range(num_steps):
#         optimizer.zero_grad()
#         current_gram = compute_gram_matrix(latent_refined)
#         style_loss = torch.nn.functional.mse_loss(current_gram, target_gram.to("cpu"))
        
#         decoded = vae_cpu.decode(latent_refined)
#         # If decoded output is not a tensor, try to extract from .sample or dict.
#         if not isinstance(decoded, torch.Tensor):
#             if hasattr(decoded, "sample"):
#                 decoded = decoded.sample
#             elif isinstance(decoded, dict) and "sample" in decoded:
#                 decoded = decoded["sample"]
#         current_features = extract_features(decoded)  # Extracted on CPU.
#         content_loss = torch.nn.functional.mse_loss(current_features, target_content.to("cpu"))
        
#         tv_loss = total_variation_loss(latent_refined)
#         loss = style_loss + content_weight * content_loss + tv_weight * tv_loss
#         loss.backward()
#         optimizer.step()
    
#     refined = latent_refined.detach().to(original_device)
#     # Restore the original VAE on the original device.
#     # (We don't modify the global model.unet or other parts.)
#     vae_cpu.to(original_device)
#     return refined

from torch.utils.checkpoint import checkpoint

def style_refine_advanced(latent, target_gram, target_content, extract_features, model, 
                            num_steps=20, lr=0.01, tv_weight=0.001, content_weight=0.1):
    """
    Refine the latent representation to better match the target style and content,
    using mixed precision on GPU and checkpointing the VAE decoder to reduce memory usage.
    
    The loss is a weighted sum of:
      - Style loss (MSE between the latent's Gram matrix and target Gram matrix)
      - Content loss (MSE between extracted features of the decoded latent and target content features)
      - Total Variation (TV) loss for spatial smoothness.
    
    Args:
        latent: The initial blended latent tensor, shape (1, C, H, W) on GPU.
        target_gram: The target Gram matrix for style.
        target_content: The target content features.
        extract_features: Function to extract content features from a decoded image.
        model: The StableDiffusionXLPipeline model.
        num_steps: Number of optimization steps.
        lr: Learning rate.
        tv_weight: Weight for TV loss.
        content_weight: Weight for content loss.
    
    Returns:
        The refined latent representation (on the original GPU device).
    """
    latent_refined = latent.clone().detach().float()
    latent_refined.requires_grad_(True)
    
    optimizer = torch.optim.Adam([latent_refined], lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    
    def total_variation_loss(img):
        return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
               torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    
    for step in range(num_steps):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            current_gram = compute_gram_matrix(latent_refined)
            style_loss = torch.nn.functional.mse_loss(current_gram, target_gram.to(latent_refined.device))
            
            # Use checkpointing to wrap the VAE decoder.
            decoded = checkpoint(lambda x: model.vae.decode(x), latent_refined)
            if not isinstance(decoded, torch.Tensor):
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample
                elif isinstance(decoded, dict) and "sample" in decoded:
                    decoded = decoded["sample"]
            
            # Downsample the decoded image to 224x224 for feature extraction.
            decoded_down = torch.nn.functional.interpolate(decoded, size=(224, 224),
                                                             mode='bilinear', align_corners=False)
            # Offload the heavy CLIP extraction to CPU.
            decoded_cpu = decoded_down.detach().to("cpu")
            current_features = extract_features(decoded_cpu)
            current_features = current_features.to(latent_refined.device)
            content_loss = torch.nn.functional.mse_loss(current_features, target_content.to(latent_refined.device))
            
            tv_loss = total_variation_loss(latent_refined)
            loss = style_loss + content_weight * content_loss + tv_weight * tv_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % 5 == 0:
            torch.cuda.empty_cache()
    
    return latent_refined.detach()



# -----------------------------------------------------------------------------
# Advanced Multi-Style Reference Blending (REBIGRAM++)
# -----------------------------------------------------------------------------
def images_encoding_rebigram_plus_plus(model: StableDiffusionXLPipeline,
                                       images: list[np.ndarray],
                                       blending_weights: list[float],
                                       normal_famous_scaling: list[str],
                                       handler: Handler,
                                       extract_features_func=extract_features):
    """
    Encode multiple images using the VAE and blend their latent representations using
    a robust Euclidean barycenter with subsequent multi-objective style refinement.
    
    REBIGRAM++ computes:
      1. A robust global barycenter of the latents.
      2. A weighted blended Gram matrix (style target).
      3. A weighted blended content target computed by decoding each latent and extracting features.
      4. Refines the barycenter latent using style, content, and total variation losses.
    
    Args:
        model: The StableDiffusionXLPipeline model.
        images: List of numpy arrays, each representing an image.
        blending_weights: List of floats (should sum to 1).
        normal_famous_scaling: List of classifications ("n" for normal, "f" for famous) for each image.
        handler: An instance for handling style arguments.
        extract_features_func: Function to extract content features from a decoded image.
            Defaults to our built-in CLIP-based extractor (which runs on CPU).
    
    Returns:
        The advanced blended latent representation (REBIGRAM++).
    """
    # Validate input lengths.
    assert len(images) == len(blending_weights), "Mismatch between images and blending_weights."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."
    assert len(normal_famous_scaling) == len(images), "Mismatch between scaling classifications and images."

    valid_latents = []
    valid_weights = []
    gram_matrices = []
    content_features = []

    # Process each image.
    for img, weight, scaling_type in zip(images, blending_weights, normal_famous_scaling):
        model.vae.to(dtype=torch.float32)
        if weight > 0.0:
            if scaling_type == "n":
                handler.register(normal_sa_args)
            elif scaling_type == "f":
                handler.register(famous_sa_args)
            else:
                raise ValueError(f"Invalid scaling type: {scaling_type}")
            
            scaled_image = torch.from_numpy(img).float() / 255.
            permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            latent_output = model.vae.encode(permuted_image.to(model.vae.device))
            latent_img = latent_output['latent_dist'].mean * model.vae.config.scaling_factor

            valid_latents.append(latent_img)
            valid_weights.append(weight)
            gram_matrices.append(compute_gram_matrix(latent_img))
            
            # Decode the latent to get a reference image, then extract content features.
            decoded_img = model.vae.decode(latent_img)
            content_features.append(extract_features_func(decoded_img))
    
    if not valid_latents:
        raise ValueError("No valid latent representations obtained.")

    valid_weights_tensor = torch.tensor(valid_weights, device=model.vae.device, dtype=torch.float32)
    valid_weights_tensor = valid_weights_tensor / valid_weights_tensor.sum()
    
    # Step 1: Compute robust global barycenter.
    blended_latent = robust_euclidean_barycenter(valid_latents, valid_weights_tensor)
    
    # Step 2: Compute blended style (Gram) target.
    blended_gram = sum(w * g for w, g in zip(valid_weights, gram_matrices))
    blended_gram = blended_gram / sum(valid_weights)
    
    # Step 3: Compute blended content target.
    blended_content = sum(w * f for w, f in zip(valid_weights, content_features))
    blended_content = blended_content / sum(valid_weights)
    
    # Step 4: Refine the barycenter latent using advanced multi-objective loss.
    with torch.enable_grad():
        refined_latent = style_refine_advanced(blended_latent, blended_gram, blended_content,
                                               extract_features_func, model, num_steps=100, lr=0.01,
                                               tv_weight=0.001, content_weight=0.1)
    
    model.vae.to(dtype=torch.float16)
    return refined_latent