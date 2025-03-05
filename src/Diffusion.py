"""
Diffusion.py

This file contains the implementation of the Noise Prediction function to generate Latent Representations,
the Denoising Step function, and the whole DDIM Process.
Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""



from __future__ import annotations
import torch

from tqdm import tqdm
from typing import Callable
from diffusers import StableDiffusionXLPipeline

from .Tokenization_and_Embedding import embeddings_ensemble_with_neg_conditioning
from .Encode_Image import image_encoding
T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T


# Defining a type alias for the Diffusion Inversion Process type of callable.
Diff_Inversion_Process_Callback = Callable[[StableDiffusionXLPipeline, int, T, dict[str, T]], dict[str, T]]





def Generate_Noise_Prediction(model: StableDiffusionXLPipeline, latent: T, t: T, context: T, guidance_scale: float, added_cond_kwargs: dict[str, T]):

    # 1) Duplicate Latent Input: Create a batch of two identical latent representations.
    double_input_latents = torch.cat([latent] * 2)

    # 2) Generate Noise Predictions: Use the model's UNet to generate noise predictions for the duplicated latents.
    noise_prediction = model.unet(double_input_latents, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]

    # 3) Split Noise Predictions: Split the noise predictions into unconditional and conditional components.
    noise_prediction_unconditioned, noise_prediction_text = noise_prediction.chunk(2)

    # 4) Apply Guidance: Combine the unconditional and conditional noise predictions using the guidance scale.
    noise_prediction = noise_prediction_unconditioned + guidance_scale * (noise_prediction_text - noise_prediction_unconditioned)

    # 5) Return Noise Prediction: Return the combined noise prediction.
    return noise_prediction







def Denoising_next_step(model: StableDiffusionXLPipeline, model_output: T, timestep: int, sample: T) -> T:

    # 1) Calculate Current and Next Timesteps: Compute the current and next timesteps for the denoising process.
    current_timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep

    # 2) Calculate Beta Product: Compute the beta cumulative product for the current timestep.
    alpha_prod_t = model.scheduler.alphas_cumprod[int(current_timestep)] if current_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[int(next_timestep)]

    # 3) Calculate Beta Product: Compute the beta cumulative product for the current timestep.
    beta_prod_t = 1 - alpha_prod_t

    # 4) Compute Next Original Sample: Calculate the next original sample using the current sample and model output.
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    # 5) Compute Next Sample Direction: Determine the direction for the next sample.
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output

    # 6) Compute Next Sample: Combine the next original sample and next sample direction to get the next sample.
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

    # 7) Return Next Sample: Return the computed next sample.
    return next_sample








def DDIM_Process(model: StableDiffusionXLPipeline, z0, prompt, guidance_scale) -> T:

    # 1) Initialize Latent List: Start with the initial latent representation.
    latent_list = [z0]

    # 2) Encode Text with Negative Conditioning: Generate text embeddings and conditioning keywords for the prompt, including also negative conditioning.
    added_cond_kwargs, prompt_embedding = embeddings_ensemble_with_neg_conditioning(model, prompt)

    # 3) Prepare Latent for Inference: Clone and detach the initial latent, and convert it to half precision.
    latent = z0.clone().detach().half()

    # 4) Denoising Loop: Perform the denoising process over the specified number of inference steps.
    for i in tqdm(range(model.scheduler.num_inference_steps)):

        # 4.1) Get Current Timestep: Retrieve the current timestep.
        current_timestep = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]

        # 4.2) Generate Noise Prediction: Use the model to predict noise for the current latent and timestep.
        noise_prediction = Generate_Noise_Prediction(model, latent, current_timestep, prompt_embedding, guidance_scale, added_cond_kwargs)

        # 4.3) Compute Next Latent: Compute the next latent representation using the noise prediction.
        latent = Denoising_next_step(model, noise_prediction, current_timestep, latent)

        # 4.4) Append Latent to List: Append the new latent to the list of all latents.
        latent_list.append(latent)

    # 5) Return Sequence of Latents: Concatenate all latents and reverse their order.
    return torch.cat(latent_list).flip(0)






def extract_latent_and_inversion(zts, offset: int = 0) -> [T, Diff_Inversion_Process_Callback]:

    def callback_on_step_end(pipeline: StableDiffusionXLPipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:

        latents = callback_kwargs['latents']

        # Update the first latent tensor with the corresponding tensor from ddim_result.
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {'latents': latents}

    # Return the initial latent tensor and the callback function.
    return  zts[offset], callback_on_step_end






@torch.no_grad()
def DDIM_Inversion_Process(model: StableDiffusionXLPipeline, x0: np.ndarray, prompt: str, num_inference_steps: int, guidance_scale,) -> T:

    # 1) Encode Image: Encode the input image into a latent representation using the model's VAE.
    encoded_img = image_encoding(model, x0)

    # 2) Set Timesteps: Set the timesteps for the diffusion process.
    model.scheduler.set_timesteps(num_inference_steps, device=encoded_img.device)

    # 3) Perform DDIM Loop: Perform the DDIM denoising loop to generate a sequence of latent representations.
    latent_repr_sequence = DDIM_Process(model, encoded_img, prompt, guidance_scale)

    # 4) Return Sequence of Latents: Return the sequence of latent representations generated by the DDIM loop.
    return latent_repr_sequence
