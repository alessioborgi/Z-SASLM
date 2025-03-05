"""
ControlNet.py

This file contains the implementation of the ControlNet over SDXL, applying 
the Style-Alignment.

Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""

from __future__ import annotations
import torch
from PIL import Image
from typing import Any
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T



def concat_zero_control(control_reisduel: T) -> T:

    # 1) Calculate Batch Size: Determine the half batch size of the input tensor.
    b = control_reisduel.shape[0] // 2

    # 2) Create Zero Tensor: Generate a zero tensor with the same shape as one sample of the input tensor.
    zerso_reisduel = torch.zeros_like(control_reisduel[0:1])

    # 3) Concatenate Tensors: Concatenate the zero tensor to the input tensor at specific positions to create a new tensor.
    return torch.cat((zerso_reisduel, control_reisduel[:b], zerso_reisduel, control_reisduel[b:]))


@torch.no_grad()
def SDXL_ControlNet_Model(
    pipeline: StableDiffusionXLControlNetPipeline,
    prompt: str | list[str] = None,
    prompt_2: str | list[str] | None = None,
    image: PipelineImageInput = None,
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str | list[str] | None = None,
    negative_prompt_2: str | list[str] | None = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    latents: TN = None,
    prompt_embeds: TN = None,
    negative_prompt_embeds: TN = None,
    pooled_prompt_embeds: TN = None,
    negative_pooled_prompt_embeds: TN = None,
    cross_attention_kwargs: dict[str, Any] | None = None,
    controlnet_conditioning_scale: float | list[float] = 1.0,
    control_guidance_start: float | list[float] = 0.0,
    control_guidance_end: float | list[float] = 1.0,
    original_size: tuple[int, int] = None,
    crops_coords_top_left: tuple[int, int] = (0, 0),
    target_size: tuple[int, int] | None = None,
    negative_original_size: tuple[int, int] | None = None,
    negative_crops_coords_top_left: tuple[int, int] = (0, 0),
    negative_target_size:tuple[int, int] | None = None,
    clip_skip: int | None = None,
) -> list[Image]:

    # 1) Check if the controlnet module is compiled and get the original module if so.
    controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

    # Align format for control guidance parameters: Ensure control_guidance_start and control_guidance_end are lists of the same length.
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )

    # 2) Check inputs. Raise error if not correct: Validate the provided inputs to ensure they are correct and compatible.
    pipeline.check_inputs(
        prompt,
        prompt_2,
        image,
        1,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
    )

    # 3) Set the guidance scale for the pipeline: This will be used to control the influence of the guidance during the diffusion process.
    pipeline._guidance_scale = guidance_scale

    # 4) Define call parameters: Determine batch size based on the type of the prompt.
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1  # Single prompt case
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)  # Multiple prompts case
    else:
        batch_size = prompt_embeds.shape[0]  # Prompt embeddings case

    # Set the device for execution (e.g., CPU or GPU).
    device = pipeline._execution_device

    # 5) Encode input prompt: Retrieve the LoRA scale for cross attention if available.
    prompt_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    # Encode the input prompts into embeddings using the pipeline's text encoder
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt,
        prompt_2,
        device,
        1,
        True,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=prompt_encoder_lora_scale,
        clip_skip=clip_skip,
    )

    # 6) Prepare Image: Check if the controlnet model is an instance of ControlNetModel.
    if isinstance(controlnet, ControlNetModel):

        # Prepare the input image for the pipeline.
        image = pipeline.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=True,
            guess_mode=False,
        )
        # Get the height and width of the prepared image.
        height, width = image.shape[-2:]

        # Stack the image tensor to match the number of images per prompt.
        image = torch.stack([image[0]] * num_images_per_prompt + [image[1]] * num_images_per_prompt)
    else:
        assert False  # Raise an assertion error if controlnet is not an instance of ControlNetModel


    # 7) Prepare Timesteps: Set the timesteps for the diffusion process using the scheduler.
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    # Retrieve the timesteps from the scheduler.
    timesteps = pipeline.scheduler.timesteps

    # 8) Prepare latent variables: Determine the number of channels for the latent variables based on the UNet configuration.
    num_channels_latents = pipeline.unet.config.in_channels

    # Prepare the latents using the pipeline's method, which handles initialization and shape adjustments.
    latents = pipeline.prepare_latents(
        1 + num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # Optionally get Guidance Scale Embedding.
    timestep_cond = None

    # 9) Prepare extra step kwargs. Prepare additional arguments for the diffusion step, such as noise and eta values.
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    # Create tensor stating which controlnets to keep. Initialize a list to keep track of which controlnets to use at each timestep
    controlnet_keep = []
    # Loop through each timestep to determine which controlnets to use based on the guidance schedule.
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        # Append the keep value for the current timestep.
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)


    # Prepare added time ids & embeddings, and determine the original size of the image if not provided.
    if isinstance(image, list):
        original_size = original_size or image[0].shape[-2:]
    else:
        original_size = original_size or image.shape[-2:]

    # Set the target size for the image if not provided.
    target_size = target_size or (height, width)

    # Initialize additional text embeddings with pooled prompt embeddings.
    add_text_embeds = pooled_prompt_embeds

    # Determine the projection dimension for the text encoder.
    if pipeline.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

    # Generate additional time IDs based on the image and text encoder settings.
    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    # Generate negative additional time IDs if negative sizes are provided.
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipeline._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    # 10) Stack the prompt embeddings to match the number of images per prompt.
    prompt_embeds = torch.stack([prompt_embeds[0]] + [prompt_embeds[1]] * num_images_per_prompt)
    negative_prompt_embeds = torch.stack([negative_prompt_embeds[0]] + [negative_prompt_embeds[1]] * num_images_per_prompt)
    negative_pooled_prompt_embeds = torch.stack([negative_pooled_prompt_embeds[0]] + [negative_pooled_prompt_embeds[1]] * num_images_per_prompt)
    add_text_embeds = torch.stack([add_text_embeds[0]] + [add_text_embeds[1]] * num_images_per_prompt)

    # 11) Concatenate negative and positive prompt embeddings.
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    # Move the embeddings and time IDs to the appropriate device (e.g., GPU).
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(1 + num_images_per_prompt, 1)

    # Update the batch size to include the number of images per prompt.
    batch_size = num_images_per_prompt + 1


    # 12) Denoising loop: Calculate the number of warmup steps needed for the scheduler.
    warmup_steps_number = len(timesteps) - num_inference_steps * pipeline.scheduler.order

    # Check if UNet and ControlNet modules are compiled and if PyTorch version is >= 2.1.
    is_unet_compiled = is_compiled_module(pipeline.unet)
    is_controlnet_compiled = is_compiled_module(pipeline.controlnet)
    is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

    # Prepare additional conditioning arguments with text embeddings and time IDs.
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # Prepare ControlNet prompt embeddings by duplicating the prompt embeddings.
    controlnet_prompt_embeddings = torch.cat((prompt_embeds[1:batch_size], prompt_embeds[1:batch_size]))

    # Prepare ControlNet additional conditioning arguments by duplicating the items in added_cond_kwargs.
    controlnet_added_cond_kwargs = {key: torch.cat((item[1:batch_size], item[1:batch_size])) for key, item in added_cond_kwargs.items()}

    # Use the pipeline's progress bar to track progress.
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:

        # Loop through each timestep in the denoising process.
        for i, t in enumerate(timesteps):

            # Mark the beginning of a CUDA graph step if using compiled modules and PyTorch >= 2.1.
            if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                torch._inductor.cudagraph_mark_step_begin()

            # Expand the latents if using classifier-free guidance by concatenating them.
            latent_model_input = torch.cat([latents] * 2)

            # Scale the model input for the current timestep.
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # Prepare input for ControlNet inference by slicing the latent model input.
            control_model_input = torch.cat((latent_model_input[1:batch_size], latent_model_input[batch_size+1:]))

            # Determine the conditioning scale for the current timestep.
            if isinstance(controlnet_keep[i], list):
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = float(controlnet_conditioning_scale)
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]

            # Perform ControlNet inference if conditioning scale is greater than 0.
            if cond_scale > 0:
                down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeddings,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                # Ensure mid block residuals are compatible with zero control.
                mid_block_res_sample = concat_zero_control(mid_block_res_sample)
                down_block_res_samples =  [concat_zero_control(down_block_res_sample) for down_block_res_sample in down_block_res_samples]
            else:
                mid_block_res_sample = down_block_res_samples = None

            # Predict the noise residual using the UNet model.
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # Perform guidance by combining unconditional and conditional noise predictions.
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample (x_t -> x_t-1).
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # Update the progress bar after each step.
            if i == len(timesteps) - 1 or ((i + 1) > warmup_steps_number and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()



    # 13) Manually handle VAE upcasting for max memory savings: Check if VAE needs to be upcasted to float32 for stability.
    if pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast:

        pipeline.upcast_vae()  # Upcast VAE to float32.
        # Convert latents to the dtype of the VAE's post-quantization convolution parameters.
        latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)

    # Ensure the VAE is in float32 mode, as it may overflow in float16.
    needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

    # 14) Decode Latents: If upcasting is required, upcast the VAE and convert latents accordingly.
    if needs_upcasting:
        pipeline.upcast_vae()  # Upcast VAE to float32
        latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)

    # Decode the latents to generate the final image.
    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]

    # 15) Apply Watermarks: If upcasting was performed, cast the VAE back to float16.
    if needs_upcasting:
        pipeline.vae.to(dtype=torch.float16)

    # Apply watermark to the image if a watermark is provided in the pipeline.
    if pipeline.watermark is not None:
        image = pipeline.watermark.apply_watermark(image)

    # 16) Post-process Image: Post-process the image to convert it to the desired output format (e.g., PIL).
    image = pipeline.image_processor.postprocess(image, output_type='pil')

    # 17) Free Up Resources: Offload all models from memory to free up resources.
    pipeline.maybe_free_model_hooks()

    # 18) Return Final Image: Return the final processed image.
    return image

