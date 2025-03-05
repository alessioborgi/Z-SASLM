"""
Tokenization_and_Embedding.py

This file contains the implementation of the Tokenization and Embedding procedure for the prompts, 
together with the Embeddings Ensemble, both with and without negative conditioning.

Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""


from __future__ import annotations
import torch
from diffusers import StableDiffusionXLPipeline

T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T

def prompt_tokenizazion_and_embedding(prompt: str, tokenizer, text_encoder, device):

    # 1) Tokenize the Input Prompt: Tokenize the input prompt using the provided tokenizer, with padding and truncation.
    prompt_tokenized = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')

    # 2) Extract Token IDs: Extract the input IDs (token indices) from the tokenized inputs.
    prompt_tokenized_ids = prompt_tokenized.input_ids

    # 3) Generate Embeddings: Use torch.no_grad() to disable gradient computation for the following operations.
    with torch.no_grad():
        # Generate embeddings for the tokenized input IDs using the text encoder.
        # The embeddings include output hidden states.
        prompt_embeddings = text_encoder(
            prompt_tokenized_ids.to(device),    # Move input IDs to the specified device (e.g., GPU).
            output_hidden_states=True,          # Request hidden states from the encoder.
        )

    # 4) Extract Pooled Output Embeddings: Extract the pooled output embeddings (first element of the tuple returned by the encoder).
    pooled_prompt_embeddings = prompt_embeddings[0]

    # 5) Extract Hidden State Embeddings: Extract the hidden state embeddings from the second last layer of the encoder.
    prompt_embeddings = prompt_embeddings.hidden_states[-2]

    # 6) Handle Empty Prompt Case: If the prompt is empty, return zero tensors as Negative Embeddings.
    if prompt == '':
        # Create a zero tensor with the same shape as the hidden state embeddings.
        negative_prompt_embeddings = torch.zeros_like(prompt_embeddings)
        # Create a zero tensor with the same shape as the pooled output embeddings.
        negative_pooled_prompt_embeddings = torch.zeros_like(pooled_prompt_embeddings)
        # Return the zero tensors for both negative embeddings and pooled negative embeddings.
        return negative_prompt_embeddings, negative_pooled_prompt_embeddings

    # 7) Returns the generated embeddings: Return the hidden state embeddings and the pooled output embeddings.
    return prompt_embeddings, pooled_prompt_embeddings






def embeddings_ensemble(model: StableDiffusionXLPipeline, prompt: str) -> tuple[dict[str, T], T]:

    # 1) Get the Device: Get the device (e.g., CPU or GPU) on which the model is being executed.
    device = model._execution_device

    # 2) Generate Text Embeddings:
    # Generate text embeddings using the first set of tokenizer and text encoder.
    prompt_embeddings, pooled_prompt_embeddings, = prompt_tokenizazion_and_embedding(prompt, model.tokenizer, model.text_encoder, device)

    # Generate text embeddings using the second set of tokenizer and text encoder.
    prompt_embeddings_2, pooled_prompt_embeds2, = prompt_tokenizazion_and_embedding( prompt, model.tokenizer_2, model.text_encoder_2, device)

    # 3) Concatenate Prompt Embeddings: Concatenate the embeddings from both sets of encoders along the last dimension.
    prompt_embeddings = torch.cat((prompt_embeddings, prompt_embeddings_2), dim=-1)

    # 4) Get Text Encoder Projection Dimension: Retrieve the projection dimension from the configuration of the second text encoder
    prompt_encoder_projection_dim = model.text_encoder_2.config.projection_dim

    # 5) Generate Additional Time IDs: Generate additional time IDs required for conditioning.
    conditioning_time_ids = model._get_add_time_ids((1024, 1024),
                                           (0, 0),
                                           (1024, 1024),
                                           torch.float16,
                                           prompt_encoder_projection_dim).to(device)

    # 6) Prepare Additional Condition Keyword Arguments: Prepare additional condition keyword arguments required for the model.
    conditioning_kwargs = {"text_embeds": pooled_prompt_embeds2, "time_ids": conditioning_time_ids}

    # 7) Return the Additional Condition Keyword Arguments and Concatenated Embeddings:Return the prepared additional condition keyword arguments and concatenated prompt embeddings
    return conditioning_kwargs, prompt_embeddings






def embeddings_ensemble_with_neg_conditioning(model: StableDiffusionXLPipeline, prompt: str) -> tuple[dict[str, T], T]:

    # 1) Encode Text with Given Prompt using Text Embedding Ensemble Encode Text with Given Prompt: Generate text embeddings and conditioning keywords for the given prompt.
    conditioning_kwargs, prompt_embeddings_concat = embeddings_ensemble(model, prompt)

    # 2) Encode Text with Empty Prompt: Generate text embeddings and conditioning keywords for an empty prompt (negative conditioning).
    unconditioning_kwargs, prompt_embeds_uncond = embeddings_ensemble(model, "")

    # 3) Concatenate Positive and Negative Embeddings: Concatenate the embeddings from the negative and positive prompts.
    prompt_embeddings_concat = torch.cat((prompt_embeds_uncond, prompt_embeddings_concat, ))

    # 4) Concatenate Positive and Negative Conditioning Keywords: Concatenate the conditioning keywords from the negative and positive prompts.
    conditioning_unconditioning_kwargs = {"text_embeds": torch.cat((unconditioning_kwargs["text_embeds"],
                                                                    conditioning_kwargs["text_embeds"])),
                         "time_ids": torch.cat((unconditioning_kwargs["time_ids"],
                                                conditioning_kwargs["time_ids"])),}

    # 5) Return Combined Conditioning Keywords and Embeddings: Return the combined conditioning keywords and concatenated embeddings.
    return conditioning_unconditioning_kwargs, prompt_embeddings_concat