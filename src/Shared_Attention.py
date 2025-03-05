"""
Shared_Attention.py

This file contains the implementation of the Shared Attention Mechanism.

Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 1, 2024
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
from . import AdaIN
from . import StyleAlignedArgs
from diffusers import StableDiffusionXLPipeline


T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T

class DefaultAttentionProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.processor = attention_processor.AttnProcessor2_0() # from diffusers.models import attention_processor

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)

class SharedAttentionProcessor(DefaultAttentionProcessor):

    def shifted_scaled_dot_product_attention(self, attn: attention_processor.Attention, query: T, key: T, value: T) -> T:
      # don't be scare by Einstein notation, it's a "fancy" way to do matrix multiplication, here the dot product in attn. QK^T.
      # Recall query shape: (batch, heads, tokens, dim) while key shape: (batch, heads, 2*tokens, dim) since it has the ref. also
      logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale # should be equal to "logits = torch.matmul(query, key.transpose(-1, -2)) * attn.scale " and scale should be 1/sqrt(dim)
      # logits shape: (batch, heads, tokens, 2*tokens)
      logits[:, :, :, query.shape[2]:] += self.style_alignment_score_shift # shift by a scalar the attn weights corresponding ONLY to the REFERENCE Keys (see slicing).
      probs = logits.softmax(-1) # softmax along the attention weights of 1 token w.r.t every other token
      return torch.einsum('bhqk,bhkd->bhqd', probs, value) # last dot product with V

    def shared_call(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2) # result shape (batch, #tokens, channel) as in the above picture

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) # linear layer "channels" -> "heads * dim_heads"
        key = attn.to_k(hidden_states) # same as above
        value = attn.to_v(hidden_states) # same as above
        inner_dim = key.shape[-1] # get "heads * dim_heads" value
        head_dim = inner_dim // attn.heads # infer "dim_head" by dividing for the number of heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # shape all back to (batch, heads, tokens, dim_head)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # same as above
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # same as above
        # if self.step >= self.start_inject:
        # Adaptive Normalization of Q and K (and eventually V)
        if self.adain_queries:
            query = AdaIN.adain(query)
        if self.adain_keys:
            key = AdaIN.adain(key)
        if self.adain_values: # usually false
            value = AdaIN.adain(value)
        # shared attention layer
        # Q, V and K shape = (batch, heads, tokens, dim_head)
        if self.share_attention:
            key = AdaIN.concat_first(key, -2, scale=self.style_alignment_score_scale) # create Keys = [K_t, K_r] -> shape: (batch, heads, 2*tokens, dim_head)
            value = AdaIN.concat_first(value, -2) # create Values = [V_t, V_r] -> shape: (batch, heads, 2*tokens, dim_head)
            if self.style_alignment_score_shift != 0:
                hidden_states = self.shifted_scaled_dot_product_attention(attn, query, key, value,)
            else:
                hidden_states = nnf.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                ) # hidden_states output shape -> (batch, heads, tokens, dim_head) since Q not double (.,.,tokens,.) and att = softmax(Q * K'/sqrt(dim_head)) * V
        else:
            hidden_states = nnf.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        # now heads concatenation for later re-projection as in standard Multi Head Attention
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # transpose -> (b, t, h, d_h); reshape -> (b, t, h*d_h)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) # to_out[0] = Linear(in_features = heads * dim_heads, out_features = dim_heads, bias=True)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # hidden_states shape -> (batch, tokens, dim_head) = (batch, pixels, channels)
        if input_ndim == 4:
            # shape it back to a "feature_maps" ready for convolution
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width) # transpose -> (b, c, pixels); reshape to create an "image" -> (b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual # residuals were initial input hidden_states untouched, before attention mechanismf

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):

        hidden_states = self.shared_call(attn, hidden_states, hidden_states, attention_mask, **kwargs)

        return hidden_states

    def __init__(self, style_aligned_args: StyleAlignedArgs):
        super().__init__()
        self.share_attention = style_aligned_args.share_attention
        self.adain_queries = style_aligned_args.adain_queries
        self.adain_keys = style_aligned_args.adain_keys
        self.adain_values = style_aligned_args.adain_values
        self.style_alignment_score_scale = style_aligned_args.style_alignment_score_scale
        self.style_alignment_score_shift = style_aligned_args.style_alignment_score_shift
        

def _get_switch_vec(total_num_layers, level): # level is percentage of layers to keep (just self.attn). Output what layers to switch to shared attn.
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool) # change everything so output a full zero (false) vector
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool) # keep all layers so a full one (true) vector
    to_flip = level > .5 # if level is above half, we have compute as it if was 1 - level and then take the complement (not = ~) of the resulting vector
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers) # number of layers to keep rounded down
    vec = torch.arange(total_num_layers) # creates vector = [0, 1, 2, 3, ..., tot_num_layers]
    vec = vec % (total_num_layers // num_switch) # put a zero on each layer to keep (1 every tot_num//num_switch).
    # E.G every 3 layer keep 1 -> [0, 1, 2, 0, 1, 2, 0, 1, ...].
    vec = vec == 0 # boolean mask
    # The % is not okay when level > 0.5 since it alway gives 1, that's why we make the flipping trick
    if to_flip:
        vec = ~vec
    return vec # return a boolean mask vector indicating which layer to keep (true) and which to switch (false) to custom shared attn.

def init_attention_processors(pipeline: StableDiffusionXLPipeline, style_aligned_args: StyleAlignedArgs | None = None):
    attn_procs = {}
    unet = pipeline.unet # extract the Unet from the StableDiffusionXLPipeline
    number_of_self, number_of_cross = 0, 0 # these two variables are like...useless but ok just in case for counting/debugging
    num_self_layers = len([name for name in unet.attn_processors.keys() if 'attn1' in name]) # count number of self_attn layers (i.e the ones with "attn1" in the name)
    if style_aligned_args is None:
        only_self_vec = _get_switch_vec(num_self_layers, 1) # indicates to switch all self attn. layers to shared attn.
    else:
        only_self_vec = _get_switch_vec(num_self_layers, style_aligned_args.only_self_level) # indicate to switch some self attn. layers to shared attn dependeing on level arg
    # Iterate through all layers: if cross.attn = default attn; if self.attn = switch to shared depending on switch vec computed with "level" arg
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attention = 'attn1' in name
        if is_self_attention:
            number_of_self += 1
            # recall only_self_vec is long "number of self.attn layers" not al layers
            if style_aligned_args is None or only_self_vec[i // 2]: # I SUPPOSE self attn and cross attn layers are alternating, so i//2
                attn_procs[name] = DefaultAttentionProcessor()
            else:
                attn_procs[name] = SharedAttentionProcessor(style_aligned_args)
        else:
            number_of_cross += 1
            attn_procs[name] = DefaultAttentionProcessor()
    # Library call -> set all assigned attention processors classes (see markdown to know how to implement those). No weights changed.
    unet.set_attn_processor(attn_procs)


def register_shared_norm(pipeline: StableDiffusionXLPipeline,
                         share_group_norm: bool = True,
                         share_layer_norm: bool = True, ):
    def register_norm_forward(norm_layer: nn.GroupNorm | nn.LayerNorm) -> nn.GroupNorm | nn.LayerNorm:
        # register the original forward method as an attribute (creates one if not present)
        if not hasattr(norm_layer, 'orig_forward'):
            setattr(norm_layer, 'orig_forward', norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T: # perform Group/Layer Normalization with reference img.
            n = hidden_states.shape[-2] # hs shape: (batch, head, #tokens, channels). Channels and dim_head are synonym in this notebook.
            hidden_states = AdaIN.concat_first(hidden_states, dim=-2) # concat with reference img hidden_states shape (b, h, 2*t, c)
            hidden_states = orig_forward(hidden_states) # original normalization this time with also reference hidden states
            return hidden_states[..., :n, :] # retain just original hidden states ("tokens") i.e. the first half of them

        norm_layer.forward = forward_ #set the new forward that will first concat. the ref. features and then call the original forward
        return norm_layer

    # get the normalization layers from the pipeline iterating it recursively and save them in a dictionary
    def get_norm_layers(pipeline_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]):
        if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
            norm_layers_['layer'].append(pipeline_)
        if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
            norm_layers_['group'].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {'group': [], 'layer': []}
    get_norm_layers(pipeline.unet, norm_layers)
    # modify the norm layers making them shared
    return [register_norm_forward(layer) for layer in norm_layers['group']] + [register_norm_forward(layer) for layer in
                                                                               norm_layers['layer']]
