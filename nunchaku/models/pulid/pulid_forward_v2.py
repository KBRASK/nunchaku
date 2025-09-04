"""
This module implements the PuLID forward function for the :class:`nunchaku.models.transformers.NunchakuFluxTransformer2DModel`,

.. note::
    This module is adapted from the original PuLID repository:
    https://github.com/ToTheBeginning/PuLID
"""

from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from ...ops.fused import fused_gelu_mlp
from ...utils import get_precision
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.flux import NunchakuFluxFA2Processor, NunchakuFluxFP16AttnProcessor
from ..embeddings import NunchakuFluxPosEmbed, pack_rotemb
from ..linear import SVDQW4A4Linear
from ..normalization import NunchakuAdaLayerNormZero, NunchakuAdaLayerNormZeroSingle
from ..utils import fuse_linears
from ..transformers.utils import NunchakuModelLoaderMixin, pad_tensor


def pulid_forward_v2(
    self,
    hidden_states: torch.Tensor,
    id_embeddings=None,
    id_weight=None,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    start_timestep: float | None = None,
    end_timestep: float | None = None,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    pulid_ca_idx = 0
    hidden_states = self.x_embedder(hidden_states)
    if timestep.numel() > 1:
        timestep_float = timestep.flatten()[0].item()
    else:
        timestep_float = timestep.item()

    if start_timestep is not None and start_timestep > timestep_float:
        id_embeddings = None
    if end_timestep is not None and end_timestep < timestep_float:
        id_embeddings = None

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    txt_tokens = encoder_hidden_states.shape[1]
    img_tokens = hidden_states.shape[1]

    assert image_rotary_emb.ndim == 6
    assert image_rotary_emb.shape[0] == 1
    assert image_rotary_emb.shape[1] == 1
    assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
    # [1, tokens, head_dim / 2, 1, 2] (sincos)
    image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
    rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]  # .to(self.dtype)
    rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]  # .to(self.dtype)
    rotary_emb_single = image_rotary_emb

    rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
    rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
    rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
            joint_attention_kwargs=joint_attention_kwargs,
        )
        # controlnet residual
        if controlnet_block_samples is not None:
            raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")
        if id_embeddings is not None and index_block % 2 == 0:
            ip = id_weight * self.pulid_ca[pulid_ca_idx](id_embeddings, hidden_states)
            hidden_states = hidden_states + ip
            pulid_ca_idx += 1

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    for index_block, block in enumerate(self.single_transformer_blocks):
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=rotary_emb_single,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            raise NotImplementedError("Controlnet is not supported for FluxTransformer2DModelV2 for now")
        if id_embeddings is not None and index_block % 4 == 0:
            ip = id_weight * self.pulid_ca[pulid_ca_idx](id_embeddings, hidden_states[:, txt_tokens:])
            hidden_states[:, txt_tokens:] = hidden_states[:, txt_tokens:] + ip
            pulid_ca_idx += 1
    hidden_states = hidden_states[:, txt_tokens:]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
