import torch
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from torch import nn

from .linear import NunchakuLinear


class NunchakuBaseAttention(nn.Module):
    def __init__(self, processor: str = "flashattn2", *args, **kwargs):
        super(NunchakuBaseAttention, self).__init__()
        self.processor = None
        self.set_processor(processor)

    def set_processor(self, processor: str):
        raise NotImplementedError("Subclass must implement this method")


def _patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            _patch_linear(child, linear_cls, **kwargs)
    return module


class NunchakuFeedForward(FeedForward):
    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        self.net = _patch_linear(ff.net, NunchakuLinear, **kwargs)
        # for int4, we shift the activation of mlp_fc2 to make it unsigned
        self.net[2].act_unsigned = self.net[2].precision != "nvfp4"
        self.act_mlp = nn.GELU()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0].proj(hidden_states)
        
        hidden_states = self.act_mlp(hidden_states)
        main_input = hidden_states + 0.171875
        hidden_states = self.net[2].forward_split(main_input, hidden_states)
        # main_output, lora_output = self.net[0].proj(hidden_states, split=True)
        
        # main_output = self.act_mlp(main_output)
        # lora_output = self.act_mlp(lora_output)
        # main_output += 0.171875
        # hidden_states = self.net[2].forward_split(main_output, lora_output)
        return hidden_states
