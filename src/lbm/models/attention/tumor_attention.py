import copy
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention


class TumorBiasedAttention(Attention):
    """Tumor-Biased Attention Mechanism (TuBAM).

    Architecture- and weight-identical to the wrapped diffusers ``Attention``. During
    training, the logits of self-attention receive the additive bias
    ``bias_weight * m m^T``, where ``m`` is the tumor mask flattened at the token
    resolution. In eval mode, or without a mask, it is standard attention.
    """

    @classmethod
    def from_attention(cls, attn: Attention) -> "TumorBiasedAttention":
        """Convert an existing ``Attention`` module without rebuilding it."""
        new_attn = copy.deepcopy(attn)
        new_attn.__class__ = cls
        new_attn.tumor_mask = None
        new_attn.bias_weight = 0.0
        return new_attn

    def set_tumor_bias(self, tumor_mask: Optional[torch.Tensor], bias_weight: float):
        """Store the tumor mask (B, 1, H, W) and the bias strength for the next forward."""
        self.tumor_mask = tumor_mask
        self.bias_weight = float(bias_weight)

    def _token_mask(self, n_tokens: int, batch_heads: int) -> Optional[torch.Tensor]:
        """Return the flattened tumor mask (B * heads, N) at the token grid, or None."""
        mask = self.tumor_mask
        if mask is None:
            return None
        h, w = mask.shape[-2:]
        scale = (h * w / n_tokens) ** 0.5
        size = (round(h / scale), round(w / scale))
        if size[0] * size[1] != n_tokens:
            return None
        mask = F.interpolate(mask.float(), size=size, mode="nearest").flatten(1)
        return mask.repeat_interleave(batch_heads // mask.shape[0], dim=0)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        """Scaled dot-product attention with the training-only tumor bias of Eq. (5)."""
        is_self_attention = encoder_hidden_states is None
        if is_self_attention:
            encoder_hidden_states = hidden_states

        query = self.head_to_batch_dim(self.to_q(hidden_states))
        key = self.head_to_batch_dim(self.to_k(encoder_hidden_states))
        value = self.head_to_batch_dim(self.to_v(encoder_hidden_states))

        scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask

        if self.training and is_self_attention and self.bias_weight > 0:
            m = self._token_mask(scores.shape[-1], scores.shape[0])
            if m is not None:
                bias = m.unsqueeze(-1) * m.unsqueeze(-2)
                scores = scores + self.bias_weight * bias.to(scores.dtype)

        probs = torch.softmax(scores, dim=-1)
        hidden_states = self.batch_to_head_dim(torch.bmm(probs, value))
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)
