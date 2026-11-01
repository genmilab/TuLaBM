import torch
from diffusers.models.attention import BasicTransformerBlock

from lbm.models.attention import TumorBiasedAttention
from lbm.models.unets import DiffusersUNet2DCondWrapper


def make_unet():
    """Build a small conditional U-Net whose attention layers are replaced by TuBAM."""
    torch.manual_seed(0)
    unet = DiffusersUNet2DCondWrapper(
        in_channels=4,
        out_channels=4,
        up_block_types=["CrossAttnUpBlock2D"],
        down_block_types=["CrossAttnDownBlock2D"],
        cross_attention_dim=[32],
        block_out_channels=[32],
        transformer_layers_per_block=[1],
        attention_head_dim=[4],
        norm_num_groups=8,
        layers_per_block=1,
    )
    for module in unet.modules():
        if isinstance(module, BasicTransformerBlock):
            module.attn1 = TumorBiasedAttention.from_attention(module.attn1)
            module.attn2 = TumorBiasedAttention.from_attention(module.attn2)
    return unet


def run(unet, mask, weight):
    """Run one forward pass with an empty conditioning dictionary."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 16)
    t = torch.tensor([500, 500])
    return unet(x, t, conditioning={"cond": {}}, tumor_mask=mask, bias_weight=weight)


def test_bias_reaches_attention_layers():
    unet = make_unet().train()
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    run(unet, mask, 0.5)
    layers = [m for m in unet.modules() if isinstance(m, TumorBiasedAttention)]
    assert layers and all(m.tumor_mask is mask and m.bias_weight == 0.5 for m in layers)


def test_bias_changes_training_output_only():
    unet = make_unet()
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    unet.train()
    with torch.no_grad():
        plain = run(unet, None, 0.0)
        biased = run(unet, mask, 2.0)
    assert not torch.allclose(plain, biased)
    unet.eval()
    with torch.no_grad():
        plain = run(unet, None, 0.0)
        biased = run(unet, mask, 2.0)
    assert torch.allclose(plain, biased)


def test_matches_standard_attention_without_mask():
    unet = make_unet().eval()
    reference = make_unet().eval()
    for module in reference.modules():
        if isinstance(module, BasicTransformerBlock):
            for name in ("attn1", "attn2"):
                attn = getattr(module, name)
                attn.__class__ = TumorBiasedAttention.__mro__[1]
    with torch.no_grad():
        assert torch.allclose(
            run(unet, None, 0.0), run(reference, None, 0.0), atol=1e-5
        )
