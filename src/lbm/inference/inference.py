import logging

import PIL
from PIL import Image
import torch
from torchvision.transforms import ToPILImage, ToTensor

from lbm.models.lbm import LBMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: LBMModel,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):
    """
    Evaluate the model on an image coming from the source distribution and generate a new image from the target distribution.

    Args:
        model (LBMModel): The model to evaluate.
        source_image (PIL.Image.Image): The source image to evaluate the model on.
        num_sampling_steps (int): The number of sampling steps to use for the model.

    Returns:
        PIL.Image.Image: The generated image.
    """

    ori_w, ori_h = source_image.size
    if source_image.size != (256, 256):
        source_image = source_image.resize((256, 256), Image.BILINEAR)

    device = next(model.parameters()).device
    img_tensor = ToTensor()(source_image).unsqueeze(0) * 2 - 1
    batch = {"source_image": img_tensor.to(device=device, dtype=torch.bfloat16)}

    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)
    if output_image.size != (ori_w, ori_h):
        output_image = output_image.resize((ori_w, ori_h), Image.BILINEAR)

    return output_image
