import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from lbm.inference import evaluate, get_model_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TuLaBM inference on 2D NC-MRI slices."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--final_dir", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(ckpt_path: str, device: str):
    """Load a trained TuLaBM checkpoint (.ckpt or .safetensors) in bfloat16."""
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = get_model_from_checkpoint(
        ckpt_path, torch_dtype=torch.bfloat16, device=device
    )
    model.eval()
    return model


def copy_png(src: Path, dst: Path):
    """Copy a reference PNG (target or mask) next to the prediction, unchanged."""
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).save(dst)


@torch.no_grad()
def run_single_slice(model, img_path: Path, args, out_root: Path) -> float:
    """Synthesize one CE slice from an NC slice and return the model time in seconds."""
    stem = img_path.name.replace(".source.png", "")
    src = Image.open(img_path).convert("RGB")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = evaluate(model, src, args.num_inference_steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    pred = np.asarray(out.convert("L"), dtype=np.uint8)
    (out_root / "output").mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(out_root / "output" / f"{stem}.png")

    if args.target_dir:
        copy_png(
            Path(args.target_dir) / f"{stem}.target.png",
            out_root / "target" / f"{stem}.png",
        )
    if args.mask_dir:
        copy_png(
            Path(args.mask_dir) / f"{stem}.mask.png", out_root / "mask" / f"{stem}.png"
        )
    return elapsed


def main():
    """Run TuLaBM on every ``*.source.png`` slice of a folder."""
    args = parse_args()
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

    model = load_model(args.ckpt_path, args.device)
    out_root = Path(args.final_dir)
    images = sorted(Path(args.source_dir).glob("*.source.png"))
    logger.info(f"Number of images: {len(images)}")

    times = [run_single_slice(model, p, args, out_root) for p in images]
    if times:
        warm = times[1:] if len(times) > 1 else times
        logger.info(
            f"Synthesized {len(times)} slices with {args.num_inference_steps} steps; "
            f"mean time {np.mean(warm):.4f} s/slice (first slice excluded as warm-up)."
        )


if __name__ == "__main__":
    main()
