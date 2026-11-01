import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def load(path: Path) -> np.ndarray:
    """Load an 8-bit grayscale PNG as float32 in [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, target: np.ndarray, region: np.ndarray = None) -> float:
    """PSNR in dB for data range 1, over the whole image or over a boolean region."""
    diff = (pred - target) ** 2
    mse = diff[region].mean() if region is not None else diff.mean()
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


def ssim(pred: np.ndarray, target: np.ndarray, region: np.ndarray = None) -> float:
    """SSIM in % (Gaussian window, sigma 1.5), averaged over the image or over a region."""
    value, ssim_map = structural_similarity(
        target,
        pred,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
    )
    return float(100.0 * (ssim_map[region].mean() if region is not None else value))


def summarize(values):
    """Return 'mean ± std' of a list, or 'n/a' when empty."""
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}" if values else "n/a"


def main():
    """Compute whole-image and tumor-region PSNR/SSIM for an inference output folder."""
    parser = argparse.ArgumentParser(description="Evaluate TuLaBM predictions.")
    parser.add_argument("--pred_dir", required=True, help="folder of predicted PNGs")
    parser.add_argument(
        "--target_dir", required=True, help="folder of reference CE PNGs"
    )
    parser.add_argument("--mask_dir", default=None, help="folder of tumor mask PNGs")
    parser.add_argument("--out_csv", default=None, help="optional per-slice CSV")
    args = parser.parse_args()

    rows = []
    for pred_path in sorted(Path(args.pred_dir).glob("*.png")):
        target_path = Path(args.target_dir) / pred_path.name
        if not target_path.exists():
            continue
        pred, target = load(pred_path), load(target_path)
        row = {
            "slice": pred_path.stem,
            "psnr": psnr(pred, target),
            "ssim": ssim(pred, target),
        }
        if args.mask_dir is not None:
            mask_path = Path(args.mask_dir) / pred_path.name
            if mask_path.exists():
                region = load(mask_path) > 0.5
                if region.any():
                    row["tumor_psnr"] = psnr(pred, target, region)
                    row["tumor_ssim"] = ssim(pred, target, region)
        rows.append(row)

    if not rows:
        raise SystemExit("No prediction/target pairs found.")

    for key, name in [
        ("psnr", "PSNR (dB)"),
        ("ssim", "SSIM (%)"),
        ("tumor_psnr", "Tumor PSNR (dB)"),
        ("tumor_ssim", "Tumor SSIM (%)"),
    ]:
        values = [r[key] for r in rows if key in r]
        print(f"{name:>16}: {summarize(values)}  (n = {len(values)})")

    if args.out_csv:
        keys = ["slice", "psnr", "ssim", "tumor_psnr", "tumor_ssim"]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
