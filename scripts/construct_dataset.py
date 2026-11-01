import csv
import math
import os
import random
import shutil
import tarfile
from collections import defaultdict

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from tqdm import tqdm

SPLIT_COUNTS = defaultdict(int)
VOLUME_COUNTS = defaultdict(int)


TRAIN_BASE_DIR = "data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
TEST_BASE_DIR = "data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"

FINAL_OUT = "data/BraTS2023_GLI_imgs"
TMP_PNG_ROOT = "data/tmp_pngs"


TRAIN_RATIO = 0.8
FILES_PER_TAR = 2000

MIN_FOREGROUND_RATIO = {"train": 0.03, "validation": 0.03, "test": None}

SEED = 42
random.seed(SEED)


def volume_percentile_norm(vol, p_low=0.1, p_high=99.9):
    v_min, v_max = np.percentile(vol, (p_low, p_high))
    vol = np.clip(vol, v_min, v_max)

    return ((vol - v_min) / (v_max - v_min + 1e-8)).astype(np.float32)


def center_pad_2d(x, target_size=256):
    h, w = x.shape
    pad_h = target_size - h
    pad_w = target_size - w

    return np.pad(
        x,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
        mode="constant",
    )


def compute_distance_to_boundary(mask: np.ndarray, d_max=30.0):
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    boundary = find_boundaries(mask, mode="inner")
    dist = distance_transform_edt(~boundary)
    dist = dist * mask
    dist = np.clip(dist, 0, d_max) / (d_max + 1e-8)

    return dist.astype(np.float32)


def is_valid_slice(slice_2d, split):
    if MIN_FOREGROUND_RATIO[split] is None:
        return True

    thresh = np.percentile(slice_2d, 10)
    mask = slice_2d > thresh

    return (mask.sum() / mask.size) > MIN_FOREGROUND_RATIO[split]


def get_subjects(base_dir):
    return sorted(d for d in os.listdir(base_dir) if d.startswith("BraTS-"))


def split_train_val(subjects):
    random.shuffle(subjects)
    n = len(subjects)

    n_train = math.ceil(TRAIN_RATIO * n)

    return {"train": subjects[:n_train], "validation": subjects[n_train:]}


def save_split_csv(splits, test_subjects, out_path="data/BraTS2023_GLI_splits.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "split"])
        for split, subs in splits.items():
            for sid in subs:
                writer.writerow([sid, split])
        for sid in test_subjects:
            writer.writerow([sid, "test"])


def save_preprocessed_volume(vol: np.ndarray, ref_nii: nib.Nifti1Image, out_path: str):
    nii = nib.Nifti1Image(
        vol.astype(np.float32), affine=ref_nii.affine, header=ref_nii.header
    )
    nib.save(nii, out_path)


def to_uint8(x):
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def process_subject(subject_id, base_dir, out_dir, split):

    subj_path = os.path.join(base_dir, subject_id)

    t1n_nii = nib.load(
        os.path.join(subj_path, [f for f in os.listdir(subj_path) if "t1n." in f][0])
    )
    t1c_nii = nib.load(
        os.path.join(subj_path, [f for f in os.listdir(subj_path) if "t1c." in f][0])
    )

    t1n = volume_percentile_norm(t1n_nii.get_fdata())
    t1c = volume_percentile_norm(t1c_nii.get_fdata())

    save_preprocessed_volume(
        t1n,
        t1n_nii,
        os.path.join(
            subj_path,
            os.path.basename(t1n_nii.get_filename()).replace(
                ".nii.gz", "_preprocessed.nii.gz"
            ),
        ),
    )

    save_preprocessed_volume(
        t1c,
        t1c_nii,
        os.path.join(
            subj_path,
            os.path.basename(t1c_nii.get_filename()).replace(
                ".nii.gz", "_preprocessed.nii.gz"
            ),
        ),
    )

    seg = None
    seg_path = os.path.join(subj_path, f"{subject_id}-seg.nii.gz")
    if os.path.exists(seg_path):
        seg = nib.load(seg_path).get_fdata()
    elif split != "test":
        raise FileNotFoundError(
            f"Missing segmentation for training subject: {seg_path}"
        )

    for z in range(t1n.shape[2]):
        inp = t1n[:, :, z]
        tgt = t1c[:, :, z]

        if split != "test" and not is_valid_slice(inp, split):
            continue

        inp = center_pad_2d(inp)
        tgt = center_pad_2d(tgt)

        base = f"{subject_id}_slice_{z:03d}"

        Image.fromarray(to_uint8(inp)).save(os.path.join(out_dir, f"{base}.source.png"))
        Image.fromarray(to_uint8(tgt)).save(os.path.join(out_dir, f"{base}.target.png"))

        if seg is not None:
            mask = center_pad_2d(seg[:, :, z])
            mask = (mask > 0).astype(np.uint8)

            dist_map = compute_distance_to_boundary(mask)
            dist_u8 = (np.clip(dist_map, 0, 1) * 255).astype(np.uint8)

            Image.fromarray(mask * 255).save(os.path.join(out_dir, f"{base}.mask.png"))
            Image.fromarray(dist_u8).save(os.path.join(out_dir, f"{base}.dist.png"))

        SPLIT_COUNTS[split] += 1


def make_tar_shards(png_dir, out_dir, split):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(png_dir))

    for i in range(0, len(files), FILES_PER_TAR):
        with tarfile.open(
            os.path.join(out_dir, f"{split}_shard_{i//FILES_PER_TAR:05d}.tar"), "w"
        ) as tar:
            for f in files[i : i + FILES_PER_TAR]:
                tar.add(os.path.join(png_dir, f), arcname=f)


def main():
    train_subjects = get_subjects(TRAIN_BASE_DIR)
    test_subjects = get_subjects(TEST_BASE_DIR)

    splits = split_train_val(train_subjects)
    save_split_csv(splits, test_subjects)

    for split, subs in splits.items():
        print(f"\nProcessing {split} set...")
        split_dir = os.path.join(TMP_PNG_ROOT, split)
        os.makedirs(split_dir, exist_ok=True)

        for s in tqdm(subs):
            VOLUME_COUNTS[split] += 1
            process_subject(s, TRAIN_BASE_DIR, split_dir, split)

    print("\nProcessing test set...")
    test_dir = os.path.join(TMP_PNG_ROOT, "test")
    os.makedirs(test_dir, exist_ok=True)

    for s in tqdm(test_subjects):
        VOLUME_COUNTS["test"] += 1
        process_subject(s, TEST_BASE_DIR, test_dir, "test")

    for split in ["train", "validation", "test"]:
        make_tar_shards(
            os.path.join(TMP_PNG_ROOT, split), os.path.join(FINAL_OUT, split), split
        )

    print("\n========== DATASET STATISTICS ==========")
    for split in ["train", "validation", "test"]:
        print(
            f"{split.capitalize():>10}: "
            f"{VOLUME_COUNTS[split]} volumes | "
            f"{SPLIT_COUNTS[split]} slice pairs"
        )
    print("=======================================")

    shutil.rmtree(TMP_PNG_ROOT)
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
