import os
import tarfile
import shutil


INPUT_TAR_DIR = "data/BraTS2023_GLI_imgs/test"
OUTPUT_ROOT = "data/BraTS2023_GLI_test"

SOURCE_DIR = os.path.join(OUTPUT_ROOT, "source")
TARGET_DIR = os.path.join(OUTPUT_ROOT, "target")
MASK_DIR = os.path.join(OUTPUT_ROOT, "mask")

VALID_EXTS = (".png", ".jpg", ".jpeg")


def setup_dirs():
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)


def process_tar(tar_path):
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            name = os.path.basename(member.name)

            if not name.lower().endswith(VALID_EXTS):
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            if ".source." in name:
                out_path = os.path.join(SOURCE_DIR, name)
            elif ".target." in name:
                out_path = os.path.join(TARGET_DIR, name)
            elif ".mask." in name:
                out_path = os.path.join(MASK_DIR, name)
            else:
                continue

            with open(out_path, "wb") as f:
                shutil.copyfileobj(extracted, f)


def main():
    setup_dirs()

    tar_files = sorted([f for f in os.listdir(INPUT_TAR_DIR) if f.endswith(".tar")])

    print(f"Found {len(tar_files)} tar files")

    for tar_name in tar_files:
        tar_path = os.path.join(INPUT_TAR_DIR, tar_name)
        print(f"Processing {tar_name}")
        process_tar(tar_path)

    print("Done. Data organized successfully.")


if __name__ == "__main__":
    main()
