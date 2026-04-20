from pathlib import Path
import shutil
import tarfile
import zipfile
import time

import requests
from tqdm import tqdm
from torchvision.datasets import VOCDetection


CONSTRUCTION_PPE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
)


def download_file(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    max_retries = 8
    chunk_size = 1024 * 1024
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {}
        mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print(f"[*] Resuming download from byte {existing_size}: {url}")
        else:
            print(f"[*] Downloading: {url}")

        try:
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            if response.status_code == 416:
                print(f"[+] Download already complete: {dest_path}")
                return dest_path
            response.raise_for_status()

            if existing_size > 0 and response.status_code == 200:
                existing_size = 0
                mode = "wb"

            content_length = int(response.headers.get("content-length", 0))
            total_size = existing_size + content_length if content_length else None

            with tqdm(
                total=total_size,
                initial=existing_size,
                unit="iB",
                unit_scale=True,
                desc=dest_path.name,
            ) as pbar:
                with dest_path.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"[+] Download complete: {dest_path}")
            return dest_path
        except requests.RequestException as exc:
            print(f"[!] Download interrupted ({attempt}/{max_retries}): {exc}")
            if attempt >= max_retries:
                raise
            time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"Failed to download after {max_retries} attempts: {url}")
    return dest_path


def extract_archive(archive_path, extract_dir):
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting: {archive_path} -> {extract_dir}")

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif archive_path.suffix in {".tar", ".gz"} or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    print(f"[+] Extraction complete: {archive_path}")


def setup_voc(root_dir="./datasets"):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("Preparing PASCAL VOC 2007")
    print("=" * 60)
    print("[*] Downloading VOC trainval...")
    VOCDetection(root=str(root_dir), year="2007", image_set="trainval", download=True)
    print("[*] Downloading VOC test...")
    VOCDetection(root=str(root_dir), year="2007", image_set="test", download=True)
    print("[+] VOC 2007 is ready.")


def setup_coco(root_dir="./datasets/coco"):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("Preparing MS COCO 2017")
    print("Note: full train2017 is large and needs a lot of disk space.")
    print("=" * 60)

    urls = {
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    }

    for file_name in ("annotations_trainval2017.zip", "val2017.zip"):
        archive_path = download_file(urls[file_name], root_dir / file_name)
        extract_archive(archive_path, root_dir)

    choice = input("\n[?] Download full COCO train2017 as well? (~18GB) (y/n): ").strip().lower()
    if choice == "y":
        archive_path = download_file(urls["train2017.zip"], root_dir / "train2017.zip")
        extract_archive(archive_path, root_dir)
        print("[+] COCO 2017 train/val is ready.")
    else:
        print("[!] Skipped train2017. val2017 + annotations are ready.")


def find_yolo_dataset_root(extract_root):
    extract_root = Path(extract_root)
    candidates = [extract_root]
    candidates.extend([path for path in extract_root.iterdir() if path.is_dir()])

    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find YOLO dataset structure in {extract_root}. "
        f"Expected images/ and labels/ folders."
    )


def setup_construction_ppe(root_dir="./datasets/construction_ppe"):
    root_dir = Path(root_dir)
    temp_dir = root_dir.parent / "_construction_ppe_tmp"

    print("\n" + "=" * 60)
    print("Preparing Construction-PPE")
    print("=" * 60)
    print(f"[*] Source: {CONSTRUCTION_PPE_URL}")

    temp_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)

    archive_path = download_file(CONSTRUCTION_PPE_URL, temp_dir / "construction-ppe.zip")
    extract_root = temp_dir / "extracted"
    extract_archive(archive_path, extract_root)

    dataset_root = find_yolo_dataset_root(extract_root)
    for item_name in ["images", "labels", "data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        src = dataset_root / item_name
        if not src.exists():
            continue
        dst = root_dir / item_name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    shutil.rmtree(temp_dir)
    print(f"[+] Construction-PPE is ready at: {root_dir}")
    print("[+] You can now train with: python train.py --dataset CONSTRUCTION_PPE --model faster_rcnn --amp")


def main():
    print("Dataset download helper")
    print("1. PASCAL VOC 2007")
    print("2. MS COCO 2017")
    print("3. Construction-PPE")
    print("4. VOC + COCO + Construction-PPE")

    choice = input("Choose an option (1/2/3/4): ").strip()

    if choice == "1":
        setup_voc()
    elif choice == "2":
        setup_coco()
    elif choice == "3":
        setup_construction_ppe()
    elif choice == "4":
        setup_voc()
        setup_coco()
        setup_construction_ppe()
    else:
        print("Invalid input. Exit.")


if __name__ == "__main__":
    main()
