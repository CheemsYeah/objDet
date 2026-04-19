# download_datasets.py
import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
from torchvision.datasets import VOCDetection


def download_file(url, dest_path):
    """带进度条的文件下载函数"""
    if os.path.exists(dest_path):
        print(f"[-] 文件已存在: {dest_path}，跳过下载。")
        return

    print(f"[*] 正在下载: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)) as t:
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
    print(f"[+] 下载完成: {dest_path}")


def extract_file(file_path, extract_dir):
    """解压 zip 或 tar 文件"""
    print(f"[*] 正在解压: {file_path} 到 {extract_dir} ... (这可能需要一些时间)")
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    print(f"[+] 解压完成: {file_path}")


def setup_voc(root_dir='./datasets'):
    """准备 PASCAL VOC 数据集"""
    print("\n" + "=" * 50)
    print("准备下载 PASCAL VOC (2007) 数据集...")
    print("=" * 50)
    # 利用 torchvision 内置的方法自动下载并解压 VOC
    # 这会生成 datasets/VOCdevkit/VOC2007 目录
    os.makedirs(root_dir, exist_ok=True)
    print("[*] 下载 Train/Val 集...")
    VOCDetection(root=root_dir, year='2007', image_set='trainval', download=True)
    print("[*] 下载 Test 集...")
    VOCDetection(root=root_dir, year='2007', image_set='test', download=True)
    print("\n[✔] PASCAL VOC 准备完毕！")


def setup_coco(root_dir='./datasets/coco'):
    """准备 MS COCO 2017 数据集"""
    print("\n" + "=" * 50)
    print("准备下载 MS COCO 2017 数据集...")
    print("⚠️ 注意：COCO 训练集包含 11.8 万张图片，下载与解压需要约 20GB 空间！")
    print("=" * 50)

    os.makedirs(root_dir, exist_ok=True)

    # COCO 官方下载链接
    urls = {
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",  # 验证集图像 (约 1GB)
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        # 标签标注 (约 241MB)
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip"  # 训练集图像 (约 18GB)
    }

    # 1. 先下载并解压标注和验证集 (体积小，用于快速测试代码)
    for file_name in ["annotations_trainval2017.zip", "val2017.zip"]:
        file_path = os.path.join(root_dir, file_name)
        download_file(urls[file_name], file_path)
        # 解压后检查目录是否存在，不存在则解压
        target_folder = file_name.replace('.zip', '').replace('_trainval2017', '')
        if not os.path.exists(os.path.join(root_dir, target_folder)):
            extract_file(file_path, root_dir)

    # 2. 询问是否下载庞大的训练集
    choice = input("\n[?] 标注文件和验证集已就绪。是否继续下载完整的 COCO 训练集 (train2017.zip, 约 18GB)? (y/n): ")
    if choice.lower() == 'y':
        file_path = os.path.join(root_dir, "train2017.zip")
        download_file(urls["train2017.zip"], file_path)
        if not os.path.exists(os.path.join(root_dir, "train2017")):
            extract_file(file_path, root_dir)
        print("\n[✔] MS COCO 2017 完整数据准备完毕！")
    else:
        print("\n[!] 已跳过下载完整的 COCO 训练集。你可以使用 val2017 验证集来跑通代码逻辑。")


if __name__ == "__main__":
    print("欢迎使用目标检测数据集一键下载脚本！")
    print("1. 仅下载 PASCAL VOC (推荐用于快速实验，约 1-2GB)")
    print("2. 仅下载 MS COCO 2017 (工业级数据，体积巨大)")
    print("3. 两者全部下载")

    choice = input("请输入对应的数字 (1/2/3): ")

    if choice == '1':
        setup_voc()
    elif choice == '2':
        setup_coco()
    elif choice == '3':
        setup_voc()
        setup_coco()
    else:
        print("无效输入，脚本退出。")