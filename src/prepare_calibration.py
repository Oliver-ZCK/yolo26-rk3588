"""准备 INT8 量化标定集：COCO val2017 的固定随机子集。

标定只需要有代表性的图片，不需要标注；256 张足够 rknn-toolkit2 统计
各层激活的分布范围。整包 val2017 下载一次，后续 mAP 评估用同一份；
标定抽样用固定种子，评估时排除这 256 张即可保证两者不重叠。

数据落在仓库外 ~/datasets/coco，产物是 dataset.txt（rknn build 的输入）。
官方源国内直连只有 ~55 KB/s，实测 hf-mirror 的 LibreYOLO/coco-val2017
更快；--zip 可直接指定已下载好的包跳过下载。

用法:
    python3 src/prepare_calibration.py                # 缺 zip 时自动下载
    python3 src/prepare_calibration.py --zip PATH    # 用已下载好的 zip
"""

import argparse
import random
import urllib.request
import zipfile
from pathlib import Path

import cv2

from stages import PROJECT_ROOT

DATA_ROOT = Path.home() / "datasets" / "coco"
IMAGES_DIR = DATA_ROOT / "images" / "val2017"
DEFAULT_ZIP = DATA_ROOT / "downloads" / "coco-val2017.zip"
DATASET_TXT = PROJECT_ROOT / "outputs" / "rknn_int8_build" / "dataset.txt"

# 镜像在前（hf-mirror 单连接也限速，大文件建议外部并发下载后 --zip 指入）
SOURCES = [
    "https://hf-mirror.com/datasets/LibreYOLO/coco-val2017/resolve/main/coco-val2017.zip",
    "https://images.cocodataset.org/zips/val2017.zip",
]
N_CALIB = 256
SEED = 0


def download(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for url in SOURCES:
        print(f"下载 {url}")
        try:
            def hook(n: int, bs: int, total: int) -> None:
                done = n * bs
                if done % (50 * 1024 * 1024) < bs:
                    print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB")
            urllib.request.urlretrieve(url, dst, reporthook=hook)
            return
        except Exception as e:
            print(f"  失败: {e}，换下一个源")
    raise SystemExit("所有源都下载失败")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="已下载的 val2017.zip")
    ap.add_argument("--num", type=int, default=N_CALIB, help="标定图数量")
    args = ap.parse_args()

    if not IMAGES_DIR.is_dir() or len(list(IMAGES_DIR.glob("*.jpg"))) < 5000:
        zip_path = args.zip
        if not zip_path.exists():
            download(zip_path)
        print(f"解压 {zip_path}")
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_ROOT)
        # zip 内部结构不统一（官方包 val2017/ 前缀，镜像包 coco-val2017/images/val2017/），
        # 解压后按"装着 5000 张图的目录"定位，归位到统一的 images/val2017
        if len(list(IMAGES_DIR.glob("*.jpg"))) < 5000:
            for d in DATA_ROOT.rglob("val2017"):
                if d.is_dir() and len(list(d.glob("*.jpg"))) >= 5000:
                    IMAGES_DIR.parent.mkdir(parents=True, exist_ok=True)
                    d.rename(IMAGES_DIR)
                    break

    images = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"val2017 共 {len(images)} 张")
    if len(images) < 5000:
        print("图片数不足 5000，解压可能不完整")
        return 1

    # 固定种子：任何人重跑都得到同一份标定集，评估时可精确排除
    picked = random.Random(SEED).sample(images, args.num)

    DATASET_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in picked:
        img = cv2.imread(str(p))  # 顺带校验文件完好，坏图换下一张
        if img is None:
            print(f"  跳过坏图 {p.name}")
            continue
        lines.append(str(p))
    DATASET_TXT.write_text("\n".join(lines) + "\n")
    print(f"标定集 {len(lines)} 张 -> {DATASET_TXT.relative_to(PROJECT_ROOT)}")
    print(f"  示例: {Path(lines[0]).name} ... {Path(lines[-1]).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
