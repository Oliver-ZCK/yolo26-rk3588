"""真机 vs 模拟器的 INT8 输出张量逐通道对比——量化 7 pt 乐观偏差的归因。

板端 main.cc --dump-raw 存模型原生 int8 + sidecar 属性；本脚本对同一张图按
部署参数在模拟器重建推理（eval_map.py 同源路径），两侧统一到 float 后对比：
逐通道 diff 分布、框/分数分组、皮尔逊相关——判断 NPU 执行噪声是全网均匀
还是从某类算子开始发散。

用法:
    python3 src/diff_raw.py <image.jpg> <board_raw.int8>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eval_map import CALIB_TXT, IMGSZ, ONNX_PATH, PROJECT_ROOT
from rknn.api import RKNN
from stages import IOU


def sim_single(image: Path) -> np.ndarray:
    """按部署参数在模拟器重建 INT8，返回 float 输出 [84, 8400]（坐标已乘回 640）。"""
    import contextlib

    from ultralytics import YOLO

    build_onnx = PROJECT_ROOT / "outputs" / "rknn_int8_build" / "yolo26n-int8norm.onnx"
    build_dir = PROJECT_ROOT / "outputs" / "map_build_int8_diff"

    yolo = YOLO(str(PROJECT_ROOT / "models" / "yolo26n.pt"))
    yolo.predict(source=str(image), imgsz=IMGSZ, rect=False, verbose=False)
    im0 = cv2.imread(str(image))
    x = yolo.predictor.preprocess([im0])
    u8 = (x[0].permute(1, 2, 0).numpy() * 255).astype("uint8")[None]

    rknn = RKNN(verbose=False)
    build_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(build_dir):
        if rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                       target_platform="rk3588", quantized_dtype="w8a8",
                       quantized_algorithm="normal",
                       auto_hybrid_cos_thresh=0.999) != 0:
            raise SystemExit("config 失败")
        if rknn.load_onnx(model=str(build_onnx)) != 0 or \
           rknn.build(do_quantization=True, dataset=str(CALIB_TXT), auto_hybrid=True) != 0:
            raise SystemExit("build 失败")
        if rknn.init_runtime() != 0:
            raise SystemExit("init_runtime 失败")
        sim = rknn.inference(inputs=[u8], data_format="nhwc")[0].reshape(84, -1).copy()
        rknn.release()
    sim[:4] *= IMGSZ  # int8norm 语义：前 4 通道是 ÷640 的归一化坐标
    return sim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("board_raw", type=Path)
    args = ap.parse_args()

    meta = json.loads(Path(str(args.board_raw) + ".json").read_text())
    raw = np.fromfile(args.board_raw, dtype=np.int8 if meta["type"] == "INT8" else np.uint16)
    board = (raw.astype(np.float32) - meta["zp"]) * meta["scale"]
    board = board.reshape(meta["channels"], meta["anchors"])
    if board[:4].max() < 10.0:  # 部署模型是 int8norm：坐标通道量级 ~1
        board[:4] *= IMGSZ

    sim = sim_single(args.image)

    diff = np.abs(board - sim)
    corr_all = np.corrcoef(board.ravel(), sim.ravel())[0, 1]
    print(f"图片 {args.image.name}  输出 [84,{meta['anchors']}]  全局皮尔逊相关 {corr_all:.6f}")
    print(f"{'组':<14}{'mean|diff|':>12}{'p95|diff|':>12}{'max|diff|':>12}{'相关':>10}")
    for tag, sl in [("框 cx/cy", slice(0, 2)), ("框 w/h", slice(2, 4)),
                    ("分数 前40类", slice(4, 44)), ("分数 后40类", slice(44, 84))]:
        d, b, s = diff[sl], board[sl], sim[sl]
        c = np.corrcoef(b.ravel(), s.ravel())[0, 1]
        print(f"{tag:<14}{d.mean():>12.4f}{np.percentile(d, 95):>12.4f}{d.max():>12.4f}{c:>10.5f}")

    # 量纲换算：int8 步长 = scale；差异按"多少个量化步长"计才好横向比较
    print(f"\nint8 输出步长 scale={meta['scale']:.6f}  "
          f"全局 mean|diff| = {diff.mean() / meta['scale']:.2f} 步长")

    # 逐通道 max diff 的 top 10：看发散集中在框通道还是某些类别通道
    per_ch = diff.max(axis=1)
    order = np.argsort(per_ch)[::-1][:10]
    print("\n逐通道 max|diff| top 10（0-3 框 / 4-83 类别）:")
    print("  " + "  ".join(f"ch{i}:{per_ch[i]:.3f}" for i in order))

    # 空间模式：大差异锚点是否集中（>1 int8 步长的锚点占比）
    hot = diff > meta["scale"]
    print(f"\n>1 步长锚点占比：框通道 {hot[:4].mean():.1%}  分数通道 {hot[4:].mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
