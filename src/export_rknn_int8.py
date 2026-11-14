"""导出 YOLO26n 的 RKNN 模型（INT8 量化），标定 + 模拟器验证。

与 FP16（export_rknn.py）只差 build 一步：do_quantization=True 按标定集
统计每层激活的分布，权重和激活都压到 int8。板端与验证框架无需改动——
输出类型、zp/scale 反量化、_NormalizeCoords 坐标归一化都按张量属性自动
识别（FP16 阶段已就位）。

量化误差比 FP16 大一个量级，容差不能沿用。下面的阈值是按 w8a8 粒度的
粗估上限占位，首跑后按 bus.jpg 与标定外图片的实测分布收紧——
和 FP16 的 8.6e-3 一样，阈值要有推导依据，不拍脑袋。

关键机制：one2many 输出把框坐标（0~640 px）和类别分数（0~1）拼在同一
张量里，w8a8 对整张量只有一个 scale，步长 651/255≈2.55 会把所有类别
分数打成 0（bus.jpg 实测 0 检出、类别 max diff 恰等于基准最高分 0.903）。
所以导出前把框通道除以输入尺寸归一化到 0~1——复刻 ultralytics RKNN
INT8 导出插入的 _NormalizeCoords，板端 Decode 的 normalized_coords
分支负责乘回。

用法:
    python3 src/prepare_calibration.py    # 先准备 dataset.txt
    python3 src/export_rknn_int8.py [--algorithm normal|mmse|kl_divergence]
"""

import argparse
import contextlib
from pathlib import Path
from importlib.metadata import version

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
import torch
from rknn.api import RKNN
from ultralytics import YOLO
from ultralytics.utils import YAML

from export_rknn import IMAGE, ONNX_PATH, WEIGHTS, detect, draw, raw_tensor, tensor_diff
from stages import (
    CONF,
    IOU,
    IMGSZ,
    PROJECT_ROOT,
    compare,
    fmt,
    load_stage,
    sha256,
    stage_paths,
    write_comparison,
    write_stage,
)

TARGET = "rk3588"
BUILD_DIR = PROJECT_ROOT / "outputs" / "rknn_int8_build"
MODEL_DIR = PROJECT_ROOT / "models" / "yolo26n_int8_rknn_model"
STAGE = "rknn_int8"
REF_STAGE = "pytorch"
SECOND_REF_STAGE = "onnx"
DATASET_TXT = BUILD_DIR / "dataset.txt"

# ---- 粗估上限，首跑实测后收紧（占位依据：w8a8 激活 1/127 量化步长） ----
TOL_BOX_REL = 6e-2      # 框通道归一化误差上限（FP16 为 2e-2）
TOL_CLS = 2e-2          # 类别通道绝对误差上限（FP16 为 2e-3）
TOL_PX = 8.0            # 检测框角点偏差上限（FP16 为 2.0）
TOL_CONF = 5e-2         # 置信度偏差上限（FP16 为 5e-3）


def normalize_box_channels(src: Path, dst: Path, imgsz: int) -> None:
    """输出张量的框通道除以输入尺寸（复刻 ultralytics 的 _NormalizeCoords）。

    机制见模块 docstring。opset 19 的 Slice 以 initializer 传参；
    最终输出改名 output0_norm，板端按索引取输出，不依赖名字。
    """
    model = onnx.load(str(src))
    graph = model.graph
    out = graph.output[0].name
    dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    ch = dims[1]

    names = [f"{out}/{tag}" for tag in ("zero", "four", "ch", "axis", "scale")]
    init = [
        helper.make_tensor(names[0], TensorProto.INT64, [1], [0]),
        helper.make_tensor(names[1], TensorProto.INT64, [1], [4]),
        helper.make_tensor(names[2], TensorProto.INT64, [1], [ch]),
        helper.make_tensor(names[3], TensorProto.INT64, [1], [1]),
        helper.make_tensor(names[4], TensorProto.FLOAT, [1], [float(imgsz)]),
    ]
    graph.initializer.extend(init)

    boxes, boxes_n, cls, merged = (f"{out}/{t}" for t in ("boxes", "boxes_norm", "cls", "norm"))
    graph.node.extend([
        helper.make_node("Slice", [out, names[0], names[1], names[3]], [boxes]),
        helper.make_node("Div", [boxes, names[4]], [boxes_n]),
        helper.make_node("Slice", [out, names[1], names[2], names[3]], [cls]),
        helper.make_node("Concat", [boxes_n, cls], [merged], axis=1),
    ])
    graph.output[0].CopyFrom(helper.make_tensor_value_info(merged, TensorProto.FLOAT, dims))
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def build_metadata(model: YOLO, opset: int, algorithm: str) -> dict:
    """字段集与 FP16 版一致，量化参数换成本次标定的实际值。"""
    return {
        "description": f"Ultralytics YOLO26n model exported to RKNN (INT8 w8a8, calib={algorithm})",
        "author": "Ultralytics",
        "date": __import__("datetime").datetime.now().astimezone().isoformat(),
        "version": version("ultralytics"),
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": int(max(model.stride)),
        "task": model.task,
        "head": type(model.model.model[-1]).__name__,
        "batch": 1,
        "imgsz": IMGSZ,
        "names": model.names,
        "args": {
            "batch": 1,
            "name": TARGET,
            "quantize": 8,
            "quantize_algorithm": algorithm,
            "opset": opset,
            "simplify": True,
        },
        "channels": 3,
        "end2end": bool(model.end2end),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"],
                    help="标定算法；mmse 精度通常更好但标定慢数倍")
    ap.add_argument("--dtype", default="w8a8", choices=["w8a8"],
                    help="w8a8：激活 int8，速度上限最高。w8a16（激活 int16）能同时解决"
                         "类别分数清零与 DFL 噪声，但实测 RK3588 不支持（config 报错），"
                         "那是 RK3576/RK3562 的能力，故不列")
    ap.add_argument("--cos-thresh", type=float, default=None,
                    help="自动混合量化：量化后余弦相似度低于该值的层保留浮点。"
                         "DFL 框解码对 int8 敏感，0.999 可把坐标误差拉回个位数 px")
    args = ap.parse_args()

    for p in (WEIGHTS, IMAGE, ONNX_PATH, DATASET_TXT):
        if not p.exists():
            print(f"缺少文件: {p}" + ("，先跑 src/prepare_calibration.py" if p == DATASET_TXT else ""))
            return 1
    if not (PROJECT_ROOT / "outputs" / f"{REF_STAGE}.json").exists():
        print(f"缺少 PyTorch 基准: outputs/{REF_STAGE}.json，先跑 src/baseline_pt.py")
        return 1

    n_calib = len(DATASET_TXT.read_text().splitlines())
    print(f"标定集 {n_calib} 张  算法 {args.algorithm}")

    model = YOLO(str(WEIGHTS))
    model.predict(source=str(IMAGE), imgsz=IMGSZ, conf=CONF, iou=IOU,
                   rect=False, verbose=False)
    im0 = cv2.imread(str(IMAGE))
    x = model.predictor.preprocess([im0])
    u8 = (x[0].permute(1, 2, 0).numpy() * 255).astype("uint8")[None]

    pt_raw = raw_tensor(model, x)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    n_anchors = sess.get_outputs()[0].shape[-1]
    onnx_raw = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].reshape(-1, n_anchors)
    opset = onnx.load(str(ONNX_PATH)).opset_import[0].version

    rknn = RKNN(verbose=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    rknn_path = MODEL_DIR / f"{ONNX_PATH.stem}-{TARGET}-int8.rknn"

    # 框通道归一化是 int8 激活的专属需求（单 scale 步长 2.5px 的机制见 docstring）；
    # w8a16 激活步长 ~0.02px，整个问题不存在，无需改图
    if args.dtype == "w8a8":
        onnx_norm = BUILD_DIR / f"{ONNX_PATH.stem}-int8norm.onnx"
        normalize_box_channels(ONNX_PATH, onnx_norm, IMGSZ)
        print(f"图改写 {onnx_norm.relative_to(PROJECT_ROOT)}（框通道 ÷ {IMGSZ}，保类别分数）")
    else:
        onnx_norm = ONNX_PATH

    with contextlib.chdir(BUILD_DIR):
        checks = [
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        target_platform=TARGET, quantized_dtype=args.dtype,
                        quantized_algorithm=args.algorithm,
                        auto_hybrid_cos_thresh=args.cos_thresh),
            rknn.load_onnx(model=str(onnx_norm)),
            rknn.build(do_quantization=True, dataset=str(DATASET_TXT)),
            rknn.init_runtime(),
        ]
        if any(rc != 0 for rc in checks):
            print(f"RKNN 流程失败，返回码 {checks}")
            return 1
        rk_raw = rknn.inference(inputs=[u8], data_format="nhwc")[0].reshape(-1, n_anchors)
        if args.dtype == "w8a8":
            # 框通道乘回输入尺寸，与 PyTorch 同一像素量纲再比对（板端 Decode 同样乘回）
            rk_raw[:4] *= IMGSZ
        rc = rknn.export_rknn(str(rknn_path))
    if rc != 0:
        print(f"export_rknn 失败，返回码 {rc}")
        return 1

    from export_rknn import build_metadata as _fp16_meta  # 参考字段集，不直接复用
    del _fp16_meta
    metadata_path = MODEL_DIR / "metadata.yaml"
    YAML.save(str(metadata_path), build_metadata(model, opset, args.algorithm))

    print(f"\nRKNN INT8 {rknn_path.relative_to(PROJECT_ROOT)}  "
          f"({rknn_path.stat().st_size / 1e6:.2f} MB，FP16 为 7.45 MB)")

    # ---- 层1：原始张量 ----
    diffs = {
        "pytorch_vs_rknn_int8": tensor_diff(pt_raw, rk_raw),
        "onnx_vs_rknn_int8": tensor_diff(onnx_raw, rk_raw),
    }
    print("\n[层1] 原始张量")
    for tag, d in diffs.items():
        print(f"  {tag:<22} 框 max|d|={d['box_max_abs_diff']:>10.4f} px  "
              f"归一化={d['box_rel_diff']:.3e}  类别 max|d|={d['cls_max_abs_diff']:.3e}")

    # ---- 层2：检测框 ----
    cand = detect(rk_raw, model.names, im0)
    reports = {}
    for ref_stage in (REF_STAGE, SECOND_REF_STAGE):
        path = PROJECT_ROOT / "outputs" / f"{ref_stage}.json"
        if not path.exists():
            continue
        ref = load_stage(ref_stage)
        box_report = compare(ref["detections"], cand)
        reports[ref_stage] = box_report
        print(f"\n[层2] 检测框 INT8 vs {ref_stage}")
        print(f"  基准 {box_report['ref_count']} 个 / INT8 {box_report['cand_count']} 个")
        print(f"  漏检 {len(box_report['missing'])}  多检 {len(box_report['extra'])}")
        print(f"  最大坐标偏差 {fmt(box_report['max_coord_diff_px'], '.3f')} px  "
              f"最大置信度偏差 {fmt(box_report['max_conf_diff'], '.6f')}  "
              f"最小匹配 IoU {fmt(box_report['min_matched_iou'], '.6f')}")

    json_path = write_stage(
        STAGE,
        {
            "source": rknn_path.name,
            "source_sha256": sha256(rknn_path),
            "image": IMAGE.name,
            "image_sha256": sha256(IMAGE),
            "imgsz": IMGSZ,
            "conf": CONF,
            "iou": IOU,
            "letterbox": "square",
            "inference_path": "one2many + NMS",
            "env": {
                "rknn_toolkit2": version("rknn-toolkit2"),
                "target_platform": TARGET,
                "quantized_dtype": args.dtype,
                "quantize_algorithm": args.algorithm,
                "auto_hybrid_cos_thresh": args.cos_thresh,
                "calibration_images": n_calib,
                "verified_on": "rknn-toolkit2 simulator",
            },
            "detections": cand,
        },
    )
    _, img_path = stage_paths(STAGE)
    cv2.imwrite(str(img_path), draw(cand, im0, model.names))
    print(f"\nINT8 检测已写入 {json_path.relative_to(PROJECT_ROOT)}")

    all_passed = True
    for ref_stage, box_report in reports.items():
        diff = diffs[f"{ref_stage}_vs_rknn_int8"]
        passed = (
            diff["box_rel_diff"] <= TOL_BOX_REL
            and diff["cls_max_abs_diff"] <= TOL_CLS
            and not box_report["missing"]
            and not box_report["extra"]
            and box_report["max_coord_diff_px"] is not None
            and box_report["max_coord_diff_px"] <= TOL_PX
            and box_report["max_conf_diff"] <= TOL_CONF
        )
        all_passed &= passed
        path = write_comparison(ref_stage, STAGE, {
            "tensor": diff,
            "boxes": box_report,
            "tolerances": {"box_rel": TOL_BOX_REL, "cls": TOL_CLS,
                           "px": TOL_PX, "conf": TOL_CONF, "note": "粗估占位，待实测收紧"},
            "passed": passed,
        })
        print(f"报告已写入 {path.relative_to(PROJECT_ROOT)}  passed={passed}")

    print(f"\n结论：{'INT8(模拟器) 与基准一致（按占位容差）' if all_passed else '存在偏差，看上面的实测量级再定阈值'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
