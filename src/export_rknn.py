"""导出 YOLO26n 的 RKNN 模型（FP16），并与 PyTorch 基准逐层比对。

复用已验证的 models/yolo26n.onnx。一次 build 同时产出固化的 .rknn 与模拟器验证
结果，两者来自同一张图——模拟器验过的图变换就是板端要跑的那张图。

用法:
    python3 src/export_rknn.py
"""

import contextlib
from datetime import datetime
from importlib.metadata import version

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from rknn.api import RKNN
from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.plotting import Annotator, colors

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

WEIGHTS = PROJECT_ROOT / "models" / "yolo26n.pt"
IMAGE = PROJECT_ROOT / "assets" / "bus.jpg"
ONNX_PATH = PROJECT_ROOT / "models" / "yolo26n.onnx"
BUILD_DIR = PROJECT_ROOT / "outputs" / "rknn_build"
MODEL_DIR = PROJECT_ROOT / "models" / "yolo26n_fp16_rknn_model"
STAGE = "rknn"
REF_STAGE = "pytorch"
SECOND_REF_STAGE = "onnx"

TARGET = "rk3588"

# FP16 尾数 10 位：640 量级的坐标 ULP 为 0.25~0.5 px，误差沿网络深度累积。
# 实测模拟器上 5.48 px（归一化 8.6e-3），取 2 倍余量覆盖真机 NPU 的舍入差异。
TOL_BOX_REL = 2e-2      # 框通道归一化误差上限（除以输入边长）
TOL_CLS = 2e-3          # 类别通道绝对误差上限
TOL_PX = 2.0            # 检测框角点偏差上限（px）
TOL_CONF = 5e-3         # 置信度偏差上限


def raw_tensor(model: YOLO, x: torch.Tensor) -> np.ndarray:
    """模型前向的原始头输出，形状 (4 + nc, anchors)。"""
    with torch.no_grad():
        out = model.model(x)
    t = out if isinstance(out, torch.Tensor) else next(i for i in out if isinstance(i, torch.Tensor))
    return t.detach().cpu().numpy().reshape(-1, t.shape[-1])


def detect(tensor: np.ndarray, names: dict, im0: np.ndarray) -> list[dict]:
    """原始头输出 -> 原图坐标系的检测框。

    头输出是 cxcywh，non_max_suppression 期望的正是这个格式，不能当 xyxy 用。
    """
    pred = non_max_suppression(
        torch.from_numpy(tensor[None]).float(),
        conf_thres=CONF, iou_thres=IOU, nc=len(names),
    )[0].numpy()
    boxes = scale_boxes((IMGSZ, IMGSZ), pred[:, :4], im0.shape[:2])
    return [
        {
            "class_id": int(c),
            "class_name": names[int(c)],
            "confidence": round(float(s), 4),
            "xyxy": [round(float(v), 2) for v in b],
        }
        for b, s, c in zip(boxes, pred[:, 4], pred[:, 5])
    ]


def build_metadata(model: YOLO, opset: int) -> dict:
    """metadata.yaml 的内容，供 ultralytics 的 RKNNBackend 还原推理参数。

    字段集与 ultralytics 原生 export(format="rknn") 产出的 metadata.yaml 一致。
    """
    return {
        "description": "Ultralytics YOLO26n model exported to RKNN (FP16)",
        "author": "Ultralytics",
        "date": datetime.now().astimezone().isoformat(),
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
            "quantize": 16,
            "opset": opset,
            "simplify": True,
        },
        "channels": 3,
        "end2end": bool(model.end2end),
    }


def tensor_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """两个原始头输出之间的误差统计。

    框坐标量级是像素，类别分数量级是概率，归一化分母各按物理量程取。
    """
    d = np.abs(a - b)
    return {
        "shape": list(a.shape),
        "max_abs_diff": float(d.max()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "box_max_abs_diff": float(d[:4].max()),
        "box_rmse": float(np.sqrt((d[:4] ** 2).mean())),
        "box_rel_diff": float(d[:4].max() / IMGSZ),
        "cls_max_abs_diff": float(d[4:].max()),
        "cls_rmse": float(np.sqrt((d[4:] ** 2).mean())),
    }


def draw(detections: list[dict], im0: np.ndarray, names: dict) -> np.ndarray:
    annotator = Annotator(im0.copy(), example=str(names.values()))
    for d in detections:
        annotator.box_label(
            d["xyxy"], f"{d['class_name']} {d['confidence']:.2f}", color=colors(d["class_id"], True)
        )
    return annotator.result()


def main() -> int:
    for p in (WEIGHTS, IMAGE, ONNX_PATH):
        if not p.exists():
            print(f"缺少文件: {p}")
            return 1
    if not (PROJECT_ROOT / "outputs" / f"{REF_STAGE}.json").exists():
        print(f"缺少 PyTorch 基准: outputs/{REF_STAGE}.json，先跑 src/baseline_pt.py")
        return 1

    model = YOLO(str(WEIGHTS))
    model.predict(source=str(IMAGE), imgsz=IMGSZ, conf=CONF, iou=IOU,
                   rect=False, verbose=False)
    im0 = cv2.imread(str(IMAGE))
    x = model.predictor.preprocess([im0])
    # RKNN 的归一化参数已烧进模型，运行时直接收 uint8 原图
    u8 = (x[0].permute(1, 2, 0).numpy() * 255).astype("uint8")[None]
    print(f"PyTorch 输入 {tuple(x.shape)} {x.dtype}   RKNN 输入 {u8.shape} {u8.dtype}")

    pt_raw = raw_tensor(model, x)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    n_anchors = sess.get_outputs()[0].shape[-1]
    onnx_raw = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].reshape(-1, n_anchors)

    opset = onnx.load(str(ONNX_PATH)).opset_import[0].version

    rknn = RKNN(verbose=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    rknn_path = MODEL_DIR / f"{ONNX_PATH.stem}-{TARGET}.rknn"

    # build 会在当前工作目录落下 check0/check2/check3 三个中间图，切目录隔离
    with contextlib.chdir(BUILD_DIR):
        checks = [
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        target_platform=TARGET),
            rknn.load_onnx(model=str(ONNX_PATH)),
            rknn.build(do_quantization=False, rknn_batch_size=1),
            rknn.init_runtime(),
        ]
        if any(rc != 0 for rc in checks):
            print(f"RKNN 流程失败，返回码 {checks}")
            return 1
        rk_raw = rknn.inference(inputs=[u8], data_format="nhwc")[0].reshape(-1, n_anchors)
        rc = rknn.export_rknn(str(rknn_path))
    if rc != 0:
        print(f"export_rknn 失败，返回码 {rc}")
        return 1

    metadata_path = MODEL_DIR / "metadata.yaml"
    YAML.save(str(metadata_path), build_metadata(model, opset))

    print(f"\nRKNN {rknn_path.relative_to(PROJECT_ROOT)}  "
          f"({rknn_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  输入 {u8.shape} uint8      输出 {tuple(rk_raw.shape)}")
    print(f"  metadata {metadata_path.relative_to(PROJECT_ROOT)}")

    # ---- 层1：原始张量 ----
    diffs = {
        "pytorch_vs_onnx": tensor_diff(pt_raw, onnx_raw),
        "onnx_vs_rknn": tensor_diff(onnx_raw, rk_raw),
        "pytorch_vs_rknn": tensor_diff(pt_raw, rk_raw),
    }
    print("\n[层1] 原始张量")
    for tag, d in diffs.items():
        print(f"  {tag:<18} 框 max|d|={d['box_max_abs_diff']:>10.4f} px  "
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
        print(f"\n[层2] 检测框 RKNN vs {ref_stage}")
        print(f"  基准 {box_report['ref_count']} 个 / RKNN {box_report['cand_count']} 个")
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
                "float_dtype": "float16",
                "verified_on": "rknn-toolkit2 simulator",
            },
            "detections": cand,
        },
    )
    _, img_path = stage_paths(STAGE)
    cv2.imwrite(str(img_path), draw(cand, im0, model.names))
    print(f"\nRKNN 检测已写入 {json_path.relative_to(PROJECT_ROOT)}")
    print(f"标注图已写入 {img_path.relative_to(PROJECT_ROOT)}")

    for ref_stage, box_report in reports.items():
        diff = diffs[f"{ref_stage}_vs_rknn"]
        passed = (
            diff["box_rel_diff"] <= TOL_BOX_REL
            and diff["cls_max_abs_diff"] <= TOL_CLS
            and not box_report["missing"]
            and not box_report["extra"]
            and box_report["max_coord_diff_px"] is not None
            and box_report["max_coord_diff_px"] <= TOL_PX
            and box_report["max_conf_diff"] <= TOL_CONF
        )
        path = write_comparison(ref_stage, STAGE, {
            "tensor": diff,
            "boxes": box_report,
            "passed": passed,
        })
        print(f"报告已写入 {path.relative_to(PROJECT_ROOT)}  passed={passed}")

    passed = reports and all(
        diffs[f"{s}_vs_rknn"]["box_rel_diff"] <= TOL_BOX_REL
        and diffs[f"{s}_vs_rknn"]["cls_max_abs_diff"] <= TOL_CLS
        and not r["missing"] and not r["extra"]
        and r["max_coord_diff_px"] is not None
        and r["max_coord_diff_px"] <= TOL_PX and r["max_conf_diff"] <= TOL_CONF
        for s, r in reports.items()
    )
    print(f"\n结论：{'RKNN 与 PyTorch 一致' if passed else '存在偏差，需排查'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
