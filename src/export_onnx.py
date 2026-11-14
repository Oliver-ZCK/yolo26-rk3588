"""导出 YOLO26n 的 ONNX，并与 PyTorch 基准逐层比对。

用法:
    python3 src/export_onnx.py
"""

import cv2
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

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
REF_STAGE = "pytorch"
STAGE = "onnx"

OPSET = 19

TOL_TENSOR = 1e-2
TOL_PX = 1.0
TOL_CONF = 0.01


def export_onnx() -> Path:
    model = YOLO(str(WEIGHTS))
    out = Path(
        model.export(format="onnx", imgsz=IMGSZ, opset=OPSET, dynamic=False,
                     simplify=True, batch=1, verbose=False)
    )
    if out.resolve() != ONNX_PATH.resolve():
        out.replace(ONNX_PATH)
    return ONNX_PATH


def graph_io(path: Path) -> tuple[list, list]:
    graph = onnx.load(str(path)).graph
    dims = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    return dims(graph.input[0]), dims(graph.output[0])


def raw_output(model: YOLO, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model.model(x)
    if isinstance(out, torch.Tensor):
        return out.cpu().numpy()
    for item in out:
        if isinstance(item, torch.Tensor):
            return item.cpu().numpy()
    raise TypeError(f"未找到张量输出: {type(out)}")


def detections_of(result, names: dict) -> list[dict]:
    return [
        {
            "class_id": int(b.cls),
            "class_name": names[int(b.cls)],
            "confidence": round(float(b.conf), 4),
            "xyxy": [round(float(v), 2) for v in b.xyxy[0]],
        }
        for b in result.boxes
    ]


def main() -> int:
    ref = load_stage(REF_STAGE) if (PROJECT_ROOT / "outputs" / f"{REF_STAGE}.json").exists() else None
    if ref is None:
        print(f"缺少 PyTorch 基准: outputs/{REF_STAGE}.json，先跑 src/baseline_pt.py")
        return 1

    path = export_onnx()
    in_shape, out_shape = graph_io(path)
    print(f"ONNX {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"  input  {in_shape}")
    print(f"  output {out_shape}\n")

    model = YOLO(str(WEIGHTS))
    model.predict(source=str(IMAGE), imgsz=IMGSZ, conf=CONF, iou=IOU,
                   rect=False, verbose=False)
    im0 = cv2.imread(str(IMAGE))
    x = model.predictor.preprocess([im0])
    print(f"输入张量 {tuple(x.shape)} {x.dtype}")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_raw = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    pt_raw = raw_output(model, x)

    diff = np.abs(onnx_raw[0] - pt_raw[0])
    tensor_report = {
        "shape": list(pt_raw.shape),
        "max_abs_diff": float(diff.max()),
        "box_max_abs_diff": float(diff[:4].max()),
        "cls_max_abs_diff": float(diff[4:].max()),
    }
    print("\n[层1] 原始张量 PT vs ONNX")
    print(f"  形状 {tensor_report['shape']}")
    print(f"  最大误差 {tensor_report['max_abs_diff']:.3e}  "
          f"(框通道 {tensor_report['box_max_abs_diff']:.3e} / 类别通道 {tensor_report['cls_max_abs_diff']:.3e})")

    onnx_model = YOLO(str(path))
    result = onnx_model.predict(source=str(IMAGE), imgsz=IMGSZ, conf=CONF, iou=IOU,
                                rect=False, verbose=False)[0]
    cand = detections_of(result, model.names)

    json_path = write_stage(
        STAGE,
        {
            "source": path.name,
            "source_sha256": sha256(path),
            "image": IMAGE.name,
            "image_sha256": sha256(IMAGE),
            "imgsz": IMGSZ,
            "conf": CONF,
            "iou": IOU,
            "letterbox": "square",
            "inference_path": "one2many + NMS",
            "env": {"opset": OPSET},
            "detections": cand,
        },
    )
    _, img_path = stage_paths(STAGE)
    cv2.imwrite(str(img_path), result.plot())
    print(f"\nONNX 检测已写入 {json_path.relative_to(PROJECT_ROOT)}")
    print(f"标注图已写入 {img_path.relative_to(PROJECT_ROOT)}")

    box_report = compare(ref["detections"], cand)
    print("\n[层2] 检测框 ONNX vs PyTorch 基准")
    print(f"  基准 {box_report['ref_count']} 个 / ONNX {box_report['cand_count']} 个")
    print(f"  漏检 {len(box_report['missing'])}  多检 {len(box_report['extra'])}")
    print(f"  最大坐标偏差 {fmt(box_report['max_coord_diff_px'], '.3f')} px  "
          f"最大置信度偏差 {fmt(box_report['max_conf_diff'], '.6f')}  "
          f"最小匹配 IoU {fmt(box_report['min_matched_iou'], '.4f')}")

    passed = (
        tensor_report["max_abs_diff"] <= TOL_TENSOR
        and not box_report["missing"]
        and not box_report["extra"]
        and box_report["max_coord_diff_px"] is not None
        and box_report["max_coord_diff_px"] <= TOL_PX
        and box_report["max_conf_diff"] <= TOL_CONF
    )

    report_path = write_comparison(REF_STAGE, STAGE, {
        "tensor": tensor_report,
        "boxes": box_report,
        "passed": passed,
    })
    print(f"\n报告已写入 {report_path.relative_to(PROJECT_ROOT)}")
    print(f"结论：{'ONNX 与 PyTorch 一致' if passed else '存在偏差，需排查'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
