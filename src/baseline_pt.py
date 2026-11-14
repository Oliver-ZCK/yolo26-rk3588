"""YOLO26 的 PyTorch 基准，作为各阶段转换的比对锚点。

用法:
    python3 src/baseline_pt.py
"""

import cv2
import ultralytics
from ultralytics import YOLO

from stages import CONF, IOU, IMGSZ, PROJECT_ROOT, sha256, stage_paths, write_stage

WEIGHTS = PROJECT_ROOT / "models" / "yolo26n.pt"
IMAGE = PROJECT_ROOT / "assets" / "bus.jpg"
STAGE = "pytorch"

# bus.jpg 的公开预期检出结果
EXPECTED = {"bus": 1, "person": 4}


def main() -> int:
    for p in (WEIGHTS, IMAGE):
        if not p.exists():
            print(f"缺少文件: {p}")
            return 1

    model = YOLO(str(WEIGHTS))
    # rect=False: 方形 letterbox，与固定 640x640 的 ONNX/RKNN 输入一致
    result = model.predict(source=str(IMAGE), imgsz=IMGSZ, conf=CONF, iou=IOU,
                            rect=False, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            {
                "class_id": int(box.cls),
                "class_name": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "xyxy": [round(v, 2) for v in (x1, y1, x2, y2)],
            }
        )

    print(f"权重 {WEIGHTS.name}   图片 {IMAGE.name} {result.orig_shape[1]}x{result.orig_shape[0]}")
    print(f"imgsz={IMGSZ}  conf={CONF}  推理路径 one2many + NMS\n")
    for d in detections:
        print(f"  {d['class_name']:<8} {d['confidence']:.3f}  {d['xyxy']}")

    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    print(f"\n类别统计: {counts}")

    matched = all(counts.get(k, 0) == v for k, v in EXPECTED.items())
    print(f"对照官方样例预期 {EXPECTED}: {'一致' if matched else '不一致'}")

    json_path = write_stage(
        STAGE,
        {
            "source": WEIGHTS.name,
            "source_sha256": sha256(WEIGHTS),
            "image": IMAGE.name,
            "image_sha256": sha256(IMAGE),
            "imgsz": IMGSZ,
            "conf": CONF,
            "iou": IOU,
            "letterbox": "square",
            "inference_path": "one2many + NMS",
            "env": {"ultralytics": ultralytics.__version__},
            "detections": detections,
        },
    )
    _, img_path = stage_paths(STAGE)
    cv2.imwrite(str(img_path), result.plot())
    print(f"\n基准已写入 {json_path.relative_to(PROJECT_ROOT)}")
    print(f"标注图已写入 {img_path.relative_to(PROJECT_ROOT)}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
