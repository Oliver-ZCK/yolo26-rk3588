"""各阶段产物的统一格式与比对。

PyTorch / ONNX / RKNN 三个阶段产出同一种 JSON，板端 C++ 也按同一格式输出，
任意两个阶段可直接比对，不必为每个阶段单独写一套对比脚本。
"""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SCHEMA_VERSION = 1
MATCH_IOU = 0.5

# 各阶段共用的推理配置。三个阶段必须用同一组值，否则检测结果不可比。
IMGSZ = 640
CONF = 0.25
IOU = 0.45


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_paths(stage: str) -> tuple[Path, Path]:
    return OUTPUTS_DIR / f"{stage}.json", OUTPUTS_DIR / f"{stage}.jpg"


def write_stage(stage: str, record: dict) -> Path:
    """写 outputs/<stage>.json。record 需含 source/image/imgsz/conf/letterbox/inference_path/detections。"""
    json_path, _ = stage_paths(stage)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA_VERSION, "stage": stage, **record}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return json_path


def load_stage(stage: str) -> dict:
    json_path, _ = stage_paths(stage)
    return json.loads(json_path.read_text())


def write_comparison(ref_stage: str, cand_stage: str, report: dict) -> Path:
    path = OUTPUTS_DIR / f"compare_{ref_stage}_{cand_stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reference": ref_stage, "candidate": cand_stage, **report}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def fmt(value: float | None, spec: str, placeholder: str = "n/a") -> str:
    """误差指标在没有可比数据时是 None，格式化时统一显示占位符。"""
    return placeholder if value is None else format(value, spec)


def iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(ref: list[dict], cand: list[dict]) -> dict:
    """按类别 + IoU 贪心匹配两阶段的检测结果。

    没有匹配上的检测不参与误差统计，此时三项误差指标为 None 而非 0——
    "没有可比数据"与"误差为零"必须能区分开。
    """
    used: set[int] = set()
    missing: list[dict] = []
    max_px: float | None = None
    max_conf: float | None = None
    min_iou: float | None = None

    for r in ref:
        best, best_iou = None, 0.0
        for i, c in enumerate(cand):
            if i in used or c["class_id"] != r["class_id"]:
                continue
            v = iou(r["xyxy"], c["xyxy"])
            if v > best_iou:
                best, best_iou = i, v
        if best is None or best_iou < MATCH_IOU:
            missing.append(r)
            continue
        used.add(best)
        c = cand[best]
        coord_diff = max(abs(a - b) for a, b in zip(r["xyxy"], c["xyxy"]))
        conf_diff = abs(r["confidence"] - c["confidence"])
        max_px = coord_diff if max_px is None else max(max_px, coord_diff)
        max_conf = conf_diff if max_conf is None else max(max_conf, conf_diff)
        min_iou = best_iou if min_iou is None else min(min_iou, best_iou)

    return {
        "ref_count": len(ref),
        "cand_count": len(cand),
        "missing": missing,
        "extra": [c for i, c in enumerate(cand) if i not in used],
        "max_coord_diff_px": None if max_px is None else round(max_px, 4),
        "max_conf_diff": None if max_conf is None else round(max_conf, 6),
        "min_matched_iou": None if min_iou is None else round(min_iou, 6),
    }
