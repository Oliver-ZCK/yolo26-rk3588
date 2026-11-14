"""摄像头链路的精度基准：板端那一帧的检测结果 vs PC 侧 PyTorch 基准。

板端把喂给模型的那一帧原图存成 PNG，同时把该帧的检测结果落 JSON。
这里对同一张 PNG 跑 PyTorch 基准，再逐框比对——和 verify_board.py 同一套判定口径，
只是比对对象从固定图片换成摄像头抓的帧。

用法:
    python3 src/verify_camera.py
"""

import argparse
import json
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from stages import (
    CONF,
    IOU,
    IMGSZ,
    PROJECT_ROOT,
    compare,
    fmt,
    iou,
    sha256,
    write_comparison,
    write_stage,
)

DEFAULT_RAW = PROJECT_ROOT / "outputs" / "camera_raw.json"
DEFAULT_FRAME = PROJECT_ROOT / "outputs" / "cam_frame.png"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo26n.pt"

# 与 verify_board.py 同一口径：板端隔着 FP16 的 NPU 算子，坐标允许 2 px 抖动
TOL_PX = 2.0
TOL_CONF = 0.01


def run_pt(frame: Path, weights: Path, conf: float = CONF) -> list[dict]:
    """对任意图片跑 PyTorch 基准，参数与 baseline_pt.py 完全一致。"""
    model = YOLO(str(weights))
    result = model.predict(source=str(frame), imgsz=IMGSZ, conf=conf, iou=IOU,
                            rect=False, verbose=False)[0]
    return [
        {
            "class_id": int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "xyxy": [round(float(v), 2) for v in box.xyxy[0]],
        }
        for box in result.boxes
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=DEFAULT_RAW, help="板端拉回的 camera_out.json")
    ap.add_argument("--frame", type=Path, default=DEFAULT_FRAME, help="板端存下的那一帧原图")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="PyTorch 权重")
    ap.add_argument("--stage", default="camera", help="落盘 stage 名；INT8 用 camera_int8，避免覆盖 FP16 证据")
    ap.add_argument("--tol-px", type=float, default=TOL_PX, help="坐标偏差容差；INT8 需按实测放宽")
    ap.add_argument("--tol-conf", type=float, default=TOL_CONF, help="置信度偏差容差")
    args = ap.parse_args()

    if not args.board.exists():
        print(f"找不到板端输出: {args.board}\n先跑 board/deploy_camera.sh")
        return 1
    if not args.frame.exists():
        print(f"找不到基准帧: {args.frame}\n板端可能没抓到有目标的帧")
        return 1

    raw = json.loads(args.board.read_text())
    env = raw.get("env", {})
    print("\n[板端摄像头链路]")
    print(f"  runtime API   {env.get('runtime_api', 'n/a')}")
    print(f"  NPU driver    {env.get('npu_driver', 'n/a')}")
    print(f"  输入 / 输出   {env.get('input', 'n/a')}  ->  {env.get('output', 'n/a')}")
    if "fps" in env:
        print(f"  处理帧率      {env['fps']:.2f} FPS   帧数 {env.get('frames', 0)}"
              f"   丢帧 {env.get('dropped', 0)}")
    if "npu_ms" in env:
        print(f"  NPU 算子      {env['npu_ms']:.3f} ms")

    frame_detections = raw.get("detections", [])
    if not frame_detections:
        print("  ! 板端这一帧没有检出任何目标，没有精度基准可比")
        return 1

    ref = run_pt(args.frame, args.weights)
    print(f"\n[精度基准] 同一帧：板端 {len(frame_detections)} 个框 / "
          f"PyTorch {len(ref)} 个框")
    for d in ref:
        print(f"    基准  {d['class_name']:<10} {d['confidence']:.4f}  {d['xyxy']}")
    for d in frame_detections:
        print(f"    板端  {d['class_name']:<10} {d['confidence']:.4f}  {d['xyxy']}")

    rep = compare(ref, frame_detections)

    # 量化弱检出：INT8 的置信度噪声会把 FP32 的弱检出（<CONF）推过阈值线，
    # 摄像头场景里几乎每帧都有这类边缘目标。判定规则：多检的框若能在
    # 低阈值基准（conf=0.05）里找到同类同位（IoU>=0.5）的对应，就是同一个
    # 真实目标被量化搬动了分数，不算多检；凭空出现、基准里完全没有的框
    # 才是真多检。FP16 链路 extra 为空时此规则不激活。
    weak = []
    true_extra = rep["extra"]
    if true_extra:
        ref_low = run_pt(args.frame, args.weights, conf=0.05)
        weak, true_extra = [], []
        for e in rep["extra"]:
            best, best_iou = None, 0.0
            for g in ref_low:
                if g["class_id"] != e["class_id"]:
                    continue
                v = iou(e["xyxy"], g["xyxy"])
                if v > best_iou:
                    best, best_iou = g, v
            if best is not None and best_iou >= 0.5:
                e["pt_conf"] = best["confidence"]
                weak.append(e)
            else:
                true_extra.append(e)

    # 容差只约束"配对成功"的框；仅有弱检出时没有配对，容差无从谈起，
    # 通过与否由 missing / 真多检 / 弱检出规则决定
    matched = rep["max_coord_diff_px"] is not None
    passed = (
        not rep["missing"]
        and not true_extra
        and (not matched or (rep["max_coord_diff_px"] <= args.tol_px
                             and rep["max_conf_diff"] <= args.tol_conf))
    )
    print(f"\n  漏检 {len(rep['missing'])}  多检 {len(true_extra)}"
          f"  量化弱检出 {len(weak)}")
    print(f"  最大坐标偏差 {fmt(rep['max_coord_diff_px'], '.3f')} px (阈值 {args.tol_px})   "
          f"最大置信度偏差 {fmt(rep['max_conf_diff'], '.6f')} (阈值 {args.tol_conf})   "
          f"最小匹配 IoU {fmt(rep['min_matched_iou'], '.6f')}")
    for d in rep["missing"]:
        print(f"    漏检 {d['class_name']} {d['confidence']:.4f} {d['xyxy']}")
    for d in true_extra:
        print(f"    多检 {d['class_name']} {d['confidence']:.4f} {d['xyxy']}")
    for d in weak:
        print(f"    量化弱检出 {d['class_name']} {d['confidence']:.4f} {d['xyxy']}"
              f"  (FP32 同框 {d['pt_conf']:.4f}，低于阈值，不算多检)")

    record = {
        "source": raw.get("model", ""),
        "image": args.frame.name,
        "image_sha256": sha256(args.frame),
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "letterbox": "square",
        "inference_path": "one2many + NMS",
        "env": {**env, "verified_on": "RK3588 NPU (librknnrt), camera"},
        "detections": frame_detections,
    }
    json_path = write_stage(args.stage, record)
    print(f"\n板端结果已写入 {json_path.relative_to(PROJECT_ROOT)}")

    # PC 侧对同一帧的基准也落一个 stage，和其他阶段一样可回溯
    pt_record = {
        "source": args.weights.name,
        "source_sha256": sha256(args.weights),
        "image": args.frame.name,
        "image_sha256": sha256(args.frame),
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "letterbox": "square",
        "inference_path": "one2many + NMS",
        "env": {"ultralytics": ultralytics.__version__},
        "detections": ref,
    }
    pt_path = write_stage(f"{args.stage}_pt", pt_record)
    print(f"PC 基准已写入 {pt_path.relative_to(PROJECT_ROOT)}")

    path = write_comparison(f"{args.stage}_pt", args.stage,
                            {"boxes": rep, "weak_detections": weak, "passed": passed})
    print(f"报告 {path.relative_to(PROJECT_ROOT)}  passed={passed}")
    print(f"\n结论：{'摄像头链路与 PC 基准一致' if passed else '存在偏差，需排查'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
