"""把板端 C++ 输出的 JSON 转成统一 schema，并与 PC 侧各阶段比对。

板子只做推理和落 JSON，判定逻辑全在这里：换容差、换基准、换统计口径都不必
重新交叉编译。

用法:
    python3 src/verify_board.py --model models/yolo26n_fp16_rknn_model/yolo26n-rk3588.rknn
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from stages import (
    CONF,
    IOU,
    IMGSZ,
    PROJECT_ROOT,
    compare,
    fmt,
    load_stage,
    sha256,
    write_comparison,
    write_stage,
)

DEFAULT_RAW = PROJECT_ROOT / "outputs" / "board_raw.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo26n_fp16_rknn_model" / "yolo26n-rk3588.rknn"
DEFAULT_IMAGE = PROJECT_ROOT / "assets" / "bus.jpg"

# 板端与 PC 基准之间隔着一层 FP16 的 NPU 算子，坐标允许 2 px 抖动；
# 置信度允许 0.01 抖动。前者由 letterbox 舍入与 FP16 尾数共同决定，
# 后者由检测头在低置信锚点上的排序敏感度决定。
TOL_PX = 2.0
TOL_CONF = 0.01


def check_identity(raw: dict, model: Path, image: Path) -> list[str]:
    """确认板端跑的就是本地这份模型和这张图。

    板子上没有做 sha256，只回传了文件名和字节数，靠这两项加上原图尺寸
    交叉确认——防止拿旧模型或换图的结果来比对。
    """
    problems = []
    if model.exists():
        if raw.get("model") != model.name:
            problems.append(f"板端模型名 {raw.get('model')!r} != 本地 {model.name!r}")
        if raw.get("model_bytes") != model.stat().st_size:
            problems.append(
                f"板端模型字节数 {raw.get('model_bytes')} != 本地 {model.stat().st_size}"
            )
    if image.exists():
        if raw.get("image") != image.name:
            problems.append(f"板端图片名 {raw.get('image')!r} != 本地 {image.name!r}")
        local = cv2.imread(str(image))
        if local is not None:
            size = raw.get("image_size", {})
            if (size.get("width"), size.get("height")) != (local.shape[1], local.shape[0]):
                problems.append(
                    f"板端图片尺寸 {size} != 本地 {local.shape[1]}x{local.shape[0]}"
                )
    return problems


def to_schema(raw: dict, model: Path, image: Path) -> dict:
    """板端 JSON -> stages.py 的统一格式。"""
    detections = [
        {
            "class_id": int(d["class_id"]),
            "class_name": d["class_name"],
            "confidence": round(float(d["confidence"]), 4),
            "xyxy": [round(float(v), 2) for v in d["xyxy"]],
        }
        for d in raw["detections"]
    ]
    env = dict(raw.get("env", {}))
    env["verified_on"] = "RK3588 NPU (librknnrt)"
    if model.exists():
        env["model_sha256"] = sha256(model)
    if image.exists():
        env["image_sha256"] = sha256(image)
    return {
        "source": raw.get("model", model.name),
        "image": raw.get("image", image.name),
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "letterbox": "square",
        "inference_path": "one2many + NMS",
        "env": env,
        "detections": detections,
    }


def parse_perf_detail(text: str) -> dict:
    """解析板端 rknn_perf_detail 的算子耗时汇总表。

    表里每一行是一种算子类型，CPUTime 非零说明有算子没落到 NPU 上——
    这是模拟器验不出来、只有真机能回答的问题。
    """
    rows = []
    in_table = False
    for line in text.splitlines():
        if "Operator Time Consuming Ranking Table" in line:
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) != 7 or not parts[1].isdigit():
            continue
        if parts[0] == "Total":
            break
        rows.append({
            "op": parts[0],
            "calls": int(parts[1]),
            "cpu_us": int(parts[2]),
            "gpu_us": int(parts[3]),
            "npu_us": int(parts[4]),
            "total_us": int(parts[5]),
            "ratio": float(parts[6].rstrip("%")),
        })

    data_ops = {"InputOperator", "OutputOperator"}
    compute_on_cpu = [r for r in rows if r["op"] not in data_ops and r["cpu_us"] > 0]
    graph_us = 0
    for line in text.splitlines():
        if "Total Operator Elapsed Per Frame Time" in line:
            graph_us = int(line.split(":")[-1].strip())
    mem_kb = 0.0
    for line in text.splitlines():
        if "Total Memory Read/Write Per Frame Size" in line:
            mem_kb = float(line.split(":")[-1].strip())

    return {
        "rows": rows,
        "cpu_us": sum(r["cpu_us"] for r in rows),
        "npu_us": sum(r["npu_us"] for r in rows),
        "graph_us": graph_us,
        "mem_read_write_kb": mem_kb,
        "compute_on_cpu": compute_on_cpu,
    }


def report_perf(detail: str) -> dict:
    perf = parse_perf_detail(detail)
    if not perf["rows"]:
        print("  算子明细      格式不匹配，未能解析")
        return perf

    print(f"  算子图耗时    {perf['graph_us'] / 1000:.3f} ms   "
          f"NPU {perf['npu_us'] / 1000:.3f} ms   CPU {perf['cpu_us'] / 1000:.3f} ms")
    print(f"  内存读写      {perf['mem_read_write_kb'] / 1024:.1f} MB/帧")
    print(f"  算子耗时前 5：")
    for r in sorted(perf["rows"], key=lambda x: -x["total_us"])[:5]:
        print(f"    {r['op']:<16} {r['calls']:>4} 次  {r['total_us'] / 1000:>7.3f} ms  "
              f"{r['ratio']:>5.2f}%  CPU {r['cpu_us']} us")

    if perf["compute_on_cpu"]:
        print("  ! 有计算算子回落 CPU：")
        for r in perf["compute_on_cpu"]:
            print(f"    {r['op']}  CPU {r['cpu_us']} us")
    else:
        print("  无计算算子回落 CPU（Input/Output 是数据搬运，本就在 CPU）")
    return perf


def report_env(raw: dict) -> None:
    env = raw.get("env", {})
    print("\n[板端环境]")
    print(f"  runtime API   {env.get('runtime_api', 'n/a')}")
    print(f"  NPU driver    {env.get('npu_driver', 'n/a')}")
    print(f"  输入 / 输出   {env.get('input', 'n/a')}  ->  {env.get('output', 'n/a')}")
    if "npu_ms" in env:
        print(f"  NPU 算子      {env['npu_ms']:.3f} ms   {env.get('fps', 0):.1f} FPS")
    if "set_ms" in env:
        print(f"  inputs_set    {env['set_ms']:.3f} ms")
        print(f"  rknn_run      {env['run_ms']:.3f} ms")
    if "get_ms" in env:
        print(f"  outputs_get   {env['get_ms']:.3f} ms")
    if "wall_ms" in env:
        print(f"  单帧总耗时    {env['wall_ms']:.3f} ms   "
              f"最快 {env.get('wall_min_ms', 0):.3f} / 最慢 {env.get('wall_max_ms', 0):.3f}")

    detail = env.get("perf_detail")
    if detail:
        report_perf(detail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=DEFAULT_RAW, help="板端拉回来的原始 JSON")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="板端实际加载的 .rknn")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="板端实际推理的原图")
    ap.add_argument("--stage", default="board", help="落盘 stage 名；INT8 用 board_int8，避免覆盖 FP16 证据")
    ap.add_argument("--tol-px", type=float, default=TOL_PX, help="坐标偏差容差；INT8 量化误差大，需按实测放宽")
    ap.add_argument("--tol-conf", type=float, default=TOL_CONF, help="置信度偏差容差")
    args = ap.parse_args()

    if not args.board.exists():
        print(f"找不到板端输出: {args.board}\n先跑 board/deploy.sh")
        return 1
    raw = json.loads(args.board.read_text())
    if "image_size" not in raw or "env" not in raw:
        print("板端输出是旧格式（缺 image_size / env），先用 board/build.sh 重新编译后再跑")
        return 1

    problems = check_identity(raw, args.model, args.image)
    if problems:
        print("板端结果与本地文件对不上：")
        for p in problems:
            print(f"  ! {p}")
        return 1

    record = to_schema(raw, args.model, args.image)
    json_path = write_stage(args.stage, record)
    print(f"板端结果已写入 {json_path.relative_to(PROJECT_ROOT)}")
    report_env(raw)

    cand = record["detections"]
    print(f"\n[层2] 检测框 板端 vs PyTorch 基准")
    passed_all = True
    for ref_stage in ("pytorch", "onnx", "rknn"):
        if not (PROJECT_ROOT / "outputs" / f"{ref_stage}.json").exists():
            continue
        ref = load_stage(ref_stage)
        rep = compare(ref["detections"], cand)
        passed = (
            not rep["missing"]
            and not rep["extra"]
            and rep["max_coord_diff_px"] is not None
            and rep["max_coord_diff_px"] <= args.tol_px
            and rep["max_conf_diff"] <= args.tol_conf
        )
        passed_all &= passed
        print(f"\n  vs {ref_stage}:  基准 {rep['ref_count']} / 板端 {rep['cand_count']}  "
              f"漏检 {len(rep['missing'])}  多检 {len(rep['extra'])}")
        print(f"    最大坐标偏差 {fmt(rep['max_coord_diff_px'], '.3f')} px (阈值 {args.tol_px})   "
              f"最大置信度偏差 {fmt(rep['max_conf_diff'], '.6f')} (阈值 {args.tol_conf})   "
              f"最小匹配 IoU {fmt(rep['min_matched_iou'], '.6f')}")
        for d in rep["missing"]:
            print(f"    漏检 {d['class_name']} {d['confidence']:.4f} {d['xyxy']}")
        for d in rep["extra"]:
            print(f"    多检 {d['class_name']} {d['confidence']:.4f} {d['xyxy']}")
        path = write_comparison(ref_stage, args.stage, {"boxes": rep, "passed": passed})
        print(f"    报告 {path.relative_to(PROJECT_ROOT)}  passed={passed}")

    print(f"\n结论：{'板端与 PC 基准一致' if passed_all else '存在偏差，需排查'}")
    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(main())
