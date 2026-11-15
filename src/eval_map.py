"""COCO val2017 全量 mAP 评估：FP32 / FP16 / INT8 三档精度标尺。

三档必须同协议：square letterbox 640、conf=0.001（低阈值让 PR 曲线完整）、
NMS iou=0.45、max_det=300——任何一档协议不同，「掉点」就掺了水分。
GT 两套标尺：--gt official 用官方 instances_val2017.json（含 iscrowd 人群
标注与 ignore 规则，可与官方基准对表，缺省）；--gt yolo 用标定 zip 自带
YOLO 标签构建的旧标尺（历史报告口径，全部 iscrowd=0）。部署标尺下标定
256 张默认排除；--protocol official 为 FP32 官方协议对照（rect + NMS
iou 0.7，全量 5000 张含标定——FP32 与标定集无污染一说）。

FP32 走 ultralytics predict；FP16/INT8 走 rknn-toolkit2 模拟器，输入路径
与 export_rknn(_int8).py 已验证的一致。mAP 计算用 pycocotools COCOeval。

用法:
    python3 src/eval_map.py --stage fp32                          # 官方 GT，部署标尺
    python3 src/eval_map.py --stage fp32 --gt yolo                # 旧 YOLO 标签 GT（历史口径）
    python3 src/eval_map.py --stage fp32 --protocol official      # 官方协议对照（rect + iou 0.7）
    python3 src/eval_map.py --stage fp16 [--limit 500] [--gt yolo]
    python3 src/eval_map.py --stage int8 [--limit 500] [--cos-thresh 0.99|0]
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rknn.api import RKNN
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression

from stages import IMGSZ, IOU, PROJECT_ROOT

DATA_ROOT = Path.home() / "datasets" / "coco"
IMAGES_DIR = DATA_ROOT / "images" / "val2017"
LABELS_DIR = DATA_ROOT / "labels" / "val2017"
CALIB_TXT = PROJECT_ROOT / "outputs" / "rknn_int8_build" / "dataset.txt"
GT_JSON = PROJECT_ROOT / "outputs" / "map_gt.json"          # --gt yolo 的缓存
ANN_JSON = DATA_ROOT / "annotations" / "instances_val2017.json"  # --gt official

WEIGHTS = PROJECT_ROOT / "models" / "yolo26n.pt"
ONNX_PATH = PROJECT_ROOT / "models" / "yolo26n.onnx"
FP16_RKNN = PROJECT_ROOT / "models" / "yolo26n_fp16_rknn_model" / "yolo26n-rk3588.rknn"
INT8_RKNN = PROJECT_ROOT / "models" / "yolo26n_int8_rknn_model" / "yolo26n-rk3588-int8.rknn"

# COCO 协议的评估口径：conf 压到 0.001 让 PR 曲线覆盖全区间，
# NMS iou 沿用本工程各阶段共用的 0.45；官方协议对照用 ultralytics val 的 0.7
CONF_EVAL = 0.001
MAX_DET = 300
SUBSET_SEED = 0
IOU_OFFICIAL = 0.7


def image_list(limit: int | None, exclude_calib: bool = True) -> list[Path]:
    """评估集：val2017 全量，部署标尺排除标定 256 张；--limit 取固定种子的子集。

    官方协议对照不排除——FP32 与标定集无污染一说，全量 5000 才是官方口径。
    """
    calib = set()
    if exclude_calib:
        calib = {Path(line.strip()).name for line in CALIB_TXT.read_text().splitlines() if line.strip()}
    images = sorted(p for p in IMAGES_DIR.glob("*.jpg") if p.name not in calib)
    if limit:
        images = sorted(random.Random(SUBSET_SEED).sample(images, limit))
    return images


def build_gt(images: list[Path], names: dict) -> dict:
    """YOLO 标签 -> COCO GT json。image_id 用文件名数字段，与官方 ID 一致。"""
    cat = [{"id": c, "name": name} for c, name in names.items()]
    imgs, anns, aid = [], [], 0
    for p in images:
        im = cv2.imread(str(p))
        if im is None:
            raise SystemExit(f"读图失败: {p}")
        h, w = im.shape[:2]
        img_id = int(p.stem)
        imgs.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        label = LABELS_DIR / f"{p.stem}.txt"
        if not label.exists():  # 无目标的纯背景图，本来就没有 GT
            continue
        for line in label.read_text().splitlines():
            c, cx, cy, bw, bh = (float(v) for v in line.split())
            bw, bh = bw * w, bh * h
            anns.append({
                "id": aid, "image_id": img_id, "category_id": int(c),
                "bbox": [(cx * w) - bw / 2, (cy * h) - bh / 2, bw, bh],
                "area": bw * bh, "iscrowd": 0,
            })
            aid += 1
    return {"images": imgs, "annotations": anns, "categories": cat}


def pt_stage(images: list[Path], protocol: str = "deploy") -> list[dict]:
    """分块预测：conf=0.001 时 CPU NMS 的中间张量大，一次性喂全量被 OOM 杀过。

    deploy：square letterbox + NMS iou 0.45，与 FP16/INT8/板端同口径；
    official：rect 保长宽比 + NMS iou 0.7——ultralytics val 的默认口径，对表官方基准。
    """
    iou_nms, rect = (IOU, False) if protocol == "deploy" else (IOU_OFFICIAL, True)
    model = YOLO(str(WEIGHTS))
    dets = []
    chunk = 100
    for i in range(0, len(images), chunk):
        batch = images[i:i + chunk]
        results = model.predict(source=[str(p) for p in batch], imgsz=IMGSZ, conf=CONF_EVAL,
                                iou=iou_nms, rect=rect, max_det=MAX_DET, stream=True, verbose=False)
        for p, r in zip(batch, results):
            img_id = int(p.stem)
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                dets.append({"image_id": img_id, "category_id": int(b.cls),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(b.conf)})
        done = min(i + chunk, len(images))
        print(f"    {done}/{len(images)}", flush=True)
    return dets


def rknn_stage(images: list[Path], stage: str, cos_thresh: float | None = None) -> list[dict]:
    """模拟器前向，输出统一到原图坐标系的检出清单。

    模拟器只认 load_onnx + build 的会话内模型（load_rknn 仅真机可用，
    这正是 export_rknn.py「一次 build 同时产出固化模型与模拟器验证」的
    原因），所以按部署时的原参数重建：FP16 不量化；INT8 复刻改图 ONNX +
    w8a8/normal + 同一份标定集，与已部署模型同源。

    cos_thresh: None = 部署档 0.999；0 = 关闭 auto-hybrid（纯 INT8 对照）；
    >0 = 扫描该阈值。不同阈值各用独立 build 目录，中间图互不覆盖。
    """
    import contextlib

    if stage == "fp16":
        build_onnx, build_dir = ONNX_PATH, PROJECT_ROOT / "outputs" / "map_build_fp16"
        quant = dict(do_quantization=False, rknn_batch_size=1)
        cfg = dict()
    else:
        build_onnx = PROJECT_ROOT / "outputs" / "rknn_int8_build" / "yolo26n-int8norm.onnx"
        if not build_onnx.exists():
            from export_rknn_int8 import normalize_box_channels
            normalize_box_channels(ONNX_PATH, build_onnx, IMGSZ)
        thresh = 0.999 if cos_thresh is None else cos_thresh
        tag = "deploy" if cos_thresh is None else ("nohybrid" if thresh <= 0 else f"ct{thresh:g}")
        build_dir = PROJECT_ROOT / "outputs" / f"map_build_int8_{tag}"
        quant = dict(do_quantization=True, dataset=str(CALIB_TXT))
        cfg = dict(quantized_dtype="w8a8", quantized_algorithm="normal")
        if thresh > 0:
            # 教训实证：阈值在 config 里只是存着，开关是 build(auto_hybrid=True)——
            # 部署版曾只配阈值不开关，mAP 三点扫描发现三位小数全同才揭穿
            cfg["auto_hybrid_cos_thresh"] = thresh
            quant["auto_hybrid"] = True

    yolo = YOLO(str(WEIGHTS))
    yolo.predict(source=str(images[0]), imgsz=IMGSZ, rect=False, verbose=False)

    rknn = RKNN(verbose=False)
    build_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(build_dir):  # build 会落 check0/2/3 中间图，隔离
        if rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                       target_platform="rk3588", **cfg) != 0:
            raise SystemExit("rknn.config 失败")
        if rknn.load_onnx(model=str(build_onnx)) != 0:
            raise SystemExit("load_onnx 失败")
        if rknn.build(**quant) != 0:
            raise SystemExit("build 失败")
        rknn_path = build_dir / "model.rknn"  # 落盘量体积：扫描曲线的代价轴
        rknn.export_rknn(str(rknn_path))
        if rknn.init_runtime() != 0:
            raise SystemExit("init_runtime 失败")

    int8_norm = stage == "int8"
    dets = []
    t0 = time.time()
    for idx, p in enumerate(images):
        im0 = cv2.imread(str(p))
        x = yolo.predictor.preprocess([im0])
        u8 = (x[0].permute(1, 2, 0).numpy() * 255).astype("uint8")[None]
        raw = rknn.inference(inputs=[u8], data_format="nhwc")[0].reshape(4 + 80, -1)
        if int8_norm:
            raw[:4] *= IMGSZ
        pred = non_max_suppression(torch.from_numpy(raw[None]).float(), CONF_EVAL, IOU,
                                   nc=80, max_det=MAX_DET)[0].numpy()
        boxes = scale_boxes((IMGSZ, IMGSZ), pred[:, :4], im0.shape[:2])
        img_id = int(p.stem)
        for b, s, c in zip(boxes, pred[:, 4], pred[:, 5]):
            x1, y1, x2, y2 = (float(v) for v in b)
            dets.append({"image_id": img_id, "category_id": int(c),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        if (idx + 1) % 100 == 0:
            rate = (idx + 1) / (time.time() - t0)
            print(f"    {idx + 1}/{len(images)}  {rate:.2f} img/s  "
                  f"剩余约 {(len(images) - idx - 1) / rate / 60:.0f} 分钟", flush=True)
    rknn.release()
    size_mb = round(rknn_path.stat().st_size / 1e6, 2)
    print(f"    模型体积 {size_mb} MB（FP16 档 7.45 / 纯 INT8 档 4.81）", flush=True)
    return dets, size_mb


def run_cocoeval(gt: dict, dets: list[dict], img_ids: list[int]) -> dict:
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    coco_dt = coco.loadRes(dets)
    ev = COCOeval(coco, iouType="bbox")
    ev.cocoDt = coco_dt
    ev.params.imgIds = img_ids  # GT 缓存按全量构建，--limit 子集靠这里圈定
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return dict(zip(keys, (float(v) for v in ev.stats)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--limit", type=int, default=None, help="固定种子抽 N 张先跑（三档一致才可比）")
    ap.add_argument("--cos-thresh", type=float, default=None,
                    help="仅 int8：auto-hybrid 阈值；缺省=部署档 0.999，0=关闭 hybrid（对照）")
    ap.add_argument("--from-board", metavar="JSON", default=None,
                    help="跳过推理，直接评估板端 --dir 批量输出的 json（COCO 格式）")
    ap.add_argument("--gt", choices=["official", "yolo"], default="official",
                    help="GT 标尺：official=官方标注含 iscrowd（缺省，可与官方对表）；"
                         "yolo=标定 zip 标签构建的旧标尺（历史报告口径）")
    ap.add_argument("--protocol", choices=["deploy", "official"], default="deploy",
                    help="deploy=square 640 + NMS iou 0.45（部署标尺，缺省）；"
                         "official=rect + NMS iou 0.7（ultralytics val 口径，仅 fp32，对表官方基准）")
    args = ap.parse_args()

    if args.protocol == "official":
        if args.gt != "official":
            raise SystemExit("--protocol official 需要官方 GT（--gt official）")
        if args.stage != "fp32":
            raise SystemExit("official 协议对照仅支持 --stage fp32（rect 与板端 square 640 不可比）")
    exclude_calib = args.protocol != "official"
    iou_nms = IOU if args.protocol == "deploy" else IOU_OFFICIAL

    images = image_list(args.limit, exclude_calib)
    print(f"评估集 {len(images)} 张（{'排除标定 256' if exclude_calib else '官方全量口径，含标定 256'}）"
          f"  协议 conf={CONF_EVAL} iou={iou_nms} max_det={MAX_DET} "
          f"{'rect' if args.protocol == 'official' else 'square'}  GT={args.gt}")

    cat_ids = None
    if args.gt == "official":
        if not ANN_JSON.exists():
            raise SystemExit(f"官方标注不存在: {ANN_JSON}\n"
                             "python3 src/prepare_calibration.py --annotations 下载，"
                             "或已下载的 zip 用 --annotations-zip 指入")
        gt = json.loads(ANN_JSON.read_text())
        # ultralytics 类别序 = 官方类目 id 升序；检出的 0~79 序号按此映射回官方 id 再评估
        cat_ids = sorted(c["id"] for c in gt["categories"])
        if len(cat_ids) != 80:
            raise SystemExit(f"官方标注类别数异常: {len(cat_ids)}")
        print(f"GT 官方标注：{len(gt['images'])} 张 / {len(gt['annotations'])} 框（含 iscrowd）")
    else:
        full = image_list(None, exclude_calib)
        if not GT_JSON.exists():
            names = YOLO(str(WEIGHTS)).names
            t0 = time.time()
            gt = build_gt(full, names)
            GT_JSON.write_text(json.dumps(gt))
            print(f"GT 构建（全量缓存）{len(gt['annotations'])} 个框 -> {GT_JSON.name}（{time.time() - t0:.0f}s）")
        else:
            gt = json.loads(GT_JSON.read_text())
            if len(gt["images"]) != len(full):
                raise SystemExit(f"GT 缓存 {len(gt['images'])} 张与全量 {len(full)} 张不一致，"
                                 f"删除 {GT_JSON.name} 重跑")
            print(f"GT 缓存 {len(gt['annotations'])} 个框（{len(gt['images'])} 张图）")

    if args.from_board:
        dets = json.loads(Path(args.from_board).read_text())["detections"]
        model_desc = f"板端 {Path(args.from_board).name}"
        size_mb = None
    elif args.stage == "fp32":
        dets = pt_stage(images, args.protocol)
        model_desc = WEIGHTS.name + ("（官方协议对照 rect+iou0.7）" if args.protocol == "official" else "")
        size_mb = None
    else:
        dets, size_mb = rknn_stage(images, args.stage, args.cos_thresh)
        if args.stage == "fp16":
            model_desc = f"{FP16_RKNN.name}（模拟器按部署参数重建）"
        else:
            thresh = 0.999 if args.cos_thresh is None else args.cos_thresh
            model_desc = f"INT8 auto_hybrid_cos_thresh={thresh:g}（模拟器重建）"
    print(f"{args.stage}: {len(dets)} 个检出（{model_desc}）")

    if cat_ids is not None:
        for d in dets:
            d["category_id"] = cat_ids[d["category_id"]]
    metrics = run_cocoeval(gt, dets, [int(p.stem) for p in images])
    report = {
        "stage": args.stage, "model": model_desc,
        "protocol": {"imgsz": IMGSZ, "conf": CONF_EVAL, "iou_nms": iou_nms,
                     "max_det": MAX_DET, "letterbox": "square" if args.protocol == "deploy" else "rect",
                     "gt": args.gt, "excluded_calib": 256 if exclude_calib else 0,
                     "limit": args.limit, "subset_seed": SUBSET_SEED if args.limit else None,
                     "cos_thresh": (0.999 if args.cos_thresh is None else args.cos_thresh)
                     if args.stage == "int8" else None},
        "images": len(images), "detections": len(dets), "model_size_mb": size_mb,
        "metrics": {k: round(v, 5) for k, v in metrics.items()},
    }
    suffix = f"_{args.limit}" if args.limit else ""
    if args.from_board:
        suffix = f"_board_{Path(args.from_board).stem}{suffix}"
    else:
        if args.stage == "int8" and args.cos_thresh is not None:
            suffix += "-nohybrid" if args.cos_thresh <= 0 else f"-ct{args.cos_thresh:g}"
    if args.gt == "official":
        suffix += "_gtofficial"
    if args.protocol == "official":
        suffix += "_officialproto"
    out = PROJECT_ROOT / "outputs" / f"map_{args.stage}{suffix}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"报告 -> {out.relative_to(PROJECT_ROOT)}")

    m = metrics
    print(f"\n  mAP@0.5:0.95  {m['AP']:.4f}    mAP@0.5  {m['AP50']:.4f}    AP75 {m['AP75']:.4f}")
    print(f"  AP小/中/大    {m['APs']:.4f} / {m['APm']:.4f} / {m['APl']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
