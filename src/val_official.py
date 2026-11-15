"""官方协议交叉验证：用 ultralytics 原生 val 流程量 FP32 基准。

eval_map.py --protocol official 用自写 predict 路径复刻官方口径（rect +
NMS iou 0.7 + COCOeval）；本脚本把评估交给 ultralytics 自己的 val 管线
（含其内部 NMS 与组批实现，is_coco 终评同样走 COCOeval）。两条路互证：
若数字接近，与官方 40.9 的残差只能归给权重版本；若明显更高，说明自写
路径还有细节未对齐。

数据 yaml 由脚本生成（绝对路径指向 ~/datasets/coco；图片、YOLO 标签、
官方标注已在标准位置，val 不会触发下载）。

用法:
    python3 src/val_official.py
"""
import pathlib

import yaml
from ultralytics import YOLO

from stages import PROJECT_ROOT

DATA_ROOT = pathlib.Path.home() / "datasets" / "coco"


def main() -> int:
    model = YOLO(str(PROJECT_ROOT / "models" / "yolo26n.pt"))
    cfg = dict(path=str(DATA_ROOT), train="images/val2017",  # 占位：yaml 校验强制要求，val 模式不加载
               val="images/val2017", is_coco=True,
               names={int(k): v for k, v in model.names.items()})
    data = PROJECT_ROOT / "outputs" / "coco_val_official.yaml"
    data.write_text(yaml.dump(cfg, allow_unicode=True))
    print(f"data yaml -> {data}")

    r = model.val(data=str(data), imgsz=640, plots=False)
    b = r.box
    print("\nultralytics 原生 val：")
    print(f"  mAP@0.5:0.95 = {b.map * 100:.2f}   mAP@0.5 = {b.map50 * 100:.2f}   AP75 = {b.map75 * 100:.2f}")
    print("  对照：自写官方协议 40.33 / 官方 README 40.9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
