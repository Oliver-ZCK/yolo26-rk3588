# 复现手册

PC 侧需 **x86_64 Linux**（rknn-toolkit2 官方 wheel 仅提供该平台）。依赖版本约束是实测得来的，放宽会踩坑：

```bash
pip install "ultralytics==8.4.180" \
            "rknn-toolkit2>=2.3.2,<2.5" \
            "onnx>=1.16.1,<1.19" \
            "setuptools<82"
```

- rknn-toolkit2 **2.5.0 有 YOLO11/YOLO26 构建失败的已知 bug**（airockchip/rknn-toolkit2#581），不要升级；
- opset 由 ultralytics 的 RKNN 导出路径强制为 19。

权重不进仓库，一条命令获取（放入 `models/`）：

```bash
cd models && python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" && cd ..
```

FP16 链路（每步自带验证）：

```bash
python3 src/baseline_pt.py    # PyTorch 基准
python3 src/export_onnx.py    # ONNX 导出 + 验证
python3 src/export_rknn.py    # RKNN FP16 导出 + 模拟器验证
```

INT8 链路（先备标定集）：

```bash
python3 src/prepare_calibration.py    # 下载 COCO val2017（优先 hf-mirror 源）并固定种子抽 256 张
python3 src/export_rknn_int8.py       # 改图 + w8a8 标定 + 模拟器验证
```

mAP 标尺（GT 用官方 COCO 标注：`python3 src/prepare_calibration.py --annotations` 下载，
走 hf-mirror 单文件，官方 S3 直连不可用；全量三档合计约 40 分钟；
历史报告为旧 YOLO 标签 GT 口径，复现加 `--gt yolo`）：

```bash
python3 src/eval_map.py --stage fp32
python3 src/eval_map.py --stage fp32 --protocol official   # 官方协议对照（rect + iou 0.7，全量 5000）
python3 src/val_official.py                                # ultralytics 原生 val 交叉验证
python3 src/eval_map.py --stage fp16
python3 src/eval_map.py --stage int8
python3 src/eval_map.py --stage int8 --cos-thresh 0.9995   # auto-hybrid 阈值扫描（0=关闭，作纯 INT8 对照）
python3 src/eval_map.py --stage int8 --from-board outputs/board_int8_eval.json --limit 200   # 板端批量输出（子集试点）
python3 src/eval_map.py --stage fp16 --from-board outputs/board_fp16_full.json   # 板端全量 4744 张
python3 src/eval_map.py --stage int8 --from-board outputs/board_int8_full.json
```

标定数据落在仓库外 `~/datasets/coco`；`prepare_calibration.py` 以 seed=0 抽 256 张写入 `dataset.txt`，已下载过 zip 的用 `--zip` 指入跳过；mAP 评估排除这 256 张保证与标定集不重叠。

板端批量评估（COCO 协议与 PC 侧一致）：

```bash
rsync -a --files-from=outputs/full_images.txt ~/datasets/coco/images/val2017/ $BOARD:/home/zck/yolo26/eval_full/images/
# 板上：./yolo26 <model.rknn> x --dir ~/yolo26/eval_full/images --eval-out ~/yolo26/eval_full/<tag>.json
#   conf=0.001 可用 --eval-conf 覆盖；FP16 模型与 INT8 模型共用同一命令
```

板端（需 RK3588 与 PC 同网；目标地址用环境变量 `BOARD` / `REMOTE` 覆盖，PC 侧解释器用 `PY` 覆盖；摄像头设备走符号链接 `/dev/video-camera0`——裸节点只出全黑帧且节点号随内核漂移；交叉编译用正点原子 buildroot SDK，`build.sh` 默认路径可用 `SDK` 环境变量覆盖）：

```bash
board/build.sh
board/deploy.sh models/yolo26n_fp16_rknn_model/yolo26n-rk3588.rknn --bench 20
board/deploy.sh models/yolo26n_int8_rknn_model/yolo26n-rk3588-int8.rknn --int8 -- \
    --stage board_int8 --tol-px 25 --tol-conf 0.08
board/deploy_camera.sh models/yolo26n_int8_rknn_model/yolo26n-rk3588-int8.rknn
```

部署脚本自动完成：上传 → 板上运行 → 拉回 JSON 与基准帧 → PC 侧比对 → 输出 passed / failed。`--` 之后的参数传给 PC 侧校验脚本。

硬件边界：PC 侧（含模拟器验证）在任意 x86_64 Linux 上可完整复现；板端链路需要 RK3588 硬件（librknnrt 2.3.2 + NPU driver 0.9.8 + MIPI 摄像头）。
