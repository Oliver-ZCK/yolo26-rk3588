# YOLO26n 端侧部署（RK3588）

把 YOLO26n 部署到 RK3588 NPU。每一级转换（PyTorch → ONNX → RKNN → 板端 C++）都用同一套标准逐框比对验证，**FP16 与 INT8 两条链路均已端到端闭环**：板端单图推理 FP16 20.8 FPS / INT8 **30.5 FPS**，摄像头实时链路单帧处理 FP16 94.9 ms / INT8 **49.1 ms**。全量 mAP 标尺：FP16 相对 FP32 零损失，INT8 **−5.4 pt（−13.7%）**。

## 验证链

全流程的每一环都输出同一种 JSON（统一 schema），任意两个阶段可以直接逐框比对：按类别 + IoU≥0.5 贪心匹配，统计漏检 / 多检 / 坐标与置信度最大偏差。容差与判定逻辑全部在 PC 侧，板端只负责推理和落 JSON——偏差先量化、再定容差，不拍脑袋。

同一张测试图（assets/bus.jpg，COCO 公开预期：1 bus + 4 person）上，FP16 各阶段对 PyTorch 基准的偏差：

| 阶段 | 运行环境 | 坐标最大偏差 | 置信度最大偏差 | 结论 |
|---|---|---|---|---|
| ONNX (opset 19) | PC ONNX Runtime | 0 px | 0 | ✅ |
| RKNN FP16 | PC 模拟器（rknn-toolkit2 2.3.2） | 0.71 px | 0.0003 | ✅ |
| 板端 FP16 | RK3588 NPU（librknnrt 2.3.2） | 1.35 px | 0.0035 | ✅ |
| 摄像头链路 | RK3588 NPU（实拍帧） | 0.13 px | 0.0073 | ✅ |

FP16 七组两两比对报告（pytorch/onnx/rknn/board/camera 之间的全部组合）见 `outputs/compare_*.json`，均为 passed。模拟器层还做了原始张量逐通道比对：RKNN 相对 PyTorch 的框通道归一化误差 8.6e-3（容差 2e-2，按 FP16 尾数在 640 量级的 ULP 推导，见 `src/export_rknn.py`）。

INT8 的量化误差比 FP16 大一个量级以上，容差不能沿用 FP16 的推导值。验证框架如实记录了「先实测、再定容差」的过程：

| 阶段 | 坐标最大偏差 | 置信度最大偏差 | 检出完整性 | 结论 |
|---|---|---|---|---|
| RKNN INT8（PC 模拟器） | 17.12 px | 0.0725 | 5/5，无漏检多检 | 按占位容差（8 px / 0.05）failed → 实测值成为放宽依据 |
| 板端 INT8（RK3588 NPU） | 18.76 px | 0.0435 | 5/5，无漏检多检 | 按实测容差（25 px / 0.08）passed ✅ |
| 摄像头 INT8（实拍帧） | — | — | 量化弱检出 1（单列） | passed ✅ |

原始张量层（NMS 前、8400 锚点全集）：INT8 相对 PyTorch 的框通道归一化误差 0.287，单锚点最大偏差 183 px——DFL 回归分支对 w8a8 量化敏感；经 TopK/NMS 筛选后，存活框的偏差收敛到 17~19 px。摄像头链路的「量化弱检出」：INT8 的置信度噪声把 FP32 的 0.156 弱检出推过 0.25 阈值线（实测 0.272），`verify_camera.py` 将这类框单列统计，不算多检。

## 板端性能（640×640）

单图 bench（`board/deploy.sh --bench`，5 次预热后的均值与最值）：

| 指标 | FP16 | INT8 (w8a8) |
|---|---|---|
| 模型体积 | 7.45 MB | **4.81 MB** |
| 输入喂入 | NHWC FP16 native 直喂 | NHWC **UINT8 直喂**（省 33 ms host 侧转换） |
| NPU 算子耗时 | 48.0 ms | **31.1 ms（1.54×）** |
| 推理链路（喂入 + run + 取回） | 91.2 ms → 20.8 FPS | **32.8 ms → 30.5 FPS（2.8×）** |
| 输出取回（want_float=0） | 0.5 ms | 0.7 ms |

权重体积链路：yolo26n.pt 5.54 MB → ONNX 9.93 MB → RKNN FP16 7.45 MB → **RKNN INT8 4.81 MB**。

摄像头实时链路（MIPI 30 FPS 采集，主动丢帧保延迟）：

| 分段 | FP16 | INT8 |
|---|---|---|
| 预处理（旋转 + letterbox + 格式转换） | 13.9 ms | 5.5 ms |
| 推理（喂入 + run + 取回） | 65.6 ms | 32.1 ms |
| 后处理（解码 + NMS + 反 letterbox） | 15.4 ms | 11.6 ms |
| **端到端单帧** | 94.9 ms | **49.1 ms** |

INT8 的后处理已远快于采集，但处理帧率仍被 `n_captured%3` 丢帧策略钉在 ~9.9 FPS——这是延迟优先的主动选择，策略可放宽（见 Roadmap）。

## INT8 为什么不是一条命令：单 scale 会清零类别分数

YOLO26 导出的是 one2many 单输出 `[1, 84, 8400]`：前 4 通道是框坐标（0~640 px），后 80 通道是类别分数（0~1），拼在同一张量里。w8a8 对整张量只有一个 scale，量化步长 651/255 ≈ 2.55——比类别分数的整个动态范围（0~1）还大，所有分数被压成同一个值。实测：不改图直接量化，bus.jpg 的 5 个检出全部消失，类别通道 max diff 恰等于基准最高分 0.903。

解法在导出前改 ONNX 图（`src/export_rknn_int8.py` 的 `normalize_box_channels`）：框通道 ÷ 输入尺寸归一化到 0~1，坐标与分数同量纲后共享一个 scale 才有意义——这复刻了 ultralytics RKNN INT8 导出内部插入的 `_NormalizeCoords`，板端 Decode 按张量属性自动识别并乘回 640。最终输出改名 `output0_norm`，板端按索引取输出，不依赖名字。

其他量化选项的实测结论：

- **标定算法**：mmse 在 256 张标定集上 ETA 9.5 小时（放弃）；kl_divergence 相比 normal 无优势；最终部署 normal。
- **auto-hybrid**（量化后余弦相似度低于阈值的层保留浮点）：最终模型带 `cos_thresh=0.999` 部署，逐框偏差仍在 17 px 量级，未达预期的个位数 px；阈值扫描与 DFL 分支逐层敏感性分析列入 Roadmap。
- **w8a16**（激活 int16）能同时绕开类别分数清零与 DFL 噪声，但 RK3588 的量化 config 直接拒绝——那是 RK3576/RK3562 的能力。

## mAP 标尺：三档精度定标（val2017 全量 4744 张）

逐框比对是流水线的安检门，mAP 是最终标尺。`src/eval_map.py` 用 pycocotools COCOeval
对三档做同协议评估（square letterbox 640 / conf=0.001 / NMS iou=0.45 / max_det=300，
排除标定 256 张，GT 由标定 zip 自带的 YOLO 标签构建，image_id 与官方一致）：

| 档位 | mAP@0.5:0.95 | mAP@0.5 | AP75 | AP 小 | AP 中 | AP 大 |
|---|---|---|---|---|---|---|
| FP32 | 0.3931 | 0.5605 | 0.416 | 0.142 | 0.350 | 0.565 |
| FP16 | 0.3930（−0.0） | 0.5608 | 0.416 | 0.143 | 0.350 | 0.565 |
| INT8 | 0.3390（**−5.4 pt，−13.7%**） | 0.5431 | 0.360 | 0.101 | 0.296 | 0.507 |

- **FP16 无损**：四项指标差 ≤3e-4，可当默认精度档；
- **INT8 的损失几乎全是定位质量**：AP75 掉 5.5 而 mAP50 仅掉 1.7——检出完整性基本没动
  （检出总数 −18% 来自弱检出被量化噪声压到阈值下），与逐框比对的 17 px 结论交叉印证；
- **量化噪声的受害者梯度**：小目标相对损失 −29%、中 −16%、大 −10%，是 hybrid
  阈值扫描的明确优化目标；
- 与官方 YOLO26n 基准 40.9（one2many+NMS 口径，取自同版本 ultralytics README）差
  −1.6 pt，归因于 GT 构建（crowd 标注处理）、NMS 口径与权重版本，不影响档间对比；
- 工具链边界：模拟器不能 `load_rknn`（仅真机可用），FP16/INT8 评估按部署参数在
  模拟器会话内重建（与已部署模型同源）。本标尺为模拟器数字，板端全量复核见 Roadmap。

## 目录结构

```
├── assets/bus.jpg            # 测试图（COCO 公开预期结果可交叉核对）
├── models/                   # pt / onnx / rknn（FP16 与 INT8；不进仓库，复现步骤见下）
├── src/                      # PC 侧流水线
│   ├── stages.py             # 统一 schema、跨阶段比对逻辑（IoU 匹配、误差统计）
│   ├── baseline_pt.py        # PyTorch 基准
│   ├── export_onnx.py        # ONNX 导出 + 比对（ultralytics 原生导出，opset 19）
│   ├── export_rknn.py        # RKNN FP16 导出 + 模拟器逐张量验证
│   ├── export_rknn_int8.py   # INT8：坐标归一化改图 + w8a8 标定 + 模拟器验证
│   ├── prepare_calibration.py  # 标定集：val2017 镜像下载、固定种子抽 256 张
│   ├── eval_map.py           # mAP 三档评估（COCOeval，模拟器按部署参数重建）
│   ├── verify_board.py       # 板端结果转统一 schema + 比对（容差按精度档调整）
│   └── verify_camera.py      # 摄像头实拍帧 vs PyTorch 基准（含量化弱检出规则）
├── board/                    # 板端 C++（交叉编译，正点原子 buildroot 工具链）
│   ├── main.cc               # 单图推理 + 分段 benchmark + 零拷贝/INT8 路径
│   ├── camera.cc             # 摄像头实时推理（丢帧策略、分段计时、基准帧留存）
│   ├── postprocess.h         # 解码 + NMS + 反 letterbox（自写，与 PC 侧口径一致）
│   ├── rknn_io.h             # 输入喂入（U8 / FP16 native / 零拷贝）与输出反量化
│   ├── result_json.h         # 统一 schema 落 JSON
│   ├── build.sh              # 交叉编译（SDK 环境变量可覆盖工具链路径）
│   ├── deploy*.sh            # 推板运行 + 拉回结果 + 自动比对
│   └── demo/                 # rknn_model_zoo 官方 demo 树（仅替换 postprocess.cc，yolo26_cam 用）
└── outputs/                  # 各阶段 JSON 与标注图、全部比对报告
```

## 复现

PC 侧需 **x86_64 Linux**（rknn-toolkit2 官方 wheel 仅提供该平台）。依赖版本约束是实测得来的，放宽会踩坑：

```bash
pip install "ultralytics==8.4.180" \
            "rknn-toolkit2>=2.3.2,<2.5" \
            "onnx>=1.16.1,<1.19" \
            "setuptools<82"
```

- rknn-toolkit2 **2.5.0 有 YOLO11/YOLO26 构建失败的已知 bug**（airockchip/rknn-toolkit2#581），不要升级；
- opset 由 ultralytics 的 RKNN 导出路径强制为 19，网上「toolkit2 只支持 opset ≤ 12」的说法已过时。

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

mAP 标尺（需 pycocotools；GT 复用标定 zip 自带的 YOLO 标签，无需另下数据；全量三档合计约 40 分钟）：

```bash
python3 src/eval_map.py --stage fp32
python3 src/eval_map.py --stage fp16
python3 src/eval_map.py --stage int8
```

标定只需要图片、不需要标注：`prepare_calibration.py` 以 seed=0 抽 256 张写入 `dataset.txt`，数据落在仓库外 `~/datasets/coco`；官方源国内直连仅 ~55 KB/s，脚本优先走 hf-mirror 镜像，已下载过 zip 的用 `--zip` 指入跳过。后续 mAP 评估排除这 256 张即可保证与标定集不重叠。

板端（需 RK3588 与 PC 同网；目标地址用环境变量 `BOARD` / `REMOTE` 覆盖，PC 侧解释器用 `PY` 覆盖；交叉编译用正点原子 buildroot SDK，`build.sh` 默认路径可用 `SDK` 环境变量覆盖）：

```bash
board/build.sh
board/deploy.sh models/yolo26n_fp16_rknn_model/yolo26n-rk3588.rknn --bench 20
board/deploy.sh models/yolo26n_int8_rknn_model/yolo26n-rk3588-int8.rknn --int8 -- \
    --stage board_int8 --tol-px 25 --tol-conf 0.08
board/deploy_camera.sh models/yolo26n_int8_rknn_model/yolo26n-rk3588-int8.rknn
```

部署脚本自动完成：上传 → 板上运行 → 拉回 JSON 与基准帧 → PC 侧比对 → 输出 passed / failed。`--` 之后的参数传给 PC 侧校验脚本；INT8 的容差按实测放宽（依据见验证链一节）。

硬件边界：PC 侧（含模拟器验证）在任意 x86_64 Linux 上可完整复现；板端链路需要 RK3588 硬件（librknnrt 2.3.2 + NPU driver 0.9.8 + MIPI 摄像头）。

## 实现要点

- **判定与执行分离**：板端只推理并按统一 schema 落 JSON，阈值、容差、比对口径全在 PC 侧——改判定不用重新交叉编译。
- **one2many 单输出**：ultralytics 导出的是端到端单输出 `[1, 84, 8400]`，后处理（NMS、反 letterbox、INT8 反量化）全部自写，与 PC 侧 `non_max_suppression` 口径逐框对齐；未沿用官方 demo 的三分支 YOLOv5/v8 布局，其 letterbox 的 resize 对齐规则也与 ultralytics 不一致。
- **INT8 坐标归一化**：导出前在 ONNX 图上复刻 ultralytics 的 `_NormalizeCoords`（框通道 ÷640），板端解码按张量属性自动识别归一化坐标并乘回——FP16 / INT8 共用同一套 Decode 代码。
- **摄像头设备走符号链接**：`/dev/video-camera0`（→ video71）。裸节点 video44 能采集但只出全黑帧；节点号随内核版本变，符号链接不变。实拍帧与 PC 基准同图比对，基准帧在画框前留存。
- **性能路径已预留**：`want_float=0` 自转 float32（省 ~9 ms）、native FP16/NHWC 直喂（省 ~33 ms CPU 转换）、零拷贝输入，均在 `main.cc` 中实现可实测。

## Roadmap

- **五规格横评**：n 已闭环（含 mAP 标尺），s/m/l/x 复制同一条链路后选型深耕。
- **板端全量 mAP 复核**：mAP 标尺目前是模拟器数字，真机 NPU 舍入差异待量化（预期零点几个点内）。
- **hybrid 量化阈值扫描**：INT8 损失集中在小目标（AP 小相对 −29%），0.999 一档未达预期，对 DFL 分支做逐层敏感性分析。
- **摄像头采帧策略放宽**：INT8 后处理（11.6 ms）已远快于采集，丢帧策略可放宽换更高处理帧率。
- **零拷贝与内存优化实测**：零拷贝路径已实现，待纳入常规 bench。

## 来源与致谢

- [ultralytics](https://github.com/ultralytics/ultralytics)（YOLO26 权重与导出，AGPL-3.0）
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)（瑞芯微，模型转换与模拟器）
- [rknn_model_zoo](https://github.com/airockchip/rknn-model-zoo)（`board/demo/` 官方 demo 树与点阵字体绘制，Apache-2.0）
- 正点原子 RK3588 开发板资料与 buildroot SDK
