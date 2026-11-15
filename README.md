# YOLO26n 端侧部署（RK3588）

把 YOLO26n 部署到 RK3588 NPU。每一级转换（PyTorch → ONNX → RKNN → 板端 C++）都用同一套标准逐框比对验证，**FP16 与 INT8 两条链路均已端到端闭环**：板端单图推理 FP16 20.8 FPS / INT8 **30.5 FPS**，摄像头实时链路单帧处理 FP16 94.9 ms / INT8 **49.1 ms**。精度标尺（val2017 全量 4744 张，官方 COCO 标注 GT，真机最终口径）：FP16 相对 FP32 零损失（0.3979 vs 0.3973）；INT8 **0.2786（−11.9 pt，相对 −30%）**，模拟器口径 0.3501 **系统性乐观 7.2 pt**——量化指标必须真机复核。

## 验证链

全流程的每一环都输出同一种 JSON（统一 schema），任意两个阶段可以直接逐框比对：按类别 + IoU≥0.5 贪心匹配，统计漏检 / 多检 / 坐标与置信度最大偏差。容差与判定逻辑全部在 PC 侧，板端只负责推理和落 JSON。

同一张测试图（assets/bus.jpg，COCO 公开预期：1 bus + 4 person）上，FP16 各阶段对 PyTorch 基准的偏差：

| 阶段 | 运行环境 | 坐标最大偏差 | 置信度最大偏差 | 结论 |
|---|---|---|---|---|
| ONNX (opset 19) | PC ONNX Runtime | 0 px | 0 | ✅ |
| RKNN FP16 | PC 模拟器（rknn-toolkit2 2.3.2） | 0.71 px | 0.0003 | ✅ |
| 板端 FP16 | RK3588 NPU（librknnrt 2.3.2） | 1.35 px | 0.0035 | ✅ |
| 摄像头链路 | RK3588 NPU（实拍帧） | 0.13 px | 0.0073 | ✅ |

七个两两比对组合见 `outputs/compare_*.json`，均 passed。INT8 各级偏差：

| 阶段 | 坐标最大偏差 | 置信度最大偏差 | 检出完整性 | 结论 |
|---|---|---|---|---|
| RKNN INT8（PC 模拟器） | 17.12 px | 0.0725 | 5/5，无漏检多检 | passed ✅ |
| 板端 INT8（RK3588 NPU） | 18.76 px | 0.0435 | 5/5，无漏检多检 | passed ✅ |
| 摄像头 INT8（实拍帧） | — | — | 量化弱检出 1（单列） | passed ✅ |

原始张量层（NMS 前、8400 锚点全集）：FP16 框通道归一化误差 8.6e-3；INT8 达 0.287、单锚点最大偏差 183 px，经 TopK/NMS 筛选后存活框偏差收敛到 17~19 px。摄像头 INT8 会把 FP32 的弱检出推过阈值（0.156 → 0.272），按「量化弱检出」单列统计，不算多检。

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

## INT8 为什么不是一条命令：单 scale 会清零类别分数

YOLO26 导出 one2many 单输出 `[1, 84, 8400]`：前 4 通道是框坐标（0~640 px），后 80 通道是类别分数（0~1），拼在同一张量里。w8a8 对整张量只有一个 scale，量化步长 ≈2.55——比类别分数的整个动态范围还大，所有分数被压成同一个值。实测不改图直接量化，bus.jpg 的 5 个检出全部消失。

解法在导出前改 ONNX 图（`src/export_rknn_int8.py` 的 `normalize_box_channels`）：框通道 ÷ 输入尺寸归一化到 0~1，坐标与分数同量纲后共享一个 scale 才有意义——复刻 ultralytics RKNN INT8 导出内部插入的 `_NormalizeCoords`，板端解码按张量属性自动识别并乘回 640，FP16 / INT8 共用同一套 Decode 代码。

其他实测：标定算法 normal 胜出（mmse ETA 9.5 小时、kl_divergence 无优势）；w8a16 能同时绕开类别分数清零与 DFL 噪声，但 RK3588 的量化 config 直接拒绝——那是 RK3576/RK3562 的能力。

## mAP 标尺：三档定标与官方对齐

三档同协议（square letterbox 640 / conf=0.001 / NMS iou=0.45 / max_det=300，官方 COCO 标注 GT，排除标定 256 张）。FP16 四项指标差 ≤3e-4，可当默认精度档：

| 档位 | mAP@0.5:0.95 | mAP@0.5 | AP75 | AP 小 | AP 中 | AP 大 |
|---|---|---|---|---|---|---|
| FP32 | 0.3973 | 0.5674 | 0.420 | 0.197 | 0.432 | 0.581 |
| FP16 | 0.3972（−0.0） | 0.5677 | 0.420 | 0.197 | 0.432 | 0.580 |
| INT8 | 0.3501（**−4.7 pt，−11.9%**） | 0.5549 | 0.371 | 0.151 | 0.379 | 0.533 |

FP32 基准与官方 YOLO26n（40.9）对齐：自写官方协议 40.33、ultralytics 原生 val 40.23 双路径互证，残差 ~0.6 归官方评测环境差异。评估路径自写的原因：官方 val 只能跑 .pt/.onnx（RKNN 模型模拟器不能 `load_rknn`、板端需 librknnrt），三档可比必须同一套代码路径。

## 模拟器 vs 真机：量化指标必须真机复核

200 张固定子集、同协议对比（旧标尺口径）：

| 配置 | 运行处 | mAP | AP75 | AP小 | 体积 | 板端推理 |
|---|---|---|---|---|---|---|
| FP16 | 模拟器 / 板端 | 0.4406 / **0.4421** | 0.465 / 0.467 | 0.208 / 0.209 | 7.45 MB | 63.7 ms |
| 纯 INT8 | 模拟器 / 板端 | 0.3913 / **0.3201** | 0.415 / 0.309 | 0.164 / 0.081 | 4.81 MB | 25.7 ms |
| DFL 定点（回归分支 9 层 FP16） | 板端 | 0.3243 | 0.323 | 0.084 | 4.98 MB | ~29 ms |
| auto-hybrid 0.9999 | 模拟器 | 0.4119 | 0.443 | 0.178 | 7.48 MB | — |

FP16 两端一致（差 0.15 pt），评估链路互证；INT8 真机比模拟器低 7 pt：AP小 腰斩、AP75 崩 10 pt——单图逐框比对看不到的差距，在小目标上被 NPU 执行噪声放大。**模拟器的 PTQ 数字系统性乐观，部署决策必须以真机标尺为准。**

## INT8 掉点的归因：分歧全在框坐标通道，宽高最狠

板端（`--dump-raw` 存模型原生 int8）与模拟器的 8400 锚点 × 84 通道输出逐通道对比：

| 通道组 | mean\|diff\|（两图） | max\|diff\| | >1 int8 步长占比 |
|---|---|---|---|
| 框 w/h | 6.7 / 9.5 px | 202 / 370 px | 高 |
| 框 cx/cy | 4.1 / 4.8 px | 102 / 130 px | ~75% |
| 类别分数 | **0.000** | ≤0.86（个别锚点） | **0.0%** |

类别分数通道基本 bit 级一致，分歧全在框坐标——两端检出集合相同，崩的是定位质量。DFL 的宽高由两组距离预测**相减**得到，误差结构性相加（w/h 发散是 cx/cy 的两倍）；w/h 均值 6.7 px 对 20 px 的小目标即 IoU 碎裂。真机 INT8 的损失因此集中在定位（AP75 −36%）与小目标（AP小 −55%）。

## PTQ 到顶：hybrid 死路，下一步 QAT

- **auto-hybrid 从未生效**（阈值只是 config 参数，开关是 `build(auto_hybrid=True)`，部署版漏了后者）；真启用后非单调（保浮点层改变下游激活分布，标定 scale 失配）、体积涨回 FP16 量级、无甜点；
- 手动 hybrid 把 DFL 回归分支保 FP16 仅换回 **+0.4 pt**——真机 INT8 的损失大头在 NPU 执行差异，不在量化参数；
- **真机口径的交易**：INT8 用 11.9 pt mAP 换 2.5× 速度、−35% 体积。实时场景值，精度敏感场景直接 FP16（0.442 无损）。

## 板端全量复核（4744 张）：最终口径

FP16 / INT8 各跑一遍全量评估（4744 张，与模拟器标尺同协议同 GT，官方 COCO 标注）：

| 档位 | 体积 | mAP | mAP50 | AP75 | AP小 | AP中 | AP大 |
|---|---|---|---|---|---|---|---|
| FP32 基准（PC） | — | 0.3973 | — | — | — | — | — |
| FP16 板端 | 7.45 MB | **0.3979** | 0.5682 | 0.4198 | 0.1962 | 0.4328 | 0.5809 |
| INT8 板端 | 4.81 MB | **0.2786** | 0.5116 | 0.2686 | 0.0887 | 0.2841 | 0.4742 |

FP16 板端 ≥ FP32 基准：整条板端链路与 PC 侧对齐到无损——INT8 的全部损失只能归给量化本身。模拟器乐观偏差在全量坐实：**7.2 pt**（0.3501 vs 0.2786）。全量真机吞吐（单帧含 pre/infer/post）：FP16 88.1 ms / INT8 46.7 ms，端到端 1.9×、纯 NPU 2.4×。

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
│   ├── prepare_calibration.py  # 标定集与官方标注下载：固定种子抽 256 张
│   ├── eval_map.py           # mAP 三档评估（COCOeval，双 GT 标尺，模拟器按部署参数重建）
│   ├── val_official.py       # ultralytics 原生 val 交叉验证（官方协议第二路径）
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

依赖约束、标定/转换/评估/板端部署的完整命令见 [docs/REPRO.md](docs/REPRO.md)。

## Roadmap

已完成：FP16/INT8 双精度部署闭环、mAP 三档标尺、hybrid 扫描、真机复核与差异归因、板端全量复核、GT 标尺升级与全档位官方 GT 重测。范围聚焦：五规格横评裁撤，集中优化 n。当前待办：

- **量化感知训练（QAT）**：PTQ 已实测到顶（最敏感分支保 FP16 仅 +0.4 pt），INT8 精度要往下打只剩 QAT。
- **NPU 多核（core_mask）**：RK3588 三核 NPU 当前单核推理，多核存在速度空间，待实测。
- **摄像头采帧策略放宽**：INT8 后处理已远快于采集，丢帧策略可放宽换更高处理帧率。
- **零拷贝与内存优化实测**：零拷贝路径已实现，待纳入常规 bench。

## 来源与致谢

- [ultralytics](https://github.com/ultralytics/ultralytics)（YOLO26 权重与导出，AGPL-3.0）
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)（瑞芯微，模型转换与模拟器）
- [rknn_model_zoo](https://github.com/airockchip/rknn-model-zoo)（`board/demo/` 官方 demo 树与点阵字体绘制，Apache-2.0）
- 正点原子 RK3588 开发板资料与 buildroot SDK
