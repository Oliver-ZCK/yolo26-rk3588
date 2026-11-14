#!/bin/bash
# 把摄像头实时推理程序推到 RK3588，跑一段，把统计和基准帧拉回 PC。
#
# 用法: deploy_camera.sh <model.rknn> [--seconds N] [--display]
# 环境变量: BOARD=root@192.168.1.10  REMOTE=/home/zck/yolo26
#
# 默认不带 --display：SSH 会话里没有 X display，imshow 会失败，
# 只跑推理并统计速度。有显示器时在板子本地终端直接跑 ./camera 更直观。
set -euo pipefail

BOARD="${BOARD:-root@192.168.1.10}"
REMOTE="${REMOTE:-/home/zck/yolo26}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/board/build/camera"
PY="${PY:-/home/dr-wei-arm/anaconda3/envs/py3.11-tk2-2.3.2/bin/python}"

MODEL="${1:?用法: deploy_camera.sh <model.rknn> [--seconds N] [--display]}"
shift

[[ -f "$MODEL" ]] || { echo "找不到模型: $MODEL"; exit 1; }
[[ -x "$BIN" ]] || { echo "还没编译，先跑 board/build.sh"; exit 1; }

RUN_SECONDS=30
DISPLAY_OPT="--no-display"
EXTRA=()
VERIFY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds) RUN_SECONDS="$2"; shift 2 ;;
    --display) DISPLAY_OPT=""; shift ;;
    --) shift; VERIFY_ARGS=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

MODEL="$(readlink -f "$MODEL")"
TAG="$(basename "$MODEL" .rknn)"
REMOTE_DIR="$REMOTE/$TAG"
ARGS=("$DISPLAY_OPT" --seconds "$RUN_SECONDS" --save-frame cam_frame.png "${EXTRA[@]}")

echo ">> 上传到 $BOARD:$REMOTE_DIR"
ssh "$BOARD" mkdir -p "$REMOTE_DIR"
scp -q "$BIN" "$MODEL" "$BOARD:$REMOTE_DIR/"

echo ">> 板上跑摄像头实时推理（${ARGS[*]}）"
ssh "$BOARD" "cd '$REMOTE_DIR' && ./camera $(basename "$MODEL") ${ARGS[*]}"

echo ">> 拉回统计与基准帧"
scp -q "$BOARD:$REMOTE_DIR/camera_out.json" "$ROOT/outputs/camera_raw.json"
scp -q "$BOARD:$REMOTE_DIR/cam_frame.png" "$ROOT/outputs/cam_frame.png"

exec "$PY" "$ROOT/src/verify_camera.py" ${VERIFY_ARGS[@]+"${VERIFY_ARGS[@]}"}
