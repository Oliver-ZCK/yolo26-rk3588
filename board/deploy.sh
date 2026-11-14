#!/bin/bash
# 把模型和板端程序推到 RK3588，跑一次推理，把结果拉回 PC 侧比对。
#
# 用法: deploy.sh <model.rknn> [--int8] [--bench N] [--detail]
# 环境变量: BOARD=root@192.168.1.10  REMOTE=/home/zck/yolo26
#
# 首次连接没有配免密时 ssh/scp 会提示输密码；配好免密后整条链路可以无人值守。
set -euo pipefail

BOARD="${BOARD:-root@192.168.1.10}"
REMOTE="${REMOTE:-/home/zck/yolo26}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/board/build/yolo26"
IMG="$ROOT/assets/bus.jpg"
PY="${PY:-/home/dr-wei-arm/anaconda3/envs/py3.11-tk2-2.3.2/bin/python}"

MODEL="${1:?用法: deploy.sh <model.rknn> [--int8] [--bench N] [--detail]}"
shift

[[ -f "$MODEL" ]] || { echo "找不到模型: $MODEL"; exit 1; }
[[ -x "$BIN" ]] || { echo "还没编译，先跑 board/build.sh"; exit 1; }
[[ -f "$IMG" ]] || { echo "找不到图片: $IMG"; exit 1; }

MODEL="$(readlink -f "$MODEL")"
TAG="$(basename "$MODEL" .rknn)"
REMOTE_DIR="$REMOTE/$TAG"

# -- 之前的参数给板端二进制，之后的给 PC 侧校验（如 --stage board_int8 --tol-px 20）
BOARD_ARGS=() VERIFY_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; VERIFY_ARGS=("$@"); break; fi
  BOARD_ARGS+=("$1"); shift
done

echo ">> 上传到 $BOARD:$REMOTE_DIR"
ssh "$BOARD" mkdir -p "$REMOTE_DIR"
scp -q "$BIN" "$IMG" "$MODEL" "$BOARD:$REMOTE_DIR/"

echo ">> 板上推理"
ssh "$BOARD" "cd '$REMOTE_DIR' && ./yolo26 $(basename "$MODEL") $(basename "$IMG") ${BOARD_ARGS[*]-}"

echo ">> 拉回结果并比对"
scp -q "$BOARD:$REMOTE_DIR/board_out.json" "$ROOT/outputs/board_raw.json"
exec "$PY" "$ROOT/src/verify_board.py" --model "$MODEL" --image "$IMG" ${VERIFY_ARGS[@]+"${VERIFY_ARGS[@]}"}
