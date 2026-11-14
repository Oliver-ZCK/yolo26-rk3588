#!/bin/bash
# 把官方 demo 版摄像头推理程序推到 RK3588。
#
# 用法: deploy_demo.sh [--clean]
# 环境变量: BOARD=root@192.168.1.10  REMOTE=/home/zck/yolo26  CLEAN=/home/zck
#
# 传到板子上的三个文件：
#   $REMOTE/yolo26_cam                   推理程序
#   $REMOTE/model/coco_80_labels_list.txt  类别名（官方 post_process 用相对路径读它）
#   $REMOTE/yolo26n-rk3588.rknn          FP16 模型
#
# 依赖的 librknnrt / libopencv_* / librga / libturbojpeg 板子的 /usr/lib 里都有，
# 不需要一起推。
#
# 默认只覆盖同名文件，不动板子上其它东西。
# --clean 才会先清空 $CLEAN 目录——那是破坏性操作，用之前先确认板子上没有要留的东西。
set -euo pipefail

BOARD="${BOARD:-root@192.168.1.10}"
REMOTE="${REMOTE:-/home/zck/yolo26}"
CLEAN="${CLEAN:-/home/zck}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/board/build/yolo26_cam"
LABELS="$ROOT/board/demo/model/coco_80_labels_list.txt"
MODEL="$ROOT/models/yolo26n_fp16_rknn_model/yolo26n-rk3588.rknn"

CLEAN_FIRST=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN_FIRST=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

[[ -x "$BIN" ]] || { echo "还没编译，先跑 board/build.sh"; exit 1; }
[[ -f "$LABELS" ]] || { echo "找不到类别名文件: $LABELS"; exit 1; }
[[ -f "$MODEL" ]] || { echo "找不到模型: $MODEL"; exit 1; }

if [[ "$CLEAN_FIRST" == 1 ]]; then
  echo ">> 清空 $BOARD:$CLEAN"
  ssh "$BOARD" "find '$CLEAN' -mindepth 1 -maxdepth 1 -exec rm -rf {} \;"
fi

echo ">> 上传程序、类别名、模型到 $BOARD:$REMOTE"
ssh "$BOARD" "mkdir -p '$REMOTE/model'"
scp -q "$BIN" "$BOARD:$REMOTE/yolo26_cam"
scp -q "$LABELS" "$BOARD:$REMOTE/model/coco_80_labels_list.txt"
scp -q "$MODEL" "$BOARD:$REMOTE/$(basename "$MODEL")"
ssh "$BOARD" "chmod +x '$REMOTE/yolo26_cam'"

echo ">> 板上的内容"
ssh "$BOARD" "ls -l '$REMOTE' '$REMOTE/model'"

cat <<EOF

板子上运行（要在板子本地终端跑，SSH 会话里没有 X display，imshow 起不来）：

  cd $REMOTE
  ./yolo26_cam $(basename "$MODEL") 71

  参数：<模型> [相机节点]   相机节点默认 44，按 'q' 退出

EOF
