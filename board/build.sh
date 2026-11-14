#!/bin/bash
# 交叉编译板端推理程序。
#
# 用正点原子 RK3588 SDK 自带的 buildroot 工具链，而不是宿主机 g++ 加 --sysroot：
# 工具链包装器已经带上了正确的 libc / libstdc++ 版本、动态链接器和 C++ ABI 选项，
# 编译出来的二进制才能在板子的 rootfs 上直接跑。
#
# 产出两个程序：
#   build/yolo26     本工程自己写的单图推理程序
#   build/yolo26_cam 官方 rknn_model_zoo demo 树（只替换了 postprocess.cc）
set -euo pipefail

SDK="${SDK:-/home/dr-wei-arm/atk_dlrk3588_linux6.1/buildroot/output/alientek_rk3588}"
SYSROOT="$SDK/host/aarch64-buildroot-linux-gnu/sysroot"
CXX="$SDK/host/bin/aarch64-buildroot-linux-gnu-g++"
CC="$SDK/host/bin/aarch64-buildroot-linux-gnu-gcc"

cd "$(dirname "$0")"
mkdir -p build

INC=(-I"$SYSROOT/usr/include/rknn" -I"$SYSROOT/usr/include/opencv4" -Idemo/utils)
# rpath 指向板上的 /usr/lib，librknnrt.so 在那里
# highgui 是摄像头实时显示用的（imshow / waitKey）
LIBS=(-L"$SYSROOT/usr/lib" -lrknnrt -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
      -lopencv_videoio -lopencv_highgui -Wl,-rpath,/usr/lib)

# utils/ 是 rknn_model_zoo 的点阵字体与绘图实现（Apache-2.0），
# 用来替代不可用的 cv::putText（buildroot OpenCV 未编 freetype）
"$CC" -O2 -c demo/utils/image_drawing.c -o build/image_drawing.o "${INC[@]}"

"$CXX" -O2 -std=c++17 -Wall -Wextra "${INC[@]}" -o build/yolo26 main.cc \
  "${LIBS[@]}" build/image_drawing.o

"$CXX" -O2 -std=c++17 -Wall -Wextra "${INC[@]}" -o build/camera camera.cc \
  "${LIBS[@]}" build/image_drawing.o

# ---------------------------------------------------------------------------
# 官方 demo 树：除了 postprocess.cc，其余文件与 rknn_model_zoo 完全一致。
# image_utils.c 依赖 RGA(im2d) 和 stb/turbojpeg，官方 CMake 里的这几个库
# 在正点原子的 buildroot sysroot 里都有。
# ---------------------------------------------------------------------------
DEMO_INC=(-I"$SYSROOT/usr/include/rknn" -I"$SYSROOT/usr/include/opencv4"
          -Idemo -Idemo/utils -Idemo/3rdparty/librga/include -Idemo/3rdparty/stb_image)
DEMO_LIBS=(-L"$SYSROOT/usr/lib" -lrknnrt -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
           -lopencv_videoio -lopencv_highgui -lrga -lturbojpeg -lpthread
           -Wl,-rpath,/usr/lib)

"$CC" -O2 -Wno-write-strings -Wno-return-type -c demo/utils/image_drawing.c \
  -o build/demo_image_drawing.o "${DEMO_INC[@]}"
"$CC" -O2 -Wno-write-strings -Wno-return-type -c demo/utils/image_utils.c \
  -o build/demo_image_utils.o "${DEMO_INC[@]}"
"$CC" -O2 -Wno-write-strings -Wno-return-type -c demo/utils/file_utils.c \
  -o build/demo_file_utils.o "${DEMO_INC[@]}"

"$CXX" -O2 -std=c++11 -Wno-write-strings -Wno-return-type "${DEMO_INC[@]}" \
  -o build/yolo26_cam demo/main.cc demo/postprocess.cc demo/rknpu2/yolo11.cc \
  "${DEMO_LIBS[@]}" build/demo_image_drawing.o build/demo_image_utils.o build/demo_file_utils.o

file build/yolo26 build/camera build/yolo26_cam | sed 's/, BuildID[^,]*//'
ls -la build/yolo26 build/camera build/yolo26_cam
