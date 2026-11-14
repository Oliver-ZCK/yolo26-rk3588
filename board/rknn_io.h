// RKNN runtime 的数据搬运：张量属性查询、输入喂法、输出取回。
//
// 单图模式（main.cc）和摄像头模式（camera.cc）共用这一份，
// 保证两条链路的喂入格式和取回方式完全一致——
// 精度基准要能说明"摄像头那一帧"和"单图那一帧"走的是同一条推理链路。

#pragma once

#include <rknn_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace board {

inline void Check(int ret, const char* what) {
  if (ret != 0) {
    std::fprintf(stderr, "%s 失败，ret=%d\n", what, ret);
    std::exit(1);
  }
}

inline std::string BaseName(const std::string& path) {
  const size_t pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

inline std::vector<uint8_t> ReadFile(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "打不开文件: %s\n", path.c_str());
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(n);
  if (std::fread(buf.data(), 1, n, f) != static_cast<size_t>(n)) {
    std::fprintf(stderr, "读取不完整: %s\n", path.c_str());
    std::exit(1);
  }
  std::fclose(f);
  return buf;
}

inline const char* FmtName(rknn_tensor_format f) {
  switch (f) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    case RKNN_TENSOR_NC1HWC2: return "NC1HWC2";
    default: return "UNDEFINED";
  }
}

inline const char* TypeName(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_FLOAT32: return "FLOAT32";
    case RKNN_TENSOR_FLOAT16: return "FLOAT16";
    case RKNN_TENSOR_INT8: return "INT8";
    case RKNN_TENSOR_UINT8: return "UINT8";
    case RKNN_TENSOR_INT16: return "INT16";
    default: return "OTHER";
  }
}

// 打印一个张量的全部关键属性（官方 model-zoo dump_tensor_attr 的形式）。
// 逻辑属性（RKNN_QUERY_INPUT_ATTR）描述模型怎么用这个张量，
// native 属性（RKNN_QUERY_NATIVE_INPUT_ATTR）描述 NPU 上实际是什么布局，
// 两者不一定相同——INT8 反量化和零拷贝都以 native 属性为准。
inline void DumpTensorAttr(const char* tag, const rknn_tensor_attr& a) {
  std::printf("  %s\n", tag);
  std::printf("    name=%-24s index=%u n_dims=%u n_elems=%u size=%u stride=%u\n", a.name, a.index,
              a.n_dims, a.n_elems, a.size, a.size_with_stride);
  std::printf("    dims=[");
  for (uint32_t i = 0; i < a.n_dims; ++i) std::printf("%s%u", i ? "," : "", a.dims[i]);
  std::printf("]\n");
  std::printf("    fmt=%s type=%s qnt=%d zp=%d scale=%.6f pass_through=%u\n", FmtName(a.fmt),
              TypeName(a.type), (int)a.qnt_type, a.zp, a.scale, a.pass_through);
}

// 喂给 runtime 的数据格式
enum class Feed {
  U8,    // uint8 NHWC，runtime 负责转成模型原生格式
  Fp16,  // FP16 NHWC，与模型原生格式一致，runtime 只做拷贝
};

// 零拷贝输入：官方 model-zoo yolo11_zero_copy.cc 的做法。
//
// rknn_inputs_set 会在 CPU 上把数据重排成 NPU 硬件要的布局，实测 40 ms，
// 比 NPU 图内算子（40.6 ms）还贵。零拷贝把输入 buffer 映射到 DDR，
// 数据自己写进去，runtime 不再复制也不再重排。
class ZeroCopyInput {
 public:
  void Init(rknn_context ctx, const rknn_tensor_attr& native_attr) {
    ctx_ = ctx;
    mem_ = rknn_create_mem(ctx, native_attr.size_with_stride);
    if (mem_ == nullptr) {
      std::fprintf(stderr, "零拷贝输入内存申请失败\n");
      std::exit(1);
    }
    attr_ = native_attr;
    const int ret = rknn_set_io_mem(ctx, mem_, &attr_);
    if (ret != RKNN_SUCC) {
      std::fprintf(stderr, "rknn_set_io_mem 失败，ret=%d\n", ret);
      std::exit(1);
    }
  }

  // blob 是 native 格式的连续数据（FP16 NHWC）
  void Write(const cv::Mat& blob) {
    const int h = attr_.dims[1], w = attr_.dims[2];  // NHWC: [N, H, W, C]
    cv::Mat dst(h, w, CV_16FC3, mem_->virt_addr);
    blob.copyTo(dst);
  }

  ~ZeroCopyInput() {
    if (mem_ != nullptr) rknn_destroy_mem(ctx_, mem_);
  }

 private:
  rknn_context ctx_ = 0;
  rknn_tensor_mem* mem_ = nullptr;
  rknn_tensor_attr attr_{};
};

// 输入通道：普通路径走 rknn_inputs_set，零拷贝路径直接写进映射好的 DDR。
struct InputSink {
  rknn_context ctx = 0;
  ZeroCopyInput* zero_copy = nullptr;  // 非空即走零拷贝
  Feed feed = Feed::U8;

  void Submit(const cv::Mat& blob) {
    if (zero_copy != nullptr) {
      zero_copy->Write(blob);
      return;
    }
    rknn_input in{};
    in.index = 0;
    if (feed == Feed::U8) {
      in.type = RKNN_TENSOR_UINT8;  // 归一化参数已烧进模型，运行时收 uint8 原图
      in.fmt = RKNN_TENSOR_NHWC;
    } else {
      in.type = RKNN_TENSOR_FLOAT16;
      in.fmt = RKNN_TENSOR_NHWC;
    }
    in.size = static_cast<uint32_t>(blob.total() * blob.elemSize());
    in.buf = blob.data;
    Check(rknn_inputs_set(ctx, 1, &in), "rknn_inputs_set");
  }
};

// 把 uint8 RGB 转成模型原生的 FP16。
//
// 喂 uint8 时 runtime 要在 CPU 上把 123 万个像素逐个转成 FP16，实测这步要
// 33 ms，比 NPU 算子本身还贵。自己转好再喂，runtime 只剩一次拷贝。
// 原生布局是 NHWC（RKNN_QUERY_NATIVE_INPUT_ATTR 查到的），所以只做逐元素
// 类型转换，不转置：转置会把连续写变成大步长写，cache 命中率崩掉。
// 归一化参数仍由模型内部的预处理算子负责，这里保持 0~255 原值。
inline cv::Mat ToFp16(const cv::Mat& rgb_u8) {
  cv::Mat fp16;
  rgb_u8.convertTo(fp16, CV_16F);
  return fp16;
}

// 读输出：want_float=0 拿模型的原生数据，自己转成 float32。
// runtime 内置的 float 转换实测 9 ms，OpenCV 向量化转只要 1~2 ms。
// INT8 走 convertTo 的 alpha/beta，一次遍历就把 (q - zp) * scale 算完。
inline cv::Mat ToFloat32(const void* buf, size_t n_elems, rknn_tensor_type type, int32_t zp,
                         float scale) {
  if (type == RKNN_TENSOR_FLOAT16) {
    cv::Mat raw(1, static_cast<int>(n_elems), CV_16F, const_cast<void*>(buf));
    cv::Mat f32;
    raw.convertTo(f32, CV_32F);
    return f32;
  }
  cv::Mat raw(1, static_cast<int>(n_elems), CV_8S, const_cast<void*>(buf));
  cv::Mat f32;
  raw.convertTo(f32, CV_32F, scale, -static_cast<float>(zp) * scale);
  return f32;
}

}  // namespace board
