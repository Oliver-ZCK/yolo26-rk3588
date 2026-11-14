// RK3588 板端推理：加载 .rknn 跑单图检测，输出统一 schema 的 JSON。
//
// 板子只负责推理和落 JSON，判定逻辑放在 PC 侧的 src/verify_board.py——
// 阈值、容差、比对规则都可以随时改，不必重新交叉编译。
//
// 覆盖模拟器验不到的部分：.rknn 二进制本身、FP16 在 NPU 上的量化误差、
// 板端输入预处理与 PC 基准的几何一致性。
//
// INT8 坐标约定：ultralytics 仅在 format=rknn 且 quantize=8 时插入
// _NormalizeCoords，导出时把框坐标归一化到 0~1。不乘回输入尺寸的话框会
// 缩到 1/640 且不报任何错。

#include <rknn_api.h>

#include <dirent.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "labels.h"
#include "postprocess.h"
#include "result_json.h"
#include "rknn_io.h"

namespace {

using board::BaseName;
using board::Check;
using board::Det;
using board::DumpTensorAttr;
using board::Feed;
using board::InputSink;
using board::ReadFile;
using board::ToFloat32;
using board::ToFp16;
using board::TypeName;
using board::ZeroCopyInput;

constexpr const char* kRule = "--------------------------------------------------------------";

double Ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Perf {
  double wall_ms = 0.0;  // 一帧全流程的平均
  double set_ms = 0.0;   // rknn_inputs_set 的平均
  double run_ms = 0.0;   // rknn_run 的平均
  double get_ms = 0.0;   // rknn_outputs_get + release 的平均
  double npu_ms = 0.0;   // rknn_query PERF_RUN 的平均
  double wall_min_ms = 0.0;
  double wall_max_ms = 0.0;
  int loops = 0;
};

// 逐次计时并报最值：平均值会被首帧的内存池预热拉偏，
// 只看均值会分不清"稳定 68 ms"和"20 帧里有一帧 500 ms"。
//
// 每轮都取回并释放输出：输出缓冲不释放，runtime 就无法复用输入缓冲，
// 下一次 rknn_inputs_set 会一直等到上一轮彻底结束，整条循环退化成
// 2 倍 NPU 时间的串行执行。
Perf Bench(InputSink& sink, const cv::Mat& blob, int loops, int warmup, bool consume,
           int want_float) {
  for (int i = 0; i < warmup; ++i) {
    sink.Submit(blob);
    Check(rknn_run(sink.ctx, nullptr), "rknn_run");
  }

  Perf perf;
  perf.loops = loops;
  std::vector<double> wall, set, run, get;
  wall.reserve(loops);
  set.reserve(loops);
  run.reserve(loops);
  get.reserve(loops);

  for (int i = 0; i < loops; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    sink.Submit(blob);
    const auto t1 = std::chrono::steady_clock::now();
    Check(rknn_run(sink.ctx, nullptr), "rknn_run");
    const auto t2 = std::chrono::steady_clock::now();

    if (consume) {
      rknn_output out{};
      out.index = 0;
      out.want_float = want_float;
      out.is_prealloc = 0;
      Check(rknn_outputs_get(sink.ctx, 1, &out, nullptr), "rknn_outputs_get");
      rknn_outputs_release(sink.ctx, 1, &out);
    }
    const auto t3 = std::chrono::steady_clock::now();

    set.push_back(Ms(t0, t1));
    run.push_back(Ms(t1, t2));
    get.push_back(Ms(t2, t3));
    wall.push_back(Ms(t0, t3));

    rknn_perf_run r{};
    if (rknn_query(sink.ctx, RKNN_QUERY_PERF_RUN, &r, sizeof(r)) == 0)
      perf.npu_ms += r.run_duration / 1000.0;
  }

  auto mean = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
  };
  perf.set_ms = mean(set);
  perf.run_ms = mean(run);
  perf.get_ms = mean(get);
  perf.wall_ms = mean(wall);
  perf.npu_ms /= loops;

  std::vector<double> sorted = wall;
  std::sort(sorted.begin(), sorted.end());
  perf.wall_min_ms = sorted.front();
  perf.wall_max_ms = sorted.back();
  return perf;
}

std::vector<std::string> ListJpgs(const std::string& dir) {
  std::vector<std::string> files;
  DIR* d = opendir(dir.c_str());
  if (d == nullptr) return files;
  while (dirent* ent = readdir(d)) {
    const std::string name = ent->d_name;
    const size_t dot = name.find_last_of('.');
    if (dot == std::string::npos) continue;
    const std::string ext = name.substr(dot + 1);
    if (ext == "jpg" || ext == "JPG" || ext == "jpeg" || ext == "png")
      files.push_back(dir + "/" + name);
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  return files;
}

// mAP 批量评估：目录内全部图片过一遍推理，检出按 COCO result 格式落一个
// json，PC 侧 pycocotools 直接消费。协议与 src/eval_map.py 对齐（square
// letterbox 640 / conf=0.001 / NMS iou=0.45 / max_det=300）。模拟器不能
// load_rknn，hybrid 定点模型的 mAP 只能在真机上量。
struct EvalCtx {
  rknn_context ctx;
  InputSink* sink;
  rknn_tensor_attr out_attr;
  int na, nc;
  bool quant_out;
  std::string model_path;
  size_t model_bytes;
};

int RunEvalDir(const EvalCtx& e, const std::string& dir, const std::string& out_path,
               float conf) {
  const std::vector<std::string> files = ListJpgs(dir);
  if (files.empty()) {
    std::fprintf(stderr, "目录里没有图片: %s\n", dir.c_str());
    return 1;
  }

  std::FILE* f = std::fopen(out_path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "写不了输出文件: %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(f,
               "{\"model\":\"%s\",\"model_bytes\":%zu,\"images\":%zu,"
               "\"protocol\":{\"imgsz\":640,\"conf\":%.4f,\"iou_nms\":0.45,\"max_det\":300},"
               "\"detections\":[",
               e.model_path.c_str(), e.model_bytes, files.size(), conf);

  double pre_ms = 0, infer_ms = 0, post_ms = 0;
  size_t n_dets = 0;
  const size_t elems = static_cast<size_t>(4 + e.nc) * e.na;
  for (size_t i = 0; i < files.size(); ++i) {
    cv::Mat img = cv::imread(files[i]);
    if (img.empty()) {
      std::fprintf(stderr, "读图失败，跳过: %s\n", files[i].c_str());
      continue;
    }
    // COCO val2017 文件名是前导零数字，atoll 正好还原官方 image_id
    const long long image_id = std::atoll(BaseName(files[i]).c_str());

    const auto t0 = std::chrono::steady_clock::now();
    cv::Mat canvas, rgb;
    const board::LetterboxInfo info = board::Letterbox(img, canvas);
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    cv::Mat blob = e.sink->feed == Feed::Fp16 ? ToFp16(rgb) : rgb;
    const auto t1 = std::chrono::steady_clock::now();
    e.sink->Submit(blob);
    Check(rknn_run(e.ctx, nullptr), "rknn_run");
    rknn_output out{};
    out.index = 0;
    out.want_float = 0;
    out.is_prealloc = 0;
    Check(rknn_outputs_get(e.ctx, 1, &out, nullptr), "rknn_outputs_get");
    const auto t2 = std::chrono::steady_clock::now();
    const cv::Mat f32 =
        ToFloat32(out.buf, elems, e.out_attr.type, e.out_attr.zp, e.out_attr.scale);
    rknn_outputs_release(e.ctx, 1, &out);
    const std::vector<Det> dets = board::Decode(f32.ptr<float>(), e.na, e.quant_out, info,
                                                img.rows, img.cols, conf, 300);
    const auto t3 = std::chrono::steady_clock::now();

    for (const auto& d : dets) {
      std::fprintf(f,
                   "%s{\"image_id\":%lld,\"category_id\":%d,"
                   "\"bbox\":[%.2f,%.2f,%.2f,%.2f],\"score\":%.4f}",
                   n_dets ? "," : "", image_id, d.cls, d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1,
                   d.score);
      ++n_dets;
    }
    pre_ms += Ms(t0, t1);
    infer_ms += Ms(t1, t2);
    post_ms += Ms(t2, t3);

    if ((i + 1) % 20 == 0 || i + 1 == files.size())
      std::printf("  %zu/%zu  平均 pre %.1f / infer %.1f ms\n", i + 1, files.size(),
                  pre_ms / (i + 1), infer_ms / (i + 1));
  }
  std::fprintf(f, "]}");
  std::fclose(f);

  std::printf("批量评估完成：%zu 张，检出 %zu 条\n", files.size(), n_dets);
  std::printf("  平均 pre %.2f ms / infer %.2f ms / post %.2f ms\n", pre_ms / files.size(),
              infer_ms / files.size(), post_ms / files.size());
  std::printf("  -> %s\n", out_path.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "用法: %s <model.rknn> <image.jpg> [--int8] [--bench N] [--detail] [--out path]\n"
                 "            [--zero-copy] [--input-fp16] [--input-u8] [--keep-output]\n"
                 "      %s <model.rknn> --dir <image_dir> --eval-out board_eval.json "
                 "[--eval-conf 0.001]   # mAP 批量评估\n",
                 argv[0], argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string img_path = argv[2];
  std::string out_path = "board_out.json";
  std::string eval_dir, eval_out = "board_eval.json";
  float eval_conf = 0.001f;
  bool int8 = false, detail = false, keep_output = false;
  int bench_loops = 0;
  bool fp16_input = false, force_u8 = false, zero_copy = false;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--int8")) int8 = true;
    else if (!std::strcmp(argv[i], "--detail")) detail = true;
    else if (!std::strcmp(argv[i], "--keep-output")) keep_output = true;
    else if (!std::strcmp(argv[i], "--input-fp16")) fp16_input = true;
    else if (!std::strcmp(argv[i], "--input-u8")) force_u8 = true;
    else if (!std::strcmp(argv[i], "--zero-copy")) zero_copy = true;
    else if (!std::strcmp(argv[i], "--dir") && i + 1 < argc) eval_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--eval-out") && i + 1 < argc) eval_out = argv[++i];
    else if (!std::strcmp(argv[i], "--eval-conf") && i + 1 < argc)
      eval_conf = static_cast<float>(std::atof(argv[++i]));
    else if (!std::strcmp(argv[i], "--bench") && i + 1 < argc) bench_loops = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--out")) out_path = argv[++i];
  }

  std::vector<uint8_t> model = ReadFile(model_path);  // rknn_init 的 model 形参是非 const void*
  rknn_context ctx;
  Check(rknn_init(&ctx, model.data(), static_cast<uint32_t>(model.size()),
                  detail ? RKNN_FLAG_COLLECT_PERF_MASK : 0, nullptr),
        "rknn_init（报模型版本不兼容说明 .rknn 的工具链比板子 librknnrt 新）");

  rknn_sdk_version ver{};
  rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));

  rknn_input_output_num io{};
  Check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "查询输入输出数量");
  if (io.n_input != 1 || io.n_output != 1) {
    std::fprintf(stderr, "模型有 %u 输入 / %u 输出，本程序只处理单输入单输出\n", io.n_input,
                 io.n_output);
    return 1;
  }

  rknn_tensor_attr in_attr{};
  in_attr.index = 0;
  Check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)), "查询输入属性");

  // native 属性描述 NPU 上实际落地的张量布局，和逻辑属性不一定相同：
  // 输入能不能绕开 runtime 的 host 侧格式转换、INT8 输出该按什么布局反量化，
  // 都以 native 为准。
  rknn_tensor_attr in_native{};
  in_native.index = 0;
  const bool has_in_native =
      rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_native, sizeof(in_native)) == 0;

  const int in_nd = static_cast<int>(in_attr.n_dims);
  const int H = in_attr.fmt == RKNN_TENSOR_NHWC ? in_attr.dims[in_nd - 3] : in_attr.dims[in_nd - 2];
  const int W = in_attr.fmt == RKNN_TENSOR_NHWC ? in_attr.dims[in_nd - 2] : in_attr.dims[in_nd - 1];
  if (in_nd < 3 || H != board::kImgsz || W != board::kImgsz) {
    std::fprintf(stderr, "输入尺寸不是 %dx%d，导出时需对齐\n", board::kImgsz, board::kImgsz);
    return 1;
  }

  rknn_tensor_attr out_attr{};
  out_attr.index = 0;
  Check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr)), "查询输出属性");
  rknn_tensor_attr out_native{};
  out_native.index = 0;
  const bool has_out_native =
      rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &out_native, sizeof(out_native)) == 0;

  const int out_nd = static_cast<int>(out_attr.n_dims);
  const int na = out_attr.dims[out_nd - 1];
  const int nc = out_attr.dims[out_nd - 2] - 4;
  if (nc != board::kNumClasses || na <= 0) {
    std::fprintf(stderr, "输出形状不是 [1, %d, anchors]，实际 [%d, %d]\n",
                 4 + board::kNumClasses, out_attr.dims[out_nd - 2], na);
    return 1;
  }

  // native 输入本身就是 FP16/NHWC 时，直接喂原生格式：runtime 只剩一次拷贝，
  // 不用在 CPU 上把 123 万个像素逐个转成 FP16（实测 33 ms）。
  // 布局必须逐字节对齐 native，转置过反而多一道重排。
  const bool native_fp16_nhwc = has_in_native && in_native.type == RKNN_TENSOR_FLOAT16 &&
                                in_native.fmt == RKNN_TENSOR_NHWC && in_native.n_dims == 4;
  const bool feed_fp16 = fp16_input || (native_fp16_nhwc && !force_u8);

  // 量化输出：runtime 给的是原始 int8，ultralytics 导出时还插了 _NormalizeCoords
  // 把框坐标归一化到 0~1。两者都按张量属性判断，--int8 只是人工兜底。
  const bool quant_out = out_attr.type == RKNN_TENSOR_INT8;
  if (int8 && !quant_out) {
    std::fprintf(stderr, "--int8 与模型输出类型 %s 不符\n", TypeName(out_attr.type));
    return 1;
  }

  char input_desc[128], output_desc[128];
  std::snprintf(input_desc, sizeof(input_desc), "[1,%d,%d,3] %s", H, W,
                feed_fp16 ? "NHWC FLOAT16" : "NHWC UINT8");
  std::snprintf(output_desc, sizeof(output_desc), "[1,%d,%d]  %d 类 x %d 锚点", 4 + nc, na, nc, na);

  std::printf("%s\n", kRule);
  // 按 张量属性 显示真实精度，--int8 只是人工兜底校验
  std::printf("%s  %s  (%.2f MB)\n", quant_out ? "INT8" : "FP16", model_path.c_str(),
              model.size() / 1e6);
  std::printf("%s\n", kRule);
  std::printf("  runtime API %s\n  NPU driver  %s\n", ver.api_version, ver.drv_version);
  DumpTensorAttr("输入（逻辑）", in_attr);
  if (has_in_native) DumpTensorAttr("输入（native）", in_native);
  DumpTensorAttr("输出（逻辑）", out_attr);
  if (has_out_native) DumpTensorAttr("输出（native）", out_native);
  std::printf("  输入 %s\n  输出 %s\n", input_desc, output_desc);
  if (native_fp16_nhwc && !fp16_input)
    std::printf("  (native 输入就是 FP16/NHWC，已自动改用原生格式喂入；--input-u8 可退回)\n");

  cv::Mat img, canvas, rgb, blob;
  board::LetterboxInfo info{};
  const Feed feed = feed_fp16 ? Feed::Fp16 : Feed::U8;
  if (eval_dir.empty()) {  // --dir 批量模式逐图处理，跳过基准图的读入与预处理
    img = cv::imread(img_path);
    if (img.empty()) {
      std::fprintf(stderr, "读图失败: %s\n", img_path.c_str());
      return 1;
    }
    info = board::Letterbox(img, canvas);
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);  // RKNN 输入按 RGB 排布
    blob = feed_fp16 ? ToFp16(rgb) : rgb;
  }

  // 零拷贝：输入 buffer 映射到 DDR，数据自己写进去，runtime 不做格式重排。
  // 只支持 FP16+NHWC，因为那才是 native 布局；喂别的格式 runtime 照样要重排。
  ZeroCopyInput zc;
  if (zero_copy) {
    if (!has_in_native) {
      std::fprintf(stderr, "查不到 native 输入属性，零拷贝不可用\n");
      return 1;
    }
    if (!feed_fp16) {
      std::fprintf(stderr, "零拷贝要求按 native 格式喂入，去掉 --input-u8\n");
      return 1;
    }
    zc.Init(ctx, in_native);
  }
  InputSink sink{ctx, zero_copy ? &zc : nullptr, feed};

  if (!eval_dir.empty()) {
    const EvalCtx e{ctx, &sink, out_attr, na, nc, quant_out, model_path, model.size()};
    return RunEvalDir(e, eval_dir, eval_out, eval_conf);
  }

  // 输出统一走 want_float=0：拿模型原生数据（FP16 或 INT8），自己转 float32，
  // 省掉 runtime 内置的那次 float 转换（实测 9 ms）。
  constexpr int want_float = 0;

  // 预处理耗时：letterbox + 通道转换。bench 的 set/run/get 不含这两步，
  // 端到端延迟要把它们算进来。
  double pre_ms = 0.0;
  {
    const int n = 20;
    cv::Mat pre_blob;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
      cv::Mat canvas_i, rgb_i;
      board::Letterbox(img, canvas_i);
      cv::cvtColor(canvas_i, rgb_i, cv::COLOR_BGR2RGB);
      pre_blob = feed_fp16 ? ToFp16(rgb_i) : rgb_i;
    }
    pre_ms = Ms(t0, std::chrono::steady_clock::now()) / n;
    // 预处理结果本身不用，但要让编译器知道它有副作用，否则整段循环会被优化掉
    static volatile size_t pre_guard = 0;
    pre_guard ^= pre_blob.total();
  }

  Perf perf;
  if (bench_loops > 0) {
    perf = Bench(sink, blob, bench_loops, 5, !keep_output, want_float);
    std::printf("\n性能  (%d 次，5 次预热，%s)\n", bench_loops,
                keep_output ? "不取回输出" : "每轮取回并释放输出");
    std::printf("  NPU 算子耗时 %7.3f ms   %7.1f FPS   (rknn_query PERF_RUN)\n", perf.npu_ms,
                perf.npu_ms > 0 ? 1000.0 / perf.npu_ms : 0.0);
    std::printf("  %-14s %5.3f ms\n", zero_copy ? "写零拷贝内存" : "rknn_inputs_set", perf.set_ms);
    std::printf("  rknn_run       %5.3f ms\n", perf.run_ms);
    std::printf("  outputs_get    %5.3f ms   (含 release)\n", perf.get_ms);
    std::printf("  预处理          %5.3f ms   (letterbox + 通道转换，不在上面的合计里)\n",
                pre_ms);
    std::printf("  推理链路合计   %7.3f ms   %7.1f FPS   最快 %.3f / 最慢 %.3f\n", perf.wall_ms,
                1000.0 / perf.wall_ms, perf.wall_min_ms, perf.wall_max_ms);
    std::printf("  端到端         %7.3f ms   %7.1f FPS   (推理链路 + 预处理)\n",
                perf.wall_ms + pre_ms, 1000.0 / (perf.wall_ms + pre_ms));
  }

  sink.Submit(blob);
  Check(rknn_run(ctx, nullptr), "rknn_run");

  rknn_output out{};
  out.index = 0;
  out.want_float = want_float;
  out.is_prealloc = 0;
  Check(rknn_outputs_get(ctx, 1, &out, nullptr), "rknn_outputs_get");
  const size_t elems = static_cast<size_t>(4 + nc) * na;
  const size_t expect_bytes =
      elems * (out_attr.type == RKNN_TENSOR_FLOAT16 ? sizeof(uint16_t) : sizeof(int8_t));
  if (out.size != expect_bytes) {
    std::fprintf(stderr, "输出字节数 %u 与形状不符（期望 %zu）\n", out.size, expect_bytes);
    return 1;
  }

  // rknn_outputs_get 给的是逻辑布局 [1, 84, 8400]，和 PC 侧 one2many 输出一致
  const cv::Mat f32 = ToFloat32(out.buf, elems, out_attr.type, out_attr.zp, out_attr.scale);
  const float* p = f32.ptr<float>();
  const std::vector<Det> dets =
      board::Decode(p, na, quant_out, info, img.rows, img.cols);

  float box_max = 0.f, score_max = 0.f;
  for (int c = 0; c < 4; ++c)
    for (int a = 0; a < na; ++a) box_max = std::max(box_max, p[c * na + a]);
  for (int c = 4; c < 4 + nc; ++c)
    for (int a = 0; a < na; ++a) score_max = std::max(score_max, p[c * na + a]);
  std::printf("\n  框坐标通道最大值 %10.4f   类别分数最大值 %.4f\n", box_max, score_max);
  if (quant_out && box_max < 10.f)
    std::printf("  ! 框坐标量级异常：INT8 模型坐标被归一化到 0~1\n");

  std::printf("\n检测 %zu 个\n", dets.size());
  std::printf("  %-8s %-9s %8s %8s %8s %8s\n", "class_id", "score", "x1", "y1", "x2", "y2");
  for (const auto& d : dets)
    std::printf("  %-8d %-9.4f %8.2f %8.2f %8.2f %8.2f\n", d.cls, d.score, d.x1, d.y1, d.x2, d.y2);

  std::string perf_detail;
  if (detail) {
    rknn_perf_detail pd{};
    if (rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof(pd)) == 0 && pd.perf_data)
      perf_detail = pd.perf_data;
  }

  board::ResultJson result;
  result.model = BaseName(model_path);
  result.model_bytes = model.size();
  result.image = BaseName(img_path);
  result.image_width = img.cols;
  result.image_height = img.rows;
  result.detections = dets;
  result.precision = quant_out ? "int8" : "fp16";
  result.runtime_api = ver.api_version;
  result.npu_driver = ver.drv_version;
  result.input_desc = input_desc;
  result.output_desc = output_desc;
  result.loops = perf.loops;
  result.npu_ms = perf.npu_ms;
  result.wall_ms = perf.wall_ms;
  result.set_ms = perf.set_ms;
  result.run_ms = perf.run_ms;
  result.get_ms = perf.get_ms;
  result.wall_min_ms = perf.wall_min_ms;
  result.wall_max_ms = perf.wall_max_ms;
  result.perf_detail = perf_detail;
  board::WriteResultJson(out_path, result);
  std::printf("  -> %s\n", out_path.c_str());

  rknn_outputs_release(ctx, 1, &out);
  rknn_destroy(ctx);
  return 0;
}
