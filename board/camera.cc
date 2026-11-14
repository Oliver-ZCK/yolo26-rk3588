// RK3588 摄像头实时推理：MIPI CSI -> NPU -> LCD。
//
// 摄像头部分参考官方 rknn_model_zoo 的 yolo11 相机 demo
// （/home/dr-wei-arm/AI/yolo11/cpp/main.cc），沿用它的两点：
//   1. 画面旋转 90 度（摄像头物理竖装，不转的话框和画面对不上）
//   2. 用 utils/ 下的点阵字体画框——buildroot 的 OpenCV 没编 freetype，
//      cv::putText 不可用
// 相机设备不沿用编号硬编码：video44 这种裸节点能采集但出全黑帧（v4l2-ctl
// 验证过），正点原子预留的 /dev/video-camera0 符号链接才指向真 ISP 输出，
// 且节点号随内核版本变、链接不变。
//
// 推理与后处理走本项目已逐像素验证过的 postprocess.h，不抄官方的
// convert_image_with_letterbox 和三分支 post_process：
// 前者的 resize 尺寸会对齐到 4/2 的倍数，与 ultralytics 不一致；
// 后者的输出布局是 YOLOv5/v8 老式三路，和本项目的 one2many 单输出不兼容。
//
// 速度基准：分段统计预处理 / 推理 / 后处理 / 显示的耗时，
// 给出"处理帧率"（系统能力）和"实时帧率"（含丢帧后的实际节奏）。
// 精度基准：--save-frame 把喂给模型的那一帧原图存成 PNG，
// PC 侧拿这张图跑 PyTorch 基准，与板端结果逐框比对。
//
// 用法: camera <model.rknn> [cam_index] [--int8] [--no-display] [--seconds N]
//                      [--save-frame out.png] [--no-drop]

#include <rknn_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image_drawing.h"
#include "labels.h"
#include "postprocess.h"
#include "result_json.h"
#include "rknn_io.h"

using board::Det;

namespace {

using Clock = std::chrono::steady_clock;

// 正点原子把 MIPI ISP 输出包在符号链接里，见头注释
constexpr const char* kDefaultCam = "/dev/video-camera0";
constexpr const char* kRule = "--------------------------------------------------------------";

double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// imshow 不会缩放图像：窗口铺满屏幕后，画面仍按原始像素 1:1 画在左上角。
// 这里等比缩放到目标尺寸再居中贴到黑底上，比例和相机帧一致，画框位置才不会偏。
void FitToScreen(const cv::Mat& src, cv::Mat& dst, const cv::Size& out) {
  const float s = std::min(out.width / static_cast<float>(src.cols),
                            out.height / static_cast<float>(src.rows));
  const cv::Size fit(static_cast<int>(src.cols * s), static_cast<int>(src.rows * s));
  cv::Mat scaled;
  cv::resize(src, scaled, fit);
  dst.setTo(cv::Scalar(0, 0, 0));
  scaled.copyTo(dst(cv::Rect((out.width - fit.width) / 2, (out.height - fit.height) / 2, fit.width,
                             fit.height)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "用法: %s <model.rknn> [cam|设备路径] [--int8] [--no-display] [--seconds N]\n"
                 "            [--save-frame out.png] [--no-drop]\n",
                 argv[0]);
    return 1;
  }

  const std::string model_path = argv[1];
  std::string cam = kDefaultCam;
  bool int8 = false, display = true, drop = true;
  double seconds = 0.0;
  std::string save_frame;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--int8")) int8 = true;
    else if (!std::strcmp(argv[i], "--no-display")) display = false;
    else if (!std::strcmp(argv[i], "--no-drop")) drop = false;
    else if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--save-frame") && i + 1 < argc) save_frame = argv[++i];
    else cam = argv[i];
  }

  std::vector<uint8_t> model = board::ReadFile(model_path);
  rknn_context ctx;
  board::Check(rknn_init(&ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr),
               "rknn_init");

  rknn_sdk_version ver{};
  rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));

  rknn_tensor_attr in_attr{};
  in_attr.index = 0;
  board::Check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)), "查询输入属性");
  rknn_tensor_attr in_native{};
  in_native.index = 0;
  const bool has_in_native =
      rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_native, sizeof(in_native)) == 0;

  rknn_tensor_attr out_attr{};
  out_attr.index = 0;
  board::Check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr)), "查询输出属性");
  const int out_nd = static_cast<int>(out_attr.n_dims);
  const int na = out_attr.dims[out_nd - 1];
  const int nc = out_attr.dims[out_nd - 2] - 4;
  const bool quant_out = out_attr.type == RKNN_TENSOR_INT8;
  if (int8 && !quant_out) {
    std::fprintf(stderr, "--int8 与模型输出类型 %s 不符\n", board::TypeName(out_attr.type));
    return 1;
  }

  // native 输入就是 FP16/NHWC 时直接喂原生格式，runtime 只剩一次拷贝，
  // 不必在 CPU 上把 123 万个像素逐个转成 FP16
  const bool native_fp16_nhwc = has_in_native && in_native.type == RKNN_TENSOR_FLOAT16 &&
                                in_native.fmt == RKNN_TENSOR_NHWC && in_native.n_dims == 4;
  const bool feed_fp16 = native_fp16_nhwc;
  const board::Feed feed = feed_fp16 ? board::Feed::Fp16 : board::Feed::U8;

  std::printf("%s\n%s  %s  (%.2f MB)  cam=%s\n", kRule, quant_out ? "INT8" : "FP16",
              model_path.c_str(), model.size() / 1e6, cam.c_str());
  std::printf("  runtime API %s / driver %s\n", ver.api_version, ver.drv_version);
  std::printf("  输入 %s   输出 [1,%d,%d]\n", feed_fp16 ? "NHWC FLOAT16" : "NHWC UINT8", 4 + nc, na);
  std::printf("%s\n", kRule);

  // 显式指定 CAP_V4L2：按索引打开会先撞一遍 GStreamer（v4l2src 分配缓冲
  // 失败）才落到 V4L2；纯数字参数仍按节点索引打开
  cv::VideoCapture cap;
  if (cam.find_first_not_of("0123456789") == std::string::npos)
    cap.open(std::stoi(cam), cv::CAP_V4L2);
  else
    cap.open(cam, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "打不开相机 %s —— 默认走 /dev/video-camera0 符号链接，用 ls -l /dev/video* 确认\n",
                 cam.c_str());
    return 1;
  }
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  board::InputSink sink{ctx, nullptr, feed};

  if (display) {
    cv::namedWindow("yolo26", cv::WINDOW_NORMAL);
    cv::setWindowProperty("yolo26", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  }

  // 循环外分配，避免每帧 malloc
  cv::Mat canvas(board::kImgsz, board::kImgsz, CV_8UC3);
  cv::Mat frame, rgb, screen;
  image_buffer_t draw_buf{};
  char text[160];

  cv::Size disp(0, 0);  // 首帧后按窗口实际像素确定
  bool first = true;

  int n_captured = 0, n_processed = 0;
  double sum_pre = 0, sum_infer = 0, sum_post = 0, sum_show = 0;
  double npu_ms_sum = 0;

  cv::Mat saved_frame;
  std::vector<Det> saved_dets;
  bool saved = false;

  const auto t_start = Clock::now();
  auto t_last = t_start;
  double fps_display = 0.0;

  while (true) {
    // 处理慢于采集时丢帧：实时链路里延迟优先于帧率，
    // 一直拿最新的帧，而不是把延迟越积越长
    if (drop) {
      cap.grab();
      ++n_captured;
      if (n_captured % 3) continue;
      if (!cap.retrieve(frame) || frame.empty()) continue;
    } else {
      cap >> frame;
      ++n_captured;
      if (frame.empty()) {
        std::fprintf(stderr, "取帧失败\n");
        break;
      }
    }

    const auto t_pre0 = Clock::now();

    // MIPI 摄像头物理竖装，转正后画框坐标才对得上
    cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
    const board::LetterboxInfo info = board::Letterbox(frame, canvas);
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);  // RKNN 输入按 RGB 排布
    cv::Mat blob = feed_fp16 ? board::ToFp16(rgb) : rgb;

    const auto t_pre1 = Clock::now();

    sink.Submit(blob);
    board::Check(rknn_run(ctx, nullptr), "rknn_run");

    rknn_output out{};
    out.index = 0;
    out.want_float = 0;  // 拿原生数据，自己转 float32
    out.is_prealloc = 0;
    board::Check(rknn_outputs_get(ctx, 1, &out, nullptr), "rknn_outputs_get");

    const auto t_infer1 = Clock::now();

    const size_t elems = static_cast<size_t>(4 + nc) * na;
    const cv::Mat f32 = board::ToFloat32(out.buf, elems, out_attr.type, out_attr.zp, out_attr.scale);
    const std::vector<Det> dets =
        board::Decode(f32.ptr<float>(), na, quant_out, info, frame.rows, frame.cols);
    rknn_outputs_release(ctx, 1, &out);

    const auto t_post1 = Clock::now();

    // 精度基准：留一帧原图给 PC 侧跑 PyTorch 基准，有检出就锁定该帧。
    // 整段零检出也存第一帧——空结果本身需要证据：是场景里没目标还是
    // 链路有问题，PC 基准对同一帧一跑便知。
    // 必须在画框之前克隆：框和 FPS 文字是直接画在 frame 上的，画完之后
    // 这张图就不再等于喂给模型的输入，PC 基准要与板端同一条输入才可比。
    if (!save_frame.empty() && !saved) {
      saved_frame = frame.clone();
      saved_dets = dets;
      saved = true;
    }

    rknn_perf_run perf_run{};
    if (rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run)) == 0)
      npu_ms_sum += perf_run.run_duration / 1000.0;

    // 画框画在 BGR 的原图上：框坐标本来就是原图坐标系，与通道序无关
    draw_buf.height = frame.rows;
    draw_buf.width = frame.cols;
    draw_buf.width_stride = frame.step[0];
    draw_buf.virt_addr = frame.data;
    draw_buf.format = IMAGE_FORMAT_RGB888;
    draw_buf.size = static_cast<int>(frame.total() * frame.elemSize());

    for (const auto& d : dets) {
      draw_rectangle(&draw_buf, d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1, COLOR_GREEN, 3);
      std::snprintf(text, sizeof(text), "%s %.1f%%", board::ClassName(d.cls), d.score * 100);
      draw_text(&draw_buf, text, d.x1, d.y1 - 5, COLOR_RED, 10);
    }

    const auto now = Clock::now();
    fps_display = 1.0 / std::chrono::duration<double>(now - t_last).count();
    t_last = now;
    std::snprintf(text, sizeof(text), "%.1f FPS  %zu det", fps_display, dets.size());
    draw_text(&draw_buf, text, 10, 20, COLOR_YELLOW, 12);

    const auto t_show0 = Clock::now();

    if (display) {
      if (disp.width > 0) {
        FitToScreen(frame, screen, disp);
      } else {
        screen = frame;
      }
      cv::imshow("yolo26", screen);
      const int key = cv::waitKey(1);
      if (key == 'q' || key == 27) break;

      // 窗口尺寸要等窗口真正建出来才查得到，所以第一帧先按原尺寸显示
      if (first) {
        const cv::Rect wr = cv::getWindowImageRect("yolo26");
        if (wr.width > 0 && wr.height > 0) {
          disp = cv::Size(wr.width, wr.height);
          std::printf("相机画面 %dx%d  LCD 显示区 %dx%d\n", frame.cols, frame.rows, disp.width,
                      disp.height);
          first = false;
        }
      }
    }

    const auto t_show1 = Clock::now();

    sum_pre += Ms(t_pre0, t_pre1);
    sum_infer += Ms(t_pre1, t_infer1);
    sum_post += Ms(t_infer1, t_post1);
    sum_show += Ms(t_show0, t_show1);
    ++n_processed;

    if (seconds > 0 &&
        std::chrono::duration<double>(Clock::now() - t_start).count() >= seconds) {
      break;
    }
  }

  const double total_s = std::chrono::duration<double>(Clock::now() - t_start).count();
  const double n = n_processed > 0 ? n_processed : 1;
  const double avg_total = (sum_pre + sum_infer + sum_post + sum_show) / n;

  std::printf("\n%s\n", kRule);
  std::printf("采集 %d 帧，处理 %d 帧，耗时 %.1f 秒\n", n_captured, n_processed, total_s);
  std::printf("  处理帧率    %6.2f FPS   (系统能力，不含采集等待)\n",
              total_s > 0 ? n_processed / total_s : 0.0);
  std::printf("  平均单帧    %6.2f ms\n", avg_total);
  std::printf("    预处理     %6.2f ms   (旋转 + letterbox + 通道转换)\n", sum_pre / n);
  std::printf("    推理       %6.2f ms   (喂入 + rknn_run + 取回输出)\n", sum_infer / n);
  std::printf("    后处理     %6.2f ms   (解码 + NMS + 反 letterbox)\n", sum_post / n);
  std::printf("    显示       %6.2f ms\n", sum_show / n);
  std::printf("  NPU 算子    %6.2f ms   (rknn_query PERF_RUN)\n", npu_ms_sum / n);

  if (!save_frame.empty()) {
    cv::imwrite(save_frame, saved_frame);
    std::printf("  基准帧已存 %s  (%dx%d, %zu 个框)%s\n", save_frame.c_str(), saved_frame.cols,
                saved_frame.rows, saved_dets.size(),
                saved_dets.empty() ? "  ! 整段零检出，存的是第一帧，供 PC 侧排查" : "");
  }

  board::ResultJson result;
  result.model = board::BaseName(model_path);
  result.model_bytes = model.size();
  result.image = saved ? board::BaseName(save_frame) : "camera";
  result.image_width = saved ? saved_frame.cols : 0;
  result.image_height = saved ? saved_frame.rows : 0;
  result.detections = saved_dets;
  result.precision = quant_out ? "int8" : "fp16";
  result.runtime_api = ver.api_version;
  result.npu_driver = ver.drv_version;
  result.input_desc = feed_fp16 ? "NHWC FLOAT16" : "NHWC UINT8";
  result.output_desc = "[1,84,8400]  80 类 x 8400 锚点";
  result.fps = total_s > 0 ? n_processed / total_s : 0.0;
  result.frames = n_processed;
  result.dropped = n_captured - n_processed;
  result.loops = n_processed;
  result.npu_ms = npu_ms_sum / n;
  result.wall_ms = avg_total;
  char seg[192];
  std::snprintf(seg, sizeof(seg),
                "预处理 %.2f ms | 推理 %.2f ms | 后处理 %.2f ms | 显示 %.2f ms | 采集 %d 帧",
                sum_pre / n, sum_infer / n, sum_post / n, sum_show / n, n_captured);
  result.perf_detail = seg;
  board::WriteResultJson("camera_out.json", result);
  std::printf("  -> camera_out.json\n");

  rknn_destroy(ctx);
  return 0;
}
