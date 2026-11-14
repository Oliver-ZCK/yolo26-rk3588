// 检测头输出的解码链路：letterbox -> decode -> 类别内 NMS -> 反 letterbox -> 边界裁剪。
//
// 每一环的语义都对齐 ultralytics：LetterBox(scale_fill=false, center=true,
// padding_value=114)、non_max_suppression(per-class)、scale_boxes + clip_boxes。
// 任何一环与基准不一致，都会直接体现在与 PT 基准的逐框比对上。
//
// letterbox 的填充值必须是 114 而不是 0：填充带占 640 宽度的 12.5%，
// 参与后续所有卷积，纯黑与灰的响应差异会让贴着填充边的目标框偏移数像素。

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include "labels.h"

namespace board {

constexpr int kImgsz = 640;
constexpr float kConf = 0.25f;
constexpr float kIou = 0.45f;
constexpr uchar kPadValue = 114;  // ultralytics LetterBox.padding_value

struct Det {
  float x1, y1, x2, y2, score;
  int cls;
};

struct LetterboxInfo {
  // ultralytics 的 scale_boxes 不带 ratio_pad 时用缩放后尺寸反推增益：
  // gain_x = new_w / w、gain_y = new_h / h，与 letterbox 的 r 相差舍入尾数
  float gain_x, gain_y;
  int pad_x, pad_y;
};

inline float Iou(const Det& a, const Det& b) {
  const float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = iw * ih;
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

// 类别内分别 NMS：同一目标被判成两种类时，两个框都要保留。
inline std::vector<Det> NmsPerClass(std::vector<Det> dets) {
  std::vector<Det> kept;
  for (int c = 0; c < kNumClasses; ++c) {
    std::vector<Det> same;
    for (const auto& d : dets)
      if (d.cls == c) same.push_back(d);
    std::sort(same.begin(), same.end(),
              [](const Det& a, const Det& b) { return a.score > b.score; });
    for (const auto& d : same) {
      bool suppressed = false;
      for (const auto& k : kept) {
        if (k.cls == c && Iou(d, k) > kIou) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) kept.push_back(d);
    }
  }
  std::sort(kept.begin(), kept.end(),
            [](const Det& a, const Det& b) { return a.score > b.score; });
  return kept;
}

inline LetterboxInfo Letterbox(const cv::Mat& img, cv::Mat& canvas) {
  const int h = img.rows, w = img.cols;
  const float r = std::min(static_cast<float>(kImgsz) / h, static_cast<float>(kImgsz) / w);
  const int nh = static_cast<int>(std::round(h * r));
  const int nw = static_cast<int>(std::round(w * r));

  // ultralytics 把余量对半分后按 round(x±0.1) 分配，保证奇数余量两侧不丢像素
  const float dw = (kImgsz - nw) / 2.0f;
  const float dh = (kImgsz - nh) / 2.0f;
  const int left = static_cast<int>(std::round(dw - 0.1));
  const int top = static_cast<int>(std::round(dh - 0.1));

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  canvas = cv::Mat(kImgsz, kImgsz, CV_8UC3, cv::Scalar(kPadValue, kPadValue, kPadValue));
  resized.copyTo(canvas(cv::Rect(left, top, nw, nh)));
  return {static_cast<float>(nw) / w, static_cast<float>(nh) / h, left, top};
}

// 取一个元素并还原成浮点。INT8 输出时 runtime 给的是量化后的原始 int8，
// 按张量的 zp/scale 反量化，这一步放在 main.cc 的搬运层做（见 ToFloat32），
// 几何解码这一层只认 float。
//
// p 布局 [1, 4+nc, anchors]，第 c 通道第 a 个值取 p[c * na + a]
//
// normalized_coords: ultralytics 仅在 format=rknn 且 quantize=8 时插入
// _NormalizeCoords，框坐标被归一化到 0~1。不乘回输入尺寸的话框会缩到 1/640
// 且不报任何错。
inline std::vector<Det> Decode(const float* p, int na, bool normalized_coords,
                                const LetterboxInfo& info, int img_h, int img_w) {
  std::vector<Det> dets;
  for (int a = 0; a < na; ++a) {
    float best = 0.f;
    int cls = -1;
    for (int c = 0; c < kNumClasses; ++c) {
      const float s = p[(4 + c) * na + a];
      if (s > best) {
        best = s;
        cls = c;
      }
    }
    if (best <= kConf) continue;

    float cx = p[0 * na + a], cy = p[1 * na + a];
    float bw = p[2 * na + a], bh = p[3 * na + a];
    if (normalized_coords) {
      cx *= kImgsz;
      cy *= kImgsz;
      bw *= kImgsz;
      bh *= kImgsz;
    }
    dets.push_back({cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, best, cls});
  }

  dets = NmsPerClass(std::move(dets));

  // 反 letterbox 后裁剪到原图范围，与 ultralytics scale_boxes 的 clip_boxes 一致
  for (auto& d : dets) {
    d.x1 = (d.x1 - info.pad_x) / info.gain_x;
    d.x2 = (d.x2 - info.pad_x) / info.gain_x;
    d.y1 = (d.y1 - info.pad_y) / info.gain_y;
    d.y2 = (d.y2 - info.pad_y) / info.gain_y;
    d.x1 = std::clamp(d.x1, 0.f, static_cast<float>(img_w));
    d.x2 = std::clamp(d.x2, 0.f, static_cast<float>(img_w));
    d.y1 = std::clamp(d.y1, 0.f, static_cast<float>(img_h));
    d.y2 = std::clamp(d.y2, 0.f, static_cast<float>(img_h));
  }
  return dets;
}

}  // namespace board
