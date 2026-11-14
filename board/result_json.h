// 统一 schema 的结果 JSON。
//
// 单图模式和摄像头模式写同一种格式，PC 侧的比对脚本才能用同一套逻辑读。
// 字段与 src/verify_board.py 期望的结构一一对应。

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "labels.h"
#include "postprocess.h"

namespace board {

struct ResultJson {
  std::string model;
  size_t model_bytes = 0;
  std::string image;
  int image_width = 0;
  int image_height = 0;
  std::vector<Det> detections;

  std::string precision;
  std::string runtime_api;
  std::string npu_driver;
  std::string input_desc;
  std::string output_desc;

  // 推理链路的分段耗时，单位毫秒
  int loops = 0;
  double npu_ms = 0.0;
  double wall_ms = 0.0;
  double set_ms = 0.0;
  double run_ms = 0.0;
  double get_ms = 0.0;
  double wall_min_ms = 0.0;
  double wall_max_ms = 0.0;

  // 摄像头模式：整段运行的实时帧率与丢帧统计
  double fps = 0.0;
  int frames = 0;
  int dropped = 0;

  std::string perf_detail;
};

inline void WriteResultJson(const std::string& path, const ResultJson& r) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "写不了输出文件: %s\n", path.c_str());
    std::exit(1);
  }

  // env 里的字段按需拼接，末尾不能留逗号——空的可选字段曾让 JSON 直接解析失败
  std::vector<std::string> env;
  env.push_back("\"precision\": \"" + r.precision + "\"");
  env.push_back("\"runtime_api\": \"" + r.runtime_api + "\"");
  env.push_back("\"npu_driver\": \"" + r.npu_driver + "\"");
  env.push_back("\"input\": \"" + r.input_desc + "\"");
  env.push_back("\"output\": \"" + r.output_desc + "\"");
  char buf[128];
  if (r.loops > 0) {
    std::snprintf(buf, sizeof(buf), "\"npu_ms\": %.3f", r.npu_ms);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"wall_ms\": %.3f", r.wall_ms);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"set_ms\": %.3f", r.set_ms);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"run_ms\": %.3f", r.run_ms);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"get_ms\": %.3f", r.get_ms);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"wall_min_ms\": %.3f", r.wall_min_ms);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"wall_max_ms\": %.3f", r.wall_max_ms);
    env.push_back(buf);
  }
  if (r.fps > 0.0) {
    std::snprintf(buf, sizeof(buf), "\"fps\": %.2f", r.fps);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"frames\": %d", r.frames);
    env.push_back(buf);
    std::snprintf(buf, sizeof(buf), "\"dropped\": %d", r.dropped);
    env.push_back(buf);
  }
  if (!r.perf_detail.empty()) {
    std::string escaped;
    for (char c : r.perf_detail) {
      if (c == '\n')
        escaped += "\\n";
      else if (c == '"' || c == '\\')
        escaped += '\\', escaped += c;
      else if (c == '\t')
        escaped += "\\t";
      else if (static_cast<unsigned char>(c) >= 0x20)
        escaped += c;
    }
    env.push_back("\"perf_detail\": \"" + escaped + "\"");
  }

  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"model\": \"%s\",\n", r.model.c_str());
  std::fprintf(f, "  \"model_bytes\": %zu,\n", r.model_bytes);
  std::fprintf(f, "  \"image\": \"%s\",\n", r.image.c_str());
  std::fprintf(f, "  \"image_size\": {\"width\": %d, \"height\": %d},\n", r.image_width, r.image_height);
  std::fprintf(f, "  \"detections\": [\n");
  for (size_t i = 0; i < r.detections.size(); ++i) {
    const Det& d = r.detections[i];
    std::fprintf(f,
                 "    {\"class_id\": %d, \"class_name\": \"%s\", \"confidence\": %.4f, "
                 "\"xyxy\": [%.2f, %.2f, %.2f, %.2f]}%s\n",
                 d.cls, ClassName(d.cls), d.score, d.x1, d.y1, d.x2, d.y2,
                 i + 1 < r.detections.size() ? "," : "");
  }
  std::fprintf(f, "  ],\n");
  std::fprintf(f, "  \"env\": {\n");
  for (size_t i = 0; i < env.size(); ++i)
    std::fprintf(f, "    %s%s\n", env[i].c_str(), i + 1 < env.size() ? "," : "");
  std::fprintf(f, "  }\n");
  std::fprintf(f, "}\n");
  std::fclose(f);
}

}  // namespace board
