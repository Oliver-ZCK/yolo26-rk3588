// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ---------------------------------------------------------------------------
// 这是本工程对 rknn_model_zoo 的 postprocess.cc 的唯一一处替换。
//
// 官方实现按 YOLOv8 / YOLO11 的三分支 DFL 输出解码：模型有 3 个尺度分支，
// 每支输出 box(64 通道 DFL 分布) 和 score(80 类)，框坐标靠 DFL 积分还原。
//
// YOLO26 移除了 DFL（reg_max=1），检测头改成 one2many，模型只有 1 个输出
//     [1, 4+nc, na] = [1, 84, 8400]
// 其中 na = 80² + 40² + 20² 个锚点，通道 0..3 是 cx/cy/w/h，4..83 是类别分数。
//
// 官方 post_process() 里 output_per_branch = n_output / 3 = 1 / 3 = 0，
// box_idx = 0、score_idx = 1，会去读 output_attrs[1] 和 outputs[1]
// （两个数组都只有 1 个元素）——越界，官方二进制跑不了 YOLO26 的模型文件。
//
// 保持不变的部分：对外接口（init/deinit_post_process、coco_cls_to_name、
// post_process）、逐类别 NMS 语义、标签文件的加载方式。
// ---------------------------------------------------------------------------

#include "yolo11.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

namespace
{

// 置信度阈值以上的候选框上限。正常场景远达不到，防止异常张量把内存吃光。
constexpr size_t kMaxCandidates = 4096;

struct Cand
{
    float x1, y1, x2, y2;  // letterbox 坐标系
    float score;
    int cls;
};

float IoU(const Cand &a, const Cand &b)
{
    const float iw = fmax(0.f, fmin(a.x2, b.x2) - fmax(a.x1, b.x1));
    const float ih = fmax(0.f, fmin(a.y2, b.y2) - fmax(a.y1, b.y1));
    const float inter = iw * ih;
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

}  // namespace

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results)
{
    if (app_ctx == nullptr || outputs == nullptr || letter_box == nullptr || od_results == nullptr)
    {
        return -1;
    }

    memset(od_results, 0, sizeof(*od_results));

    const rknn_tensor_attr *out_attr = &app_ctx->output_attrs[0];
    if (out_attr->n_dims != 3 || out_attr->dims[0] != 1)
    {
        printf("unexpected output shape: n_dims=%d dims=[%d %d %d]\n", out_attr->n_dims, out_attr->dims[0],
               out_attr->dims[1], out_attr->dims[2]);
        return -1;
    }
    const int nc = out_attr->dims[1] - 4;  // 类别数
    const int na = out_attr->dims[2];      // 锚点数

    rknn_output *_outputs = (rknn_output *)outputs;
    const void *buf = _outputs[0].buf;
    if (buf == nullptr)
    {
        return -1;
    }

    // want_float = !is_quant：FP16 模型取回的是 float32；INT8 模型取回的是原始 int8
    const bool quant = app_ctx->is_quant;
    const float q_scale = quant ? out_attr->scale : 1.0f;
    const int32_t q_zp = quant ? out_attr->zp : 0;

    // 输出张量的逻辑布局是 NCHW，形状 [1, 4+nc, na]，第 c 通道第 a 个值在 c*na+a
    auto value_at = [&](int c, int a) -> float {
        const size_t idx = static_cast<size_t>(c) * na + a;
        if (quant)
        {
            return (static_cast<const int8_t *>(buf)[idx] - q_zp) * q_scale;
        }
        return static_cast<const float *>(buf)[idx];
    };

    std::vector<Cand> cands;
    cands.reserve(256);
    for (int a = 0; a < na; ++a)
    {
        float best = 0.f;
        int cls = -1;
        for (int c = 0; c < nc; ++c)
        {
            const float s = value_at(4 + c, a);
            if (s > best)
            {
                best = s;
                cls = c;
            }
        }
        if (cls < 0 || best <= conf_threshold)
        {
            continue;
        }

        float cx = value_at(0, a);
        float cy = value_at(1, a);
        float bw = value_at(2, a);
        float bh = value_at(3, a);
        if (quant)
        {
            // INT8 导出时 ultralytics 在检测头后插了 _NormalizeCoords，
            // 框坐标是相对输入尺寸的 0~1，不乘回去框会缩到 1/640
            cx *= app_ctx->model_width;
            cy *= app_ctx->model_height;
            bw *= app_ctx->model_width;
            bh *= app_ctx->model_height;
        }
        cands.push_back({cx - bw / 2.f, cy - bh / 2.f, cx + bw / 2.f, cy + bh / 2.f, best, cls});
        if (cands.size() >= kMaxCandidates)
        {
            break;
        }
    }

    // 逐类别 NMS：同一目标被判成两种类时两个框都保留，与 ultralytics 一致
    std::vector<Cand> kept;
    for (int c = 0; c < nc; ++c)
    {
        std::vector<Cand> same;
        for (const auto &d : cands)
        {
            if (d.cls == c)
            {
                same.push_back(d);
            }
        }
        std::sort(same.begin(), same.end(), [](const Cand &x, const Cand &y) { return x.score > y.score; });
        for (const auto &d : same)
        {
            bool suppressed = false;
            for (const auto &k : kept)
            {
                if (k.cls == c && IoU(d, k) > nms_threshold)
                {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed)
            {
                kept.push_back(d);
            }
        }
    }
    std::sort(kept.begin(), kept.end(), [](const Cand &x, const Cand &y) { return x.score > y.score; });

    // letterbox 坐标 -> 原图坐标。letter_box->scale 是缩放比，x_pad/y_pad 是居中填充量。
    // 内容区尺寸 = 模型输入尺寸 - 两侧填充；填充为奇数时这里会差 1 px。
    const float s = letter_box->scale;
    const float content_w = app_ctx->model_width - 2 * letter_box->x_pad;
    const float content_h = app_ctx->model_height - 2 * letter_box->y_pad;

    int count = 0;
    for (const auto &d : kept)
    {
        if (count >= OBJ_NUMB_MAX_SIZE)
        {
            break;
        }
        float x1 = fmin(fmax(d.x1 - letter_box->x_pad, 0.f), content_w);
        float y1 = fmin(fmax(d.y1 - letter_box->y_pad, 0.f), content_h);
        float x2 = fmin(fmax(d.x2 - letter_box->x_pad, 0.f), content_w);
        float y2 = fmin(fmax(d.y2 - letter_box->y_pad, 0.f), content_h);

        od_results->results[count].box.left = (int)(x1 / s);
        od_results->results[count].box.top = (int)(y1 / s);
        od_results->results[count].box.right = (int)(x2 / s);
        od_results->results[count].box.bottom = (int)(y2 / s);
        od_results->results[count].prop = d.score;
        od_results->results[count].cls_id = d.cls;
        count++;
    }
    od_results->count = count;
    return 0;
}

int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{

    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
