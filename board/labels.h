// COCO 80 类的类别名。
//
// 单图模式和摄像头模式都要用，单独放一个头，避免两份拷贝不同步。
// 顺序必须与 ultralytics 的 model.names 一致，class_id 直接按下标取。

#pragma once

namespace board {

inline constexpr const char* const kClassNames[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

inline constexpr int kNumClasses = 80;

inline const char* ClassName(int cls) {
  return (cls >= 0 && cls < kNumClasses) ? kClassNames[cls] : "?";
}

}  // namespace board
