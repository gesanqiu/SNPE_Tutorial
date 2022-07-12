/*
 * @Description: Common header.
 * @version: 2.0
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-07-09 13:07:37
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-07-11 20:10:36
 */
#pragma once

#include <algorithm>
#include <functional>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "Logger.h"

#ifndef EXPORT_API
#define EXPORT_API __attribute__ ((visibility("default")))
#endif

// Inference hardware runtime.
typedef enum runtime {
    CPU = 0,
    GPU,
    GPU_16,
    DSP,
    AIP
}runtime_t;

static float calcIoU(const cv::Rect& a, const cv::Rect& b) {
    float xOverlap = std::max(
        0.,
        std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x) + 1.);
    float yOverlap = std::max(
        0.,
        std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y) + 1.);
    float intersection = xOverlap * yOverlap;
    float unio =
        (a.width + 1.) * (a.height + 1.) +
        (b.width + 1.) * (b.height + 1.) - intersection;
    return intersection / unio;
}