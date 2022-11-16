/*
 * @Description: Decode stream pipeline.
 * @version: 2.2
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-10-11 10:46:12
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-10-19 06:16:14
 */
#pragma once

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <gst/gst.h>

#include "SafeQueue.h"

/*
 * 
 * gst-launch-1.0 uridecodebin uri="" ! qtivtransform ! video/x-raw,format=RGB,width=1920,height=1080 ! appsink drop=true sync=false
 * 
 */

class VideoPipelineConfig {
public:
    std::string url;
    int cameraID;
    int streamWidth;
    int streamHeight;
    int streamFramerateN;
    int streamFramerateD;
    std::string convertFormat;
    bool isDropBuffer;
    bool isSync;
};

class VideoPipeline {
public:
    VideoPipeline (const VideoPipelineConfig& config);
    ~VideoPipeline(void);
    bool Create   (void);
    bool Start    (void);
    void Destroy  (void);
    void SetUserData(std::shared_ptr<SafetyQueue<cv::Mat>> user_data);

    VideoPipelineConfig config;
    GstElement* pipeline;
    GstElement* source;
    GstElement* srcFilter;
    GstElement* queue;
    GstElement* converter;
    GstElement* convFilter;
    GstElement* appsink;
    uint32_t    bus_watch_id;

    bool dump;
    std::shared_ptr<SafetyQueue<cv::Mat>> productQueue;
};