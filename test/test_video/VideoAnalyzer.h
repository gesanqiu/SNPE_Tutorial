/*
 * @Description: Inference decoded stream with libYOLOv5s.so.
 * @version: 2.2
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-10-11 11:50:34
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-10-19 06:14:01
 */
#pragma once

#include <thread>
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <map>

#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>
#include <mosquitto.h>

#include "YOLOv5s.h"
#include "YOLOv5sImpl.h"
#include "utils.h"
#include "SafeQueue.h"

struct MQTTClientConfig {
    std::string brokerIP;
    int brokerPort;
    int keepAlive;
    int QoS;
    std::string topicName;
    bool isSendBase64;
};

class VideoAnalyzer {
public:
    VideoAnalyzer();
    ~VideoAnalyzer();
    bool Init(Json::Value& model, Json::Value& mqtt);
    bool DeInit();
    bool Start();
    void SetUserData(std::shared_ptr<SafetyQueue<cv::Mat>> user_data);

private:
    void ParseConfig(Json::Value& root, yolov5::ObjectDetectionConfig& config);

private:
    bool isRunning;
    void InferenceFrame();
    std::shared_ptr<std::thread> inferThread;

    MQTTClientConfig mqttConfig;
    struct mosquitto* mqttClient;

    std::unordered_map<std::string, std::shared_ptr<yolov5::ObjectDetection>> detectors;
    std::unordered_map<std::string, std::vector<std::string>> labels;
    std::unordered_map<std::string, std::vector<float>> thresholds;
    std::shared_ptr<SafetyQueue<cv::Mat>> consumeQueue;
};