/*
 * @Description: Inference decoded stream with libYOLOv5s.so.
 * @version: 2.2
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-10-11 11:50:40
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-10-25 06:36:24
 */

#include <iostream>
#include <fstream>
#include <sys/time.h>

#include "VideoAnalyzer.h"

static std::string base64Decode(const char* Data, int DataByte)
{
    const char DecodeTable[] =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        62, // '+'
        0, 0, 0,
        63, // '/'
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
        0, 0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
        0, 0, 0, 0, 0, 0,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
    };
    std::string strDecode;
    int nValue;
    int i = 0;
    while (i < DataByte) {
        if (*Data != '\r' && *Data != '\n') {
            nValue = DecodeTable[*Data++] << 18;
            nValue += DecodeTable[*Data++] << 12;
            strDecode += (nValue & 0x00FF0000) >> 16;
            if (*Data != '=') {
                nValue += DecodeTable[*Data++] << 6;
                strDecode += (nValue & 0x0000FF00) >> 8;
                if (*Data != '=') {
                    nValue += DecodeTable[*Data++];
                    strDecode += nValue & 0x000000FF;
                }
            }
            i += 4;
        }
        else {
            Data++;
            i++;
        }
    }
    return strDecode;
}
 
 
static std::string base64Encode(const unsigned char* Data, int DataByte)
{
    const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string strEncode;
    unsigned char Tmp[4] = { 0 };
    int LineLength = 0;
    for (int i = 0; i < (int)(DataByte / 3); i++) {
        Tmp[1] = *Data++;
        Tmp[2] = *Data++;
        Tmp[3] = *Data++;
        strEncode += EncodeTable[Tmp[1] >> 2];
        strEncode += EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
        strEncode += EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
        strEncode += EncodeTable[Tmp[3] & 0x3F];
        if (LineLength += 4, LineLength == 76) { strEncode += "\r\n"; LineLength = 0; }
    }

    int Mod = DataByte % 3;
    if (Mod == 1) {
        Tmp[1] = *Data++;
        strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
        strEncode += EncodeTable[((Tmp[1] & 0x03) << 4)];
        strEncode += "==";
    }
    else if (Mod == 2) {
        Tmp[1] = *Data++;
        Tmp[2] = *Data++;
        strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
        strEncode += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
        strEncode += EncodeTable[((Tmp[2] & 0x0F) << 2)];
        strEncode += "=";
    }
 
 
    return strEncode;
}
 
//imgType: png bmp jpg jpeg
static std::string Mat2Base64(const cv::Mat& img, std::string imgType)
{
    std::string img_data;
    std::vector<uchar> vecImg;
    std::vector<int> vecCompression_params;
    vecCompression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    vecCompression_params.push_back(90);
    imgType = "." + imgType;
    cv::imencode(imgType, img, vecImg, vecCompression_params);
    img_data = base64Encode(vecImg.data(), vecImg.size());
    return img_data;
}
 
 
static cv::Mat Base2Mat(std::string& base64_data)
{
    cv::Mat img;
    std::string s_mat;
    s_mat = base64Decode(base64_data.data(), base64_data.size());
    std::vector<char> base64_img(s_mat.begin(), s_mat.end());
    img = cv::imdecode(base64_img, cv::IMREAD_COLOR);
    return img;
}

static runtime_t device2runtime(std::string&& device)
{
    std::transform(device.begin(), device.end(), device.begin(),
        [](unsigned char ch){ return tolower(ch); });

    if (0 == device.compare("cpu")) {
        return CPU;
    } else if (0 == device.compare("gpu")) {
        return GPU;
    } else if (0 == device.compare("gpu_float16")) {
        return GPU_FLOAT16;
    } else if (0 == device.compare("dsp")) {
        return DSP;
    } else if (0 == device.compare("aip")) {
        return AIP;
    } else { 
        return CPU;
    }
}

void VideoAnalyzer::InferenceFrame()
{
    std::shared_ptr<cv::Mat> image;

    while (isRunning) {
        Json::Value root;
        for (auto& [k, v] : detectors) {
            std::vector<yolov5::ObjectData> results;
            consumeQueue->consumption(image);
            v->Detect(*image.get(), results);

            for (auto& result : results) {
                if (result.confidence >= thresholds[k][result.label]) {
                    Json::Value object;
                    object["bbox"]["x"] = result.bbox.x;
                    object["bbox"]["y"] = result.bbox.y;
                    object["bbox"]["width"] = result.bbox.width;
                    object["bbox"]["height"] = result.bbox.height;
                    object["confidence"] = result.confidence;
                    object["label"] = labels[k][result.label];
                    object["model"] = k;
                    root["results"].append(object);
                }
            }
        }
        if (root.isMember("results")) {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            long ts = tv.tv_sec * 1000 + tv.tv_usec / 1000;
            root["timestamp"] = std::to_string(ts);
        }
        if (mqttConfig.isSendBase64) root["image"] = Mat2Base64(*(image.get()), "jpg");
        // LOG_INFO("inference result: {}", root.toStyledString());
        std::string message = root.toStyledString();
        if (!root.isNull()) mosquitto_publish(mqttClient, nullptr, mqttConfig.topicName.data(), message.size(), message.data(), mqttConfig.QoS, false);
    }
}

void VideoAnalyzer::ParseConfig(Json::Value& root, yolov5::ObjectDetectionConfig& config)
{
    config.model_path = root["model-path"].asString();
    config.runtime = device2runtime(root["runtime"].asString());
    config.labels = root["labels"].asInt();
    config.grids = root["grids"].asInt();
    if (root["input-layers"].isArray()) {
        int sz = root["input-layers"].size();
        for (int i = 0; i < sz; ++i)
            config.inputLayers.push_back(root["input-layers"][i].asString());
    }
    if (root["output-layers"].isArray()) {
        int sz = root["output-layers"].size();
        for (int i = 0; i < sz; ++i)
            config.outputLayers.push_back(root["output-layers"][i].asString());
    }
    if (root["output-tensors"].isArray()) {
        int sz = root["output-tensors"].size();
        for (int i = 0; i < sz; ++i)
            config.outputTensors.push_back(root["output-tensors"][i].asString());
    }
}

VideoAnalyzer::VideoAnalyzer()
{
    isRunning = false;
}

VideoAnalyzer::~VideoAnalyzer()
{
    DeInit();
}

bool VideoAnalyzer::Init(Json::Value& model, Json::Value& mqtt)
{
    mqttConfig.brokerIP = mqtt["ip"].asString();
    mqttConfig.brokerPort = mqtt["port"].asInt();
    mqttConfig.topicName = mqtt["topic-name"].asString();
    mqttConfig.keepAlive = mqtt["keepalive"].asInt();
    mqttConfig.QoS = mqtt["QoS"].asInt();
    mqttConfig.isSendBase64 = mqtt["send-base64"].asBool();

    if (model.isArray()) {
        int sz = model.size();
        for (int i = 0; i < sz; ++i) {
            std::string modelName = model[i]["model-name"].asString();
            std::ifstream in(model[i]["label-path"].asString());
            std::string line;
            std::vector<std::string> label;
            while (getline(in, line)){
                label.push_back(line);
            }
            this->labels[modelName] = label;
            in.close();

            in.open(model[i]["threshold-path"].asString());
            std::vector<float> threshold;
            while (getline(in, line)){
                threshold.push_back(std::stof(line));
            }
            this->thresholds[modelName] = threshold;
            in.close();

            std::shared_ptr<yolov5::ObjectDetection> detector = std::shared_ptr<yolov5::ObjectDetection>(new yolov5::ObjectDetection());
            yolov5::ObjectDetectionConfig config;
            ParseConfig(model[i], config);
            detector->Init(config);
            detector->SetScoreThreshold(model[i]["global-threshold"].asFloat(), 0.5);
            this->detectors[modelName] = detector;
        }
    }
    
    mosquitto_lib_init();
    mqttClient = mosquitto_new(nullptr, true, nullptr);
    mosquitto_connect_async(mqttClient, mqttConfig.brokerIP.data(), mqttConfig.brokerPort, mqttConfig.keepAlive);

    return true;
}

bool VideoAnalyzer::DeInit()
{
    mosquitto_disconnect(mqttClient);
    mosquitto_loop_stop(mqttClient, true);

    if (inferThread) {
        isRunning = false;
        inferThread->join();
        inferThread = nullptr;
    }
}

bool VideoAnalyzer::Start()
{
    mosquitto_loop_start(mqttClient);

    isRunning = true;
    if (!(inferThread = std::make_shared<std::thread>(std::bind(&VideoAnalyzer::InferenceFrame, this)))) {
        LOG_ERROR("Failed to new a std::thread object");
        isRunning = false;
        return false;
    }

    return true;
}

void VideoAnalyzer::SetUserData(std::shared_ptr<SafetyQueue<cv::Mat>> user_data)
{
    consumeQueue = user_data;
}