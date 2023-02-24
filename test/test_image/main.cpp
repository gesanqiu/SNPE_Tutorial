/*
 * @Description: Test program of yolov5s. 
 * @version: 2.1
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-05-18 16:51:10
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2023-02-24 15:02:47
 */

#include <string>
#include <vector>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <json-glib/json-glib.h>

#include "YOLOv5s.h"
#include "YOLOv5sImpl.h"
#include "utils.h"

static runtime_t device2runtime(std::string & device)
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
    } else if (0 == device.compare("dsp_fixed8")) {
        return DSP_FIXED8;
    } else if (0 == device.compare("aip")) {
        return AIP;
    } else { 
        return CPU;
    }
}

static bool validateInput(const char* name, const std::string& value) 
{ 
    if (!value.compare ("")) {
        LOG_ERROR("You must specify an input file!");
        return false;
    }

    struct stat statbuf;
    if (0 == stat(value.c_str(), &statbuf)) {
        return true;
    }

    LOG_ERROR("Can't stat input image file: %s", value.c_str());
    return false;
}

DEFINE_string(input, "./image.jpg", "Input image file for this test program.");
DEFINE_validator(input, &validateInput);

static bool validateLabels(const char* name, const std::string& value) 
{ 
    if (!value.compare ("")) {
        LOG_ERROR("You must specify a labels file!");
        return false;
    }

    struct stat statbuf;
    if (0 == stat(value.c_str(), &statbuf)) {
        return true;
    }

    LOG_ERROR("Can't stat labels file: %s", value.c_str());
    return false;
}

DEFINE_string(labels, "./labels.txt", "Labels file for the yolov5s model.");
DEFINE_validator(labels, &validateLabels);

static bool validateConfigPath(const char* name, const std::string& value) 
{ 
    if (0 == value.compare ("")) {
        LOG_ERROR("You must specify a dlc file!");
        return false;
    }

    struct stat statbuf;
    if (0 == stat(value.c_str(), &statbuf)) {
        return true;
    }

    LOG_ERROR("Can't stat model file: %s", value.c_str());
    return false;
}

DEFINE_string(config_path, "./config.json", "Model config file path.");
DEFINE_validator(config_path, &validateConfigPath);

DEFINE_double(confidence, 0.5, "Confidence Threshold.");
DEFINE_double(nms, 0.5, "NMS Threshold.");

static bool parse_args(yolov5::ObjectDetectionConfig& config, const std::string& path)
{
    JsonParser* parser = NULL;
    JsonNode*   root   = NULL;
    JsonObject* object = NULL;
    GError*     error  = NULL;
    bool        ret    = false;

    if (!(parser = json_parser_new())) {
        LOG_ERROR("Failed to new a object with type JsonParser");
        return false;
    }

    if (json_parser_load_from_file(parser, (const gchar *)path.c_str(),&error)) {
        if (!(root = json_parser_get_root(parser))) {
            LOG_ERROR("Failed to get root node from JsonParser");
            goto done;
        }

        if (JSON_NODE_HOLDS_OBJECT(root)) {
            if (!(object = json_node_get_object(root))) {
                LOG_ERROR("Failed to get object from JsonNode");
                goto done;
            }

            if (json_object_has_member(object, "model-path")) {
                std::string mp((const char*)json_object_get_string_member(object, "model-path"));
                LOG_INFO("model-path: {}", mp);
                config.model_path = mp;
            }

            if (json_object_has_member(object, "runtime")) {
                std::string r((const char*)json_object_get_string_member(object, "runtime"));
                LOG_INFO("runtime: {}", r);
                config.runtime = device2runtime(r);
            }

            if (json_object_has_member(object, "labels")) {
                int l = json_object_get_int_member(object, "labels");
                LOG_INFO("labels: {}", l);
                config.labels = l;
            }

            if (json_object_has_member(object, "grids")) {
                int g = json_object_get_int_member(object, "grids");
                LOG_INFO("grids: {}", g);
                config.grids = g;
            }

            if (json_object_get_array_member(object, "input-layers")) {
                JsonArray* a = json_object_get_array_member(object, "input-layers");
                std::vector<std::string> ils;
                for (int i = 0; i < json_array_get_length(a); i++) {
                    ils.emplace_back(json_array_get_string_element(a, i));
                    LOG_INFO("input-layers[{}]: {}", i, ils[i]);
                }
                config.inputLayers = ils;
            }

            if (json_object_get_array_member(object, "output-layers")) {
                JsonArray* a = json_object_get_array_member(object, "output-layers");
                std::vector<std::string> ols;
                for (int i = 0; i < json_array_get_length(a); i++) {
                    ols.emplace_back(json_array_get_string_element(a, i));
                    LOG_INFO("output-layers[{}]: {}", i, ols[i]);
                }
                config.outputLayers = ols;
            }

            if (json_object_get_array_member(object, "output-tensors")) {
                JsonArray* a = json_object_get_array_member(object, "output-tensors");
                std::vector<std::string> olt;
                for (int i = 0; i < json_array_get_length(a); i++) {
                    olt.emplace_back(json_array_get_string_element(a, i));
                    LOG_INFO("output-tensors[{}]: {}", i, olt[i]);
                }
                config.outputTensors = olt;
            }
        }
    } else {
        LOG_ERROR("Failed to parse json string {}, {}", error->message, path.c_str());
        g_error_free (error);
        goto done;
    }

    ret = true;

done:
    g_object_unref (parser);

    return ret;
}

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    std::vector<std::string> labels;
    std::ifstream in(FLAGS_labels);
    std::string line;
    while (getline(in, line)){
        labels.push_back(line);
    }

    yolov5::ObjectDetectionConfig config;
    parse_args(config, FLAGS_config_path);

    std::vector<std::shared_ptr<yolov5::ObjectDetection> > vec_alg;
    for (int i = 0; i < 1; i++) {
        std::shared_ptr<yolov5::ObjectDetection> alg = std::shared_ptr<yolov5::ObjectDetection>(new yolov5::ObjectDetection());
        alg->Init(config);
        alg->SetScoreThreshold(FLAGS_confidence, FLAGS_nms);
        vec_alg.push_back(alg);
    }

    for (int i = 0; i < 1; i++) {
        cv::Mat img = cv::imread(FLAGS_input);
        cv::Mat rgb_img;

        cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

        std::vector<yolov5::ObjectData> vec_res;

        vec_alg[i]->Detect(rgb_img, vec_res);

        LOG_INFO("result size: {}", vec_res.size());

        for (size_t j = 0; j < vec_res.size(); j++) {
            yolov5::ObjectData result = vec_res[j];
            LOG_INFO("[{}, {}, {}, {}, {}, {}]", result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height, result.confidence, result.label);
            cv::rectangle(img, cv::Rect(result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height), cv::Scalar(0, 255, 0), 3);
            cv::Point position = cv::Point(result.bbox.x, result.bbox.y - 10);
            cv::putText(img, labels[result.label], position, cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, 0.3);
        }

        std::string output_path = "./object_detection_result_" + std::to_string(i) + ".jpg";

        cv::imwrite(output_path, img);
    }

    google::ShutDownCommandLineFlags();
    return 0;
}
