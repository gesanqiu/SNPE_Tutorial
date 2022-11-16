/*
 * @Description: Test program of yolov5s. 
 * @version: 2.2
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-05-18 16:51:10
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-11-16 07:45:48
 */

#include <string>
#include <vector>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

#include "VideoPipeline.h"
#include "VideoAnalyzer.h"

static GMainLoop* g_main_loop = NULL;

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

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    Json::Reader reader;
    Json::Value root;
    std::ifstream in(FLAGS_config_path, std::ios::binary);
    reader.parse(in, root);

    VideoPipelineConfig m_vpConfig;
    m_vpConfig.url = root["pipeline-config"]["camera-url"].asString();
    m_vpConfig.cameraID = root["pipeline-config"]["camera-id"].asInt();
    m_vpConfig.streamWidth = root["pipeline-config"]["stream-width"].asInt();
    m_vpConfig.streamHeight = root["pipeline-config"]["stream-height"].asInt();
    m_vpConfig.streamFramerateN = root["pipeline-config"]["fps-n"].asInt();
    m_vpConfig.streamFramerateD = root["pipeline-config"]["fps-d"].asInt();
    m_vpConfig.convertFormat = root["pipeline-config"]["output-format"].asString();
    m_vpConfig.isDropBuffer = true;
    m_vpConfig.isSync = false;
    VideoPipeline* m_vp;
    VideoAnalyzer* m_va;
    std::shared_ptr<SafetyQueue<cv::Mat>> imageQueue = std::make_shared<SafetyQueue<cv::Mat>>();

    gst_init(&argc, &argv);

    if (!(g_main_loop = g_main_loop_new(NULL, FALSE))) {
        LOG_ERROR("Failed to new a object with type GMainLoop");
        goto exit;
    }

    m_vp = new VideoPipeline(m_vpConfig);

    if (!m_vp->Create()) {
        LOG_ERROR("Pipeline Create failed: lack of elements");
        goto exit;
    }
    m_vp->SetUserData(imageQueue);
    m_vp->Start();

    m_va = new VideoAnalyzer();
    if (!m_va->Init(root["model-configs"], root["mqtt-config"])) {
        LOG_ERROR("VideoAnalyzer Init failed!");
        goto exit;
    }
    m_va->SetUserData(imageQueue);
    m_va->Start();

    g_main_loop_run(g_main_loop);

exit:
    if (g_main_loop) g_main_loop_unref(g_main_loop);

    if (m_vp) {
        delete m_vp;
        m_vp = NULL;
    }

    google::ShutDownCommandLineFlags();
    return 0;
}
