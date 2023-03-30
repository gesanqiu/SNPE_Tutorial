/*
 * @Description: Object detection algorithm handler.
 * @version: 2.1
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-05-17 20:27:51
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2023-02-24 15:14:36
 */

#ifndef __YOLOV5S_IMPL_H__
#define __YOLOV5S_IMPL_H__

#include <vector>
#include <string>
#if defined(WIN32) || defined(_WIN32)
#include <time.h>
#else
#include <unistd.h>
#endif
#include <memory>

#include "SNPETask.h"
#include "YOLOv5s.h"

namespace yolov5 {

class ObjectDetectionImpl {
public:
    ObjectDetectionImpl();
    ~ObjectDetectionImpl();
    bool Detect(const cv::Mat& image, std::vector<ObjectData>& results);
    bool Initialize(const ObjectDetectionConfig& config);
    bool DeInitialize();

    bool SetScoreThresh(const float& conf_thresh, const float& nms_thresh = 0.5) noexcept {
        this->m_nmsThresh  = nms_thresh;
        this->m_confThresh = conf_thresh;
        return true;
    }

    bool SetROI(const cv::Rect& roi) {
        this->m_roi = roi;
        return true;
    }

    bool RegisterPreProcess(pre_process_t func) {
        this->m_preProcess = func;
        m_isRegisteredPreProcess = true;
        return true;
    }

    bool RegisterPreProcess(post_process_t func) {
        this->m_postProcess = func;
        m_isRegisteredPostProcess = true;
        return true;
    }


    bool IsInitialized() const {
        return m_isInit;
    }

    static std::vector<ObjectData> nms(std::vector<ObjectData> winList, const float& nms_thresh) {
        if (winList.empty()) {
            return winList;
        }

        std::sort(winList.begin(), winList.end(), [] (const ObjectData& left, const ObjectData& right) {
            if (left.confidence > right.confidence) {
                return true;
            } else {
                return false;
            }
        });

        std::vector<bool> flag(winList.size(), false);
        for (int i = 0; i < winList.size(); i++) {
            if (flag[i]) {
                continue;
            }

            for (int j = i + 1; j < winList.size(); j++) {
                if (calcIoU(winList[i].bbox, winList[j].bbox) > nms_thresh) {
                    flag[j] = true;
                }
            }
        }

        std::vector<ObjectData> ret;
        for (int i = 0; i < winList.size(); i++) {
            if (!flag[i])
                ret.push_back(winList[i]);
        }

        return ret;
    }

private:
    bool m_isInit = false;
    bool m_isRegisteredPreProcess = false;
    bool m_isRegisteredPostProcess = false;

    bool PreProcess(const cv::Mat& frame);
    bool PostProcess(std::vector<ObjectData>& results, int64_t time);

    pre_process_t m_preProcess;
    post_process_t m_postProcess;

    std::unique_ptr<snpetask::SNPETask> m_task;
    std::vector<std::string> m_inputLayers;
    std::vector<std::string> m_outputLayers;
    std::vector<std::string> m_outputTensors;

    int m_labels;
    int m_grids;
    float* m_output;

    cv::Rect m_roi = {0, 0, 0, 0};
    uint32_t m_minBoxBorder = 16;
    float m_nmsThresh = 0.5f;
    float m_confThresh = 0.5f;
    float m_scale;
    int m_xOffset, m_yOffset;
};

} // namespace yolov5

#endif // __YOLOV5S_IMPL_H__
