/*
 * @Description: Implementation of object detection algorithm handler.
 * @version: 2.1
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-05-17 20:28:01
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2023-02-24 15:16:01
 */

#include <math.h>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "YOLOv5sImpl.h"

namespace yolov5 {

ObjectDetectionImpl::ObjectDetectionImpl() : m_task(nullptr) {

}

ObjectDetectionImpl::~ObjectDetectionImpl() {
    DeInitialize();
}

bool ObjectDetectionImpl::Initialize(const ObjectDetectionConfig& config)
{
    m_task = std::move(std::unique_ptr<snpetask::SNPETask>(new snpetask::SNPETask()));

    m_inputLayers = config.inputLayers;
    m_outputLayers = config.outputLayers;
    m_outputTensors = config.outputTensors;
    m_labels = config.labels;
    m_grids = config.grids;

    m_task->setOutputLayers(m_outputLayers);

    if (!m_task->init(config.model_path, config.runtime)) {
        LOG_ERROR("Can't init snpetask instance.");
        return false;
    }

    m_output = new float[m_grids * m_labels];

    m_isInit = true;
    return true;
}

bool ObjectDetectionImpl::DeInitialize()
{
    if (m_task) {
        m_task->deInit();
        m_task.reset(nullptr);
    }

    if (m_output) {
        delete[] m_output;
        m_output = nullptr;
    }

    m_isInit = false;
    return true;
}

bool ObjectDetectionImpl::PreProcess(const cv::Mat& image)
{
    auto inputShape = m_task->getInputShape(m_inputLayers[0]);

    size_t batch = inputShape[0];
    size_t inputHeight = inputShape[1];
    size_t inputWidth = inputShape[2];
    size_t channel = inputShape[3];

    if (m_task->getInputTensor(m_inputLayers[0]) == nullptr) {
        LOG_ERROR("Empty input tensor");
        return false;
    }

    cv::Mat input(inputHeight, inputWidth, CV_32FC3, m_task->getInputTensor(m_inputLayers[0]));

    if (image.empty()) {
        LOG_ERROR("Invalid image!");
        return false;
    }

    int imgWidth = image.cols;
    int imgHeight = image.rows;

    m_scale = std::min(inputHeight /(float)imgHeight, inputWidth / (float)imgWidth);
    int scaledWidth = imgWidth * m_scale;
    int scaledHeight = imgHeight * m_scale;
    m_xOffset = (inputWidth - scaledWidth) / 2;
    m_yOffset = (inputHeight - scaledHeight) / 2;

    cv::Mat inputMat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat roiMat(inputMat, cv::Rect(m_xOffset, m_yOffset, scaledWidth, scaledHeight));
    cv::resize(image, roiMat, cv::Size(scaledWidth, scaledHeight), cv::INTER_LINEAR);

    inputMat.convertTo(input, CV_32FC3);
    input /= 255.0f;
}

bool ObjectDetectionImpl::Detect(const cv::Mat& image,
    std::vector<ObjectData>& results)
{

    if (m_roi.empty()) {
        if (m_isRegisteredPreProcess) m_preProcess(image); 
        else PreProcess(image);
    } else {
        auto roi_image = image(m_roi);
        if (m_isRegisteredPreProcess) m_preProcess(roi_image); 
        else PreProcess(roi_image);
    }

    int64_t start = GetTimeStamp_ms();
    if (!m_task->execute()) {
        LOG_ERROR("SNPETask execute failed.");
        return false;
    }

    if (m_isRegisteredPostProcess) m_postProcess(results);
    else PostProcess(results, GetTimeStamp_ms() - start);

    return true;
}

bool ObjectDetectionImpl::PostProcess(std::vector<ObjectData> &results, int64_t time)
{
    float strides[3] = {8, 16, 32};
    float anchorGrid[][6] = {
        {10, 13, 16, 30, 33, 23},       // 8*8
        {30, 61, 62, 45, 59, 119},      // 16*16
        {116, 90, 156, 198, 373, 326},  // 32*32
    };

    // copy all outputs to one array.
    // [80 * 80 * 3 * 85]----\
    // [40 * 40 * 3 * 85]--------> [25200 * 85]
    // [20 * 20 * 3 * 85]----/
    float* tmpOutput = m_output;
    for (size_t i = 0; i < 3; i++) {
        auto outputShape = m_task->getOutputShape(m_outputTensors[i]);
        const float *predOutput = m_task->getOutputTensor(m_outputTensors[i]);

        int batch = outputShape[0];
        int height = outputShape[1];
        int width = outputShape[2];
        int channel = outputShape[3];

        for (int j = 0; j < height; j++) {      // 80/40/20
            for (int k = 0; k < width; k++) {   // 80/40/20
                int anchorIdx = 0;
                for (int l = 0; l < 3; l++) {   // 3
                    for (int m = 0; m < channel / 3; m++) {     // 85
                        if (m < 2) {
                            float value = *predOutput;
                            float gridValue = m == 0 ? k : j;
                            *tmpOutput = (value * 2 - 0.5 + gridValue) * strides[i];
                        } else if (m < 4) {
                            float value = *predOutput;
                            *tmpOutput = value * value * 4 * anchorGrid[i][anchorIdx++];
                        } else {
                            *tmpOutput = *predOutput;
                        }
                        tmpOutput++;
                        predOutput++;
                    }
                }
            }
        }
    }
    
    std::vector<int> boxIndexs;
    std::vector<float> boxConfidences;
    std::vector<ObjectData> winList;

    for (int i = 0; i< m_grids; i++) {   // 共计MODEL_OUTPUT_GRIDS个grid要统计
        // 每个grid有m_labels个数据要处理
        // m_labels = 85 ----> [bbox_x, bbox_y, bbox_width, bbox_height, bbox_score, prob0, prob1,..., prob84]
        float boxConfidence = m_output[i * m_labels + 4];
        if (boxConfidence > 0.001) {
            boxIndexs.push_back(i);     // 先对bbox进行粗筛，并保存索引index
            boxConfidences.push_back(boxConfidence);
        }
    }

    float curMaxScore = 0.0f;
    float curMinScore = 0.0f;

    for (size_t i = 0; i < boxIndexs.size(); i++) {
        int curIdx = boxIndexs[i];
        float curBoxConfidence = boxConfidences[i];
        int max_idx = 5;
        for (int j = 6; j < m_labels; j++) {    // 处理目标分类概率即m_labels中的[prob0, prob1,..., prob84]
            if (m_output[curIdx * m_labels + j] > m_output[curIdx * m_labels + max_idx]) max_idx = j;
        }

        float score = curBoxConfidence * m_output[curIdx * m_labels + max_idx];
        if (score > m_confThresh) {
            ObjectData rect;
            rect.bbox.width = m_output[curIdx * m_labels + 2];
            rect.bbox.height = m_output[curIdx * m_labels + 3];
            rect.bbox.x = std::max(0, static_cast<int>(m_output[curIdx * m_labels] - rect.bbox.width / 2)) - m_xOffset;
            rect.bbox.y = std::max(0, static_cast<int>(m_output[curIdx * m_labels + 1] - rect.bbox.height / 2)) - m_yOffset;

            rect.bbox.width /= m_scale;
            rect.bbox.height /= m_scale;
            rect.bbox.x /= m_scale;
            rect.bbox.y /= m_scale;
            rect.confidence = score;
            rect.label = max_idx - 5;
            rect.time_cost = time;
            winList.push_back(rect);
        }
    }

    winList = nms(winList, m_nmsThresh);

    for (size_t i = 0; i < winList.size(); i++) {
        if (winList[i].bbox.width >= m_minBoxBorder || winList[i].bbox.height >= m_minBoxBorder) {
            if (!m_roi.empty()) {
                winList[i].bbox.x += m_roi.x;
                winList[i].bbox.y += m_roi.y;
            }
            results.push_back(winList[i]);
        }
    }

    return true;
}

} // namespace yolov5
