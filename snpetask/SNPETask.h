/*
 * @Description: Inference SDK based on SNPE. 
 * @version: 1.1
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-05-17 20:28:01
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2023-02-23 08:22:00
 */

#ifndef __SNPE_TASK_H__
#define __SNPE_TASK_H__

#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>

#include "SNPE/SNPE.h"
#include "SNPE/SNPEUtil.h"
#include "SNPE/SNPEBuilder.h"
#include "DlSystem/DlVersion.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/DlError.h"
#include "DlSystem/TensorShape.h"
#include "DlContainer/DlContainer.h"

#include "utils.h"

namespace snpetask {

class SNPETask {
public:
    SNPETask();
    ~SNPETask();

    bool init(const std::string& model_path, const runtime_t runtime);
    bool deInit();
    bool setOutputLayers(std::vector<std::string>& outputLayers);

    std::vector<size_t> getInputShape(const std::string& name);
    std::vector<size_t> getOutputShape(const std::string& name);

    float* getInputTensor(const std::string& name);
    float* getOutputTensor(const std::string& name);

    bool isInit() {
        return m_isInit;
    }

    bool execute();

private:
    bool m_isInit = false;

    Snpe_DlContainer_Handle_t m_container;
    Snpe_SNPE_Handle_t m_snpe;
    Snpe_Runtime_t m_runtime;
    Snpe_RuntimeList_Handle_t m_runtimeList;
    Snpe_StringList_Handle_t m_outputLayers;

    std::map<std::string, std::vector<size_t> > m_inputShapes;
    std::map<std::string, std::vector<size_t> > m_outputShapes;

    std::vector<Snpe_IUserBuffer_Handle_t> m_inputUserBuffers;
    std::vector<Snpe_IUserBuffer_Handle_t> m_outputUserBuffers;
    Snpe_UserBufferMap_Handle_t m_inputUserBufferMap;
    Snpe_UserBufferMap_Handle_t m_outputUserBufferMap;

    std::unordered_map<std::string, std::vector<uint8_t>> m_inputTensors;
    std::unordered_map<std::string, std::vector<uint8_t>> m_outputTensors;
};

}    // namespace snpetask

#endif    // __SNPE_TASK_H__