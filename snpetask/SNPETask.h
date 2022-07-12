/*
 * @Description: Inference SDK based on SNPE. 
 * @version: 1.1
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-05-17 20:28:01
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-07-12 08:58:26
 */

#ifndef __SNPE_TASK_H__
#define __SNPE_TASK_H__

#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "DlSystem/TensorShape.hpp"
#include "DlContainer/IDlContainer.hpp"

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

    std::unique_ptr<zdl::DlContainer::IDlContainer> m_container;
    std::unique_ptr<zdl::SNPE::SNPE> m_snpe;
    zdl::DlSystem::Runtime_t m_runtime;
    zdl::DlSystem::StringList m_outputLayers;

    std::map<std::string, std::vector<size_t> > m_inputShapes;
    std::map<std::string, std::vector<size_t> > m_outputShapes;

    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer> > m_inputUserBuffers;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer> > m_outputUserBuffers;
    zdl::DlSystem::UserBufferMap m_inputUserBufferMap;
    zdl::DlSystem::UserBufferMap m_outputUserBufferMap;
    std::unordered_map<std::string, float*> m_inputTensors;
    std::unordered_map<std::string, float*> m_outputTensors;
};

}    // namespace snpetask

#endif    // __SNPE_TASK_H__