/*
 * @Description: Inference SDK based on SNPE.
 * @version: 1.1
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-05-18 09:48:36
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2023-03-04 07:18:49
 */


#include "SNPETask.h"

namespace snpetask{

static size_t calcSizeFromDims(const size_t* dims, size_t rank, size_t elementSize)
{
    if (rank == 0) return 0;
    size_t size = elementSize;
    while (rank--) {
        size *= *dims;
        dims++;
    }
    return size;
}

static void createUserBuffer(Snpe_UserBufferMap_Handle_t userBufferMapHandle,
                      std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                      std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackedBuffersHandle,
                      Snpe_TensorShape_Handle_t bufferShapeHandle,
                      const char* name,
                      size_t bufferElementSize)
{
    // Calculate the stride based on buffer strides, assuming tightly packed.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    // Note: Buffer stride is usually known and does not need to be calculated.
    std::vector<size_t> strides(Snpe_TensorShape_Rank(bufferShapeHandle));
    strides[strides.size() - 1] = sizeof(float);
    size_t stride = strides[strides.size() - 1];
    for (size_t i = Snpe_TensorShape_Rank(bufferShapeHandle) - 1; i > 0; i--)
    {
        stride *= Snpe_TensorShape_At(bufferShapeHandle, i);
        strides[i - 1] = stride;
    }
    Snpe_TensorShape_Handle_t stridesHandle = Snpe_TensorShape_CreateDimsSize(strides.data(), Snpe_TensorShape_Rank(bufferShapeHandle));
    size_t bufSize = calcSizeFromDims(Snpe_TensorShape_GetDimensions(bufferShapeHandle), Snpe_TensorShape_Rank(bufferShapeHandle), bufferElementSize);
    LOG_INFO("Create [{}] buffer size: {}.", name, bufSize);
    // set the buffer encoding type
    Snpe_UserBufferEncoding_Handle_t userBufferEncodingFloatHandle = Snpe_UserBufferEncodingFloat_Create();
    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));
    // create SNPE user buffer from the user-backed buffer
    snpeUserBackedBuffersHandle.push_back(Snpe_Util_CreateUserBuffer(applicationBuffers.at(name).data(),
                                                  bufSize,
                                                  stridesHandle,
                                                  userBufferEncodingFloatHandle));
    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    Snpe_UserBufferMap_Add(userBufferMapHandle, name, snpeUserBackedBuffersHandle.back());
    Snpe_UserBufferEncodingFloat_Delete(userBufferEncodingFloatHandle);
}

SNPETask::SNPETask()
{
    Snpe_DlVersion_Handle_t versionHandle = Snpe_Util_GetLibraryVersion();
    LOG_INFO("Using SNPE: {}", Snpe_DlVersion_ToString(versionHandle));
    Snpe_DlVersion_Delete(versionHandle);

    m_container = nullptr;
    m_snpe = nullptr;
    m_runtimeList = nullptr;
    m_outputLayers = nullptr;
    m_inputUserBufferMap = nullptr;
    m_outputUserBufferMap = nullptr;
}

SNPETask::~SNPETask()
{

}

bool SNPETask::init(const std::string& model_path, const runtime_t runtime)
{
    switch (runtime) {
        case CPU:
            m_runtime = SNPE_RUNTIME_CPU;
            break;
        case GPU:
            m_runtime = SNPE_RUNTIME_GPU;
            break;
        case GPU_FLOAT16:
            m_runtime = SNPE_RUNTIME_GPU_FLOAT16;
            break;
        case DSP:
            m_runtime = SNPE_RUNTIME_DSP;
            break;
        case DSP_FIXED8:
            m_runtime = SNPE_RUNTIME_DSP_FIXED8_TF;
            break;
        case AIP:
            m_runtime = SNPE_RUNTIME_AIP_FIXED8_TF;
            break;
        default:
            m_runtime = SNPE_RUNTIME_CPU;
            break;
    }

    if (!Snpe_Util_IsRuntimeAvailable(m_runtime)) {
        LOG_ERROR("Selected runtime not supported. Falling back to CPU.");
        m_runtime = SNPE_RUNTIME_CPU;
    }

    m_container = Snpe_DlContainer_Open(model_path.c_str());
    Snpe_SNPEBuilder_Handle_t snpeBuilderHandle = Snpe_SNPEBuilder_Create(m_container);
    Snpe_PerformanceProfile_t profile = SNPE_PERFORMANCE_PROFILE_BURST;
    if (nullptr == m_runtimeList) m_runtimeList = Snpe_RuntimeList_Create();
    Snpe_RuntimeList_Add(m_runtimeList, m_runtime);
    Snpe_RuntimeList_Add(m_runtimeList, SNPE_RUNTIME_CPU);
    Snpe_SNPEBuilder_SetRuntimeProcessorOrder(snpeBuilderHandle, m_runtimeList);
    if (Snpe_SNPEBuilder_SetOutputLayers(snpeBuilderHandle, m_outputLayers)) {
        LOG_ERROR("Snpe_SNPEBuilder_SetOutputLayers failed: {}", Snpe_ErrorCode_GetLastErrorString());
        return false;
    }
    Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(snpeBuilderHandle, true);
    Snpe_SNPEBuilder_SetPerformanceProfile(snpeBuilderHandle, profile);
    m_snpe = Snpe_SNPEBuilder_Build(snpeBuilderHandle);
    if (nullptr == m_snpe) {
        const char* errStr = Snpe_ErrorCode_GetLastErrorString();
        LOG_ERROR("SNPE build failed: {}", errStr);
        return false;
    }

    // get input tensor names of the network that need to be populated
    Snpe_StringList_Handle_t inputNamesHandle = Snpe_SNPE_GetInputTensorNames(m_snpe);
    if (nullptr == inputNamesHandle) throw std::runtime_error("Error obtaining input tensor names");
    assert(Snpe_StringList_Size(inputNamesHandle) > 0);

    // create SNPE user buffers for each application storage buffer
    if (nullptr == m_inputUserBufferMap) m_inputUserBufferMap = Snpe_UserBufferMap_Create();
    for (size_t i = 0; i < Snpe_StringList_Size(inputNamesHandle); ++i) {
        const char* name = Snpe_StringList_At(inputNamesHandle, i);
        // get attributes of buffer by name
        auto bufferAttributesOptHandle = Snpe_SNPE_GetInputOutputBufferAttributes(m_snpe, name);
        if (nullptr == bufferAttributesOptHandle) {
            LOG_ERROR("Error obtaining attributes for input tensor: {}", name);
            return false;
        }

        auto bufferShapeHandle = Snpe_IBufferAttributes_GetDims(bufferAttributesOptHandle);
        std::vector<size_t> tensorShape;
        for (size_t j = 0; j < Snpe_TensorShape_Rank(bufferShapeHandle); j++) {
            tensorShape.push_back(Snpe_TensorShape_At(bufferShapeHandle, j));
        }
        m_inputShapes.emplace(name, tensorShape);

        // size_t bufferElementSize = Snpe_IBufferAttributes_GetElementSize(bufferAttributesOptHandle);
        createUserBuffer(m_inputUserBufferMap, m_inputTensors, m_inputUserBuffers, bufferShapeHandle, name, sizeof(float));

        Snpe_IBufferAttributes_Delete(bufferAttributesOptHandle);
        Snpe_TensorShape_Delete(bufferShapeHandle);
    }
    Snpe_StringList_Delete(inputNamesHandle);

    // get output tensor names of the network that need to be populated
    if (nullptr == m_outputUserBufferMap) m_outputUserBufferMap = Snpe_UserBufferMap_Create();
    Snpe_StringList_Handle_t outputNamesHandle = Snpe_SNPE_GetOutputTensorNames(m_snpe);
    if (nullptr == outputNamesHandle) throw std::runtime_error("Error obtaining input tensor names");
    assert(Snpe_StringList_Size(outputNamesHandle) > 0);

    // create SNPE user buffers for each application storage buffer
    for (size_t i = 0; i < Snpe_StringList_Size(outputNamesHandle); ++i) {
        const char* name = Snpe_StringList_At(outputNamesHandle, i);
        // get attributes of buffer by name
        auto bufferAttributesOptHandle = Snpe_SNPE_GetInputOutputBufferAttributes(m_snpe, name);
        if (!bufferAttributesOptHandle) {
            LOG_ERROR("Error obtaining attributes for input tensor: {}", name);
            return false;
        }

        auto bufferShapeHandle = Snpe_IBufferAttributes_GetDims(bufferAttributesOptHandle);
        std::vector<size_t> tensorShape;
        for (size_t j = 0; j < Snpe_TensorShape_Rank(bufferShapeHandle); j++) {
            tensorShape.push_back(Snpe_TensorShape_At(bufferShapeHandle, j));
        }
        m_outputShapes.emplace(name, tensorShape);

        // size_t bufferElementSize = Snpe_IBufferAttributes_GetElementSize(bufferAttributesOptHandle);
        createUserBuffer(m_outputUserBufferMap, m_outputTensors, m_outputUserBuffers, bufferShapeHandle, name, sizeof(float));

        Snpe_IBufferAttributes_Delete(bufferAttributesOptHandle);
        Snpe_TensorShape_Delete(bufferShapeHandle);
    }

    Snpe_StringList_Delete(outputNamesHandle);
    Snpe_SNPEBuilder_Delete(snpeBuilderHandle);

    m_isInit = true;

    return true;
}

bool SNPETask::deInit()
{
    if (nullptr != m_runtimeList) Snpe_RuntimeList_Delete(m_runtimeList);
    for (auto& input : m_inputUserBuffers) {
        if (nullptr != input) Snpe_IUserBuffer_Delete(input);
    }
    m_inputUserBuffers.clear();
    for (auto& output : m_outputUserBuffers) {
        if (nullptr != output) Snpe_IUserBuffer_Delete(output);
    }
    m_outputUserBuffers.clear();

    if (nullptr != m_inputUserBufferMap) Snpe_UserBufferMap_Delete(m_inputUserBufferMap);
    if (nullptr != m_outputUserBufferMap) Snpe_UserBufferMap_Delete(m_outputUserBufferMap);

    if (nullptr != m_snpe) Snpe_SNPE_Delete(m_snpe);
    if (nullptr != m_container) Snpe_DlContainer_Delete(m_container);

    return true;
}

bool SNPETask::setOutputLayers(std::vector<std::string>& outputLayers)
{
    if (nullptr == m_outputLayers) m_outputLayers = Snpe_StringList_Create();

    for (size_t i = 0; i < outputLayers.size(); i ++) {
        if (SNPE_SUCCESS != Snpe_StringList_Append(m_outputLayers, outputLayers[i].c_str())) {
            LOG_ERROR("Append output name: {} failed: {}.", outputLayers[i], Snpe_ErrorCode_GetLastErrorString());
            return false;
        }
    }

    return true;
}

std::vector<size_t> SNPETask::getInputShape(const std::string& name)
{
    if (isInit()) {
        if (m_inputShapes.find(name) != m_inputShapes.end()) {
            return m_inputShapes.at(name);
        }
        LOG_ERROR("Can't find any input layer named {}", name.c_str());
        return {};
    } else {
        LOG_ERROR("The getInputShape() needs to be called after SNPETask is initialized!");
        return {};
    }
}

std::vector<size_t> SNPETask::getOutputShape(const std::string& name)
{
    if (isInit()) {
        if (m_outputShapes.find(name) != m_outputShapes.end()) {
            return m_outputShapes.at(name);
        }
        LOG_ERROR("Can't find any ouput layer named {}", name.c_str());
        return {};
    } else {
        LOG_ERROR("The getOutputShape() needs to be called after SNPETask is initialized!");
        return {};
    }
}

float* SNPETask::getInputTensor(const std::string& name)
{
    if (isInit()) {
        if (m_inputTensors.find(name) != m_inputTensors.end()) {
            return reinterpret_cast<float*>(m_inputTensors.at(name).data());
        }
        LOG_ERROR("Can't find any input tensor named {}", name.c_str());
        return nullptr;
    } else {
        LOG_ERROR("The getInputTensor() needs to be called after SNPETask is initialized!");
        return nullptr;
    }
}

float* SNPETask::getOutputTensor(const std::string& name)
{
    if (isInit()) {
        if (m_outputTensors.find(name) != m_outputTensors.end()) {
            return reinterpret_cast<float*>(m_outputTensors.at(name).data());
        }
        LOG_ERROR("Can't find any output tensor named {}", name.c_str());
        return nullptr;
    } else {
        LOG_ERROR("The getOutputTensor() needs to be called after SNPETask is initialized!");
        return nullptr;
    }
}

bool SNPETask::execute()
{
    if (SNPE_SUCCESS != Snpe_SNPE_ExecuteUserBuffers(m_snpe, m_inputUserBufferMap, m_outputUserBufferMap)) {
        LOG_ERROR("SNPETask execute failed: {}", Snpe_ErrorCode_GetLastErrorString());
        return false;
    }

    return true;
}

}   // namespace snpetask
