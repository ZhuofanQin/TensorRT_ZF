#ifndef OneHot1DPlugin_H
#define OneHot1DPlugin_H

#include "NvInferPlugin.h"
#include "common/plugin.h"
#include <string>
#include <cuda_runtime.h>

#define CLASS_NUM (10)

namespace nvinfer1
{
namespace plugin
{
class OneHot1DPlugin : public IPluginV2Ext
{
public:
    OneHot1DPlugin();

    OneHot1DPlugin(const void* data, size_t length);

    OneHot1DPlugin(int _class_num, int _length);

    const char* getPluginType () const noexcept override;

    const char* getPluginVersion () const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
        noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;
        
    IPluginV2Ext* clone() const noexcept override;

private:
    int32_t class_num, length;
    std::string mNamespace;
};

class OneHot1DPluginCreator : public nvinfer1::IPluginCreator
{
public:
    OneHot1DPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif