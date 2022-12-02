#ifndef WINDOWPROCESSPLUGIN_H
#define WINDOWPROCESSPLUGIN_H

#include "NvInferPlugin.h"
#include "common/plugin.h"
#include <string>
#include <cuda_runtime.h>

#define WINDOW_SIZE (7)

namespace nvinfer1
{
namespace plugin
{
class WindowProcessPlugin : public IPluginV2
{
public:
    WindowProcessPlugin();

    WindowProcessPlugin(const void* data, size_t length);

    const char* getPluginType () const noexcept override;

    const char* getPluginVersion () const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    void configureWithFormat (Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;

    IPluginV2* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    int32_t H, W, C, shift_size;
    std::string mNamespace;
};

class WindowProcessPluginCreator : public nvinfer1::IPluginCreator
{
public:
    WindowProcessPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif