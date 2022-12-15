#include "oneHot1DPlugin.h"
#include <iostream>

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

using namespace nvinfer1;
using nvinfer1::plugin::OneHot1DPlugin;

static const char* MY_TOPK_PLUGIN_VERSION{"1"};
static const char* MY_TOPK_PLUGIN_NAME{"OneHot1DPlugin_TRT"};

OneHot1DPlugin::OneHot1DPlugin() {}

OneHot1DPlugin::OneHot1DPlugin(const void* data, size_t data_length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    class_num = readFromBuffer<int32_t>(d);
    length = readFromBuffer<int32_t>(d);
    PLUGIN_VALIDATE(d == a + data_length);
}

OneHot1DPlugin::OneHot1DPlugin(int _class_num, int _length) : class_num(_class_num), length(_length) {}

const char* OneHot1DPlugin::getPluginType () const noexcept
{
    return MY_TOPK_PLUGIN_NAME;
}

const char* OneHot1DPlugin::getPluginVersion () const noexcept
{
    return MY_TOPK_PLUGIN_VERSION;
}

int32_t OneHot1DPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims OneHot1DPlugin::getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept
{
    try
    {
        PLUGIN_ASSERT(index == 0);
        PLUGIN_ASSERT(nbInputDims == 1);
        PLUGIN_ASSERT(inputs[0].nbDims == 1);
        Dims outputDims;
        outputDims.nbDims = 2;
        outputDims.d[0] = CLASS_NUM;
        outputDims.d[1] = inputs[0].d[0];
        return outputDims;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return Dims{};
}

bool OneHot1DPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kINT32) && format == PluginFormat::kLINEAR);
}

// void OneHot1DPlugin::configureWithFormat (Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept
// {
//     k = MY_K;
//     length = inputDims[0].d[0];
//     PLUGIN_ASSERT(nbOutputs == 1);
//     PLUGIN_ASSERT(inputDims[0].nbDims == 1);
//     PLUGIN_ASSERT(inputDims[0].d[0] == 324000);
//     PLUGIN_ASSERT(k <= length);
// }

void OneHot1DPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    class_num = outputDims[0].d[0];
    length = inputDims[0].d[0];
    PLUGIN_ASSERT(class_num == CLASS_NUM);
    PLUGIN_ASSERT(inputDims[0].nbDims == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(outputDims[0].nbDims == 2);
    PLUGIN_ASSERT(outputDims[0].d[1] == length);
}

int32_t OneHot1DPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void OneHot1DPlugin::terminate() noexcept {};

size_t OneHot1DPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return 0;
}

size_t OneHot1DPlugin::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 2;
}

void OneHot1DPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    writeToBuffer<int32_t>(d, class_num);
    writeToBuffer<int32_t>(d, length);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void OneHot1DPlugin::destroy() noexcept
{
    delete this;
}

bool OneHot1DPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    noexcept
{
    return false;
}

bool OneHot1DPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

nvinfer1::DataType OneHot1DPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

IPluginV2Ext* OneHot1DPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new OneHot1DPlugin(class_num, length);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void OneHot1DPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* OneHot1DPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



using nvinfer1::plugin::OneHot1DPluginCreator;

PluginFieldCollection OneHot1DPluginCreator::mFC{};

OneHot1DPluginCreator::OneHot1DPluginCreator()
{
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

const char* OneHot1DPluginCreator::getPluginName() const noexcept
{
    return MY_TOPK_PLUGIN_NAME;
}

const char* OneHot1DPluginCreator::getPluginVersion() const noexcept
{
    return MY_TOPK_PLUGIN_VERSION;
}

const PluginFieldCollection* OneHot1DPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* OneHot1DPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        auto* plugin = new OneHot1DPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* OneHot1DPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        return new OneHot1DPlugin(serialData, serialLength);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void OneHot1DPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* OneHot1DPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}