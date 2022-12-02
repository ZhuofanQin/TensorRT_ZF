#include "myTopKPlugin.h"
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
using nvinfer1::plugin::MyTopKPlugin;

static const char* MY_TOPK_PLUGIN_VERSION{"1"};
static const char* MY_TOPK_PLUGIN_NAME{"MyTopKPlugin_TRT"};

MyTopKPlugin::MyTopKPlugin() {}

MyTopKPlugin::MyTopKPlugin(const void* data, size_t data_length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    k = readFromBuffer<int32_t>(d);
    length = readFromBuffer<int32_t>(d);
    PLUGIN_VALIDATE(d == a + data_length);
}

MyTopKPlugin::MyTopKPlugin(int _k, int _length) : k(_k), length(_length) {}

const char* MyTopKPlugin::getPluginType () const noexcept
{
    return MY_TOPK_PLUGIN_NAME;
}

const char* MyTopKPlugin::getPluginVersion () const noexcept
{
    return MY_TOPK_PLUGIN_VERSION;
}

int32_t MyTopKPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims MyTopKPlugin::getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept
{
    try
    {
        PLUGIN_ASSERT(index == 0);
        PLUGIN_ASSERT(nbInputDims == 1);
        PLUGIN_ASSERT(inputs[0].nbDims == 1);
        Dims outputDims;
        outputDims.nbDims = 1;
        outputDims.d[0] = MY_K;
        return outputDims;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return Dims{};
}

bool MyTopKPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kFLOAT || type == DataType::kINT32) && format == PluginFormat::kLINEAR);
}

// void MyTopKPlugin::configureWithFormat (Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept
// {
//     k = MY_K;
//     length = inputDims[0].d[0];
//     PLUGIN_ASSERT(nbOutputs == 1);
//     PLUGIN_ASSERT(inputDims[0].nbDims == 1);
//     PLUGIN_ASSERT(inputDims[0].d[0] == 324000);
//     PLUGIN_ASSERT(k <= length);
// }

void MyTopKPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    k = outputDims[0].d[0];
    length = inputDims[0].d[0];
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inputDims[0].nbDims == 1);
    PLUGIN_ASSERT(k == MY_K);
    PLUGIN_ASSERT(k <= length);
    // PLUGIN_ASSERT(length == 324000);
}

int32_t MyTopKPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void MyTopKPlugin::terminate() noexcept {};

size_t MyTopKPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return 0;
}

size_t MyTopKPlugin::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 2;
}

void MyTopKPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    writeToBuffer<int>(d, k);
    writeToBuffer<int>(d, length);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void MyTopKPlugin::destroy() noexcept
{
    delete this;
}

bool MyTopKPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    noexcept
{
    return false;
}

bool MyTopKPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

nvinfer1::DataType MyTopKPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

IPluginV2Ext* MyTopKPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new MyTopKPlugin(k, length);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void MyTopKPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* MyTopKPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



using nvinfer1::plugin::MyTopKPluginCreator;

PluginFieldCollection MyTopKPluginCreator::mFC{};

MyTopKPluginCreator::MyTopKPluginCreator()
{
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

const char* MyTopKPluginCreator::getPluginName() const noexcept
{
    return MY_TOPK_PLUGIN_NAME;
}

const char* MyTopKPluginCreator::getPluginVersion() const noexcept
{
    return MY_TOPK_PLUGIN_VERSION;
}

const PluginFieldCollection* MyTopKPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* MyTopKPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        auto* plugin = new MyTopKPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* MyTopKPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        return new MyTopKPlugin(serialData, serialLength);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void MyTopKPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* MyTopKPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}