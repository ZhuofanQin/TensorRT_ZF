#include "WindowProcessPlugin.h"

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
using nvinfer1::plugin::WindowProcessPlugin;

static const char* WINDOW_PROCESS_PLUGIN_VERSION{"1"};
static const char* WINDOW_PROCESS_PLUGIN_NAME{"WindowProcessPlugin"};

WindowProcessPlugin::WindowProcessPlugin() {}

WindowProcessPlugin::WindowProcessPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    H = readFromBuffer<int32_t>(d);
    W = readFromBuffer<int32_t>(d);
    C = readFromBuffer<int32_t>(d);
    shift_size = readFromBuffer<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

const char* WindowProcessPlugin::getPluginType () const noexcept
{
    return WINDOW_PROCESS_PLUGIN_NAME;
}

const char* WindowProcessPlugin::getPluginVersion () const noexcept
{
    return WINDOW_PROCESS_PLUGIN_VERSION;
}

int32_t WindowProcessPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims WindowProcessPlugin::getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputDims == 1);
        PLUGIN_ASSERT(index == 0);
        PLUGIN_ASSERT(inputs[0].nbDims == 4);
        return inputs[0];
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return Dims{};
}

bool WindowProcessPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

void WindowProcessPlugin::configureWithFormat (Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept
{
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    C = inputDims[0].d[3];
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inputDims[0].nbDims == 4);
    PLUGIN_ASSERT(inputDims[0].d[0] == 1);
    PLUGIN_ASSERT(!((H % WINDOW_SIZE) || (W % WINDOW_SIZE)));
    shift_size = WINDOW_SIZE / 2;
}

int32_t WindowProcessPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void WindowProcessPlugin::terminate() noexcept {};

size_t WindowProcessPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return 0;
}

size_t WindowProcessPlugin::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 4;
}

void WindowProcessPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    writeToBuffer<size_t>(d, H);
    writeToBuffer<size_t>(d, W);
    writeToBuffer<size_t>(d, C);
    writeToBuffer<size_t>(d, shift_size);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void WindowProcessPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2* WindowProcessPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new WindowProcessPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void WindowProcessPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* WindowProcessPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



using nvinfer1::plugin::WindowProcessPluginCreator;

PluginFieldCollection WindowProcessPluginCreator::mFC{};

WindowProcessPluginCreator::WindowProcessPluginCreator()
{
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

const char* WindowProcessPluginCreator::getPluginName() const noexcept
{
    return WINDOW_PROCESS_PLUGIN_NAME;
}

const char* WindowProcessPluginCreator::getPluginVersion() const noexcept
{
    return WINDOW_PROCESS_PLUGIN_VERSION;
}

const PluginFieldCollection* WindowProcessPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WindowProcessPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        auto* plugin = new WindowProcessPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WindowProcessPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        return new WindowProcessPlugin(serialData, serialLength);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void WindowProcessPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* WindowProcessPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}