#include "oneHot1DPlugin.h"

using nvinfer1::plugin::OneHot1DPlugin;

template <typename T>
__global__ void oneHot1D_kernel(T* input, T* output)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = (input[threadIdx.x] == blockIdx.x);
}

int32_t OneHot1DPlugin::enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept 
{
    unsigned int num_threads = length;
    unsigned int num_blocks = class_num;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    PLUGIN_ASSERT(num_threads <= prop.maxThreadsPerBlock);

    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(num_threads, 1);
    oneHot1D_kernel<<<grid_dim, block_dim, 0, stream>>>((int*) inputs[0], (int*) outputs[0]);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}