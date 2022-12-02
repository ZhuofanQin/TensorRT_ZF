#include "WindowProcessPlugin.h"

using nvinfer1::plugin::WindowProcessPlugin;

template <typename T>
__global__ void slide_window_kernel(T* inputs, T* outputs, int32_t shift_size)
{
    int32_t index = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
    int32_t offset = blockDim.x * (gridDim.x * ((blockIdx.y + shift_size) % gridDim.y) + ((blockIdx.x + shift_size) % gridDim.x)) + threadIdx.x; 
    outputs[index] = inputs[offset];
}

int32_t WindowProcessPlugin::enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept 
{
    dim3 grid_dim(H, W);
    dim3 block_dim(C);
    slide_window_kernel<<<grid_dim, block_dim, 0, stream>>>((float*) inputs[0], (float*) outputs[0], shift_size);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}