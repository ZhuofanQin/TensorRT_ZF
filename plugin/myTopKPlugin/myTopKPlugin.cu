#include "myTopKPlugin.h"
#include <float.h>
#define maxThreadsPerBlock (1024)

using nvinfer1::plugin::MyTopKPlugin;

template <typename T>
__device__ void swap(T &a, T &b){
    T t = a;
    a = b;
    b = t;
}

template <typename T>
__global__ void bitonic_sort_stage1(T* arr, int32_t* ind, int length){
    __shared__ T shared_arr[maxThreadsPerBlock];
    __shared__ int32_t shared_ind[maxThreadsPerBlock];
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < length)
        shared_arr[threadIdx.x] = arr[tid];
    else
        shared_arr[threadIdx.x] = FLT_MIN;
    shared_ind[threadIdx.x] = tid;
    __syncthreads();
    
    for(unsigned int i=2; i<=blockDim.x; i<<=1){
        for(unsigned int j=i>>1; j>0; j>>=1){
            unsigned int tid_comp = threadIdx.x ^ j;
            if(tid_comp > threadIdx.x){
                if((threadIdx.x & i)==0){
                    if(shared_arr[threadIdx.x]<shared_arr[tid_comp]){
                        swap(shared_arr[threadIdx.x],shared_arr[tid_comp]);
                        swap(shared_ind[threadIdx.x],shared_ind[tid_comp]);
                    }
                }
                else{
                    if(shared_arr[threadIdx.x]>shared_arr[tid_comp]){
                        swap(shared_arr[threadIdx.x],shared_arr[tid_comp]);
                        swap(shared_ind[threadIdx.x],shared_ind[tid_comp]);
                    }
                }
            }
            __syncthreads();
        }
    }
    if(blockIdx.x%2){
        arr[tid] = shared_arr[blockDim.x - threadIdx.x - 1];
        ind[tid] = shared_ind[blockDim.x - threadIdx.x - 1];
    }
    else{
        arr[tid] = shared_arr[threadIdx.x];
        ind[tid] = shared_ind[threadIdx.x];
    }
}

template <typename T>
__global__ void bitonic_sort_stage2(T* arr, int32_t* ind, int i, int j){

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid_comp = tid ^ j;
    if(tid_comp > tid){
        if((tid & i)==0){
            if(arr[tid]<arr[tid_comp]){
                swap(arr[tid],arr[tid_comp]);
                swap(ind[tid],ind[tid_comp]);
            }
        }
        else{
            if(arr[tid]>arr[tid_comp]){
                swap(arr[tid],arr[tid_comp]);
                swap(ind[tid],ind[tid_comp]);
            }
        }
    }
}

int32_t MyTopKPlugin::enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept 
{
    float const* const* in_datas = reinterpret_cast<float const* const*>(inputs);
    float* const* out_datas = reinterpret_cast<float* const*>(outputs);

    unsigned int extended_num;
    for(extended_num = 1; extended_num < length; extended_num<<=1){;}
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // unsigned int num_threads = prop.maxThreadsPerBlock;
    unsigned int num_threads = maxThreadsPerBlock;
    unsigned int num_blocks = extended_num / num_threads;
    if(!num_blocks)
    {
        num_threads = extended_num;
        num_blocks = 1;
    }
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(num_threads, 1);
    float* ptr_arr;
    int32_t* ptr_ind;
    cudaMalloc((void**)&ptr_arr, extended_num * sizeof(float));
    cudaMalloc((void**)&ptr_ind, extended_num * sizeof(int32_t));
    cudaMemcpy(ptr_arr, in_datas[0], length * sizeof(float), cudaMemcpyHostToDevice);
    bitonic_sort_stage1<<<grid_dim, block_dim, 0, stream>>>(ptr_arr, ptr_ind, length);
    for(unsigned int i= num_threads<<1; i<=extended_num; i<<=1){
        for(unsigned int j=i>>1; j>0; j>>=1){
            bitonic_sort_stage2<<<grid_dim, block_dim, 0, stream>>>(ptr_arr, ptr_ind, i, j);
        }
    }
    cudaMemcpy(out_datas[0], ptr_arr, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(out_datas[1], ptr_ind, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaFree(ptr_arr);
    cudaFree(ptr_ind);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}