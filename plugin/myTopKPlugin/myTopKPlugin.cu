#include "myTopKPlugin.h"

using nvinfer1::plugin::MyTopKPlugin;

template <typename T>
__device__ void swap(T &a, T &b){
    T t = a;
    a = b;
    b = t;
}

template <typename T>
__global__ void bitonic_sort_stage1(T* arr){
    extern __shared__ T shared_arr[];
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    shared_arr[threadIdx.x] = arr[tid];
    __syncthreads();
    
    for(unsigned int i=2; i<=blockDim.x; i<<=1){
        for(unsigned int j=i>>1; j>0; j>>=1){
            unsigned int tid_comp = threadIdx.x ^ j;
            if(tid_comp > threadIdx.x){
                if((threadIdx.x & i)==0){ //ascending
                    if(shared_arr[threadIdx.x]>shared_arr[tid_comp]){
                        swap(shared_arr[threadIdx.x],shared_arr[tid_comp]);
                    }
                }
                else{ //desending
                    if(shared_arr[threadIdx.x]<shared_arr[tid_comp]){
                        swap(shared_arr[threadIdx.x],shared_arr[tid_comp]);
                    }
                }
            }
            __syncthreads();
        }
    }
    if(blockIdx.x%2)
        arr[tid] = shared_arr[blockDim.x - threadIdx.x - 1];
    else
        arr[tid] = shared_arr[threadIdx.x];
}

template <typename T>
__global__ void bitonic_sort_stage2(T* arr, int i, int j){

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid_comp = tid ^ j;
    if(tid_comp > tid){
        if((tid & i)==0){ //ascending
            if(arr[tid]>arr[tid_comp]){
                swap(arr[tid],arr[tid_comp]);
            }
        }
        else{ //desending
            if(arr[tid]<arr[tid_comp]){
                swap(arr[tid],arr[tid_comp]);
            }
        }
    }
}

int32_t MyTopKPlugin::enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept 
{
    unsigned int extended_num;
    for(extended_num = 1; extended_num < length; extended_num<<=1){;}
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    unsigned int num_threads = prop.maxThreadsPerBlock;
    unsigned int num_blocks = extended_num / num_threads;
    if(!num_blocks)
    {
        num_threads = extended_num;
        num_blocks = 1;
    }
    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(num_threads, 1);
    bitonic_sort_stage1<<<grid_dim, block_dim, num_threads * sizeof(int), stream>>>((int*) inputs[0]);
    for(unsigned int i= num_threads<<1; i<=extended_num; i<<=1){
        for(unsigned int j=i>>1; j>0; j>>=1){
            bitonic_sort_stage2<<<grid_dim, block_dim>>>((int*) inputs[0], i, j);
        }
    }
    cudaMemcpy(outputs[0], inputs[0], k * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}