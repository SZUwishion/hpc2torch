#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename T>
__global__ void gather_kernel(T const *data, int64_t const* indices,
                            int indices_size, T *output, int output_size,
                            int pre_size, int axis_size, int post_size, int num_elem_per_thread)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(y >= indices_size || x >= pre_size * post_size) {
        return;
    }

    for(int i = 0; i < num_elem_per_thread; i++) {
        const int idx = x * num_elem_per_thread + i;
        if(idx >= pre_size * post_size) {
            break;
        }

        const int pre_idx = idx / post_size;
        const int post_idx = idx % post_size;
        const int indices_idx = y;

        const int data_idx = pre_idx * axis_size * post_size + indices[indices_idx] * post_size + post_idx;
        const int output_idx = pre_idx * indices_size * post_size + indices_idx * post_size + post_idx;

        output[output_idx] = data[data_idx];
    }
}

extern "C" void gather(void const* data, int64_t const* indices,
                        int indices_size, void* output,  int output_size,
                        int pre_size, int axis_size, int post_size,
                        const int elem_size)
{
    // 将input都看成三部分，pre_size是axis前面的大小, axis_size是axis上的大小, post_size是axis后的大小
    // 根据gather的定义，output也可以相应看成三部分
    dim3 block_size(128, 1);
    dim3 grid_size((pre_size * post_size + block_size.x - 1) / block_size.x,
                    (indices_size + block_size.y - 1) / block_size.y);
    // x维度计算output的pre_size * post_size部分, y维度计算output的indices_size部分，通过计算x和y的索引可以得出input三个部分对应的索引
    int num_elem_per_thread = 2;
    switch(elem_size) {
        case 2:
            gather_kernel<<<grid_size, block_size>>>((half const*)data, indices, indices_size, (half*)output, output_size, pre_size, axis_size, post_size, num_elem_per_thread);
            break;
        case 4:
            gather_kernel<<<grid_size, block_size>>>((float const*)data, indices, indices_size, (float*)output, output_size, pre_size, axis_size, post_size, num_elem_per_thread);
            break;
        case 8:
            gather_kernel<<<grid_size, block_size>>>((double const*)data, indices, indices_size, (double*)output, output_size, pre_size, axis_size, post_size, num_elem_per_thread);
            break;
        default:
            printf("Unsupported data type\n");
            break;
    }
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(error));
    // }
}