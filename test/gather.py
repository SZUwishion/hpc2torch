import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor
def test(inputShape, indexShape, axis, test_dtype, device):
    print(
        f"Testing Softmax on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(torch.int64).to(device)

    rank = len(inputShape)
    outTensor = gather(rank, axis, inputTensor, indexTensor)#

    Q_output = torch.zeros(outTensor.shape, device=device, dtype=test_dtype)
    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    if device == "cuda":
        torch_gather_time = performance.CudaProfile(
            (gather, (rank, axis, inputTensor, indexTensor))
        )

        index_size = indexTensor.numel()
        pre_size = int(np.prod(inputShape[:axis]))
        post_size = int(np.prod(inputShape[axis + 1 :]))
        axis_size = inputShape[axis]
        output_size = outTensor.numel()

        kernel_type = np.argmax(np.array([pre_size * index_size, post_size * index_size, pre_size * post_size]))
            
        lib.gather.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        custom_gather_time = performance.CudaProfile(
            (
                lib.gather,
                (
                    input_ptr,
                    index_ptr,
                    index_size,
                    output_ptr,
                    output_size,
                    pre_size,
                    axis_size,
                    post_size,
                    test_dtype.itemsize,
                    kernel_type,
                )
            )
        )
        
    performance.logBenchmark(torch_gather_time, custom_gather_time)

    tmpa = outTensor.to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2), (2, 2), 0, torch.float32, "cuda"),
        ((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((3, 4, 4, 5, 2), (2, 3), 0, torch.float32, "cuda"),
        ((3, 4, 4, 5, 2), (2, 3), 2, torch.float32, "cuda"),
        ((1024, 1024, 1024), (1), 0, torch.float32, "cuda"),
        ((1024, 1024, 1024), (1), 1, torch.float32, "cuda"),
        ((1024, 1024, 1024), (1), 2, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),
        ((1024, 1024), (128, 128), 0, torch.float32, "cuda"),
        ((1024, 1024), (128, 128), 1, torch.float32, "cuda"),

        ((3, 2), (2, 2), 0, torch.float16, "cuda"),
        ((3, 2), (1, 2), 1, torch.float16, "cuda"),
        ((3, 4, 4, 5, 2), (2, 3), 0, torch.float16, "cuda"),
        ((3, 4, 4, 5, 2), (2, 3), 2, torch.float16, "cuda"),
        ((1024, 1024, 1024), (1), 0, torch.float16, "cuda"),
        ((1024, 1024, 1024), (1), 1, torch.float16, "cuda"),
        ((1024, 1024, 1024), (1), 2, torch.float16, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float16, "cuda"),
        ((1024, 1024), (128, 128), 0, torch.float16, "cuda"),
        ((1024, 1024), (128, 128), 1, torch.float16, "cuda"),
         
]
filtered_test_cases = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape , indexShape, axis, test_dtype, device)