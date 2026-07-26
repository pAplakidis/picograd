import ctypes
import Metal
import numpy as np

prg = """
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum_kernel(
    const device float *input      [[ buffer(0) ]],
    device float *output           [[ buffer(1) ]],
    constant uint &N               [[ buffer(2) ]],
    threadgroup float *shared_mem  [[ threadgroup(0) ]],
    uint tid                       [[ thread_position_in_threadgroup ]],
    uint gid                       [[ thread_position_in_grid ]],
    uint group_id                  [[ threadgroup_position_in_grid ]],
    uint threads_per_group         [[ threads_per_threadgroup ]]
) {
    // Safe load (avoid out-of-bounds)
    float val = 0.0;
    if (gid < N) {
        val = input[gid];
    }

    shared_mem[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial sum
    if (tid == 0) {
        output[group_id] = shared_mem[0];
    }
}
"""

# Create Metal device
device = Metal.MTLCreateSystemDefaultDevice()
if device is None:
    raise RuntimeError("No Metal device available")

# Compile
lib, err = device.newLibraryWithSource_options_error_(prg, None, None)
if lib is None:
    raise RuntimeError(err.localizedDescription())

kernel = lib.newFunctionWithName_("reduce_sum_kernel")
pso, _ = device.newComputePipelineStateWithFunction_error_(kernel, None)

# Input
array_length = 1024
input_data = np.random.rand(array_length).astype(np.float32)
expected_sum = np.sum(input_data)

# Config
threads_per_group = min(256, array_length)
num_groups = (array_length + threads_per_group - 1) // threads_per_group

# Buffers
input_buf = device.newBufferWithLength_options_(
    array_length * 4, Metal.MTLResourceStorageModeShared
)

output_buf = device.newBufferWithLength_options_(
    num_groups * 4, Metal.MTLResourceStorageModeShared
)

n_np = np.array([array_length], dtype=np.uint32)
n_buf = device.newBufferWithBytes_length_options_(
    n_np, 4, Metal.MTLResourceStorageModeShared
)

# Copy input
input_array = (ctypes.c_float * array_length).from_buffer(
    input_buf.contents().as_buffer(array_length * 4)
)
np.copyto(np.ctypeslib.as_array(input_array), input_data)

# Command setup
command_queue = device.newCommandQueue()
command_buffer = command_queue.commandBuffer()

encoder = command_buffer.computeCommandEncoder()
encoder.setComputePipelineState_(pso)

encoder.setBuffer_offset_atIndex_(input_buf, 0, 0)
encoder.setBuffer_offset_atIndex_(output_buf, 0, 1)
encoder.setBuffer_offset_atIndex_(n_buf, 0, 2)

# Allocate shared memory
encoder.setThreadgroupMemoryLength_atIndex_(
    threads_per_group * 4, 0
)

# Dispatch
grid_size = Metal.MTLSizeMake(array_length, 1, 1)
group_size = Metal.MTLSizeMake(threads_per_group, 1, 1)

encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
encoder.endEncoding()

# Run
command_buffer.commit()
command_buffer.waitUntilCompleted()

# Read partial sums
output_array = (ctypes.c_float * num_groups).from_buffer(
    output_buf.contents().as_buffer(num_groups * 4)
)

partials = np.ctypeslib.as_array(output_array)

# Final reduction on CPU
result = np.sum(partials)

# Verify
print(f"CPU sum: {expected_sum}")
print(f"GPU sum: {result}")

assert np.allclose(result, expected_sum, atol=1e-5)
print("Reduce sum kernel PASSED!")
