import ctypes
import Metal
import numpy as np

prg = """
kernel void add_kernel(constant float *a [[ buffer(0) ]],
                       constant float *b [[ buffer(1) ]],
                       device float *c [[ buffer(2) ]],
                       uint id [[ thread_position_in_grid ]]) {
    c[id] = a[id] + b[id];
}
"""

# create a Metal device, library, and kernel function
device = Metal.MTLCreateSystemDefaultDevice()
library = device.newLibraryWithSource_options_error_(prg, None, None)[0]
kernel_function = library.newFunctionWithName_("add_kernel")

# create buffers
array_length = 5
buffer_length = array_length * 4  # 4 bytes per float

a = np.random.rand(array_length).astype(np.float32)
b = np.random.rand(array_length).astype(np.float32)

a_buf = device.newBufferWithLength_options_(a.size, Metal.MTLResourceStorageModeShared)
b_buf= device.newBufferWithLength_options_(b.size, Metal.MTLResourceStorageModeShared)
assert a.size == b.size, "Input arrays must have the same size"
c_buf= device.newBufferWithLength_options_(a.size, Metal.MTLResourceStorageModeShared)

# populate input buffers
a_array = (ctypes.c_float * array_length).from_buffer(a_buf.contents().as_buffer(buffer_length))  # Map the Metal buffer to a Python array
b_array = (ctypes.c_float * array_length).from_buffer(b_buf.contents().as_buffer(buffer_length))  # Map the Metal buffer to a Python array
np.copyto(np.ctypeslib.as_array(a_array), a)    # populate
np.copyto(np.ctypeslib.as_array(b_array), b)    # populate

# command queue and command buffer
commandQueue = device.newCommandQueue()
commandBuffer = commandQueue.commandBuffer()

# set the kernel function and buffers
pso = device.newComputePipelineStateWithFunction_error_(kernel_function, None)[0]
computeEncoder = commandBuffer.computeCommandEncoder()
computeEncoder.setComputePipelineState_(pso)
computeEncoder.setBuffer_offset_atIndex_(a_buf, 0, 0)
computeEncoder.setBuffer_offset_atIndex_(b_buf, 0, 1)
computeEncoder.setBuffer_offset_atIndex_(c_buf, 0, 2)

# threadgroup/grid size
threadsPerThreadgroup = Metal.MTLSizeMake(array_length, 1, 1)
print(pso.maxTotalThreadsPerThreadgroup())
threadgroupSize = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)

# dispatch the kernel
computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup, threadgroupSize)
computeEncoder.endEncoding()

# commit the command buffer (execute)
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

# map the Metal buffer to a Python array (gpu -> cpu)
output_data = (ctypes.c_float * array_length).from_buffer(c_buf.contents().as_buffer(buffer_length))
np.copyto(np.ctypeslib.as_array(output_data), np.ctypeslib.as_array(output_data))  # Copy the data from the Metal buffer to a NumPy array

output_python = a + b
print(output_python)
print(output_data)
assert np.allclose(output_data, output_python), "Output does not match reference!"
print("Reference matches output!")
