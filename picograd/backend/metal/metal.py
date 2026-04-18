import os
import time
import ctypes
from typing import List, Tuple, Optional
import Metal
import numpy as np

from picograd.backend.device import DeviceManager
from picograd.print_utils import *

try:
  device = Metal.MTLCreateSystemDefaultDevice()
  if device is None:
    raise RuntimeError("No Metal-compatible GPU found")
except Exception as e:
  print(f"Error initializing Metal device: {e}")
  device = None

DEBUG = int(os.getenv("DEBUG", 0))

class MetalDeviceManager(DeviceManager):
  def __init__(self, device_name):
    super().__init__(device_name)
    self.dev_name = device_name
    self.device = Metal.MTLCreateSystemDefaultDevice()
    if DEBUG >= 1: print("** Opened device", device_name)
    self.max_grid_size = (2147483647, 65535, 65535)
    self.max_block_size = (1024, 1024, 64)

  # -------
  # GENERIC DEVICE INTERFACE METHODS
  # -------

  def allocate_device_memory(self, x):
    """Allocate device memory and return a Metal buffer."""
    if isinstance(x, np.ndarray):
      nbytes = x.nbytes
    elif isinstance(x, int):
      nbytes = x
    else:
      raise ValueError(f"Unsupported type for device memory allocation: {type(x)}")
    return self.device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)

  def copy_data_to_device(self, d_buf, h_buf: np.ndarray):
    """Copy data from host to device."""
    assert d_buf.length() >= h_buf.nbytes
    buf = d_buf.contents().as_buffer(h_buf.nbytes)
    dst = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    src = h_buf.ctypes.data
    ctypes.memmove(dst, src, h_buf.nbytes)

  def copy_data_to_host(self, d_buf, h_buf: np.ndarray):
    assert d_buf.length() >= h_buf.nbytes
    buf = d_buf.contents().as_buffer(h_buf.nbytes)
    src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    dst_ptr = h_buf.ctypes.data
    ctypes.memmove(dst_ptr, src_ptr, h_buf.nbytes)

  def copy_device_to_device(self, d_src, d_dst, size: int):
    assert d_src.length() >= size and d_dst.length() >= size
    src_buf = d_src.contents().as_buffer(size)
    dst_buf = d_dst.contents().as_buffer(size)
    src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(src_buf))
    dst_ptr = ctypes.addressof(ctypes.c_char.from_buffer(dst_buf))
    ctypes.memmove(dst_ptr, src_ptr, size)

  def free_device_tensor(self, d_buf):
    """Free device memory (Metal buffers are automatically managed, so this is a no-op)."""
    pass

  def compile_kernel(self, src: str, kernel_name: str):
    """Compile a Metal kernel from source code and return the kernel function."""
    if DEBUG >= 3: print(f"{color_green('[Metal]')} Compiling kernel {color_green(kernel_name)}")
    library, error = self.device.newLibraryWithSource_options_error_(src, None, None)
    if error:
      raise RuntimeError(f"Failed to compile Metal kernel '{kernel_name}': {error.localizedDescription()}")
    return library.newFunctionWithName_(kernel_name)

  # FIXME: this must match cuda args and scheduler standard
  def launch_kernel(
    self,
    kernel,
    grid_size: Tuple[int, int, int],
    block_size: Tuple[int, int, int],
    buffers: List,
    shared_mem: int = 0,
    n_flops: Optional[int] = None
  ):
    """Launch a Metal kernel with the specified buffers and execution configuration."""

    if DEBUG >= 3: print(f"{color_green('[Metal]')} Launching kernel '{kernel.name()}' with grid size {grid_size} and block size {block_size}")
    start = time.time()

    command_queue = self.device.newCommandQueue()
    command_buffer = command_queue.commandBuffer()
    compute_encoder = command_buffer.computeCommandEncoder()
    compute_encoder.setComputePipelineState_(self.device.newComputePipelineStateWithFunction_error_(kernel, None)[0])
    for i, buf in enumerate(buffers):
      compute_encoder.setBuffer_offset_atIndex_(buf, 0, i)
    compute_encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(*grid_size), Metal.MTLSizeMake(*block_size))
    compute_encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    end = time.time()
    elapsed_ms = (end - start) * 1000

    gflops = None
    if n_flops is not None:
      gflops = (n_flops / (elapsed_ms / 1000)) / 1e9
    if DEBUG >= 3:
      if gflops:
        print(f"{color_green('[Metal]')} Kernel '{kernel.name()}' executed {n_flops} FLOPs in {elapsed_ms:.4f} ms ({gflops:.4f} GFLOPs)")
      else:
        print(f"{color_green('[Metal]')} Kernel '{kernel.name()}' execution completed in {color_red(f'{elapsed_ms:.4f} ms')}")

    return elapsed_ms, gflops if n_flops is not None else None
