import os
import time
import ctypes
import hashlib
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Optional

import Metal
import objc
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
PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))

class MetalDeviceManager(DeviceManager):
  def __init__(self, device_name):
    super().__init__(device_name)
    self.dev_name = device_name
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.command_queue = self.device.newCommandQueue()
    if DEBUG >= 1: print("** Opened device", device_name)
    self.max_grid_size = (2147483647, 65535, 65535)
    self.max_block_size = (1024, 1024, 64)
    self._kernel_cache = {}

  # -------
  # GENERIC DEVICE INTERFACE METHODS
  # -------
  
  def sync(self):
    """Synchronize the device. Does nothing since Metal is synchronous by default."""
    pass

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
    """Free device memory by making the Metal buffer purgeable."""
    if d_buf is not None:
      d_buf.setPurgeableState_(Metal.MTLPurgeableStateEmpty)

  def compile_kernel(self, src: str, kernel_name: str):
    key = hashlib.md5(src.encode()).digest()
    if key in self._kernel_cache:
      if DEBUG >= 3 and not PSEUDO_DEBUG: print(f"{color_green('[Metal]')} Cache hit {color_green(kernel_name)}")
      return self._kernel_cache[key]
    if DEBUG >= 3 and not PSEUDO_DEBUG: print(f"{color_green('[Metal]')} Compiling kernel {color_green(kernel_name)}")
    if DEBUG >= 4: self.print_metal_ir(kernel_name, src)
    if DEBUG >= 5: self.print_metal_objdump(kernel_name, src)

    library, error = self.device.newLibraryWithSource_options_error_(src, None, None)
    if error:
      raise RuntimeError(f"Failed to compile Metal kernel '{kernel_name}': " f"{error.localizedDescription()}")
    func = library.newFunctionWithName_(kernel_name)
    self._kernel_cache[key] = func
    return func

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

    with objc.autorelease_pool():
      command_buffer = self.command_queue.commandBuffer()
      compute_encoder = command_buffer.computeCommandEncoder()
      kernel_name = str(kernel.name())
      pipeline, error = self.device.newComputePipelineStateWithFunction_error_(kernel, None)
      if error:
        raise RuntimeError(f"Failed to create Metal pipeline for '{kernel_name}': {error.localizedDescription()}")
      compute_encoder.setComputePipelineState_(pipeline)
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

  def _metal_tool(self, tool: str):
    """
    Return xcrun command for a Metal tool.

    Xcode 26+ may install the Metal compiler as a separate toolchain.
    """
    cmd = ["xcrun", "--toolchain", "metal", "-f", tool]

    try:
      result = subprocess.run(cmd, check=True, capture_output=True, text=True)
      return result.stdout.strip()
    except subprocess.CalledProcessError:
      result = subprocess.run(
        ["xcrun", "-f", tool],
        check=True,
        capture_output=True,
        text=True,
      )
      return result.stdout.strip()


  def print_metal_ir(self, kernel_name: str, src: str):
    """
    Compile generated MSL separately with Apple's command-line
    compiler and print human-readable AIR/LLVM IR.
    """

    if not shutil.which("xcrun"):
      print("[WARN] xcrun not found — cannot inspect Metal IR")
      return

    try:
      metal = self._metal_tool("metal")
    except (subprocess.CalledProcessError, FileNotFoundError):
      print("[WARN] Metal compiler toolchain not found")
      return

    with tempfile.TemporaryDirectory() as tmpdir:
      src_path = os.path.join(tmpdir, f"{kernel_name}.metal")
      ir_path = os.path.join(tmpdir, f"{kernel_name}.ll")

      with open(src_path, "w") as f:
        f.write(src)

      try:
        subprocess.run(
          [
            metal,
            "-O2",
            "-S",
            "-emit-llvm",
            src_path,
            "-o",
            ir_path,
          ],
          check=True,
          capture_output=True,
          text=True,
        )

        with open(ir_path, "r") as f:
          ir = f.read()

        if not PSEUDO_DEBUG:
          print(f"\n===== [Metal AIR/LLVM IR for kernel {kernel_name}] =====")
          print(ir)
          print("========================================================\n")

      except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to generate Metal AIR/LLVM IR for {kernel_name}")

        if e.stderr:
          print(e.stderr)


  def print_metal_objdump(self, kernel_name: str, src: str):
    """
    Compile MSL -> AIR -> metallib and inspect the resulting
    Metal library.

    NOTE: metal-objdump does NOT give us an officially documented
    equivalent of NVIDIA SASS.
    """

    try:
      metal = self._metal_tool("metal")
      metallib = self._metal_tool("metallib")
      metal_objdump = self._metal_tool("metal-objdump")
      metal_lipo = self._metal_tool("metal-lipo")
    except (subprocess.CalledProcessError, FileNotFoundError):
      print("[WARN] Metal command-line inspection tools not available")
      return

    with tempfile.TemporaryDirectory() as tmpdir:
      src_path = os.path.join(tmpdir, f"{kernel_name}.metal")
      air_path = os.path.join(tmpdir, f"{kernel_name}.air")
      metallib_path = os.path.join(tmpdir, f"{kernel_name}.metallib")

      with open(src_path, "w") as f:
        f.write(src)

      try:
        # MSL -> AIR
        subprocess.run(
          [
            metal,
            "-O2",
            "-c",
            src_path,
            "-o",
            air_path,
          ],
          check=True,
          capture_output=True,
          text=True,
        )

        # AIR -> metallib
        subprocess.run(
          [
            metallib,
            air_path,
            "-o",
            metallib_path,
          ],
          check=True,
          capture_output=True,
          text=True,
        )

        # Inspect Metal IR/library contents.
        objdump = subprocess.run(
          [
            metal_objdump,
            "-d",
            metallib_path,
          ],
          check=True,
          capture_output=True,
          text=True,
        )

        if not PSEUDO_DEBUG:
          print(f"\n===== [Metal object dump for kernel {kernel_name}] =====")
          print(objdump.stdout)
          print("=======================================================\n")

        # Show architecture slices if available.
        try:
          archs = subprocess.run(
            [
              metal_lipo,
              metallib_path,
              "-archs",
            ],
            check=True,
            capture_output=True,
            text=True,
          )

          if not PSEUDO_DEBUG:
            print(f"===== [Metal architectures for {kernel_name}] =====")
            print(archs.stdout.strip())
            print("=================================================\n")

        except subprocess.CalledProcessError:
          pass

      except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to inspect Metal kernel {kernel_name}")

        if e.stderr:
          print(e.stderr)

    
