import os
import time
import ctypes # TODO: replace ctypes with pycuda
import hashlib
import tempfile
import subprocess
import numpy as np
from typing import Tuple, List, Optional

from .error import CUDA_ERRORS
from .types import *
from picograd.backend.device import DeviceManager
from picograd.print_utils import *
from picograd.viz import recorder as viz

try:
  cuda = ctypes.CDLL('libcuda.so')
  nvrtc = ctypes.CDLL('libnvrtc.so')
except OSError as e:
  # raise RuntimeError("Could not load CUDA libraries. Make sure CUDA is installed and the libraries are in your library path.") from e
  print("Could not load CUDA libraries. Make sure CUDA is installed and the libraries are in your library path.")

DEBUG = int(os.getenv("DEBUG", 0))
KERNELS_PATH = "picograd/backend/cuda/kernels"
PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))  # if 1, generate assembly code as string but don't print (helps with segfaults)
CUDA_MEM_TRACE = int(os.getenv("CUDA_MEM_TRACE", 0))

TILE_SIZE = 16

def _dev_name(device):
  return getattr(device, "name", str(device))


class CudaDeviceManager(DeviceManager):
  def __init__(self, device_name):
    super().__init__(device_name)
    self.tile_size = TILE_SIZE
    self.dev_name = device_name

    self.ctx = CUcontext()
    self.module = None
    self.kernels = {}
    self._kernel_artifacts = {}
    self._kernel_names = {}
    self._alloc_sizes = {}
    self.active_alloc_bytes = 0
    self.peak_alloc_bytes = 0
    self.init_cuda()

    self.start_event = CUevent()
    self.end_event = CUevent()
    self.check_cuda(cuda.cuEventCreate(ctypes.byref(self.start_event), 0), "cuEventCreate (start)")
    self.check_cuda(cuda.cuEventCreate(ctypes.byref(self.end_event), 0), "cuEventCreate (end)")
    if DEBUG >= 1: print("** Opened device", device_name)
    self.max_grid_size = (2147483647, 65535, 65535)
    self.max_block_size = (1024, 1024, 64)

  def __del__(self):
    unloaded_modules = set()
    for module, _ in self.kernels.values():
      module_id = module.value
      if module_id in unloaded_modules: continue
      self.check_cuda(cuda.cuModuleUnload(module), "cuModuleUnload")
      unloaded_modules.add(module_id)

    self.check_cuda(cuda.cuEventDestroy(self.start_event), "cuEventDestroy (start)")
    self.check_cuda(cuda.cuEventDestroy(self.end_event), "cuEventDestroy (end)")

    self.check_cuda(cuda.cuCtxDestroy(self.ctx), "cuCtxDestroy")

  @staticmethod
  def check_cuda(result: int, func_name: str = "", sync=False):
    """Checks if CUDA function call was successful. Raises RuntimeError if not."""
    if result != 0:
      err_msg = CUDA_ERRORS.get(result, f"Unknown error code {result}")
      raise RuntimeError(f"[CUDA ERROR] {func_name} failed: {err_msg} (code {result})")
    if sync: cuda.cuCtxSynchronize()  # synchronous wait for CUDA ops to finish

  def check_nvrtc(self, result: int, func_name: str):
    """Checks if NVRTC function call was successful. Raises RuntimeError if not."""

    if result != 0:
      log_size = ctypes.c_size_t()
      nvrtc.nvrtcGetProgramLogSize(self.program, ctypes.byref(log_size))
      log = ctypes.create_string_buffer(log_size.value)
      nvrtc.nvrtcGetProgramLog(self.program, log)
      raise RuntimeError(f"[NVRTC ERROR] {func_name} failed with code {result}:\n{log.value.decode()}")

  @staticmethod
  def load_kernel(file_path: str) -> str:
    """Reads a kernel file and returns its contents as a string."""
    with open(os.path.join(KERNELS_PATH, file_path), 'r') as f:
      return f.read()

  def inspect_ptx_and_sass(self, kernel_name: str, ptx_str: str):
    artifacts = {"ptx": ptx_str, "warnings": []}

    if DEBUG >= 4 and not PSEUDO_DEBUG:
      print(f"\n===== [NVRTC Generated PTX for kernel {kernel_name}] =====")
      print(ptx_str)
      print("=================================\n")

    with tempfile.TemporaryDirectory() as tmpdir:
      ptx_path = os.path.join(tmpdir, "kernel.ptx")
      cubin_path = os.path.join(tmpdir, "kernel.cubin")

      with open(ptx_path, "w", encoding="utf-8") as f:
        f.write(ptx_str)

      arch = "sm_89"
      try:
        subprocess.run(["ptxas", ptx_path, "-o", cubin_path, f"-arch={arch}"], check=True, capture_output=True, text=True)
        sass_output = subprocess.run(["nvdisasm", cubin_path], check=True, capture_output=True, text=True)
        artifacts["sass"] = sass_output.stdout
        if DEBUG >= 5 and not PSEUDO_DEBUG:
          print(f"\n===== [SASS Assembly for kernel {kernel_name}] =====")
          print(sass_output.stdout)
          print("===========================\n")
      except FileNotFoundError:
        warning = "ptxas or nvdisasm not found in PATH"
        artifacts["warnings"].append(warning)
        if DEBUG >= 4:
          print(f"[WARN] {warning} — cannot print SASS")
      except subprocess.CalledProcessError as e:
        warning = f"Failed to generate SASS: {e}"
        artifacts["warnings"].append(warning)
        if DEBUG >= 4:
          print(f"[ERROR] {warning}")

    return artifacts

  def  init_cuda(self):
    """Gets CUDA device and context, then initializes CUDA driver API."""
    self.check_cuda(cuda.cuInit(0), "cuInit")
    device = CUdevice() 
    self.check_cuda(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    self.check_cuda(cuda.cuCtxCreate(ctypes.byref(self.ctx), 0, device), "cuCtxCreate")
    if DEBUG >= 3 and not PSEUDO_DEBUG: print(f"{color_green('[Cuda]')} Device initialized")

  def cuda_malloc(self, size: int) -> CUdeviceptr:
    """Allocates device memory and returns a pointer to it."""

    ptr = CUdeviceptr()
    self.check_cuda(cuda.cuMemAlloc(ctypes.byref(ptr), size), "cuMemAlloc")
    self._alloc_sizes[ptr.value] = size
    self.active_alloc_bytes += size
    self.peak_alloc_bytes = max(self.peak_alloc_bytes, self.active_alloc_bytes)
    if CUDA_MEM_TRACE:
      print(f"[Cuda-Mem] alloc {size} active={self.active_alloc_bytes} peak={self.peak_alloc_bytes}")
    return ptr

  def cuda_free(self, ptr: CUdeviceptr):
    """Frees device memory pointed to by ptr."""
    ptr_value = getattr(ptr, "value", None)
    if ptr_value is None or ptr_value not in self._alloc_sizes: return
    size = self._alloc_sizes.pop(ptr_value)
    result = cuda.cuMemFree(ptr)
    if result == 1: # CUDA_ERROR_INVALID_VALUE: stale/double free, already gone from accounting.
      self.active_alloc_bytes -= size
      if CUDA_MEM_TRACE:
        print(f"[Cuda-Mem] stale-free {size} active={self.active_alloc_bytes} peak={self.peak_alloc_bytes}")
      return
    self.check_cuda(result, "cuMemFree")
    self.active_alloc_bytes -= size
    if CUDA_MEM_TRACE:
      print(f"[Cuda-Mem] free {size} active={self.active_alloc_bytes} peak={self.peak_alloc_bytes}")

  def cuda_memcpy_htod(self, dst: CUdeviceptr, src: ctypes.c_void_p, size: int):
    """Copies data from host to device memory."""
    self.check_cuda(cuda.cuMemcpyHtoD(dst, ctypes.c_void_p(src), size), "cuMemcpyHtoD", sync=True)

  def cuda_memcpy_dtoh(self, dst: ctypes.c_void_p, src: CUdeviceptr, size):
    """Copies data from device to host memory."""
    self.check_cuda(cuda.cuMemcpyDtoH(ctypes.c_void_p(dst), src, size), "cuMemcpyDtoH", sync=True)

  def cuda_memcpy_dtod(self, dst: CUdeviceptr, src: CUdeviceptr, size: int):
    """Copies data inside the same device."""
    return self.check_cuda(cuda.cuMemcpyDtoD(dst, src, size), "cuMemcpyDtoD", sync=True)

  # -------
  # GENERIC DEVICE INTERFACE METHODS
  # -------

  def sync(self):
    """Synchronizes the device."""
    self.check_cuda(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

  def allocate_device_memory(self, x) -> ctypes.c_void_p:
    """Allocate device memory for tensor."""

    if isinstance(x, np.ndarray):
      nbytes = x.nbytes
    elif isinstance(x, int):
      nbytes = x
    else:
      raise ValueError("allocate_device_memory expects an integer (number of bytes) or a numpy ndarray.")
    return self.cuda_malloc(nbytes)

  def copy_data_to_device(self, d_T: ctypes.c_void_p, T_flat: np.ndarray):
    """Copy data from host to device."""
    self.cuda_memcpy_htod(d_T, T_flat.ctypes.data, T_flat.nbytes)

  def copy_data_to_host(self, d_T: ctypes.c_void_p, T_flat: np.ndarray):
    """Copy data from device to host."""
    self.cuda_memcpy_dtoh(T_flat.ctypes.data, d_T, T_flat.nbytes)

  def copy_device_to_device(self, d_src: ctypes.c_void_p, d_dst: ctypes.c_void_p, size: int):
    """Copy data inside the same device"""
    self.cuda_memcpy_dtod(d_dst, d_src, size)

  def free_device_tensor(self, d_T: ctypes.c_void_p):
    """Free tensor from device memory."""
    if d_T is None: return
    self.cuda_free(d_T)

  def compile_kernel(self, src: str, kernel_name: str) -> CUfunction:
    compile_start = time.time()
    kernel_name_bytes = kernel_name if isinstance(kernel_name, bytes) else kernel_name.encode("utf-8")
    kernel_name_str = kernel_name_bytes.decode()
    cache_key = (kernel_name_bytes, hashlib.sha256(src.encode()).digest())
    if cache_key in self.kernels:
      if DEBUG >= 3 and not PSEUDO_DEBUG:
        print(f"{color_green('[Cuda]')} Fetching compiled kernel {color_green(kernel_name_str)}.")
      viz.record("compile", device=_dev_name(self.dev_name), name=kernel_name_str, source=src, artifacts=self._kernel_artifacts.get(cache_key), cache_hit=True, duration_ms=(time.time() - compile_start) * 1000.0)
      self._kernel_names[str(self.kernels[cache_key][1])] = kernel_name_str
      return self.kernels[cache_key][1]

    if DEBUG >= 3 and not PSEUDO_DEBUG:
      print(f"{color_green('[Cuda]')} Compiling kernel {color_green(kernel_name_str)}")

    self.program = nvrtcProgram()
    nvrtc.nvrtcCreateProgram.restype = nvrtcResult
    nvrtc.nvrtcCreateProgram(
              ctypes.byref(self.program),
              ctypes.c_char_p(src.encode()),
              ctypes.c_char_p(f"{kernel_name_str}.cu".encode()),
              0,
              None,
              None
    )

    # compile to PTX
    opts = [
      b"--fmad=false",
      b"--gpu-architecture=compute_75",
    ]
    if DEBUG >= 4:
      opts += [
        b"-G",
        b"--device-debug",
        b"--generate-line-info",
        b"-lineinfo"
      ]
    self.check_nvrtc(
      nvrtc.nvrtcCompileProgram(self.program, len(opts), (ctypes.c_char_p * len(opts))(*opts)),
      "nvrtcCompileProgram"
    )

    # get PTX code
    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(self.program, ctypes.byref(ptx_size))
    ptx = (ctypes.c_char * ptx_size.value)()
    nvrtc.nvrtcGetPTX(self.program, ptx)
    ptx_str = ctypes.string_at(ptx, ptx_size.value).decode()

    artifacts = self.inspect_ptx_and_sass(kernel_name_str, ptx_str)

    # load PTX module
    self.module = CUmodule()
    self.check_cuda(cuda.cuModuleLoadData(ctypes.byref(self.module), ptx), "cuModuleLoadData")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(self.program))

    # get kernel function
    kfunc = CUfunction()
    self.check_cuda(cuda.cuModuleGetFunction(ctypes.byref(kfunc), self.module, ctypes.c_char_p(kernel_name_bytes)), "cuModuleGetFunction")
    self.kernels[cache_key] = (self.module, kfunc)
    self._kernel_artifacts[cache_key] = artifacts
    self._kernel_names[str(kfunc)] = kernel_name_str
    viz.record("compile", device=_dev_name(self.dev_name), name=kernel_name_str, source=src, artifacts=artifacts, cache_hit=False, duration_ms=(time.time() - compile_start) * 1000.0)
    return kfunc

  def launch_kernel(
      self,
      kfunc: CUfunction,
      grid: Tuple,
      block: Tuple,
      args: List[ctypes.c_void_p],
      shared_mem: int = 0,
      n_flops: Optional[int] = None
    ) -> Tuple[float, Optional[float]]:
    """Launches a CUDA kernel with the given grid and block dimensions and arguments."""

    if DEBUG >= 3 and not PSEUDO_DEBUG:
      print(f"{color_green('[Cuda]')} Launching kernel {color_yellow(kfunc)} with grid {color_yellow(grid)} and block {color_yellow(block)}")

    # FIXME: start_event and end_event cause Segmentation fault (undeterministically) for consecutive kernel launches
    # event-based profiling
    # self.check_cuda(cuda.cuEventRecord(self.start_event, 0), "cuEventRecord (start)")
    start = time.time()

    # launch kernel
    if not len(grid) == 3 or not len(block) == 3:
      raise ValueError(f"Unsupported grid/block dimensions: grid={grid}, block={block}. Must be 2D or 3D.")

    cuda.cuLaunchKernel.restype = CUresult
    cuda.cuLaunchKernel.argtypes = [
      CUfunction,
      ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # gridDim (blocks in x, y, z)
      ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # blockDim (threads per block in x, y, z)
      ctypes.c_uint,                                # sharedMemBytes (shared memory per block in bytes)
      ctypes.c_void_p,                              # hStream (CUstream, 0 for default)
      ctypes.POINTER(ctypes.c_void_p),              # void** kernelParams (array of pointers to arguments or NULL)
      ctypes.c_void_p                               # void** extra (reserved for future use, usually NULL)
    ]

    arg_buff = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    self.check_cuda(cuda.cuLaunchKernel(
      kfunc,
      grid[0], grid[1], grid[2],      # grid dimensions (blocks)
      block[0], block[1], block[2],   # block dimensions (threas per block)
      shared_mem, 0,                           # shared mem and stream
      arg_buff, 0
    ), "cuLaunchKernel", sync=True)

    # profiling results
    # self.check_cuda(cuda.cuEventRecord(self.end_event, 0), "cuEventRecord (end)")
    # self.check_cuda(cuda.cuEventSynchronize(self.end_event), "cuEventSynchronize")
    # elapsed_ms = ctypes.c_float()
    # self.check_cuda(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), self.start_event, self.end_event), "cuEventElapsedTime")
    end = time.time()
    elapsed_ms = (end - start) * 1000.0

    # compute GFLOPs
    gflops = None
    if n_flops is not None:
      # elapsed_s = elapsed_ms.value / 1000.0
      elapsed_s = elapsed_ms / 1000.0
      gflops = n_flops / (elapsed_s * 1e9)
      if DEBUG >= 3 and not PSEUDO_DEBUG:
        print(f"{color_yellow('[Cuda-Perf]')} Kernel time: {color_red(f'{elapsed_ms:.4f} ms — GFLOPs: {gflops:.2f}')}")
    else:
      if DEBUG >= 3 and not PSEUDO_DEBUG:
        # print(f"{color_yellow('[Cuda-Perf]')} Kernel time: {elapsed_ms.value:.3f} ms")
        print(f"{color_yellow('[Cuda-Perf]')} Kernel time: {elapsed_ms:.4f} ms")
    viz.record("launch", device=_dev_name(self.dev_name), kernel=self._kernel_names.get(str(kfunc), str(kfunc)), grid=grid, block=block, shared_mem=shared_mem, arg_count=len(args), elapsed_ms=elapsed_ms, gflops=gflops)
    
    return elapsed_ms, gflops
