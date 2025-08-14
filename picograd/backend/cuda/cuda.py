import os
import time
import ctypes
import tempfile
import subprocess
from typing import Tuple, List, Optional

from picograd.print_utils import *
from .error import CUDA_ERRORS
from .utils import *
from .types import *

try:
  cuda = ctypes.CDLL('libcuda.so')
  nvrtc = ctypes.CDLL('libnvrtc.so')
except OSError as e:
  # raise RuntimeError("Could not load CUDA libraries. Make sure CUDA is installed and the libraries are in your library path.") from e
  print("Could not load CUDA libraries. Make sure CUDA is installed and the libraries are in your library path.")

KERNELS_PATH = "picograd/backend/cuda/kernels"
PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))  # if 1, generate assembly code as string but don't print (helps with segfaults)


class CudaDevice:
  def __init__(self, debug: int = 1):
    self.debug = debug  # debug levels: 0: no debug, 1: basic debug, 2: kernel level
    self.ctx = CUcontext()
    self.module = None
    self.kernels = {}
    self.init_cuda()

    self.start_event = CUevent()
    self.end_event = CUevent()
    self.check_cuda(cuda.cuEventCreate(ctypes.byref(self.start_event), 0), "cuEventCreate (start)")
    self.check_cuda(cuda.cuEventCreate(ctypes.byref(self.end_event), 0), "cuEventCreate (end)")

  def __del__(self):
    self.check_cuda(cuda.cuEventDestroy(self.start_event), "cuEventDestroy (start)")
    self.check_cuda(cuda.cuEventDestroy(self.end_event), "cuEventDestroy (end)")

    # TODO: garbage collect self.allocations and self.kernels

    if self.module: self.check_cuda(cuda.cuModuleUnload(self.module), "cuModuleUnload")
    self.check_cuda(cuda.cuCtxDestroy(self.ctx), "cuCtxDestroy")

  @staticmethod
  def check_cuda(result: int, func_name: str = "", sync=False):
    """Checks if CUDA function call was successful. Raises RuntimeError if not."""
    if result != 0:
      err_msg = CUDA_ERRORS.get(result, f"Unknown error code {result}")
      raise RuntimeError(f"[CUDA ERROR] {func_name} failed: {err_msg} (code {result})")
    if sync: cuda.cuCtxSynchronize()  # synchronous wait for CUDA ops to finish

  @staticmethod
  def load_kernel(file_path: str) -> str:
    """Reads a kernel file and returns its contents as a string."""

    with open(os.path.join(KERNELS_PATH, file_path), 'r') as f:
      kernel_code = f.read()
    return kernel_code

  def check_nvrtc(self, result: int, func_name: str):
    """Checks if NVRTC function call was successful. Raises RuntimeError if not."""

    if result != 0:
      log_size = ctypes.c_size_t()
      nvrtc.nvrtcGetProgramLogSize(self.program, ctypes.byref(log_size))
      log = ctypes.create_string_buffer(log_size.value)
      nvrtc.nvrtcGetProgramLog(self.program, log)
      raise RuntimeError(f"[NVRTC ERROR] {func_name} failed with code {result}:\n{log.value.decode()}")

  def print_ptx_and_sass(self, kernel_name: str, ptx_str: str):
    if not PSEUDO_DEBUG:
      print(f"\n===== [NVRTC Generated PTX for kernel {kernel_name}] =====")
      print(ptx_str)
      print("=================================\n")

    if self.debug >= 4:
      with tempfile.TemporaryDirectory() as tmpdir:
        ptx_path = os.path.join(tmpdir, "kernel.ptx")
        cubin_path = os.path.join(tmpdir, "kernel.cubin")

        with open(ptx_path, "w") as f:
          f.write(ptx_str)

        arch = "sm_89"  # match your GPU
        try:
          subprocess.run(
              ["ptxas", ptx_path, "-o", cubin_path, f"-arch={arch}"],
              check=True
          )
          sass_output = subprocess.run(
              ["nvdisasm", cubin_path],
              check=True,
              capture_output=True,
              text=True
          )
          if not PSEUDO_DEBUG:
            print(f"\n===== [SASS Assembly for kernel {kernel_name}] =====")
            print(sass_output.stdout)
            print("===========================\n")
        except FileNotFoundError:
          print("[WARN] ptxas or nvdisasm not found in PATH — cannot print SASS")
        except subprocess.CalledProcessError as e:
          print(f"[ERROR] Failed to generate SASS: {e}")

  def  init_cuda(self):
    """Gets CUDA device and context, then initializes CUDA driver API."""

    if self.debug >= 2 and not PSEUDO_DEBUG:
      print(f"{color_green("[Cuda]")} Initializing...")

    self.check_cuda(cuda.cuInit(0), "cuInit")
    device = CUdevice() 
    self.check_cuda(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    self.check_cuda(cuda.cuCtxCreate(ctypes.byref(self.ctx), 0, device), "cuCtxCreate")

  def compile_kernel(self, src: str, kernel_name: str):
    if kernel_name in self.kernels:
      if self.debug >= 2 and not PSEUDO_DEBUG:
        print(f"{color_green('[Cuda]')} Fetching compiled kernel {color_green(kernel_name)}.")
      return self.kernels[kernel_name]

    if self.debug >= 2 and not PSEUDO_DEBUG:
      print(f"{color_green("[Cuda]")} Compiling kernel {color_green(kernel_name)}")

    self.program = nvrtcProgram()
    nvrtc.nvrtcCreateProgram.restype = nvrtcResult
    nvrtc.nvrtcCreateProgram(
              ctypes.byref(self.program),
              ctypes.c_char_p(src.encode()),
              ctypes.c_char_p(f"{kernel_name.decode()}.cu".encode()),
              0,
              None,
              None
    )

    # compile to PTX
    opts = [
      b"--fmad=false",
      b"--gpu-architecture=compute_75",
    ]
    if self.debug >= 2 and not PSEUDO_DEBUG:
      opts += [
        b"--device-debug",
        b"--generate-line-info",
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

    # Print PTX and SASS (intermediate repr and assembly)
    if self.debug >= 3:
      self.print_ptx_and_sass(kernel_name, ptx_str)

    # load PTX module
    self.module = CUmodule()
    self.check_cuda(cuda.cuModuleLoadData(ctypes.byref(self.module), ptx), "cuModuleLoadData")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(self.program))

    # get kernel function
    kfunc = CUfunction()
    self.check_cuda(cuda.cuModuleGetFunction(ctypes.byref(kfunc), self.module, ctypes.c_char_p(kernel_name)), "cuModuleGetFunction")
    self.kernels[kernel_name] = kfunc
    return kfunc

  def cuda_malloc(self, size: int) -> CUdeviceptr:
    """Allocates device memory and returns a pointer to it."""

    ptr = CUdeviceptr()
    self.check_cuda(cuda.cuMemAlloc(ctypes.byref(ptr), size), "cuMemAlloc")
    return ptr

  def cuda_free(self, ptr: CUdeviceptr):
    """Frees device memory pointed to by ptr."""
    self.check_cuda(cuda.cuMemFree(ptr), "cuMemFree")

  def memcpy_htod(self, dst: CUdeviceptr, src: ctypes.c_void_p, size: int):
    """Copies data from host to device memory."""
    self.check_cuda(cuda.cuMemcpyHtoD(dst, ctypes.c_void_p(src), size), "cuMemcpyHtoD", sync=True)

  def memcpy_dtoh(self, dst: ctypes.c_void_p, src: CUdeviceptr, size):
    """Copies data from device to host memory."""
    self.check_cuda(cuda.cuMemcpyDtoH(ctypes.c_void_p(dst), src, size), "cuMemcpyDtoH", sync=True)

  def launch_kernel(
      self,
      kfunc: CUfunction,
      grid: Tuple,
      block: Tuple,
      args: List[ctypes.c_void_p],
      n_flops: Optional[int] = None
    ):
    """Launches a CUDA kernel with the given grid and block dimensions and arguments."""

    if self.debug >= 2 and not PSEUDO_DEBUG:
      print(f"{color_green("[Cuda]")} Launching kernel {color_yellow(kfunc)} with grid {color_yellow(grid)} and block {color_yellow(block)}")

    # FIXME: start_event and end_event cause Segmentation fault (undeterministically) for consecutive kernel launches
    # event-based profiling
    # self.check_cuda(cuda.cuEventRecord(self.start_event, 0), "cuEventRecord (start)")
    start = time.time()

    # launch kernel
    if not len(grid) == 3 or not len(block) == 3:
      raise ValueError(f"Unsupported grid/block dimensions: grid={grid}, block={block}. Must be 2D or 3D.")

    arg_buff = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    self.check_cuda(cuda.cuLaunchKernel(
      kfunc,
      grid[0], grid[1], grid[2],      # grid dimensions (blocks)
      block[0], block[1], block[2],   # block dimensions (threas per block)
      0, 0,                           # shared mem and stream
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
    if n_flops is not None:
      # elapsed_s = elapsed_ms.value / 1000.0
      elapsed_s = elapsed_ms / 1000.0
      gflops = n_flops / (elapsed_s * 1e9)
      if self.debug >= 1 and not PSEUDO_DEBUG:
        # print(f"{color_yellow("[Cuda-Perf]")} Kernel time: {color_red(f"{elapsed_ms.value:.3f} ms — GFLOPs: {gflops:.2f}")}")
        print(f"{color_yellow("[Cuda-Perf]")} Kernel time: {color_red(f"{elapsed_ms:.4f} ms — GFLOPs: {gflops:.2f}")}")
    else:
      if self.debug >= 1 and not PSEUDO_DEBUG:
        # print(f"{color_yellow('[Cuda-Perf]')} Kernel time: {elapsed_ms.value:.3f} ms")
        print(f"{color_yellow('[Cuda-Perf]')} Kernel time: {elapsed_ms:.4f} ms")
