import ctypes

# CUDA driver API types and constants
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_void_p

# NVRTC types
nvrtcProgram = ctypes.c_void_p
nvrtcResult = ctypes.c_int
