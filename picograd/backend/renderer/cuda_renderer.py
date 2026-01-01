from .cstyle import CStyleRenderer

class CUDARenderer(CStyleRenderer):
  device = "CUDA"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152

  kernel_typedef = "extern \"C\" __global__ void"
  elementwise_kernel_name = "E"

  def __init__(self, arch:str):
    self.arch = arch
    # self.tensor_cores = tc.cuda_sm89 if int(arch[3:]) >= 89 else tc.cuda_sm80 if int(arch[3:]) >= 80 else tc.cuda_sm75 if int(arch[3:]) >= 75 else []

  def __reduce__(self):
    return self.__class__, (self.arch,)

  # TODO: make this generic
  def elementwise(self, op, dtype, arg, shape: tuple[int,...]):
    alu = self.op_to_alu(op)
    # size_t = np.prod(shape)
    size_t = '_'.join([str(s) for s in shape])
    kernel_name = f"{self.elementwise_kernel_name}_{size_t}"

    func = f"""
{self.kernel_typedef} {kernel_name}(float *data0, float *data1, float *data2)
{{
  int gidx0 = blockIdx.x; /* 16 */
  float val0 = *(data1 + gidx0);
  float val1 = *(data2 + gidx0);
  *(data0 + gidx0) = (val0 {alu} val1);
}}
    """
    return func, kernel_name

  def reduce(self, op, dtype, arg, shape: tuple[int,...], dim: int):
    raise NotImplementedError("CUDA reduce not implemented yet")

  # TODO: reduce (sum, max, min, std, argmax, argmin)
