extern "C" __global__ void relu_kernel(const float *input, float *output, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
  }
}