extern "C" __global__ void relu_kernel(float *input, float *output, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
  }
}