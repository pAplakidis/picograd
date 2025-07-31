extern "C" __global__ void relu_back_kernel(float *input, float *grad_output, float *grad_input, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
  }
}
