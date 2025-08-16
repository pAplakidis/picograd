extern "C" __global__ void softmax_kernel(float *input, float *output, int batch_size, int n_classes)
{
  int row = blockIdx.x; // each block handles one row
  int tid = threadIdx.x;

  float FLT_MAX = 3.402823466e+38F;

  extern __shared__ float shmem[];

  if (row < batch_size)
  {
    const float *row_input = input + row * n_classes;
    float *row_output = output + row * n_classes;

    // compute max value in the row for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < n_classes; i += blockDim.x)
    {
      max_val = fmaxf(max_val, row_input[i]);
    }

    // reduction: find global max
    shmem[tid] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
      if (tid < stride)
      {
        shmem[tid] = fmaxf(shmem[tid], shmem[tid + stride]);
      }
      __syncthreads();
    }
    max_val = shmem[0]; // global max for this row

    // compute exp(x - max) and accumulate sum
    float sum_val = 0.0f;
    for (int i = tid; i < n_classes; i += blockDim.x)
    {
      float exp_val = expf(row_input[i] - max_val);
      row_output[i] = exp_val;
      sum_val += exp_val;
    }

    // reduction: sum of exps
    shmem[tid] = sum_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
      if (tid < stride)
      {
        shmem[tid] += shmem[tid + stride];
      }
      __syncthreads();
    }
    float denom = shmem[0]; // global sum

    // normalize
    for (int i = tid; i < n_classes; i += blockDim.x)
    {
      row_output[i] /= denom;
    }
  }
}
