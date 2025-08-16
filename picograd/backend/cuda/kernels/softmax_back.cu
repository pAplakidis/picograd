extern "C" __global__ void softmax_back_kernel(
    float *grad_output,
    float *softmax_output,
    float *grad_input,
    int batch_size,
    int n_classes)
{
  int row = blockIdx.x; // one block per row
  int tid = threadIdx.x;

  extern __shared__ float shmem[];

  if (row < batch_size)
  {
    const float *row_grad_out = grad_output + row * n_classes;
    const float *row_softmax = softmax_output + row * n_classes;
    float *row_grad_in = grad_input + row * n_classes;

    // compute dot = sum_j grad_output[j] * softmax_output[j]
    float dot = 0.0f;
    for (int i = tid; i < n_classes; i += blockDim.x)
    {
      dot += row_grad_out[i] * row_softmax[i];
    }

    // reduction sum
    shmem[tid] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
      if (tid < stride)
      {
        shmem[tid] += shmem[tid + stride];
      }
      __syncthreads();
    }
    dot = shmem[0]; // global dot product for this row

    // compute grad_input[i] = softmax_output[i] * (grad_output[i] - dot)
    for (int i = tid; i < n_classes; i += blockDim.x)
    {
      row_grad_in[i] = row_softmax[i] * (row_grad_out[i] - dot);
    }
  }
}
