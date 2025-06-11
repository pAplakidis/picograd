extern "C" __global__ void conv2d_backward_kernel(
    const float *__restrict__ a_padded, // padded input: (BS, C_in, H_pad, W_pad)
    const float *__restrict__ w,        // weights: (C_out, C_in, K, K)
    const float *__restrict__ grad_out, // grad output: (BS, C_out, H_out, W_out)
    float *grad_a_padded,               // grad input padded: (BS, C_in, H_pad, W_pad)
    float *grad_w,                      // grad weights: (C_out, C_in, K, K)
    float *grad_b,                      // grad bias: (C_out)
    int BS, int C_in, int C_out,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int kernel_size,
    int stride,
    int padding)
{
  // Compute thread indices
  int batch = blockIdx.x;
  int out_c = blockIdx.y;
  int i = threadIdx.y; // output height index
  int j = threadIdx.x; // output width index

  if (i >= H_out || j >= W_out)
    return;

  // Index into grad_out
  int grad_out_idx = ((batch * C_out + out_c) * H_out + i) * W_out + j;
  float grad_out_val = grad_out[grad_out_idx];

  // Accumulate bias gradient atomically
  atomicAdd(&grad_b[out_c], grad_out_val);

  // For each input channel, update grad_w and grad_a_padded
  for (int in_c = 0; in_c < C_in; ++in_c)
  {
    // Coordinates in padded input
    int h_start = i * stride;
    int w_start = j * stride;

    // Compute grad_w and grad_a_padded contributions
    for (int p = 0; p < kernel_size; ++p)
    {
      for (int q = 0; q < kernel_size; ++q)
      {
        int a_idx = ((batch * C_in + in_c) * H_pad + (h_start + p)) * W_pad + (w_start + q);
        int w_idx = ((out_c * C_in + in_c) * kernel_size + p) * kernel_size + q;

        float a_val = a_padded[a_idx];
        float w_val = w[w_idx];

        // grad_w update: accumulate input * grad_out_val atomically
        atomicAdd(&grad_w[w_idx], a_val * grad_out_val);

        // grad_a_padded update: accumulate weight * grad_out_val atomically
        atomicAdd(&grad_a_padded[a_idx], w_val * grad_out_val);
      }
    }
  }
}