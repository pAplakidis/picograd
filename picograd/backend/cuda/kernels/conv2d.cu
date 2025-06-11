extern "C" __global__ void conv2d_kernel(
    const float *__restrict__ input,  // [BS, C_in, H_in, W_in]
    const float *__restrict__ weight, // [C_out, C_in, K_h, K_w]
    const float *__restrict__ bias,   // [C_out]
    float *output,                    // [BS, C_out, H_out, W_out]
    int BS, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    int stride, int padding)
{
  int n = blockIdx.x;                                  // batch index
  int out_c = blockIdx.y;                              // output channel
  int out_idx = threadIdx.x + blockDim.x * blockIdx.z; // linear output pixel index

  int total_out_pixels = H_out * W_out;
  if (out_idx >= total_out_pixels)
    return;

  int i = out_idx / W_out; // output row
  int j = out_idx % W_out; // output col

  float value = bias[out_c];

  for (int in_c = 0; in_c < C_in; ++in_c)
  {
    for (int kh = 0; kh < K_h; ++kh)
    {
      for (int kw = 0; kw < K_w; ++kw)
      {
        int h_in = i * stride - padding + kh;
        int w_in = j * stride - padding + kw;

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in)
        {
          // Calculate input and weight offsets
          int input_offset = n * C_in * H_in * W_in + in_c * H_in * W_in + h_in * W_in + w_in;
          int weight_offset = out_c * C_in * K_h * K_w + in_c * K_h * K_w + kh * K_w + kw;

          value += input[input_offset] * weight[weight_offset];
        }
      }
    }
  }

  // Store result
  int output_offset = n * C_out * H_out * W_out + out_c * H_out * W_out + i * W_out + j;
  output[output_offset] = value;
}
