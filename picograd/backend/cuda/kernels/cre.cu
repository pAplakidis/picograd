extern "C" __global__ void cross_entropy_kernel(const float *__restrict__ z,
                                                const int *__restrict__ y,
                                                float *__restrict__ losses,
                                                int batch_size,
                                                int num_classes)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size)
  {
    int label = y[idx];
    if (label >= 0 && label < num_classes)
    {
      float p = z[idx * num_classes + label];
      // clamp to avoid log(0)
      if (p < 1e-7f)
        p = 1e-7f;
      if (p > 1.0f - 1e-7f)
        p = 1.0f - 1e-7f;
      losses[idx] = -logf(p);
    }
    else
    {
      // invalid label, set loss=0 (or could signal error)
      losses[idx] = 0.0f;
    }
  }
}
