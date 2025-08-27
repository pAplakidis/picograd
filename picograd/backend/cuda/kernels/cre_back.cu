extern "C" __global__ void cross_entropy_back_kernel(
    const float *__restrict__ z, // [batch_size, num_classes]
    const int *__restrict__ y,   // [batch_size]
    float *__restrict__ grad,    // [batch_size, num_classes]
    int batch_size, int num_classes)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size * num_classes)
  {
    int row = i / num_classes;
    int col = i % num_classes;
    int label = y[row];

    float one_hot = (col == label) ? 1.0f : 0.0f;
    grad[i] = (z[i] - one_hot) / batch_size;
  }
}