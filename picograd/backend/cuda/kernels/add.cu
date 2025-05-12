extern "C" __global__ void add_kernel(
    float *A, float *B, float *C,
    int dim1, int dim2, int dim3)
{
  // 3D thread indexing
  int idx1 = blockIdx.z * blockDim.z + threadIdx.z;
  int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
  int idx3 = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx1 < dim1 && idx2 < dim2 && idx3 < dim3)
  {
    // Row-major linear index calculation
    int idx = idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    C[idx] = A[idx] + B[idx];
  }
}