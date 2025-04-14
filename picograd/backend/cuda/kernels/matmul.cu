const int TILE_SIZE = 16;

extern "C" __global__ void matmul_tiled_kernel(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
  {
    // Load tiles into shared memory
    int a_col = t * TILE_SIZE + threadIdx.x;
    int b_row = t * TILE_SIZE + threadIdx.y;

    if (row < M && a_col < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if (b_row < K && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Compute partial sum
    for (int k = 0; k < TILE_SIZE; ++k)
    {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
}
