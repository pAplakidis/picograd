// Kernel name: transpose_back_kernel
// in_ptr  : const float*  (grad_out, permuted layout)
// out_ptr : float*        (a.grad, original layout; will be atomically += )
// shape   : const long long*  (original a.shape, length ndim)
// in_stride : const long long* (strides for grad_out layout, i.e. permuted strides)
// out_stride: const long long* (strides for a.grad layout, i.e. original strides)
// axes    : const int*    (forward permutation axes, length ndim)
// ndim    : int
// total_elems : long long  (product of original shape elements)
extern "C" __global__ void transpose_back_kernel(
    const float *__restrict__ in_ptr,
    float *__restrict__ out_ptr,
    const long long *__restrict__ shape,
    const long long *__restrict__ in_stride,
    const long long *__restrict__ out_stride,
    const int *__restrict__ axes,
    const int ndim,
    const long long total_elems)
{
  // 1D global thread index
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_elems)
    return;

  // decode gid -> coords in the ORIGINAL layout (row-major semantics described by 'shape')
  // We decode from least-significant dimension last: using modulo/division.
  // coords[0..ndim-1] correspond to dimension order used with out_stride.
  // We allocate small local array on stack; limit ndim to reasonable number (e.g. <= 12).
  const int MAXD = 12;
  long long coords_local[MAXD]; // ensure MAXD >= max possible ndim
  // If ndim > MAXD you'll need to increase MAXD at compile time.
  long long tmp = gid;
  for (int d = ndim - 1; d >= 0; --d)
  {
    long long dim = shape[d];
    coords_local[d] = tmp % dim;
    tmp = tmp / dim;
  }

  // compute linear indices
  long long in_lin = 0;
  long long out_lin = 0;

  // in_ptr expects coordinates in permuted order: coord_perm[k] = coords[ axes[k] ]
  // so: in_lin = sum_k ( coords[ axes[k] ] * in_stride[k] )
  // out_lin = sum_k ( coords[k] * out_stride[k] )
  for (int k = 0; k < ndim; ++k)
  {
    long long c_out = coords_local[k];
    out_lin += c_out * out_stride[k];

    int axis_for_k = axes[k];                  // which original dim maps to position k in permuted tensor
    long long c_in = coords_local[axis_for_k]; // permuted coordinate value
    in_lin += c_in * in_stride[k];
  }

  // atomic add (float)
  atomicAdd(&out_ptr[out_lin], in_ptr[in_lin]);
}
