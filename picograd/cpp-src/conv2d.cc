#include <iostream>

// g++ -shared -o conv2d.so -fPIC conv2d.cc

extern "C" {
int conv2d(int in_ch, int out_ch, int kernel_size, int padding, int h_out,
           int w_out, int h, int w, float ***out_data, int out_data_size,
           float ***kernel_data, int kernel_data_size, float *b_data,
           int b_data_size, float ***data, int data_size) {
  int i_idx;
  int j_idx;

  // FIXME: segfaults
  std::cout << "Hello from C++" << std::endl;

  for (int o_c; o_c < out_ch; o_c++) {
    for (int i_c; i_c < in_ch; i_c++) {
      i_idx = 0 - padding;
      for (int i = 0; i < h_out; i++) {
        j_idx = 0 - padding;
        for (int j = 0; j < w_out; j++) {
          for (int k; k < kernel_size; k++) {
            for (int l; l < kernel_size; k++) {

              if (i_idx + k < 0 || j_idx + l < 0 || i_idx + k >= h or
                  j_idx + l >= w) {
                out_data[o_c][i][j] += b_data[o_c];
              }
              out_data[o_c][i][j] +=
                  data[i_c][i_idx + k][j_idx + l] * kernel_data[o_c][k][l] +
                  b_data[o_c];
            }
          }
        }
      }
    }
  }
  return 0;
}
}
