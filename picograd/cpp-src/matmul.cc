#include <iostream>

void matmul(double **a, int *size_a, double **b, int *size_b, double **c) {
  if (size_a[1] != size_b[0]) {
    std::cout << "size_a[1] must be equal to size_b[0]" << std::endl;
    return;
  }

  for (int i = 0; i < size_a[0]; i++) {
    for (int j = 0; j < size_b[1]; j++) {
      c[i][j] = 0;
      for (int k = 0; k < size_a[1]; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void test_matmul() {
  // A (2x3)
  int size_a[2] = {2, 3};
  double **a = new double *[size_a[0]];
  a[0] = new double[size_a[1]]{1, 2, 3};
  a[1] = new double[size_a[1]]{4, 5, 6};

  // B (3x2)
  int size_b[2] = {3, 2};
  double **b = new double *[size_b[0]];
  b[0] = new double[size_b[1]]{7, 8};
  b[1] = new double[size_b[1]]{9, 10};
  b[2] = new double[size_b[1]]{11, 12};

  // C (2x2)
  int size_c[2] = {size_a[0], size_b[1]};
  double **c = new double *[size_c[0]];
  for (int i = 0; i < size_c[0]; i++) {
    c[i] = new double[size_c[1]]{0};
  }

  matmul(a, size_a, b, size_b, c);
  std::cout << "Resultant matrix C:" << std::endl;
  for (int i = 0; i < size_c[0]; i++) {
    for (int j = 0; j < size_c[1]; j++) {
      std::cout << c[i][j] << " ";
    }
    std::cout << std::endl;
  }

  for (int i = 0; i < size_a[0]; i++)
    delete[] a[i];
  delete[] a;
  for (int i = 0; i < size_b[0]; i++)
    delete[] b[i];
  delete[] b;
  for (int i = 0; i < size_c[0]; i++)
    delete[] c[i];
  delete[] c;
}
