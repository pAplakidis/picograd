#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# setup import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device
from picograd.backend.linearizer import *


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

device = None if IN_GITHUB_ACTIONS else Device(Devices.METAL)
lazy = True
serial = True
if device is not None:
  print("[*] Using device", device.name, "\n")


@unittest.skipIf(IN_GITHUB_ACTIONS, "GPU compiler tests require Metal/CUDA")
class TestCompilerOps(unittest.TestCase):
  def setUp(self):
    np.random.seed(42)

  def generate_square_tensors(self):
    a = Tensor.random((32, 32), lazy=lazy, device=device, name="a")
    b = Tensor.random((32, 32), lazy=lazy, device=device, name="b")
    c = Tensor.random((32, 32), lazy=lazy, device=device, name="c")
    return a, b, c

  def assert_tensor_equal(self, tensor, expected, atol=1e-5, rtol=1e-5):
    actual = tensor.numpy()

    self.assertEqual(
      actual.shape,
      expected.shape,
      f"Shape mismatch: got {actual.shape}, expected {expected.shape}",
    )
    self.assertTrue(np.all(np.isfinite(actual)), "Tensor contains NaN or Inf values")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

  # ---------------------------------------------------------
  # NumPy reference implementations
  # ---------------------------------------------------------

  @staticmethod
  def numpy_conv2d(x, weight, stride=1, padding=0, dilation=1, bias=None):
    """
    Independent vectorized NumPy Conv2D reference.

    Input:
      x:      (B, IC, H, W)
      weight: (OC, IC, KH, KW)

    Output:
      (B, OC, OH, OW)

    Uses:
      np.pad
      np.lib.stride_tricks.sliding_window_view
      np.einsum
    """

    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)

    B, IC, H, W = x.shape
    OC, weight_ic, KH, KW = weight.shape

    assert IC == weight_ic, f"Input channels {IC} do not match weight channels {weight_ic}"

    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    if pad_h > 0 or pad_w > 0:
      x = np.pad(
        x,
        (
          (0, 0),
          (0, 0),
          (pad_h, pad_h),
          (pad_w, pad_w),
        ),
        mode="constant",
        constant_values=0,
      )

    effective_kh = dilation_h * (KH - 1) + 1
    effective_kw = dilation_w * (KW - 1) + 1

    windows = np.lib.stride_tricks.sliding_window_view(
      x,
      window_shape=(effective_kh, effective_kw),
      axis=(-2, -1),
    )

    windows = windows[:, :, ::stride_h, ::stride_w, :, :]
    windows = windows[:, :, :, :, ::dilation_h, ::dilation_w]

    assert windows.shape[-2:] == (KH, KW), (
      f"Reference convolution produced kernel windows "
      f"{windows.shape[-2:]}, expected {(KH, KW)}"
    )

    # windows: B, IC, OH, OW, KH, KW
    # weight:  OC, IC, KH, KW
    # output:  B, OC, OH, OW
    out = np.einsum("bihwkl,oikl->bohw", windows, weight, optimize=True)

    if bias is not None:
      bias = np.asarray(bias, dtype=np.float32)
      out = out + bias.reshape(1, OC, 1, 1)

    return out.astype(np.float32, copy=False)

  def assert_lazy_conv_matches_numpy(
    self,
    input_data,
    weight_data,
    stride=1,
    padding=0,
    dilation=1,
    bias_data=None,
    atol=1e-4,
    rtol=1e-4,
  ):
    input_data = np.asarray(input_data, dtype=np.float32)
    weight_data = np.asarray(weight_data, dtype=np.float32)

    if bias_data is not None:
      bias_data = np.asarray(bias_data, dtype=np.float32)

    OC, IC, _, _ = weight_data.shape

    expected = self.numpy_conv2d(
      input_data,
      weight_data,
      stride=stride,
      padding=padding,
      dilation=dilation,
      bias=bias_data,
    )

    input_tensor = Tensor(input_data.copy(), lazy=True, device=device, name="conv_input")
    weight_tensor = Tensor(weight_data.copy(), lazy=True, device=device, name="conv_weight")

    bias_tensor = None
    if bias_data is not None:
      bias_tensor = Tensor(bias_data.copy(), lazy=True, device=device, name="conv_bias")

    output_tensor = input_tensor.conv2d(
      weight_tensor,
      in_channels=IC,
      out_channels=OC,
      stride=stride,
      padding=padding,
      bias=bias_tensor,
    )

    actual = output_tensor.numpy()

    self.assertEqual(
      actual.shape,
      expected.shape,
      f"Conv2D shape mismatch: got {actual.shape}, expected {expected.shape}",
    )
    self.assertTrue(np.all(np.isfinite(actual)), "Conv2D output contains NaN or Inf")

    np.testing.assert_allclose(
      actual,
      expected,
      atol=atol,
      rtol=rtol,
      err_msg=(
        f"Lazy Conv2D mismatch\n"
        f"input={input_data.shape}\n"
        f"weight={weight_data.shape}\n"
        f"stride={stride}\n"
        f"padding={padding}"
      ),
    )

    return actual, expected

  # --------------- Elementwise Ops ----------------

  def test_add(self):
    a, b, _ = self.generate_square_tensors()
    d = a + b
    check = a.data + b.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler add OK")

  def test_mul_add(self):
    a, b, c = self.generate_square_tensors()
    d = a * b + c
    check = a.data * b.data + c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler mul + add OK")

  def test_mul_add_shared_input(self):
    a, b, c = self.generate_square_tensors()
    d = a * b + a * c
    check = a.data * b.data + a.data * c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler shared input graph OK")

  def test_repeated_expression(self):
    a, b, c = self.generate_square_tensors()
    d = a * b + a * b + c
    check = a.data * b.data + a.data * b.data + c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler repeated expression OK")

  def test_residual(self):
    a, b, _ = self.generate_square_tensors()
    d = a * b + a
    check = a.data * b.data + a.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler residual OK")

  # --------------- Unary Ops ----------------

  def test_unsqueeze(self):
    a = Tensor.random((2, 3, 4), lazy=lazy, device=device, name="a")
    cases = [
      (0, (1, 2, 3, 4)),
      (1, (2, 1, 3, 4)),
      (2, (2, 3, 1, 4)),
      (3, (2, 3, 4, 1)),
      (-1, (2, 3, 4, 1)),
      (-2, (2, 3, 1, 4)),
      (-3, (2, 1, 3, 4)),
      (-4, (1, 2, 3, 4)),
    ]
    for axis, expected_shape in cases:
      with self.subTest(axis=axis):
        out = a.unsqueeze(axis)
        self.assertEqual(out.shape, expected_shape)
    print("[+] Compiler unsqueeze shapes OK")

  def test_unsqueeze_strides(self):
    a = Tensor.random((2, 3, 4), lazy=lazy, device=device, name="a")
    out = a.unsqueeze(-1)
    self.assertEqual(out.shape, (2, 3, 4, 1))
    self.assertEqual(out.strides, (12, 4, 1, 0))
    out = a.unsqueeze(-3)
    self.assertEqual(out.shape, (2, 1, 3, 4))
    self.assertEqual(out.strides, (12, 0, 4, 1))
    print("[+] Compiler unsqueeze strides OK")

  # --------------- Reduce Ops ----------------

  def test_sum_axis_0(self):
    a, _, _ = self.generate_square_tensors()
    d = a.sum(axis=0)
    check = a.data.sum(axis=0)

    self.assert_tensor_equal(d, check)
    print("[+] Compiler reduce axis=0 OK")

  def test_sum_axis_1(self):
    a, _, _ = self.generate_square_tensors()
    d = a.sum(axis=1)
    check = a.data.sum(axis=1)

    self.assert_tensor_equal(d, check)
    print("[+] Compiler reduce axis=1 OK")

  # --------------- Matmul Ops ----------------

  def test_matmul_square(self):
    a, b, _ = self.generate_square_tensors()
    d = a @ b
    check = a.data @ b.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler square matmul OK")

  def test_dot_square(self):
    a, b, _ = self.generate_square_tensors()
    d = a.dot(b)
    check = a.data @ b.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler square dot OK")

  def test_matmul_non_square_cases(self):
    shapes = [
      ((2, 4), (4, 3)),
      ((8, 16), (16, 4)),
      ((5, 7), (7, 2)),
      ((1, 32), (32, 1)),
      ((64, 8), (8, 128)),
    ]

    for shape_a, shape_b in shapes:
      with self.subTest(shape_a=shape_a, shape_b=shape_b):
        a = Tensor.random(shape_a, lazy=lazy, device=device, name="a")
        b = Tensor.random(shape_b, lazy=lazy, device=device, name="b")

        d = a @ b
        check = a.data @ b.data

        self.assert_tensor_equal(d, check)

    print("[+] Compiler non-square matmul OK")

  def test_dot_non_square_cases(self):
    shapes = [
      ((2, 4), (4, 3)),
      ((8, 16), (16, 4)),
      ((5, 7), (7, 2)),
      ((1, 32), (32, 1)),
      ((64, 8), (8, 128)),
    ]

    for shape_a, shape_b in shapes:
      with self.subTest(shape_a=shape_a, shape_b=shape_b):
        a = Tensor.random(shape_a, lazy=lazy, device=device, name="a")
        b = Tensor.random(shape_b, lazy=lazy, device=device, name="b")

        d = a.dot(b)
        check = a.data @ b.data

        self.assert_tensor_equal(d, check)

    print("[+] Compiler non-square dot OK")

  def test_matmul_add(self):
    a, b, c = self.generate_square_tensors()
    d = a @ b + c
    check = a.data @ b.data + c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler matmul + add OK")

  # =========================================================
  # Conv2D
  # =========================================================

  def test_conv2d_known_values(self):
    """
    Hand-verifiable basic convolution.

    Catches:
      - _pool indexing
      - kernel indexing
      - reductions
      - realization
      - output shape
    """

    input_data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    weight_data = np.ones((1, 1, 3, 3), dtype=np.float32)

    expected = np.array(
      [[[
        [45.0, 54.0],
        [81.0, 90.0],
      ]]],
      dtype=np.float32,
    )

    numpy_result = self.numpy_conv2d(input_data, weight_data)
    np.testing.assert_array_equal(numpy_result, expected)

    actual, _ = self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    print("[+] Compiler conv2d known-values OK")

  def test_conv2d_1x1(self):
    input_data = np.random.randn(2, 3, 6, 7).astype(np.float32)
    weight_data = np.random.randn(5, 3, 1, 1).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d 1x1 OK")

  def test_conv2d_multi_input_channel(self):
    """
    Explicitly tests reduction across IC.

    Removing:
      out.sum(axis=2)

    should make this test fail.
    """

    input_data = np.arange(2 * 5 * 5, dtype=np.float32).reshape(1, 2, 5, 5) / 10.0
    weight_data = np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3) / 10.0

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d multi-input-channel OK")

  def test_conv2d_multi_output_channel(self):
    input_data = np.random.randn(1, 2, 6, 6).astype(np.float32)
    weight_data = np.random.randn(5, 2, 3, 3).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d multi-output-channel OK")

  def test_conv2d_batch(self):
    input_data = np.random.randn(3, 2, 7, 7).astype(np.float32)
    weight_data = np.random.randn(4, 2, 3, 3).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d batch OK")

  def test_conv2d_stride_2(self):
    input_data = np.random.randn(2, 3, 9, 9).astype(np.float32)
    weight_data = np.random.randn(5, 3, 3, 3).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data, stride=2)
    print("[+] Compiler conv2d stride=2 OK")

  def test_conv2d_stride_3(self):
    input_data = np.random.randn(1, 2, 11, 13).astype(np.float32)
    weight_data = np.random.randn(3, 2, 3, 3).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data, stride=3)
    print("[+] Compiler conv2d stride=3 OK")

  def test_conv2d_non_square_input(self):
    input_data = np.random.randn(2, 2, 7, 11).astype(np.float32)
    weight_data = np.random.randn(3, 2, 3, 3).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d non-square input OK")

  def test_conv2d_rectangular_kernel(self):
    input_data = np.random.randn(1, 2, 8, 11).astype(np.float32)
    weight_data = np.random.randn(4, 2, 3, 2).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d rectangular kernel OK")

  def test_conv2d_even_kernel(self):
    """
    Explicit regression test against the old CPU Conv2D odd-kernel restriction.
    """

    input_data = np.random.randn(2, 3, 8, 9).astype(np.float32)
    weight_data = np.random.randn(4, 3, 2, 4).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d even kernel OK")

  def test_conv2d_large_kernel(self):
    input_data = np.random.randn(1, 3, 12, 13).astype(np.float32)
    weight_data = np.random.randn(4, 3, 5, 5).astype(np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    print("[+] Compiler conv2d 5x5 OK")

  def test_conv2d_bias(self):
    input_data = np.random.randn(2, 3, 7, 8).astype(np.float32)
    weight_data = np.random.randn(4, 3, 3, 3).astype(np.float32)
    bias_data = np.array([0.25, -0.5, 1.5, -2.0], dtype=np.float32)

    self.assert_lazy_conv_matches_numpy(input_data, weight_data, bias_data=bias_data)
    print("[+] Compiler conv2d bias OK")

  def test_conv2d_zero_input(self):
    input_data = np.zeros((2, 3, 7, 7), dtype=np.float32)
    weight_data = np.random.randn(4, 3, 3, 3).astype(np.float32)

    actual, expected = self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    np.testing.assert_array_equal(actual, np.zeros_like(expected))

    print("[+] Compiler conv2d zero input OK")

  def test_conv2d_zero_kernel(self):
    input_data = np.random.randn(2, 3, 7, 7).astype(np.float32)
    weight_data = np.zeros((4, 3, 3, 3), dtype=np.float32)

    actual, expected = self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    np.testing.assert_array_equal(actual, np.zeros_like(expected))

    print("[+] Compiler conv2d zero kernel OK")

  def test_conv2d_identity_1x1(self):
    input_data = np.random.randn(2, 1, 6, 9).astype(np.float32)
    weight_data = np.ones((1, 1, 1, 1), dtype=np.float32)

    actual, _ = self.assert_lazy_conv_matches_numpy(input_data, weight_data)
    np.testing.assert_allclose(actual, input_data, atol=1e-6, rtol=1e-6)

    print("[+] Compiler conv2d 1x1 identity OK")

  # TODO: Enable when lazy Conv2D padding is implemented.
  #
  # def test_conv2d_padding_1_known_values(self):
  #   input_data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
  #   weight_data = np.ones((1, 1, 3, 3), dtype=np.float32)
  #
  #   expected = np.array(
  #     [[[
  #       [10.0, 18.0, 24.0, 18.0],
  #       [27.0, 45.0, 54.0, 39.0],
  #       [51.0, 81.0, 90.0, 63.0],
  #       [42.0, 66.0, 72.0, 50.0],
  #     ]]],
  #     dtype=np.float32,
  #   )
  #
  #   numpy_result = self.numpy_conv2d(input_data, weight_data, padding=1)
  #   np.testing.assert_array_equal(numpy_result, expected)
  #
  #   actual, _ = self.assert_lazy_conv_matches_numpy(input_data, weight_data, padding=1)
  #   np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
  #
  #   print("[+] Compiler conv2d padding=1 known-values OK")

  # TODO: Enable when lazy Conv2D padding is implemented.
  #
  # def test_conv2d_padding_multi_channel(self):
  #   input_data = np.random.randn(2, 3, 8, 9).astype(np.float32)
  #   weight_data = np.random.randn(5, 3, 3, 3).astype(np.float32)
  #
  #   self.assert_lazy_conv_matches_numpy(input_data, weight_data, padding=1)
  #   print("[+] Compiler conv2d padded multi-channel OK")

  # TODO: Enable when lazy Conv2D padding is implemented.
  #
  # def test_conv2d_padding_with_bias(self):
  #   input_data = np.random.randn(2, 3, 8, 9).astype(np.float32)
  #   weight_data = np.random.randn(5, 3, 3, 3).astype(np.float32)
  #   bias_data = np.random.randn(5).astype(np.float32)
  #
  #   self.assert_lazy_conv_matches_numpy(
  #     input_data,
  #     weight_data,
  #     padding=1,
  #     bias_data=bias_data,
  #   )
  #
  #   print("[+] Compiler conv2d padding + bias OK")

  # TODO: Enable when lazy Conv2D padding is implemented.
  #
  # def test_conv2d_stride_and_padding(self):
  #   input_data = np.random.randn(2, 2, 9, 11).astype(np.float32)
  #   weight_data = np.random.randn(4, 2, 3, 3).astype(np.float32)
  #
  #   self.assert_lazy_conv_matches_numpy(input_data, weight_data, stride=2, padding=1)
  #   print("[+] Compiler conv2d stride + padding OK")

  def test_conv2d_random_cases(self):
    """
    Randomized regression matrix.

    Padding cases are excluded until lazy padding is implemented.
    """

    cases = [
      # B, IC, OC, H, W, KH, KW, stride, padding
      (1, 1, 1, 5, 5, 3, 3, 1, 0),
      (2, 1, 3, 6, 7, 2, 2, 1, 0),
      (1, 3, 4, 8, 9, 3, 3, 2, 0),
      (3, 2, 5, 7, 10, 3, 2, 1, 0),
      (2, 4, 3, 9, 8, 1, 1, 1, 0),
      (1, 2, 4, 10, 11, 5, 3, 1, 0),

      # TODO: Enable when lazy Conv2D padding is implemented.
      # (1, 1, 2, 5, 5, 3, 3, 1, 1),
      # (2, 3, 4, 8, 9, 3, 3, 2, 1),
      # (1, 2, 3, 7, 10, 2, 4, 1, 2),
    ]

    for B, IC, OC, H, W, KH, KW, stride, padding in cases:
      with self.subTest(
        B=B,
        IC=IC,
        OC=OC,
        H=H,
        W=W,
        KH=KH,
        KW=KW,
        stride=stride,
        padding=padding,
      ):
        input_data = np.random.randn(B, IC, H, W).astype(np.float32)
        weight_data = np.random.randn(OC, IC, KH, KW).astype(np.float32)

        self.assert_lazy_conv_matches_numpy(
          input_data,
          weight_data,
          stride=stride,
          padding=padding,
        )

    print("[+] Compiler randomized conv2d cases OK")

  def test_conv2d_output_shapes(self):
    cases = [
      # input, kernel, stride, padding, expected H/W
      ((1, 1, 4, 4), (1, 1, 3, 3), 1, 0, (2, 2)),
      ((2, 3, 9, 9), (5, 3, 3, 3), 2, 0, (4, 4)),
      ((1, 2, 8, 10), (4, 2, 2, 3), 1, 0, (7, 8)),

      # TODO: Enable when lazy Conv2D padding is implemented.
      # ((1, 1, 4, 4), (1, 1, 3, 3), 1, 1, (4, 4)),
      # ((2, 3, 9, 9), (5, 3, 3, 3), 2, 1, (5, 5)),
    ]

    for input_shape, kernel_shape, stride, padding, expected_hw in cases:
      with self.subTest(
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        stride=stride,
        padding=padding,
      ):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*kernel_shape).astype(np.float32)

        expected = self.numpy_conv2d(
          input_data,
          weight_data,
          stride=stride,
          padding=padding,
        )

        input_tensor = Tensor(input_data, lazy=True, device=device)
        weight_tensor = Tensor(weight_data, lazy=True, device=device)

        output = input_tensor.conv2d(
          weight_tensor,
          in_channels=kernel_shape[1],
          out_channels=kernel_shape[0],
          stride=stride,
          padding=padding,
        )

        expected_shape = (
          input_shape[0],
          kernel_shape[0],
          expected_hw[0],
          expected_hw[1],
        )

        self.assertEqual(expected.shape, expected_shape)
        self.assertEqual(output.shape, expected_shape)

    print("[+] Compiler conv2d output shapes OK")

  def test_conv2d_repeated_realize(self):
    input_data = np.random.randn(1, 2, 6, 6).astype(np.float32)
    weight_data = np.random.randn(3, 2, 3, 3).astype(np.float32)

    expected = self.numpy_conv2d(input_data, weight_data)

    input_tensor = Tensor(input_data, lazy=True, device=device, name="input")
    weight_tensor = Tensor(weight_data, lazy=True, device=device, name="weight")

    output = input_tensor.conv2d(
      weight_tensor,
      in_channels=2,
      out_channels=3,
    )

    first = output.numpy().copy()
    second = output.numpy().copy()

    np.testing.assert_allclose(first, expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(second, expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)

    print("[+] Compiler conv2d repeated realize OK")


if __name__ == "__main__":
  if serial:
    runner = unittest.TextTestRunner(failfast=True, verbosity=2)
    unittest.main(testRunner=runner)
  else:
    unittest.main(verbosity=2)
