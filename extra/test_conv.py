import os
import sys
import numpy as np

sys.path.insert(
  0,
  os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
)

from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device


def test_conv2d_basic():
  print("[*] Testing conv2d basic")

  input_data = np.arange(16, dtype=np.float32)
  weight_data = np.ones((1, 1, 3, 3), dtype=np.float32)

  # Known mathematical result.
  expected = np.array(
    [[[
      [45.0, 54.0],
      [81.0, 90.0],
    ]]],
    dtype=np.float32,
  )

  # Eager CPU reference.
  cpu_device = Device(Devices.CPU)

  cpu_input = Tensor(
    input_data,
    lazy=False,
    device=cpu_device,
    name="cpu_input",
  ).reshape((1, 1, 4, 4))

  cpu_weight = Tensor(
    weight_data,
    lazy=False,
    device=cpu_device,
    name="cpu_weight",
  )

  cpu_out = cpu_input.conv2d(
    cpu_weight,
    in_channels=1,
    out_channels=1,
  ).data

  print("CPU output:")
  print(cpu_out)

  np.testing.assert_allclose(
    cpu_out,
    expected,
    rtol=1e-5,
    atol=1e-5,
  )

  # Lazy Metal implementation.
  metal_device = Device(Devices.METAL)

  metal_input = Tensor(
    input_data,
    lazy=True,
    device=metal_device,
    name="metal_input",
  ).reshape((1, 1, 4, 4))

  metal_weight = Tensor(
    weight_data,
    lazy=True,
    device=metal_device,
    name="metal_weight",
  )

  metal_out_tensor = metal_input.conv2d(
    metal_weight,
    in_channels=1,
    out_channels=1,
  )

  # Important: numpy() realizes the lazy graph.
  metal_out = metal_out_tensor.numpy()

  print("Metal output:")
  print(metal_out)

  np.testing.assert_allclose(
    metal_out,
    cpu_out,
    rtol=1e-5,
    atol=1e-5,
  )

  print("[+] conv2d basic passed")


if __name__ == "__main__":
  test_conv2d_basic()
