# CUDA Lazy Metal Parity Plan

## Goal

Make lazy CUDA path match current lazy Metal behavior for tests and `examples/MNIST_simple.py`, then verify on real CUDA machine.

## Current Risk Summary

- CUDA likely has both memory leaks and correctness bugs.
- Local machine has no CUDA (`is_cuda_available() == False`), so final validation must run on CUDA host.
- Avoid Metal-style pipeline cache by kernel name; generated kernels are stride-specialized and same names can have different source.

## Main Issues

1. CUDA compile cache keyed by kernel name only.
   - File: `picograd/backend/cuda/cuda.py`
   - Current: `self.kernels[kernel_name] = kfunc`
   - Problem: lazy renderer emits stride-specialized source under same kernel name, e.g. `E_ADD_16_128` can differ by strides.
   - Fix: key compile cache by source hash plus kernel name, not name alone.

2. CUDA module leak.
   - File: `picograd/backend/cuda/cuda.py`
   - Current: `self.module` overwritten on every compile; old modules not unloaded.
   - Fix: store module per cached kernel and unload all modules in `__del__`.

3. Device buffer ownership is unsafe.
   - File: `picograd/tensor.py`
   - Current: `__del__` frees `_device_data` for every tensor.
   - Problem: `detach()`, movement ops, `_pool()`, reshape/view/expand/transpose share same `device_data`; CUDA can double-free or free parent buffer while views still use it.
   - Fix: add `_owns_device_data` flag. Only owning tensors free data. Views/detach/shared-buffer movement tensors set `_owns_device_data=False`.

4. `free_device_tensor(None)` hazard.
   - Files: `picograd/tensor.py`, `picograd/backend/cuda/cuda.py`
   - Current: `device_data` setter calls `free_device_tensor(self._device_data)` unguarded.
   - Fix: guard `None` in both setter and CUDA `free_device_tensor`.

5. CUDA launch geometry mismatch.
   - Files: `picograd/backend/scheduler.py`, `picograd/backend/renderer/cuda_renderer.py`
   - Current scheduler launches `grid=(numel,1,1), block=(1024,1,1)` for CUDA-style kernels.
   - Problem: CUDA renderer sometimes uses global thread id and sometimes uses `blockIdx.x`; this causes massive overlaunch or incorrect indexing.
   - Fix: standardize CUDA renderer on `idx = blockIdx.x * blockDim.x + threadIdx.x`, and scheduler on `grid=((numel + block - 1)//block, 1, 1)`.

6. CUDA renderer contiguous path bug.
   - File: `picograd/backend/renderer/cuda_renderer.py`
   - Current elementwise contiguous path computes global id then overwrites it with `blockIdx.x`.
   - Fix: remove second assignment; use global id consistently.

7. CUDA static path is weaker than lazy path.
   - File: `picograd/backend/cuda/ops.py`
   - Risks: old kernels ignore strides/views, `cre_back.cu` launch is likely wrong, comments mention illegal address in `relu_back`.
   - Scope: focus lazy compiler path first; do not rely on static CUDA CE/backward for MNIST lazy.

8. Labels dtype can break static CUDA CE.
   - File: `picograd/tensor.py`
   - Current `to(CUDA)` casts all data to float32.
   - Risk: labels become float but CUDA CE kernels expect `int*`.
   - Scope: lazy CE reads labels on host; static CUDA CE needs separate dtype fix if targeted.

## Implementation Steps

1. Add device ownership metadata in `Tensor`.
   - Add constructor arg `_owns_device_data=True`.
   - Set `self._owns_device_data = _owns_device_data`.
   - In `__del__`, free `_device_data` only if owned.
   - In `device_data` setter, free old value only if owned and old is not `None`.
   - New allocated outputs own buffers.
   - Shared-buffer views/detach/movement ops set `_owns_device_data=False`.

2. Harden CUDA frees.
   - In `CudaDeviceManager.free_device_tensor`, return immediately if pointer is `None`.
   - Consider setting pointer to `None` after free in calling code.

3. Fix CUDA compile/module cache.
   - Add `hashlib` import.
   - Store `self.kernels = {cache_key: (module, function)}` or separate `self.modules = {cache_key: module}`.
   - `cache_key = (kernel_name, sha256(src.encode()).digest())`.
   - On hit, return cached function.
   - In `__del__`, unload every stored module exactly once.
   - Do not cache by kernel name alone.

4. Fix CUDA renderer elementwise indexing.
   - Use one global id assignment.
   - Contiguous and non-contiguous branches both use same global id.
   - Ensure kernel names still include shape but cache key protects stride-specialized source.

5. Fix scheduler launch geometry for CUDA.
   - In `run_elementwise_kernel`, `run_unary_kernel`, `run_rows_kernel`, `run_reduce_kernel`:
   - If `self.mngr.dev_name == "CUDA"`, use `block=(min(256 or max_block_size[0], numel),1,1)` and `grid=(ceil(numel/block[0]),1,1)`.
   - Keep Metal behavior unchanged if currently passing tests.

6. Add CUDA selectable lazy tests.
   - File: `tests/test_lazy.py`
   - Allow env override: `PICOGRAD_TEST_DEVICE=CUDA|METAL`.
   - Default stays Metal locally, skip in GitHub Actions.
   - Use same lazy tests on CUDA machine.

7. Run local non-CUDA checks.
   - `python3.12 -m unittest discover -s tests -p "test_*.py" -v`
   - Metal lazy suite if available.

8. Run CUDA machine verification.
   - `PICOGRAD_TEST_DEVICE=CUDA LAZY=1 python3.12 -m unittest tests/test_lazy.py -v`
   - `PICOGRAD_TEST_DEVICE=CUDA LAZY=1 python3.12 -m unittest discover -s tests -p "test_*.py" -v`
   - MNIST smoke:
     `LAZY=1 PICOGRAD_DEVICE=CUDA python3.12 examples/MNIST_simple.py`
   - If script lacks `PICOGRAD_DEVICE`, patch it to select device from env before running.

## Expected Failure Modes To Watch

- `CUDA_ERROR_ILLEGAL_ADDRESS`: likely bad launch geometry, stale/double-freed buffer, or static kernel path accidentally used.
- Wrong lazy matmul/sum grads: likely kernel cache source/name collision or renderer indexing bug.
- OOM after many iterations: module leak or tensor buffer ownership still wrong.
- Labels errors in CE: dtype path using static CUDA loss instead of lazy host label path.

## Validation Target

- Lazy CUDA test suite passes same assertions as lazy Metal suite.
- MNIST simple runs for at least 500 batches without OOM/illegal address.
- Loss trend uses batch mean and should decrease on real MNIST with current Adam setup.
