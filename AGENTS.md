# AGENTS.md

## Commands
- CI installs `numpy graphviz torch` and system `graphviz`; local full discovery also needs `pytest` because `tests/test_lazy.py` imports it before skipping.
- Full test command: `python3 -m unittest discover -s tests -p "test_*.py" -v`.
- Focus one file: `python3 -m unittest tests/test_ops.py -v`.
- Focus one test: `python3 -m unittest tests.test_ops.TestBinaryOps.test_add -v`.
- No `pyproject.toml`, package install metadata, lint, formatter, or typecheck config exists.

## Test Gotchas
- `tests/test_lazy.py` hardcodes `Device(Devices.METAL)` and is skipped only when `GITHUB_ACTIONS=true`; local full discovery needs Metal/pyobjc or must avoid this file.
- `tests/test_lazy.py` imports `pytest` before its GitHub Actions skip, so full discovery needs `pytest` installed even though tests run through `unittest`.
- CPU-focused suites are `tests/test_ops.py`, `tests/test_module.py`, and `tests/test_rnn.py`; they manually add repo root to `sys.path`.
- Tests import `picograd`, which imports `draw_utils`, so Python `graphviz` is required even for tests that do not draw graphs.

## Architecture
- Public package surface is `picograd/__init__.py`: `Tensor`, `Device`, `Devices`, `draw_dot`, `dtypes`.
- `picograd/tensor.py` wires autograd, eager ops, lazy realization, device movement, and env flags `DEBUG`, `VERBOSE`, `LAZY`.
- Eager CPU math lives in `picograd/backend/cpu/ops.py`; `picograd/backend/function.py` maps high-level `OPS` to backend op classes.
- Lazy/device codegen path is `Tensor.realize()` -> `backend/linearizer.py` -> `backend/scheduler.py` -> renderer under `backend/renderer/` -> CUDA or Metal device manager.
- NN layers live under `picograd/nn/`; optimizers and losses are top-level `picograd/optim.py` and `picograd/loss.py`.

## Devices And Builds
- Default device is CPU; CUDA/Metal are opt-in through explicit `Device(Devices.CUDA/METAL)`.
- `LAZY=1` only changes tensor laziness; realization needs CUDA/Metal because `Tensor.get_renderer()` has no CPU renderer.
- CUDA backend uses driver libraries `libcuda.so` and `libnvrtc.so`; `is_cuda_available()` only checks `nvidia-smi`.
- Metal backend imports `Metal` from pyobjc at module import time.
- C++ library build is deprecated and unused by current Python path; if needed, script is `picograd/build.sh`, not repo-root `build.sh`.
- Generated/runtime outputs are ignored under `graphs/`, `media/`, `lib/`, `mnist_data/`, `*.so`, and `__pycache__/`.

## Style
- Existing Python uses 2-space indentation and compact one-line methods; follow local style when editing nearby code.
