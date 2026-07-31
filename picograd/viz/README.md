# Picograd Viz

JSON trace recorder and static viewer with event spans, kernel source, UOps, and backend code artifacts.

## Record

Set `VIZ=1` before importing picograd. Disabled recorder calls are no-ops.

```python
from picograd.viz import record, record_kernel

record("step", name="forward")
record_kernel(name="add", device="cpu", source="kernel source")
```

Trace output defaults to `picograd/viz/traces/latest.json`. Override with `PICOGRAD_VIZ_PATH`.

```sh
VIZ=1 PICOGRAD_VIZ_PATH=/tmp/latest.json python3 script.py
```

Picograd records tensor ops, eager function timings, lazy schedules, generated kernels, compile events, launches, and copies. Kernel and compile events also carry `source_hash` and artifact payloads when available. Traces flush on process exit with `atexit`; call `flush()` for explicit writes.

## View

```sh
python3 -m picograd.viz.serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/index.html`. Viewer fetches `traces/latest.json` from this package directory.

Viewer includes:

- Performance event spans with ms labels
- Clickable kernel rows and spans
- Tabs for Source, UOps, Assembly, and raw JSON
- Search, kind, device, sort, and timeline zoom controls

For custom trace paths:

```sh
python3 -m picograd.viz.serve --trace /tmp/latest.json
```

## Format

```json
{
  "metadata": {"format": "picograd-viz-trace", "version": 1},
  "events": [{"kind": "kernel", "ts_ms": 0.0, "data": {}}]
}
```

Serializer keeps tensors, enums, functions, and unknown objects JSON-safe. Kernel source is stored in full when supplied, and `source_hash` is added automatically for trace joins.
