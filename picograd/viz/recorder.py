import atexit
import enum
import inspect
import json
import os
import sys
import time
import math
import hashlib
from datetime import datetime, timezone

import numpy as np

_START = time.perf_counter()
_ENABLED = os.environ.get("VIZ") == "1"
_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "traces", "latest.json")
_PATH = os.environ.get("PICOGRAD_VIZ_PATH", _DEFAULT_PATH)
_EVENTS = []


def enabled():
  return _ENABLED


def trace_path():
  return _PATH


def _tensor(value):
  return value.__class__.__name__ == "Tensor" and hasattr(value, "_shape")


def _safe(value, depth=0):
  if depth > 6:
    return repr(value)
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if value is None or isinstance(value, (bool, int, str)):
    return value
  if isinstance(value, bytes):
    return value.decode("utf-8", "replace")
  if isinstance(value, enum.Enum):
    return {"type": value.__class__.__name__, "name": value.name, "value": _safe(value.value, depth + 1)}
  if isinstance(value, np.ndarray):
    return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "size": int(value.size), "nbytes": int(value.nbytes)}
  if isinstance(value, np.generic):
    return _safe(value.item(), depth + 1)
  if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
    return {"type": "function", "name": getattr(value, "__qualname__", getattr(value, "__name__", repr(value))), "module": getattr(value, "__module__", None)}
  if _tensor(value):
    return {
      "type": "Tensor",
      "id": id(value),
      "shape": _safe(getattr(value, "shape", None), depth + 1),
      "dtype": _safe(getattr(value, "dtype", None), depth + 1),
      "device": _safe(getattr(value, "device", None), depth + 1),
      "requires_grad": _safe(getattr(value, "requires_grad", None), depth + 1),
    }
  if isinstance(value, dict):
    return {str(_safe(k, depth + 1)): _safe(v, depth + 1) for k, v in value.items()}
  if isinstance(value, (list, tuple, set)):
    return [_safe(v, depth + 1) for v in value]
  try:
    json.dumps(value)
    return value
  except TypeError:
    return repr(value)


def _hash_text(value):
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata():
  return {
    "format": "picograd-viz-trace",
    "version": 1,
    "enabled": _ENABLED,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "pid": os.getpid(),
    "python": sys.version,
    "path": _PATH,
    "event_count": len(_EVENTS),
  }


def record(kind, **data):
  if not _ENABLED:
    return None
  if isinstance(data.get("source"), str):
    data.setdefault("source_hash", _hash_text(data["source"]))
  event = {"kind": str(kind), "ts_ms": round((time.perf_counter() - _START) * 1000, 3), "data": _safe(data)}
  _EVENTS.append(event)
  return event


def record_kernel(name=None, source=None, device=None, args=None, global_size=None, local_size=None, **data):
  return record("kernel", name=name, source=source, device=device, args=args, global_size=global_size, local_size=local_size, **data)


def clear():
  _EVENTS.clear()


def flush(path=None):
  if not _ENABLED:
    return None
  out = path or _PATH
  directory = os.path.dirname(out)
  if directory:
    os.makedirs(directory, exist_ok=True)
  payload = {"metadata": _metadata(), "events": _EVENTS}
  tmp = out + ".tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, allow_nan=False, indent=2, sort_keys=True)
    f.write("\n")
  os.replace(tmp, out)
  return out


atexit.register(flush)
