"""Bandwidth-audit arms for the shared ``verity-vllm bench`` driver.

``PLAIN`` installs the stack's plain kernels. ``AUDIT`` installs the encoded
and receipted stack and is gated against ``PLAIN``. Both run with the same
fill-sk16 reader over the paged KV pool and report its byte share as an extra.

The GPU libraries are built outside the lease's Python process and named by:

* ``VERITY_VLLM_BENCH_WT_LIB``
* ``VERITY_VLLM_BENCH_READER_LIB``
* ``VERITY_VLLM_BENCH_PAGED_LIB``
* ``VERITY_VLLM_BENCH_TOPUP_LIB``
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_LIB_VARS = {
    "wt_lib": "VERITY_VLLM_BENCH_WT_LIB",
    "reader_lib": "VERITY_VLLM_BENCH_READER_LIB",
    "paged_lib": "VERITY_VLLM_BENCH_PAGED_LIB",
    "topup_lib": "VERITY_VLLM_BENCH_TOPUP_LIB",
}


def _libraries() -> dict[str, str]:
    paths = {name: os.environ.get(variable, "") for name, variable in _LIB_VARS.items()}
    missing = [variable for name, variable in _LIB_VARS.items() if not paths[name]]
    if missing:
        raise RuntimeError(f"bandwidth benchmark arm needs {', '.join(missing)}")
    absent = [path for path in paths.values() if not Path(path).is_file()]
    if absent:
        raise RuntimeError(f"bandwidth benchmark arm libraries do not exist: {absent}")
    return paths


def _digest(paths: list[Path]) -> str:
    out = hashlib.sha256()
    for path in paths:
        raw_name = path.name.encode()
        out.update(len(raw_name).to_bytes(4, "big"))
        out.update(raw_name)
        with path.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                out.update(chunk)
    return out.hexdigest()


class _Installed:
    def __init__(self, target: Any, *, encoded: bool) -> None:
        from benchmarks.vllm_encoded import serve

        libs = _libraries()
        kv_receipts = os.environ.get("VERITY_VLLM_BENCH_KV_RECEIPTS", "on")
        if kv_receipts not in ("on", "off"):
            raise RuntimeError("VERITY_VLLM_BENCH_KV_RECEIPTS must be 'on' or 'off'")
        self.encoded = encoded
        self.libs = libs
        self.ctas = int(os.environ.get("VERITY_VLLM_BENCH_FILLER_CTAS", "32"))
        if self.ctas < 1:
            raise RuntimeError("VERITY_VLLM_BENCH_FILLER_CTAS must be positive")
        args = SimpleNamespace(
            arm="stack" if encoded else "pstack",
            k=int(os.environ.get("VERITY_VLLM_BENCH_CODEC_ROUNDS", "3")),
            kv_receipts=encoded and kv_receipts == "on",
            select_trials=int(os.environ.get("VERITY_VLLM_BENCH_SELECT_TRIALS", "5")),
            wt_lib=libs["wt_lib"],
            reader_lib=libs["reader_lib"],
            paged_lib=libs["paged_lib"],
        )
        self.install = serve.Install(args)
        serve.hook_runner(target.hooks, self.install)
        self.topup: Any | None = None
        self.snapshots: list[dict[str, float]] = []
        self.closed = False
        self.kernel = ""

    def ready(self, target: Any) -> None:
        import torch

        from benchmarks.kv_filler.bench import FILL_WARPS, RUN_SEED, Topup, mode_of
        from verity_vllm.engine.engine_profile import get_model_runner

        if self.install.stack is None:
            raise RuntimeError("bandwidth benchmark arm did not install during model loading")
        raw = target.engine._engine.llm
        runner = get_model_runner(raw)
        tensors = []
        for cache in runner.kv_caches:
            tensors.extend(cache if isinstance(cache, (list, tuple)) else (cache,))
        seen: set[int] = set()
        regions = []
        for tensor in tensors:
            if tensor.data_ptr() not in seen:
                seen.add(tensor.data_ptr())
                regions.append((tensor.data_ptr(), tensor.numel() * tensor.element_size()))
        if not regions:
            raise RuntimeError("bandwidth benchmark arm found no paged KV cache regions")
        self.topup = Topup(self.libs["topup_lib"])
        first, second = torch.cuda.Stream.priority_range()
        least = max(first, second)
        torch.cuda.synchronize()
        self.topup.start(
            regions,
            mode_of("fill"),
            512 << 20,
            3,
            least,
            RUN_SEED,
            ctas=self.ctas,
            warps=FILL_WARPS["fill"],
        )
        here = Path(__file__).resolve().parent
        self.kernel = _digest([
            Path(__file__),
            here / "serve.py",
            here / "stack.py",
            here / "paged.cu",
            *(Path(self.libs[name]) for name in ("wt_lib", "reader_lib", "paged_lib", "topup_lib")),
        ])

    @property
    def stack(self) -> Any:
        if self.install.stack is None:
            raise RuntimeError("bandwidth benchmark arm is not ready")
        return self.install.stack

    def snapshot(self) -> dict[str, float]:
        if self.topup is None:
            return {}
        filler = float(sum(record[2] for record in self.topup.log()))
        weight_bytes = float(sum(linear.nbytes for linear in self.stack.linears))
        value = {
            "filler_bytes": filler,
            "passes": float(self.stack.passes),
            "registered_weight_bytes": float(self.stack.passes) * weight_bytes,
        }
        self.snapshots.append(value)
        return value

    def extras(self) -> dict[str, float]:
        if len(self.snapshots) < 2:
            return {}
        before, after = self.snapshots[-2:]
        filler = after["filler_bytes"] - before["filler_bytes"]
        registered = after["registered_weight_bytes"] - before["registered_weight_bytes"]
        total = filler + registered
        return {
            "filler_bytes": filler,
            "registered_weight_bytes": registered,
            "filler_share": filler / total if total > 0 else 0.0,
            "receipt_passes": after["passes"] - before["passes"],
            "filler_ctas": float(self.ctas),
        }

    def gate(self) -> dict[str, Any]:
        gates = self.install.record.get("gates", {})
        return {
            "passed": bool(gates.get("layers") and gates.get("step")),
            "layers": int(gates.get("layers", 0)),
            "kv_receipts": bool(self.stack.kv_receipts),
            "filler_ctas": self.ctas,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.topup is not None:
            self.topup.stop()


class _ArmModule:
    precision = "fp8"
    protocols: tuple[str, ...] = ()
    resident = False

    def __init__(self, arm_id: str, *, encoded: bool, control: str | None) -> None:
        self.id = arm_id
        self.encoded = encoded
        self.control = control
        self.state: _Installed | None = None

    def install(self, target: Any) -> _Installed:
        self.state = _Installed(target, encoded=self.encoded)
        return self.state

    def gate(self, target: Any) -> dict[str, Any]:
        return self._state().gate()

    def spans(self, target: Any) -> dict[str, float]:
        return self._state().snapshot()

    def extras(self, target: Any) -> dict[str, float]:
        return self._state().extras()

    def kernel_digest(self, target: Any) -> str:
        return self._state().kernel

    def _state(self) -> _Installed:
        if self.state is None:
            raise RuntimeError(f"bandwidth benchmark arm {self.id!r} is not installed")
        return self.state


PLAIN = _ArmModule("bandwidth-plain", encoded=False, control=None)
AUDIT = _ArmModule("bandwidth-audit", encoded=True, control=PLAIN.id)

__all__ = ["AUDIT", "PLAIN"]
