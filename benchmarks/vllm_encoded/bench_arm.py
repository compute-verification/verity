"""Bandwidth-audit arms for the shared ``verity-bench vllm`` driver.

``PLAIN`` installs the stack's plain kernels. ``AUDIT`` installs the encoded and receipted stack and is gated against
``PLAIN``. Both run with the same fill-sk16 top-up (the filler) over the paged KV pool and report its share of filler
plus weight-read bytes as an extra. They are the credited lines (``CREDITED``). ``BandwidthArm`` builds the ablations
under ids of their own (README.md): ``topup=False`` (or ``VERITY_VLLM_BENCH_TOPUP=off``) never starts the top-up, and
``spans=True`` (or ``VERITY_VLLM_BENCH_SPANS=on``) times each step's parts (`step_spans.py`). Either refuses credit:
under a credited id its gate fails, so the run has no line. ``topup_sms=F`` (or ``VERITY_VLLM_BENCH_TOPUP_SMS``) places
the top-up on F SMs of its own and the engine on the rest (`kv_filler/green.py`, green contexts), split before vLLM
starts; 0 interleaves it with the engine on every SM. A partition whose confinement check fails, or a run off the
step's stream, fails the gate.

The GPU libraries are built outside the lease's Python process and named by:

* ``VERITY_VLLM_BENCH_WT_LIB``
* ``VERITY_VLLM_BENCH_READER_LIB``
* ``VERITY_VLLM_BENCH_PAGED_LIB``
* ``VERITY_VLLM_BENCH_TOPUP_LIB``
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))   # last: the harness imports this module, and HERE's other names are generic

import step_spans  # noqa: E402

_LIB_VARS = {
    "wt_lib": "VERITY_VLLM_BENCH_WT_LIB",
    "reader_lib": "VERITY_VLLM_BENCH_READER_LIB",
    "paged_lib": "VERITY_VLLM_BENCH_PAGED_LIB",
    "topup_lib": "VERITY_VLLM_BENCH_TOPUP_LIB",
}
#: the lines a table may cite as the audit's cost: only with the top-up running and no spans
CREDITED = ("bandwidth-plain", "bandwidth-audit")
#: the top-up's SMs when an arm and VERITY_VLLM_BENCH_TOPUP_SMS leave them open (0: interleaved on every SM)
TOPUP_SMS = "0"


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


def _install_gate_passed(record: dict[str, Any], *, encoded: bool) -> bool:
    gates = record.get("gates", {})
    common = bool(gates.get("layers") and gates.get("configs_checked"))
    return common and (not encoded or bool(gates.get("step")))


def _switch(value: bool | None, variable: str, default: str) -> bool:
    """An arm's own setting, else its environment variable's ('on' or 'off')."""
    if value is not None:
        return bool(value)
    got = os.environ.get(variable, default)
    if got not in ("on", "off"):
        raise RuntimeError(f"{variable} must be 'on' or 'off'")
    return got == "on"


def _sms(value: int | None) -> int:
    """An arm's top-up SMs, else VERITY_VLLM_BENCH_TOPUP_SMS's, else TOPUP_SMS."""
    got = str(value) if value is not None else os.environ.get("VERITY_VLLM_BENCH_TOPUP_SMS", TOPUP_SMS)
    if not got.isdigit():
        raise RuntimeError("VERITY_VLLM_BENCH_TOPUP_SMS must be a nonnegative integer (0: interleaved)")
    return int(got)


def credit(arm_id: str, *, topup: bool, spans: bool, empty_runs: int) -> dict[str, Any]:
    """Whether the arm's line is creditable: a credited id, the top-up on and reading in every measured run, no spans.
    ``passes`` is False only for a credited id refused, whose gate then fails."""
    refused = []
    if not topup:
        refused.append("top-up off (topup=False or VERITY_VLLM_BENCH_TOPUP=off): an ablation, not the audit")
    if empty_runs:
        refused.append(f"the top-up read nothing in {empty_runs} measured run(s)")
    if spans:
        refused.append("spans on (spans=True or VERITY_VLLM_BENCH_SPANS=on): their marks are in the step's time")
    out: dict[str, Any] = {"credited": arm_id in CREDITED and not refused, "refused": refused,
                           "passes": arm_id not in CREDITED or not refused}
    if not out["passes"]:
        out["fix"] = (f"{arm_id} is a credited line: run an ablation under an id of its own, e.g. "
                      "bench_arm.AUDIT_NO_TOPUP or BandwidthArm('bandwidth-audit-…', encoded=True, "
                      "control='bandwidth-plain', topup=False)")
    return out


class _TopupLog:
    """The top-up's chunk log read on from where the last read stopped: its bytes, chunks and their GPU time."""

    def __init__(self, topup: Any) -> None:
        self.topup, self.read = topup, 0
        self.bytes = 0.0
        self.busy_ms = 0.0

    def update(self) -> None:
        if self.topup is None:
            return
        records = self.topup.log(self.read)
        self.read += len(records)
        for record in records:
            self.bytes += record[2]
            self.busy_ms += record[1] - record[0]

    def progress(self) -> int:
        if self.topup is None:
            return 0
        out = (ctypes.c_uint64 * 2)()
        self.topup.so.tu_progress(out)
        return int(out[0])


class _Installed:
    def __init__(self, target: Any, *, encoded: bool, topup: bool, spans: bool, kv_receipts: bool,
                 codec_rounds: int, topup_sms: int) -> None:
        from benchmarks.vllm_encoded import serve

        libs = _libraries()
        self.encoded, self.use_topup, self.use_spans = encoded, topup, spans
        self.libs = libs
        self.ctas = int(os.environ.get("VERITY_VLLM_BENCH_FILLER_CTAS", "32"))
        if self.ctas < 1:
            raise RuntimeError("VERITY_VLLM_BENCH_FILLER_CTAS must be positive")
        self.topup_sms, self.green, self.step_stream, self.off_step_runs = topup_sms, None, None, 0
        if topup_sms:
            self._partition(topup_sms)
        args = SimpleNamespace(
            arm="stack" if encoded else "pstack",
            k=codec_rounds,
            kv_receipts=encoded and kv_receipts,
            select_trials=int(os.environ.get("VERITY_VLLM_BENCH_SELECT_TRIALS", "5")),
            wt_lib=libs["wt_lib"],
            reader_lib=libs["reader_lib"],
            paged_lib=libs["paged_lib"],
            spans=spans,
        )
        self.install = serve.Install(args)
        serve.hook_runner(target.hooks, self.install)
        self.topup: Any | None = None
        self.log = _TopupLog(None)
        self.started = False
        self.snapshots: list[dict[str, Any]] = []
        self.closed = False
        self.kernel = ""
        self.run_start: int | None = None
        self.measured = self.empty_runs = 0

    def _partition(self, fill_sms: int) -> None:
        """The SMs split before vLLM starts: the step's stream torch's current one, its SM count the device's, and the
        stack's side streams on its context. The top-up library is opened bare here (kv_filler's bench, which wraps it,
        puts word_codec first on sys.path, ahead of the stack's install); `Topup` opens the same one at `ready`."""
        import torch

        from benchmarks.kv_filler import green

        self.green_lib = SimpleNamespace(so=ctypes.CDLL(str(Path(self.libs["topup_lib"]).resolve())))
        self.green, self.step_stream, self.new_stream = green.partition(self.green_lib, torch, fill_sms)
        import stack as S   # benchmarks.vllm_encoded.serve put HERE first; the install imports this module

        if Path(S.__file__).resolve().parent != HERE:
            raise RuntimeError(f"the stack module is {S.__file__}, not {HERE / 'stack.py'}")
        S.new_stream = self.new_stream

    def ready(self, target: Any) -> None:
        import torch

        from benchmarks.kv_filler.bench import FILL_WARPS, RUN_SEED, Topup, mode_of
        from verity_vllm.engine import handle

        if self.install.stack is None:
            raise RuntimeError("bandwidth benchmark arm did not install during model loading")
        if self.green is not None:
            from benchmarks.kv_filler import green

            self.green["spread"] = green.spread_checks(self.green_lib, torch, self.step_stream, self.new_stream,
                                                       self.green)
        if self.use_topup:
            runner = handle.model_runner(target.engine.vllm)
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
            self.log = _TopupLog(self.topup)
        self.started = True
        self.kernel = _digest([
            Path(__file__),
            HERE / "serve.py",
            HERE / "stack.py",
            HERE / "step_spans.py",
            HERE / "paged.cu",
            HERE.parent / "kv_filler" / "green.py",
            *(Path(self.libs[name]) for name in ("wt_lib", "reader_lib", "paged_lib", "topup_lib")),
        ])

    @property
    def stack(self) -> Any:
        if self.install.stack is None:
            raise RuntimeError("bandwidth benchmark arm is not ready")
        return self.install.stack

    def snapshot(self) -> dict[str, Any]:
        if not self.started:
            return {}
        self.log.update()
        weight_bytes = float(sum(linear.nbytes for linear in self.stack.linears))
        value: dict[str, Any] = {
            "filler_bytes": self.log.bytes,
            "passes": float(self.stack.passes),
            "registered_weight_bytes": float(self.stack.passes) * weight_bytes,
        }
        if self.use_spans:
            value["span"] = {**self.stack.spans.read(),
                             "topup": {"busy_ms": self.log.busy_ms, "chunks": float(self.log.read),
                                       "bytes": self.log.bytes}}
        self.snapshots.append(value)
        return value

    def extras(self) -> dict[str, float]:
        if len(self.snapshots) < 2:
            return {}
        before, after = self.snapshots[-2:]
        filler = after["filler_bytes"] - before["filler_bytes"]
        registered = after["registered_weight_bytes"] - before["registered_weight_bytes"]
        total = filler + registered
        out = {
            "filler_bytes": filler,
            "registered_weight_bytes": registered,
            "filler_share": filler / total if total > 0 else 0.0,
            "receipt_passes": after["passes"] - before["passes"],
            "filler_ctas": float(self.ctas),
        }
        if self.green is not None:
            out["topup_sms"] = float(self.green["fill_sms"])
            out["step_sms"] = float(self.green["step_sms"])
        if "span" in after and "span" in before:
            steps = step_spans.per_step(before["span"], after["span"])
            out.update({f"span_per_step.{key}": value for key, value in steps.items()})
            main = sum(after["span"][kind]["ms"][name] - before["span"][kind]["ms"][name]
                       for kind in step_spans.KINDS for name in step_spans.MAIN)
            busy = after["span"]["topup"]["busy_ms"] - before["span"]["topup"]["busy_ms"]
            out["span.main_ms"] = main
            out["span.topup_busy_ms"] = busy
            out["span.topup_busy_share"] = busy / main if main > 0 else 0.0
        return out

    def begin_run(self, phase: str) -> None:
        if self.step_stream is not None:
            import torch

            if torch.cuda.current_stream().cuda_stream != self.step_stream.cuda_stream:
                self.off_step_runs += 1
        self.run_start = self.log.progress() if phase == "measure" else None

    def end_run(self) -> None:
        if self.run_start is None:
            return
        self.measured += 1
        if self.use_topup and self.log.progress() <= self.run_start:
            self.empty_runs += 1
        self.run_start = None

    def gate(self, arm_id: str) -> dict[str, Any]:
        gates = self.install.record.get("gates", {})
        judged = credit(arm_id, topup=self.use_topup, spans=self.use_spans, empty_runs=self.empty_runs)
        placed = self.green is None or (bool(self.green.get("spread", {}).get("confined")) and not self.off_step_runs)
        out = {
            "passed": _install_gate_passed(self.install.record, encoded=self.encoded) and judged["passes"] and placed,
            "layers": int(gates.get("layers", 0)),
            "kv_receipts": bool(self.stack.kv_receipts),
            "filler_ctas": self.ctas,
            "topup": self.use_topup,
            "spans": self.use_spans,
            "topup_sms": self.topup_sms,
            "measured_runs": self.measured,
            "credit": judged,
        }
        if self.green is not None:
            out["partition"] = {**self.green, "off_step_runs": self.off_step_runs}
        return out

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.topup is not None:
            self.topup.stop()


class BandwidthArm:
    """A bandwidth arm: the encoded stack (``encoded``) or its plain forms, token-gated against ``control``. Each
    setting left None is its environment variable's: ``topup`` VERITY_VLLM_BENCH_TOPUP (default on), ``spans``
    VERITY_VLLM_BENCH_SPANS (default off), ``kv_receipts`` VERITY_VLLM_BENCH_KV_RECEIPTS (default on, encoded only),
    ``codec_rounds`` VERITY_VLLM_BENCH_CODEC_ROUNDS (default 3), ``topup_sms`` VERITY_VLLM_BENCH_TOPUP_SMS (default
    ``TOPUP_SMS``)."""

    precision = "fp8"
    protocols: tuple[str, ...] = ()
    resident = False

    def __init__(self, arm_id: str, *, encoded: bool, control: str | None, topup: bool | None = None,
                 spans: bool | None = None, kv_receipts: bool | None = None, codec_rounds: int | None = None,
                 topup_sms: int | None = None) -> None:
        self.id = arm_id
        self.encoded = encoded
        self.control = control
        self.topup, self.spans_on, self.kv_receipts, self.codec_rounds = topup, spans, kv_receipts, codec_rounds
        self.topup_sms = topup_sms
        self.state: _Installed | None = None

    def settings(self) -> dict[str, Any]:
        """The settings an install takes: the arm's own, else the environment's."""
        rounds = self.codec_rounds
        if rounds is None:
            rounds = int(os.environ.get("VERITY_VLLM_BENCH_CODEC_ROUNDS", "3"))
        return {
            "topup": _switch(self.topup, "VERITY_VLLM_BENCH_TOPUP", "on"),
            "spans": _switch(self.spans_on, "VERITY_VLLM_BENCH_SPANS", "off"),
            "kv_receipts": _switch(self.kv_receipts, "VERITY_VLLM_BENCH_KV_RECEIPTS", "on"),
            "codec_rounds": rounds,
            "topup_sms": _sms(self.topup_sms),
        }

    def install(self, target: Any) -> _Installed:
        self.state = _Installed(target, encoded=self.encoded, **self.settings())
        return self.state

    def gate(self, target: Any) -> dict[str, Any]:
        arm = getattr(target, "arm", None)
        return self._state().gate(getattr(arm, "id", None) or self.id)

    def spans(self, target: Any) -> dict[str, Any]:
        return self._state().snapshot()

    def extras(self, target: Any) -> dict[str, float]:
        return self._state().extras()

    def begin_run(self, target: Any) -> None:
        self._state().begin_run(getattr(target, "phase", ""))

    def end_run(self, target: Any) -> None:
        self._state().end_run()

    def kernel_digest(self, target: Any) -> str:
        return self._state().kernel

    def _state(self) -> _Installed:
        if self.state is None:
            raise RuntimeError(f"bandwidth benchmark arm {self.id!r} is not installed")
        return self.state


PLAIN = BandwidthArm("bandwidth-plain", encoded=False, control=None)
AUDIT = BandwidthArm("bandwidth-audit", encoded=True, control=PLAIN.id)
PLAIN_NO_TOPUP = BandwidthArm("bandwidth-plain-no-topup", encoded=False, control=PLAIN.id, topup=False)
AUDIT_NO_TOPUP = BandwidthArm("bandwidth-audit-no-topup", encoded=True, control=PLAIN.id, topup=False)
PLAIN_SPANS = BandwidthArm("bandwidth-plain-spans", encoded=False, control=PLAIN.id, spans=True)
AUDIT_SPANS = BandwidthArm("bandwidth-audit-spans", encoded=True, control=PLAIN.id, spans=True)

__all__ = ["AUDIT", "AUDIT_NO_TOPUP", "AUDIT_SPANS", "CREDITED", "PLAIN", "PLAIN_NO_TOPUP", "PLAIN_SPANS",
           "BandwidthArm", "credit"]
