"""Value-correctness acceptance check for the native rows (VALUE-FIRST §5): the bytes the committer BOUND equal the exact required
values, at the right identity and occurrence — including repeated calls of the same module and buffer reuse across steps.  Matching
final logits is not enough, so two independent references are compared against the committed bytes of every acquired tensor:

  (T) hook-time reference tap: an independent host copy of each module output taken in a SEPARATE forward hook at the moment the
      output exists (before vLLM's in-place fused add-RMSNorm overwrites its input and the residual buffer).  Detects deferred-copy
      hazards, buffer reuse across steps / calls, wrong ordering of the acquired tensors.
  (R) re-execution of the registered relation: for modules whose forward is a pure function of tapped inputs (Linear -> GEMM,
      RMSNorm, SiluAndMul, RotaryEmbedding), the SAME pinned batch-invariant kernel is run again after the step on copies of the
      tapped inputs (host copies taken in the pre-hook, moved back to the device they came from) and its output compared
      bit-for-bit with the committed bytes.  (Attention needs the KV cache state, the sampler needs the sampling metadata: T only.)
      Bit-identity of a re-execution is exactly what `VLLM_BATCH_INVARIANT=1` promises.

The references are host tensors: a run's every output does not fit beside the engine on the device (GLM-5.3 TP8 B8 commits 25.6 GB
a rank in one run, with about 9 GB of the GPU free).

Run by `commit_delta.py --value-check` on one extra instrumented run; result in `value_check.json` (+ printed).  Labels: measured.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import torch

from verity_vllm.commit.opened import OpenedReader, OpeningNotVerified
from verity_vllm.engine.profiles import layer_classes as LC


def _cls(m: Any) -> str:
    names = {c.__name__ for c in type(m).__mro__}
    for k in LC.TAPPED:
        if k in names:
            return k
    if names & LC.LINEAR:
        return LC.LINEAR_GROUP
    return type(m).__name__


def _host(a: Any) -> Any:
    """A tensor's copy in host memory, taken now (the device-to-host copy is ordered after the kernel that produced it); else `a`."""
    return a.detach().to("cpu", copy=True) if isinstance(a, torch.Tensor) else a


class ValueChecker:
    """Install AFTER the committer (hooks run in registration order: the committer acquires first, then the tap clones)."""

    def __init__(self, committer, max_reexec_per_step: int = 10**9) -> None:
        self.c = committer
        self.model = committer.model
        self.handles: list = []
        self.steps: list[dict] = []          # per step: {"name": {"in": [...], "out": [...]}}
        self.cur: dict | None = None
        self.max_reexec = max_reexec_per_step
        self.modules = dict(committer._modules)

    def install(self) -> None:
        for name, m in self.c._modules:
            self.handles.append(m.register_forward_pre_hook(self._pre(name)))
            self.handles.append(m.register_forward_hook(self._post(name)))
        smp = getattr(self.c.runner, "sampler", None)
        if isinstance(smp, torch.nn.Module):
            self.handles.append(smp.register_forward_hook(self._post("runner.sampler")))
        # step boundaries: piggy-back on the committer's _begin_step / _end_step
        me = self
        orig_begin, orig_end = self.c._begin_step, self.c._end_step

        def begin():
            orig_begin()
            me.cur = {}
            me.steps.append(me.cur)

        def end():
            orig_end()
            me.cur = None

        self.c._begin_step, self.c._end_step = begin, end
        self._restore = (orig_begin, orig_end)

    def uninstall(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self.c._begin_step, self.c._end_step = self._restore

    def _pre(self, name):
        def pre(module, args):
            if self.cur is None:
                return None
            d = self.cur.setdefault(name, {"in": [], "in_dev": [], "out": [], "calls": 0, "single": []})
            d["in"].append([_host(a) for a in args])
            d["in_dev"].append([a.device if isinstance(a, torch.Tensor) else None for a in args])
            return None
        return pre

    def _post(self, name):
        def post(module, args, output):
            if self.cur is None:
                return None
            d = self.cur.setdefault(name, {"in": [], "in_dev": [], "out": [], "calls": 0, "single": []})
            d["calls"] += 1
            d["single"].append(isinstance(output, torch.Tensor))
            if isinstance(output, torch.Tensor):
                outs = [_host(output)]
            elif isinstance(output, (tuple, list)):
                outs = [_host(o) if isinstance(o, torch.Tensor) else None for o in output]
            else:
                st = getattr(output, "sampled_token_ids", None)
                outs = [_host(st)] if isinstance(st, torch.Tensor) else []
            d["out"].append(outs)
            return None
        return post

    @staticmethod
    def _committed(reader, step: int, m):
        """Committed bytes of tensor `m`, opened from the committer and verified against the run root by `reader`
        (`commit.opened.OpenedReader`); a read that does not verify raises `OpeningNotVerified`."""
        return None if m is None else reader.read_meta(step, m)

    @staticmethod
    def _bytes(t: torch.Tensor) -> bytes:
        return t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes() if t.dtype != torch.uint8 else t.contiguous().reshape(-1).cpu().numpy().tobytes()

    @torch.inference_mode()
    def compare(self, reader) -> dict:
        """After finalize(): committed bytes (opened and verified against the run root by `reader`, an `commit.opened.OpenedReader`) vs
        (T) tap clones and (R) re-execution.  A tensor whose opening does not verify is `not_verified` (FAIL)."""
        t0 = time.perf_counter()
        res = {"steps": len(self.steps), "tensors": 0, "tap_equal": 0, "tap_mismatch": [], "reexec_tensors": 0, "reexec_equal": 0,
               "reexec_mismatch": [], "reexec_skipped": {}, "repeated_calls": 0, "missing_in_commit": [], "not_verified": [], "by_class": {}}
        for step, taps in enumerate(self.steps):
            metas_by: dict[str, list] = {}
            for m in sorted(self.c._layouts.get(step, []), key=lambda m: m.ordinal):
                metas_by.setdefault(m.name, []).append(m)            # repeated calls of one module -> several metas, in call order

            def meta_at(key: str, call_idx: int):
                lst = metas_by.get(key) or []
                return lst[call_idx] if call_idx < len(lst) else None
            for name, d in taps.items():
                mc = LC.SAMPLER if name == "runner.sampler" else _cls(self.modules[name])
                bc = res["by_class"].setdefault(mc, {"tensors": 0, "tap_equal": 0, "reexec": 0, "reexec_equal": 0})
                if d["calls"] > 1:
                    res["repeated_calls"] += 1
                for call_idx, outs in enumerate(d["out"]):
                    for k, o in enumerate(outs):
                        if o is None:
                            continue
                        if name == "runner.sampler":
                            key = f"{name}/sampled_token_ids"
                        else:
                            key = name if d["single"][call_idx] else f"{name}/{k}"
                        m = meta_at(key, call_idx)
                        try:
                            committed = self._committed(reader, step, m)
                        except OpeningNotVerified as e:
                            res["not_verified"].append((step, key, str(e)[:200]))
                            continue
                        if committed is None:
                            res["missing_in_commit"].append((step, key))
                            continue
                        res["tensors"] += 1
                        bc["tensors"] += 1
                        if committed == self._bytes(o):
                            res["tap_equal"] += 1
                            bc["tap_equal"] += 1
                        else:
                            res["tap_mismatch"].append((step, key, m.nbytes))
                # (R) re-execution on tapped inputs
                if mc in LC.REEXEC and d["in"] and name in self.modules:
                    mod = self.modules[name]
                    for call_idx, args in enumerate(d["in"][: self.max_reexec]):
                        try:
                            args2 = [a.to(dv, copy=True) if isinstance(a, torch.Tensor) else a for a, dv in zip(args, d["in_dev"][call_idx])]
                            out2 = mod(*args2)
                        except Exception as e:  # noqa: BLE001
                            res["reexec_skipped"][mc] = repr(e)[:160]
                            break
                        single = isinstance(out2, torch.Tensor)
                        outs2 = [out2] if single else [o if isinstance(o, torch.Tensor) else None for o in out2] if isinstance(out2, (tuple, list)) else []
                        for k, o2 in enumerate(outs2):
                            if o2 is None:
                                continue
                            key = name if single else f"{name}/{k}"
                            m = meta_at(key, call_idx)
                            try:
                                committed2 = self._committed(reader, step, m)
                            except OpeningNotVerified as e:
                                res["not_verified"].append((step, key, str(e)[:200]))
                                continue
                            if committed2 is None:
                                continue
                            res["reexec_tensors"] += 1
                            bc["reexec"] += 1
                            if committed2 == self._bytes(o2):
                                res["reexec_equal"] += 1
                                bc["reexec_equal"] += 1
                            else:
                                res["reexec_mismatch"].append((step, key, m.nbytes))
        res["tap_mismatch_n"] = len(res["tap_mismatch"])
        res["reexec_mismatch_n"] = len(res["reexec_mismatch"])
        res["tap_mismatch"] = res["tap_mismatch"][:40]
        res["reexec_mismatch"] = res["reexec_mismatch"][:40]
        res["missing_in_commit"] = res["missing_in_commit"][:40]
        res["not_verified_n"] = len(res["not_verified"])
        res["not_verified"] = res["not_verified"][:40]
        res["value_source"] = reader.record()
        res["wall_s"] = time.perf_counter() - t0
        res["verdict"] = "PASS" if (res["tap_mismatch_n"] == 0 and res["reexec_mismatch_n"] == 0 and res["tensors"] > 0 and not res["missing_in_commit"]
                                    and res["not_verified_n"] == 0) else "FAIL"
        self.steps = []
        return res


def run_value_check(com: Any, run: Callable[[], Any], restore: Callable[[], None]) -> dict:
    """One extra instrumented run with reference taps + re-execution of the registered relation: `run()` executes the workload with `com` and
    a `ValueChecker` installed, then the committed bytes -- opened and verified against that run's root -- are compared; `restore()` runs
    once both are uninstalled.  The result carries the run root and the committer's errors."""
    com.reset()
    com.install()
    vc = ValueChecker(com)
    vc.install()
    run()
    rc = com.finalize()
    vres = vc.compare(OpenedReader(com, rc))
    vc.uninstall()
    com.uninstall()
    com.reset()
    restore()
    vres.update({"run_root": rc.run_root.hex(), "errors": rc.extra.get("errors")})
    return vres
