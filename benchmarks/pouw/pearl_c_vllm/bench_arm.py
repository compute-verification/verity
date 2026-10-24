"""Pearl-C (``pearl-c-sm120-v1-h3``) as a `verity-bench` experimental arm (``--arm-module pearl_c_vllm.bench_arm:PEARL_C``
with ``benchmarks/pouw`` on ``PYTHONPATH``), on one RTX PRO 6000.

The arm serves the preset's model in BF16 with every linear computed and committed by `pouw_pearl_c_device.PearlCDeviceRun`,
installed as `e2e.install` does (sites by type, the kernel directory's cubin, the weight roots, the epoch from a
challenger's salt coin) but attached through the harness's ``target.hooks``, so the runner uninstalls it with the engine.
One pass per engine step: ``llm_engine.step`` is wrapped to open `Stack.pass_` around each forward, which bounds a pass's
tiles by the step's scheduled tokens, so the full sweep's long prefills fit the pass buffers that one generate's pass
would overflow.

Its tokens are Pearl-C's own y (Pearl's FP8 quantization), so no stock arm is a bit-exact control (``control=None``); the
arm's gate is Pearl-C's own, `e2e.gate`'s call check: at every distinct (n, k), served y against x·Wᵀ within `ERR` rel
RMS, the same call's commitment twice bit for bit, and the ``hash_leaf``-omitted negative control changing the
transcript root.

Env: ``PEARLC_KERNEL`` (default the tree's ``benchmarks/pouw/pearl_c_sm120``), ``PEARLC_CUBIN`` (default the directory's),
``PEARLC_SCHEDULE`` (default ``deferred``, the served schedule), ``PEARLC_SCHEME``, ``PEARLC_BENCH_EPOCH``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: `e2e.gate`'s bound on served y against x·Wᵀ (rel RMS), Pearl's FP8 quantization noise
ERR = 0.08


class _Coins:
    """`e2e.Coins`, trimmed: one epoch's challenger, each coin opened once."""

    def __init__(self, epoch: int) -> None:
        from verity.core.protocols.pouw import verifier
        from verity.core.protocols.pouw.coins import Challenger

        self.challenger, self.opened = Challenger(epoch, verifier.coins), {}
        self.commitment = self.challenger.commitment

    def open(self, label: str, after: bytes) -> Any:
        from verity.core.protocols.pouw.protocol import DomainError

        if label in self.opened:
            if self.opened[label].after != after:
                raise DomainError(f"epoch {self.commitment.epoch}'s {label!r} coin is already open, after another record")
            return self.opened[label]
        self.opened[label] = self.challenger.open(label, after)
        return self.opened[label]


class PearlCArm:
    id = "pouw-pearl-c"
    protocols: tuple[str, ...] = ()
    precision = "bfloat16"
    control = None
    resident = True
    #: `PearlCDeviceRun`'s options beside the install's
    run_options: dict[str, Any] = {}

    def __init__(self) -> None:
        self.kernel_dir = Path(os.environ.get("PEARLC_KERNEL", str(HERE.parent / "pearl_c_sm120")))
        self.cubin = os.environ.get("PEARLC_CUBIN") or None
        self.schedule = os.environ.get("PEARLC_SCHEDULE", "deferred")
        self.epoch = int(os.environ.get("PEARLC_BENCH_EPOCH", "20261014"))
        self.run = self.stack = self.weights = None

    def install(self, target: Any) -> "PearlCArm":
        return self

    def engine_overrides(self, target: Any) -> dict[str, Any]:
        """Headroom for the run's own device memory, which sits beside vLLM's budget: B̃ about a byte per weight, the
        pass's leaf and flag buffers, and the per-shape ring buffers whose A (FP32) and C̃/U scale with the step's rows
        times n. Without the utilization cap the block mode's 0.90 left nothing (OOM at 94.8 of 95 GiB,
        r20261014-062817-4c45); without the step cap a 16,384-token prefill step's rings cost about 36 GiB and the full
        sweep's big points OOMed in the gate (93.5 GiB torch-allocated, r20261014-065819-c0c3). The step cap applies to
        this arm's engine only, so its chunking differs from the controls'; the per-token ratio stays the metric."""
        return {"gpu_memory_utilization": float(os.environ.get("PEARLC_GPU_UTIL", "0.55")),
                "max_num_batched_tokens": int(os.environ.get("PEARLC_STEP_TOKENS", "4096"))}

    def kernel_digest(self, target: Any) -> str | None:
        path = Path(self.cubin) if self.cubin else self.kernel_dir / "pearl_c_sm120.cubin"
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def _capacity(self, target: Any, tile: int, row_align: int, ns: list[int],
                  leaf_default: int, rows_default: int) -> tuple[int, int]:
        """One engine step's tiles and screened rows, with headroom, at least the device run's defaults."""
        cdiv = lambda a, b: -(-a // b)  # noqa: E731
        cfg = target.engine.vllm.llm_engine.vllm_config.scheduler_config
        step_rows = int(getattr(cfg, "max_num_batched_tokens", 8192) or 8192)
        tiles = 2 * cdiv(step_rows, tile) * sum(cdiv(n, tile) for n in ns)
        rows = 2 * len(ns) * cdiv(step_rows, row_align) * row_align
        pow2 = lambda x: 1 << max(0, x - 1).bit_length()  # noqa: E731
        return max(leaf_default, pow2(tiles)), max(rows_default, pow2(rows))

    def ready(self, target: Any) -> None:
        from verity.core.protocols.pouw.audit import weights_root
        from verity_vllm.linear import vllm_site
        from verity_vllm.linear.api import Arm as StackArm
        from verity_vllm.linear.api import PassSpec, Stack
        from verity_vllm.protocol_options import pouw_pearl_c_device as PPD

        scheme = os.environ.get("PEARLC_SCHEME", "pearl-c-sm120-v1-h3")
        sites = vllm_site.discover(target.model)
        weights = {site.weight_id: layer.weight.data for site, layer in sites}
        bad = PPD.refusals(scheme, {w: tuple(t.shape) for w, t in weights.items()})
        if bad:
            raise RuntimeError(f"{scheme} refuses the model: " + "; ".join(bad))
        kernel = PPD.Kernel.load(str(self.kernel_dir), self.cubin)
        salt_key = PPD.new_salt_key(scheme)
        roots = PPD.weight_roots(kernel, weights, scheme, salt_key=salt_key)
        coins = _Coins(self.epoch)
        opening = coins.open("salt", weights_root(roots))
        tiles, rows = self._capacity(target, kernel.run.TILE, PPD.ROW_ALIGN,
                                     [t.shape[0] for t in weights.values()], PPD.LEAF_CAPACITY, PPD.FLAG_ROWS)
        run = PPD.PearlCDeviceRun(PPD.scheme_of(scheme), f"pouw:{scheme}", kernel, weights, roots,
                                  coins.commitment, opening, schedule=self.schedule,
                                  capacity=tiles, flag_rows=rows, salt_key=salt_key, **self.run_options)
        stack = Stack([site for site, _ in sites], PPD.PearlCExecutor(run))
        bad = stack.refusals()
        if bad:
            raise RuntimeError(f"{scheme} refuses the model: " + "; ".join(bad))
        vllm_site.attach(stack, sites, target.hooks)
        spec = PassSpec(hashing=True, schedule=self.schedule, run_id=os.environ.get("RESEARCH_RUN_ID"))
        stack.activate(StackArm("pearlc", "bf16", run.entry, spec=spec))

        eng = target.engine.vllm.llm_engine

        def make(original: Any) -> Any:
            def step(*a: Any, **kw: Any) -> Any:
                if stack.active is None or stack.open_pass is not None:
                    return original(*a, **kw)
                with stack.pass_():
                    return original(*a, **kw)

            return step

        target.hooks.wrap(eng, "step", make)
        self.run, self.stack, self.weights = run, stack, weights

    def gate(self, target: Any) -> dict[str, Any]:
        """`e2e.gate`'s call check, between passes: y against x·Wᵀ, the commitment twice, and the negative control."""
        import torch

        run, weights = self.run, self.weights
        checks, seen = [], set()
        gen = torch.Generator(device=next(iter(weights.values())).device).manual_seed(20260930)
        for w, W in weights.items():
            if tuple(W.shape) in seen:
                continue
            seen.add(tuple(W.shape))
            for m in (16, 64):
                x = torch.randn(m, W.shape[1], device=W.device, generator=gen).to(torch.bfloat16)
                commits, served = [], None
                for omit in ((), (), ("hash_leaf",)):
                    run.begin()
                    run.poison()
                    run.omit = frozenset(omit)
                    y = run.gemm(w, x, W, None)
                    served = y if served is None else served
                    commits.append({k: v for k, v in run.end().items() if k in ("transcript_root", "activation_roots")})
                run.omit = frozenset()
                ref = x.float() @ W.float().T
                err = float(((served.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
                same = commits[0] == commits[1]
                control = commits[2]["transcript_root"] != commits[0]["transcript_root"]
                checks.append({"n": W.shape[0], "k": W.shape[1], "m": m, "rel_rms": err, "commitment_repeats": same,
                               "omitted_leaves_rejected": control, "ok": err < ERR and same and control})
        run.details["passes"].clear()
        return {"passed": all(c["ok"] for c in checks), "scheme": run.scheme.name, "schedule": self.schedule,
                "checks": checks}

    def spans(self, target: Any) -> dict[str, Any]:
        return {} if self.stack is None else {"passes": float(len(self.stack.passes))}

    def close(self, target: Any = None) -> None:
        if self.stack is not None and self.stack.open_pass is None:
            self.stack.activate(None)


class PearlCWholeArm(PearlCArm):
    """`PearlCArm` with each decode step replayed as one CUDA graph (`pouw_pearl_c_whole.WholeStep`, `e2e.py
    --graphs --whole-step --whole-defer`): vLLM's FULL decode graphs at the preset's capture sizes, without torch.compile,
    each captured with every call's graph steps inside it and each call's tile hashing beside the next call, so a decode
    step launches one graph after one slot upload and runs no call's Python. Prefill and mixed steps run eagerly, each
    call of at most `pouw_pearl_c_graphs.GRAPH_ROWS` rows as its own graph. The serial schedule (the graph path's), one
    pass per engine step as `PearlCArm`'s, and its gate is `PearlCArm`'s call check with the calls graphed, then
    `e2e.whole_check`: a forward's calls captured whole and replayed, bit for bit the eager calls' commitment.

    The PoUW option refuses this engine (`pouw.refusals`: compiled execution), so the arm installs as `PearlCArm` does;
    the protocol arm reaching it is vllm's change. ``PEARLC_WHOLE_EXECUTION`` is the engine's execution label:
    ``compiled`` (default, vLLM's kernels without batch invariance, as `stock-fp8@compiled` runs) or ``bi-compiled``;
    ``PEARLC_CAPTURE_SIZES`` overrides the preset's capture sizes (each at most 256 rows). A retained pass's whole capture
    keeps one stage a weight, sized at the first capture, so until vllm keys them by m₁₆ the arm starts at one size only
    (the point's batch), and its whole check runs at the largest size."""

    id = "pouw-pearl-c-whole"
    run_options = {"graphs": True, "whole_defer": True}

    def __init__(self) -> None:
        super().__init__()
        self.schedule = "serial"
        self.execution = os.environ.get("PEARLC_WHOLE_EXECUTION", "compiled")
        self.whole, self.totals, self.capture = None, {}, [1, 8, 32, 128]

    def engine_overrides(self, target: Any) -> dict[str, Any]:
        sizes = os.environ.get("PEARLC_CAPTURE_SIZES")
        self.capture = [int(x) for x in sizes.split(",")] if sizes else [1, 8, 32, 128]
        return {**super().engine_overrides(target), "enforce_eager": False, "execution": self.execution,
                "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                       "cudagraph_capture_sizes": self.capture}}

    def ready(self, target: Any) -> None:
        from verity_vllm.engine.engine_profile import get_model_runner
        from verity_vllm.protocol_options.pouw_pearl_c_whole import WholeStep

        super().ready(target)
        self.whole = WholeStep(self.run, self.stack, get_model_runner(target.engine.vllm), target.hooks)
        self.whole.prepare()
        self.totals = dict(self.whole.take())

    def gate(self, target: Any) -> dict[str, Any]:
        import importlib.util

        got = super().gate(target)
        spec = importlib.util.spec_from_file_location("pearl_c_e2e", HERE / "e2e.py")
        e2e = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(e2e)
        whole = e2e.whole_check(self.run, self.weights, m=max(self.capture))
        self.run.details["passes"].clear()
        return {**got, "passed": bool(got["passed"] and whole.get("ok")), "whole": whole}

    def spans(self, target: Any) -> dict[str, Any]:
        """`PearlCArm`'s, and `WholeStep`'s counts since `ready` (the runner differences two reads around a run)."""
        out = super().spans(target)
        if self.whole is not None:
            for k, v in self.whole.take().items():
                self.totals[k] = self.totals.get(k, 0) + v
            out.update({f"whole_{k}": float(v) for k, v in self.totals.items()})
        return out


PEARL_C = PearlCArm()
PEARL_C_WHOLE = PearlCWholeArm()
