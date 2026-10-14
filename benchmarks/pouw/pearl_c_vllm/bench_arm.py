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
        from verity.core.protocols.pouw.coins import Challenger

        self.challenger, self.opened = Challenger(epoch), {}
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

    def __init__(self) -> None:
        self.kernel_dir = Path(os.environ.get("PEARLC_KERNEL", str(HERE.parent / "pearl_c_sm120")))
        self.cubin = os.environ.get("PEARLC_CUBIN") or None
        self.schedule = os.environ.get("PEARLC_SCHEDULE", "deferred")
        self.epoch = int(os.environ.get("PEARLC_BENCH_EPOCH", "20261014"))
        self.run = self.stack = self.weights = None

    def install(self, target: Any) -> "PearlCArm":
        return self

    def engine_overrides(self, target: Any) -> dict[str, Any]:
        """Headroom for the run's own device memory (B̃ about a byte per weight, the pass's leaf and flag buffers),
        which sits beside vLLM's budget: without this the block mode's 0.90 utilization leaves nothing (OOM at 94.8
        of 95 GiB, r20261014-062817-4c45)."""
        return {"gpu_memory_utilization": float(os.environ.get("PEARLC_GPU_UTIL", "0.55"))}

    def kernel_digest(self, target: Any) -> str | None:
        path = Path(self.cubin) if self.cubin else self.kernel_dir / "pearl_c_sm120.cubin"
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def _capacity(self, target: Any, tile: int, row_align: int, ns: list[int],
                  leaf_default: int, rows_default: int) -> tuple[int, int]:
        """One engine step's tiles and screened rows, with headroom, at least the device run's defaults."""
        cdiv = lambda a, b: -(-a // b)  # noqa: E731
        cfg = target.engine._engine.llm.llm_engine.vllm_config.scheduler_config
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
                                  capacity=tiles, flag_rows=rows, salt_key=salt_key)
        stack = Stack([site for site, _ in sites], PPD.PearlCExecutor(run))
        bad = stack.refusals()
        if bad:
            raise RuntimeError(f"{scheme} refuses the model: " + "; ".join(bad))
        vllm_site.attach(stack, sites, target.hooks)
        spec = PassSpec(hashing=True, schedule=self.schedule, run_id=os.environ.get("RESEARCH_RUN_ID"))
        stack.activate(StackArm("pearlc", "bf16", run.entry, spec=spec))

        eng = target.engine._engine.llm.llm_engine

        def make(original: Any) -> Any:
            def step(*a: Any, **kw: Any) -> Any:
                if stack.active is None or stack._pass is not None:
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

    def close(self, target: Any) -> None:
        if self.stack is not None and self.stack._pass is None:
            self.stack.activate(None)


PEARL_C = PearlCArm()
