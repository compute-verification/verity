"""Pearl-C (``pearl-c-sm120-v1-h3``) as a `verity-bench` experimental arm (``--arm-module pearl_c_vllm.bench_arm:PEARL_C``
with ``benchmarks/pouw`` on ``PYTHONPATH``), on one RTX PRO 6000; with ``PEARLC_KERNEL`` the tree's
``benchmarks/pouw/pearl_c_sm90``, ``pearl-c-h100-v1-h3`` on one H100 or H200.

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

Env: ``PEARLC_KERNEL`` (default the tree's ``benchmarks/pouw/pearl_c_sm120``), ``PEARLC_CUBIN`` (default the directory's
build.sh cubin, ``<directory name>.cubin``), ``PEARLC_SCHEDULE`` (default ``deferred``, the served schedule),
``PEARLC_SCHEME`` (default the kernel directory's record, `KERNEL_SCHEMES`), ``PEARLC_BENCH_EPOCH``. The salt coin
is a challenger's held in this process (``in-process``, a cost measurement) unless ``PEARLC_CHALLENGER_GATE`` (infra's gate
as an ssh command under the lane's gate key) and ``PEARLC_CHALLENGER_PEER`` (this node's name in the challenger host's
peers) are set: then it is a live session's on the auditor host, reserved at install with its salt coin alone
(`window.SALT_FAMILIES`), opened on its salt epoch's root (`salt.epoch_open_root` of the weights' root and epoch 0, the
arm's one install under a fresh hm96 key) and its salt coin opened after it (`_LiveSession`), which the device run checks
(`pouw_pearl_c_host.epoch_of`); the gate's result names the session (``coins``), and the arm's close ends it complete.
``PEARLC_CHALLENGER_LANE`` (default ``vllm``) is the lane the gate key is registered for: while that lane's open period
is H1's the arm refuses before reserving (`pouw_h1_window.period_reason`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
settings = of(__file__)
#: `e2e.gate`'s bound on served y against x·Wᵀ (rel RMS), Pearl's FP8 quantization noise
ERR = 0.08
#: the scheme each of the tree's kernel directories runs (`pouw_pearl_c_device.SCHEMES`' device records)
KERNEL_SCHEMES = {"pearl_c_sm120": "pearl-c-sm120-v1-h3", "pearl_c_sm90": "pearl-c-h100-v1-h3"}


class _Coins:
    """`e2e.EpochCoins`, trimmed: one epoch's challenger, each coin opened once."""

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


class _LiveSession:
    """The epoch's salt coin from a session of the auditor host's challenger, through infra's gate: ``gate`` its argv
    prefix as one string (``ssh -i KEY -o IdentitiesOnly=yes research@HOST``, the lane's own gate key) and ``peer`` this
    node's name in the host's peers (``--peer``), whose 127.0.0.1 the challenger reaches through the node's certifier
    coin link. Reserved at
    construction (`commit_coins.Gate`: ``reserve --peer NAME``, Pearl-C's layout, its salt coin alone,
    `window.SALT_FAMILIES`), opened on salt epoch ``k``'s root over the weights' root once they are committed, its
    salt coin opened after it (`pouw_h1_window.open_session`), closed with the arm, so it ends complete. Refused before
    reserving while ``lane``'s open period in the auditor's ledger (``period``) is H1's, and before its open when the
    reserve joined another period (`pouw_h1_window.reserve_salt`)."""

    #: the arm's salt epoch over its weights' root: one install, under a fresh hm96 key
    k = 0

    def __init__(self, gate: str, peer: str, epoch: int, lane: str) -> None:
        from verity_vllm.pipeline.commit_coins import Gate
        from verity_vllm.protocols.pouw.h1 import window as PHW

        self.gate = Gate(gate)
        self.reserved, self.period = PHW.reserve_salt(self.gate.call, self.gate.ledger, lane, peer, epoch)
        self.client = self.commitment = None
        self.carried: dict[str, Any] = {}

    def open(self, entry: str, root: bytes) -> Any:
        from verity_vllm.protocols.pouw.h1 import window as PHW

        self.client, self.commitment, opening, _, self.carried = PHW.open_session(
            entry, self.gate.call, self.reserved["session"], root, self.k)
        return opening

    def close(self) -> None:
        if self.client is not None:
            client, self.client = self.client, None
            client.close()


class PearlCArm:
    id = "pouw-pearl-c"
    protocols: tuple[str, ...] = ()
    precision = "bfloat16"
    control = None
    resident = True
    #: `PearlCDeviceRun`'s options beside the install's
    run_options: dict[str, Any] = {}

    def __init__(self) -> None:
        self.kernel_dir = Path(settings.PEARLC_KERNEL or HERE.parent / "pearl_c_sm120")
        self.cubin = settings.PEARLC_CUBIN
        self.scheme = settings.PEARLC_SCHEME or KERNEL_SCHEMES.get(self.kernel_dir.name, "pearl-c-sm120-v1-h3")
        self.schedule = settings.PEARLC_SCHEDULE
        self.epoch = settings.PEARLC_BENCH_EPOCH
        self.gate_command = settings.PEARLC_CHALLENGER_GATE
        self.peer = settings.PEARLC_CHALLENGER_PEER
        self.lane = settings.PEARLC_CHALLENGER_LANE
        self.run = self.stack = self.weights = self.session = None
        self.coins: dict[str, Any] = {"source": "in-process"}

    def install(self, target: Any) -> "PearlCArm":
        return self

    def engine_overrides(self, target: Any) -> dict[str, Any]:
        """Headroom for the run's own device memory, which sits beside vLLM's budget: B̃ about a byte per weight, the
        pass's leaf and flag buffers, and the per-shape ring buffers whose A (FP32) and C̃/U scale with the step's rows
        times n. Without the utilization cap the block mode's 0.90 left nothing (OOM at 94.8 of 95 GiB);
        without the step cap a 16,384-token prefill step's rings cost about 36 GiB and the full sweep's big points OOMed
        in the gate (93.5 GiB torch-allocated). The step cap applies to
        this arm's engine only, so its chunking differs from the controls'; the per-token ratio stays the metric."""
        return {"gpu_memory_utilization": settings.PEARLC_GPU_UTIL,
                "max_num_batched_tokens": settings.PEARLC_STEP_TOKENS}

    def kernel_digest(self, target: Any) -> str | None:
        path = Path(self.cubin) if self.cubin else self.kernel_dir / f"{self.kernel_dir.name}.cubin"
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
        from verity.core.protocols.pouw.salt import epoch_open_root
        from verity_vllm.protocols.linear import vllm_site
        from verity_vllm.protocols.linear.api import Arm as StackArm
        from verity_vllm.protocols.linear.api import PassSpec, Stack
        from verity_vllm.protocols.pouw import pearl_c as PPD

        scheme = self.scheme
        sites = vllm_site.discover(target.model)
        weights = {site.weight_id: layer.weight.data for site, layer in sites}
        bad = PPD.refusals(scheme, {w: tuple(t.shape) for w, t in weights.items()})
        if bad:
            raise RuntimeError(f"{scheme} refuses the model: " + "; ".join(bad))
        if (self.gate_command is None) != (self.peer is None):
            raise RuntimeError("live coins take PEARLC_CHALLENGER_GATE (the gate's ssh command) and PEARLC_CHALLENGER_PEER "
                               "(this node's name in the challenger host's peers) together")
        kernel = PPD.Kernel.load(str(self.kernel_dir), self.cubin)
        if self.gate_command is not None:
            self.session = _LiveSession(self.gate_command, self.peer, self.epoch, self.lane)
        salt_key = PPD.new_salt_key(scheme)
        roots = PPD.weight_roots(kernel, weights, scheme, salt_key=salt_key)
        if self.session is not None:
            opening = self.session.open(f"pouw:{scheme}", weights_root(roots))
            commitment = self.session.commitment
            self.coins = {"source": "challenger", "session": self.session.carried, "pass_index": self.session.k,
                          "salt_after": epoch_open_root(weights_root(roots), self.session.k).hex()}
        else:
            coins = _Coins(self.epoch)
            opening, commitment = coins.open("salt", weights_root(roots)), coins.commitment
        tiles, rows = self._capacity(target, kernel.run.TILE, PPD.ROW_ALIGN,
                                     [t.shape[0] for t in weights.values()], PPD.LEAF_CAPACITY, PPD.FLAG_ROWS)
        try:
            run = PPD.PearlCDeviceRun(PPD.scheme_of(scheme), f"pouw:{scheme}", kernel, weights, roots,
                                      commitment, opening, schedule=self.schedule,
                                      capacity=tiles, flag_rows=rows, salt_key=salt_key,
                                      pass_index=None if self.session is None else self.session.k, **self.run_options)
        except BaseException:
            if self.session is not None:
                self.session.close()
            raise
        run.commits["coin_source"] = self.coins["source"]
        if self.session is not None:
            run.commits["session"] = self.session.carried
        stack = Stack([site for site, _ in sites], PPD.PearlCExecutor(run))
        bad = stack.refusals()
        if bad:
            raise RuntimeError(f"{scheme} refuses the model: " + "; ".join(bad))
        vllm_site.attach(stack, sites, target.hooks)
        spec = PassSpec(hashing=True, schedule=self.schedule, run_id=settings.RESEARCH_RUN_ID)
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
                commits, served, at = [], None, run.epoch_calls()
                for omit in ((), (), ("hash_leaf",)):
                    run.begin(at=at)
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
                "coins": self.coins, "checks": checks}

    def spans(self, target: Any) -> dict[str, Any]:
        return {} if self.stack is None else {"passes": float(len(self.stack.passes))}

    def close(self, target: Any = None) -> None:
        try:
            if self.stack is not None and self.stack.open_pass is None:
                self.stack.activate(None)
        finally:
            if self.session is not None:
                self.session.close()


class PearlCWholeArm(PearlCArm):
    """`PearlCArm` with each decode step replayed as one CUDA graph (`pouw_pearl_c_whole.WholeStep`, `e2e.py
    --graphs --whole-step --whole-defer --whole-defer-a --whole-fork-stats`): vLLM's FULL decode graphs at the preset's
    capture sizes, without torch.compile, each captured with every call's graph steps inside it, each call's tile hashing
    beside the next call, its A tree beside its own forming and GEMM (`whole_defer_a`, which needs a kernel whose pipeline
    splits A's commitment), the GEMM leaving the tree's CTA its SM (`whole_tree_sm`), and its A scales beside its A
    commitment (`whole_fork_stats`), so a decode step launches one
    graph after one slot upload and runs no call's Python. Prefill and mixed steps run eagerly, each call of at most
    `pouw_pearl_c_graphs.GRAPH_ROWS` rows as its own graph. The serial schedule (the graph path's), one pass per engine
    step as `PearlCArm`'s, each pass's commitment read back at the next pass's end (`lag_ends`), so the host prepares the
    next step while the GPU runs this one; its gate is `PearlCArm`'s call check with the calls graphed, then
    `e2e.whole_check`: a forward's calls captured whole and replayed, bit for bit the eager calls' commitment.

    The PoUW option refuses this engine (`pouw.refusals`: compiled execution), so the arm installs as `PearlCArm` does;
    the protocol arm reaching it is vllm's change. ``PEARLC_WHOLE_EXECUTION`` is the engine's execution label:
    ``compiled`` (default, vLLM's kernels without batch invariance, as `stock-fp8@compiled` runs) or ``bi-compiled``;
    ``PEARLC_CAPTURE_SIZES`` overrides the preset's capture sizes (each at most 256 rows); ``PEARLC_WHOLE_DEFER_A=0``
    keeps each call's A tree on the capturing stream (a kernel whose pipeline doesn't split A's commitment needs it),
    ``PEARLC_WHOLE_FORK_STATS=0`` each call's A scales (`whole_fork_stats`: by default they run beside its A commitment),
    ``PEARLC_WHOLE_TREE_SM=0`` gives the GEMM beside A's tree every SM, ``PEARLC_WHOLE_SHAPE_JOINS=1`` joins each call's
    side steps only before the next call of its shape (`whole_shape_joins`; off by default: its side steps slow
    the next calls' GEMMs), ``PEARLC_LAG_ENDS=0`` reads each pass's
    commitment back at its own end, and ``PEARLC_ASIDE_ENDS=0`` runs a lagged end's tile tree on the step's stream (by
    default, `aside_ends`, on a stream of its own, so the next step doesn't wait for it).  On pearl_c_sm120 (`RTX_ONLY`)
    the stats stream also prefetches each call's B̃ into L2 (`whole_prefetch_b`, run.py's ``prefetch_b`` in its
    ``PEARLC_PREFETCH_FORM``, default form 2; ``PEARLC_WHOLE_PREFETCH_B=0`` turns it off).  There a step
    that runs eagerly (a prefill or mixed step) takes ``deferred-a`` in place of the serial schedule (`whole_eager`,
    `PearlCDeviceRun.eager_forward`: its calls' tile hashing and A trees on side lanes); ``PEARLC_WHOLE_EAGER`` names
    another schedule, on any kernel, or 0 none.
    A retained pass's whole capture keeps a stage for each weight and m₁₆, and the whole check runs at the largest size."""

    id = "pouw-pearl-c-whole"
    run_options = {"graphs": True, "whole_defer": True, "whole_defer_a": True, "whole_fork_stats": True, "lag_ends": True,
                   "whole_tree_sm": True, "aside_ends": True, "whole_prefetch_b": True, "whole_eager": "deferred-a"}
    #: the options measured on pearl_c_sm120 alone (B̃'s prefetch is its pipeline's), which another kernel's arm leaves out
    RTX_ONLY = ("whole_prefetch_b", "whole_eager")

    def __init__(self) -> None:
        super().__init__()
        if self.kernel_dir.name != "pearl_c_sm120":
            self.run_options = {k: v for k, v in self.run_options.items() if k not in self.RTX_ONLY}
        for option, env in (("whole_defer_a", "PEARLC_WHOLE_DEFER_A"), ("whole_fork_stats", "PEARLC_WHOLE_FORK_STATS"),
                            ("lag_ends", "PEARLC_LAG_ENDS"), ("whole_tree_sm", "PEARLC_WHOLE_TREE_SM"),
                            ("aside_ends", "PEARLC_ASIDE_ENDS"), ("whole_prefetch_b", "PEARLC_WHOLE_PREFETCH_B")):
            if not settings.get(env):
                self.run_options = {k: v for k, v in self.run_options.items() if k != option}
        if settings.PEARLC_WHOLE_SHAPE_JOINS:
            self.run_options = {**self.run_options, "whole_shape_joins": True}
        if "whole_defer_a" not in self.run_options:
            self.run_options = {k: v for k, v in self.run_options.items() if k != "whole_tree_sm"}
        if "lag_ends" not in self.run_options:
            self.run_options = {k: v for k, v in self.run_options.items() if k != "aside_ends"}
        eager = settings.PEARLC_WHOLE_EAGER
        if eager == "0":
            self.run_options = {k: v for k, v in self.run_options.items() if k != "whole_eager"}
        elif eager:
            self.run_options = {**self.run_options, "whole_eager": eager}
        self.schedule = "serial"
        self.vllm_execution = settings.PEARLC_WHOLE_EXECUTION
        # Not ``execution``: the harness reads that as "build at vLLM's serving defaults", and this engine is FULL
        # decode graphs at compilation mode 0 (no torch.compile), which PoUW requires.
        self.engine_label = "graphs-mode0" + ("" if self.vllm_execution == "compiled" else f"-{self.vllm_execution}")
        self.whole, self.totals, self.capture = None, {}, [1, 8, 32, 128]

    def engine_overrides(self, target: Any) -> dict[str, Any]:
        self.capture = list(settings.PEARLC_CAPTURE_SIZES or (1, 8, 32, 128))
        return {**super().engine_overrides(target), "enforce_eager": False, "execution": self.vllm_execution,
                "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                       "cudagraph_capture_sizes": self.capture}}

    def ready(self, target: Any) -> None:
        from verity_vllm.engine import handle
        from verity_vllm.protocols.pouw.pearl_c.whole import WholeStep

        super().ready(target)
        self.whole = WholeStep(self.run, self.stack, handle.model_runner(target.engine.vllm), target.hooks)
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
