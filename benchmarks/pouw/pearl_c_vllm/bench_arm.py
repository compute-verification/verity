"""A useful-work scheme's kernels as a `verity-bench` experimental arm (``--arm-module pearl_c_vllm.bench_arm:PEARL_C``
with ``benchmarks/pouw`` on ``PYTHONPATH``) on one card: the arm takes the scheme's registered name (``PearlCArm(NAME)``;
`PEARL_C` is ``pearl-plus``'s) and serves the PoUW kernel scheme that scheme runs on the leased card
(`PearlCArm.kernel`: ``pearl-c-sm120-v1-h3`` on an RTX PRO 6000, ``pearl-c-h100-v1-h3`` on an H100 or H200).

The arm serves the preset's model in BF16 with every linear computed and committed by `pouw_pearl_c_device.PearlCDeviceRun`,
installed as `e2e.install` does (sites by type, the kernel directory's cubin, the fixed operands' roots, the epoch from a
challenger's salt coin) but attached through the harness's ``target.hooks``, so the runner uninstalls it with the engine.
One pass per engine step: ``llm_engine.step`` is wrapped to open `Stack.pass_` around each forward, which bounds a pass's
tiles by the step's scheduled tokens, so the full sweep's long prefills fit the pass buffers that one generate's pass
would overflow.
It holds the device run directly, not Pearl+ attached by name, because it times and replays device internals no served
setting exposes.

Its tokens are Pearl-C's own y (Pearl's FP8 quantization), so no stock arm is a bit-exact control (``control=None``), and
each call's noise is seeded by its index across the epoch, so no two of the engine's passes need agree either (the gate's
`runner.PER_PASS_TOKENS`); the arm's gate is Pearl-C's own, `e2e.gate`'s call check: at every distinct (n, k), served y
against x·Wᵀ within `ERR` rel RMS, and the same call twice with equal tile digests under different roots (its salts
are fresh; `e2e.call_verdict`).

Env: ``PEARLC_KERNEL`` (default the tree's directory of the kernel scheme on the card, `kernel_directory`), ``PEARLC_CUBIN``
(default the directory's build.sh cubin, ``<directory name>.cubin``), ``PEARLC_SCHEDULE`` (default ``deferred``, the
served schedule), ``PEARLC_BENCH_EPOCH``. The salt coin
is a challenger's held in this process (``in-process``, a cost measurement) unless ``PEARLC_CHALLENGER_GATE`` (infra's gate
as an ssh command under the lane's gate key) and ``PEARLC_CHALLENGER_PEER`` (this node's name in the challenger host's
peers) are set: then it is a live session's on the auditor device, reserved at install with its salt coin alone
(`window.SALT_FAMILIES`), opened on its salt epoch's root (`salt.epoch_open_root` of the fixed operands' root and epoch 0, the
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
#: the tree's kernel directories, one per kernel scheme family and build target (`kernel_directory`)
KERNELS = HERE.parents[2] / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "prover" / "kernels"
settings = of(__file__)
#: `e2e.gate`'s bound on served y against x·Wᵀ (rel RMS), Pearl's FP8 quantization noise
ERR = 0.08


def _e2e() -> Any:
    """`e2e.py`, loaded from its file: the call check (`call_checks` on `shape_cases`) and `whole_check`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pearl_c_e2e", HERE / "e2e.py")
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)
    return e2e


def kernel_directory(scheme: str, card: str) -> Path:
    """The tree's kernel directory of the PoUW kernel scheme ``scheme`` on the card ``card`` (its short name): its
    family's, built for the card's target (``pearl_c_sm90`` for ``pearl-c-h100-v1-h3`` on an H200)."""
    from verity.protocols.accounting.compute.pouw.verifier.kernel_schemes import named
    from verity_bench.schemes import arch

    return KERNELS / f"{named(scheme).family.replace('-', '_')}_{arch(card).removesuffix('a').replace('_', '')}"


class _Coins:
    """`e2e.Coins`, trimmed: one epoch's challenger, each coin opened once."""

    def __init__(self, epoch: int) -> None:
        from verity.protocols.accounting.compute.pouw.verifier import verifier
        from verity.protocols.accounting.compute.pouw.verifier.coins import Challenger

        self.challenger, self.opened = Challenger(epoch, verifier.coins), {}
        self.commitment = self.challenger.commitment

    def open(self, label: str, after: bytes) -> Any:
        from verity.protocols.accounting.compute.pouw.verifier.protocol import DomainError

        if label in self.opened:
            if self.opened[label].after != after:
                raise DomainError(f"epoch {self.commitment.epoch}'s {label!r} coin is already open, after another record")
            return self.opened[label]
        self.opened[label] = self.challenger.open(label, after)
        return self.opened[label]


class _LiveSession:
    """The epoch's salt coin from a session of the auditor device's challenger, through infra's gate: ``gate`` its argv
    prefix as one string (``ssh -i KEY -o IdentitiesOnly=yes research@HOST``, the lane's own gate key) and ``peer`` this
    node's name in the host's peers (``--peer``), whose 127.0.0.1 the challenger reaches through the node's attester
    coin link. Reserved at
    construction (`commit_coins.ChallengerGate`: ``reserve --peer NAME``, Pearl-C's layout, its salt coin alone,
    `window.SALT_FAMILIES`), opened on salt epoch ``k``'s root over the fixed operands' root once they are committed, its
    salt coin opened after it (`pouw_h1_window.open_session`), closed with the arm, so it ends complete. Refused before
    reserving while ``lane``'s open period in the auditor's ledger (``period``) is H1's, and before its open when the
    reserve joined another period (`pouw_h1_window.reserve_salt`)."""

    #: the arm's salt epoch over its fixed operands' root: one install, under a fresh hm96 key
    k = 0

    def __init__(self, gate: str, peer: str, epoch: int, lane: str, run: Any = None) -> None:
        from verity_vllm.pipeline.commit_coins import ChallengerGate
        from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

        self.gate = ChallengerGate(gate, run)
        self.reserved, self.period = PHW.reserve_salt(self.gate.call, self.gate.ledger, lane, peer, epoch)
        self.client = self.commitment = None
        self.carried: dict[str, Any] = {}

    def open(self, entry: str, root: bytes) -> Any:
        from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

        self.client, self.commitment, opening, _, self.carried = PHW.open_session(
            entry, self.gate.call, self.reserved["session"], root, self.k)
        return opening

    def close(self) -> None:
        if self.client is not None:
            client, self.client = self.client, None
            client.close()


class LiveDraw:
    """`verify_run.verify_passes`' ``draw`` from a sample session of its own on the auditor device's challenger, through
    infra's gate (``gate``, ``peer`` and ``lane`` as `_LiveSession`'s; `run` its gate's subprocess in tests): called on
    the run's commitment C, it reserves a session withheld with the sampling coin alone (``reserve --withhold``,
    `window.DRAW_FAMILIES`, `pouw_h1_window.reserve_draw` as H1's window samples), opens it on C and takes its
    ``pouw/draw`` coin after C (`pouw_h1_window.DrawSession`), whose full opening and commitment the session releases at
    its close.
    Refused (`CoinError`) before reserving while ``lane`` has a period open in the auditor's ledger, and before its open
    when the reserve joined one, so the sample counts toward no period's sessions; and when the gate reserves it plain.
    With ``period``, the Pearl-C period the run's salt session joined, the sample must be the lane's first reserve after
    that period closed (`pouw_h1_window.after_close`, the compose's rule): refused before reserving unless the ledger
    closes it with no reserve of the lane since, and before its coin is asked unless the reserve is that first one.
    Returns (the coin's 64 bytes, the coin record's ``draw``: `verify_run`'s module docstring, "The coin record")."""

    def __init__(self, gate: str, peer: str, lane: str, run: Any = None, at: str = "127.0.0.1:0",
                 period: str | None = None) -> None:
        from verity_vllm.pipeline.commit_coins import ChallengerGate

        self.gate, self.peer, self.lane, self.at, self.period = ChallengerGate(gate, run), peer, lane, at, period

    def __call__(self, root: bytes, asking: Any) -> tuple[bytes, dict[str, Any]]:
        from verity.protocols.accounting.compute.pouw.verifier import window as Wd
        from verity.protocols.proof.spot.verifier.challenger.challenger import CoinError
        from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

        lines = self.gate.ledger()
        period = PHW.open_period(lines, self.lane)
        if period is not None:
            raise CoinError(f"refused before reserving the run's draw session: lane {self.lane}'s period "
                            f"{period['period']} is open, and the draw joins no period (close it first)")
        if self.period is not None:
            closed, after = PHW.after_close(lines, self.lane, self.period)
            if not closed:
                raise CoinError(f"refused before reserving the run's draw session: the auditor's ledger doesn't close "
                                f"the run's Pearl-C period {self.period}, which the draw must follow")
            if after is not None:
                raise CoinError(f"refused before reserving the run's draw session: lane {self.lane} reserved session "
                                f"{after.get('session')} after its Pearl-C period {self.period} closed, so no draw now "
                                "is the lane's first reserve after it, the one draw a compose takes")
        reserved = PHW.reserve_draw(self.gate.call, self.peer)
        if reserved.get("period") is not None:
            raise CoinError(f"draw session {reserved['session']} joined period {reserved['period']}: refused before its "
                            "open, since the draw joins no period")
        if self.period is not None:
            _, after = PHW.after_close(self.gate.ledger(), self.lane, self.period)
            if after is None or after.get("session") != reserved["session"]:
                raise CoinError(f"draw session {reserved['session']} is not lane {self.lane}'s first reserve after its "
                                f"Pearl-C period {self.period} closed (that is "
                                f"{None if after is None else after.get('session')}): refused before its coin is asked")
        asking({"reserved": reserved})
        session = PHW.DrawSession("pearl-plus verify", self.at, reserved, PHW.gate_of(self.gate.call))
        opening = session.record(root)
        return opening.coin, {"from": "challenger", "label": Wd.DRAW, "index": 0, "reserved": reserved,
                              "session": session.session, "commitment": session.commitment.to_json(),
                              "opening": opening.to_json()}


def live_draw(gate: str | None, peer: str | None, lane: str, run: Any = None,
              period: str | None = None) -> LiveDraw | None:
    """`LiveDraw` under ``gate`` and ``peer`` after the run's Pearl-C period ``period``, None when neither is given (the
    node's sample); ValueError for one alone."""
    if (gate, peer) == (None, None):
        return None
    if gate is None or peer is None:
        raise ValueError("the live draw takes PEARLC_CHALLENGER_GATE (the gate's ssh command) and PEARLC_CHALLENGER_PEER "
                         "(this node's name in the challenger host's peers) together")
    return LiveDraw(gate, peer, lane, run, period=period)


class PearlCArm:
    id = "pouw-pearl-c"
    protocols: tuple[str, ...] = ()
    precision = "bfloat16"
    control = None
    resident = True
    #: `PearlCDeviceRun`'s options beside the install's
    run_options: dict[str, Any] = {}

    def __init__(self, protocol: str | None = None) -> None:
        """``protocol`` is the useful-work scheme's registered name; an arm built only for its gate (`gate`) names none."""
        self.protocol = protocol
        self._kernel: tuple[str, Path] | None = None
        self.cubin = settings.PEARLC_CUBIN
        self.schedule = settings.PEARLC_SCHEDULE
        self.epoch = settings.PEARLC_BENCH_EPOCH
        self.gate_command = settings.PEARLC_CHALLENGER_GATE
        self.peer = settings.PEARLC_CHALLENGER_PEER
        self.lane = settings.PEARLC_CHALLENGER_LANE
        self.run = self.stack = self.weights = self.session = None
        self.coins: dict[str, Any] = {"source": "in-process"}

    def kernel(self, card: str | None = None) -> tuple[str, Path]:
        """The kernel scheme and kernel directory the arm serves with, fixed at the first call: the scheme the arm's
        served useful-work scheme's declaration runs (its ``kernels``, `scheme.declared`; `verity_bench.schemes.served`) on the card ``card`` (its short name;
        where None, the leased card's, `kit.card_short` of the run's stamp), for the served model's weights, which
        registration pins, and its directory ``PEARLC_KERNEL`` or the tree's (`kernel_directory`)."""
        if self._kernel is None:
            from verity.protocols.accounting.scheme import declared
            from verity_bench import kit
            from verity_bench.schemes import arch, served, useful_work

            if self.protocol is None:
                raise ValueError(f"{self.id} names no useful-work scheme: build it as {type(self).__name__}(NAME), NAME "
                                 "a registered compute scheme's name")
            card = card or kit.card_short(kit.host_stamp()["card"])
            scheme = declared(served(useful_work(self.protocol)), "kernels")(card, pinned=True, arch=arch(card))
            self._kernel = scheme, Path(settings.PEARLC_KERNEL or kernel_directory(scheme, card))
        return self._kernel

    @property
    def scheme(self) -> str:
        return self.kernel()[0]

    @property
    def kernel_dir(self) -> Path:
        return self.kernel()[1]

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
        from verity.protocols.accounting.compute.pouw.verifier.audit import weights_root
        from verity.protocols.run import schemes
        from verity.protocols.accounting.scheme import declared
        from verity.hooks import ComputeBlockWith
        from verity_vllm import messages
        from verity_vllm.protocols.linear import vllm_site
        from verity_vllm.protocols.linear.api import Arm as StackArm
        from verity_vllm.protocols.linear.api import PassSpec, Stack
        from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
        from verity.protocols.accounting.compute.pouw.prover.serving import pearl_c as PPD
        from verity_torch.pouw.pearl_c import run_device as PPDR

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
        weight_salts = PPDR.fresh_weight_salts(weights)
        roots = PPDR.weight_roots(kernel, weights, scheme, weight_salts=weight_salts)
        if self.session is not None:
            opening = self.session.open(f"pouw:{scheme}", weights_root(roots))
            commitment = self.session.commitment
            self.coins = {"source": "challenger", "session": self.session.carried, "pass_index": self.session.k,
                          "salt_after": declared(schemes.get("pearl-plus"), "salt_open_root")(weights_root(roots),
                                                                                             self.session.k).hex()}
        else:
            coins = _Coins(self.epoch)
            opening, commitment = coins.open("salt", weights_root(roots)), coins.commitment
        tiles, rows = self._capacity(target, kernel.run.TILE, PPD.ROW_ALIGN,
                                     [t.shape[0] for t in weights.values()], PPD.LEAF_CAPACITY, PPD.FLAG_ROWS)
        try:
            run = PPDR.PearlCDeviceRun(pouw_schemes.get(scheme), f"pouw:{scheme}", kernel, weights, roots,
                                      commitment, opening, schedule=self.schedule,
                                      capacity=tiles, flag_rows=rows, weight_salts=weight_salts,
                                      pass_index=None if self.session is None else self.session.k, **self.run_options)
        except BaseException:
            if self.session is not None:
                self.session.close()
            raise
        run.commits["coin_source"] = self.coins["source"]
        if self.session is not None:
            run.commits["session"] = self.session.carried
        stack = Stack([site for site, _ in sites], PPDR.PearlCExecutor(run))
        bad = stack.refusals()
        if bad:
            raise RuntimeError(f"{scheme} refuses the model: " + "; ".join(bad))
        messages.current().request(run.entry, ComputeBlockWith(stack))
        vllm_site.attach(sites, target.hooks)
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
        """`e2e.gate`'s call check (`e2e.call_checks`, each pass on a twin of the run): y against x·Wᵀ and the call twice under
        the salt."""
        import torch

        run, weights, e2e = self.run, self.weights, _e2e()
        cases = e2e.shape_cases(weights, (16, 64), torch.Generator(device=next(iter(weights.values())).device).manual_seed(20260930))
        checks = [{"n": W.shape[0], "k": W.shape[1], "m": x.shape[0], **v}
                  for (_, W, x), v in zip(cases, e2e.call_checks(run, cases), strict=True)]
        from verity_bench.vllm.runner import PER_PASS_TOKENS

        return {"passed": all(c["ok"] for c in checks), "scheme": run.scheme.name, "schedule": self.schedule,
                "coins": self.coins, "checks": checks, PER_PASS_TOKENS: True}

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

    def __init__(self, protocol: str | None = None) -> None:
        super().__init__(protocol)
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

    def kernel(self, card: str | None = None) -> tuple[str, Path]:
        """`PearlCArm.kernel`, the `RTX_ONLY` options dropped from `run_options` where its directory is another kernel's."""
        got = super().kernel(card)
        if got[1].name != "pearl_c_sm120":
            self.run_options = {k: v for k, v in self.run_options.items() if k not in self.RTX_ONLY}
        return got

    def engine_overrides(self, target: Any) -> dict[str, Any]:
        self.capture = list(settings.PEARLC_CAPTURE_SIZES or (1, 8, 32, 128))
        return {**super().engine_overrides(target), "enforce_eager": False, "execution": self.vllm_execution,
                "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                       "cudagraph_capture_sizes": self.capture}}

    def ready(self, target: Any) -> None:
        from verity_vllm.engine import handle
        from verity_torch.pouw.pearl_c.whole import WholeStep

        super().ready(target)
        self.whole = WholeStep(self.run, self.stack, handle.model_runner(target.engine.vllm), target.hooks)
        self.whole.prepare()
        self.totals = dict(self.whole.take())

    def gate(self, target: Any) -> dict[str, Any]:
        got = super().gate(target)
        whole = _e2e().whole_check(self.run, self.weights, m=max(self.capture))
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


PEARL_C = PearlCArm("pearl-plus")
PEARL_C_WHOLE = PearlCWholeArm("pearl-plus")
