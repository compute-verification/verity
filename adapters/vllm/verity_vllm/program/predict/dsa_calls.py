"""The calls the DeepSeek decoders' forwards are written with (`dsa._Model`, `dsv4._Model`): kinds, collectives, GEMMs and the
sampler over one rank's net, each token-count-dependent selection at the call's launch (`common._Launch`)."""

from __future__ import annotations

from typing import Any, NoReturn, Protocol

from verity.circuits.defs import Builder
from verity.circuits.refs import Coll, PartLog, Refs, concat, strided_view
from verity.circuits.types import Type

from .common import (
    Unexpressible,
    _arr_shape,
    _Launch,
    _Peers,
    _ptypes,
    _Stochastic,
    kind,
    select_token,
)


def _refuse(fact: str) -> NoReturn:
    raise Unexpressible(f"vocabulary cannot express: {fact}")


class _CallsNet(Protocol):
    """What the calls both decoders write with (`_Calls`) read off a rank's net, which `dsa.Net` and `dsv4.Net` each state."""

    C: dict[str, Any]
    _leaves: list[tuple[str, Type, object]]
    settings: Any
    world: int
    dot: Any
    tp: Any

    @property
    def capture_sizes(self) -> tuple[int, ...]: ...

    @property
    def max_num_seqs(self) -> int | None: ...


class _Calls:
    """The calls a decoder's forward is written with, emitted into `B` over one rank's net: the kinds, the collectives, the
    GEMMs and the sampler; every token-count-dependent selection reads the call's launch (`launch`).  `dsa._Model` and
    `dsv4._Model` are each one decoder's forward over them; `batched` and `table_reads` are `_Model`'s forms, off in any other."""

    reads_launch = True
    batched = False
    table_reads = False

    def __init__(self, net: _CallsNet, B: Builder, W: Coll, peers: _Peers) -> None:
        self.net, self.C, self.B, self.peers = net, net.C, B, peers
        self.launch = _Launch(capture_sizes=net.capture_sizes, max_num_seqs=net.max_num_seqs or 0)
        self.w = {key: W[i] for i, (_, _, key) in enumerate(net._leaves) if key is not None}
        self.logs: dict[int, dict[str, PartLog]] = {}
        self.topk: dict = {}
        self._kinds: dict[tuple, object] = {}
        self._pads: dict[tuple[int, int], Refs] = {}
        C = self.C
        self.H, self.EPS, self.G = int(C["H"]), float(C["EPS"]), int(C["FP8_BLOCK"])

    def k(self, role: str, **statics):
        key = (role, repr(sorted(statics.items())))
        if key not in self._kinds:
            self._kinds[key] = kind(role, **statics)
        return self._kinds[key]

    def call(self, fn, *refs: Refs) -> Coll:
        return self.B.call(fn, *(Coll(t, r) for t, r in zip(_ptypes(fn), refs)))

    def pad(self, w: int, bits: int, words: int, count: int) -> Refs:
        """``count`` words of ``bits``: a slice of the Program's one block of them (`vllm_dsa.pad_words`), as `vllm_dsa` pads."""
        if (w, bits) not in self._pads:
            self._pads[(w, bits)] = self.B.batch(self.k("const", W=w, BITS=bits), axes=(), n=words).refs
        return self._pads[(w, bits)].slice(0, count)

    def sample(self, row: Refs, event: int, pos: int, policy: _Stochastic | None = None) -> Coll:
        """The token of the row's logits (`common.select_token`): this rank's vocabulary slice, gathered across the ranks."""
        V, W = int(self.C["V"]), self.net.world
        lg = self.cublas([row], self.w["lm_head"], sampled=True)
        gathered = self.call(self.k("all_gather", WORLD=W, N=V), self.peers.rows(lg)[0])
        return select_token(self.call, self.k, gathered.refs, V * W, pos, policy)

    # -- collectives (`collectives.AllReduceRule`, `FusedAllReduceRmsNormRule`)

    def all_reduce(self, rows: list[Refs]) -> list[Refs]:
        """One all-reduce over the rows on the route vLLM takes at the launch's tokens (`tp_allreduce.served_route`), the captured one
        -> each row's sum."""
        from verity_vllm.definitions.registry import tp_allreduce as AR
        from verity_vllm.program.frontend.rules.vllm_bindings.collectives import (
            _ROUTE_KINDS,
        )
        n, N = len(rows), rows[0].count
        M = self.launch.tokens(n)
        setting, route = AR.served_route(self.net.settings, M, N)
        c = AR.covered(setting, N, route, M) if setting else None
        if c is None:
            _refuse(f"an all-reduce of {M} x {N} on route {route!r} under {self.net.settings} (no capture covers it: tp_allreduce.covered)")
        try:
            _, st = AR.route_binding(c, self.net.world, M, N, n)
        except ValueError as e:
            _refuse(str(e))
        out = self.call(self.k(_ROUTE_KINDS[c.definition], **st), self.peers.whole(rows)).refs
        return [out.slice(i * N, (i + 1) * N) for i in range(n)]

    def reduce_norm(self, parts: list[Refs], res: list[Refs], w: Coll) -> tuple[list[Refs], list[Refs]]:
        """The all-reduce of the rows' partials, the residual add and the RMSNorm -> (normed rows, residual rows): FlashInfer's fused
        kernel where vLLM fuses (its FlashInfer all-reduce on and M within `fused_max_tokens`), else the all-reduce then the fused norm."""
        from verity_vllm.definitions.registry import tp_allreduce as AR
        M, N, W, s = self.launch.tokens(len(parts)), parts[0].count, self.net.world, self.net.settings
        if s["flashinfer"] and AR.fused_max_tokens(N, W) >= M:
            setting, _ = AR.served_route(s, M, N)
            c = AR.covered(setting, N, "fused", M) if setting else None
            if c is None:
                _refuse(f"a fused all-reduce + norm of {M} x {N} under {s} (no capture covers it: tp_allreduce.covered)")
            fn = self.k("all_reduce_rmsnorm_mnnvl", WORLD=W, N=N, SHOT=AR.mnnvl_shot(M, N, W), CLUSTER=AR.mnnvl_grid(M, N)[1], EPS=self.EPS,
                        FMA=bool(c.statics["FMA"]))
            outs = [self.call(fn, p, r, w.refs) for p, r in zip(self.peers.rows(parts), res)]
        else:
            fn = self.k("rmsnorm_fused", N=N, EPS=self.EPS, TARGET=self.net.tp, M=M)
            outs = [self.call(fn, x, r, w.refs) for x, r in zip(self.all_reduce(parts), res)]
        return [o[0].refs for o in outs], [o[1].refs for o in outs]

    # -- GEMMs

    def fp8_linear(self, xs: list[Refs], w: Coll, ws: Coll, rows: int | None = None) -> list[Refs]:
        """vLLM's block-FP8 linear (`DynamicDeepGemmFp8LinearRule`): FlashInfer below 32 rows, else DeepGEMM (its output port 2).
        The rows are the launch's, or `rows` for a call outside the graphs (kv_b_proj in the dense-MHA prefill)."""
        N, K = _arr_shape(w.type)
        small = (self.launch.tokens(len(xs)) if rows is None else int(rows)) < 32
        fn = self.k("flashinfer_fp8_linear" if small else "deepgemm_fp8_linear", K=K, N=N)
        outs = [self.call(fn, x, w.refs, ws.refs) for x in xs]
        return [(o if small else o[2]).refs for o in outs]

    def cublas(self, xs: list[Refs], w: Coll, f32: bool = False, sampled: bool = False) -> list[Refs]:
        """An unbiased bf16 GEMM on cuBLAS's captured route at the launch's tokens (its sequences for the `sampled` rows): bf16 out
        (`CublasLinearRule`), or f32 (`CublasRouterRule`: the router family's route, or a direct call's where the shape has its own)."""
        from verity_vllm.definitions.registry.bf16_gemm_check import (
            default_cublas_route,
        )
        from verity_vllm.program.frontend.rules.vllm_bindings.bf16_gemm import (
            _f32out_phase,
        )
        M, (N, K) = self.launch.tokens(len(xs), sampled), _arr_shape(w.type)
        phase = _f32out_phase(K, N) if f32 else "linear"
        route = default_cublas_route(phase, M, K, N)
        if route is None:
            _refuse(f"a bf16 {phase} GEMM of M={M}, K={K}, N={N} (no captured vLLM-default cuBLAS route)")
        split = route[0] == "splitk"
        role = ("cublas_splitk_gemm_f32out" if split else "cublas_gemm_f32out") if f32 else ("cublas_splitk_gemm" if split else "gemm_dot")
        fn = self.k(role, K=K, N=N, DOT=self.net.dot, **route[1])
        return [self.call(fn, x, w.refs).refs for x in xs]

    def router(self, xs: list[Refs], w: Coll) -> list[Refs]:
        """`GateLinear`'s f32 logits on the tier vLLM dispatches (`LlBf16GemmRule`, `CublasRouterRule`)."""
        from verity_vllm.definitions.registry.bf16_gemm_check import (
            DEFAULT_LL_BF16_FMA,
            dispatch_sm90,
        )
        M, (N, K) = self.launch.tokens(len(xs)), _arr_shape(w.type)
        tier, st = dispatch_sm90(M, K, N)
        if tier == "cublas":
            return self.cublas(xs, w, f32=True)
        if tier == "dotprod":
            fn = self.k("ll_bf16_dotprod", N=N, K=K, FMA=DEFAULT_LL_BF16_FMA, **st)
        else:
            fn = self.k("ll_bf16_splitk", N=N, K=K, DOT=self.net.dot, **st)
        return [self.call(fn, x, w.refs).refs for x in xs]

    def absorb(self, xs: list[Refs], w: Coll, B: int, K: int, N: int) -> list[Refs]:
        """MLA's latent absorption `bmm(x^T, W)^T` per row (`MlaAbsorbTransposePatternRule`): each row's B heads against W^T, the
        [B][K][N] weight read as [B][N][K] (the frontend's permute view: one strided part per head)."""
        fn = self.k("batched_gemm", B=B, K=K, N=N, DOT=self.net.dot)
        wt = concat([strided_view(w, b * K * N, K, 1, N, N * K) for b in range(B)])
        return [self.call(fn, x, wt).refs for x in xs]

    def silu(self, rows: list[Refs]) -> list[Refs]:
        fn = self.k("silu_mul", I=rows[0].count // 2)
        return [self.call(fn, r).refs for r in rows]
