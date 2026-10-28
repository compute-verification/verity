"""The DeepSeek sparse-attention MoE decoder (`FAMILY_FACTS[...]["arch"]` "dsa-moe": GLM-5.3's `glm_moe_dsa`, which vLLM serves as
`DeepseekV32ForCausalLM`) for `build.build_program`: one tensor-parallel rank of a block-FP8 checkpoint on cc 9.0 without batch
invariance under vLLM d9105ea80's served defaults, written with the calls the Build's rules make (`rules.vllm_dsa`,
`vllm_bindings.fp8`, `bf16_gemm`, `collectives`, `vllm_moe_fp8`, `vllm_moe_router`) at the selections they read, each at the call's
launch (`common._Launch`: the batch a request's calls run in without batch invariance, else the call's own rows): the captured cuBLAS
and all-reduce routes (`registry.bf16_gemm_check`, `registry.tp_allreduce`), vLLM's tuned fused-MoE tile for the device
(`engine.profiles.devices`) and FA3's split count (`registry.sparse_mla_split`).  The configuration is the generic profile's
(`generic.derive_config` at the rank) and the weights record its map (`generic._mla_weight_field`) in the family's module order.

Vocabulary: EmbeddingShard and the embedding's all-reduce; RMSNorm (layer 0's input norm); per layer the block-FP8 linears (FlashInfer
below 32 rows, else DeepGEMM), DsaRMSNorm of q_c and kv_c, the interleaved RoPE of k_pe and of each head's q_pe, the latent absorptions
(BatchedGemm), SparseMlaSplit over the layer's latent history, and on an indexer layer the bf16 weights projection, DsaIndexerKRow,
DsaIndexerQ, DsaMqaLogits and DsaTopKShortRow; the all-reduce + residual + RMSNorm (FlashInfer's fused MNNVL kernel within its token
budget, else the all-reduce and RMSNormFusedCuda); the SiLU MLP below FIRST_DENSE, and from it on the MoE block: the f32 router GEMM,
the shared expert, MoeRouterSigmoidBias, Fp8GroupQuantUe8m0, the block-FP8 expert GEMMs per (token, slot) pair, MoeSum and the
shared expert's add; then the last row's logits, their all-gather and the token select (greedy, or the seeded temperature + top-p
select: `common.select_token`).  A fact outside it raises `Unexpressible`, φ's
when the decoder is made and a launch's (a prefill past the decode threshold, a row of more than INDEX_TOPK keys, a GEMM, all-reduce
or MoE tile no record covers) when the launch is written.
"""

from __future__ import annotations

from verity.ml.circuits.defs import Builder
from verity.ml.circuits.refs import Coll, PartLog, Refs, concat, strided_view
from verity.ml.circuits.types import Type, Value

from .common import Unexpressible, _arr_shape, _Launch, _Peers, _ptypes, _Stochastic, kind, select_token, served_class
from .phi import Phi


def _refuse(fact: str):
    raise Unexpressible(f"vocabulary cannot express: {fact}")


def _at(t: Type, path: tuple) -> Type:
    for p in path:
        t = t.elem if isinstance(p, int) else dict(t.fields)[p]
    return t


class Net:
    """One rank's decoder: the configuration, the weights record (`leaves`), the peers a pass reads (`peer_shapes`) and the target's
    selections the forward (`model`) reads."""

    DOMAIN = ((8,), "eager", ("float8_e4m3fn",))

    def __init__(self, phi: Phi, M: int, history: int) -> None:
        from verity_vllm.engine.profiles import family_facts as FF
        from verity_vllm.observe.fold.profiles.generic import _MLA_SCALES, _mla_weight_field, derive_config
        from verity_vllm.program import model as analytic
        from verity_vllm.program.frontend import targets as TG
        from verity_vllm.program.frontend.rules.vllm_dsa import FA3_SPARSE_GEOMETRY
        from verity.ml.definitions.registry import dsa
        from verity.ml.definitions.registry.tp_allreduce import settings_of
        hf = phi.hf_config
        self.world, self.rank = int(phi.tp["world"]), int(phi.tp.get("rank", 0))
        try:
            C, _ = derive_config(hf, repo=phi.model.get("repo"), revision=phi.model.get("revision"), world=self.world)
        except (KeyError, ValueError) as e:
            raise Unexpressible(f"config.json: {e}") from None
        self.C, self.tp, self.device = C, phi.target_profile(), phi.target.get("device_name")
        self.indexers = frozenset(C.get("INDEXER_LAYERS") or ())
        facts = []
        if self.tp.compute_capability != (9, 0):
            facts.append(f"target cc {self.tp.capability_str} (the served defaults are stated on cc 9.0)")
        if not self.device:
            facts.append("a target without a device name (vLLM keys its tuned fused-MoE tiles by it)")
        if self.tp.moe_backend != "triton":
            facts.append(f"moe_backend {self.tp.moe_backend!r} (the decoder writes the Triton experts, moe_backend 'triton')")
        if C["tie_word_embeddings"]:
            facts.append("tied embeddings")
        if C["ROPE_NEOX"] or not C["INDEX_ROPE_INTERLEAVE"]:
            facts.append("a NeoX-style RoPE (the DSA kernels rotate interleaved pairs)")
        if (C["N_GROUP"], C["TOPK_GROUP"], C["RENORMALIZE"], C["SCORING"], bool(C["GATE_IN_EXPERTS"])) != (1, 1, True, "sigmoid", True):
            facts.append(f"a router of {C['N_GROUP']} groups (top {C['TOPK_GROUP']}), renormalize {C['RENORMALIZE']}, {C['SCORING']} scores, "
                         f"gate in the experts {C['GATE_IN_EXPERTS']} (MoeRouterSigmoidBias: one group, renormalized, sigmoid, in the "
                         f"shared-expert forward)")
        if (int(C["QK_ROPE"]), int(C["KV_LORA"])) not in FA3_SPARSE_GEOMETRY:
            facts.append(f"FA3's sparse MLA at rope {C['QK_ROPE']}, latent {C['KV_LORA']} ({sorted(FA3_SPARSE_GEOMETRY)})")
        if 0 not in self.indexers:
            facts.append("a first layer without an indexer (an IndexShare layer reads the top-k of the indexer layer before it)")
        try:
            dsa.check_row_statics(int(C["INDEX_HEAD_DIM"]), int(C["QK_ROPE"]))
            dsa.check_eps(float(C["INDEX_K_EPS"]))
            dsa.check_logit_statics(int(C["INDEX_HEADS"]), int(C["INDEX_HEAD_DIM"]))
            self.dot = TG.tensor_core_dot(self.tp)
        except (ValueError, TG.UnregisteredTarget) as e:
            facts.append(str(e))
        if facts:
            _refuse("; ".join(facts))
        self.settings = settings_of(bool(phi.engine.get("disable_custom_all_reduce")), phi.engine.get("env") or {})
        ff = FF.facts_for(hf)
        self.model_class, self.NL = served_class(ff["vllm_class"]), int(C["NL"])
        self.nodes_per_row_layer = 26 + int(C["NH"]) + 4 * int(C["TOPK"])
        T, field = analytic.mla_weights_type(C), _mla_weight_field(hf, C)
        self._leaves: list[tuple[str, Type, object]] = []

        def add(name: str, layer: int | None = None) -> None:
            path = field(name)
            if path is not None:
                self._leaves.append((name, _at(T, path), path[-1] if layer is None else (layer, path[-1])))
            elif _MLA_SCALES.match(name):
                self._leaves.append((name, Value(32), None))

        add("model.embed_tokens.weight")
        for L in range(self.NL):
            for rest in ff["params"]["order"]:
                add(f"model.layers.{L}.{rest}", L)
        add("model.norm.weight")
        add("lm_head.weight")
        for L in range(self.NL):
            for rest in ff["params"]["buffers"]:
                add(f"model.layers.{L}.{rest}")

    def leaves(self) -> list[tuple[str, Type]]:
        return [(name, t) for name, t, _ in self._leaves]

    def peer_shapes(self, r: int) -> list[tuple[int, int]]:
        """A pass of r rows: the embedding's all-reduce, each layer's attention and MLP all-reduces, the sampled row's logits gather."""
        H = int(self.C["H"])
        return [(r, H)] * (1 + 2 * self.NL) + [(1, int(self.C["V"]))]

    def model(self, B: Builder, W: Coll, peers: _Peers) -> "_Model":
        return _Model(self, B, W, peers)


class _Model:
    """The forward over one call's rows, emitted into `B`, each layer's latent and index-K histories carried across calls as
    `vllm_dsa` keeps them (one `PartLog` per history); every token-count-dependent selection reads the call's launch (`launch`)."""

    reads_launch = True

    def __init__(self, net: Net, B: Builder, W: Coll, peers: _Peers) -> None:
        self.net, self.C, self.B, self.peers, self.launch = net, net.C, B, peers, _Launch()
        self.w = {key: W[i] for i, (_, _, key) in enumerate(net._leaves) if key is not None}
        self.logs: dict[int, dict[str, PartLog]] = {}
        self._kinds: dict[tuple, object] = {}
        C = self.C
        self.H, self.EPS, self.G = int(C["H"]), float(C["EPS"]), int(C["FP8_BLOCK"])

    def k(self, role: str, **statics):
        key = (role, repr(sorted(statics.items())))
        if key not in self._kinds:
            self._kinds[key] = kind(role, **statics)
        return self._kinds[key]

    def call(self, fn, *refs: Refs) -> Coll:
        return self.B.call(fn, *(Coll(t, r) for t, r in zip(_ptypes(fn), refs)))

    def forward(self, toks: list[Coll], pos0: int) -> list[Refs]:
        """Token rows (I32 values) at positions pos0.. -> the final-normed hidden rows."""
        C, net, M = self.C, self.net, len(toks)
        V = int(C["V"])
        emb = self.k("embedding_shard", VS=V, H=self.H, START=net.rank * V)
        res = self.all_reduce([self.call(emb, t.refs, self.w["embed"].refs).refs for t in toks])
        norm = self.k("rmsnorm", N=self.H, EPS=self.EPS, TARGET=net.tp, M=self.launch.tokens(M))
        x = [self.call(norm, r, self.w[(0, "ln1")].refs).refs for r in res]
        for L in range(net.NL):
            x, res = self.reduce_norm(self.attention(L, x, pos0), res, self.w[(L, "ln2")])
            nxt = self.w[(L + 1, "ln1")] if L + 1 < net.NL else self.w["norm"]
            x, res = self.reduce_norm(self.moe(L, x) if L >= int(C["FIRST_DENSE"]) else self.mlp(L, x), res, nxt)
        return x

    def sample(self, row: Refs, event: int, pos: int, policy: _Stochastic | None = None) -> Coll:
        """The token of the row's logits (`common.select_token`): this rank's vocabulary slice, gathered across the ranks."""
        V, W = int(self.C["V"]), self.net.world
        lg = self.cublas([row], self.w["lm_head"], sampled=True)
        gathered = self.call(self.k("all_gather", WORLD=W, N=V), self.peers.rows(lg)[0])
        return select_token(self.call, self.k, gathered.refs, V * W, self.net.tp, event, pos, policy)

    # -- collectives (`collectives.AllReduceRule`, `FusedAllReduceRmsNormRule`)

    def all_reduce(self, rows: list[Refs]) -> list[Refs]:
        """One all-reduce over the rows on the route vLLM takes at the launch's tokens (`tp_allreduce.served_route`), the captured one
        -> each row's sum."""
        from verity_vllm.program.frontend.rules.vllm_bindings.collectives import _ROUTE_KINDS
        from verity.ml.definitions.registry import tp_allreduce as AR
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
        from verity.ml.definitions.registry import tp_allreduce as AR
        M, N, W, s = self.launch.tokens(len(parts)), parts[0].count, self.net.world, self.net.settings
        if s["flashinfer"] and M <= AR.fused_max_tokens(N, W):
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

    def fp8_linear(self, xs: list[Refs], w: Coll, ws: Coll) -> list[Refs]:
        """vLLM's block-FP8 linear (`DynamicDeepGemmFp8LinearRule`): FlashInfer below 32 rows, else DeepGEMM (its output port 2)."""
        N, K = _arr_shape(w.type)
        small = self.launch.tokens(len(xs)) < 32
        fn = self.k("flashinfer_fp8_linear" if small else "deepgemm_fp8_linear", K=K, N=N)
        outs = [self.call(fn, x, w.refs, ws.refs) for x in xs]
        return [(o if small else o[2]).refs for o in outs]

    def cublas(self, xs: list[Refs], w: Coll, f32: bool = False, sampled: bool = False) -> list[Refs]:
        """An unbiased bf16 GEMM on cuBLAS's captured route at the launch's tokens (its sequences for the `sampled` rows): bf16 out
        (`CublasLinearRule`), or f32 (`CublasRouterRule`: the router family's route, or a direct call's where the shape has its own)."""
        from verity_vllm.program.frontend.rules.vllm_bindings.bf16_gemm import _f32out_phase
        from verity.ml.definitions.registry.bf16_gemm_check import default_cublas_route
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
        from verity.ml.definitions.registry.bf16_gemm_check import DEFAULT_LL_BF16_FMA, dispatch_sm90
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

    # -- the layer

    def attention(self, L: int, x: list[Refs], pos0: int) -> list[Refs]:
        """One layer's attention over the rows x at positions pos0.. (`DsaNormRopeRule`, `DsaQRule`, `DsaIndexerAttnRule`, op-major
        over the rows) -> the o projection's partials."""
        from verity_vllm.program.frontend.rules.vllm_dsa import DECODE_THRESHOLD, DECODE_THRESHOLD_DEFAULT, FA3_SPARSE_GEOMETRY
        from verity.ml.definitions.registry.sparse_mla_split import launch_num_splits
        C, M = self.C, len(x)
        QL, KVL, R, NOPE, VH, NH = (int(C[k]) for k in ("Q_LORA", "KV_LORA", "QK_ROPE", "QK_NOPE", "V_HEAD", "NH"))
        IH, ID, TOPK = int(C["INDEX_HEADS"]), int(C["INDEX_HEAD_DIM"]), int(C["INDEX_TOPK"])
        D, indexer = NOPE + R, L in self.net.indexers
        w = lambda f: self.w[(L, f)]  # noqa: E731
        limit = DECODE_THRESHOLD.get(NH, DECODE_THRESHOLD_DEFAULT)
        if M > limit:
            _refuse(f"a call of {M} query rows, a prefill (decode threshold {limit} at {NH} query heads per rank), which the dense-MHA "
                    f"prefill backend serves")
        logs = self.logs.setdefault(L, {"latent": PartLog(), "ik": PartLog(), "iks": PartLog()})
        keys = [len(logs["latent"]) + i + 1 for i in range(M)]
        if keys[-1] > TOPK:
            _refuse(f"a row of {keys[-1]} keys > TOPK {TOPK} (the top-k kernels' radix selection)")
        cs = [self.w["cos_sin"].refs.slice((pos0 + i) * R, (pos0 + i + 1) * R) for i in range(M)]
        qkv = self.fp8_linear(x, w("qkv_a_w"), w("qkv_a_ws"))
        wkp = self.cublas(x, w("wk_w")) if indexer else None
        qn, kn, kr = self.k("dsa_rmsnorm", N=QL, EPS=self.EPS), self.k("dsa_rmsnorm", N=KVL, EPS=self.EPS), self.k("dsa_kv_rope", D=R)
        qc = [self.call(qn, q.slice(0, QL), w("q_a_ln").refs).refs for q in qkv]
        kvc = [self.call(kn, q.slice(QL, QL + KVL), w("kv_a_ln").refs).refs for q in qkv]
        kpe = [self.call(kr, q.slice(QL + KVL, QL + KVL + R), c).refs for q, c in zip(qkv, cs)]
        if indexer:
            fn = self.k("dsa_indexer_k_row", D=ID, ROPE=R, EPS=float(C["INDEX_K_EPS"]))
            slots = [self.call(fn, k.slice(0, ID), w("k_norm_w").refs, w("k_norm_b").refs, c) for k, c in zip(wkp, cs)]
        qb = self.fp8_linear(qc, w("q_b_w"), w("q_b_ws"))
        ql = self.absorb([concat([q.slice(h * D, h * D + NOPE) for h in range(NH)]) for q in qb], w("w_uk_t"), NH, NOPE, KVL)
        iqb = self.fp8_linear(qc, w("wq_b_w"), w("wq_b_ws")) if indexer else None
        qr = self.k("dsa_q_rope", D=R)
        qpe = [concat([self.call(qr, q.slice(h * D + NOPE, (h + 1) * D), c).refs for h in range(NH)]) for q, c in zip(qb, cs)]
        for i in range(M):
            logs["latent"].append(concat([kvc[i], kpe[i]]))
        if indexer:
            fn = self.k("dsa_indexer_q", H=IH, D=ID, ROPE=R)
            iq = [self.call(fn, q, c, k.slice(ID, ID + IH)) for q, c, k in zip(iqb, cs, wkp)]
            for s in slots:
                logs["ik"].append(s[0].refs)
                logs["iks"].append(s[1].refs)
            select = self.k("dsa_topk_short_row", TOPK=TOPK)
            for i, n in enumerate(keys):
                self.call(self.k("dsa_mqa_logits", N=n, H=IH, D=ID), iq[i][0].refs, iq[i][1].refs, logs["ik"].cut(n), logs["iks"].cut(n))
                self.call(select, self.call(self.k("const", W=32, BITS=n)).refs)
        G = FA3_SPARSE_GEOMETRY[(R, KVL)]
        batch = [min(n, TOPK) for n in self.launch.keys(keys)]
        by_keys = dict(zip(batch, launch_num_splits(batch, int(self.net.tp.num_sms), TOPK, 0)))
        splits = [by_keys[n] for n in keys]
        att = [self.call(self.k("sparse_mla_split", T=n, NS=ns, NH=NH, DR=R, DV=KVL, BN=G["BN"], DOT=G["DOT"], INV=G["INV"],
                                SM_SCALE=float(C["ATTN_SCALE"])), ql[i], qpe[i], logs["latent"].cut(n)).refs
               for i, (n, ns) in enumerate(zip(keys, splits))]
        return self.fp8_linear(self.absorb(att, w("w_uv"), NH, KVL, VH), w("o_w"), w("o_ws"))

    def mlp(self, L: int, x: list[Refs]) -> list[Refs]:
        w = lambda f: self.w[(L, f)]  # noqa: E731
        return self.fp8_linear(self.silu(self.fp8_linear(x, w("gate_up_w"), w("gate_up_ws"))), w("down_w"), w("down_ws"))

    def moe(self, L: int, x: list[Refs]) -> list[Refs]:
        """The MoE block over the rows x (`vllm_moe_router`, `vllm_moe_fp8`, op-major): the router logits, the shared expert, each row's
        top-k, its block-FP8 quantization, then per (token, slot) pair p = t * TOPK + k the w13 GEMM, the activation, its quantization
        and the routed w2 GEMM; each row's sum of its TOPK down rows, plus the shared expert's."""
        from verity_vllm.engine.profiles.devices import fused_moe_config
        from verity_vllm.program.frontend import targets as TG
        from verity.ml.definitions.registry.moe import router_kernel
        C, M, H, G = self.C, len(x), self.H, self.G
        E, TOPK, I = int(C["NUM_EXPERTS"]), int(C["TOPK"]), int(C["I"])
        w = lambda f: self.w[(L, f)]  # noqa: E731
        logits = self.router(x, w("gate_w"))
        shared = self.fp8_linear(self.silu(self.fp8_linear(x, w("sh_gate_up_w"), w("sh_gate_up_ws"))), w("sh_down_w"), w("sh_down_ws"))
        kernel = router_kernel(E, M, TOPK)
        if kernel is None:
            _refuse(f"the grouped top-k launch at E={E}, top {TOPK} (registry.moe.router_kernel pins no kernel)")
        rt = self.k("moe_router_sigmoid_bias", E=E, TOPK=TOPK, RSF=float(C["ROUTED_SCALING_FACTOR"]), KERNEL=kernel)
        top = [self.call(rt, lg, w("gate_bias").refs) for lg in logits]
        xq = [self.call(self.k("fp8_group_quant_ue8m0", K=H, G=G), r) for r in x]
        cfg = fused_moe_config(E, I, self.net.device, "fp8_w8a8", (G, G), self.launch.tokens(M))
        if cfg is None:
            _refuse(f"no tuned fused-MoE config for E={E}, N={I} on {self.net.device!r} (engine.profiles.devices.FUSED_MOE_CONFIGS)")
        BM, BN, KB, warps = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], min(cfg["BLOCK_SIZE_K"], G), cfg["num_warps"]
        swap = BM < 64 and BN >= 64
        if (why := TG.moe_fp8_tuned_tile_refusal(self.net.tp, BM, BN, KB, warps, swap)):
            _refuse(why)
        st = {"E": E, "G": G, "KB": KB, "SWAP": swap, "DOT": TG.moe_expert_fp8_launch_dot(self.net.tp, BM, BN, warps, swap)}
        up, down = self.k("moe_expert_gemm_dot", K=H, N=2 * I, **st), self.k("moe_expert_gemm_w_dot", K=I, N=H, **st)
        pairs = range(M * TOPK)
        ids = [top[p // TOPK][1].refs.slice(p % TOPK, p % TOPK + 1) for p in pairs]
        u = [self.call(up, xq[p // TOPK][0].refs, xq[p // TOPK][1].refs, ids[p], w("w13").refs, w("w13_ws").refs).refs for p in pairs]
        aq = [self.call(self.k("fp8_group_quant_ue8m0", K=I, G=G), a) for a in self.silu(u)]
        d = [self.call(down, aq[p][0].refs, aq[p][1].refs, ids[p], top[p // TOPK][0].refs.slice(p % TOPK, p % TOPK + 1), w("w2").refs,
                       w("w2_ws").refs).refs for p in pairs]
        msum = self.k("moe_sum", TOPK=TOPK, H=H)
        routed = [self.call(msum, concat(d[t * TOPK:(t + 1) * TOPK])).refs for t in range(M)]
        add = self.k("residual_add", N=H)
        return [self.call(add, s, r).refs for s, r in zip(shared, routed)]
