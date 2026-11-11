"""The DeepSeek sparse-attention MoE decoder (`FAMILY_FACTS[...]["arch"]` "dsa-moe": GLM-5.3's `glm_moe_dsa`, which vLLM serves as
`DeepseekV32ForCausalLM`) for `build.build_program`: one tensor-parallel rank of a block-FP8 checkpoint on cc 9.0 without batch
invariance under vLLM d9105ea80's served defaults, written with the calls the Build's rules make (`rules.vllm_dsa`,
`vllm_bindings.fp8`, `bf16_gemm`, `collectives`, `vllm_moe_fp8`, `vllm_moe_router`) at the selections they read, each at the call's
launch (`common._Launch`: the batch a request's calls run in without batch invariance, else the call's own rows): the captured cuBLAS
and all-reduce routes (`registry.bf16_gemm_check`, `registry.tp_allreduce`), vLLM's tuned fused-MoE tile for the device
(`engine.profiles.devices`) and FA3's split count (`registry.sparse_mla_split`).  The configuration is the generic profile's
(`generic.derive_config` at the rank) and the weights record its map (`generic._mla_weight_field`) in the family's module order.
A request after a history of C positions (`build.build_program`'s C: a prefix-cache hit, or where a Commit window carries it in)
reads each layer's history as Inputs (`Net.prefix_params`): every layer's latent rows and every indexer layer's keys and scales,
which open its layers' logs (`_Model.open_with`) before its first call.
The compiled execution is the target's compilation "breakable": vLLM serves `glm_moe_dsa` without torch.compile
(`VLLM_USE_BREAKABLE_CUDAGRAPH`, config/vllm.py:721-737), the same kernels replayed from CUDA graphs: a uniform decode step's FULL
graph, another step's PIECEWISE graphs (`targets.graph_launch`).  Both pad alike, so every selection a launch reads is at the step's
padded tokens (`targets.graph_tokens`), and no all-reduce route, cuBLAS route or MoE tile reads the mode (d9105ea80's communicators
choose by bytes alone), but the indexer's and the sparse MLA's: graph breaks on the real batch under PIECEWISE graphs, captured at the
padded batch under a FULL graph, its padding rows without keys (`_Launch.keys`); and the logits', computed outside the graphs.

Vocabulary: EmbeddingShard and the embedding's all-reduce; RMSNorm (layer 0's input norm); per layer the block-FP8 linears (FlashInfer
below 32 rows, else DeepGEMM), DsaRMSNorm of q_c and kv_c, the interleaved RoPE of k_pe and of each head's q_pe, the latent absorptions
(BatchedGemm), SparseMlaSplit over the layer's latent history, and on an indexer layer the bf16 weights projection, DsaIndexerKRow,
DsaIndexerQ, DsaMqaLogits and DsaTopKShortRow; a step's prefills (more rows than the decode threshold) of at most INDEX_TOPK keys
by dense MHA instead (`use_dense_mha`, `_Model._dense_prefill`): kv_b_proj, FA3's prefill heads (Attention_v4, AttentionSplit, and
with the lse AttentionHeadLse and AttentionHeadSplitLse where the step has context) and MergeAttnStates, while longer prefills take
the decode kernels; the all-reduce + residual + RMSNorm (FlashInfer's fused MNNVL kernel within its token
budget, else the all-reduce and RMSNormFusedCuda); the SiLU MLP below FIRST_DENSE, and from it on the MoE block: the f32 router GEMM,
the shared expert, MoeRouterSigmoidBias, Fp8GroupQuantUe8m0, the block-FP8 expert GEMMs per (token, slot) pair, MoeSum and the
shared expert's add; then the last row's logits, their all-gather and the token select (greedy, or the seeded temperature + top-p
select: `common.select_token`).  A row of more than INDEX_TOPK keys selects its top-k as the `vllm_dsa` rule does, in the circuit,
at the row's class (`vllm_dsa.long_row_keys`: its keys rounded up to a multiple of INDEX_TOPK, the logits past its own `PAD_SCORE`
and their history rows zero): DsaTopKMask marks the logits' top-TOPK set, DsaButterflySettings sets the concentrator for it, and
both kinds of layer read the TOPK rows DsaButterflyRows routes at those settings, in ascending index order (the row
`VERITY_DSA_TOPK=canonical` serves); below its class through DsaButterflyRowsPad, the history and its pad two arguments, so a
row's history is the row before's plus one part in the descriptor.  In class G's batched form a call's rows past INDEX_TOPK go in
blocks (`vllm_dsa.long_row_blocks`), each row read at its block's keys and selecting among its own (DsaTopKMaskCounted), each
op over a block's rows in turn, and a token's latent row is one DsaLatentRow, as the rule writes them; under table reads
(`Registered.dsa_table_reads`) a long row's selection is DsaTopKIndex's positions and both kinds of layer read their rows at
them by DsaHistoryReads (`program.dsa_table`).  A fact outside it raises
`Unexpressible`, φ's when the decoder is made and a launch's (a step whose prefills the batch does not fix, more context than one
chunk is sure to hold, a GEMM, all-reduce or MoE tile no record covers) when the launch is written.
"""

from __future__ import annotations

import re
from typing import Mapping, NamedTuple, Sequence

from verity.circuits.defs import Builder
from verity.circuits.refs import Coll, Concat, PartLog, Refs, concat, strided_view
from verity.circuits.types import Type, Value

from .common import (Unexpressible, _arr, _arr_shape, _Launch, _Peers, _ptypes, _Stochastic, graph_capture_sizes, kind,
                     select_token, served_class)
from .phi import Phi


def _refuse(fact: str):
    raise Unexpressible(f"vocabulary cannot express: {fact}")


def run_field(layer: int, pos: int) -> str:
    """The `run` field `<layer>/topk_switches[t=P]` of layer `layer`'s Waksman settings for the row at position `pos`, as the
    Commit's run values (`commit.committer.run_values`) and the replay (`check.replay`) name it.  No Program declares a `run` input:
    a row past INDEX_TOPK computes its selection (`_Model.attention`)."""
    return f"model.layers.{layer}.self_attn.attn/topk_switches[t={pos}]"


_RUN_FIELD = re.compile(r"model\.layers\.(\d+)\.self_attn\.attn/topk_switches\[t=(\d+)\]")


def run_bits(layout: dict, orders: Mapping[tuple[int, int], Sequence[int]], topk: int, pos0: int = 0) -> list[int]:
    """The leaves of a `run` input (one bit each, in its record's field order) from what the served run did: `layout` is its
    `RunInputs.layout`, and `orders[(layer, pos)]` the `topk` (INDEX_TOPK) key indices that layer `layer`'s indexer
    wrote for the row at position `pos`, in the order it wrote them (its `topk_indices_buffer` row), each index one of the row's
    own keys, the wrapper's rows pos0..pos (a request's pos0 is 0). Each field is the Waksman settings that route those keys to
    that order (`dsa_topk.waksman_settings`)."""
    from verity.definitions.registry.dsa_topk import waksman_settings
    bits: list[int] = []
    for f in layout["fields"]:
        m = _RUN_FIELD.fullmatch(f["name"])
        if m is None or f["offset"] != len(bits):
            raise ValueError(f"run field {f['name']!r} at leaf {f['offset']} is not a top-k order's (`run_field`) at leaf {len(bits)}")
        key = (int(m.group(1)), int(m.group(2)))
        if key not in orders:
            raise KeyError(f"no served top-k order for layer {key[0]}'s row at position {key[1]} (run field {f['name']!r})")
        n, order = key[1] + 1 - int(pos0), [int(x) - int(pos0) for x in orders[key]]
        if len(order) != int(topk):
            raise ValueError(f"run field {f['name']!r}: an order of {len(order)} slots, not INDEX_TOPK {topk}")
        sw = waksman_settings(order, n)
        if len(sw) != f["leaves"]:
            raise ValueError(f"run field {f['name']!r}: {len(sw)} settings for {f['leaves']} leaves ({n} keys)")
        bits += sw
    return bits


def _at(t: Type, path: tuple) -> Type:
    for p in path:
        t = t.elem if isinstance(p, int) else dict(t.fields)[p]
    return t


class Net:
    """One rank's decoder: the configuration, the weights record (`leaves`), the peers a pass reads (`peer_shapes`) and the target's
    selections the forward (`model`) reads."""

    DOMAIN = ((8,), ("eager", "compiled"), ("float8_e4m3fn",))

    def __init__(self, phi: Phi, M: int, history: int) -> None:
        from verity_vllm.engine.profiles import family_facts as FF
        from verity_vllm.engine.profiles.engine_classes import DSA_TOPK_ENV
        from verity_vllm.observe.fold.profiles.generic import _MLA_SCALES, _mla_weight_field, derive_config
        from verity_vllm.program import model as analytic
        from verity_vllm.program.frontend import targets as TG
        from verity_vllm.program.frontend.rules.vllm_dsa import FA3_SPARSE_GEOMETRY
        from verity.definitions.registry import dsa
        from verity.definitions.registry.tp_allreduce import settings_of
        hf = phi.hf_config
        self.world, self.rank = int(phi.tp["world"]), int(phi.tp.get("rank", 0))
        try:
            C, _ = derive_config(hf, repo=phi.model.get("repo"), revision=phi.model.get("revision"), world=self.world)
        except (KeyError, ValueError) as e:
            raise Unexpressible(f"config.json: {e}") from None
        self.C, self.tp, self.device = C, phi.target_profile(), phi.target.get("device_name")
        self.indexers = frozenset(C.get("INDEXER_LAYERS") or ())
        facts = []
        if phi.execution == "compiled" and self.tp.compilation != "breakable":
            facts.append(f"a compiled execution by {self.tp.compilation!r} (vLLM serves glm_moe_dsa's eager kernels from breakable CUDA "
                         f"graphs, the target's compilation 'breakable')")
        self.capture_sizes, self.max_num_seqs = graph_capture_sizes(phi), (phi.engine or {}).get("max_num_seqs")
        if self.capture_sizes and type(self.max_num_seqs) is not int:
            facts.append(f"breakable CUDA graphs at engine arg max_num_seqs {self.max_num_seqs!r}, not a stated count (a uniform decode "
                         f"step replays a FULL graph up to it)")
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
        env = phi.engine.get("env")
        self.long_rows = None if env is None or all(env.get(k) == v for k, v in DSA_TOPK_ENV.items()) else (
            f"a row past INDEX_TOPK ({int(C['INDEX_TOPK'])}) keys under an engine whose environment sets "
            f"{', '.join(f'{k}={env.get(k)!r}' for k in DSA_TOPK_ENV)}, not {DSA_TOPK_ENV} (the kernels' own order past INDEX_TOPK, "
            f"not the ascending row DsaButterflyRows reads; serve it under the DSA top-k plugin's canonical row)")
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

    def past_topk(self, keys: int) -> bool:
        """Whether a row of up to `keys` keys selects its top-k from its logits: a row past INDEX_TOPK."""
        return int(keys) > int(self.C["INDEX_TOPK"])

    def peer_shapes(self, r: int) -> list[tuple[int, int]]:
        """A pass of r rows: the embedding's all-reduce, each layer's attention and MLP all-reduces, the sampled row's logits gather."""
        H = int(self.C["H"])
        return [(r, H)] * (1 + 2 * self.NL) + [(1, int(self.C["V"]))]

    def edge_groups(self, keys: int) -> tuple[str, ...]:
        """The history groups of a request whose rows read up to `keys` keys that some row of it reads, which a Commit window's edge
        carries (`segment.edge_of`): the latent rows, and the indexer's keys and scales where a row reads past INDEX_TOPK (below, its
        top-k selection keeps every key and reads no logit: `past_topk`)."""
        return ("prefix", *(("prefix_ik", "prefix_iks") if self.past_topk(keys) else ()))

    def prefix_params(self, C: int) -> list[tuple[str, Type]]:
        """The Inputs a request's history of its first C positions opens with (`_Model.open_with`), as `vllm_dsa` keeps it: every
        layer's latent rows `prefix` (bf16 [NL, C, KV_LORA + QK_ROPE]: the normed kv_c, then the roped k_pe), and the indexer
        layers', in layer order, index-K slots `prefix_ik` (e4m3 [NI, C, INDEX_HEAD_DIM]) and their scales `prefix_iks` (f32 [NI, C])."""
        NI, row = len(self.indexers), int(self.C["KV_LORA"]) + int(self.C["QK_ROPE"])
        return [("prefix", _arr((self.NL, C, row), 16)), ("prefix_ik", _arr((NI, C, int(self.C["INDEX_HEAD_DIM"])), 8)),
                ("prefix_iks", _arr((NI, C), 32))]

    def model(self, B: Builder, W: Coll, peers: _Peers) -> "_Model":
        return _Model(self, B, W, peers)

    def dense_plan(self, k, M: int, pos0: int, context: int, dense: list[tuple[int, int]]) -> "DensePlan":
        """The selections of a dense-MHA prefill of M rows at pos0.. after `context` history rows (`_Model._dense_prefill`), each
        kind `k(role, **statics)`'s, or the refusal."""
        import numpy as np
        from verity.definitions.numerics import fa3_model
        from verity.definitions.registry import fa3_split
        from verity.definitions.registry.merge_attn_states import NEG_INF
        from verity_vllm.program.frontend import targets as TG
        C, tp = self.C, self.tp
        R, NOPE, VH, NH = (int(C[k]) for k in ("QK_ROPE", "QK_NOPE", "V_HEAD", "NH"))
        D = NOPE + R
        if VH != D:
            _refuse(f"a dense-MHA prefill at v head {VH} vs qk head {D}: Attention_v4 states one head dim for q, k and v")
        if np.float32(C["ATTN_SCALE"]) != np.float32(D ** -0.5):
            _refuse(f"a dense-MHA prefill at scale {C['ATTN_SCALE']!r}, not f32(D^-0.5) at D={D}, the word Attention_v4 folds in")
        if not TG.fa3_check_inf(tp):
            _refuse(f"a dense-MHA prefill on FA{tp.effective_flash_attn_version()} without FA3's check-inf-per-iteration construction")
        if context != pos0:
            _refuse(f"a dense-MHA prefill at position {pos0} over a layer history of {context} rows")
        extends = [s for s in dense if s[1] > 0]
        longest, held = max(n + p for n, p in dense), sum(p for _, p in extends)
        if held > min(8 * longest, 65536):
            _refuse(f"a dense-MHA step's context of {held} rows, past the {min(8 * longest, 65536)} rows of the smallest workspace "
                    f"`plan_mla_context_chunks` may cut it at (several context chunks)")
        max_q, rows, me = max(n for n, _ in dense), [n for n, _ in dense], dense.index((M, pos0))
        try:
            geom = TG.attention_geometry(tp, D, max_q, NH, NH)
            BM = TG.fa3_tile_m(D, max_q, 1)
        except TG.UnregisteredTarget as e:
            _refuse(str(e))
        BN, st = geom["BN"], {"BN": geom["BN"], "DOT": geom["DOT"], "INV": geom["INV"]}

        def splits(keys=None, at=me, q=rows) -> int:
            if tp.batch_invariant:
                return 1
            try:
                static, dyn = fa3_model.split_plan(q, NH, NH, D, num_sms=int(tp.num_sms), Dv=D, seqlens_k=keys)
                if dyn[at] > 1:
                    fa3_split.check_combine_bound(static)
            except ValueError as e:
                _refuse(f"FA3's split count of a dense-MHA launch of {q} query rows over {keys or q} keys: {e}")
            return dyn[at]

        def row(role: str, i: int, **kw):
            split = {} if ns == 1 else {"PER": fa3_split.split_per(M, i, BM, BN, ns)}
            return k(role, T=i + 1, **kw, D=D, MASKED_FROM=TG.fa3_masked_from(tp, i + 1, NH, NH, D, max_q, i), **st, **split)

        ns = splits()
        if not extends:
            return DensePlan(sum(rows), held, [row("attention_dot", i, NH=1, KVH=1) for i in range(M)])
        cfn = empty = None
        if context:
            nc = splits([p for _, p in extends], extends.index((M, pos0)), [n for n, _ in extends])
            nb = -(-context // BN)
            cfn = k("attention_head_lse", T=context, D=D, MASKED_FROM=nb - 1, **st, **({} if nc == 1 else {"PER": -(-nb // nc)}))
        else:
            empty = (k("const", W=16, BITS=0), k("const", W=32, BITS=NEG_INF))
        merge = k("merge_attn_states", NH=NH, D=D)
        return DensePlan(sum(rows), held, [row("attention_head_lse", i) for i in range(M)], cfn, merge, empty)


class DensePlan(NamedTuple):
    """A dense-MHA prefill's selections (`Net.dense_plan`): kv_b_proj's rows (the step's prefills'), the context's rows `held`, each
    row's head kind, and where some prefill of the step extends, MergeAttnStates, with the context's head kind where this request
    has context, else the zero output and lse -inf consts."""
    rows: int
    held: int
    heads: list
    context: object = None
    merge: object = None
    empty: tuple | None = None


class _Model:
    """The forward over one call's rows, emitted into `B`, each layer's latent and index-K histories carried across calls as
    `vllm_dsa` keeps them (one `PartLog` per history); every token-count-dependent selection reads the call's launch (`launch`).
    `topk` is the last indexer layer's selection by position, which an IndexShare layer reads.  `batched`: class G's batched form
    (`build.build_program`'s `batched`), whose latent rows are one Definition each and whose rows past INDEX_TOPK go in blocks
    (`vllm_dsa.long_row_blocks`), each op over a block's rows in turn, so that `prune.batched_form` batches them.  `table_reads`:
    in that form, those rows by table reads (`build.build_program`'s `table_reads`): `topk` holds a long row's `DsaTopKIndex`
    positions, and each layer reads its rows at them (`DsaHistoryReads`) in place of the rows network."""

    reads_launch = True
    batched = False
    table_reads = False

    def __init__(self, net: Net, B: Builder, W: Coll, peers: _Peers) -> None:
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

    def open_with(self, prefix: list[Coll], C: int) -> None:
        """Every layer's history opens with its C rows of `prefix` (`Net.prefix_params`: the latent rows, then the indexer layers'
        index-K slots and scales), read and never computed: the request's own rows follow at positions C.."""
        latent, ik, iks = prefix
        for L in range(self.net.NL):
            logs = self.logs.setdefault(L, {"latent": PartLog(), "ik": PartLog(), "iks": PartLog()})
            if len(logs["latent"]):
                raise AssertionError(f"layer {L}'s history opens once, before its first call")
            for j in range(C):
                logs["latent"].append(latent[L][j].refs)
        for n, L in enumerate(sorted(self.net.indexers)):
            for j in range(C):
                self.logs[L]["ik"].append(ik[n][j].refs)
                self.logs[L]["iks"].append(iks[n][j].refs)

    def edge_rows(self, groups: tuple[str, ...]) -> list[Coll]:
        """Every layer's rows so far of each history group in `groups` (`Net.edge_groups`), its history's and its own, as
        `Net.prefix_params` lays out a history of them: wires and no gates."""
        E, net = len(self.logs[0]["latent"]), self.net
        logs = {"prefix": [self.logs[L]["latent"] for L in range(net.NL)],
                **{g: [self.logs[L][h] for L in sorted(net.indexers)] for g, h in (("prefix_ik", "ik"), ("prefix_iks", "iks"))}}
        if any(len(log) != E for g in groups for log in logs[g]):
            raise AssertionError(f"every layer's history holds the {E} rows of layer 0's")
        return [Coll(t, concat([log.cut(E) for log in logs[g]])) for g, t in net.prefix_params(E) if g in groups]

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
        return select_token(self.call, self.k, gathered.refs, V * W, pos, policy)

    # -- collectives (`collectives.AllReduceRule`, `FusedAllReduceRmsNormRule`)

    def all_reduce(self, rows: list[Refs]) -> list[Refs]:
        """One all-reduce over the rows on the route vLLM takes at the launch's tokens (`tp_allreduce.served_route`), the captured one
        -> each row's sum."""
        from verity_vllm.program.frontend.rules.vllm_bindings.collectives import _ROUTE_KINDS
        from verity.definitions.registry import tp_allreduce as AR
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
        from verity.definitions.registry import tp_allreduce as AR
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
        from verity_vllm.program.frontend.rules.vllm_bindings.bf16_gemm import _f32out_phase
        from verity.definitions.registry.bf16_gemm_check import default_cublas_route
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
        from verity.definitions.registry.bf16_gemm_check import DEFAULT_LL_BF16_FMA, dispatch_sm90
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
        from verity_vllm.program.frontend.rules.vllm_dsa import (DECODE_THRESHOLD, DECODE_THRESHOLD_DEFAULT, FA3_SPARSE_GEOMETRY, PAD_SCORE,
                                                                 long_row_blocks, long_row_keys, pad_words)
        from verity.definitions.registry.sparse_mla_split import launch_num_splits
        C, M = self.C, len(x)
        QL, KVL, R, NOPE, VH, NH = (int(C[k]) for k in ("Q_LORA", "KV_LORA", "QK_ROPE", "QK_NOPE", "V_HEAD", "NH"))
        IH, ID, TOPK = int(C["INDEX_HEADS"]), int(C["INDEX_HEAD_DIM"]), int(C["INDEX_TOPK"])
        D, indexer, E = NOPE + R, L in self.net.indexers, self.launch.entry(M)
        w = lambda f: self.w[(L, f)]  # noqa: E731
        prefills = self.launch.prefills((E, pos0), DECODE_THRESHOLD.get(NH, DECODE_THRESHOLD_DEFAULT))
        dense = prefills if prefills and max(n + p for n, p in prefills) <= TOPK else []
        logs = self.logs.setdefault(L, {"latent": PartLog(), "ik": PartLog(), "iks": PartLog()})
        keys = [len(logs["latent"]) + i + 1 for i in range(M)]
        cs = [self.w["cos_sin"].refs.slice((pos0 + i) * R, (pos0 + i + 1) * R) for i in range(M)]
        qkv = self.fp8_linear(x, w("qkv_a_w"), w("qkv_a_ws"))
        wkp = self.cublas(x, w("wk_w")) if indexer else None
        qn = self.k("dsa_rmsnorm", N=QL, EPS=self.EPS)
        qc = [self.call(qn, q.slice(0, QL), w("q_a_ln").refs).refs for q in qkv]
        if self.batched:
            lr = self.k("dsa_latent_row", KVL=KVL, R=R, EPS=self.EPS)
            lat = [self.call(lr, q.slice(QL, QL + KVL + R), w("kv_a_ln").refs, c).refs for q, c in zip(qkv, cs)]
            kvc, kpe = [r.slice(0, KVL) for r in lat], [r.slice(KVL, KVL + R) for r in lat]
        else:
            kn, kr = self.k("dsa_rmsnorm", N=KVL, EPS=self.EPS), self.k("dsa_kv_rope", D=R)
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
        context = len(logs["latent"])
        for i in range(M):
            logs["latent"].append(concat([kvc[i], kpe[i]]))
        own = (E, pos0) in dense
        mqa = self.launch.keys([] if own else keys, without=dense)
        if indexer:
            fn = self.k("dsa_indexer_q", H=IH, D=ID, ROPE=R)
            iq = [self.call(fn, q, c, k.slice(ID, ID + IH)) for q, c, k in zip(iqb, cs, wkp)]
            for s in slots:
                logs["ik"].append(s[0].refs)
                logs["iks"].append(s[1].refs)
            select = self.k("dsa_topk_short_row", TOPK=TOPK)
            self.topk = {}
            blocks = long_row_blocks(keys, TOPK) if self.batched and mqa else []
            for i, n in enumerate(keys if mqa else ()):
                if blocks and n > TOPK:
                    break
                logits = self.call(self.k("dsa_mqa_logits", N=n, H=IH, D=ID), iq[i][0].refs, iq[i][1].refs, logs["ik"].cut(n),
                                   logs["iks"].cut(n))
                if n <= TOPK:
                    self.call(select, self.call(self.k("const", W=32, BITS=n)).refs)
                    continue
                if self.net.long_rows:
                    _refuse(self.net.long_rows)
                N = long_row_keys(n, TOPK)
                scores = concat([logits.refs, self.pad(32, PAD_SCORE, pad_words(TOPK), N - n)]) if N > n else logits.refs
                mask = self.call(self.k("dsa_topk_mask", N=N, TOPK=TOPK), scores)
                self.topk[pos0 + i] = self.call(self.k("dsa_butterfly_settings", N=N, TOPK=TOPK), mask.refs)
            for Nb, rows in blocks:
                if self.net.long_rows:
                    _refuse(self.net.long_rows)
                N, lg = long_row_keys(Nb, TOPK), self.k("dsa_mqa_logits", N=Nb, H=IH, D=ID)
                ik, iks = logs["ik"].cut(Nb), logs["iks"].cut(Nb)
                logits = [self.call(lg, iq[i][0].refs, iq[i][1].refs, ik, iks).refs for i in rows]
                counts = [self.pad(32, keys[i], 1, 1) for i in rows]
                tail = [self.pad(32, PAD_SCORE, pad_words(TOPK), N - Nb)] if N > Nb else []
                fm = self.k("dsa_topk_mask_counted", N=N, TOPK=TOPK)
                fs = self.k("dsa_topk_index" if self.table_reads else "dsa_butterfly_settings", N=N, TOPK=TOPK)
                masks = [self.call(fm, concat([s, *tail]) if tail else s, c).refs for s, c in zip(logits, counts)]
                for i, m in zip(rows, masks):
                    self.topk[pos0 + i] = self.call(fs, m)
        if own:
            return self.fp8_linear(self._dense_prefill(L, qb, qpe, kvc, kpe, context, pos0, dense, E), w("o_w"), w("o_ws"))
        G = FA3_SPARSE_GEOMETRY[(R, KVL)]
        batch = [min(n, TOPK) for n in mqa]
        by_keys = dict(zip(batch, launch_num_splits(batch, int(self.net.tp.num_sms), TOPK, 0)))
        rows_of: dict[int, Refs] = {}
        for Nb, rows in (long_row_blocks(keys, TOPK) if self.batched else ()):
            N, hist = long_row_keys(Nb, TOPK), logs["latent"].cut(Nb)
            fr, rest = ((self.k("dsa_history_reads", V=Nb, TOPK=TOPK, H=KVL + R), ()) if self.table_reads else
                        (self.k("dsa_butterfly_rows_pad", T=Nb, N=N, TOPK=TOPK, W=KVL + R),
                         (self.pad(16, 0, pad_words(TOPK, KVL + R), (N - Nb) * (KVL + R)),)) if N > Nb else
                        (self.k("dsa_butterfly_rows", N=N, TOPK=TOPK, W=KVL + R), ()))
            for i in rows:
                rows_of[i] = self.call(fr, self.topk[pos0 + i].refs, hist, *rest).refs
        att = []
        for i, n in enumerate(keys):
            kv = rows_of[i] if i in rows_of else logs["latent"].cut(n)
            if n > TOPK and i not in rows_of:
                N = long_row_keys(n, TOPK)
                if N > n:
                    kv = self.call(self.k("dsa_butterfly_rows_pad", T=n, N=N, TOPK=TOPK, W=KVL + R), self.topk[pos0 + i].refs, kv,
                                   self.pad(16, 0, pad_words(TOPK, KVL + R), (N - n) * (KVL + R))).refs
                else:
                    kv = self.call(self.k("dsa_butterfly_rows", N=N, TOPK=TOPK, W=KVL + R), self.topk[pos0 + i].refs, kv).refs
            att.append(self.call(self.k("sparse_mla_split", T=min(n, TOPK), NS=by_keys[min(n, TOPK)], NH=NH, DR=R, DV=KVL, BN=G["BN"],
                                        DOT=G["DOT"], INV=G["INV"], SM_SCALE=float(C["ATTN_SCALE"])), ql[i], qpe[i], kv).refs)
        return self.fp8_linear(self.absorb(att, w("w_uv"), NH, KVL, VH), w("o_w"), w("o_ws"))

    def _dense_prefill(self, L: int, qb: list[Refs], qpe: list[Refs], kvc: list[Refs], kpe: list[Refs], context: int, pos0: int,
                       dense: list[tuple[int, int]], entry: int) -> list[Refs]:
        """The rows' attention by dense MHA (`MLACommonBaseImpl.forward_mha`, `FlashAttnPrefillBackend`), the step's prefills `dense`
        (rows, start) all of at most TOPK keys (`use_dense_mha`), this request's `context` history rows before them -> rows [NH * VH].
        `entry`: its sequence's rows in the launch, of which the call computes the first.

        kv_b_proj over the prefills' latent rows (eager, at their row count), then per row i and head h FA3's causal varlen self-attention
        over the request's own rows 0..i (`run_prefill_new_tokens`), bound per head as `DsaDenseMhaRule` binds it, at the split count
        `fa3_model.split_plan` gives the request among the prefills.  Where some prefill of the step has context (`has_context`), each
        head returns its lse (`AttentionHeadLse_v1`, split `AttentionHeadSplitLse_v1`), the request's context partial is FA3's
        non-causal attention over its gathered history (`_compute_prefill_context`: kv_b_proj over the context rows of the chunk, the
        prefills with context, `run_prefill_context_chunk`) or, without context, the zero output with lse -inf vLLM neutralizes it to,
        and `MergeAttnStates_v1` merges the two per token.  More context than one chunk is sure to hold is refused."""
        C, M = self.C, len(qb)
        KVL, R, NOPE, VH, NH = (int(C[k]) for k in ("KV_LORA", "QK_ROPE", "QK_NOPE", "V_HEAD", "NH"))
        D, KW = NOPE + R, NOPE + VH
        plan = self.net.dense_plan(self.k, entry, pos0, context, dense)
        plan = plan._replace(heads=plan.heads[:M])

        def w(f: str):
            return self.w[(L, f)]

        def head_logs(rows: list[Refs], pe: list[Refs]) -> tuple[list[tuple[PartLog, PartLog]], list[int]]:
            """Per head the key rows [k_nope | k_pe] and the value rows, kv_b_proj's `rows` split per head ([k_nope | v] each), and
            the key log's part count after each row."""
            logs, ends = [(PartLog(), PartLog()) for _ in range(NH)], [0]
            for r, p in zip(rows, pe):
                for h, (klog, vlog) in enumerate(logs):
                    for part in (r.slice(h * KW, h * KW + NOPE), p):
                        for x in (part.parts if isinstance(part, Concat) else (part,)):
                            klog.append(x)
                    vlog.append(r.slice(h * KW + NOPE, (h + 1) * KW))
                ends.append(len(logs[0][0]))
            return logs, ends

        def heads(fn, q: list[Refs], T: int, logs) -> list:
            (pairs, ends) = logs
            return [self.call(fn, q[h], klog.cut(ends[T]), vlog.cut(T)) for h, (klog, vlog) in enumerate(pairs)]

        kvb = self.fp8_linear(kvc, w("kv_b_w"), w("kv_b_ws"), rows=plan.rows)
        sfx = head_logs(kvb, kpe)
        qs = [[concat([q.slice(h * D, h * D + NOPE), pe.slice(h * R, (h + 1) * R)]) for h in range(NH)] for q, pe in zip(qb, qpe)]
        if plan.merge is None:
            return [concat([o.refs for o in heads(fn, qs[i], i + 1, sfx)]) for i, fn in enumerate(plan.heads)]
        if context:
            hist = self.logs[L]["latent"].parts[:context]
            ckvb = self.fp8_linear([r.slice(0, KVL) for r in hist], w("kv_b_w"), w("kv_b_ws"), rows=plan.held)
            pre = head_logs(ckvb, [r.slice(KVL, KVL + R) for r in hist])
        else:
            zero, ninf = (self.call(fn).refs for fn in plan.empty)
            empty = (concat([zero] * (NH * D)), concat([ninf] * NH))
        out = []
        for i, fn in enumerate(plan.heads):
            s = heads(fn, qs[i], i + 1, sfx)
            p = heads(plan.context, qs[i], context, pre) if context else None
            po, pl = (concat([o[0].refs for o in p]), concat([o[1].refs for o in p])) if p else empty
            out.append(self.call(plan.merge, po, pl, concat([o[0].refs for o in s]), concat([o[1].refs for o in s])).refs)
        return out

    def mlp(self, L: int, x: list[Refs]) -> list[Refs]:
        w = lambda f: self.w[(L, f)]  # noqa: E731
        return self.fp8_linear(self.silu(self.fp8_linear(x, w("gate_up_w"), w("gate_up_ws"))), w("down_w"), w("down_ws"))

    def moe(self, L: int, x: list[Refs]) -> list[Refs]:
        """The MoE block over the rows x (`vllm_moe_router`, `vllm_moe_fp8`, op-major): the router logits, the shared expert, each row's
        top-k, its block-FP8 quantization, then per (token, slot) pair p = t * TOPK + k the w13 GEMM, the activation, its quantization
        and the routed w2 GEMM; each row's sum of its TOPK down rows, plus the shared expert's."""
        from verity_vllm.engine.profiles.devices import fused_moe_config
        from verity_vllm.program.frontend import targets as TG
        from verity.definitions.registry.moe import router_kernel
        C, M, H, G = self.C, len(x), self.H, self.G
        E, TOPK, I = int(C["NUM_EXPERTS"]), int(C["TOPK"]), int(C["I"])
        w = lambda f: self.w[(L, f)]  # noqa: E731
        logits = self.router(x, w("gate_w"))
        shared = self.fp8_linear(self.silu(self.fp8_linear(x, w("sh_gate_up_w"), w("sh_gate_up_ws"))), w("sh_down_w"), w("sh_down_ws"))
        kernel = router_kernel(E, self.launch.tokens(M), TOPK)
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
