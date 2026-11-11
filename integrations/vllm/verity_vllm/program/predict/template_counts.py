"""n_t(c): each template's instances in a request Program of the sparse-attention decoder (`dsa`), by descriptor id, as a template
partition counts them (`partition_object.TemplatePartition`, a batch counting its members), from the request's unit key
(`steps.RequestUnit.key`: LP, T, the policy, its history C, its prefill chunks, its launch, its edge and its length rule) and the
registered φ, without building the Program.

Each forward pass is one launch, and its (Definition, count) list is `step_needs.StepNeeds`' selections at the request's own rows:
at the launch's tokens, key counts and prefills, as `dsa._Model` reads them. A unit's n_t sums its passes' lists, each kept to what
the registered compiler's form keeps (`prune.dead_free`):

- the last layer computes what the samples read: where FlashInfer's fused all-reduce + norm runs per row, the sampled row's queries,
  attention, MLP and norms; where one all-reduce reads every row (a launch past `fused_max_tokens`), every row's but the final
  norm's; nothing of a pass that samples nothing. Every layer's key rows stay, the request's later rows, its edge and its samples
  reading them;
- a row within INDEX_TOPK keys reads no indexer logit (its top-k keeps every key), so its logits, query and top-k are dead, and an
  index-K slot lives only where a row past INDEX_TOPK of its layer reads it, or the edge does: its bf16 projection is then
  narrowed to the K columns (`prune.narrowed`) unless that row's query reads its weights too;
- a dense-MHA prefill's queries skip the latent absorption, which is dead;
- each Definition `prune.dead_free_versions` names is its dead-free version;
- the form is batched (`prune.batched_form`, class G's): each token's latent row is one `DsaLatentRow`, and a row past INDEX_TOPK
  is read at its block's keys (`vllm_dsa.long_row_blocks`), its logits there, its own key count a one-word pad, and its counted
  mask, settings and rows at the block's class.

Consts count, a pad block as its words. A const called alone (the select's settings under a seeded policy, a dense prefill's empty
context) is a root primitive, one instance here, which `TemplatePartition` refuses. `tests/program/predict/test_template_counts.py`
holds n_t to the built Programs' root nodes, and to their template partitions where the partition applies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity_vllm.program.prune import COLUMNS, dead_free_versions

from .common import Unexpressible, _Launch, in_launch, kind, launch_batch, request_passes
from .step_needs import Need, StepNeeds

__all__ = ["TemplateCounts"]


@dataclass(frozen=True)
class _Pass:
    """A forward pass of the request's m rows at positions p.. in its launch: the launch's tokens `M`, its sequences `S` where a
    sample follows (else None), whether its all-reduces are FlashInfer's fused kernel, the rows' key counts, the step's dense-MHA
    prefills (`own`: the request's among them), the attention launch's key counts `mqa`, its split count by kept keys, and each row
    past INDEX_TOPK's block keys (`vllm_dsa.long_row_blocks`)."""
    m: int
    p: int
    S: int | None
    M: int
    fused: bool
    keys: list[int]
    dense: list[tuple[int, int]]
    own: bool
    mqa: list[int]
    splits: dict[int, int]
    block: dict[int, int]


class TemplateCounts(StepNeeds):
    """`StepNeeds`' decoder for φ (one tensor-parallel rank), each kind named by its dead-free version's descriptor id (`kind`),
    and n_t of a request unit's Program (`unit`, `request`)."""

    def __init__(self, phi) -> None:
        super().__init__(phi)
        self._fns: dict[str, Any] = {}

    def kind(self, role: str, **statics: object) -> Need:
        """The vocabulary's Definition for `role` at `statics` as a template partition names it, or `Unexpressible`."""
        key = (role, repr(sorted(statics.items(), key=lambda kv: kv[0])))
        if key not in self._memo:
            self._memo[key] = self._named(role, kind(role, **statics))
        return self._memo[key]

    def _named(self, role: str, f) -> Need:
        if (v := dead_free_versions().get(getattr(f, "base_id", None))) is not None:
            f = bind(v, **f.bindings)
        self._fns.setdefault(tid := descriptor_id(f), f)
        return Need(role, tid)

    def _narrowed(self, nd: Need, N: int) -> Need:
        """`nd` over its first N output columns (`prune.narrowed`), where it is a `COLUMNS` linear."""
        f = None if nd.gap else self._fns[nd.id]
        if f is None or getattr(f, "base_id", None) not in COLUMNS:
            return nd
        return self._named(nd.route, bind(f.defn, **{**f.bindings, "N": N}))

    def unit(self, u) -> Counter[str]:
        """n_t of a `steps.RequestUnit`'s Program (`steps.request_program`), from its key's fields."""
        return self.request(u.LP, u.T, u.policy, C=u.C, chunks=u.chunks, batch=u.batch(), edge=u.edge, stop=u.stop)

    def request(self, LP: int, T: int, policy: Mapping[str, Any] | None = None, *, C: int = 0, chunks: Sequence[int] = (),
                batch: list | None = None, edge: Sequence[str] = (), stop: Mapping[str, Any] | None = None) -> Counter[str]:
        """n_t of the request Program `build.build_program` writes for (LP, T) under `policy` (φ's `sampling`, greedy without one)
        after C rows of history, in its prefill `chunks` at its launch `batch` (else φ's workload's, `common.launch_batch`), with its
        `edge` groups, its seed rule where `policy` is stochastic (at its ``rid``) and length rule `stop`, in the registered
        compiler's form: descriptor id -> instances."""
        from verity_vllm.program.frontend import seed_rule as SEED
        from verity_vllm.program.frontend import stop_rule as SR
        net, NL, TOPK = self.net, int(self.C["NL"]), self.index_topk
        _, chunks, rows, sampled = request_passes("request", LP, T, 1, C, tuple(chunks or ()), False, bool(edge))
        batch = launch_batch(self.phi, LP, T) if batch is None else batch
        model = SimpleNamespace(reads_launch=True, launch=_Launch(capture_sizes=net.capture_sizes, max_num_seqs=net.max_num_seqs or 0))
        in_launch(model, batch, net.model_class, chunks, T)
        passes, E = [], int(C)
        for k, (m, smp) in enumerate(zip(rows, sampled)):
            if batch is not None:
                model.launch.step = k
            passes.append(self._pass(model.launch, m, E, smp))
            E += m
        long = [[i for i in range(P.m) if P.mqa and P.keys[i] > TOPK] for P in passes]
        if net.long_rows and any(long):
            raise Unexpressible(f"vocabulary cannot express: {net.long_rows}")
        last = [[] if P.S is None else [P.m - 1] if P.fused else list(range(P.m)) for P in passes]
        every = E if set(edge) - {"prefix"} else 0
        bound = max([every] + [P.block[i] for P, ls in zip(passes, long) for i in ls])
        bound_last = max([every] + [P.block[i] for P, ls, X in zip(passes, long, last) for i in ls if i in X])
        first = int(self.C["FIRST_DENSE"])
        body = Counter((L in net.indexers, L >= first) for L in range(NL - 1))
        rep = {c: min(L for L in range(NL) if (L in net.indexers, L >= first) == c) for c in body}
        out: Counter[str] = Counter()
        pads: set[tuple[int, int, int]] = set()

        def add(pairs: list[tuple[Need, int]], times: int = 1) -> None:
            for nd, c in pairs:
                if nd.gap:
                    raise Unexpressible(f"vocabulary cannot express: {nd.id}: {nd.gap}")
                if c * times:
                    out[nd.id] += c * times

        V, events = int(self.C["V"]), 0
        for P, X in zip(passes, last):
            add([(self.kind("embedding_shard", VS=V, H=self.H, START=net.rank * V), P.m), (self._all_reduce(P.M, P.m), 1),
                 (self.kind("rmsnorm", N=self.H, EPS=self.EPS, TARGET=self.tp, M=P.M), P.m)])
            for c, times in body.items():
                add(self._layer(rep[c], P, range(P.m), bound, pads), times)
            add(self._layer(NL - 1, P, X, bound_last, pads, final=True))
            if P.S is not None:
                add([(self._cublas(P.S, "lm_head"), 1), (self.kind("all_gather", WORLD=self.W, N=V), 1),
                     *self._select(policy, V * self.W, P.p + P.m - 1)])
                events += 1
        if (policy or {}).get("kind") == "stochastic":
            add([(self._named("seed", SEED.definition(SEED.rid_in(policy))), 1)])
        if stop:
            add([(self._named("stop", SR.definition(events, stop)[0]), 1)])
        add([(self.kind("const", W=w, BITS=bits), words) for w, bits, words in sorted(pads)])
        return out

    def _pass(self, launch: _Launch, m: int, p: int, sampled: bool) -> _Pass:
        from verity.ml.definitions.registry import tp_allreduce as AR
        from verity.ml.definitions.registry.sparse_mla_split import launch_num_splits
        from verity_vllm.program.frontend.rules.vllm_dsa import long_row_blocks
        M, TOPK = launch.tokens(m), self.index_topk
        prefills = launch.prefills((m, p), self.threshold)
        dense = prefills if prefills and max(n + q for n, q in prefills) <= TOPK else []
        keys, own = [p + i + 1 for i in range(m)], (m, p) in dense
        mqa = launch.keys([] if own else keys, without=dense)
        kept = [min(n, TOPK) for n in mqa]
        splits = {} if own else dict(zip(kept, launch_num_splits(kept, int(self.tp.num_sms), TOPK, 0)))
        fused = bool(self.net.settings["flashinfer"]) and M <= AR.fused_max_tokens(self.H, self.W)
        block = {i: N for N, rows in long_row_blocks(keys, TOPK) for i in rows}
        return _Pass(m, p, launch.tokens(1, sampled=True) if sampled else None, M, fused, keys, dense, own, mqa, splits, block)

    def _layer(self, L: int, P: _Pass, X: Sequence[int], bound: int, pads: set, final: bool = False) -> list[tuple[Need, int]]:
        """Layer L's calls in pass P that the Program keeps: the attention of its live query rows X (every row below the last
        layer), its index-K slots below position `bound`, then the norms and the MLP or MoE block."""
        ffn = (lambda n: self._moe(P.M, n, L)) if L >= int(self.C["FIRST_DENSE"]) else (lambda n: self._mlp(P.M, n, L))
        out = self._attention(L, P, X, bound, pads)
        if not final:
            return out + self._reduce_norm(P.M, P.m) + ffn(P.m) + self._reduce_norm(P.M, P.m)
        if P.fused:
            return out + self._reduce_norm(P.M, len(X)) + ffn(len(X)) + self._reduce_norm(P.M, len(X))
        (ar, _), (norm, _) = self._reduce_norm(P.M, P.m)
        live = 1 if X else 0
        return out + [(ar, live), (norm, P.m * live)] + [(nd, c * live) for nd, c in ffn(P.m)] + [(ar, live), (norm, live)]

    def _attention(self, L: int, P: _Pass, X: Sequence[int], bound: int, pads: set) -> list[tuple[Need, int]]:
        from verity_vllm.program.frontend.rules.vllm_dsa import FA3_SPARSE_GEOMETRY, PAD_SCORE, long_row_keys, pad_words
        C, TOPK, x, block = self.C, self.index_topk, len(X), P.block
        QL, KVL, R, NOPE, VH, NH = (int(C[k]) for k in ("Q_LORA", "KV_LORA", "QK_ROPE", "QK_NOPE", "V_HEAD", "NH"))
        IH, ID = int(C["INDEX_HEADS"]), int(C["INDEX_HEAD_DIM"])

        def fp8(w: str) -> Need:
            return self._fp8_linear(P.M, (L, w))

        out = [(fp8("qkv_a_w"), P.m), (self.kind("dsa_latent_row", KVL=KVL, R=R, EPS=self.EPS), P.m),
               (self.kind("dsa_rmsnorm", N=QL, EPS=self.EPS), x), (fp8("q_b_w"), x), (self.kind("dsa_q_rope", D=R), NH * x),
               (self.kind("batched_gemm", B=NH, K=NOPE, N=KVL, DOT=self.net.dot), 0 if P.own else x)]
        if L in self.net.indexers:
            long = [i for i in X if P.mqa and P.keys[i] > TOPK]
            slots, wk = min(max(bound - P.p, 0), P.m), self._cublas(P.M, (L, "wk_w"))
            out += [(wk, len(long)), (self._narrowed(wk, ID), slots - len(long)),
                    (self.kind("dsa_indexer_k_row", D=ID, ROPE=R, EPS=float(C["INDEX_K_EPS"])), slots),
                    (fp8("wq_b_w"), len(long)), (self.kind("dsa_indexer_q", H=IH, D=ID, ROPE=R), len(long))]
            for i in long:
                n, Nb = P.keys[i], block[i]
                N = long_row_keys(Nb, TOPK)
                out += [(self.kind("dsa_mqa_logits", N=Nb, H=IH, D=ID), 1), (self.kind("dsa_topk_mask_counted", N=N, TOPK=TOPK), 1),
                        (self.kind("dsa_butterfly_settings", N=N, TOPK=TOPK), 1)]
                pads.add((32, n, 1))
                if N > Nb:
                    pads.add((32, PAD_SCORE, pad_words(TOPK)))
        if P.own:
            return out + self._dense(L, P, X) + [(fp8("o_w"), x)]
        G = FA3_SPARSE_GEOMETRY[(R, KVL)]
        for i in X:
            n = P.keys[i]
            if n > TOPK:
                Nb = block[i]
                N = long_row_keys(Nb, TOPK)
                if N > Nb:
                    out.append((self.kind("dsa_butterfly_rows_pad", T=Nb, N=N, TOPK=TOPK, W=KVL + R), 1))
                    pads.add((16, 0, pad_words(TOPK, KVL + R)))
                else:
                    out.append((self.kind("dsa_butterfly_rows", N=N, TOPK=TOPK, W=KVL + R), 1))
            T = min(n, TOPK)
            out.append((self.kind("sparse_mla_split", T=T, NS=P.splits[T], NH=NH, DR=R, DV=KVL, BN=G["BN"], DOT=G["DOT"], INV=G["INV"],
                                  SM_SCALE=float(C["ATTN_SCALE"])), 1))
        return out + [(self.kind("batched_gemm", B=NH, K=KVL, N=VH, DOT=self.net.dot), x), (fp8("o_w"), x)]

    def _dense(self, L: int, P: _Pass, X: Sequence[int]) -> list[tuple[Need, int]]:
        """The dense-MHA prefill's calls (`dsa.Net.dense_plan`) of the live query rows X: kv_b_proj's rows, which every head reads up
        to its own, each row's heads, and where the step extends, the merges and the context's heads and rows, or the empty consts."""
        plan = self.net.dense_plan(self.kind, P.m, P.p, P.p, P.dense)
        NH, live = int(self.C["NH"]), 1 if X else 0
        out = [(self._fp8_linear(plan.rows, (L, "kv_b_w")), P.m * live), *[(plan.heads[i], NH) for i in X]]
        if plan.merge is not None:
            out.append((plan.merge, len(X)))
            out += ([(plan.context, NH * len(X)), (self._fp8_linear(plan.held, (L, "kv_b_w")), P.p * live)] if P.p else
                    [(e, live) for e in plan.empty])
        return out

    def _select(self, policy: Mapping[str, Any] | None, V: int, pos: int) -> list[tuple[Need, int]]:
        """A sample's select (`common.select_token`): greedy TokenSelect, or the seeded top-p select and its per-row consts."""
        if (policy or {}).get("kind", "greedy") != "stochastic":
            return [(self.kind("token_select", V=V), 1)]
        from verity_vllm.program.frontend.rules.vllm_sampling import f32_bits
        S = policy["splits"]
        fn = (self.kind("token_select_stochastic", V=V, CONSTRUCTION=self.tp.effective_sampler_construction()) if S == "derived"
              else self.kind("token_select_stochastic_split", V=V, S=int(S)))
        words = [(32, f32_bits(policy.get("top_p", 1.0))), (64, int(pos) & 0xFFFFFFFFFFFFFFFF), (32, f32_bits(policy["temperature"]))]
        return [(fn, 1), *[(self.kind("const", W=w, BITS=b), 1) for w, b in words + ([] if S == "derived" else [(32, int(S))])]]
