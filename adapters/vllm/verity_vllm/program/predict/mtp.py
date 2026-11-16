"""`dsa`'s decoder with its MTP draft layer (GLM-5.3's one `num_nextn_predict_layers` layer, at index NL): the draft layer's part of
the weights record (`Net`) and its draft forward over `dsa._Calls` (`Draft`), as vLLM d9105ea80 serves `method="mtp"` at
`num_speculative_tokens=5` (`phi.STATED_SPECULATIVE`).  Paths are under `vllm/`.

Draft step 0 (`Draft.step0`) runs over the target step's rows, its padded batch with the rejected positions
(`v1/worker/gpu/spec_decode/autoregressive/speculator.py:240-245, 303-321`), and emits in order:

- the prepare kernel (`_prepare_prefill_inputs_kernel`, speculator.py:714-730), wires: row j < kept - 1 takes the target's id at
  row j + 1, row kept - 1 the token the target sampled, rows 0..kept - 1 the target's positions;
- the embedding shard and its all-reduce (`models/deepseek_v32/nvidia/mtp.py:215-222`: with the vocabulary sharded the lookup is
  not folded into the norm);
- `_fused_eh_norm_kernel` (mtp.py:115-123) per row as `mtp_draft.MtpEhNorm_v1`'s cut (`mtp_draft.eh_cut`): `MtpEhZero_v1{H}`,
  then per half `TritonBlockRMSNormSumSq_v1` and H / 64 `DsaRMSNormApply_v1{G=64}`, e's with `enorm`, then the previous hidden
  row's with `hnorm`;
- `eh_proj` (mtp.py:132-134), the bf16 linear from 2H to H on cuBLAS's route (`build_glm52_plan` plans only sm103,
  `nvidia/glm52_low_latency_gemm.py:143-148`);
- the draft layer as a decoder layer with its indexer on (mtp.py:135-137, `mtp/speculator.py:37-41`): its input norm on
  eh_proj's rows, which are its residual (`nvidia/model.py:125-131`), `dsa._Model.attention` at layer NL, the post-attention fused
  all-reduce norm (model.py:152-154) and its MLP, MoE from `first_k_dense_replace` on (model.py:85-97), left un-reduced;
- the fused all-reduce norm with `shared_head.norm` (mtp.py:154-156), whose rows are the draft's hidden rows;
- at the request's sampled row, kept - 1 (speculator.py:467-482): the head GEMM, the all-gather and the greedy select
  (`_Calls.sample`, mtp.py:233-243);
- the compacted top-k (`compact_topk_indices`, mtp.py:195-202, `mtp/speculator.py:43-48`), wires: the sampled row's selection.

A draft step k from 1 on (`Draft.decode`) takes one row a request (`_prepare_decode_inputs_kernel`, speculator.py:796-847, wires):
the previous step's token, the position after its row's, and its hidden row.  It emits what step 0 does at one row, but its
attention skips the indexer and reads step 0's compacted top-k (`mtp/speculator.py:50-54`, `models/deepseek_v32/attention.py:
289-461`): `dsa._Model.attention` with `reuse`, the sampled row's key count, so FlashMLA reads that row's keys and not its own.

The weights record is the target's (`dsa.Net`) and then the draft layer's, under its checkpoint names `model.layers.NL.*`: its
`enorm`, `hnorm`, `eh_proj` and `shared_head.norm` (mtp.py:66-68, 81-83), then its decoder layer's parameters in the family's
module order and its buffers, each typed and keyed as `generic._mla_weight_field` maps layer NL of a configuration one layer
longer whose indexer layers include it.  `draft_layer` gives those names, their paths and that configuration (`DraftLayer`).

Premises, each unsettled at the pin and taken here rather than read:

- the embedding and the head are the target's `embed` and `lm_head` leaves: the draft model loads its own `embed_tokens` and
  `shared_head.head` (mtp.py:320-343), and whether `load_eagle_model` replaces them by the target's is not read;
- the draft runs the `_multi_step_decode` path, one forward a step, not the fused multi-step decode;
- draft step 0 runs at the target step's launch, its tokens and graph mode, and eagerly as the target does (the draft's own
  CUDA graphs are not read); sequence parallelism is off, so the norms are the fused all-reduce ones;
- the greedy select's tie-break and NaN handling are the draft sampler's;
- the rows past `kept` of draft step 0 reach no output: none is sampled, and every key a later row reads is one the target's
  metadata gives it up to its own position.  Their ids and positions are what the prepare kernel leaves in its buffers
  (speculator.py:714-730, 916-919, 941-943), so each is a Program input (`Leftover`) charged at its width, and so is its RoPE
  row, which no Definition reads off the table at a position that is a value: it is advice of 64 bf16 words a row.
"""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

from verity.circuits.defs import bind
from verity.circuits.refs import Coll, Refs, concat
from verity.circuits.types import Array, Value

from .dsa import Net as _TargetNet
from .dsa import _at, _Model
from .dsa_calls import _refuse
from .phi import Phi

#: the draft layer's own parameters beside its decoder layer's (mtp.py:66-68, 81-83), in module order: checkpoint suffix -> key
OWN = {"enorm.weight": "enorm", "hnorm.weight": "hnorm", "eh_proj.weight": "eh_proj", "shared_head.norm.weight": "shared_head_norm"}

class Leftover(NamedTuple):
    """A row of draft step 0 past the kept rows, as Program inputs: its id (I32) and position (I64) as the prepare kernel's buffers
    hold them, and its RoPE row ([QK_ROPE] bf16), advice."""
    id: Coll
    pos: Coll
    rope: Coll


class Step0(NamedTuple):
    """Draft step 0's outputs: the sampled token (I32), the draft's hidden row at the sampled row, which step 1 reads, and that
    row's compacted top-k selection (None for a row of at most INDEX_TOPK keys, which keeps every key).  A later step's are its
    token and hidden row beside step 0's selection and key count."""
    token: Coll
    hidden: Refs
    topk: Coll | None
    keys: int


class DraftLayer(NamedTuple):
    """The draft layer's part of the weights record as names and paths (`draft_layer`).

    `config` is the configuration the paths are into: the target's, one layer longer (NL + 1), with layer NL among its
    `INDEXER_LAYERS`.  `paths` maps each leaf the record holds for the draft layer, by checkpoint name `model.layers.NL.<rest>`,
    in record order, to its path:
    - the layer's own four (`OWN`) to the one-part path of their key;
    - its decoder layer's parameters and buffers to what `generic._mla_weight_field` gives under `config`, a path into
      `analytic.mla_weights_type(config)`;
    - a KV-cache scale (`generic._MLA_SCALES`), an f32 word in the record, to None, as the target's are unmapped."""
    config: dict[str, Any]
    paths: dict[str, tuple[str | int, ...] | None]


def draft_layer(hf: dict[str, Any], C: dict[str, Any]) -> DraftLayer:
    """The draft layer's names and paths (`DraftLayer`) for the checkpoint config `hf` and the target's configuration `C`
    (`generic.derive_config` at the rank): what `Net` builds the draft layer's leaves from, for a reader that matches checkpoint
    names to the record."""
    from verity_vllm.engine.profiles import family_facts as FF
    from verity_vllm.observe.fold.profiles.generic import _MLA_SCALES, _mla_weight_field
    NL = int(C["NL"])
    config = {**C, "NL": NL + 1, "INDEXER_LAYERS": (*(C.get("INDEXER_LAYERS") or ()), NL)}
    field = _mla_weight_field(hf, config)
    paths: dict[str, tuple[str | int, ...] | None] = {f"model.layers.{NL}.{s}": (key,) for s, key in OWN.items()}
    params = FF.facts_for(hf)["params"]
    for rest in (*params["order"], *params["buffers"]):
        name = f"model.layers.{NL}.{rest}"
        path = field(name)
        if path is not None:
            paths[name] = path
        elif _MLA_SCALES.match(name):
            paths[name] = None
    return DraftLayer(config, paths)


class Net(_TargetNet):
    """`dsa.Net` with its MTP draft layer: the weights record (`leaves`) is the target's and then the draft layer's, and `draft`
    is the draft forward over a target model's builder, weights, peers and launch."""

    def __init__(self, phi: Phi, M: int, history: int) -> None:
        from verity_vllm.definitions.registry.prims import BF16
        from verity_vllm.program import model as analytic

        from .phi import STATED_SPECULATIVE
        super().__init__(phi, M, history)
        hf, C, NL = phi.hf_config, self.C, self.NL
        facts = []
        if (phi.engine or {}).get("speculative") != STATED_SPECULATIVE:
            facts.append(f"an MTP draft under engine speculative {(phi.engine or {}).get('speculative')!r}, not {STATED_SPECULATIVE}")
        if hf.get("num_nextn_predict_layers") != 1:
            facts.append(f"{hf.get('num_nextn_predict_layers')!r} MTP layers (the draft forward is one layer's)")
        if hf.get("index_share_for_mtp_iteration") is not True:
            facts.append("an MTP layer whose steps past the first run its indexer (index_share_for_mtp_iteration off)")
        if facts:
            _refuse("; ".join(facts))
        self.steps = int(STATED_SPECULATIVE["num_speculative_tokens"])
        H = int(C["H"])
        row = Array(H, BF16)
        own_types = {"enorm": row, "hnorm": row, "eh_proj": Array(H, Array(2 * H, BF16)), "shared_head_norm": row}
        layer = draft_layer(hf, C)
        T = analytic.mla_weights_type(layer.config)
        for name, path in layer.paths.items():
            if path is None:
                self._leaves.append((name, Value(32), None))
            else:
                t = own_types[str(path[0])] if len(path) == 1 else _at(T, path)
                self._leaves.append((name, t, (NL, path[-1])))

    def draft_peer_shapes(self, r: int) -> list[tuple[int, int]]:
        """A draft step of r rows: the embedding's all-reduce, the layer's post-attention fused all-reduce norm and the shared
        head's, the sampled row's logits gather."""
        H = int(self.C["H"])
        return [(r, H)] * 3 + [(1, int(self.C["V"]))]

    def draft(self, target: _Model, W: Coll) -> Draft:
        """The draft forward over `target`'s builder, peers and launch, `W` the weights record's Input: its own model over this net
        with the draft layer the one indexer layer, so its layer history is its own."""
        view = copy.copy(self)
        view.indexers = frozenset({self.NL})
        d = Draft(view, target.B, W, target.peers)
        d.launch, d.batched, d.table_reads = target.launch, target.batched, target.table_reads
        return d


class Draft(_Model):
    """The MTP draft forward (the module docstring) over `dsa._Calls`, its layer `dsa._Model`'s at index NL."""

    net: Net

    def step0(self, ids: list[Coll], sampled: Coll, prev: list[Refs], pos0: int, kept: int, left: list[Leftover]) -> Step0:
        """Draft step 0 over the target step's rows at positions pos0..: `ids` its input ids (I32 rows), `sampled` the token it
        sampled after its kept rows, `prev` its final-normed hidden rows, `kept` the rows the prepare kernel keeps (the target's
        rows less its rejected ones), and `left` a `Leftover` for each row past them."""
        from verity_vllm.definitions.registry.prims import BF16
        M, R = len(ids), int(self.C["QK_ROPE"])
        if not (1 <= kept <= M and len(prev) == M and len(left) == M - kept):
            raise ValueError(f"draft step 0 over {M} rows keeps 1..{M} of them and takes the target's {M} hidden rows and a Leftover "
                             f"each past them: kept {kept}, {len(prev)} hidden rows, {len(left)} Leftovers")
        rows = [*(t.refs for t in ids[1:kept]), sampled.refs, *(o.id.refs for o in left)]
        pos = [self.call(self.k("const", W=64, BITS=pos0 + j)).refs for j in range(kept)] + [o.pos.refs for o in left]
        table = self.w["cos_sin"]
        rope = [o.rope.refs for o in left]
        self.w["cos_sin"] = Coll(Array(pos0 + M, Array(R, BF16)),
                                 concat([table.refs.slice(0, (pos0 + kept) * R), *rope])) if rope else table
        try:
            x = self._layer(rows, pos, prev, pos0)
        finally:
            self.w["cos_sin"] = table
        token = self.sample(x[kept - 1], 0, pos0 + kept, None)
        keys = len(self.logs[self.net.NL]["latent"]) - M + kept
        return Step0(token, x[kept - 1], self.topk.get(pos0 + kept - 1), keys)

    def decode(self, prev: Step0, k: int, pos: int) -> Step0:
        """Draft step k >= 1 at position `pos` (the sampled row's + k) over the previous step's token and hidden row."""
        if self.launch.ctx is not None:
            _refuse("a draft step past the first in a launch batch (it runs at the requests' one row each, which the target "
                    "step's batch does not state)")
        if not 1 <= k < self.net.steps:
            raise ValueError(f"draft step {k}: the steps past the first are 1..{self.net.steps - 1}")
        x = self._layer([prev.token.refs], [self.call(self.k("const", W=64, BITS=pos)).refs], [prev.hidden], pos, reuse=prev.keys)
        return Step0(self.sample(x[0], 0, pos + 1, None), x[0], prev.topk, prev.keys)

    def _layer(self, rows: list[Refs], pos: list[Refs], prev: list[Refs], pos0: int, reuse: int | None = None) -> list[Refs]:
        """The draft layer over token rows at positions pos0.. (`pos` each as an I64 value), the previous hidden rows `prev` ->
        the draft's hidden rows, normed by `shared_head.norm`."""
        from verity_vllm.definitions.registry import mtp_draft as MD
        NL, H, V = self.net.NL, self.H, int(self.C["V"])
        w = lambda f: self.w[(NL, f)]  # noqa: E731
        emb = self.k("embedding_shard", VS=V, H=H, START=self.net.rank * V)
        e = self.all_reduce([self.call(emb, t, self.w["embed"].refs).refs for t in rows])
        cut = MD.eh_cut(bind(MD.MtpEhNorm, H=H, BLOCK=1 << (H - 1).bit_length(), WARPS=4, EPS=self.EPS))
        eh = [self._eh_norm(cut, p, a, b) for p, a, b in zip(pos, e, prev)]
        res = self.cublas(eh, w(OWN["eh_proj.weight"]))
        norm = self.k("rmsnorm", N=H, EPS=self.EPS, TARGET=self.net.tp, M=self.launch.tokens(len(rows)))
        x = [self.call(norm, r, w("ln1").refs).refs for r in res]
        x, res = self.reduce_norm(self.attention(NL, x, pos0, reuse), res, w("ln2"))
        mlp = self.moe(NL, x) if int(self.C["FIRST_DENSE"]) <= NL else self.mlp(NL, x)
        x, _ = self.reduce_norm(mlp, res, w(OWN["shared_head.norm.weight"]))
        return x

    def _eh_norm(self, cut: list, pos: Refs, e: Refs, p: Refs) -> Refs:
        """One row of `_fused_eh_norm_kernel` as its cut: the position select of e, then each half's scale and its apply units, e's
        with `enorm` and p's with `hnorm` -> [enorm(e') | hnorm(p)]."""
        from verity_vllm.definitions.registry.mtp_draft import APPLY_G
        (zero, _), (sumsq, _), (apply, _) = cut
        H, NL = self.H, self.net.NL
        out = []
        for x, wf in ((self.call(zero, pos, e).refs, OWN["enorm.weight"]), (p, OWN["hnorm.weight"])):
            r = self.call(sumsq, x).refs
            wr = self.w[(NL, wf)].refs
            out += [self.call(apply, x.slice(j, j + APPLY_G), wr.slice(j, j + APPLY_G), r).refs for j in range(0, H, APPLY_G)]
        return concat(out)
