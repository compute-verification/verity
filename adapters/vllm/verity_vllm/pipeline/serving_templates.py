"""The templates of the serving rows (`pipeline.serving_rows`, whose module text states their rules): a template read through its
word link (:func:`template`), and the templates a Program row's Definition holds instances of (:func:`unit_templates`: the row
itself, a ``CUT`` family's batched members, a ``COORDS`` family's coordinates, a ``COLS`` family's column blocks, a ``PARTS``
family's parts, a ``SLICES`` family's slices, a ``KCHUNKS`` row's K-chunks, a ``CHAINS`` row's key blocks, ends and combines,
a ``STAGES`` row's stages, and an ``EXPERTS`` row's per-expert blocks and table reads and a ``SLOTS`` router row's slots), and
the unit of one output word of a row (:func:`coordinate`)."""
from __future__ import annotations

import importlib
import math
import re
from dataclasses import dataclass
from typing import Any

#: the row Definitions whose Calls are cut into template instances (their bodies batch the template), the rest being the row itself
CUT = ("RoPE_v1", "Gemm_v1", "ScaledMmS8_v1")
#: the row Definitions whose Calls are split into coordinates of themselves: coordinate n of a row is its Definition at N = 1
#: (the expert-indexed pair GEMMs, `fp8_moe_experts.pair_rows` at one column; FlashInfer's FP8 linear, `deepgemm`, one output
#: word with the row's quantisation of its own, where its ``PARTS`` quantise the row once; the block-scaled FP8 GEMM,
#: `fp8.ScaledMmFp8BlockSharedScale`, one output word, ``w[n]`` and ``ws[n // G]``), the ``coord`` layout; each family's E and
#: G where its statics have none
COORDS = {"MoeExpertFp8Gemm_v1": {}, "MoeExpertFp8GemmW_v1": {}, "FlashInferFp8Linear_v1": {"E": 1, "G": 128},
          "ScaledMmFp8BlockSharedScale_v1": {"E": 1}}
#: the ``COORDS`` families whose coordinates may also be c outputs wide (`coord_widths`): coordinate j of a row is then its
#: Definition at N = c, reading ``w[:, jc:(j + 1)c, :]`` and their one scale row ``ws[:, jc // G, :]``, and returning output
#: words [jc, (j + 1)c).  A wide coordinate reads its leading operands once for c outputs (the block-scaled FP8 GEMM's
#: activation row and scales), so its block hashes fewer rows an output
WIDE_COORDS = ("ScaledMmFp8BlockSharedScale_v1",)
#: the row Definitions elementwise along the last axis of every parameter and of the output, by that axis's static: a row W wide
#: is W / w units of its Definition at width w (a halving of W), unit j reading and returning columns [jw, jw + w), the ``cols``
#: layout (`moe.MoeSum`'s top-k sum, `tp_allreduce.AllReduceMnnvl`'s all-reduce and its residual add, torch's multimem
#: all-reduce, `tp_allreduce.AllReduceMultimem`, where `cols_refused` admits its blocks, and `b1.ResidualAdd`'s residual add);
#: and the norms' apply units, ``PARTS`` of a norm row whose G columns are blocks of the row's N (``COL_SPANS``), each reading
#: the row's scale whole
COLS = {"MoeSum_v1": "H", "AllReduceMnnvl_v1": "N", "AllReduceResidualMnnvl_v1": "N", "AllReduceMultimem_v1": "SIZE",
        "ResidualAdd_v1": "N", "RMSNormApplyCudaVector8_v1": "G", "RMSNormFusedApplyCudaVector8_v1": "G",
        "DsaRMSNormApply_v1": "G", "RmsNormMnnvlApply_v1": "G"}
#: the row Definitions that are batches along axes of their statics, which some parameters carry and the others don't: a row of
#: extents a_0, a_1, .. along its axes is a_0 a_1 .. units of its Definition at every axis 1, unit (i_0, i_1, ..) (row-major)
#: reading element i_k along axis k of each parameter that carries it and the others whole, and returning the output's element
#: (i_0, i_1, ..), its leading dimensions being the axes in order (the ``slice`` layout).  Per family: its axes, and per parameter
#: its dimension of each axis (None: it doesn't carry it).  The GEMMs of one token row into W's rows (vLLM's router GEMM, the LM
#: head, cuBLAS's split-K), torch.bmm into each member's coordinates, DSA's logits into keys, the vocabulary shard's embedding
#: into columns (every unit reads the token and gathers its column) and the all-gather into its ranks' shards; cuBLAS's GEMMs
#: at an f32 output too, split-K or not (the MoE router gate)
SLICES = {"LlBf16DotprodGemm_v1": (("N",), ((None,), (0,))), "Gemm_v2": (("N",), ((None,), (0,))),
          "CublasSplitKGemm_v1": (("N",), ((None,), (0,))), "BatchedGemm_v1": (("B", "N"), ((0, None), (0, 1))),
          "DsaMqaLogits_v3": (("N",), ((None,), (None,), (0,), (0,))), "EmbeddingShard_v1": (("H",), ((None,), (1,))),
          "AllGather_v1": (("WORLD",), ((0,),)), "CublasSplitKGemmF32Out_v1": (("N",), ((None,), (0,))),
          "GemmF32Out_v1": (("N",), ((None,), (0,)))}
#: the row Definitions whose Calls are split into parts, each a Definition of its own at the row's statics it names: per part, its
#: family, what it reads in order (``("arg", g)`` the row's argument group g, ``("out", j)`` the row's committed output member j,
#: ``("mid", name)`` the row's intermediate) and what it returns (``j`` the row's committed output member j, ``name`` an
#: intermediate: a value the row's Definition computes and serving never stores, which the reader computes by its twin,
#: `part_rows.MIDS`, and the window wires to every part that reads it, `part_rows.wire`); together they return every output
#: once, and the row computes what they compose to (tests/program/test_tp_allreduce_parts.py).  A ``COLS`` part is read in
#: column blocks, a ``COORDS`` part in coordinates, any other one whole (the ``part`` layout).
#: `tp_allreduce.AllReduceRmsNormMnnvl`: its all-reduce and residual add (r), then its norm of the committed r.
#: `deepgemm.DeepGemmFp8Linear`: its quantisation of x (q, s), then its GEMM of the committed q and s.
#: `deepgemm.FlashInferFp8Linear`: its quantisation of x once (q, s, which FlashInfer keeps in its workspace), then the same GEMM
#: of those q and s (tests/program/test_deepgemm_parts.py).
#: The RMSNorms, cut at the scale every output element reads (`cuda_rmsnorm_split`, `dsa_norm`, `tp_allreduce`): the row's sum
#: of squares (its scale, an intermediate), then its outputs in ``COLS`` column blocks of G elements, each reading its columns,
#: their weights and the scale (vLLM's vector-8 CUDA norms, the fused one with its residual add redone per block; DSA's;
#: and the fused all-reduce's norm of its committed r, after its all-reduce and residual add), so that no unit returns a whole
#: row. A row outside every decoder layer keeps its ``FINAL`` cut.
_SUMSQ = (("arg", 0),)
PARTS = {"AllReduceRmsNormMnnvl_v1": (("AllReduceResidualMnnvl_v1", (("arg", 0), ("arg", 1)), (1,)),
                                      ("RmsNormMnnvlSumSq_v1", (("out", 1),), ("rs",)),
                                      ("RmsNormMnnvlApply_v1", (("out", 1), ("arg", 2), ("mid", "rs")), (0,))),
         "DeepGemmFp8Linear_v1": (("Fp8GroupQuantUe8m0_v1", (("arg", 0),), (0, 1)),
                                  ("ScaledMmFp8BlockSharedScale_v1", (("out", 0), ("out", 1), ("arg", 1), ("arg", 2)), (2,))),
         "FlashInferFp8Linear_v1": (("FlashInferFp8Quant1x128_v1", (("arg", 0),), ("q", "s")),
                                    ("ScaledMmFp8BlockSharedScale_v1", (("mid", "q"), ("mid", "s"), ("arg", 1), ("arg", 2)), (0,))),
         "RMSNormCudaVector8_v1": (("RMSNormSumSqCudaVector8_v1", _SUMSQ, ("scale",)),
                                   ("RMSNormApplyCudaVector8_v1", (("arg", 0), ("arg", 1), ("mid", "scale")), (0,))),
         **dict.fromkeys(("RMSNormFusedCudaVector8_v1", "RMSNormFusedCudaVector8_v3"), (
             ("RMSNormFusedSumSqCudaVector8_v1", (("arg", 0), ("arg", 1)), ("scale",)),
             ("RMSNormFusedApplyCudaVector8_v1", (("arg", 0), ("arg", 1), ("arg", 2), ("mid", "scale")), (0, 1)))),
         "DsaRMSNorm_v1": (("DsaRMSNormSumSq_v1", _SUMSQ, ("r",)),
                           ("DsaRMSNormApply_v1", (("arg", 0), ("arg", 1), ("mid", "r")), (0,)))}
#: the cut of a row outside every decoder layer (`in_layer`) of a row family whose ``PARTS`` cut a norm at its scale: the final
#: norm, whose output feeds the LM head, stays one unit a token on its unsplit Definition (the row itself, ``()``; the fused
#: all-reduce's all-reduce and residual add, then its whole norm of r), where a split's apply blocks would raise the location's
#: leakage and its sum of squares rewrite the whole row's scale. Such a row is read by its ``FINAL`` cut, and never by its
#: scale cut; a row inside a layer is read by its scale cut when the partition holds it, else by its ``FINAL`` cut.
FINAL = {"AllReduceRmsNormMnnvl_v1": (PARTS["AllReduceRmsNormMnnvl_v1"][0],
                                      ("RmsNormMnnvl_v1", (("out", 1), ("arg", 2)), (0,))),
         **dict.fromkeys(("RMSNormCudaVector8_v1", "RMSNormFusedCudaVector8_v1", "RMSNormFusedCudaVector8_v3", "DsaRMSNorm_v1"), ())}
_IN_LAYER = re.compile(r"(?:^|\.)layers\.\d+\.")


def in_layer(op_path: str) -> bool:
    """Whether a row of ``op_path`` (its module path, the wrapper's prefix stripped) is inside a decoder layer: its norms and
    the fused all-reduces the layer calls (``model.layers.<n>.``), not the model's final norm (``model.norm``, or the fused
    all-reduce + norm the model itself calls)."""
    return _IN_LAYER.search(op_path) is not None


def scale_cut(word: str, spec: str) -> str | None:
    """For a row of ``spec`` whose family has a ``FINAL`` cut: ``"split"`` when template word ``word`` is only in its scale cut
    (``PARTS``), ``"whole"`` when it is the row itself or only in its ``FINAL`` cut, None when both cuts hold it (the fused
    all-reduce's residual add) or the row has no ``FINAL`` cut."""
    from verity_vllm.query.program_view import spec_base

    row, fam = spec_base(spec), spec_base(word)
    if row not in FINAL:
        return None
    if fam == row:
        return "whole"
    split, final = any(p[0] == fam for p in PARTS[row]), any(p[0] == fam for p in FINAL[row])
    return None if split == final else "split" if split else "whole"


def final_hits(ts: list[Template], hits: list[tuple[int, int]], spec: str, op_path: str) -> list[tuple[int, int]]:
    """Of the hits ``(k, units)`` of templates ``ts`` on a row of ``spec`` at ``op_path``, those that read it
    (`serving_rows.member_rows`): all of them unless the row's family has a ``FINAL`` cut.  Such a norm row is read by its scale
    cut (``PARTS``) when it is inside a decoder layer (`in_layer`) and the partition holds a part of that cut, and otherwise only
    by its ``FINAL`` cut (the row itself, or the fused all-reduce's residual add and whole norm): the final norm is never split."""
    from verity_vllm.query.program_view import spec_base

    if spec_base(spec) not in FINAL:
        return hits
    cut = {k: scale_cut(ts[k].word, spec) for k, _ in hits}
    take = "split" if in_layer(op_path) and "split" in cut.values() else "whole"
    return [(k, n) for k, n in hits if cut[k] in (None, take)]
#: the statics a ``PARTS`` row family's parts take that the row has none of: the block-FP8 linears quantise and scale in 128-wide
#: blocks (`deepgemm.FP8_BLOCK`)
PART_STATICS = {"DeepGemmFp8Linear_v1": {"G": 128}, "FlashInferFp8Linear_v1": {"G": 128}}
#: each part family, by the row families it is a part of (in its ``PARTS`` or ``FINAL`` cut)
PART_OF: dict[str, dict[str, tuple]] = {}
for _cuts in (PARTS, FINAL):
    for _row, _ps in _cuts.items():
        for _p in _ps:
            PART_OF.setdefault(_p[0], {})[_row] = _p
#: the norms' apply units (``COLS`` along G): the row static their G columns are blocks of, the row's N
COL_SPANS = {"RMSNormApplyCudaVector8_v1": "N", "RMSNormFusedApplyCudaVector8_v1": "N", "DsaRMSNormApply_v1": "N",
             "RmsNormMnnvlApply_v1": "N"}


def col_span(fam: str) -> str:
    """The row static a ``COLS`` family's column blocks divide: its own axis, or the row's ``COL_SPANS`` static."""
    return COL_SPANS.get(fam, COLS[fam])
#: the chunk families that cut a ``COORDS`` family's coordinate along K (the ``kchunk`` layout, `kchunk_rows`), by the row family:
#: a coordinate of K weights is C = K / k chunks of k (whole scale blocks and tiles), chunk 0 FIRST, chunk C - 1 LAST, each
#: continuing the one before's f32 accumulator (`fp8_moe_experts.MoeExpertFp8CoordinateChunk_v1`, the w13 launch)
KCHUNKS = {"MoeExpertFp8Gemm_v1": "MoeExpertFp8CoordinateChunk_v1"}
#: the row Definitions whose heads are cut along their keys into chains (the ``chain`` layout, `chain_rows`), by the row family,
#: the families of its units in their roles (`CHAIN_ROLES`): its block family, one key block carrying the online softmax's state
#: to the next in FA3's visit order (a chain's first visited block FIRST at its key count, every later one at BN), its end family,
#: a head's epilogue on its last block's lanes and O, the two families a key block is served as instead where the partition
#: names them, its scores family (the block's running max, lanes and bf16 probabilities) and its values family (the block's O on
#: one column block of `values_widths`, off the scores unit's probabilities), and a split row's
#: (`sparse_mla_split.SparseMlaSplit_v1`, FA3's split kernel and its combine) split end in column blocks, its combine of each
#: column block of the splits, and its split end on one column block of values units; the units are the cut the row's module
#: names (`sparse_mla.CUTS`, `sparse_mla_split.CUTS`: in whole key blocks, which circuit-check holds to the Flock block in the
#: row's place, and at each of `values_widths` in scores and values units)
CHAINS = {"SparseMla_v1": ("SparseMlaHeadBlock_v1", "SparseMlaHeadEnd_v1", "SparseMlaHeadScores_v1", "SparseMlaHeadValues_v1"),
          "SparseMlaSplit_v1": ("SparseMlaHeadBlock_v1", "SparseMlaHeadEnd_v1", "SparseMlaHeadScores_v1", "SparseMlaHeadValues_v1",
                                "SparseMlaSplitEndCols_v1", "SparseMlaCombine_v1", "SparseMlaSplitEnd_v1")}
#: a ``CHAINS`` unit's role by its index in its row family's entry
CHAIN_ROLES = ("block", "end", "scores", "values", "cols", "combine", "split")
CHAIN_OF = {f: CHAIN_ROLES[i] for fs in CHAINS.values() for i, f in enumerate(fs)}
#: the row Definitions cut into stages (the ``stage`` layout, `stage_rows`), by the row family: its stage families, each carrying
#: what the stage after it reads.  A stochastic top-p select's are every split count's top-p pipeline (stats blocks and their
#: parts, probabilities, each round's step blocks and combines) and the select's chunks (`topp_stages`); a greedy select's are
#: its chunks' partial argmaxes and their combine (`greedy_stages`).  The units are the cut the row's module names
#: (`sampling.CUTS`, `b1.CUTS`: the ``plan`` of the module ``STAGE_PLANS`` names), which circuit-check holds to the Flock block
#: in the row's place
_TOPP_STAGES = ("TopPStatsHead_v1", "TopPStatsExp_v1", "TopPProbs_v1", "TopPStep_v1", "GumbelTopPSelect_v1")
STAGES = {"GumbelTopPTokenSelect_v1": _TOPP_STAGES, "GumbelTopPTokenSelectSharedGreedy_v1": _TOPP_STAGES,
          "TokenSelect_v1": ("GreedyChunk_v2", "GreedyCombine_v2")}
STAGE_OF = {f: tuple(row for row, fs in STAGES.items() if f in fs) for fs in STAGES.values() for f in fs}
_TOPP_PLAN = "verity.definitions.registry.topp_stages"
#: the module whose ``plan`` and ``cut`` a ``STAGES`` row family's units are
STAGE_PLANS = {"GumbelTopPTokenSelect_v1": _TOPP_PLAN, "GumbelTopPTokenSelectSharedGreedy_v1": _TOPP_PLAN,
               "TokenSelect_v1": "verity.definitions.registry.greedy_stages"}
#: the row Definitions cut per expert (`expert_rows`), by the row family: its block family, a pair's GEMM over n of its expert's
#: rows (n a power of two dividing G and N, N / n blocks of the row), each block's rows and its G-block's scale row read at the
#: pair's expert from a table registered one expert a row (`gates.embedding.EmbeddingRead_v1`, ``READ``), whose key is the slot
#: id of the token's router row cut into slots (``SLOTS``)
EXPERTS = {"MoeExpertFp8Gemm_v1": "MoeExpertFp8BlockGemm_v3", "MoeExpertFp8GemmW_v1": "MoeExpertFp8BlockGemmW_v3"}
#: the fused blocks of the per-expert cut, by the row family: a block GEMM whose body reads its block's rows from the expert's
#: table itself (`boolean_fp8_moe.MoeExpertFp8ReadGemm(W)_v1`, its ``READ`` the read of a block's rows), so a block's rows are
#: never a member's output; its G-block's scale row is still a read of its own
FUSED = {"MoeExpertFp8Gemm_v1": "MoeExpertFp8ReadGemm_v1", "MoeExpertFp8GemmW_v1": "MoeExpertFp8ReadGemmW_v1"}
#: the router row Definitions cut into slots for the per-expert cut, by the row family: the same words, one output a slot's weight
#: or id (`moe.MoeRouterSigmoidBiasSlots`)
SLOTS = {"MoeRouterSigmoidBias_v1": "MoeRouterSigmoidBiasSlots_v1"}
#: the family of the per-expert cut's table reads
READ = "EmbeddingRead_v1"
#: DSA's long rows' history reads cut into reads of the request's history table at the layer (`history_rows`), by the row
#: family: its writer's family, whose units' outputs are the table's rows.  A row of TOPK reads is TOPK ``READ`` units at
#: V = the table's capacity (`verity.commitments.history.CAPACITY`) and the row's H, the ``history`` layout
HISTORY = {"DsaHistoryReads_v1": "DsaLatentRow_v1"}
#: a member's layout by the family of its template's word; any other template is the row itself
LAYOUTS = {"RoPEHead_v1": "rope", "GemmCoordinate_v1": "gemm", "ScaledMmS8Coordinate_v1": "s8", READ: "read",
           **dict.fromkeys(EXPERTS.values(), "expert"), **dict.fromkeys(FUSED.values(), "fused"),
           **dict.fromkeys(SLOTS.values(), "slots")}
#: a row template's element widths whose ports the rows commit as their little-endian bytes
PACKED = frozenset((8, 16, 32, 64))


@dataclass(frozen=True)
class Template:
    """A partition's template as the rows are read for it: its descriptor id and Definition (the population program's), the id of
    the word Definition it computes (its own for a word Definition), and its member layout (``LAYOUTS``; ``"coord"`` for a
    ``COORDS`` family at N = 1 or a ``WIDE_COORDS`` one at a power of two N dividing its G, ``"cols"`` for a ``COLS`` family, ``"part"`` for any other ``PARTS`` part, ``"slice"`` for a
    ``SLICES`` family at every axis 1, ``"kchunk"`` for a ``KCHUNKS`` chunk family, ``"chain"`` for a ``CHAINS`` unit family,
    ``"stage"`` for a ``STAGES`` stage family, ``"expert"``, ``"fused"``, ``"read"`` and ``"slots"`` for the per-expert cut's
    block GEMMs, fused blocks, table reads and router slots (``EXPERTS``, ``FUSED``, ``READ``, ``SLOTS``), ``"history"`` for a
    ``HISTORY`` row's history reads (`history_read`); ``"bits"`` for a row template that is a Boolean Definition with no word
    Definition (DSA's long row's units: the mask's cut and the index), whose ports are bits and no words: each port is committed
    as a bit row, its bits low first (`interface.unit_rows`, which holds each to whole u16 words); else ``"row"``).
    ``packed``: a row, coordinate, column block, slice, part, K-chunk, chain, stage, expert or slots template with a port that is not 16-bit
    elements, each port's element bits (its parameters', then its return fields'); its ports are committed as their elements'
    bytes. A read's (a history read's too) and a fused block's ports are bits (a key, digests, salt and path, and a fused block's operands' bytes),
    committed as Flock's bit rows."""
    id: str
    fn: Any
    word: str
    layout: str
    packed: tuple[int, ...] | None = None


_TEMPLATES: dict[str, Template] = {}


def template(t: str) -> Template:
    """Template ``t`` through its word link: the registered Definition it names, bound to exactly ``t``; the word Definition it computes
    (`trace.word_of` for a Boolean Definition), whose signature is the template's word view; refused by name when ``t`` names no
    Definition or a port of its word view is not 16-bit words (a row or coordinate template's: not whole u16 words of one ``PACKED``
    width; a Boolean row template with no word Definition is laid out as its bits, ``"bits"``)."""
    hit = _TEMPLATES.get(t)
    if hit is not None:
        return hit
    from verity.circuits import boolean as BL
    from verity.circuits.boolean.trace import word_of
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.definitions import boolean as BB
    from verity.definitions.call_scope import specialization
    from verity_vllm.query.program_view import spec_base, spec_statics

    # registers EmbeddingRead_v1, a read's and a READ static's Definition, which BB.load does not
    importlib.import_module("verity.commitments.gates.embedding")
    errors = BB.load()
    try:
        fn = specialization(spec_base(t), spec_statics(t))
    except (KeyError, TypeError, ValueError) as e:
        raise LookupError(f"template {t} names no registered Definition: {type(e).__name__}: {e}"
                          + (f" (registry modules that failed to import: {errors})" if errors else "")) from e
    fn = bind(fn) if isinstance(fn, CompositeDefinition) else fn
    if descriptor_id(fn) != t:
        raise LookupError(f"template {t} binds to {descriptor_id(fn)}, not to itself")
    params, ret = fn.signature()
    types = [ty for _, ty in params] + [ret]
    if getattr(getattr(fn, "defn", fn), "word", None) is not None:
        w = word_of(fn)
        w = bind(w) if isinstance(w, CompositeDefinition) else w
        wparams, wret = w.signature()
        if [BL.word_type(ty) for ty in types] != [ty for _, ty in wparams] + [wret]:
            raise ValueError(f"template {t}'s word view is not the signature of {descriptor_id(w)}, the word Definition it names")
        widths = {e for ty in types for e in BL.elements(ty)}
        view = [BL.word_type(ty) for ty in types]
    else:
        w = fn
        widths = {x for ty in types for x in ty.width_set()}
        view = types
    word = descriptor_id(w)
    sl = SLICES.get(spec_base(word))
    layout = ("coord" if coord_width(word) else "cols" if spec_base(word) in COLS
              else "part" if spec_base(word) in PART_OF else "slice" if sl and all(int(spec_statics(word)[a]) == 1 for a in sl[0])
              else "kchunk" if spec_base(word) in KCHUNKS.values()
              else "chain" if spec_base(word) in CHAIN_OF else "stage" if spec_base(word) in STAGE_OF
              else "history" if history_read(word) else LAYOUTS.get(spec_base(word), "row"))
    if layout == "row" and w is fn and widths == {1}:
        layout = "bits"
    packed = None
    if widths != {16} and layout not in ("s8", "read", "fused", "history", "bits"):
        ports = [*view[:-1], *(getattr(view[-1], "fields", None) or (view[-1],))]
        elems = [ty.width_set() for ty in ports]
        if layout not in ("row", "coord", "cols", "slice", "part", "kchunk", "chain", "stage", "expert", "slots") or any(
                len(e) != 1 or not e <= PACKED or sum(ty.leaf_widths()) % 16 for e, ty in zip(elems, ports)):
            raise ValueError(f"template {t}: the serving rows commit u16 words, and its ports' {'elements' if w is not fn else 'words'} "
                             f"are {sorted(widths)} bits (a row template's port may be elements of one of {sorted(PACKED)} bits, "
                             f"whole u16 words)")
        packed = tuple(next(iter(e)) for e in elems)
    hit = _TEMPLATES[t] = Template(t, fn, word, layout, packed)
    return hit


def history_read(word: str) -> bool:
    """Whether ``word`` is a ``HISTORY`` read: ``READ`` at V = a history table's capacity, which no expert table has (a template
    is its Definition, so another read at that V would be the same template, which `history_rows.Plan` refuses unless its rows
    are ``HISTORY``'s)."""
    from verity.commitments.history import CAPACITY
    from verity_vllm.query.program_view import spec_base, spec_statics

    return spec_base(word) == READ and int(spec_statics(word).get("V", 0)) == CAPACITY


def cols_refused(word: str) -> str | None:
    """Why ``word``'s ``COLS`` column blocks are not the columns of its family's rows, or None (a family elementwise at every
    statics).  torch's multimem all-reduce (`tp_allreduce.AllReduceMultimem`) reduces each owner rank's slice of its SIZE
    elements through that owner's tree, and a block's owners are not the row's: a block computes the row's columns only when
    every owner's tree is one, and is a Definition only as whole 16-byte vectors (`tp_allreduce.multimem_owners`)."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    if spec_base(word) != "AllReduceMultimem_v1":
        return None
    from verity.definitions.registry import tp_allreduce as TA
    from verity.definitions.registry import tp_allreduce_nodes as TN

    S = spec_statics(word)
    try:
        world = TA.ranks(S["WORLD"])
        trees = TN.multimem_trees(S["TREES"], world)
        TA.multimem_owners(world, int(S["SIZE"]))
    except (KeyError, TypeError, ValueError) as e:
        return f"{word}: {type(e).__name__}: {e}"
    if len(set(trees)) > 1:
        return (f"{word}: its owners' TREES are {len(set(trees))} trees, and a column block's elements are reduced through "
                f"other owners' trees than the row's")
    return None


def coord_width(word: str) -> int:
    """The outputs template ``word`` holds as a coordinate: 1 for a ``COORDS`` family at N = 1, and N for a ``WIDE_COORDS``
    family at a power of two N > 1 dividing its G; else 0 (it is no coordinate)."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    base = spec_base(word)
    if base not in COORDS:
        return 0
    S = {**COORDS[base], **spec_statics(word)}
    c = int(S["N"])
    if c == 1:
        return 1
    return c if base in WIDE_COORDS and c > 1 and c & (c - 1) == 0 and int(S["G"]) % c == 0 else 0


def coord_widths(fam: str, statics: dict[str, Any]) -> list[int]:
    """The widths c > 1 of a ``WIDE_COORDS`` family's coordinates on a row of ``statics``, coarse to fine: each power of two
    below the row's N that divides its N and its G; none for any other family."""
    if fam not in WIDE_COORDS:
        return []
    S = {**COORDS[fam], **statics}
    N, G = int(S["N"]), int(S["G"])
    return [1 << j for j in range(G.bit_length() - 1, 0, -1) if (1 << j) < N and N % (1 << j) == 0 and G % (1 << j) == 0]


def part(word: str, spec: str) -> tuple | None:
    """The ``PARTS`` entry of template word ``word`` on a row of ``spec`` when ``word``'s family is a part of the row's; else None
    (the row is ``word``'s own family)."""
    from verity_vllm.query.program_view import spec_base

    return PART_OF.get(spec_base(word), {}).get(spec_base(spec))


def values_widths(DV: int) -> list[int]:
    """The column widths of a ``CHAINS`` row's values units, coarse to fine: each G = DV / 2^j of at least 16 columns below the
    row's DV output columns, a values unit's O on G columns then at least 64 bytes."""
    return [DV >> j for j in range(1, DV.bit_length()) if DV % (1 << j) == 0 and DV >> j >= 16]


def chain_ways(spec: str) -> list[tuple[int | None, tuple[tuple[str, int], ...]]]:
    """The ways a ``CHAINS`` row ``spec``'s key blocks are served, each its unit templates with their instances a row: whole
    key blocks first (None, the cut its module names), then each key block its scores unit and values units of G columns, at
    each G of `values_widths`, coarse to fine."""
    from verity.circuits.codec import descriptor_id
    from verity.definitions.registry import sparse_mla, sparse_mla_split
    from verity_vllm.query.program_view import spec_base, spec_statics

    sp, chain = _bind(spec), {**sparse_mla.CUTS, **sparse_mla_split.CUTS}[spec_base(spec)]
    return [(g, tuple((descriptor_id(c), n) for c, n in (chain(sp) if g is None else chain(sp, cols=g))))
            for g in (None, *values_widths(int(spec_statics(spec)["DV"])))]


def chunk_widths(K: int, G: int, KB: int) -> list[int]:
    """The K-chunk widths of a ``KCHUNKS`` coordinate of K weights: each k < K that cuts it into whole scale blocks and tiles."""
    return [K // c for c in range(2, K // G + 1) if K % c == 0 and (K // c) % G == 0 and (K // c) % KB == 0]


_UNITS: dict[str, tuple[tuple[str, int], ...]] = {}


def unit_templates(spec: str) -> tuple[tuple[str, int], ...]:
    """The word Definitions a Program row's Definition (its word view's ``spec``) holds instances of, ``(descriptor id, instances)``:
    the row itself, then, for a ``CUT`` family, each Definition its body batches (a batch's members; Input batches are its
    parameters), for a ``COORDS`` family of N > 1, its coordinates (a ``WIDE_COORDS`` family's Definition at each width c of
    `coord_widths`, N / c of them, coarse to fine, then every family's at N = 1, N of them), for a ``COLS`` family,
    its Definition at each halving W / k of its width (k of them) that `cols_refused` admits, for a ``PARTS`` family, each part
    of its ``PARTS`` and ``FINAL`` cuts at the row's statics and its ``PART_STATICS`` (a ``COLS`` part also at each halving of
    its `col_span`, a ``COORDS`` part also in coordinates), for
    a ``SLICES`` family, its Definition at every axis 1 (the product of its extents of them), for a ``KCHUNKS`` row family, at
    each chunk width k (`chunk_widths`), its first, middle and last chunks (N, N (K / k - 2) and N of them), for a ``CHAINS``
    row family, its cut (`sparse_mla.head_chain`: per head the first visited block at its key count, the later blocks at BN and
    the end; `sparse_mla_split.split_chain`: a split row's at NS = 1, else per head and split its blocks so and its end in
    column blocks, and per head the combine of each column block) and then, at each width G of `values_widths`, the units the
    cut holds with each key block its scores unit and DV / G values units (a Definition two of them hold is listed once: a
    key block's scores unit, the combine of a column block), for a ``STAGES`` row family, its cut (the ``cut`` of the
    module ``STAGE_PLANS`` names: each stage Definition and its units in its ``plan``), for an ``EXPERTS`` row family, at each block of n rows
    (`block_rows`), its
    block family at N = n, the read of a block's rows, ``EmbeddingRead_v1{V=E, H=n K / 2}``, and its ``FUSED`` block reading
    them (`fused_block`; N / n of each), and the read of a G-block's scale row, ``EmbeddingRead_v1{V=E, H=2 K / G}`` (N / G of
    them), and for a ``SLOTS`` row family, its slots family at its statics (one).  None when the spec does not bind to exactly
    its own descriptor id."""
    hit = _UNITS.get(spec)
    if hit is not None:
        return hit
    from verity.circuits.codec import descriptor_id
    from verity.circuits.program import is_input
    from verity_vllm.query.program_view import spec_base, spec_statics

    sp, hit = _bind(spec), ()
    if sp is not None and sp.complete and descriptor_id(sp) == spec:
        hit = ((spec, 1),)
        if spec_base(spec) in CUT:
            hit += tuple((descriptor_id(n.fn), int(n.n) if n.form == "batch" else 1) for n in sp.body.nodes
                         if n.form in ("call", "batch") and not is_input(n.fn))
        n = int(spec_statics(spec).get("N", 1)) if spec_base(spec) in COORDS else 1
        for w in coord_widths(spec_base(spec), spec_statics(spec)) if n > 1 else ():
            if (c := _bind(spec, N=w)) is not None:
                hit += ((descriptor_id(c), n // w),)
        if n > 1 and (c := _bind(spec, N=1)) is not None:
            hit += ((descriptor_id(c), n),)
        axes = SLICES.get(spec_base(spec), ((),))[0]
        n = math.prod(int(spec_statics(spec)[a]) for a in axes)
        if n > 1 and (c := _bind(spec, **dict.fromkeys(axes, 1))) is not None:
            hit += ((descriptor_id(c), n),)
        cuts = [p[0] for c in (PARTS, FINAL) for p in c.get(spec_base(spec), ())]
        for fam in [spec_base(spec), *dict.fromkeys(cuts)]:
            own = fam == spec_base(spec)
            over = {} if own else PART_STATICS.get(spec_base(spec), {})
            if not own and fam not in COL_SPANS and (c := _bind(spec, fam, **over)) is not None:
                hit += ((descriptor_id(c), 1),)
                n = int(spec_statics(spec).get("N", 1)) if fam in COORDS else 1
                for w in coord_widths(fam, {**spec_statics(spec), **over}) if n > 1 else ():
                    if (c := _bind(spec, fam, **over, N=w)) is not None:
                        hit += ((descriptor_id(c), n // w),)
                if n > 1 and (c := _bind(spec, fam, **over, N=1)) is not None:
                    hit += ((descriptor_id(c), n),)
            if (axis := COLS.get(fam)) is not None:
                W, k = int(spec_statics(spec)[col_span(fam)]), 2
                while W % k == 0:
                    c = _bind(spec, None if own else fam, **over, **{axis: W // k})
                    if c is not None and cols_refused(descriptor_id(c)) is None:
                        hit += ((descriptor_id(c), k),)
                    k *= 2
        if (fam := KCHUNKS.get(spec_base(spec))) is not None:
            S = spec_statics(spec)
            for kc in chunk_widths(int(S["K"]), int(S["G"]), int(S["KB"])):
                C = int(S["K"]) // kc
                for first, last, n in ((True, False, 1), (False, False, C - 2), (False, True, 1)):
                    if n and (c := _bind(spec, fam, K=kc, FIRST=first, LAST=last)) is not None:
                        hit += ((descriptor_id(c), n * int(S["N"])),)
        if spec_base(spec) in CHAINS:
            hit += tuple({u: n for _g, units in chain_ways(spec) for u, n in units}.items())
        if spec_base(spec) in STAGES:
            cut = importlib.import_module(STAGE_PLANS[spec_base(spec)]).cut
            hit += tuple((descriptor_id(c), n) for c, n in cut(sp))
        if (fam := EXPERTS.get(spec_base(spec))) is not None:
            from verity.commitments.gates.embedding import read

            S = spec_statics(spec)
            E, K, N, G = (int(S[x]) for x in ("E", "K", "N", "G"))
            for n in block_rows(N, G):
                if (c := _bind(spec, fam, N=n)) is not None:
                    hit += ((descriptor_id(c), N // n), (descriptor_id(read(E, n * K // 2)), N // n))
                if (c := fused_block(spec, n)) is not None:
                    hit += ((descriptor_id(c), N // n),)
            hit += ((descriptor_id(read(E, 2 * K // G)), -(-N // G)),)
        if (fam := SLOTS.get(spec_base(spec))) is not None and (c := _bind(spec, fam)) is not None:
            hit += ((descriptor_id(c), 1),)
        if spec_base(spec) in HISTORY:
            from verity.commitments.gates.embedding import read
            from verity.commitments.history import CAPACITY

            S = spec_statics(spec)
            hit += ((descriptor_id(read(CAPACITY, int(S["H"]))), int(S["TOPK"])),)
    _UNITS[spec] = hit
    return hit


def coordinate(spec: str) -> tuple[Any, int] | None:
    """The unit of one output word of a Program row's Definition (its word view's ``spec``), and the row's output field that
    word is one of: a ``PARTS`` part in ``COORDS`` returning one committed member, at the row's ``PART_STATICS`` and N = 1 (a
    block-FP8 linear's GEMM, whose quantisation is a part of its own), else a ``COORDS`` family at N = 1, else a ``SLICES``
    family of one axis at that axis 1; word c of the field is the unit's one output word at coordinate c. None for a row of
    no such unit, whose own Definition is its unit."""
    from verity_vllm.query.program_view import spec_base

    base = spec_base(spec)
    for fam, _reads, outs in PARTS.get(base, ()):
        if (fam in COORDS and len(outs) == 1 and isinstance(outs[0], int)
                and (c := _bind(spec, fam, **PART_STATICS.get(base, {}), N=1)) is not None):
            return c, outs[0]
    if base in COORDS and (c := _bind(spec, N=1)) is not None:
        return c, 0
    axes = SLICES.get(base, ((),))[0]
    if len(axes) == 1 and (c := _bind(spec, **{axes[0]: 1})) is not None:
        return c, 0
    return None


def fused_block(spec: str, n: int) -> Any:
    """The ``FUSED`` block of an ``EXPERTS`` row ``spec`` over n of its rows: its family at the row's statics, N = n, its ``READ``
    the read of a block's rows (``EmbeddingRead_v1{V=E, H=n K / 2}``) and its ``DOT`` the Boolean version of the row's.  None when
    it does not bind."""
    from verity.circuits.codec import lookup_primitive
    from verity.circuits.defs import REGISTRY
    from verity.commitments.gates.embedding import read
    from verity.definitions import boolean as BB
    from verity_vllm.query.program_view import spec_base, spec_statics

    S = spec_statics(spec)
    try:
        dot = BB.boolean_version(lookup_primitive(REGISTRY, S["DOT"]))
    except (KeyError, TypeError, BB.NoBooleanVersion):
        return None
    return _bind(spec, FUSED[spec_base(spec)], READ=read(int(S["E"]), n * int(S["K"]) // 2), N=n, DOT=dot)


def block_rows(N: int, G: int) -> list[int]:
    """The rows n of an ``EXPERTS`` row's blocks: each power of two that divides N and G, so a block's rows are one G-block's."""
    return [1 << j for j in range(G.bit_length()) if G % (1 << j) == 0 and N % (1 << j) == 0]


def _bind(spec: str, family: str | None = None, **over: Any) -> Any:
    """The registry's Definition bound at ``spec``'s statics (``over`` replacing some), each ``{"fn": id}`` static (`spec_statics`
    reads it as the id) the primitive it names; with ``family``, that family's Definition at those of the statics it has.  None
    when it does not bind.  A caller holds the bound descriptor id to ``spec``."""
    from verity.circuits.codec import lookup_primitive
    from verity.circuits.defs import REGISTRY
    from verity.definitions.registry.catalog import load
    from verity_vllm.query.program_view import spec_base, spec_statics

    def static(v: Any) -> Any:
        try:
            return lookup_primitive(REGISTRY, v) if isinstance(v, str) else v
        except KeyError:
            return v

    reg, _errors = load()
    d = reg.defs.get(family or spec_base(spec))
    if d is None or not hasattr(d, "bind"):
        return None
    statics = {**spec_statics(spec), **over}
    try:
        return d.bind(**{k: static(v) for k, v in statics.items() if family is None or k in d.statics})
    except (TypeError, ValueError, KeyError):
        return None
