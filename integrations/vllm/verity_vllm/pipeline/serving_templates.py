"""The templates of the serving rows (`pipeline.serving_rows`, whose module text states their rules): a template read through its
word link (:func:`template`), and the templates a Program row's Definition holds instances of (:func:`unit_templates`: the row
itself, a ``CUT`` family's batched members, a ``COORDS`` family's coordinates, a ``COLS`` family's column blocks, a ``PARTS``
family's parts, a ``SLICES`` family's slices, a ``KCHUNKS`` row's K-chunks and a ``SPLITS`` row's split parts and combine)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: the row Definitions whose Calls are cut into template instances (their bodies batch the template), the rest being the row itself
CUT = ("RoPE_v1", "Gemm_v1", "ScaledMmS8_v1")
#: the row Definitions whose Calls are split into coordinates of themselves: coordinate n of a row is its Definition at N = 1
#: (the expert-indexed pair GEMMs, `fp8_moe_experts.pair_rows` at one column; FlashInfer's FP8 linear, `deepgemm`, one output
#: word with the row's quantisation of its own; the block-scaled FP8 GEMM, `fp8.ScaledMmFp8BlockSharedScale`, one output word,
#: ``w[n]`` and ``ws[n // G]``), the ``coord`` layout; each family's E and G where its statics have none
COORDS = {"MoeExpertFp8Gemm_v1": {}, "MoeExpertFp8GemmW_v1": {}, "FlashInferFp8Linear_v1": {"E": 1, "G": 128},
          "ScaledMmFp8BlockSharedScale_v1": {"E": 1}}
#: the row Definitions elementwise along the last axis of every parameter and of the output, by that axis's static: a row W wide
#: is W / w units of its Definition at width w (a halving of W), unit j reading and returning columns [jw, jw + w), the ``cols``
#: layout (`moe.MoeSum`'s top-k sum, `tp_allreduce.AllReduceMnnvl`'s all-reduce and its residual add, and torch's multimem
#: all-reduce, `tp_allreduce.AllReduceMultimem`, where `cols_refused` admits its blocks)
COLS = {"MoeSum_v1": "H", "AllReduceMnnvl_v1": "N", "AllReduceResidualMnnvl_v1": "N", "AllReduceMultimem_v1": "SIZE"}
#: the row Definitions that are batches along axes of their statics, which some parameters carry and the others don't: a row of
#: extents a_0, a_1, .. along its axes is a_0 a_1 .. units of its Definition at every axis 1, unit (i_0, i_1, ..) (row-major)
#: reading element i_k along axis k of each parameter that carries it and the others whole, and returning the output's element
#: (i_0, i_1, ..), its leading dimensions being the axes in order (the ``slice`` layout).  Per family: its axes, and per parameter
#: its dimension of each axis (None: it doesn't carry it).  The GEMMs of one token row into W's rows (vLLM's router GEMM, the LM
#: head, cuBLAS's split-K), torch.bmm into each member's coordinates, DSA's logits into keys, the vocabulary shard's embedding
#: into columns (every unit reads the token and gathers its column) and the all-gather into its ranks' shards; cuBLAS's split-K
#: at an f32 output too (the MoE router gate)
SLICES = {"LlBf16DotprodGemm_v1": (("N",), ((None,), (0,))), "Gemm_v2": (("N",), ((None,), (0,))),
          "CublasSplitKGemm_v1": (("N",), ((None,), (0,))), "BatchedGemm_v1": (("B", "N"), ((0, None), (0, 1))),
          "DsaMqaLogits_v3": (("N",), ((None,), (None,), (0,), (0,))), "EmbeddingShard_v1": (("H",), ((None,), (1,))),
          "AllGather_v1": (("WORLD",), ((0,),)), "CublasSplitKGemmF32Out_v1": (("N",), ((None,), (0,)))}
#: the row Definitions whose Calls are split into parts, each a Definition of its own at the row's statics it names: per part, its
#: family, what it reads in order (``("arg", g)`` the row's argument group g, ``("out", j)`` the row's committed output member j)
#: and the output members it returns; together they return every output once, and the row computes what they compose to
#: (tests/program/test_tp_allreduce_parts.py).  A ``COLS`` part is read in column blocks, a ``COORDS`` part in coordinates, any
#: other one whole (the ``part`` layout).  `tp_allreduce.AllReduceRmsNormMnnvl`: its all-reduce and residual add (r), then its
#: norm of the committed r.  `deepgemm.DeepGemmFp8Linear`: its quantisation of x (q, s), then its GEMM of the committed q and s
#: (tests/program/test_deepgemm_parts.py).
PARTS = {"AllReduceRmsNormMnnvl_v1": (("AllReduceResidualMnnvl_v1", (("arg", 0), ("arg", 1)), (1,)),
                                      ("RmsNormMnnvl_v1", (("out", 1), ("arg", 2)), (0,))),
         "DeepGemmFp8Linear_v1": (("Fp8GroupQuantUe8m0_v1", (("arg", 0),), (0, 1)),
                                  ("ScaledMmFp8BlockSharedScale_v1", (("out", 0), ("out", 1), ("arg", 1), ("arg", 2)), (2,)))}
#: the statics a ``PARTS`` row family's parts take that the row has none of: DeepGEMM's linear quantises and scales in
#: 128-wide blocks (`deepgemm.FP8_BLOCK`)
PART_STATICS = {"DeepGemmFp8Linear_v1": {"G": 128}}
PART_OF = {p[0]: (row, p) for row, ps in PARTS.items() for p in ps}
#: the chunk families that cut a ``COORDS`` family's coordinate along K (the ``kchunk`` layout, `kchunk_rows`), by the row family:
#: a coordinate of K weights is C = K / k chunks of k (whole scale blocks and tiles), chunk 0 FIRST, chunk C - 1 LAST, each
#: continuing the one before's f32 accumulator (`fp8_moe_experts.MoeExpertFp8CoordinateChunk_v1`, the w13 launch)
KCHUNKS = {"MoeExpertFp8Gemm_v1": "MoeExpertFp8CoordinateChunk_v1"}
#: the row Definitions whose heads are cut along their keys into split parts and a combine (the ``split`` layout, `split_rows`), by
#: the row family: its part family, one split's (o_partial, lse) at each split's key count, and its combine family over the row's
#: non-empty splits (`sparse_mla_split.SparseMlaSplitPart_v1` and `SparseMlaCombine_v1`: FA3's split kernel and its combine)
SPLITS = {"SparseMlaSplit_v1": ("SparseMlaSplitPart_v1", "SparseMlaCombine_v1")}
SPLIT_OF = {f: row for row, fs in SPLITS.items() for f in fs}
#: a member's layout by the family of its template's word; any other template is the row itself
LAYOUTS = {"RoPEHead_v1": "rope", "GemmCoordinate_v1": "gemm", "ScaledMmS8Coordinate_v1": "s8"}
#: a row template's element widths whose ports the rows commit as their little-endian bytes
PACKED = frozenset((8, 16, 32, 64))


@dataclass(frozen=True)
class Template:
    """A partition's template as the rows are read for it: its descriptor id and Definition (the population program's), the id of
    the word Definition it computes (its own for a word Definition), and its member layout (``LAYOUTS``; ``"coord"`` for a
    ``COORDS`` family at N = 1, ``"cols"`` for a ``COLS`` family, ``"part"`` for any other ``PARTS`` part, ``"slice"`` for a
    ``SLICES`` family at every axis 1, ``"kchunk"`` for a ``KCHUNKS`` chunk family, ``"split"`` for a ``SPLITS`` part or combine
    family; else ``"row"``).
    ``packed``: a row, coordinate, slice, part, K-chunk or split template with a port that is not 16-bit elements, each port's
    element bits (its parameters', then its return fields'); its ports are committed as their elements' bytes."""
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
    width)."""
    hit = _TEMPLATES.get(t)
    if hit is not None:
        return hit
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.definitions import boolean as BB
    from verity.ml.definitions.call_scope import specialization
    from verity_vllm.query.program_view import spec_base, spec_statics

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
    layout = ("coord" if spec_base(word) in COORDS and int(spec_statics(word)["N"]) == 1 else "cols" if spec_base(word) in COLS
              else "part" if spec_base(word) in PART_OF else "slice" if sl and all(int(spec_statics(word)[a]) == 1 for a in sl[0])
              else "kchunk" if spec_base(word) in KCHUNKS.values() else "split" if spec_base(word) in SPLIT_OF
              else LAYOUTS.get(spec_base(word), "row"))
    packed = None
    if widths != {16} and layout != "s8":
        ports = [*view[:-1], *(getattr(view[-1], "fields", None) or (view[-1],))]
        elems = [ty.width_set() for ty in ports]
        if layout not in ("row", "coord", "slice", "part", "kchunk", "split") or any(
                len(e) != 1 or not e <= PACKED or sum(ty.leaf_widths()) % 16 for e, ty in zip(elems, ports)):
            raise ValueError(f"template {t}: the serving rows commit u16 words, and its ports' {'elements' if w is not fn else 'words'} "
                             f"are {sorted(widths)} bits (a row template's port may be elements of one of {sorted(PACKED)} bits, "
                             f"whole u16 words)")
        packed = tuple(next(iter(e)) for e in elems)
    hit = _TEMPLATES[t] = Template(t, fn, word, layout, packed)
    return hit


def cols_refused(word: str) -> str | None:
    """Why ``word``'s ``COLS`` column blocks are not the columns of its family's rows, or None (a family elementwise at every
    statics).  torch's multimem all-reduce (`tp_allreduce.AllReduceMultimem`) reduces each owner rank's slice of its SIZE
    elements through that owner's tree, and a block's owners are not the row's: a block computes the row's columns only when
    every owner's tree is one, and is a Definition only as whole 16-byte vectors (`tp_allreduce.multimem_owners`)."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    if spec_base(word) != "AllReduceMultimem_v1":
        return None
    from verity.ml.definitions.registry import tp_allreduce as TA

    S = spec_statics(word)
    try:
        world = TA.ranks(S["WORLD"])
        trees = TA.multimem_trees(S["TREES"], world)
        TA.multimem_owners(world, int(S["SIZE"]))
    except (KeyError, TypeError, ValueError) as e:
        return f"{word}: {type(e).__name__}: {e}"
    if len(set(trees)) > 1:
        return (f"{word}: its owners' TREES are {len(set(trees))} trees, and a column block's elements are reduced through "
                f"other owners' trees than the row's")
    return None


def part(word: str, spec: str) -> tuple | None:
    """The ``PARTS`` entry of template word ``word`` on a row of ``spec`` when ``word``'s family is a part of the row's; else None
    (the row is ``word``'s own family)."""
    from verity_vllm.query.program_view import spec_base

    hit = PART_OF.get(spec_base(word))
    return hit[1] if hit is not None and hit[0] == spec_base(spec) else None


def chunk_widths(K: int, G: int, KB: int) -> list[int]:
    """The K-chunk widths of a ``KCHUNKS`` coordinate of K weights: each k < K that cuts it into whole scale blocks and tiles."""
    return [K // c for c in range(2, K // G + 1) if K % c == 0 and (K // c) % G == 0 and (K // c) % KB == 0]


_UNITS: dict[str, tuple[tuple[str, int], ...]] = {}


def unit_templates(spec: str) -> tuple[tuple[str, int], ...]:
    """The word Definitions a Program row's Definition (its word view's ``spec``) holds instances of, ``(descriptor id, instances)``:
    the row itself, then, for a ``CUT`` family, each Definition its body batches (a batch's members; Input batches are its
    parameters), for a ``COORDS`` family of N > 1, its coordinate (its Definition at N = 1, N of them), for a ``COLS`` family,
    its Definition at each halving W / k of its width (k of them) that `cols_refused` admits, for a ``PARTS`` family, each part
    at the row's statics and its ``PART_STATICS`` (a ``COLS`` part also at each halving, a ``COORDS`` part also at N = 1), for
    a ``SLICES`` family, its Definition at every axis 1 (the product of its extents of them), for a ``KCHUNKS`` row family, at
    each chunk width k (`chunk_widths`), its first, middle and last chunks (N, N (K / k - 2) and N of them), and for a
    ``SPLITS`` row family in NS > 1 splits, its part at each split's key count k (`split_keys`; NH times the splits of k keys)
    and its combine over the non-empty splits (NH).  None when the
    spec does not bind to exactly its own descriptor id."""
    hit = _UNITS.get(spec)
    if hit is not None:
        return hit
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.program import is_input
    from verity_vllm.query.program_view import spec_base, spec_statics

    sp, hit = _bind(spec), ()
    if sp is not None and sp.complete and descriptor_id(sp) == spec:
        hit = ((spec, 1),)
        if spec_base(spec) in CUT:
            hit += tuple((descriptor_id(n.fn), int(n.n) if n.form == "batch" else 1) for n in sp.body.nodes
                         if n.form in ("call", "batch") and not is_input(n.fn))
        n = int(spec_statics(spec).get("N", 1)) if spec_base(spec) in COORDS else 1
        if n > 1 and (c := _bind(spec, N=1)) is not None:
            hit += ((descriptor_id(c), n),)
        axes = SLICES.get(spec_base(spec), ((),))[0]
        n = math.prod(int(spec_statics(spec)[a]) for a in axes)
        if n > 1 and (c := _bind(spec, **dict.fromkeys(axes, 1))) is not None:
            hit += ((descriptor_id(c), n),)
        for fam in [spec_base(spec)] + [p[0] for p in PARTS.get(spec_base(spec), ())]:
            own = fam == spec_base(spec)
            over = {} if own else PART_STATICS.get(spec_base(spec), {})
            if not own and (c := _bind(spec, fam, **over)) is not None:
                hit += ((descriptor_id(c), 1),)
                n = int(spec_statics(spec).get("N", 1)) if fam in COORDS else 1
                if n > 1 and (c := _bind(spec, fam, **over, N=1)) is not None:
                    hit += ((descriptor_id(c), n),)
            if (axis := COLS.get(fam)) is not None:
                W, k = int(spec_statics(spec)[axis]), 2
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
        if (fams := SPLITS.get(spec_base(spec))) is not None and int(spec_statics(spec)["NS"]) > 1:
            S = spec_statics(spec)
            ks = split_keys(int(S["T"]), int(S["NS"]), int(S["BN"]))
            for k in dict.fromkeys(ks):
                if (c := _bind(spec, fams[0], T=k)) is not None:
                    hit += ((descriptor_id(c), ks.count(k) * int(S["NH"])),)
            if (c := _bind(spec, fams[1], NSPLIT=len(ks))) is not None:
                hit += ((descriptor_id(c), int(S["NH"])),)
    _UNITS[spec] = hit
    return hit


def split_keys(T: int, NS: int, BN: int) -> list[int]:
    """Each non-empty split's key count, in split order, of a ``SPLITS`` row of T keys in NS splits of BN-key blocks
    (`sparse_mla_split.split_blocks`: only the token's last block is partial)."""
    from verity.ml.definitions.registry.sparse_mla_split import split_blocks

    return [min(T, b1 * BN) - b0 * BN for b0, b1 in split_blocks(T, NS, BN)]


def _bind(spec: str, family: str | None = None, **over: Any) -> Any:
    """The registry's Definition bound at ``spec``'s statics (``over`` replacing some), each ``{"fn": id}`` static (`spec_statics`
    reads it as the id) the primitive it names; with ``family``, that family's Definition at those of the statics it has.  None
    when it does not bind.  A caller holds the bound descriptor id to ``spec``."""
    from verity.ml.circuits.codec import lookup_primitive
    from verity.ml.circuits.defs import REGISTRY
    from verity.ml.definitions.registry.catalog import load
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
