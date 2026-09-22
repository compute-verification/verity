"""Export one captured GEMM-coordinate verification unit as a ``typed-obligation/v0`` object.

This is the adapter's side of the Track A / Track B interface
(``docs/interface/typed-obligation-v0.md``): it reads what the serving lane
persists for one authenticated GEMM operation -- the fixture layout of
``fixtures/<bundle>/<family>_<phase>_<n>/`` (``fixture.json``,
``openings.json``) plus the bundle's ``commitments.json`` -- authenticates the
three openings under Track A's own trees, and emits the object Track B checks:

* **roots**: the two trees the openings live in (the request's replay-unit
  tree and the weights tree), by name, with the tree's ``ids_digest`` as the
  interface's ``domain`` and the framing ``verity-vllm/leaf/v1``;
* **leaves**: the three committed leaves (the activation row, the weight row,
  the output row), by Track A's canonical id, with the layout of the words
  packed into each (``bfloat16 [1536]`` -> ``Array<1536, Value<16>>``): a
  commitment-position matter, not a value;
* **references**: one per word the obligation touches -- ``x[k]``, ``W[j][k]``,
  ``out[j]`` -- each the word's canonical reference (:func:`word_reference`)
  and where it is committed (leaf, offset).  Until the typed B1 program (TA1)
  gives every word its producing gate's identity, Track A spells the reference
  of the ``k``-th word of a tensor port as ``<port id>[k]``: a member of the
  static collection of references the port denotes (IR §B.3), never "a
  component of a gate output";
* **subcircuit**: :func:`veritor.interface.gemm_coordinate` over ``K`` in
  ``veritor.tc-ampere-bf16@2`` (``TC_FAMILY``; ``@1`` until lane L8 rebound the
  store cast to the hardware NaN word) -- ``GemmCoordinate<K>``, a subcircuit of the
  registered composite ``Gemm`` over ``tc_dot`` (IR §B.7): the ``mma.sync``
  chain the numerics lane validated
  (``verity_vllm_numerics.relations.independent_gemm_coordinate``), the body
  W3c's ``DagBuilder.gemm_coordinate`` builds;
* **obligation**: the graph boundary -- ``boundary.in`` the 2K incoming words
  ``x[k]``, ``W[j][k]``, ``boundary.out`` the one produced word ``out[j]`` (IR
  §D.1) -- and the declared call interface ``GemmCoordinate(x, W[j]) ->
  out[j]`` flattened to words (IR §D.2); the binding derived from the
  request/profile/token identity; the ``target`` encoded as the function-call
  instance ``<vu canonical id>/Gemm[j]``.  The ``W[j][k]`` words are prescribed
  inputs: their ``Input`` gates (IR §E) are *outside* this subcircuit -- this
  plan does not sample them here, so they are incoming values, authenticated
  and not pinned; a plan that samples them puts ``Input`` gates with the
  registered checkpoint's words into the body, and the exporter would need the
  checkpoint, not the commitment, to write them.  No field assumes the weights
  correct.  The legacy backend statement (``KindProgram``, slots, pins) is
  derived by ``veritor.interface.lowering.lower`` and never appears here.

The object is *accepted* by every interpreter and its relation holds on the
committed words (``relations = accept``).  Since the packed-word decision (IR
§H.4; ``docs/interface/packed-word-authentication.md``) it is also
**authenticatable**: ``veritor.interface.lowering`` lowers every reference to a
``packed/v1:verity-vllm/leaf/v1:u16le:<count>:<index>`` position -- the leaf's
rank in Track A's tree and the word's row-major index in the row -- and
:func:`packed_witness` turns the fixture's three openings into a **format-3
witness** (``veritor.protocol.proofs.packed``): each whole row leaf with its
Track A authentication path, once, in the statement's tree order.  The generic
guest hashes the leaf exactly as Track A committed it, folds the path to the
committed root and reads each word at the statement-fixed index.  The record
carries the witness and the reference checker's verdict on it.

Only pure Python and numpy; nothing from torch or vLLM.  ``veritor.interface``
never imports this module (the dependency runs one way).

    PYTHONPATH=. python -m verity_vllm.typed_obligation_export \\
        fixtures/B1-.../qkv_proj_decode_0 --out fixtures/typed-obligation-v0/stage2--qkv-proj-decode-0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from veritor.core import tagged_sha256
from veritor.interface import (
    FORMAT,
    VLLM_FRAMING,
    to_statement,
    Array,
    Binding,
    Bits,
    Boundary,
    Family,
    DeclaredInterface,
    Obligation,
    Reference,
    Root,
    Target,
    Type,
    TypedObligationSet,
    Leaf,
    family_gate_set,
    gemm_coordinate,
)
from veritor.interface.programs import input_gate, make_subcircuit
from veritor.interface.typed_obligation import Definition

from veritor.protocol.proofs import Statement
from veritor.protocol.proofs.packed import (
    PackedLeaf,
    PackedWitness,
    encode_packed_witness,
    packed_table,
)

from . import hashing, identity
from .commit import Opening, fold_path, path_shape, verify_opening_in

__all__ = ["GemmCoordinateFixture", "load_fixture", "export_gemm_coordinate", "packed_witness", "packed_witness_of",
           "CoordinateUnit", "InputEquation", "ComposedExport", "export_composed", "composed_record", "input_equation_subcircuit", "main"]

#: The production family binding (lane L8, `docs/verified-backend/l8-nan-primitive.md`): version `@2` of
#: `veritor.tc-ampere-bf16` -- `f32_to_bf16` with the hardware NaN word `0x7FFF` that the Triton epilogue's
#: `cvt.rn.bf16.f32` produces (L6, measured on the A100 and the RTX 4090).  `@1` (NaN -> `0x7FC0`) stays registered;
#: the checked-in Stage-2 record `stage2--qkv-proj-decode-0-k1536.json` was exported under `@1` and is unchanged
#: (its committed words are finite, on which the two versions are equal).
TC_FAMILY = "veritor.tc-ampere-bf16@2"
TC_WIDTH = 32
WEIGHTS_TREE = "weights"


class ExportError(ValueError):
    pass


@dataclass(frozen=True)
class GemmCoordinateFixture:
    """One persisted authenticated GEMM operation (the fixture layout of ``vllm-poc/fixtures``)."""

    directory: Path
    fixture: dict[str, Any]
    openings: dict[str, Opening]
    commitments: dict[str, Any]

    @property
    def K(self) -> int:  # noqa: N802
        return int(self.fixture["K"])

    @property
    def j(self) -> int:
        return int(self.fixture["j"])


def load_fixture(directory: Path, commitments: Path | None = None) -> GemmCoordinateFixture:
    directory = Path(directory)
    commitments = commitments or directory.parent / "commitments.json"
    fixture = json.loads((directory / "fixture.json").read_text())
    openings_json = json.loads((directory / "openings.json").read_text())
    openings = {name: Opening.from_json(doc) for name, doc in openings_json.items()}
    for name in ("x", "w_row", "out_row"):
        if name not in openings:
            raise ExportError(f"{directory}: openings.json lacks {name!r}")
    return GemmCoordinateFixture(directory, fixture, openings, json.loads(commitments.read_text()))


def _words(opening: Opening) -> list[int]:
    if opening.dtype != "bfloat16":
        raise ExportError(f"{opening.canonical_id}: dtype {opening.dtype!r}, the coordinate gate takes bfloat16")
    return np.frombuffer(opening.value, dtype="<u2").astype(int).tolist()


#: Track A dtypes the exporter lays out as ``Value<16>`` words (little-endian): the BF16 activations and
#: weights are one word per element; the boundary's int32 tokens / float32 logprobs are *two* words per
#: element (low half first), so the family's 16-bit gates can pin them -- both halves are pinned, which
#: fixes the 32-bit value exactly.  The leaf hash is over the raw bytes either way (the witness carries
#: Track A's dtype/shape text verbatim), so the layout is the interface's business, not the commitment's.
WORDS_PER_ELEMENT = {"bfloat16": 1, "float16": 1, "int32": 2, "float32": 2}
WORD_BITS = 16


def words_per_element(dtype: str) -> int:
    n = WORDS_PER_ELEMENT.get(dtype)
    if n is None:
        raise ExportError(f"dtype {dtype!r} is not laid out as Value<16> words here")
    return n


def leaf_words(opening: Opening) -> list[int]:
    """Every 16-bit word of a leaf as an unsigned int, row-major (an int32 element is its two halves, low first)."""
    words_per_element(opening.dtype)
    return np.frombuffer(opening.value, dtype="<u2").astype(int).tolist()


def _type_of(opening: Opening) -> Type:
    ty: Type = Bits(WORD_BITS)
    n = words_per_element(opening.dtype)
    if n > 1:
        ty = Array(ty, n)
    for extent in reversed(opening.shape):
        ty = Array(ty, int(extent))
    return ty


def word_reference(port_id: str, offset: tuple[int, ...]) -> str:
    """Track A's provisional canonical reference of one word of a tensor port.

    ``port_id`` is the canonical id of the port (``p=.../op=.../port=0``, or a
    prescribed weight row ``w=.../c=<row>``); the port denotes a static
    collection of value references (IR §B.3) and ``offset`` selects one member.
    ``<port id>[k]`` is that member.  When the typed B1 program exists, the
    reference becomes the producing gate's identity and this spelling retires.
    """

    return port_id + "".join(f"[{k}]" for k in offset)


def _owner_of(tree: str) -> int:
    """Track B's owner of a Track A tree: ``-2`` for the weights, the replay-unit ordinal for ``ru/<request>/<n>``."""

    if tree == WEIGHTS_TREE:
        return -2
    parts = tree.split("/")
    if len(parts) == 3 and parts[0] == "ru":
        return int(parts[2])
    if parts[0] == "boundary":
        return -1
    raise ExportError(f"tree {tree!r}: no owner mapping")


def _h(tag: str, text: str) -> str:
    return str(tagged_sha256(f"verity-vllm/typed-obligation/{tag}", text.encode("utf-8")))


def authenticate_opening(commitments: dict[str, Any], opening: Opening) -> str:
    """How ``opening`` authenticates against ``commitments.json``; raises :class:`ExportError` if it does not.

    Returns ``"verify_opening_in"`` when this checkout's full verifier accepts
    it.  A weights-v2 row-block id (``w=<param>/c=<row>``, capture format
    section 11) is rejected by a v1 ``verity_vllm.identity`` grammar *before*
    any hash is checked; for such an opening the hash chain is checked here
    directly (path shape, leaf hash, fold to the recorded root) and the
    returned label says so -- the identity grammar is a drift point between
    Track A's versions, not a property of the committed bytes.
    """

    if verify_opening_in(commitments, opening):
        return "verify_opening_in"
    record = commitments["trees"].get(opening.tree)
    if record is None:
        raise ExportError(f"{opening.tree!r} is not in commitments.json")
    root, count = bytes.fromhex(record["root"]), int(record["leaf_count"])
    if opening.root != root or opening.leaf_count != count or not 0 <= opening.index < count:
        raise ExportError(f"{opening.canonical_id}: root, count or index disagree with commitments.json")
    expected = path_shape(count, opening.index)
    if len(opening.path) != len(expected) or any(
        (sibling is None) == has_sibling for sibling, has_sibling in zip(opening.path, expected, strict=True)
    ):
        raise ExportError(f"{opening.canonical_id}: authentication path has the wrong shape")
    if fold_path(opening.leaf(), opening.index, opening.path) != root:
        raise ExportError(f"{opening.canonical_id}: does not fold to the committed root")
    try:
        identity.parse(opening.canonical_id)
    except identity.IdentityError as error:
        return f"hash-chain (id outside this checkout's identity grammar: {error})"
    raise ExportError(f"{opening.canonical_id}: verify_opening_in rejects an opening whose hash chain holds")


def export_gemm_coordinate(
    fx: GemmCoordinateFixture, *, authenticate: bool = True
) -> tuple[TypedObligationSet, list[int], dict[str, str]]:
    """The ``typed-obligation/v0`` object of ``fx``, the decoded words of its positions, and how each opening authenticated.

    With ``authenticate`` (the default) every opening is first verified
    against ``commitments.json`` under Track A's framing
    (:func:`authenticate_opening`); a fixture whose openings do not
    authenticate is refused, since the object would then bind words nobody
    committed to.
    """

    x, w, out = fx.openings["x"], fx.openings["w_row"], fx.openings["out_row"]
    how: dict[str, str] = {}
    if authenticate:
        for name, opening in (("x", x), ("w_row", w), ("out_row", out)):
            how[name] = authenticate_opening(fx.commitments, opening)
    K, j = fx.K, fx.j
    x_words, w_words, out_words = _words(x), _words(w), _words(out)
    cols = fx.fixture.get("x_arg_cols")
    x_cols = list(range(K)) if cols is None else [int(c) for c in cols]
    if len(x_cols) != K:
        raise ExportError(f"{fx.directory}: {len(x_cols)} consumed x columns for K={K}")
    if w.shape not in ((K,), (1, K)):
        raise ExportError(f"{fx.directory}: weight opening shape {w.shape} is not one row of K={K}")
    w_path_prefix: tuple[int, ...] = () if w.shape == (K,) else (0,)
    if not 0 <= j < len(out_words):
        raise ExportError(f"{fx.directory}: j={j} outside the output row")

    gate_set = family_gate_set(TC_FAMILY, TC_WIDTH)
    assert gate_set is not None
    family = Family(TC_FAMILY, str(gate_set.digest), TC_WIDTH)

    roots: dict[str, Root] = {}
    for opening in (x, w, out):
        record = fx.commitments["trees"][opening.tree]
        if bytes.fromhex(record["root"]) != opening.root or int(record["leaf_count"]) != opening.leaf_count:
            raise ExportError(f"{opening.tree}: opening root/count disagree with commitments.json")
        roots[opening.tree] = Root(
            opening.tree,
            _owner_of(opening.tree),
            record["ids_digest"],
            opening.root.hex(),
            opening.leaf_count,
            VLLM_FRAMING,
        )
    leaves = [
        Leaf(opening.canonical_id, opening.tree, opening.index, opening.index, _type_of(opening))
        for opening in (x, w, out)
    ]

    program = gemm_coordinate(K)
    references = (
        [Reference(word_reference(x.canonical_id, (k,)), x.canonical_id, (k,)) for k in x_cols]
        + [Reference(word_reference(w.canonical_id, (*w_path_prefix, k)), w.canonical_id, (*w_path_prefix, k)) for k in range(K)]
        + [Reference(word_reference(out.canonical_id, (j,)), out.canonical_id, (j,))]
    )
    incoming = tuple(r.ref for r in references[: 2 * K])
    outgoing = (references[2 * K].ref,)
    cid = identity.parse(fx.fixture["canonical_id"])
    ru_ordinal = _owner_of(out.tree)
    binding = Binding(
        session=_h("session", str(cid.request_id)),
        compiled=_h("compiled", str(cid.profile_id)),
        unit=int(cid.t if cid.t is not None else 0),
        replay_unit=max(ru_ordinal, 0),
    )
    obligation = Obligation(
        binding,
        program.digest,
        tuple(sorted(roots)),
        Boundary(incoming, outgoing),
        # GemmCoordinate(x, W[j]) -> out[j]: the declared call interface, flattened to words
        DeclaredInterface(incoming, outgoing),
        Target("call", f"{fx.fixture['canonical_id']}/Gemm[{j}]"),
    )
    obj = TypedObligationSet(
        family,
        (),
        tuple(roots[t] for t in sorted(roots)),
        tuple(sorted(leaves, key=lambda v: v.id)),
        tuple(sorted(references, key=lambda r: r.ref)),
        (program,),
        (obligation,),
    )
    row = [x_words[k] for k in x_cols] + w_words + [out_words[j]]
    return obj, row, how


# --- composed export: several coordinates sharing authenticated values, plus Input equations ---


@dataclass(frozen=True)
class CoordinateUnit:
    """One GEMM-coordinate verification unit ``out[t, j]`` of a captured bundle, by its openings.

    ``canonical_id`` is the committed output row (the VU the challenge sampled), ``j`` the
    coordinate, ``x``/``w_row``/``out_row`` the three Track A openings (the producer's row at
    ``t``, the weight row leaf ``w=<param>/c=<j>`` of a weights-v2 bundle, the output row),
    ``x_cols`` W1's ``arg_cols`` slice of ``x`` when the producer is wider than ``K``.
    """

    canonical_id: str
    j: int
    x: Opening
    w_row: Opening
    out_row: Opening
    x_cols: Sequence[int] | None = None

    @property
    def K(self) -> int:  # noqa: N802
        return int(np.frombuffer(self.w_row.value, dtype="<u2").size)

    @property
    def target(self) -> str:
        return f"{self.canonical_id}/Gemm[{self.j}]"


@dataclass(frozen=True)
class InputEquation:
    """A sampled ``Input`` equation (IR §E): the committed word at ``offset`` of ``leaf`` must equal
    ``expected`` -- the registered checkpoint's word for a weight, the request's token for a prompt.
    ``expected`` is the 16-bit encoding as an int; ``source`` says where it came from (informative)."""

    leaf: Opening
    offset: tuple[int, ...]
    expected: int
    source: str = ""

    @property
    def ref(self) -> str:
        return word_reference(self.leaf.canonical_id, self.offset)


@dataclass(frozen=True)
class ComposedExport:
    """What :func:`export_composed` returns: the object, the boundary words per obligation (in the
    object's obligation order), and the openings the witness is built from, plus a composition report."""

    obj: TypedObligationSet
    values: list[list[int]]
    openings: dict[str, Opening]
    composition: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Pending:
    session: str
    target: str
    program: Any
    incoming: tuple[str, ...]
    outgoing: tuple[str, ...]
    roots: tuple[str, ...]
    row: list[int]
    replay_unit: int


def _binding(session_text: str, profile_id: str, unit: int, replay_unit: int) -> Binding:
    return Binding(session=_h("session", session_text), compiled=_h("compiled", profile_id), unit=unit, replay_unit=replay_unit)


def input_equation_subcircuit(expected: int, width: int = 16):
    """``InputEq<W>[v]``: the body of one sampled input equation -- one ``Input<W>`` gate with prescribed
    value ``v`` on the outgoing boundary (IR §E; the lowering pins the committed position to ``v``).
    ``W`` is the family's 16 (a 32-bit boundary token is pinned as its two halves)."""

    if not 0 <= int(expected) < (1 << width):
        raise ExportError(f"input equation value {expected} is not a {width}-bit word")
    return make_subcircuit([], [input_gate(width, int(expected))], [0], Definition("InputEq", (("W", width),)))


def export_composed(
    commitments: dict[str, Any],
    units: Sequence[CoordinateUnit],
    inputs: Sequence[InputEquation] = (),
    *,
    profile_id: str,
    authenticate: bool = True,
) -> ComposedExport:
    """One ``typed-obligation/v0`` object over several verification units that **share authenticated values**.

    Every coordinate unit becomes a ``GemmCoordinate<K>`` obligation; every input equation an
    ``InputEq`` obligation whose one outgoing value is the committed word it pins.  Leaves and
    references are deduplicated by identity, so two coordinates of one output row name the *same*
    ``x[k]`` references (one authenticated row read by both), two coordinates with the same ``j`` of
    one GEMM name the same ``W[j][k]`` references, and a pinned weight word is at once the outgoing
    value of its ``InputEq`` obligation and an incoming value of the coordinate that consumes it (the
    interface's one-producer rule, E_DUPLICATE_PRODUCER, applies: an input equation is stated once).

    Obligations are keyed by ``(session, compiled, unit)``: ``session`` is the request's (coordinates)
    or the weights tree's (input equations), ``compiled`` the profile's, ``unit`` an ordinal in the
    canonical order of the targets within the session (the exporter's convention for a set; the target
    carries the function-call identity).  The relation is not checked here; the caller does that
    with ``veritor.interface.check_relations`` on ``values`` or with the transparent backend on the witness.
    """

    gate_set = family_gate_set(TC_FAMILY, TC_WIDTH)
    assert gate_set is not None
    family = Family(TC_FAMILY, str(gate_set.digest), TC_WIDTH)
    openings: dict[str, Opening] = {}
    how: dict[str, str] = {}

    def take(opening: Opening) -> Opening:
        prior = openings.get(opening.canonical_id)
        if prior is not None:
            if prior.value != opening.value or prior.index != opening.index or prior.tree != opening.tree:
                raise ExportError(f"{opening.canonical_id}: two different openings of one leaf")
            return prior
        if authenticate:
            how[opening.canonical_id] = authenticate_opening(commitments, opening)
        openings[opening.canonical_id] = opening
        return opening

    roots: dict[str, Root] = {}

    def root_of(opening: Opening) -> str:
        if opening.tree not in roots:
            record = commitments["trees"][opening.tree]
            if bytes.fromhex(record["root"]) != opening.root or int(record["leaf_count"]) != opening.leaf_count:
                raise ExportError(f"{opening.tree}: opening root/count disagree with commitments.json")
            roots[opening.tree] = Root(opening.tree, _owner_of(opening.tree), record["ids_digest"], opening.root.hex(), opening.leaf_count, VLLM_FRAMING)
        return opening.tree

    references: dict[str, Reference] = {}

    def ref_of(opening: Opening, offset: tuple[int, ...]) -> str:
        rid = word_reference(opening.canonical_id, offset)
        if rid not in references:
            references[rid] = Reference(rid, opening.canonical_id, offset)
        return rid

    subcircuits: dict[str, Any] = {}
    pending: list[_Pending] = []
    shared_x: dict[str, list[str]] = {}
    shared_w: dict[str, list[str]] = {}
    for u in units:
        x, w, out = take(u.x), take(u.w_row), take(u.out_row)
        K, j = u.K, int(u.j)
        x_words, w_words, out_words = _words(x), _words(w), _words(out)
        cols = list(range(K)) if u.x_cols is None else [int(c) for c in u.x_cols]
        if len(cols) != K:
            raise ExportError(f"{u.target}: {len(cols)} consumed x columns for K={K}")
        if w.shape not in ((K,), (1, K)):
            raise ExportError(f"{u.target}: weight opening shape {w.shape} is not one row of K={K}")
        prefix: tuple[int, ...] = () if w.shape == (K,) else (0,)
        if not 0 <= j < len(out_words):
            raise ExportError(f"{u.target}: j={j} outside the output row")
        program = gemm_coordinate(K)
        subcircuits[program.digest] = program
        # x is the producer's whole row leaf, possibly [heads, D]; the GEMM consumes it flattened
        x_shape = tuple(int(e) for e in x.shape)
        x_offsets = [tuple(int(i) for i in np.unravel_index(k, x_shape)) for k in cols]
        incoming = tuple([ref_of(x, off) for off in x_offsets] + [ref_of(w, (*prefix, k)) for k in range(K)])
        outgoing = (ref_of(out, (j,)),)
        session = str(identity.parse(u.canonical_id).request_id)
        pending.append(_Pending(session, u.target, program, incoming, outgoing, tuple(sorted({root_of(x), root_of(w), root_of(out)})),
                                [x_words[k] for k in cols] + w_words + [out_words[j]], max(_owner_of(out.tree), 0)))
        shared_x.setdefault(x.canonical_id, []).append(u.target)
        shared_w.setdefault(w.canonical_id, []).append(u.target)
    pinned: dict[str, str] = {}
    for eq in inputs:
        leaf = take(eq.leaf)
        rid = ref_of(leaf, tuple(eq.offset))
        if rid in pinned:
            raise ExportError(f"{rid}: pinned twice")
        program = input_equation_subcircuit(int(eq.expected), WORD_BITS)
        subcircuits[program.digest] = program
        words = leaf_words(leaf)
        extents = [int(e) for e in leaf.shape] + ([words_per_element(leaf.dtype)] if words_per_element(leaf.dtype) > 1 else [])
        index = 0
        for extent, step in zip(extents, eq.offset, strict=True):
            index = index * int(extent) + int(step)
        if not 0 <= index < len(words):
            raise ExportError(f"{rid}: offset outside the leaf")
        session = "weights" if leaf.tree == WEIGHTS_TREE else leaf.tree
        pending.append(_Pending(session, f"{rid}/InputEq", program, (), (rid,), (root_of(leaf),), [words[index]], max(_owner_of(leaf.tree), 0)))
        pinned[rid] = eq.source
    # obligations: ordinal unit per session in canonical target order, then the interface's sort
    ordinals: dict[str, int] = {}
    built: list[tuple[Obligation, list[int], str]] = []
    for p in sorted(pending, key=lambda p: (p.session, p.target)):
        unit = ordinals.get(p.session, 0)
        ordinals[p.session] = unit + 1
        binding = _binding(p.session, profile_id, unit, p.replay_unit)
        di = DeclaredInterface(p.incoming, p.outgoing) if p.incoming else None
        built.append((Obligation(binding, p.program.digest, p.roots, Boundary(p.incoming, p.outgoing), di, Target("call", p.target)), p.row, p.target))
    built.sort(key=lambda b: (b[0].binding.session, b[0].binding.compiled, b[0].binding.unit))
    leaves = [Leaf(o.canonical_id, o.tree, o.index, o.index, _type_of(o)) for o in openings.values()]
    obj = TypedObligationSet(
        family,
        (),
        tuple(roots[t] for t in sorted(roots)),
        tuple(sorted(leaves, key=lambda v: v.id)),
        tuple(sorted(references.values(), key=lambda r: r.ref)),
        tuple(sorted(subcircuits.values(), key=lambda s: s.digest)),
        tuple(o for o, _, _ in built),
    )
    consumed_pins = {rid: [p.target for p in pending if rid in p.incoming] for rid in pinned}
    composition = {
        "obligations": [{"unit": o.binding.unit, "session": o.binding.session[:12], "target": t, "subcircuit": o.subcircuit[:12], "incoming": len(o.boundary.incoming), "outgoing": len(o.boundary.outgoing)} for o, _, t in built],
        "shared_x_rows": {leaf: targets for leaf, targets in shared_x.items() if len(targets) > 1},
        "shared_w_rows": {leaf: targets for leaf, targets in shared_w.items() if len(targets) > 1},
        "input_equations": [{"ref": rid, "source": src, "consumed_by": consumed_pins[rid]} for rid, src in pinned.items()],
        "leaves": len(leaves),
        "references": len(references),
        "authentication": how,
    }
    return ComposedExport(obj, [row for _, row, _ in built], openings, composition)


def packed_witness(fx: GemmCoordinateFixture, statement: Statement) -> PackedWitness:
    """The format-3 witness of ``statement`` (the lowering of :func:`export_gemm_coordinate`'s
    object) from the fixture's openings: per tree in the statement's tree order, the whole
    packed leaf of every rank the statement opens -- Track A's ``Opening`` carried over
    verbatim (canonical id, dtype, shape text, raw little-endian bytes, sibling/lift path).

    Nothing is recomputed: the guest re-derives the leaf hash from these parts and folds
    the path to the root the statement names, so a fixture whose openings do not
    authenticate yields a witness the guest rejects.  ``ExportError`` if the statement
    opens a rank no opening of the fixture covers.
    """

    return packed_witness_of(fx.openings.values(), statement)


def packed_witness_of(openings: Iterable[Opening], statement: Statement) -> PackedWitness:
    """:func:`packed_witness` over any collection of Track A openings (one per opened leaf)."""

    by_root_rank = {(o.root, o.index): o for o in openings}
    trees: list[tuple[PackedLeaf, ...]] = []
    for tree in packed_table(statement).trees:
        leaves = []
        for rank in tree.ranks:
            opening = by_root_rank.get((tree.commitment.root, rank))
            if opening is None:
                raise ExportError(f"the statement opens rank {rank} of root {tree.commitment.root.hex()[:16]}..., which no opening covers")
            leaves.append(
                PackedLeaf(
                    opening.canonical_id,
                    opening.dtype,
                    hashing.shape_text(opening.shape),
                    opening.value,
                    tuple(opening.path),
                )
            )
        trees.append(tuple(leaves))
    return PackedWitness(tuple(trees))


def composed_record(export: ComposedExport, *, name: str, description: str, source: str, tags: Sequence[str] = ()) -> dict[str, Any]:
    """The conformance-vector shaped record of a :func:`export_composed` object: the object, its canonical
    bytes and digest ``h_O``, the lowering, the statement (hex, ``h_L``), the boundary words per obligation
    with the relation verdict, the format-3 witness with the transparent backend's verdict, and the
    composition report (which obligations share which leaves, the input equations and their consumers)."""

    from veritor.interface.vectors import Vector
    from veritor.protocol.messages import Reject
    from veritor.protocol.proofs import TransparentBackend

    statement = to_statement(export.obj)
    witness = packed_witness_of(export.openings.values(), statement)
    proof = encode_packed_witness(witness)
    gate_set = family_gate_set(TC_FAMILY, TC_WIDTH)
    assert gate_set is not None
    try:
        TransparentBackend(gate_set).verify(statement, proof)
        witness_verdict = "accept"
    except Reject as error:
        witness_verdict = f"reject:{error.code.name}"
    record = Vector(name, description, export.obj.to_json(), values=export.values, witness=proof.hex(), witness_verdict=witness_verdict,
                    tags=["stage-2", "tc", "real-operands", "framing", "packed-leaf", "packed-word", "witness", "composed", *tags]).to_json()
    record["provenance"] = {"source": source, "composition": export.composition,
                            "openings": {cid: {"tree": o.tree, "index": o.index, "bytes": len(o.value)} for cid, o in export.openings.items()}}
    return record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_record(fx: GemmCoordinateFixture, *, source: str) -> dict[str, Any]:
    """A conformance-vector record (the shape of ``fixtures/typed-obligation-v0/*.json``) for ``fx``."""

    from veritor.interface.vectors import Vector  # the record writer; generator module, pure Python

    obj, row, how = export_gemm_coordinate(fx)
    f = fx.fixture
    name = f"stage2/{f['family']}-{f['phase']}-{fx.directory.name.rsplit('_', 1)[-1]}-k{fx.K}"
    files = {n: _sha256(fx.directory / n) for n in ("fixture.json", "openings.json", "x.u16", "w_row.u16", "out_row.u16")}
    description = (
        f"Stage-2 candidate: one real GEMM output coordinate of Qwen2.5-1.5B ({f['op']}, {f['phase']}, "
        f"K={fx.K}, N={f['N']}, j={fx.j}) captured by the serving adapter (bundle profile {f['profile_id']}), "
        f"exported by verity_vllm.typed_obligation_export from {source}; the three openings authenticate under "
        f"Track A's trees (verity-vllm/leaf/v1), the committed word {f['committed_word']} equals the mma.sync chain's "
        f"prediction {f['predicted_word']}; kernel {f['kernel']['kernel_path']} ({f['kernel']['runtime_impl']}). "
        f"Subcircuit: veritor.interface.gemm_coordinate({fx.K}) in {TC_FAMILY} -- GemmCoordinate<K>, "
        f"a function-call instance of the registered composite Gemm over tc_dot (IR B.7, D.1); the weight words "
        f"are incoming values (their Input gates are not sampled by this plan, IR E); references are the words' "
        f"canonical references, leaves the rows they are packed in. "
        f"Authenticatable by the guest through packed-word positions (packed/v1:verity-vllm/leaf/v1:u16le:<count>:<index>, "
        f"IR H.4): the format-3 witness carries the three row leaves with Track A's paths, once each, and the guest "
        f"authenticates them under Track A's framing and reads the 3073 words at their statement-fixed indices."
    )
    from veritor.protocol.proofs import TransparentBackend
    from veritor.protocol.messages import Reject

    statement = to_statement(obj)
    witness = packed_witness(fx, statement)
    proof = encode_packed_witness(witness)
    gate_set = family_gate_set(TC_FAMILY, TC_WIDTH)
    assert gate_set is not None
    try:
        TransparentBackend(gate_set).verify(statement, proof)
        witness_verdict = "accept"
    except Reject as error:
        witness_verdict = f"reject:{error.code.name}"
    record = Vector(
        name,
        description,
        obj.to_json(),
        values=[row],
        relations="accept",
        witness=proof.hex(),
        witness_verdict=witness_verdict,
        tags=["stage-2", "tc", "real-operands", "framing", "packed-leaf", "packed-word", "witness"],
    ).to_json()
    record["provenance"] = {
        "fixture": source,
        "canonical_id": f["canonical_id"],
        "x_id": f["x_id"],
        "w_id": f["w_id"],
        "committed_word": f["committed_word"],
        "predicted_word": f["predicted_word"],
        "files_sha256": files,
        "authentication": how,
        "trees": {
            t: {k: fx.commitments["trees"][t][k] for k in ("root", "leaf_count", "ids_digest", "kind")}
            for t in sorted({o.tree for o in fx.openings.values()})
        },
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("fixture", type=Path, help="a <family>_<phase>_<n>/ fixture directory")
    parser.add_argument("--commitments", type=Path, default=None, help="commitments.json (default: ../commitments.json)")
    parser.add_argument("--out", type=Path, required=True, help="where to write the record")
    parser.add_argument("--source", default=None, help="provenance label (default: the fixture path)")
    args = parser.parse_args(argv)
    fx = load_fixture(args.fixture, args.commitments)
    record = export_record(fx, source=args.source or str(args.fixture))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(
        f"{record['name']}: verdict {record['verdict']}, relations {record.get('relations')}, "
        f"authenticatable {record.get('authenticatable')} {record.get('authenticity_reasons')}, "
        f"digest {record['digest'][:16]}... -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
