"""A chunked serving-rows template's units (`serving_rows.chunked`) on the bit rows Flock's `UnitStager` stages: their rows
(`unit_rows`), a chunk's member (`unit_member`), table reads (`reads`) and statement (`statement`), the most of them one
flock-circuit statement holds (`chunk_cap`, a partition file's ``chunk_cap``), and a template's units read once cut into
its chunks (`split`).  The window's chunks (`serving_rows.members`) are laid out by `unit_rows`, and the verifier's circuit
(``benchmarks/one_stage/compose_circuit.py --template``) by `unit_rows` and `statement`, on the rows the chunks commit
(`statements`, the window's circuit record's).  Each function takes the template as `serving_rows.template` reads it.  `zk_refusal` is why ``verify --zk`` refuses a statement's form, and `zk_check` refuses a
partition file whose ``zk`` says its window is for such sessions and that names a template of that form."""
from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.commitments.windows import interface as IF, rows as CS

#: flock-circuit's spec shape (`verity/protocols/flock/prover/live/src/circuit.rs`'s `shape_in_range`, `benchmarks/one_stage/a6.py`'s `M_MAX`):
#: a statement's dense size m = k_log + nbl, nbl = max(3, ceil(log2 blocks)), is at most 35
M_MAX = 35
#: the layout mode of a table read's templated statement (`statement`; `verity_flock.tmpl.mode_of`)
READ_LAYOUT = "iv/hybrid/4"


def s8_words(K: int) -> int:
    """A ``ScaledMmS8Coordinate_v1{K}`` input row's u16 words: the K s8 codes' bytes, then the f32 scale's (A6's ``x`` / ``w``)."""
    return K // 2 + 2


#: the whole-row templates whose MUFU table reads `reads` lays out in lookup slots: the attention rows (in neither ``CUT`` nor
#: ``CHAINS``, so the whole row is the unit), and the per-key-block Calls the window circuit cuts their heads into
#: (`window_circuit.CUTS`: FA2's v18 by `AttentionHead_v19`, FA3's v4 by `AttentionHead_v20`), each reading EX2 a score or
#: RCP once
SLOT_ROWS = ("Attention_v5", "Attention_v18", "AttnBlockFirst_v1", "AttnBlock_v5", "AttnBlock_v4", "AttnHeadEnd_v1")


def reads(tp: Any) -> str:
    """How `UnitStager` lays out chunked template ``tp``'s MUFU table reads: flat for a ``STAGES`` stage on bits (its
    Boolean version, `boolean_topp_stages`: its MUFU operations are gates, so it reads no table and its unit returns its
    own outputs, the one form `verify --zk` takes, `one_stage.proofs.ks_form_why`); else in lookup slots (`read_slots`,
    each table folded once per statement) for a ``CHAINS`` key block of a whole block (NVIS = BN), for a whole attention
    row (``SLOT_ROWS``) and for its heads' per-key-block Calls: inline, their EX2 and RCP reads outnumber the rest of
    their nonzeros, so the build and the prover's memory grow with the reads, while in slots each table is folded once.
    Everything else that reads a table reads it inline, with fewer reads than a whole block's: a key block of fewer keys
    than BN (a read a key), a ``CHAINS`` end and column end (one read each) and combine (seven for five splits). The
    RMSNorm rows (``RMSNormTriton_v5``, ``RMSNormFusedCuda_v2``) read inline at their served shapes. A ``STAGES`` stage
    reads in slots: a top-p stats part or a probabilities block reads EX2 a lane, thousands of them; a greedy select's
    stage reads no table, so it lowers flat either way. A Boolean template's reads are inline whatever its layout, a key
    block's Boolean body (``SparseMlaHeadBlock_v2``) at every key count among them: a body whose MUFU is gates has no
    table to read, and one that still read a table would keep it in its own rows, so its unit returns its own outputs,
    the one form ``verify --zk`` takes (`zk_refusal`, `one_stage.proofs.ks_form_why`)."""
    from verity_vllm.pipeline import chain_rows as CH
    from verity_vllm.query.program_view import spec_base, spec_statics

    if tp.id != tp.word:
        return "inline"
    if tp.layout == "row":
        return "slots" if spec_base(tp.id) in SLOT_ROWS else "inline"
    if tp.layout == "stage":
        return "slots"
    if tp.layout != "chain" or CH.kind(tp) not in ("first", "block"):
        return "inline"
    S = spec_statics(tp.word)
    return "slots" if int(S["NVIS"]) >= int(S["BN"]) else "inline"


def statement(tp: Any) -> dict[str, Any]:
    """`UnitStager`'s statement arguments for chunked template ``tp``: a table read's (``read``, ``history``) the templated statement at
    `READ_LAYOUT` of the verifier's read template (``structural``: `verity_flock.read_tmpl`, the one template
    `Flock.ReadNet.check` accepts); every other template's flat statement, its table reads laid out as `reads` says."""
    if tp.layout in ("read", "history"):
        return {"layout": READ_LAYOUT, "structural": True}
    return {"reads": reads(tp)}


_PIECE_TABLES: dict[str, frozenset[str]] = {}


def _piece_tables(prim: Any) -> frozenset[str]:
    """The tables a lookup slot holds (`read_slots.split`'s ``slot``) that primitive ``prim``'s Flock piece reads, the piece
    lowered once on free operands as `ir_lower.definition_ands` counts it; none for a gate, a primitive with no piece, and a
    piece that lowers only on a constant operand (wiring)."""
    if prim.id not in _PIECE_TABLES:
        import inspect

        from verity.circuits.evaluation.reference import signature_of
        from verity_flock import ir_lower as IL, read_slots as RS, tail as TL

        tables: frozenset[str] = frozenset()
        if prim.params and prim.id in IL.PIECES and getattr(inspect.unwrap(IL.PIECES[prim.id]), "ands", None) is None:
            C = IL.Circuit()
            params, _ = signature_of(prim)
            IL.lower(C, prim, [C.input(f"in{j}", w) for j, w in enumerate(w for pt in params for w in pt.leaf_widths())])
            tables = frozenset(t for t, idx, w in C.lookups if RS.SLOT_TABLES.get(t) == len(idx) and w <= TL.VALUE_BITS)
        _PIECE_TABLES[prim.id] = tables
    return _PIECE_TABLES[prim.id]


def slot_tables(fn: Any) -> tuple[str, ...]:
    """The tables a lookup slot holds that Definition ``fn``'s unit reads, sorted: what `UnitStager` splits out of the unit
    under ``reads="slots"``.  Read off the Definition and never lowered: each specialized Definition's body is walked once
    and each primitive's piece lowered once (`_piece_tables`).  None for a body whose MUFU is gates."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.evaluation.reference import resolve

    memo: dict[int, frozenset[str]] = {}

    def tables(f: Any) -> frozenset[str]:
        if isinstance(f, PrimitiveDefinition):
            return _piece_tables(f)
        if id(f) not in memo:
            body = getattr(f, "body", None)
            called = {} if body is None else {id(nd.fn): nd.fn for nd in body.nodes}
            memo[id(f)] = frozenset().union(*(tables(g) for g in called.values()))
        return memo[id(f)]

    return tuple(sorted(tables(resolve(fn))))


def zk_refusal(tp: Any) -> str | None:
    """Why `verify --zk` refuses chunked template ``tp``'s `statement` for its form, known before anything is staged; None
    for a form it takes.  A flat statement whose unit reads a table a lookup slot holds (`slot_tables`), laid out in slots
    (`reads`), returns its outputs from its last tail stage, and the unit's form the sampled audit's knowledge soundness
    covers has the unit return them (`one_stage.proofs.ks_form_why`, `Flock.ProvedScope.scopeOkH`).  Only a word template's
    reads are in slots (`reads`), and the reason names it, the tables and, where one is registered, the Boolean version to
    name in its place."""
    if statement(tp).get("reads") != "slots" or not (tables := slot_tables(tp.fn)):
        return None
    from verity.circuits.codec import descriptor_id
    from verity.definitions import boolean as BB

    try:
        fix = f"name its Boolean version {descriptor_id(BB.boolean_version(tp.fn))}"
    except BB.NoBooleanVersion:
        fix = "no Boolean version of it is registered"
    return (f"{tp.id}: its statement reads {', '.join(tables)} in lookup slots and returns its outputs from a tail stage, a "
            f"form `verify --zk` refuses (`ProvedScope.scopeOkH`); {fix}")


def zk_check(partition: Mapping[str, Any], tps: Iterable[Any]) -> None:
    """A partition file's ``zk: true`` says its window is for ``verify --zk`` sessions: of the window's chunked templates
    ``tps``, the first whose statement is a form ``--zk`` refuses (its table reads in lookup slots, `zk_refusal`) is refused
    by name (ValueError), with the Boolean version to name in its place.  `serving_rows.population` calls it before any row
    is read."""
    if partition.get("zk"):
        for tp in tps:
            if (why := zk_refusal(tp)) is not None:
                raise ValueError(f"a partition for --zk sessions names {why}")


def unit_rows(tp: Any, seg: Collection[tuple[str, str]] = ()) -> tuple[list[tuple[str, str, int, list[str]]],
                                                                       list[tuple[str, tuple[int, ...]]]]:
    """Chunked template ``tp``'s rows as `UnitStager` stages them, ``(in_rows, out_rows)``: its Definition's bit rows
    (`interface.unit_rows`) under its cuts (`chain_rows.cuts`, a ``CHAINS`` combine's), parameter g's row ``in<g>``, or
    ``in<g>.<s>`` for its row s when cut. A row of ``tp`` that ``seg`` names (``(template, row)`` as `history_rows.seg`
    and `interface.seg_rows` do, a row by its port and output j by ``o<j>``) is ``hm96-sha512/row-seg/v1``, every other
    ``hm96-sha512/row/v2``; a name that is no row of ``tp`` is refused, and so is a row named with no template (a bare
    ``"o0"`` would unpack as a pair naming no template and leave the row v2). The one exception is a
    ``ScaledMmS8Coordinate``'s A6 rows, ``x`` (the token's s8
    codes then its f32 scale, row-seg) and ``w`` (``W[c, :]`` then ``sw[c]``, v2), then the v2 output row ``o0``: each operand
    packed with its scale in one row is the layout serving commits (`serving_rows`' s8 row tables), where
    `interface.unit_rows` would give ``x``, ``sx``, ``wrow`` and ``sw`` a row each."""
    from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_vllm.pipeline import chain_rows as CH
    from verity_vllm.query.program_view import spec_statics

    if any(isinstance(r, str) for r in seg):
        raise TypeError(f"{tp.id}: seg names each row as (template, row), as `history_rows.seg` does, not {list(seg)}")
    seg = {p for t, p in seg if t == tp.id}
    if tp.layout == "s8":
        if seg - {"x"}:
            raise ValueError(f"{tp.id}: its A6 rows are x row-seg and w and o0 v2, not {sorted(seg)} row-seg")
        bits = 16 * s8_words(int(spec_statics(tp.id)["K"]))
        return [("x", SEG, bits, ["x", "sx"]), ("w", V2, bits, ["wrow", "sw"])], [(V2, (0,))]
    if tp.layout not in ("coord", "cols", "slice", "kchunk", "row", "part", "chain", "stage", "slots", "read", "history", "expert",
                         "fused"):
        raise ValueError(f"{tp.id}: its layout {tp.layout!r} is not a chunked one")
    g = {nm: k for k, (nm, _ty) in enumerate(tp.fn.signature()[0])}

    def port(row: str) -> str:
        nm, _, s = row.partition("#")
        return f"in{g[nm]}" + (f".{s}" if s else "")

    lay, cuts = IF.layout(tp.fn), CH.cuts(tp) if tp.layout == "chain" else None
    named = {port(p): p for p, _w, _ps in lay.cut(cuts)} | {o: o for o, _w in lay.outs}
    if seg - set(named):
        raise ValueError(f"{tp.id}: {sorted(seg - set(named))} is no row of its {sorted(named)}")
    in_rows, out_rows = IF.unit_rows(tp.fn, cuts, {named[p] for p in seg})
    return [(port(p), *rest) for p, *rest in in_rows], out_rows


def unit_member(tp: Any, n: int, base: int = 0, name: str = "", seg: Collection[tuple[str, str]] = ()) -> CS.Member:
    """``n`` units of chunked template ``tp`` from unit ``base`` on bit rows, as `UnitStager` stages a unit: each port under
    the schema of its row of `unit_rows`, ``seg`` the window's row-seg rows (`history_rows.seg`: a history table's writer's
    output, its table's leaves), which its statement states likewise (`statements`)."""
    in_rows, out_rows = unit_rows(tp, seg)
    outs = (("o0", 1),) if tp.layout == "s8" else IF.layout(tp.fn).outs
    schemas = tuple((p, s) for p, s, _b, _ps in in_rows) + tuple((outs[j][0], s) for s, js in out_rows for j in js)
    return CS.Member(tp.id, tuple((p, b // 16) for p, _s, b, _ps in in_rows), outs, n, base, name=name, schemas=schemas)


def row_seg(m: CS.Member) -> list[tuple[str, str]]:
    """The rows member ``m`` commits as ``hm96-sha512/row-seg/v1`` rows, as ``(template, row)``."""
    return [(m.template, p) for p, s in m.schemas if s == CS.SCHEMA_HM96_SHA512_ROW_SEG]


def statements(ms: Sequence[CS.Member], template: Callable[[str], Any],
               declared: Collection[tuple[str, str]] = ()) -> dict[str, dict[str, Any]]:
    """The statement of each template of bit rows among a window's members ``ms`` (``template``: `serving_rows.template`), as
    ``compose_circuit.py --template`` composes it: its `unit_rows`, each row under the schema its members commit it under
    (`row_seg`), and its `statement`. Refused (ValueError) where two members of a template commit a row under two schemas,
    since a template has one statement, or where a row ``declared`` names (`interface.seg_rows` of the window's declared
    tables: a history's writer's output) is not committed row-seg."""
    segs: dict[str, list[tuple[str, str]]] = {}
    for m in ms:
        if m.schemas and segs.setdefault(m.template, row_seg(m)) != row_seg(m):
            raise ValueError(f"{m.template}: its members commit {segs[m.template]} and {row_seg(m)} as row-seg rows, and a "
                             f"template has one statement; commit every member's rows under one schema (`serving_rows.members`' seg)")
    if lost := sorted(set(declared) - {r for seg in segs.values() for r in seg}):
        raise ValueError(f"the declared history tables' writers' rows {lost} are not committed row-seg; build the window's "
                         f"members with the partition file's histories (`serving_rows.population`, `history_rows.seg`)")
    return {t: dict(zip(("in_rows", "out_rows"), unit_rows(template(t), seg)), **statement(template(t))) for t, seg in segs.items()}


def chunk_cap(tp: Any, seg: Collection[tuple[str, str]] = ()) -> int:
    """The most instances of chunked template ``tp`` one flock-circuit statement holds, as ``benchmarks/one_stage/a6.py``'s
    ``chunk_cap`` computes it: the template staged by `UnitStager` on its rows (`unit_rows`, ``seg`` the window's row-seg
    ones) as its `statement`, then
    ``vus_per_block`` blocks of ``2^(M_MAX - k_log)``.  Builds the circuit (seconds to minutes; a 64-key ``SparseMlaSplitPart``
    about 36 min at 71 GB): a partition file pins it as ``chunk_cap``."""
    import tempfile

    from verity_flock.class_statement import UnitStager

    in_rows, out_rows = unit_rows(tp, seg)
    with tempfile.TemporaryDirectory() as d:
        st = UnitStager(tp.fn, "unit/serving-rows-cap", in_rows, out_rows, circuit=Path(d) / "circuit.txt",
                        bind={"partition": "0" * 128, "program": "0" * 128}, **statement(tp))
    if M_MAX - st.meta["k_log"] < 3:
        raise ValueError(f"{tp.id}: k_log {st.meta['k_log']} leaves no statement inside the spec shape (m <= {M_MAX})")
    return int(st.meta["vus_per_block"]) << (M_MAX - int(st.meta["k_log"]))


def split(ms: Sequence[CS.Member], data: dict[str, Any], entries: list[list[Any]]) -> list[tuple[dict[str, Any], list[list[Any]]]]:
    """A template's chunks ``ms`` from its units read once (``data`` and ``entries`` from ``ms[0].base``): each chunk its units
    ``[base, base + n)``, and its index entries the rows whose units it holds, each with the row's whole range."""
    b0, out = ms[0].base, []
    for m in ms:
        a, b = m.base - b0, m.base - b0 + m.n
        out.append(({part: {p: v[a:b] for p, v in data[part].items()} for part in ("rows", "outs")},
                    [e for e in entries if e[0] < m.base + m.n and e[1] > m.base]))
    return out
