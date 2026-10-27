"""A chunked serving-rows template's units (`serving_rows.chunked`) on the bit rows C-Flock's `UnitStager` stages: their rows
(`unit_rows`) and table reads (`reads`), the most of them one flock-circuit statement holds (`chunk_cap`, a partition file's
``chunk_cap``), and a template's units read once cut into its chunks (`split`).  The window's chunks (`serving_rows.members`)
and the verifier's circuit (``benchmarks/one_stage/compose_circuit.py --template``) are both laid out by `unit_rows` and
`reads`.  Each function takes the template as `serving_rows.template` reads it."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity.ml.windows import interface as IF, rows as CS

#: flock-circuit's spec shape (`verity/ml/flock/live/src/circuit.rs`'s `shape_in_range`, `benchmarks/one_stage/a6.py`'s `M_MAX`):
#: a statement's dense size m = k_log + nbl, nbl = max(3, ceil(log2 blocks)), is at most 35
M_MAX = 35


def s8_words(K: int) -> int:
    """A ``ScaledMmS8Coordinate_v1{K}`` input row's u16 words: the K s8 codes' bytes, then the f32 scale's (A6's ``x`` / ``w``)."""
    return K // 2 + 2


#: the whole-row templates whose MUFU table reads `reads` lays out in lookup slots: the attention rows (in neither ``CUT`` nor
#: ``SPLITS``, so the whole row is the unit), and the per-key-block Calls the window circuit cuts their heads into
#: (`window_circuit.CUTS`: FA2's v18 by `AttentionHead_v19`, FA3's v4 by `AttentionHead_v20`), each reading EX2 a score or
#: RCP once
SLOT_ROWS = ("Attention_v5", "Attention_v18", "AttnBlockFirst_v1", "AttnBlock_v5", "AttnBlock_v4", "AttnHeadEnd_v1")


def reads(tp: Any) -> str:
    """How `UnitStager` lays out chunked template ``tp``'s MUFU table reads: in lookup slots (`read_slots`, each table folded
    once per statement) for a ``SPLITS`` part of at least one whole block (T >= BN: inline, step 16's 64-key part's EX2 reads
    are 4.5 times its other nonzeros, a 322 GB build against 75.6 GB in slots) and for a whole attention row (``SLOT_ROWS``:
    inline, the 1B decode row ``Attention_v5{T=1,NH=32,KVH=8,D=64,..}``'s 64 reads make 6.35e9 nonzeros for its 10.2M ANDs,
    622 an AND, and its ``--zk`` prover ran out of memory at 600 GiB; in slots, 314M nonzeros for 8,473,952 ANDs, 37 an AND,
    k_log 25, proved in 118 s, a 272 s session at 27 GB, accepted by Lean with its three negatives refused) and for its heads' per-key-block Calls, else inline, as step 16's 16-key part and
    five-split combine were proved, and the RMSNorm rows (``RMSNormTriton_v5``, ``RMSNormFusedCuda_v2``, 41 to 56 nonzeros
    an AND inline) at their served shapes. A ``_v2`` body reads no table (its EX2 is gates), so it lowers flat either way."""
    from verity_vllm.pipeline import split_rows as SP
    from verity_vllm.query.program_view import spec_base, spec_statics

    if tp.layout == "row":
        return "slots" if spec_base(tp.id) in SLOT_ROWS else "inline"
    if tp.layout != "split" or SP.combine(tp):
        return "inline"
    S = spec_statics(tp.word)
    return "slots" if int(S["T"]) >= int(S["BN"]) else "inline"


def unit_rows(tp: Any) -> tuple[list[tuple[str, str, int, list[str]]], list[tuple[str, tuple[int, ...]]]]:
    """Chunked template ``tp``'s rows as `UnitStager` stages them, ``(in_rows, out_rows)``: a ``ScaledMmS8Coordinate``'s A6 rows,
    ``x``     (the token's s8 codes then its f32 scale, row-seg) and ``w`` (``W[c, :]`` then ``sw[c]``, v2), then the v2 output row
    ``o0``; any other chunked template's (`serving_rows.chunked`) its Definition's bit rows (`interface.unit_rows`) under its
    cuts (`split_rows.cuts`, a ``SPLITS`` combine's), parameter g's row ``in<g>``, or ``in<g>.<s>`` for its row s when cut."""
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_vllm.pipeline import split_rows as SP
    from verity_vllm.query.program_view import spec_statics

    if tp.layout == "s8":
        bits = 16 * s8_words(int(spec_statics(tp.id)["K"]))
        return [("x", SEG, bits, ["x", "sx"]), ("w", V2, bits, ["wrow", "sw"])], [(V2, (0,))]
    if tp.layout not in ("coord", "cols", "slice", "kchunk", "row", "part", "split"):
        raise ValueError(f"{tp.id}: its layout {tp.layout!r} is not a chunked one")
    g = {nm: k for k, (nm, _ty) in enumerate(tp.fn.signature()[0])}

    def port(row: str) -> str:
        nm, _, s = row.partition("#")
        return f"in{g[nm]}" + (f".{s}" if s else "")

    in_rows, out_rows = IF.unit_rows(tp.fn, SP.cuts(tp) if tp.layout == "split" else None)
    return [(port(p), *rest) for p, *rest in in_rows], out_rows


def chunk_cap(tp: Any) -> int:
    """The most instances of chunked template ``tp`` one flock-circuit statement holds, as ``benchmarks/one_stage/a6.py``'s
    ``chunk_cap`` computes it: the template staged by `UnitStager` on its rows (`unit_rows`) with its `reads`, then
    ``vus_per_block`` blocks of ``2^(M_MAX - k_log)``.  Builds the circuit (seconds to minutes; a 64-key ``SparseMlaSplitPart``
    about 36 min at 71 GB): a partition file pins it as ``chunk_cap``."""
    import tempfile

    from verity_flock.class_statement import UnitStager

    in_rows, out_rows = unit_rows(tp)
    with tempfile.TemporaryDirectory() as d:
        st = UnitStager(tp.fn, "unit/serving-rows-cap", in_rows, out_rows, circuit=Path(d) / "circuit.txt",
                        bind={"partition": "0" * 128, "program": "0" * 128}, reads=reads(tp))
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
