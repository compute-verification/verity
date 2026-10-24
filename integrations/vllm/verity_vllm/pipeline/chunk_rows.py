"""A chunked serving-rows template's units (`serving_rows.CHUNKED`) on the bit rows C-Flock's `UnitStager` stages: their rows
(`unit_rows`), the most of them one flock-circuit statement holds (`chunk_cap`, a partition file's ``chunk_cap``), and a
template's units read once cut into its chunks (`split`).  The window's chunks (`serving_rows.members`) and the verifier's
circuit (``benchmarks/one_stage/compose_circuit.py --template``) are both laid out by `unit_rows`.  Each function takes the
template as `serving_rows.template` reads it."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity_vllm.commit import serving_rows as CS

#: flock-circuit's spec shape (`verity/ml/flock/live/src/circuit.rs`'s `shape_in_range`, `benchmarks/one_stage/a6.py`'s `M_MAX`):
#: a statement's dense size m = k_log + nbl, nbl = max(3, ceil(log2 blocks)), is at most 35
M_MAX = 35


def s8_words(K: int) -> int:
    """A ``ScaledMmS8Coordinate_v1{K}`` input row's u16 words: the K s8 codes' bytes, then the f32 scale's (A6's ``x`` / ``w``)."""
    return K // 2 + 2


def unit_rows(tp: Any) -> tuple[list[tuple[str, str, int, list[str]]], list[tuple[str, tuple[int, ...]]]]:
    """Chunked template ``tp``'s rows as `UnitStager` stages them, ``(in_rows, out_rows)``: a ``ScaledMmS8Coordinate``'s A6 rows,
    ``x``     (the token's s8 codes then its f32 scale, row-seg) and ``w`` (``W[c, :]`` then ``sw[c]``, v2); a ``COORDS``
    coordinate's, a ``COLS`` column block's, a ``SLICES`` slice's or a ``KCHUNKS`` K-chunk's ``in<g>``, one v2 row per parameter
    (its port); then the one v2 output row ``o0``."""
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_vllm.query.program_view import spec_statics

    if tp.layout == "s8":
        bits = 16 * s8_words(int(spec_statics(tp.id)["K"]))
        return [("x", SEG, bits, ["x", "sx"]), ("w", V2, bits, ["wrow", "sw"])], [(V2, (0,))]
    if tp.layout in ("coord", "cols", "slice", "kchunk"):
        params, _ret = tp.fn.signature()
        return [(f"in{g}", V2, sum(ty.leaf_widths()), [nm]) for g, (nm, ty) in enumerate(params)], [(V2, (0,))]
    raise ValueError(f"{tp.id}: its layout {tp.layout!r} is not a chunked one")


def chunk_cap(tp: Any) -> int:
    """The most instances of chunked template ``tp`` one flock-circuit statement holds, as ``benchmarks/one_stage/a6.py``'s
    ``chunk_cap`` computes it: the template staged by `UnitStager` on its rows (`unit_rows`), then ``vus_per_block`` blocks of
    ``2^(M_MAX - k_log)``.  Builds the circuit (seconds to minutes): a partition file pins it as ``chunk_cap``."""
    import tempfile

    from verity_flock.class_statement import UnitStager

    in_rows, out_rows = unit_rows(tp)
    with tempfile.TemporaryDirectory() as d:
        st = UnitStager(tp.fn, "unit/serving-rows-cap", in_rows, out_rows, circuit=Path(d) / "circuit.txt",
                        bind={"partition": "0" * 128, "program": "0" * 128})
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
