"""[E2, EOD 2026-09-11, INTEGRATOR D23] The pinned MUFU table trees -- ``table-layout/v1`` as code.

The four table gates of ``veritor.reduce-mufu@1`` (``rsqrt_approx``, ``div_full_rcp``, ``mufu_sqrt_tab``; ``div_full_scale_a`` reads no
row) and the two of ``veritor.ftz-f32@1`` (``mufu_ex2_ftz``, ``fa2_inv_sum``) take the hardware table row as a **64-bit operand**
``(index << 32) | T[index]`` (``veritor.core.reduce_gates._table_word`` refuses a row whose high word is not the index the operand
reads).  The tables are conformance evidence measured on sm_89 and shipped as int8 delta fixtures under ``vllm-poc/fixtures``
(``fa2_model.MufuTables.load`` / ``rms_triton_model.RmsTables.load`` reconstruct them by cumulative sum) -- so a table tree is a pure
function of the registry checkout, and this module builds it anywhere in seconds (no leaf files need to travel).

The rule ``table-layout/v1`` (``layout.TABLE_RULE``; digest ``layout_digest(rule=TABLE_RULE, chunk_words=TABLE_ROWS_PER_LEAF)``):

1. **Tables.** ``mufu/ex2`` (2^23 rows: ``ex2.approx.ftz`` of ``j / 2^23``), ``mufu/rcp`` (2^23: ``rcp.approx.ftz`` of ``1.m``),
   ``mufu/sqrt`` (2^24: ``sqrt.approx.ftz`` of ``1.m`` then of ``2 * 1.m``), ``mufu/rsq`` (2^24: ``rsqrt.approx.ftz`` likewise) --
   the names E4's ``ck_ftz.KindSpec.table`` already spells; one tree per table, tree id ``table/<name>``.
2. **Row.** Row ``i`` is the 64-bit word ``(i << 32) | T[i]`` as 8 little-endian bytes.
3. **Leaves.** Rows ``[k * 2048, (k + 1) * 2048)`` in index order form leaf ``k``: canonical id ``table/<name>/chunk=<k>``, dtype
   ``uint64``, shape ``[2048]`` (16 KiB -- the size of the largest interior leaf); rank = ``k``; ``k = i // 2048``, ``flat = i % 2048``.
4. **Hashes.** Exactly the interior trees' (``verity-vllm/leaf/v1`` over ``(leaf_id, dtype, "[2048]", bytes)``, node/lift/empty/ids
   framing) -- :class:`e2e.interior.tree.InteriorTree` over an :class:`e2e.interior.layout.InteriorLayout` of rule ``TABLE_RULE``.
5. **Pin.** ``protocol_instance.json#commitments.tables[<name>] = table_record(name)`` (:func:`table_record`): ``{tree, root,
   leaf_count, ids_digest, rows, row_width: 64, rule: "table-layout/v1", layout_digest, source: {file, sha256, bytes}}``; E6 pins the
   four records (``roots.json`` is this module's output, never copied by hand: ``check_tables(pins)`` recomputes).
6. **Index of a row** (what the checkers already compute): ``mufu/rcp``: ``reduce_gates.mufu_rcp_index(x) = x & 0x7FFFFF``
   (``div_full_rcp_index(b)`` after the range scaling; ``ck_ftz.fa2_inv_sum_index``); ``mufu/sqrt`` / ``mufu/rsq``:
   ``reduce_gates.mufu_sqrt_index(x) = ((exp - 127) & 1) << 23 | mantissa`` (``rsqrt_approx_index`` after the subnormal pre-scale);
   ``mufu/ex2``: ``ck_ftz.mufu_ex2_index(x)``.  :func:`open_row` takes the index and returns the opening + the word's position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e2e.interior.layout import (
    TABLE_ORIGIN,
    TABLE_ROWS_PER_LEAF,
    TABLE_RULE,
    CallSpan,
    InteriorLayout,
    LeafSpec,
    layout_digest,
    layout_pin,
)
from e2e.interior.tree import InteriorTree, Opening, read_word, verify_opening, write_tree

__all__ = [
    "ROW_WIDTH",
    "TABLES",
    "TableRow",
    "TableSpec",
    "TableWords",
    "build_table_tree",
    "check_tables",
    "leaf_id",
    "load_table",
    "open_row",
    "row_position",
    "row_word",
    "table_layout",
    "table_layout_digest",
    "table_record",
    "tree_id",
]

ROW_WIDTH = 64
TABLE_N = 1 << 23
_T0 = 0x3F800000  # every table starts at 1.0: ex2(0), rcp(1.0), sqrt(1.0), rsqrt(1.0)
_POC = Path(__file__).resolve().parents[2]  # vllm-poc/


@dataclass(frozen=True)
class TableSpec:
    """One registered MUFU table: name (E4's spelling), rows, the fixture delta file it is reconstructed from, its readers."""

    name: str
    rows: int
    fixture: str  # relative to vllm-poc/
    file: str
    domain: str
    index_rule: str
    readers: tuple[str, ...]

    @property
    def path(self) -> Path:
        return _POC / self.fixture / self.file


TABLES: dict[str, TableSpec] = {
    t.name: t
    for t in (
        TableSpec("mufu/ex2", TABLE_N, "fixtures/W11-40f0cebeb670-20260907T1800Z/mufu", "mufu_ex2_delta_int8.xz",
                  "ex2.approx.ftz.f32 of j / 2^23 (fa2_model.MufuTables.ex2)", "ck_ftz.mufu_ex2_index(x)",
                  ("MufuEx2Ftz_v1|primitive (ftz-f32@1 mufu_ex2_ftz)",)),
        TableSpec("mufu/rcp", TABLE_N, "fixtures/W11C-40f0cebeb670-20260907T1900Z/tables", "mufu_rcp_delta_int8.xz",
                  "rcp.approx.ftz.f32 of 1.m (fa2_model.MufuTables.rcp == rms_triton_model.RmsTables.rcp, one file sha 291d1c9c...)",
                  "reduce_gates.mufu_rcp_index(x) = x & 0x7FFFFF; div_full_rcp_index(b) after the div.full range scaling; ck_ftz.fa2_inv_sum_index(s)",
                  ("DivFullRcp_v1|primitive (reduce-mufu@1 div_full_rcp)", "Fa2InvSum_v1|primitive (ftz-f32@1 fa2_inv_sum)")),
        TableSpec("mufu/sqrt", 2 * TABLE_N, "fixtures/W11C-40f0cebeb670-20260907T1900Z/tables", "mufu_sqrt_delta_int8.xz",
                  "sqrt.approx.ftz.f32 of 1.m (rows 0..2^23-1) and of 2*1.m (rows 2^23..2^24-1) (rms_triton_model.RmsTables.sqrt)",
                  "reduce_gates.mufu_sqrt_index(x) = (((exp-127) & 1) << 23) | mantissa",
                  ("MufuSqrtFtz_v1|primitive (reduce-mufu@1 mufu_sqrt_tab)",)),
        TableSpec("mufu/rsq", 2 * TABLE_N, "fixtures/W11C-40f0cebeb670-20260907T1900Z/tables", "mufu_rsq_delta_int8.xz",
                  "rsqrt.approx.ftz.f32 of 1.m and of 2*1.m (rms_triton_model.RmsTables.rsq, the CUDA rms_norm kernel's MUFU.RSQ)",
                  "reduce_gates.rsqrt_approx_index(x) = mufu_sqrt_index(x scaled by 2^24 when |x| < 2^-126)",
                  ("RsqrtApprox_v1|primitive (reduce-mufu@1 rsqrt_approx)",)),
    )
}


# --- the rule as code ---------------------------------------------------------------------------------------------------------------


def tree_id(name: str) -> str:
    return f"table/{_spec(name).name}"


def leaf_id(name: str, chunk: int) -> str:
    if int(chunk) < 0:
        raise ValueError("chunk index must be >= 0")
    return f"table/{_spec(name).name}/chunk={int(chunk)}"


def table_layout_digest(rows_per_leaf: int = TABLE_ROWS_PER_LEAF) -> str:
    return layout_digest(chunk_words=rows_per_leaf, rule=TABLE_RULE)


def table_pin(rows_per_leaf: int = TABLE_ROWS_PER_LEAF) -> dict[str, Any]:
    """The rule's pin fields (``framing`` chunks-2048, ``word_order`` index, ``tree`` table/<name>, ``digest``, ``rule``)."""
    return layout_pin(rows_per_leaf, rule=TABLE_RULE)


def _spec(name: str) -> TableSpec:
    try:
        return TABLES[str(name)]
    except KeyError:
        raise KeyError(f"unknown table {name!r} (registered: {sorted(TABLES)})") from None


def table_layout(name: str, *, rows_per_leaf: int = TABLE_ROWS_PER_LEAF) -> InteriorLayout:
    """The leaves of table ``name`` (rows in index order, ``rows_per_leaf`` per leaf) as an :class:`InteriorLayout` of rule ``TABLE_RULE``:
    ``layout.position(i)`` = ``(rank, flat)`` of row ``i``; ``gates`` are the row indices."""
    spec = _spec(name)
    n = int(rows_per_leaf)
    if n < 1 or spec.rows % n:
        raise ValueError(f"rows_per_leaf {n} must divide {spec.rows}")
    idx = np.arange(spec.rows, dtype=np.int64)
    leaves = [LeafSpec(k, leaf_id(name, k), spec.name, k, ROW_WIDTH, idx[k * n:(k + 1) * n], origin=TABLE_ORIGIN) for k in range(spec.rows // n)]
    return InteriorLayout(tree_id(name), leaves, [CallSpan(spec.name, 0, spec.rows)], prefix="", chunk_words=n, rule=TABLE_RULE)


def row_position(name: str, index: int, *, rows_per_leaf: int = TABLE_ROWS_PER_LEAF) -> tuple[str, str, int, int]:
    """``(tree id, leaf id, rank, flat)`` of row ``index`` -- what a resolver's ``table_row(table, index)`` names
    (E4's ``CommittedWord(tree, leaf, shape=(rows_per_leaf,), flat, width=64)``).  No table data needed."""
    spec = _spec(name)
    i = int(index)
    if not 0 <= i < spec.rows:
        raise IndexError(f"{spec.name}: row {i} outside [0, {spec.rows})")
    k, flat = divmod(i, int(rows_per_leaf))
    return tree_id(name), leaf_id(name, k), k, flat


# --- the table data ---------------------------------------------------------------------------------------------------------------------


def load_table(name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """``(T as uint32[rows], source)`` reconstructed from the fixture delta file (cumulative sum from 1.0); ``source`` names the file,
    its sha256 and size.  Deterministic; refuses a fixture of the wrong length or first value."""
    import lzma

    spec = _spec(name)
    raw = spec.path.read_bytes()
    delta = np.frombuffer(lzma.decompress(raw), dtype=np.int8)
    if delta.shape[0] != spec.rows - 1:
        raise ValueError(f"{spec.path}: expected {spec.rows - 1} deltas, got {delta.shape[0]}")
    t = np.empty(spec.rows, dtype=np.int64)
    t[0] = _T0
    np.cumsum(delta.astype(np.int64), out=t[1:])
    t[1:] += _T0
    if t.min() < 0 or t.max() > 0xFFFFFFFF:
        raise ValueError(f"{spec.path}: reconstructed table leaves u32")
    return t.astype(np.uint32), {"fixture": spec.fixture, "file": spec.file, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def row_word(index: int | np.ndarray, value: int | np.ndarray) -> np.ndarray:
    """The 64-bit row word(s) ``(index << 32) | value`` (uint64)."""
    return (np.asarray(index, dtype=np.uint64) << np.uint64(32)) | np.asarray(value, dtype=np.uint64)


def build_table_tree(name: str, *, rows_per_leaf: int = TABLE_ROWS_PER_LEAF, table: np.ndarray | None = None) -> tuple[InteriorTree, dict[str, Any]]:
    """The committed tree of table ``name`` and its source description.  ``table`` (uint32[rows]) overrides the fixture (tests)."""
    spec = _spec(name)
    if table is None:
        table, source = load_table(name)
    else:
        table = np.asarray(table, dtype=np.uint32)
        if table.shape != (spec.rows,):
            raise ValueError(f"{spec.name}: table has shape {table.shape}, the registry says ({spec.rows},)")
        source = {"fixture": "<given>", "file": "<given>", "sha256": hashlib.sha256(table.tobytes()).hexdigest(), "bytes": int(table.nbytes)}
    lay = table_layout(name, rows_per_leaf=rows_per_leaf)
    words = row_word(np.arange(spec.rows, dtype=np.uint64), table).astype("<u8")
    n = int(rows_per_leaf)
    values = [words[k * n:(k + 1) * n].tobytes() for k in range(lay.leaf_count)]
    return InteriorTree(lay, values), source


def table_record(name: str, tree: InteriorTree, source: Mapping[str, Any]) -> dict[str, Any]:
    """The ``commitments.tables[<name>]`` record E6 pins (rule clause 5)."""
    spec = _spec(name)
    rec = tree.record()
    return {"tree": tree.tree_id, "root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec["ids_digest"], "rows": spec.rows,
            "row_width": ROW_WIDTH, "rows_per_leaf": tree.layout.chunk_words, "rule": TABLE_RULE, "layout_digest": rec["layout_digest"],
            "origin": TABLE_ORIGIN, "domain": spec.domain, "index_rule": spec.index_rule, "readers": list(spec.readers), "source": dict(source)}


# --- openings (E7 / E4) ----------------------------------------------------------------------------------------------------------------


def open_row(tree: InteriorTree, index: int) -> tuple[Opening, int, int]:
    """``(opening of the leaf holding row index, flat, row word)``: ``opening`` is a ``verity_vllm.commit.Opening`` (index = rank) over the
    16 KiB leaf; ``read_word(opening, flat)`` are the row's 8 LE bytes; ``row word == (index << 32) | T[index]``."""
    opening, flat = tree.open_gate(int(index))
    word = int.from_bytes(read_word(opening, flat), "little")
    if word >> 32 != int(index):
        raise AssertionError(f"{tree.tree_id}: row {index} carries index {word >> 32}")
    return opening, flat, word


@dataclass(frozen=True)
class TableRow:
    """The position of a committed table row in E4's ``CommittedWord`` shape (``tree, leaf, shape, flat, width``): what a
    ``WordResolver.table_row(table, index)`` returns.  ``expected()`` is the verifier's ``ExpectedValue`` (lazy ``veritor`` import)."""

    tree: str
    leaf: str
    shape: tuple[int, ...]
    flat: int
    width: int = ROW_WIDTH

    @property
    def offset(self) -> tuple[int, ...]:
        return (int(self.flat),)

    @property
    def ref(self) -> str:
        return f"{self.leaf}[{self.flat}]"

    def expected(self):
        from veritor.interface.expected import ExpectedValue

        return ExpectedValue(self.tree, self.leaf, self.offset)


class TableWords:
    """The table half of E4's ``ck_ftz.WordResolver`` and E7's row opener: ``table_row(table, index)`` names the position (no data);
    ``tree(table)`` builds (once) the committed tree, checking its root against ``pins`` (``commitments.tables``) when given, so an
    opener never opens against an unpinned or foreign root; ``open(table, index)`` = :func:`open_row`.  ``cls`` lets E4 receive its own
    ``CommittedWord`` (same field order)."""

    def __init__(self, pins: Mapping[str, Mapping[str, Any]] | None = None, *, cls: type = TableRow) -> None:
        self.pins = dict(pins) if pins else None
        self.cls = cls
        self._trees: dict[str, InteriorTree] = {}

    def table_row(self, table: str, index: int):
        if table not in TABLES or not 0 <= int(index) < TABLES[table].rows:
            return None
        if self.pins is not None and table not in self.pins:
            return None  # unpinned table: the record does not commit it
        tree_, leaf, _rank, flat = row_position(table, index, rows_per_leaf=self._rows_per_leaf(table))
        return self.cls(tree_, leaf, (self._rows_per_leaf(table),), flat, ROW_WIDTH)

    def _rows_per_leaf(self, table: str) -> int:
        return int((self.pins or {}).get(table, {}).get("rows_per_leaf", TABLE_ROWS_PER_LEAF))

    def tree(self, table: str) -> InteriorTree:
        t = self._trees.get(table)
        if t is None:
            t, _src = build_table_tree(table, rows_per_leaf=self._rows_per_leaf(table))
            if self.pins is not None:
                pin = self.pins.get(table)
                if pin is None:
                    raise KeyError(f"commitments.tables[{table!r}] is not pinned")
                if pin.get("root") != t.root.hex() or int(pin.get("leaf_count", -1)) != t.leaf_count:
                    raise ValueError(f"{table}: rebuilt root {t.root.hex()[:16]} / {t.leaf_count} leaves != pinned {str(pin.get('root'))[:16]} / {pin.get('leaf_count')}")
            self._trees[table] = t
        return t

    def open(self, table: str, index: int) -> tuple[Opening, int, int]:
        return open_row(self.tree(table), index)


# --- verification (E9 / E6) ------------------------------------------------------------------------------------------------------------


def check_tables(pins: Mapping[str, Mapping[str, Any]], *, names: tuple[str, ...] | None = None) -> list[str]:
    """Refusals of ``commitments.tables`` against a rebuild from the registry (empty = every named table's pin is what the rule yields).
    Rebuilds each table (~1-3 s each); a table absent from ``pins`` is named."""
    p: list[str] = []
    for name in names or tuple(TABLES):
        pin = pins.get(name)
        if not isinstance(pin, Mapping):
            p.append(f"TABLE_UNPINNED: commitments.tables[{name!r}] is absent")
            continue
        try:
            tree, source = build_table_tree(name, rows_per_leaf=int(pin.get("rows_per_leaf", TABLE_ROWS_PER_LEAF)))
        except (OSError, ValueError) as e:
            p.append(f"TABLE_UNBUILDABLE: {name}: {type(e).__name__}: {str(e)[:160]}")
            continue
        want = table_record(name, tree, source)
        for k in ("tree", "root", "leaf_count", "ids_digest", "rows", "row_width", "rule", "layout_digest"):
            if pin.get(k) != want[k]:
                p.append(f"TABLE_PIN_MISMATCH: {name}.{k} = {pin.get(k)!r}, the rule yields {want[k]!r}")
        if (pin.get("source") or {}).get("sha256") not in (None, want["source"]["sha256"]):
            p.append(f"TABLE_PIN_MISMATCH: {name}.source.sha256 = {pin['source'].get('sha256')!r}, the fixture is {want['source']['sha256']!r}")
    return p


# --- CLI ---------------------------------------------------------------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    """``python -m e2e.interior.tables build --out <dir> [--write-trees] [--table mufu/ex2 ...]`` -> ``<out>/roots.json`` (the four records +
    the rule's pin + MEASURED timings); ``--write-trees`` also writes ``<out>/table/<name>/{index.json, gates.bin, leaf-*.bin}`` (384 MiB
    for the four).  ``check --pins <commitments.json or protocol_instance.json>`` recomputes and compares."""
    ap = argparse.ArgumentParser(prog="python -m e2e.interior.tables")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--table", action="append", default=None)
    b.add_argument("--write-trees", action="store_true")
    b.add_argument("--openings", type=int, default=64, help="self-check: open this many random rows per table and verify")
    c = sub.add_parser("check")
    c.add_argument("--pins", required=True, help="a JSON file holding commitments.tables (or a document with commitments.tables / tables)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        doc: dict[str, Any] = {"schema": "verity-tables/roots/v1", "label": "MEASURED", "rule": TABLE_RULE, "layout": table_pin(), "tables": {}, "timings_s": {},
                               "host": os.uname().nodename, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rng = np.random.default_rng(20260911)
        for name in a.table or list(TABLES):
            t0 = time.perf_counter()
            tree, source = build_table_tree(name)
            built = time.perf_counter() - t0
            for i in rng.integers(0, TABLES[name].rows, size=max(0, int(a.openings))).tolist():
                opening, flat, word = open_row(tree, i)
                if not verify_opening(tree.root, tree.leaf_count, opening):
                    raise SystemExit(f"{name}: opening of row {i} does not verify")
            if a.write_trees:
                write_tree(tree, out / tree.tree_id)
            doc["tables"][name] = table_record(name, tree, source)
            doc["timings_s"][name] = {"build": round(built, 3), "openings_checked": int(a.openings), "total": round(time.perf_counter() - t0, 3)}
            print(f"{tree.tree_id}: {TABLES[name].rows} rows, {tree.leaf_count} leaves, root {tree.root.hex()} ({built:.2f}s build)", flush=True)
        (out / "roots.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"roots": str(out / "roots.json"), "tables": {k: v["root"] for k, v in doc["tables"].items()}}))
        return 0
    pins = json.loads(Path(a.pins).read_text())
    for key in ("commitments", "tables"):
        if isinstance(pins, Mapping) and key in pins and isinstance(pins[key], Mapping) and not any(k in TABLES for k in pins):
            pins = pins[key]
    problems = check_tables(pins)
    print(json.dumps({"problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
