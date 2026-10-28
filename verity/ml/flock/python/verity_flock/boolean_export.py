"""Boolean-circuit export of a served row's leaf templates (the website visualizer's dataset; a derived format, not a
statement).

Each leaf template of a row's program graph is lowered by :mod:`verity_flock.ir_lower` (its C-Flock template module where
one exists; ``GemmCoordinate`` as tensor-core k-steps, ``ir_lower.tc_units``, total semantics: ``fp.tc_dot16``). The unit
circuits are built with the named ``fp`` / ``gf2`` / ``unit`` functions traced, so each unit is a tree of named subcircuits
(bf16->f32, multiply, align, add, normalize, round, lookup, ...) down to its gates. Gates outside the units (the template's
tail) are lowered with their piece where ``ir_lower.PIECES`` has one; the others are "not yet lowered".

Counting: ``gf2`` commits only ANDs; XOR and NOT are free linear forms. Materialised as gates, each distinct linear form a
subcircuit uses (an AND input or a returned bit) is a chain of ``|terms| - 1`` XORs, plus a NOT when its constant is 1; a
subcircuit's counts are its own gates plus its children's (forms are not shared across a subcircuit boundary). AND counts
are the gates live in the unit (they sum to the unit's committed ANDs). Identical subcircuits (same canonical structure) are
one node, referenced with instance counts. A subcircuit with children records its parts in order and the wiring between them
(:func:`_wiring`); a gate list tags each gate with the scope of the innermost named subcircuit that made it.

Every subcircuit type also has a module (:func:`_module`): its own gates, table reads and part calls in the order the
lowering made them, with each call's inputs bound to the module's wires. Expanding a module (:func:`expand_module`) rebuilds
the subcircuit's gates to any depth from the modules alone, so no size limit applies to them.

    python -m verity_flock.boolean_export --program-graph ROW.program.json --out DIR
"""
from __future__ import annotations

import argparse
import bisect
import gc
import gzip
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.ops.boolean import softmax as SM

from . import fp, gf2, ir_lower as IL, tail_pieces as TP, topp_word as TW, unit as U
from . import circuit_types as CT
from . import type_trace as TT
from .gf2 import Circuit, const_bits

SCHEMA = "verity-boolean-circuit/v1"
#: how a gather from committed weights (the embedding's row, a MoE expert's weights) is sized: "mux", a multiplexer over the
#: candidate rows, or "opening", a commitment opening of the committed table (no gates)
GATHERS = "mux"
#: a gather over at most this many candidates is traced as one circuit; a wider one is a composed Definition of its muxes
GATHER_INLINE = 4096
#: traced function -> display name (the id is structural; renaming a display name changes no id)
DISPLAY = {
    "bf16_to_f32": "bf16→f32", "f32_to_bf16": "f32→bf16 (round)", "unpack": "unpack f32", "round_f32": "round",
    "specials": "specials (NaN/Inf)", "f32_mul": "f32 multiply", "f32_add": "f32 add", "f32_sub": "f32 subtract",
    "f32_fma": "f32 fused multiply-add", "f32_div": "f32 divide", "f32_neg": "f32 negate",
    "tc_dot16": "tensor-core k-step (total)", "tc_dot_e4m3": "FP8 tensor-core k-step (total)", "_tc_operand": "tensor-core operand decode",
    "_classes": "classify (NaN/Inf)", "_finite_or_zero": "finite stand-in", "decode": "one-hot decode",
    "lookup": "table (plain gates)", "_normalize_sig": "normalize", "shl": "shift left", "shr_sticky": "align (shift right, sticky)",
    "min_u": "min", "neg_if": "conditional negate", "select": "select", "is_zero": "zero test", "add": "integer add",
    "sub": "integer subtract", "multiply": "integer multiply", "compress_columns": "carry-save compress",
    "lzc": "leading-zero count", "shr": "shift right", "shl_window": "shift left (window)", "mux": "mux",
    "or_reduce": "OR-reduce", "and_reduce": "AND-reduce", "group_sum": "group sum (align + add + normalize)",
    "saturated": "saturation test", "decode_operand": "operand decode", "max_tree": "max tree",
    "table_read": "table read (plain gates)", "ftz": "flush to zero", "f32_gt": "f32 compare (>)",
    "mufu_ex2": "MUFU.EX2", "mufu_rcp": "MUFU.RCP", "rcp_post": "MUFU.RCP exponent", "_rsq_post": "MUFU.RSQ exponent",
    "_div_scale": "div.full range scaling", "_abs_gt_big": "|b| > 2^126", "_scale": "scale by a power of two",
    "_parity_index": "rsq / sqrt table index", "_word": "special cases",
    "tl_exp": "tl.exp (MUFU.EX2, non-ftz)", "tl_div": "div.full (scaled numerator times the reciprocal)", "f32_ge": "f32 compare (>=)",
    "f32_near": "|a - b| < 1e-9", "py_max": "max (Python)", "py_min": "min (Python)", "nan_max": "max (numpy, NaN wins)",
    "nan_min": "min (numpy, NaN wins)", "int_to_f32": "count to f32", "numkeep": "numkeep (truncate, clamp)",
    "word_eq": "word compare (==)", "nv_logf": "__nv_logf",
}
KIND = {"multiply": "multiply", "f32_mul": "multiply", "shr_sticky": "align", "add": "add", "f32_add": "add",
        "sub": "add", "_normalize_sig": "normalize", "lzc": "normalize", "shl": "normalize", "round_f32": "round",
        "f32_to_bf16": "round", "lookup": "table", "decode": "table", "bf16_to_f32": "convert", "tc_dot16": "dot-step",
        "group_sum": "add", "compress_columns": "add", "f32_fma": "fma"}

# --- tracing: which named call made each committed bit (circuit-check's attribution of redundant ANDs) --------------------

class _Call:
    __slots__ = ("fn", "start", "end", "kids")

    def __init__(self, fn, start):
        self.fn, self.start, self.end, self.kids = fn, start, start, []


_T: dict = {"C": None, "stack": None}


def _wrap(name, fn):
    def w(*a, **k):
        if TT._S["on"]:
            return TT.call(name, fn, a, k)
        st = _T["stack"]
        if st is None:
            return fn(*a, **k)
        C = next((x for x in a if isinstance(x, Circuit)), None)
        if C is not None:
            _T["C"] = C
        C = _T["C"]
        node = _Call(name, len(C.kind) if C is not None else 0)
        st[-1].kids.append(node)
        st.append(node)
        try:
            return fn(*a, **k)
        finally:
            st.pop()
            node.end = len(_T["C"].kind) if _T["C"] is not None else node.start

    w.__wrapped__ = fn
    return w


_PATCHED = False


def _patch():
    global _PATCHED
    if _PATCHED:
        return
    originals = {}
    for mod in (gf2, fp, SM, SC, U, TP, TW):
        for name in DISPLAY:
            f = getattr(mod, name, None)
            if callable(f) and not isinstance(f, type):
                originals[id(f)] = (name, f)
    wrapped = {k: _wrap(n, f) for k, (n, f) in originals.items()}
    for mod in (gf2, fp, SM, SC, U, IL, TP, TW):
        for name in dir(mod):
            f = getattr(mod, name)
            if callable(f) and id(f) in wrapped:
                setattr(mod, name, wrapped[id(f)])
    for k, f in list(IL.PIECES.items()):
        IL.PIECES[k] = _wrap(k, f)
    IL.PIECES.wrap = _wrap
    for k, (f, ins, out) in list(TW.OPS.items()):
        TW.OPS[k] = (wrapped.get(id(f), f), ins, out)
    _PATCHED = True


def trace(build) -> tuple[Circuit, _Call]:
    """Run ``build()`` (returns the Circuit it built) with the named functions traced: which call made each committed bit."""
    _patch()
    root = _Call("root", 0)
    _T.update(C=None, stack=[root])
    try:
        C = build()
    finally:
        _T.update(C=None, stack=None)
    root.end = len(C.kind)
    return C, root


# --- a table read's gates -----------------------------------------------------------------------------------------------

#: a read's low index bits by index width: the prover's inline read (``ir_lower.READ_LO``, constant design step 1: 14 for the
#: measured MUFU tables' widths); another width takes the split with the fewest rows
READ_LO = {24: 14, 23: 14}
_TABLE_GATES: dict[str, dict] = {}


def _decode_ands(k: int) -> int:
    return 0 if k <= 1 else (1 << k) + _decode_ands(k // 2) + _decode_ands(k - k // 2)


def table_gates(table: str) -> dict:
    """The plain gate circuit of a read of a pinned table, as the prover lays it out (``ir_lower._Read``, constant design step
    1): one-hot decoders of the index's low and high bits (each minterm the AND of two half minterms, a NOT per index bit); per
    value bit j and high minterm h an AND of h with the XOR of the low minterms whose word has bit j set, where that XOR is not
    empty; each value bit the XOR of its products. A library table's bits that are constant over the whole table get no gates
    (they are its constant); a program table's every bit does. The table is constants in the XOR sums."""
    if table not in _TABLE_GATES:
        import numpy as np
        from verity.ml.circuits import library

        n, width = TP.TABLE_BITS[table], TP.TABLE_VALUE_BITS.get(table, TP.VALUE_BITS)
        words = np.asarray(gf2.table(table), dtype=np.uint32)
        lib = library.table_label(TT.table_sha512(table)) == library.LIBRARY
        bits = [j for j in range(width) if not (lib and (((words >> np.uint32(j)) & np.uint32(1)).all()
                                                         or not ((words >> np.uint32(j)) & np.uint32(1)).any()))]
        lo = READ_LO.get(n) or min(range(2, n - 1), key=lambda lb: (1 << lb) + (1 << (n - lb)) * (len(bits) + 1))
        seg = words.reshape(1 << (n - lo), 1 << lo)
        ands, xor = _decode_ands(lo) + _decode_ands(n - lo), 0
        step = max(1, (1 << 22) >> lo)
        for j in bits:
            prods = 0
            for r in range(0, seg.shape[0], step):
                c = ((seg[r:r + step] >> np.uint32(j)) & np.uint32(1)).sum(axis=1, dtype=np.int64)
                prods += int((c > 0).sum())
                xor += int(np.maximum(c - 1, 0).sum())
            ands += prods
            xor += max(prods - 1, 0)
        del words, seg
        _TABLE_GATES[table] = {"and": ands, "xor": xor, "not": n, "table": table, "sha512": TT.table_sha512(table),
                               "index_bits": n, "value_bits": width, "live_value_bits": len(bits), "low_index_bits": lo,
                               "label": library.LIBRARY if lib else library.PROGRAM,
                               "built_by": "the prover's inline read (verity_flock.ir_lower._Read, constant design step 1)"}
    return _TABLE_GATES[table]


def _read_cost(entry: tuple) -> dict:
    return table_gates(TT.table_name(entry[0]))


# --- the circuit types ---------------------------------------------------------------------------------------------------

def _display(fn: str) -> str:
    return DISPLAY.get(fn, fn)


def _evaluate_type(types: dict, d: str, C: Circuit, T: int = 64, seed: int = 0) -> bool:
    """Type ``d`` (the root of ``C``'s types) against the flat circuit ``C`` on T random vectors."""
    rng = random.Random(seed)
    ins = {name: [rng.getrandbits(T) for _ in bits] for name, bits in C.inputs.items()}
    want, _fail = C.evaluate(ins, T)
    words: dict[str, object] = {}

    def word(sha, i):
        if sha not in words:
            words[sha] = gf2.table(TT.table_name(sha))
        return int(words[sha][i])

    got = CT.evaluate(types, d, [x for bits in ins.values() for x in bits], T, word)
    return got == [x for bits in want.values() for x in bits]


class Registry:
    """Every circuit type (`circuit_types`) the export builds, by its digest: each distinct sub-circuit once, shared across
    templates and rows, with its names and counts as metadata."""

    def __init__(self, out: Path | None = None):
        self.nodes: dict[str, dict] = {}
        self.types: dict[str, CT.CircuitType] = {}
        self.counts: dict[str, dict] = {}
        self.roots = self.evaluated = 0
        #: per root: the flat circuit's ANDs and its types' (their difference: the sharing a type boundary stops)
        self.shared: list[tuple[str, int, int]] = []

    def type_of(self, build, name: str, kind: str, label: str | None = None) -> str:
        """The type of the circuit ``build()`` makes, and every type below it: validated, the root evaluated against the flat
        circuit. Returns the root's digest."""
        from verity.ml.circuits import library

        _patch()
        types, root, names = TT.trace_types(build, label or library.PROGRAM)
        for d, t in types.items():
            if d not in self.types:
                CT.validate(t, lambda x: types.get(x) or self.types[x])
        C = build()
        if not _evaluate_type(types, root, C):
            raise AssertionError(f"the types of {name} disagree with its circuit")
        self.evaluated += 1
        for d, t in types.items():
            self.types.setdefault(d, t)
        for d, t in types.items():
            self._record(d, t, names.get(d, "root"), name if d == root else None, kind if d == root else None)
        self.roots += 1
        reads = sum(table_gates(TT.table_name(e[0]))["and"] * n for e, n in self._reads(root).items())
        flat = C.counts()["ands"]
        self.shared.append((name, flat, self.nodes[root]["and"] - reads))
        self.nodes[root]["flat_and"] = flat + reads
        del C
        return root

    def _reads(self, d: str, memo: dict | None = None) -> dict:
        memo = {} if memo is None else memo
        if d not in memo:
            t, out = self.types[d], {}
            for it in t.items:
                if it[0] == CT.READ:
                    out[t.tables[it[1]]] = out.get(t.tables[it[1]], 0) + 1
                elif it[0] == CT.CALL:
                    for e, n in self._reads(t.calls[it[1]], memo).items():
                        out[e] = out.get(e, 0) + n
            memo[d] = out
        return memo[d]

    def _record(self, d: str, t: CT.CircuitType, fn: str, name: str | None, kind: str | None) -> None:
        if d in self.nodes:
            return
        c = TT.counts(self.types, d, _read_cost, self.counts)
        own = TT.own_counts(t)
        calls: dict[str, int] = {}
        reads: dict[tuple, int] = {}
        for it in t.items:
            if it[0] == CT.CALL:
                calls[t.calls[it[1]]] = calls.get(t.calls[it[1]], 0) + 1
            elif it[0] == CT.READ:
                reads[t.tables[it[1]]] = reads.get(t.tables[it[1]], 0) + 1
        fn_kind = "table" if reads and not calls and not own["and"] else KIND.get(fn, "piece" if fn in IL.PIECES else "subcircuit")
        rec = {"id": d, "name": name or _display(fn), "kind": kind or fn_kind, "function": fn, "label": t.label,
               "and": c["and"], "xor": c["xor"], "not": c["not"], "in_bits": t.in_bits, "out_bits": t.out_bits,
               "in": [list(p) for p in t.ins], "out": [list(p) for p in t.outs], "own": own,
               "children": [{"ref": r, "count": n} for r, n in calls.items()]}
        if reads:
            rec["reads"] = [{"table": TT.table_name(e[0]), "sha512": e[0], "index_bits": e[1], "value_bits": e[2], "count": n,
                             **{k: v for k, v in table_gates(TT.table_name(e[0])).items() if k in ("and", "xor", "not", "label")}}
                            for e, n in reads.items()]
        self.nodes[d] = rec


def write_modules(reg: Registry, path: Path) -> int:
    """``modules.json.gz``: ``{schema, format, types: {digest: content}}`` (``circuit_types``' content, its digest the key),
    streamed; returns its bytes."""
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8") as f:
        f.write(json.dumps({"schema": SCHEMA + "/modules", "format": CT.FORMAT})[:-1] + ',"types":{')
        for n, d in enumerate(sorted(reg.types)):
            f.write(("," if n else "") + json.dumps(d) + ":" + reg.types[d].canonical().decode())
        f.write("}}")
    return path.stat().st_size


# --- the definition walker (templates without a C-Flock unit lowering, attention per T, GEMM coordinates) ---------------

def _split_params(s: str) -> dict:
    out, depth, cur, key = {}, 0, "", None
    for ch in s:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "=" and depth == 0 and key is None:
            key, cur = cur, ""
            continue
        if ch == "," and depth == 0:
            out[key] = cur
            key, cur = None, ""
            continue
        cur += ch
    if key is not None:
        out[key] = cur
    return out


def parse_id(did: str) -> tuple[str, dict]:
    if "{" not in did:
        return did, {}
    name, rest = did.split("{", 1)
    return name, _split_params(rest[:-1])


class DefStore:
    """The program graphs' Definitions (``definitions/index.json`` + ``<digest[:2]>.json.gz`` shards), by id, plus those a
    graph carries inline (``definitions``: a graph not yet catalogued)."""

    def __init__(self, pg_dir: Path):
        self.dir = Path(pg_dir) / "definitions"
        self.ix = json.loads((self.dir / "index.json").read_text())["definitions"]
        self.shards: dict[str, dict] = {}
        self.inline: dict[str, dict] = {}
        self.by_name: dict[str, list] = {}
        for did in self.ix:
            name, ps = parse_id(did)
            self.by_name.setdefault(name, []).append((did, ps))

    def add_inline(self, defs: dict) -> None:
        for did, d in defs.items():
            if did not in self.ix and did not in self.inline:
                self.inline[did] = d
                name, ps = parse_id(did)
                self.by_name.setdefault(name, []).append((did, ps))

    def get(self, did: str):
        if did in self.inline:
            return self.inline[did]
        e = self.ix.get(did)
        if e is None:
            return None
        d = e["digest"]
        if d[:2] not in self.shards:
            self.shards[d[:2]] = json.loads(gzip.decompress((self.dir / f"{d[:2]}.json.gz").read_bytes()))
        return self.shards[d[:2]][d]["definition"]

    def find(self, name: str, **want) -> dict:
        """``{T: id}`` (or ``{None: id}``) of ``name``'s definitions whose statics include ``want`` (string compare)."""
        out = {}
        for did, ps in self.by_name.get(name, []):
            if all(ps.get(k) == v for k, v in want.items()):
                out[int(ps["T"]) if "T" in ps else None] = did
        return out


_PRIMS: dict | None = None


def prims() -> dict:
    """Every registered PrimitiveDefinition by id (verity.ml.circuits.ops and the vLLM registry modules)."""
    global _PRIMS
    if _PRIMS is None:
        import importlib
        import pkgutil

        from verity.ml.circuits.defs import PrimitiveDefinition

        mods = ["verity.ml.circuits.ops.prims"]
        import verity.ml.definitions.registry as R
        mods += [f"{R.__name__}.{m.name}" for m in pkgutil.iter_modules(R.__path__) if not m.name.startswith(("quarantine",))]
        _PRIMS = {}
        for m in mods:
            try:
                mod = importlib.import_module(m)
            except Exception:  # noqa: BLE001  (a registry module that needs a GPU stack or data)
                continue
            for v in vars(mod).values():
                if isinstance(v, PrimitiveDefinition):
                    _PRIMS.setdefault(v.id, v)
    return _PRIMS


def _width_prim(pid: str):
    """A primitive the registry makes per width on demand (``BitAtx{W}``, ``TopPMaskWordx{V}``), from its factory."""
    import re

    from verity.ml.definitions.registry import sampling

    for pat, make in ((r"BitAtx(\d+)_v1", sampling.bit_at), (r"TopPMaskWordx(\d+)_v1", sampling.topp_mask_word)):
        m = re.fullmatch(pat, pid)
        if m:
            return make(int(m.group(1)))
    return None


def _words(t) -> list[int]:
    k = t[0]
    if k == "Value":
        return [t[1]]
    if k == "Array":
        return _words(t[2]) * t[1]
    if k == "Tuple":
        return [w for x in t[1] for w in _words(x)]
    raise ValueError(f"type {t!r}")


def _type_bits(t) -> int:
    k = t[0]
    if k == "Value":
        return t[1]
    if k == "Array":
        return t[1] * _type_bits(t[2])
    if k == "Tuple":
        return sum(_type_bits(x) for x in t[1])
    raise ValueError(f"type {t!r}")


def _const_value(fn: str):
    if fn.startswith("Const") and "[" in fn:
        return int(fn.split("[", 1)[1].split("]", 1)[0], 16)
    return None


#: a lane counter's value: a different constant at every step of its scan
COUNTER = object()


class Walker:
    """Definitions as trees: children are sub-definitions (``call`` x1, ``batch`` / ``scan`` x n) and primitive leaves; a
    tensor-core step is the proof block; another primitive is its ir_lower piece (lowered once per operand signature, constant
    operands a Const node gives propagated) or not yet lowered."""

    def __init__(self, reg: Registry, store: DefStore, cache: dict):
        self.reg, self.store, self.cache = reg, store, cache
        self.defs: dict[str, dict] = {}
        self.openings: set[str] = set()

    def tc_unit(self, pid: str) -> str:
        key = ("tc", pid)
        if key not in self.cache:
            if pid in IL.TC_STEPS_FP8:
                def build():
                    C = Circuit(cse=True)
                    acc = C.input("in0", 32)
                    a = [C.input(f"in{1 + j}", 8) for j in range(32)]
                    b = [C.input(f"in{33 + j}", 8) for j in range(32)]
                    C.output("out0", fp.tc_dot_e4m3(C, acc, a, b, *IL.TC_STEPS_FP8[pid]))
                    return C
            else:
                groups, W, F = IL.TC_STEPS[pid]

                def build():
                    C = Circuit(cse=True)
                    b = [C.input(f"in{j}", 16) for j in range(16)]
                    a = [C.input(f"in{16 + j}", 16) for j in range(16)]
                    acc = C.input("in32", 32)
                    C.output("out0", fp.tc_dot16(C, acc, a, b, groups=groups, W=W, F=F))
                    return C
            from verity.ml.circuits import library

            self.cache[key] = self.reg.type_of(build, f"tensor-core k-step unit ({pid}, total)", "tc-step", library.definition_label(pid))
            gc.collect()
        return self.cache[key]

    def prim(self, pid: str, consts: tuple):
        """(child entry without count) for primitive ``pid`` with operand constants ``consts`` (None = free)."""
        if pid in IL.TC_STEPS or pid in IL.TC_STEPS_FP8:
            return {"ref": self.tc_unit(pid)}
        if (m := re.fullmatch(r"GatherBf16x(\d+)_v1", pid)) and GATHERS == "mux":
            return self.gather(int(m.group(1)))
        p = prims().get(pid) or _width_prim(pid)
        if p is None:
            return {"not_yet_lowered": pid, "reason": "primitive not in the registries this export reads"}
        widths = [w for _, t in p.params for w in t.leaf_widths()]
        consts = consts if len(consts) == len(widths) else (None,) * len(widths)
        if (m := re.fullmatch(r"TopPMaskWordx(\d+)_v1", pid)):
            return {"def": self.composed(pid, int(m.group(1)), consts[-2], consts[-1])}
        free = [w for c, w in zip(consts, widths) if c is None]
        if pid in IL.PIECES and free:
            ck = ("prim", pid, consts, tuple(widths))
            if ck not in self.cache:
                def build():
                    C = Circuit(cse=True)
                    ops = [const_bits(c, w) if c is not None else C.input(f"in{j}", w) for j, (c, w) in enumerate(zip(consts, widths))]
                    C.output("out0", IL.PIECES[pid](C, *ops))
                    return C
                try:
                    self.cache[ck] = self.reg.type_of(build, f"{pid} (outside the units)", "piece")
                    gc.collect()
                except Exception as e:  # noqa: BLE001  (the piece does not take these operands)
                    self.cache[ck] = str(e)
            got = self.cache[ck]
            if got not in self.reg.nodes:
                return {"not_yet_lowered": pid, "in_bits": sum(free), "out_bits": sum(p.ret.leaf_widths()), "reason": got}
            return {"ref": got}
        if not free:
            return None
        return {"not_yet_lowered": pid, "in_bits": sum(free), "out_bits": sum(p.ret.leaf_widths()),
                "reason": next((r for f, r in NO_PIECE.items() if pid.startswith(f)), "ir_lower has no piece for this primitive yet")}

    def gather(self, V: int) -> dict:
        """``GatherBf16x{V}_v1`` as its multiplexer (``tail_pieces.gather_bf16``): one circuit up to GATHER_INLINE candidates, else a
        composed Definition of its V - 1 candidate-pair muxes and its range select (the same gates, counted)."""
        ck = ("gather", V)
        if ck not in self.cache:
            def traced(name, build):
                rid = self.reg.type_of(build, name, "piece")
                gc.collect()
                return rid

            def whole():
                C = Circuit(cse=True)
                C.output("out0", TP.gather_bf16(C, C.input("in0", 32), *[C.input(f"in{1 + j}", 16) for j in range(V)]))
                return C

            def pair():
                C = Circuit(cse=True)
                C.output("out0", gf2.mux(C, C.input("in0", 1)[0], C.input("in1", 16), C.input("in2", 16)))
                return C

            def select():
                C = Circuit(cse=True)
                tok, w = C.input("in0", 32), C.input("in1", 16)
                C.output("out0", gf2.mux(C, TP._ult_const(C, tok, V), w, const_bits(0x7FFF, 16)))
                return C

            if V <= GATHER_INLINE:
                self.cache[ck] = {"ref": traced(f"GatherBf16x{V}_v1 (multiplexer)", whole)}
            else:
                rid = "def-" + hashlib.sha256(f"gather-mux:{V}".encode()).hexdigest()[:16]
                rec = {"id": rid, "name": f"GatherBf16x{V}_v1 (multiplexer)", "kind": "definition", "in_bits": 32 + 16 * V, "out_bits": 16,
                       "composed": "tail_pieces.gather_bf16: a tree of V - 1 candidate-pair muxes by the index's low bits, then the "
                                   "range select (0x7FFF outside [0, V)); too wide to hold as one circuit",
                       "children": [{"ref": traced("mux (candidate pair)", pair), "count": V - 1},
                                    {"ref": traced(f"range select (index < {V}, else 0x7FFF)", select), "count": 1}]}
                self.defs[rid] = rec
                self._total(rec)
                self.cache[ck] = {"def": rid}
        return self.cache[ck]

    def piece(self, name: str, key: tuple) -> str:
        """The subcircuit of ``topp_word`` piece ``name`` with constant operands ``key``."""
        ck = ("tw", name, key)
        if ck not in self.cache:
            consts = ", ".join(f"operand {j} = 0x{c:x}" for j, c in enumerate(key) if c is not None)
            self.cache[ck] = self.reg.type_of(lambda: TW.build(name, key), TW.NAMES[name] + (f" ({consts})" if consts else ""), "piece")
            gc.collect()
        return self.cache[ck]

    def composed(self, pid: str, V: int, p, splits) -> str:
        """A primitive too large to hold as one circuit (``TopPMaskWordx{V}_v1``: about 10^11 ANDs at V = 128256), as a tree of
        its stages (``topp_word``'s paths) whose leaves are its pieces with instance counts."""
        ck = ("composed", pid, p, splits)
        if ck in self.cache:
            return self.cache[ck]
        counts = TW.count(V, p, splits)
        tree: dict = {}
        for (path, name, key), n in counts.items():
            node = tree
            for label in path:
                node = node.setdefault(label, {})
            leaf = node.setdefault(None, {})
            ref = self.piece(name, key)
            leaf[ref] = leaf.get(ref, 0) + n
        tag = f"{pid} p={p} splits={splits}"

        def make(label, node, where):
            rid = "def-" + hashlib.sha256(f"composed:{tag}:{where}".encode()).hexdigest()[:16]
            rec = {"id": rid, "name": label, "kind": "definition", "children": []}
            for sub, kid in node.items():
                if sub is not None:
                    rec["children"].append({"def": make(sub, kid, f"{where}/{sub}"), "count": 1})
            rec["children"] += [{"ref": r, "count": n} for r, n in node.get(None, {}).items()]
            self.defs[rid] = rec
            self._total(rec)
            return rid

        top = make(pid, tree, "")
        self.defs[top].update(in_bits=V * 32 + 64, out_bits=V, composed=(
            "topp_word: the split top-p pipeline per split count (stages, then pieces with instance counts), too large to hold "
            "as one circuit; checked against sampling.topp_keep by evaluating every piece instance on test rows"))
        self.cache[ck] = top
        return top

    def walk(self, did: str, counters: frozenset = frozenset()) -> str:
        """Definition ``did`` as a tree. Constants propagate: a gate whose operands are all constant is folded (constant-api
        §1.3: constant sub-circuits are folded before the partition), as are a scan's lane counters (a carry leaf the step
        advances from constants alone: ``counters``, of the step's carry parameter), and a gate reading a constant is lowered
        with it; a gate reading a lane counter with free operands is lowered generic, as a unit's past its constant variants."""
        tag = did if not counters else f"{did}|lane counters {sorted(counters)}"
        rid = "def-" + hashlib.sha256(tag.encode()).hexdigest()[:16]
        if rid in self.defs:
            return rid
        d = self.store.get(did)
        name = did if not counters else f"{did} (carry leaf {', '.join(str(j) for _, j in sorted(counters))} a lane counter)"
        rec = {"id": rid, "name": name, "kind": "definition", "children": []}
        self.defs[rid] = rec
        if d is None:
            rec.update(kind="not-yet-lowered", reason="definition not in the program graphs' store")
            return rid
        if d.get("form") != "composite":
            rec["children"] = [self._leaf(did, (), 1)] if self._leaf(did, (), 1) else []
            self._total(rec)
            return rid
        nodes = d["body"]["nodes"]
        rec["in_bits"] = sum(_type_bits(prm[1]) for prm in d["body"]["params"])
        rec["out_bits"] = _type_bits(d["body"]["ret_type"])
        agg: dict = {}
        opened = OPENED_IN.get(parse_id(did)[0]) if GATHERS == "opening" else None
        if opened:
            ps = d["body"]["params"]
            opens = {"table": next(p[0] for p in ps if p[1][0] == "Array"), "index": next(p[0] for p in ps if p[1][0] == "Value"),
                     "committed": "weights"}
        cval: dict[int, object] = {}

        def const_of(ref):
            if not ref or ref[3] != 1:
                return None
            if ref[0] == "n":
                return cval.get(ref[1]) if ref[2] == 0 else None
            return COUNTER if ref[0] == "p" and (ref[1], ref[2]) in counters else None

        for i, n in enumerate(nodes):
            fn, form = n["fn"], n["form"]
            count = n.get("n", 1) if form in ("batch", "scan") else 1
            if (c := _const_value(fn)) is not None:
                cval[i] = c
                continue
            if opened and fn.startswith("Gather"):
                self.openings.add(fn)
                agg.setdefault(("open", fn), [{"commitment_opening": fn, "kind": "opening", "opens": opens, "reason": opened}, 0])[1] += count
                continue
            if fn in self.store.ix and (self.store.get(fn) or {}).get("form") == "composite":
                sub = self._counters(n, fn, const_of) if form == "scan" else frozenset()
                k = ("def", fn, sub)
                agg.setdefault(k, [{"def": self.walk(fn, sub)}, 0])[1] += count
                continue
            consts = []
            axes = n.get("axes") or [0] * len(n.get("args", []))
            for a, ax in zip(n.get("args", []), axes):
                ref = (a.get("refs") or {}).get("affine")
                if form == "primitive":
                    consts.extend([const_of(ref)] * len(_words(a["type"])))
                elif form == "batch" and ax is None:
                    consts.extend([const_of(ref)] * len(_words(a["type"])))
                elif form == "batch" and a["type"][0] == "Array":
                    consts.extend([None] * len(_words(a["type"][2])))
                else:
                    consts = None
                    break
            if consts and all(c is not None for c in consts):
                v = self._fold(fn, consts) if form == "primitive" else COUNTER
                if v is not None:
                    if form == "primitive":
                        cval[i] = v
                    continue
            consts = [None if c is COUNTER else c for c in consts] if consts is not None else None
            k = ("prim", fn, tuple(consts) if consts and any(c is not None for c in consts) or form == "primitive" and consts
                 else ())
            if k not in agg:
                e = self._leaf(fn, k[2], count)
                if e is None:
                    continue
                agg[k] = [e, 0]
            agg[k][1] += count
        rec["children"] = [dict(e, count=c) for e, c in agg.values()]
        self._total(rec)
        return rid

    def _fold(self, pid: str, consts: list):
        """A gate of constant (or lane-counter) operands: its value, COUNTER when a counter reads into it, or None when it
        cannot be folded here (not one word, or no evaluator)."""
        p = prims().get(pid) or _width_prim(pid)
        if p is None or len(p.ret.leaf_widths()) != 1:
            return None
        if any(c is COUNTER for c in consts):
            return COUNTER
        try:
            return int(p.evaluate(*consts))
        except Exception:  # noqa: BLE001  (an evaluator over arrays, or a partial one: leave the gate to its piece)
            return None

    def _counters(self, n: dict, step: str, const_of) -> frozenset:
        """A scan's lane counters: the carry leaves whose initial value is constant and that the step advances from that leaf
        and constants alone (``TokenSelect``'s index, the Gumbel select's lane)."""
        init = (n["args"][0].get("refs") or {}) if n.get("args") else {}
        body = (self.store.get(step) or {}).get("body") or {}
        ret = body.get("ret") or {}
        if "concat" not in init or "concat" not in ret or "concat" not in (ret["concat"][0] or {}):
            return frozenset()
        snodes = body["nodes"]
        out = set()
        for j, (lr, orf) in enumerate(zip(init["concat"], ret["concat"][0]["concat"])):
            o = orf.get("affine")
            if const_of(lr.get("affine")) is None or not o or o[0] != "n" or o[3] != 1:
                continue
            sn = snodes[o[1]]
            refs = [(a.get("refs") or {}).get("affine") for a in sn.get("args", [])]
            if sn["form"] == "primitive" and refs and any(r and r[0] == "p" for r in refs) and all(
                    r and r[3] == 1 and ((r[0] == "p" and r[1] == 0 and r[2] == j)
                                         or (r[0] == "n" and _const_value(snodes[r[1]]["fn"]) is not None)) for r in refs):
                out.add((0, j))
        return frozenset(out)

    def _leaf(self, pid, consts, count):
        return self.prim(pid, consts)

    def _total(self, rec):
        t = {"and": 0, "xor": 0, "not": 0, "flat_and": 0, "not_yet_lowered": 0}
        opened = 0
        for c in rec["children"]:
            if "def" in c:
                s = self.defs[c["def"]]
                for k in t:
                    t[k] += s.get(k, s.get("and", 0) if k == "flat_and" else 0) * c["count"]
                opened += s.get("commitment_openings", 0) * c["count"]
                if s["kind"] == "not-yet-lowered":
                    t["not_yet_lowered"] += c["count"]
            elif "ref" in c:
                n = self.reg.nodes[c["ref"]]
                for k in ("and", "xor", "not"):
                    t[k] += n[k] * c["count"]
                t["flat_and"] += n.get("flat_and", n["and"]) * c["count"]
            elif "commitment_opening" in c:
                opened += c["count"]
            else:
                t["not_yet_lowered"] += c["count"]
        rec.update(t)
        if opened:
            rec["commitment_openings"] = opened


# --- commitment marks: the no-recompute partition (verity.ml.definitions.word, Q_word_v1{X=16,W=32,R=no-recompute}) -------------

X_BITS, W_BITS = 16, 32
#: constant-operand variants of one primitive lowered per record; further gates use the generic piece (an unrolled scan gives every
#: lane its own constant index: GumbelNoiseLane, BitAtx)
CONST_VARIANTS = 8
UNIT_PARTS_LIMIT = 4096


class Commitments:
    """Each Call specialization's cut under the no-recompute partition, by its descriptor id.

    A record: the partition's totals; ``units``, its classes (as ``word.unit_rule`` groups them: kind, head primitive @ activation,
    word gates, |out(S)|, count per Call), each with every primitive of one unit and that unit's AND/XOR/NOT from the pieces, and a
    unit of at most UNIT_PARTS_LIMIT word gates also as a part graph (its parts index the record's ``pieces``); ``committed``,
    ``[activation, body node, kind (MARK_KINDS), values]``: every value the cut commits, counted at the Definition body node that
    computes it; ``structure``, the gates in no unit (constants, wiring); ``violations``. Activations are indices into the record's
    ``scopes`` (descriptor ids, as the program graphs' Definitions). A separable Definition (a Gemm's coordinates,
    attention's heads) is its members' cuts: its record lists ``separable`` and its unit classes point at the member record holding
    them (``graph``), as in ``word.unit_rule``."""

    def __init__(self, walker: "Walker"):
        self.walker = walker
        self.records: dict[str, dict] = {}
        self.openings = walker.openings

    def of(self, fn) -> str:
        from verity.ml.circuits.codec import _spec_id
        from verity.ml.circuits.cut import separable
        from verity.ml.circuits.defs import PrimitiveDefinition
        import verity.ml.definitions.word as WQ

        fid = _spec_id(fn)
        if fid in self.records:
            return fid
        if isinstance(fn, PrimitiveDefinition):
            rec = self._primitive(fn, fid, WQ)
        elif (sep := separable(fn)) is not None:
            rec = self._combine(fid, fn.gates, [(self.records[self.of(c)], m) for c, m in sep])
        elif fn.gates > (limit := WQ.gate_limit(fn)):
            rec = _empty(fid, fn.gates)
            rec["violations"] = [{"class": "too-large", "gates": fn.gates, "limit": limit}]
        else:
            rec = self._partition(fn, fid, WQ)
        self.records[fid] = rec
        return fid

    def _boolean(self, gates: list[tuple[str, tuple]], pieces: dict) -> dict:
        """The gates' AND/XOR/NOT from their pieces (constant operands propagated), and each gate's index in ``pieces``
        (``{"ref": subcircuit}``, ``{"not_yet_lowered": primitive}``, ``{"commitment_opening": primitive}`` for the
        embedding's gather)."""
        t = {"and": 0, "xor": 0, "not": 0}
        nyl: dict[str, int] = {}
        parts: list[int] = []
        memo: dict = {}
        variants: dict[str, int] = {}
        generic = 0
        for sig in gates:
            if sig not in memo and any(c is not None for c in sig[1]):
                variants[sig[0]] = variants.get(sig[0], 0) + 1
                if variants[sig[0]] > CONST_VARIANTS:
                    generic += 1
                    sig = (sig[0], (None,) * len(sig[1]))
            if sig not in memo:
                memo[sig] = self.walker.prim(*sig)
            e = memo[sig]
            if sig[0] in self.openings:
                key = ("commitment_opening", sig[0])
            elif e is not None and "ref" in e:
                n = self.walker.reg.nodes[e["ref"]]
                for k in t:
                    t[k] += n[k]
                key = ("ref", e["ref"])
            elif e is not None and "def" in e:
                n = self.walker.defs[e["def"]]
                for k in t:
                    t[k] += n[k]
                key = ("def", e["def"])
            elif e is not None:
                nyl[sig[0]] = nyl.get(sig[0], 0) + 1
                key = ("not_yet_lowered", sig[0])
            else:
                key = ("constant", True)
            parts.append(pieces.setdefault(key, len(pieces)))
        return {**t, "not_yet_lowered": nyl, "parts": parts, "generic": generic}

    def _primitive(self, fn, fid, WQ) -> dict:
        rec = _empty(fid, 1)
        if not fn.params or fn.id in WQ.WIRING:
            rec.update(structure_gates=1, structure={fn.id: 1})
            return rec
        leaves = [(name, w) for name, t in fn.params for w in t.leaf_widths()]
        pieces: dict = {}
        b = self._boolean([(fn.id, (None,) * len(leaves))], pieces)
        reads: dict = {}
        for name, w in leaves:
            reads[name] = reads.get(name, 0) + w
        rec.update(scopes=[fid], pieces=_pieces(pieces), units_per_call=1, by_kind={"output": 1}, committed=[[0, 0, 0, 1]],
                   units=[{"kind": "output", "head": fn.id, "scope": 0, "count": 1, "word_gates": 1, "out_bits": fn.ret.w, "out_gates": 1,
                           "ok": fn.ret.w <= X_BITS or fn.ret.w <= W_BITS, "nodes": {fn.id: 1},
                           **{k: b[k] for k in ("and", "xor", "not", "not_yet_lowered", "parts")}, "part_nodes": [[0, 0]], "wiring": [],
                           "reads": [[-1, 0, w, name] for name, w in reads.items()], "outputs": [0]}])
        return rec

    def _partition(self, fn, fid, WQ) -> dict:
        from collections import Counter

        import numpy as np
        from verity.ml.circuits.codec import _spec_id
        from verity.ml.circuits.layout import WIDE_FAN_IN

        G = WQ.Graph(fn)
        R = WQ.units(G, X_BITS, W_BITS)
        owner, kinds = R["owner"], R["kinds"]
        own2, _k, head = WQ.partition(G, X_BITS)
        if not (own2 == owner).all():
            raise AssertionError(f"{fid}: the partition is not deterministic")
        from verity.ml.circuits.cut import call_scope
        c, call = call_scope(fn)
        lo = call.offset
        leaves = _leaf_lookup(fn.signature()[0])
        scopes: dict[str, int] = {}
        pieces: dict = {}
        sid: dict[int, int] = {}
        cval: dict[int, int | None] = {}
        computing = set(G.keep.tolist())

        def const(o):
            if o in cval:
                return cval[o]
            cval[o] = None
            if not 0 <= o - lo < fn.gates or o - lo in computing:
                return None
            g = c.gate(o)
            ops = g.operands
            if not g.prim.params:
                v = g.prim.evaluate()
            elif len(ops) <= WIDE_FAN_IN and all(const(x) is not None for x in ops):
                v = g.prim.evaluate(*[cval[x] for x in ops])
            else:
                return None
            cval[o] = int(v) if isinstance(v, (int, bool)) else None
            return cval[o]

        def where(i):
            g = c.gate(lo + int(G.keep[i]))
            s = g.scope
            if id(s.spec) not in sid:
                sid[id(s.spec)] = scopes.setdefault(_spec_id(s.spec), len(scopes))
            return sid[id(s.spec)], int(s.spec.body.node_at(g.index - s.offset)), g

        kind_ix = np.array([("output", "committed", "dead").index(k) for k in kinds], np.int64)
        cols = np.stack([kind_ix, R["head_prim"], R["head_scope"], R["gates"], R["out_bits"], R["out_gates"]], 1)
        keys, first, count, inv = [], [], [], np.zeros(0, np.int64)
        if len(kinds):
            keys, first, inv, count = np.unique(cols, axis=0, return_index=True, return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
        members: dict[int, list[int]] = {int(u): [] for u in first}
        for i, u in enumerate(owner.tolist()):
            if u in members:
                members[u].append(i)
        order = sorted(range(len(first)), key=lambda j: (-int(count[j]), int(keys[j][0]), int(keys[j][2])))
        ireads = np.array(G.input_reads, dtype=np.int64).reshape(-1, 2)
        rank = {j: n for n, j in enumerate(order)}
        units = []
        for j in order:
            kd, hp, hs, ng, ob, og = (int(x) for x in keys[j])
            u = int(first[j])
            gates = members[u]
            info = [where(i) for i in gates]
            sig = [(g.prim.id, tuple(const(o) for o in g.operands) if len(g.operands) <= WIDE_FAN_IN else (None,) * len(g.operands))
                   for _, _, g in info]
            b = self._boolean(sig, pieces)
            name = G.scope_names.get(hs)
            e = {"kind": ("output", "committed", "dead")[kd], "head": G.prim_names.get(hp),
                 "scope": next((s for s, _k, g in info if g.scope.spec.id == name), None), "count": int(count[j]), "word_gates": ng,
                 "out_bits": ob, "out_gates": og, "ok": bool(R["ok"][u]),
                 "nodes": dict(Counter(G.prim_names[int(G.prim[i])] for i in gates).most_common()),
                 **{x: b[x] for x in ("and", "xor", "not", "not_yet_lowered")}}
            if b["generic"]:
                e["constants_not_propagated"] = b["generic"]
            if ng <= UNIT_PARTS_LIMIT:
                e.update(parts=b["parts"], **_unit_graph(G, owner, head, inv, rank, u, gates, info, leaves, ireads))
            units.append(e)
        marks: Counter = Counter()
        for i in np.flatnonzero(head >= 0).tolist():
            s, k, _g = where(i)
            marks[(s, k, MARK_KINDS.index(kinds[int(head[i])]))] += 1
        cut = R["cut"]
        rec = _empty(fid, G.total)
        rec.update(scopes=sorted(scopes, key=scopes.__getitem__), pieces=_pieces(pieces), structure_gates=G.free,
                   redundant_gates=cut.detail.get("redundant_gates", 0),
                   units_per_call=len(kinds), by_kind=dict(Counter(kinds)),
                   committed_interior_words=sum(e["out_gates"] * e["count"] for e in units if e["kind"] == "committed"),
                   units=units, committed=[[s, k, kd, n] for (s, k, kd), n in sorted(marks.items())],
                   structure=dict(G.free_by_prim.most_common()),
                   violations=([] if cut.ok else [{"class": "cut", "codes": list(cut.codes)}])
                   + [{"class": f"{e['kind']}:{e['head']}", "out_bits": e["out_bits"], "units": e["count"]} for e in units if not e["ok"]])
        return rec

    def _combine(self, fid, gates, parts) -> dict:
        from collections import Counter

        rec = _empty(fid, gates)
        agg: dict[tuple, dict] = {}
        scopes: dict[str, int] = {}
        kinds: Counter = Counter()
        structure: Counter = Counter()
        for r, m in parts:
            for e in r["units"]:
                where = e.get("graph", r["id"])
                scope = scopes.setdefault(r["scopes"][e["scope"]], len(scopes)) if e["scope"] is not None else None
                k = (e["kind"], e["head"], scope, e["word_gates"], e["out_bits"], e["out_gates"], where)
                if k not in agg:
                    agg[k] = {x: v for x, v in e.items() if x not in ("parts", "part_nodes", "wiring", "reads", "outputs")}
                    agg[k].update(scope=scope, count=0, graph=where)
                agg[k]["count"] += e["count"] * m
            for kd, n in r["by_kind"].items():
                kinds[kd] += n * m
            for p, n in r["structure"].items():
                structure[p] += n * m
            for f in ("structure_gates", "redundant_gates", "units_per_call", "committed_interior_words"):
                rec[f] += r[f] * m
            rec["violations"] += [dict(v, units=v.get("units", 1) * m) for v in r["violations"]]
        rec.update(units=sorted(agg.values(), key=lambda e: (-e["count"], e["kind"], str(e["scope"]))), by_kind=dict(kinds),
                   scopes=sorted(scopes, key=scopes.__getitem__), structure=dict(structure.most_common()),
                   separable=[[r["id"], m] for r, m in parts])
        rec.pop("committed")
        rec.pop("pieces")
        return rec


MARK_KINDS = ("output", "committed")


def _pieces(pieces: dict) -> list[dict]:
    return [{k: v} for (k, v), _i in sorted(pieces.items(), key=lambda kv: kv[1])]


def _empty(fid, gates) -> dict:
    return {"id": fid, "word_gates": gates, "structure_gates": 0, "redundant_gates": 0, "units_per_call": 0, "by_kind": {},
            "committed_interior_words": 0, "scopes": [], "pieces": [], "units": [], "committed": [], "structure": {}, "violations": []}


def _leaf_lookup(params):
    """``leaf(k) -> (parameter name, width)`` for the Call's k-th input leaf, without listing the leaves (an embedding table
    parameter has over 10^8)."""
    from verity.ml.circuits.types import Array, Record, Tuple

    starts, names, types, o = [], [], [], 0
    for name, t in params:
        starts.append(o)
        names.append(name)
        types.append(t)
        o += t.leaves

    def leaf(k):
        i = bisect.bisect_right(starts, k) - 1
        t, k = types[i], k - starts[i]
        while True:
            if isinstance(t, Array):
                t, k = t.elem, k % t.elem.leaves
            elif isinstance(t, (Tuple, Record)):
                for f in t.fields:
                    f = f[1] if isinstance(t, Record) else f
                    if k < f.leaves:
                        t = f
                        break
                    k -= f.leaves
            else:
                return names[i], t.w

    return leaf


def _unit_graph(G, owner, head, inv, rank, u, gates, info, leaves, ireads) -> dict:
    """One unit's part graph: ``parts`` are its word gates in layout order (``part_nodes``: each one's activation and body node),
    ``wiring`` ``[src part, dst part, bits]`` inside the unit (a read through wiring stands for a read of its source); ``reads``
    ``[unit class, dst part, bits]`` of a value another unit commits, ``[-1, dst part, bits, parameter]`` of the Call's inputs;
    ``outputs``: the parts whose value the cut commits (returned, or read by another unit)."""
    import numpy as np

    at = {i: n for n, i in enumerate(gates)}
    sel = np.flatnonzero(owner[G.dst] == u)
    src, dst = G.src[sel], G.dst[sel]
    inside = owner[src] == u
    wiring = [[at[a], at[b], int(G.width[a])] for a, b in zip(src[inside].tolist(), dst[inside].tolist())]
    reads = [[rank[int(inv[owner[a]])], at[b], int(G.width[a])] for a, b in zip(src[~inside].tolist(), dst[~inside].tolist())]
    inputs: dict[tuple, int] = {}
    if len(ireads):
        for inp, b in ireads[owner[ireads[:, 1]] == u].tolist():
            name, w = leaves(G.input_ids[inp])
            inputs[(at[b], name)] = inputs.get((at[b], name), 0) + w
    reads += [[-1, d, w, name] for (d, name), w in inputs.items()]
    return {"part_nodes": [[s, k] for s, k, _g in info], "wiring": wiring, "reads": reads,
            "outputs": [at[i] for i in gates if head[i] >= 0]}


# --- templates and rows -------------------------------------------------------------------------------------------------

#: program-graph template -> spine template (C-Flock's lowering module, or the GEMM coordinate as tensor-core k-steps)
SPINE = {"RoPEHead": "rope-head", "SiluMul_v1": "silu-mul", "RMSNormFusedCuda_v2": "rmsnorm-fused-cuda",
         "RMSNormTriton_v1": "rmsnorm-triton", "GemmCoordinate": "gemm-coordinate", "AttentionHead": "attention-head"}
NOT_LOWERED: dict[str, str] = {}
#: primitive families with no piece, and why (anything else: no piece yet)
NO_PIECE: dict[str, str] = {}
#: Definitions whose Gather primitives read a committed weight table at a computed index: checked by opening it, not as a circuit
OPENED_IN = {
    "MoeExpertRow_v1": "checked by opening the committed expert weight bank at the routed expert (a commitment opening, not a "
                       "circuit), as the embedding's row gather: MoeExpertRow_v1 is the Embedding_v1 construction with the expert "
                       "axis in the vocabulary's role; as a circuit it would be an E-way mux tree per weight element (1,008 ANDs at "
                       "E = 64, 2,032 at E = 128)",
}
#: templates checked by opening the committed table, not as a circuit (benchmarks/census/subcircuits.json commitment_opening)
OPENED = {
    "Embedding_v1": "checked by opening the committed embedding table at the token's row (a commitment opening, not a circuit)",
    "EmbeddingShard_v1": "checked by opening the tensor-parallel shard of the committed embedding table at the token's row "
                         "(a commitment opening, not a circuit)",
}
DOT_SEMANTICS = {"AmpereBF16TcDot16_v1": "sm80.mma.m16n8k16.bf16", "AmpereBF16TcDot16_v2": "sm80.mma.m16n8k16.bf16"}


def _params(tname: str, tp: dict) -> dict:
    out = {}
    for k, v in tp.items():
        if k in ("dot", "inv"):
            continue
        if isinstance(v, str) and v.startswith("{"):
            v = float.fromhex(json.loads(v)["f64"])
        out[k] = v
    if tname == "GemmCoordinate":
        out["semantics"] = DOT_SEMANTICS[tp["dot"]]
    return out


def _constants(call) -> dict[int, int]:
    """The gates of ``call`` whose value is a constant: its Const gates, and every one-word gate whose operands all are (a
    constant sub-circuit, folded before the partition: constant-api §1.3)."""
    cval: dict[int, int] = {}
    for g in call:
        fam = getattr(g.prim, "family", None)
        if fam == "Const":
            cval[g.index] = int(g.prim.evaluate())
        elif fam != "Input" and g.operands and all(o in cval for o in g.operands) and len(g.prim.ret.leaf_widths()) == 1:
            try:
                cval[g.index] = int(g.prim.evaluate(*[cval[o] for o in g.operands]))
            except Exception:  # noqa: BLE001  (a partial evaluator: leave the gate to its piece)
                pass
    return cval


def _tail(reg: Registry, units, target, cache: dict) -> list[dict]:
    """The template's gates outside its units: each (primitive, constant operands) once with its count, lowered with its
    ``ir_lower`` piece where one exists, else marked not yet lowered."""
    _prog, call = IL.standalone(target)
    by = {g.index: g for g in call}
    cval = _constants(call)
    groups: dict = {}
    for g in units.tail:
        fam = getattr(g.prim, "family", None)
        if fam in ("Input", "Const") or g.index in cval:
            continue
        consts = tuple(cval.get(o) for o in g.operands)
        widths = tuple(by[o].prim.ret.w for o in g.operands)
        key = (g.prim.id, consts, widths)
        if key not in groups:
            groups[key] = [g.prim, 0]
        groups[key][1] += 1
    out = []
    for (pid, consts, widths), (prim, n) in groups.items():
        free = [w for c, w in zip(consts, widths) if c is None]
        if pid in IL.PIECES and free:
            ck = (pid, consts, widths)
            if ck not in cache:
                def build(pid=pid, consts=consts, widths=widths):
                    C = Circuit(cse=True)
                    ops = [const_bits(c, w) if c is not None else C.input(f"in{j}", w) for j, (c, w) in enumerate(zip(consts, widths))]
                    C.output("out0", IL.PIECES[pid](C, *ops))
                    return C
                try:
                    cache[ck] = reg.type_of(build, f"{pid} (outside the units)", "piece")
                    gc.collect()
                except Exception as e:  # noqa: BLE001  (the piece does not take these operands)
                    cache[ck] = str(e)
            if cache[ck] not in reg.nodes:
                out.append({"not_yet_lowered": pid, "count": n, "in_bits": sum(free), "out_bits": prim.ret.w, "reason": cache[ck]})
                continue
            out.append({"ref": cache[ck], "count": n})
        elif free:
            prev = next((e for e in out if e.get("not_yet_lowered") == pid and e["in_bits"] == sum(free)), None)
            if prev:
                prev["count"] += n
            else:
                out.append({"not_yet_lowered": pid, "count": n, "in_bits": sum(free), "out_bits": prim.ret.w,
                            "reason": "ir_lower has no piece for this primitive yet"})
    return out


def _io_bits(target) -> tuple[int, int]:
    _prog, call = IL.standalone(target)
    by = {g.index: g for g in call}
    args = list(call.args)
    leaf = {a: IL._leaf_width(target, i) for i, a in enumerate(args)}
    return sum(leaf.values()), sum(by[r].prim.ret.w if r in by else leaf[r] for r in call.returns)


def _unit(reg: Registry, units, name: str, cache: dict) -> str:
    key = ("unit", units.signature if units.builder is None else "tc")
    if key not in cache:
        cache[key] = reg.type_of(units.circuit, name, "c-flock-unit")
        gc.collect()
    return cache[key]


SPINE_UNITS = {"RoPEHead": "rope-head", "SiluMul_v1": "silu-mul", "RMSNormFusedCuda_v2": "rmsnorm-fused-cuda",
               "RMSNormTriton_v1": "rmsnorm-triton"}
HOPPER_DOT = '{"fn":"HopperBF16WgmmaDot16_v1"}'


def _attention_ids(store: DefStore, tname: str, tp: dict, statics: dict) -> dict:
    if tname == "AttentionSoftcap_v1":
        return store.find("AttentionSoftcap_v1", NH=str(tp["NH"]), KVH=str(tp["KVH"]), D=str(tp["D"]), BN=str(tp["BN"]), CAP=tp["CAP"])
    if tp.get("dot") == "HopperBF16WgmmaDot16_v1":
        return store.find("AttentionHead_v2", D=str(tp["D"]), BN=str(tp["BN"]), DOT=HOPPER_DOT, INV='{"fn":"%s"}' % tp["inv"])
    return store.find("AttentionHead_v3", D=str(tp["D"]), BN=str(tp["BN"]))


def template(reg: Registry, tname: str, tp: dict, T_range=None, cache: dict | None = None, walker: "Walker | None" = None,
             spec: str | None = None, member: str | None = None) -> dict:
    """A leaf template's record: its proof blocks under the committed cut and its tail (C-Flock's unit lowerings), or its
    Definition tree (the walker: attention per T, GEMM coordinates, every other template). ``member``: the Definition one
    instance of the group is (a GEMM's output coordinate) when the Call is a batch of them; the record then counts one."""
    cache = {} if cache is None else cache
    rid = "tpl-" + hashlib.sha256(json.dumps([tname, tp, T_range] + ([member] if member else []), sort_keys=True).encode()).hexdigest()[:16]
    base = {"id": rid, "template": tname, "params": tp}
    if tname.startswith(("Const32", "Const64")):
        return dict(base, name=tname, kind="constant", bind="run", **{"and": 0, "xor": 0, "not": 0}, children=[])
    if tname in OPENED and GATHERS == "opening":
        return dict(base, name=tname, kind="commitment-opening", reason=OPENED[tname], children=[])
    if tname in NOT_LOWERED:
        return dict(base, name=tname, kind="not-yet-lowered", reason=NOT_LOWERED[tname], children=[])
    if tname in ("AttentionHead", "AttentionSoftcap_v1"):
        ids = _attention_ids(walker.store, tname, tp, {})
        per_t = {}
        for T in range(T_range[0], T_range[1] + 1):
            if T not in ids:
                per_t[str(T)] = {"not_yet_lowered": f"T={T}", "reason": "this key count's Definition is not in the program graphs' store"}
                continue
            d = walker.defs[walker.walk(ids[T])]
            per_t[str(T)] = {"definition": d["id"], "name": d["name"], "children": [{"def": d["id"], "count": 1}],
                             **{k: d.get(k) for k in ("and", "xor", "not", "flat_and", "not_yet_lowered", "in_bits", "out_bits")}}
        top = per_t[str(T_range[1])]
        return dict(base, name=f"{tname}{json.dumps(tp, sort_keys=True)} T∈[{T_range[0]},{T_range[1]}]", kind="template",
                    children=top.get("children", []), children_at=f"T={T_range[1]}", per_key_count=per_t,
                    in_bits=top.get("in_bits"), out_bits=top.get("out_bits"))
    if member is not None:
        name = f"{tname}{json.dumps(tp, sort_keys=True)}" if member.startswith("GemmCoordinate") else member
        return _walked(base, walker, member, name)
    if tname == "GemmCoordinate":
        ids = (walker.store.find("GemmCoordinate_v2", K=str(tp["K"]), DOT=HOPPER_DOT) if tp.get("dot") == "HopperBF16WgmmaDot16_v1"
               else walker.store.find("GemmCoordinate_v1", K=str(tp["K"])))
        return _walked(base, walker, ids.get(None), f"GemmCoordinate{json.dumps(tp, sort_keys=True)}")
    if tname in SPINE_UNITS:
        from verity_numerical.bench import templates as TM

        sub = TM.subcircuit(SPINE_UNITS[tname], **_params(tname, tp))
        return _spine(reg, sub, base, walker, spec, f"{tname}{json.dumps(tp, sort_keys=True)}",
                      f"{tname}{json.dumps(_params(tname, tp), sort_keys=True)}", cache)
    return _walked(base, walker, spec, spec or tname)


def _spine(reg: Registry, sub, base: dict, walker: "Walker", spec: str | None, walked_name: str, name: str, cache: dict) -> dict:
    """A spine template's record: its C-Flock unit with the instance count and its tail, or, where the kernel's warps differ (no
    single unit), its Definition walked."""
    from verity_numerical.bench import lowerings

    mod = lowerings.registry("C-Flock").module(sub.template.id)
    if hasattr(mod, "width") and mod.width(int(sub.params["N"])) is None:
        rec = _walked(base, walker, spec, walked_name)
        rec["note"] = (f"at N = {sub.params['N']} the kernel's warps differ, so there is no single C-Flock unit "
                       f"(verity_flock.templates.{Path(mod.__file__).stem}.width); its Definition is walked instead, every "
                       f"primitive with its piece, none in a proof block")
        return rec
    low = mod.lowering(sub)
    uid = _unit(reg, low.units, f"{mod.UNIT} unit", cache)
    children = [{"ref": uid, "count": low.units.n, "c_flock_unit": True}] + _tail(reg, low.units, low.definition, cache)
    ib, ob = _io_bits(low.definition)
    return dict(base, name=name, kind="template", subcircuit=sub.id, children=children, unit=uid, in_bits=ib, out_bits=ob)


def _coordinate(store: DefStore, did: str | None) -> tuple[str, int] | None:
    """(member Definition id, members per Call) when the Definition ``did`` is only batches of one composite Definition (a
    GEMM's output coordinates, a MoE expert GEMM's, a block-scaled FP8 GEMM's), else None."""
    d = store.get(did) if did else None
    if not d or d.get("form") != "composite":
        return None
    nodes = d["body"]["nodes"]
    fns = {n["fn"] for n in nodes}
    if len(fns) != 1 or any(n["form"] != "batch" for n in nodes):
        return None
    (m,) = fns
    if (store.get(m) or {}).get("form") != "composite":
        return None
    return m, sum(n.get("n", 1) for n in nodes)


def _node_totals(node: dict, t: dict) -> dict:
    """A program-graph node's AND / XOR / NOT, and ``flat_and`` (its circuits' own ANDs, sharing across calls): its instances
    times its template's, per T where the T varies."""
    def get(x, k):
        return (x.get(k, x.get("and") if k == "flat_and" else 0) or 0)

    by_t = node.get("instances_by_T")
    if by_t:
        pk = t["per_key_count"]
        return {k: sum(n * get(pk[T], k) for T, n in by_t.items()) for k in ("and", "xor", "not", "flat_and")}
    return {k: (node.get("instances") or 0) * get(t, k) for k in ("and", "xor", "not", "flat_and")}


def _walked(base: dict, walker: "Walker", did: str | None, name: str) -> dict:
    if did is None:
        return dict(base, name=name, kind="not-yet-lowered", reason="its Definition is not in the program graphs' store", children=[])
    d = walker.defs[walker.walk(did)]
    return dict(base, name=name, kind="template", definition=d["id"], children=[{"def": d["id"], "count": 1}],
                in_bits=d.get("in_bits"), out_bits=d.get("out_bits"))


def _totals(reg: Registry, children: list[dict], defs: dict | None = None) -> dict:
    t = {"and": 0, "xor": 0, "not": 0, "flat_and": 0, "not_yet_lowered": 0}
    for c in children:
        if "def" in c:
            d = defs[c["def"]]
            for k in t:
                t[k] += (d.get(k, d.get("and") if k == "flat_and" else 0) or 0) * c["count"]
        elif "ref" in c:
            n = reg.nodes[c["ref"]]
            for k in ("and", "xor", "not"):
                t[k] += n[k] * c["count"]
            t["flat_and"] += n.get("flat_and", n["and"]) * c["count"]
        elif "commitment_opening" not in c:
            t["not_yet_lowered"] += c["count"]
    return t


def export_row(pg_path: Path, out: Path, reg: Registry, tpl_cache: dict, cache: dict, walker: "Walker", cm: "Commitments | None" = None,
               checks: dict | None = None) -> dict:
    p = json.loads(Path(pg_path).read_text())
    walker.store.add_inline(p.get("definitions") or {})
    row = p["row"]
    nodes, used = [], []
    into: dict[str, list[dict]] = {}
    for e in p.get("edges", []):
        into.setdefault(e["to"], []).append(e)
    tpl = {x["id"]: x["template"] for x in p["groups"]}
    for g in p["groups"]:
        tname, tp = g["template"], g.get("template_params") or {}
        T_range = g.get("varying", {}).get("T") if tname in ("AttentionHead", "AttentionSoftcap_v1") else None
        member = None
        if not T_range and g.get("instances") and g.get("calls") and g["instances"] != g["calls"]:
            c = _coordinate(walker.store, g.get("spec"))
            if c is None or g["instances"] != g["calls"] * c[1]:
                raise ValueError(f"{g['id']} ({g.get('spec')}): {g['instances']} instances in {g['calls']} Calls, and the Call is "
                                 f"not a batch of one Definition with that many members")
            member = c[0]
        key = json.dumps([tname, tp, T_range, member], sort_keys=True)
        if key not in tpl_cache:
            t0 = time.time()
            rec = template(reg, tname, tp, T_range, cache, walker, g.get("spec"), member)
            if rec["kind"] == "template":
                rec.update(_totals(reg, rec["children"], walker.defs))
            tpl_cache[key] = rec
            print(f"  template {rec['name'][:80]}: {rec['kind']} {time.time() - t0:.1f}s", flush=True)
        rec = tpl_cache[key]
        used.append(rec["id"])
        node = {"id": f"{row['row']}:{g['id']}", "name": g["module"] or "(request)", "kind": "program-node", "group": g["id"],
                "definition": g.get("spec") or g.get("definition"), "template": rec["id"], "instances": g.get("instances"),
                "calls": g.get("calls")}
        if member:
            node["instance"] = member
        if T_range:
            node["varying"] = {"T": T_range}
            vc = g.get("varying_calls") or {}
            heads, rest = divmod(g["instances"], g["calls"])
            if rest or sum(vc.values()) != g["calls"]:
                raise ValueError(f"{g['id']}: the T histogram ({sum(vc.values())} Calls) does not cover its {g['calls']} Calls")
            node["instances_by_T"] = {str(T): n * heads for T, n in sorted(vc.items(), key=lambda kv: int(kv[0]))}
        node["totals"] = _node_totals(node, rec)
        node.update(_roles(g, into.get(g["id"], []), tpl, walker.store))
        if cm is not None:
            node.update(_commitment(g, cm, checks))
        nodes.append(node)
    totals = {k: sum(n["totals"][k] for n in nodes) for k in ("and", "xor", "not", "flat_and")}
    return {"schema": SCHEMA + "/row", "row": row["row"], "row_key": row["serving"]["row_key"], "model": row["serving"].get("model"),
            "program_graph": Path(pg_path).name, "totals": totals, "nodes": nodes, "edges": p.get("edges", []),
            "templates": sorted(set(used))}


class _Encoded:
    """A store of Definitions encoded here, in the codec's v0 form a program graph's store holds them in, for a subcircuit no
    row carries."""

    def __init__(self):
        self.ix: dict[str, dict] = {}

    def add(self, fn) -> str:
        from verity.ml.definitions.encoding import encode_specialization

        top, defs = encode_specialization(fn)
        self.ix.update(defs)
        return top

    def get(self, did: str):
        return self.ix.get(did)


def statics_key(statics: dict) -> str:
    """Instance statics as one key: canonical JSON (sorted keys, no spaces), ``{}`` for a subcircuit whose statics never vary."""
    return json.dumps(statics, sort_keys=True, separators=(",", ":"))


def subcircuit(sub, statics=({},)) -> dict:
    """One bound subcircuit's Boolean export (``verity_numerical.bench.templates.Subcircuit``), self-contained: a template record
    per instance statics (``variants``, keyed by :func:`statics_key`; attention one per key count T), as a row's templates are
    recorded (a spine template its C-Flock unit and tail, any other its Definition walked), with every definition, subcircuit
    node and circuit type (``circuit_types`` content, keyed by digest) they reach. Raises where the subcircuit has no core
    Definition (a silicon reference)."""
    reg, cache, store = Registry(), {}, _Encoded()
    walker = Walker(reg, store, cache)
    variants = {}
    for st in statics:
        top = store.add(sub.definition(**st))
        base = {"id": "tpl-" + hashlib.sha256(json.dumps([sub.id, st], sort_keys=True).encode()).hexdigest()[:16],
                "template": sub.template.id, "params": sub.params, "statics": st}
        if sub.template.id in SPINE_UNITS.values():
            rec = _spine(reg, sub, base, walker, top, top, f"{sub.id} ({top})", cache)
        else:
            rec = _walked(base, walker, top, top)
        if rec["kind"] == "template":
            rec.update(_totals(reg, rec["children"], walker.defs))
        variants[statics_key(st)] = rec
    defs, refs, todo = {}, set(), [c for r in variants.values() for c in r["children"]]
    while todo:
        c = todo.pop()
        if "def" in c and c["def"] not in defs:
            defs[c["def"]] = walker.defs[c["def"]]
            todo += defs[c["def"]]["children"]
        elif "ref" in c:
            refs.add(c["ref"])
    types, todo = {}, list(refs)
    while todo:
        d = todo.pop()
        if d not in types:
            types[d] = reg.types[d]
            todo += list(types[d].calls)
    return {"schema": SCHEMA + "/subcircuit", "subcircuit": sub.id, "template": sub.template.id, "params": sub.params,
            "gathers": GATHERS, "library_digest": _library_digest(), "variants": variants, "definitions": defs,
            "nodes": {d: reg.nodes[d] for d in types}, "types": {"format": CT.FORMAT,
                                                                 "content": {d: t.content() for d, t in types.items()}}}


#: a template's node ANDs and its Calls' unit ANDs agree within this; beyond CROSSCHECK_FAIL (a granularity-sized error) the export fails
CROSSCHECK_AGREE, CROSSCHECK_FAIL = 0.02, 4.0


def crosscheck(doc: dict, tpl: dict, records: dict) -> dict:
    """Per template of a row: the nodes' ANDs (instances x the template's, per T) against the sum of their Calls' units under the
    no-recompute cut (each unit class's ANDs x its count x the Calls), attention at the T values that have a record."""
    out: dict = {}
    for n in doc["nodes"]:
        t = tpl[n["template"]]
        e = out.setdefault(t["name"], {"nodes": 0, "units": 0})
        if "commitment" in n:
            e["nodes"] += n["totals"]["and"]
            e["units"] += n["calls"] * sum((u.get("and") or 0) * u["count"] for u in records[n["commitment"]]["units"])
        elif "commitment_by_T" in n:
            heads = n["instances"] // n["calls"]
            for T, rid in n["commitment_by_T"].items():
                k = n["instances_by_T"].get(str(T), 0)
                e["nodes"] += k * t["per_key_count"][str(T)]["and"]
                e["units"] += k // heads * sum((u.get("and") or 0) * u["count"] for u in records[rid]["units"])
    for e in out.values():
        r = e["nodes"] / e["units"] if e["units"] else None
        e["ratio"] = r
        e["status"] = ("agree" if r is not None and abs(r - 1) <= CROSSCHECK_AGREE else
                       "differs" if r is not None and 1 / CROSSCHECK_FAIL <= r <= CROSSCHECK_FAIL else "fail" if e["units"] else "no units")
    return {k: v for k, v in out.items() if v["status"] != "no units"}


_CHECKED = {"units_per_call": "units_per_call", "gates": "word_gates", "free_gates": "structure_gates",
            "committed_interior_words": "committed_interior_words"}


def _commitment(g: dict, cm: "Commitments", checks: dict | None) -> dict:
    """The Call's cut under the no-recompute partition: its record id (``commitment``), or per T (``commitment_by_T``) for a group
    whose T varies, at the program graph's partition points (every T up to a few hundred, else the key-block boundaries), each
    checked against the program graph's own ``q_word_v1`` summary where it has one."""
    from verity.ml.definitions.encoding import word_values
    from verity.ml.definitions.word import specialization

    if g["template"].startswith(("Const32", "Const64")):
        return {}
    q = g.get("q_word_v1") or {}
    try:
        if g["definition"] in OPENED and GATHERS == "opening":
            from verity.ml.circuits.defs import PrimitiveDefinition

            fn = specialization(g["definition"], g["statics"])
            cm.openings.update(n.fn.id for n in fn.body.nodes if isinstance(n.fn, PrimitiveDefinition) and n.fn.id.startswith("Gather"))
        if g.get("varying") and len(g["varying"]) == 1 and g.get("varying_calls"):
            (a,) = g["varying"]
            vals = sorted(int(v) for v in g["varying_calls"])
            bn = g["statics"].get("BN") if a == "T" else None
            pts = word_values(vals, int(bn) if isinstance(bn, int) else None)
            ids = {v: cm.of(specialization(g["definition"], dict(g["statics"], **{a: v}))) for v in pts}
            if "per_call" in q and checks is not None:
                for end, v in (("min", vals[0]), ("max", vals[-1])):
                    r = cm.records[ids[v]]
                    ok = all(q["per_call"][f][end] == r[k] for f, k in _CHECKED.items() if f in q["per_call"])
                    checks["agree" if ok else "differ"].append(f"{g['id']} {g['definition']} {a}={v}")
            return {"commitment_by_T": {str(v): i for v, i in ids.items()},
                    "commitment_partitioned_at": "every T" if len(pts) == len(vals) else
                    f"{len(pts)} of {len(vals)} T values: the key-block boundaries and their neighbours; the others are not partitioned"}
        fid = cm.of(specialization(g["definition"], g["statics"]))
        if "per_call" in q and checks is not None:
            r = cm.records[fid]
            ok = all(q["per_call"][f] == r[k] for f, k in _CHECKED.items() if f in q["per_call"])
            checks["agree" if ok else "differ"].append(f"{g['id']} {g['definition']}")
        return {"commitment": fid}
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return {"commitment": None, "commitment_unresolved": f"{type(e).__name__}: {e}"[:300]}


def _input_role(name: str) -> str:
    role = name.split(":", 1)[0]
    return "peers" if role.startswith("peers_") else role


def _roles(g: dict, into: list[dict], tpl: dict[str, str], store: DefStore) -> dict:
    """A Call's parameters with their roles, as far as the program graph records them: a parameter another Call's output feeds
    (``edges[].ports``) is ``activations`` (``constant`` from a constant Call: a run constant, the root body's, so the parameter
    is bound ``run`` and no type holds its value); one a prescribed input feeds (``param_inputs``)
    takes the input's group (``weights``, ``prompt``, ``seed``, ``splits``, ``peers``). The Call's prescribed inputs are listed
    in ``inputs`` whether or not the graph says which parameter reads them."""
    dids = [g["spec"]] if g.get("spec") else list(g.get("definition_ids") or [])
    widths: dict[str, list[int]] = {}
    names: list[str] = []
    for did in dids:
        d = store.get(did)
        if d is None:
            continue
        for name, t in (d.get("body") or {}).get("params") or d.get("params") or []:
            if name not in widths:
                names.append(name)
            widths.setdefault(name, []).append(_type_bits(t))
    from_calls: dict[str, dict[str, int]] = {}
    for e in into:
        for x in e.get("ports") or []:
            from_calls.setdefault(x["param"], {})
            from_calls[x["param"]][e["from"]] = from_calls[x["param"]].get(e["from"], 0) + x["words"]
    pin = g.get("param_inputs")
    params = []
    for name in names:
        w = widths[name]
        roles = sorted({"constant" if tpl.get(src, "").startswith("Const") else "activations" for src in from_calls.get(name, {})}
                       | {_input_role(i) for i in (pin or {}).get(name, {})})
        params.append({"name": name, "bits": w[0] if min(w) == max(w) else [min(w), max(w)], "roles": roles or None,
                       "from_calls": from_calls.get(name, {}), "from_inputs": (pin or {}).get(name, {}) if pin is not None else None})
        if "constant" in roles:
            params[-1]["bind"] = "run"
    inputs = [{"input": k, "role": _input_role(k), "reads": v} for k, v in sorted((g.get("prescribed_reads") or {}).items())]
    return {"params": params, "inputs": inputs, "param_roles_complete": pin is not None or not inputs,
            "param_roles": "exact (edges' ports and param_inputs)" if pin is not None else
                           "activations exact from edges' ports; which parameter reads each of `inputs` is not in this program graph "
                           "(param_inputs), so input roles are not assigned to parameters"}


def _library_digest() -> str:
    from verity.ml.circuits import library

    return library.digest()


def _git_head() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    global GATHERS
    ap = argparse.ArgumentParser(prog="verity_flock.boolean_export")
    ap.add_argument("--program-graph", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gathers", choices=("mux", "opening"), default=GATHERS,
                    help="a gather from committed weights as a multiplexer (default) or as a commitment opening")
    a = ap.parse_args(argv)
    GATHERS = a.gathers
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = Registry(out)
    ntpl = out / "templates.json"
    tpl_cache, cache = {}, {}
    walker = Walker(reg, DefStore(Path(a.program_graph[0]).parent), cache)
    cm = Commitments(walker)
    checks: dict[str, list] = {"agree": [], "differ": []}
    rows = []
    for pg in a.program_graph:
        t0 = time.time()
        doc = export_row(Path(pg), out, reg, tpl_cache, cache, walker, cm, checks)
        for name in [n for n, w in gf2._TABLE_WORDS.items() if len(w) > 1 << 24]:
            del gf2._TABLE_WORDS[name]          # gemma's 2^27-word tanh tables: 512 MB each, reloaded on demand
        gc.collect()
        f = out / (doc["row_key"] + ".boolean.json")
        f.write_text(json.dumps(doc, separators=(",", ":")))
        xc = crosscheck(doc, {r["id"]: r for r in tpl_cache.values()}, cm.records)
        rows.append({"row": doc["row"], "row_key": doc["row_key"], "file": f.name, "bytes": f.stat().st_size, "nodes": len(doc["nodes"]),
                     "templates": len(doc["templates"]), "seconds": round(time.time() - t0, 1), **doc["totals"], "crosscheck": xc})
        print(f"row {doc['row']}: {len(doc['nodes'])} nodes, {len(doc['templates'])} templates, {time.time() - t0:.0f}s", flush=True)
    ntpl.write_text(json.dumps({"schema": SCHEMA + "/templates", "templates": {r["id"]: r for r in tpl_cache.values()},
                                "definitions": walker.defs}, separators=(",", ":")))
    (out / "subcircuits.json").write_text(json.dumps({"schema": SCHEMA + "/subcircuits", "nodes": reg.nodes}, separators=(",", ":")))
    module_bytes = write_modules(reg, out / "modules.json.gz")
    from verity.ml.definitions.word import query_id
    (out / "commitments.json").write_text(json.dumps({"schema": SCHEMA + "/commitments", "query": query_id(X_BITS, W_BITS),
                                                      "code": f"verity.ml.definitions.word at verity {_git_head()}", "mark_kinds": MARK_KINDS,
                                                      "records": cm.records}, separators=(",", ":")))
    nyl: dict[str, dict] = {}
    for t in tpl_cache.values():
        if t["kind"] == "not-yet-lowered":
            nyl.setdefault(t["template"], {"what": "template", "reason": t["reason"]})
        for v in [t["children"]] + [x["children"] for x in (t.get("per_key_count") or {}).values()]:
            for c in v:
                if "not_yet_lowered" in c:
                    nyl.setdefault(c["not_yet_lowered"], {"what": "primitive (outside the units)", "reason": c["reason"], "in": []})
                    if t["template"] not in nyl[c["not_yet_lowered"]]["in"]:
                        nyl[c["not_yet_lowered"]]["in"].append(t["template"])
    for d in walker.defs.values():
        for c in d["children"]:
            if "not_yet_lowered" in c:
                e = nyl.setdefault(c["not_yet_lowered"], {"what": "primitive (outside the units)", "reason": c["reason"], "in": []})
                fam = d["name"].split("{")[0]
                if fam not in e.setdefault("in", []):
                    e["in"].append(fam)
    flat_ands, type_ands = sum(f for _, f, _ in reg.shared), sum(t for _, _, t in reg.shared)
    index = {"schema": SCHEMA + "/index", "generator": "verity_flock.boolean_export (lane flock-ir-lowering)",
             "source": "internal/datasets/program-graphs/<row>.program.json (its groups' templates)",
             "rows": rows, "files": {"row": "<row_key>.boolean.json", "templates": "templates.json", "subcircuits": "subcircuits.json",
                                     "modules": "modules.json.gz: {types: {digest: circuit type content}}",
                                     "commitments": "commitments.json: {records: {Call descriptor id: its cut}}"},
             "types": {"format": CT.FORMAT, "count": len(reg.types), "bytes": module_bytes, "library_digest": _library_digest(),
                       "own_ands": sum(sum(1 for it in t.items if it[0] == CT.AND) for t in reg.types.values()),
                       "verification": f"every type validated (circuit_types.validate); every circuit root's type evaluated against "
                                       f"its flat circuit (gf2, 64 random vectors): {reg.evaluated} roots, 0 differ",
                       "sharing_across_types": {
                           "rule": "a type's gates are shared only inside it (constant-api §2.5); the flat circuits the lowering "
                                   "builds share ANDs across a call too, so their ANDs are fewer",
                           "flat_ands": flat_ands, "type_ands": type_ands, "roots": reg.roots}},
             "commitments": {"query": query_id(X_BITS, W_BITS), "records": len(cm.records),
                             "check": f"each Call's cut against its program graph's q_word_v1 (units, gates, structure gates, committed "
                                      f"interior words per Call): {len(checks['agree'])} agree, {len(checks['differ'])} differ",
                             "differ": checks["differ"][:50]},
             "not_yet_lowered": nyl}
    if GATHERS == "opening":
        index["commitment_openings"] = {**OPENED, **{f"{k} (its Gather primitives)": v for k, v in OPENED_IN.items()}}
    fails = [(r["row"], k, v["ratio"]) for r in rows for k, v in r["crosscheck"].items() if v["status"] == "fail"]
    index["crosscheck"] = {
        "rule": f"per row and template: the nodes' ANDs against their Calls' units' ANDs under the no-recompute cut; agree within "
                f"{CROSSCHECK_AGREE:.0%}, differs within x{CROSSCHECK_FAIL:g} (a template lowered as one hash-consed unit counts fewer "
                f"ANDs than its primitives' pieces one by one), fail beyond (a granularity error: the export refuses)",
        "agree": sum(v["status"] == "agree" for r in rows for v in r["crosscheck"].values()),
        "differs": sorted([r["row"], k, round(v["ratio"], 4)] for r in rows for k, v in r["crosscheck"].items() if v["status"] == "differs"),
        "fail": fails}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    if fails:
        raise SystemExit(f"cross-check failed: {fails[:10]}")
    print(json.dumps({"rows": [{k: r[k] for k in ("row", "and", "xor", "not")} for r in rows], "types": len(reg.types),
                      "roots_evaluated": reg.evaluated, "flat_ands": flat_ands, "type_ands": type_ands}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
