"""Boolean-circuit export of a leaf template or one census subcircuit (`subcircuit`, the website visualizer's dataset; a
derived format, not a statement).

A leaf template is lowered by :mod:`verity_flock.ir_lower` (its Flock template module where
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
"""
from __future__ import annotations

import gc
import gzip
import hashlib
import io
import json
import random
import re
from pathlib import Path

from verity.circuits.ops.boolean import scalar as SC
from verity.circuits.ops.boolean import softmax as SM

from . import fp, gf2, ir_lower as IL, tail_pieces as TP, unit as U
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
    TW = IL.compositions()[IL.TOPP_WORD]
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
        from verity.circuits import library

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


class CircuitTypeTable:
    """Every circuit type (`circuit_types`) the export builds, by its digest: each distinct sub-circuit once, shared across
    templates and rows, with its names and counts as metadata."""

    def __init__(self, out: Path | None = None):
        self.nodes: dict[str, dict] = {}
        self.types: dict[str, CT.CircuitType] = {}
        self.counts: dict[str, dict] = {}
        self.roots = self.evaluated = 0
        #: per root: the flat circuit's ANDs and its types' (their difference: the sharing a type boundary stops)
        self.shared: list[tuple[str, int, int]] = []

    def type_of(self, build, name: str, kind: str, label: str | None = None, circuit: Circuit | None = None) -> str:
        """The type of the circuit ``build()`` makes, and every type below it: validated, the root evaluated against the flat
        circuit (``circuit``, one ``build()`` made already, else built here). Returns the root's digest."""
        from verity.circuits import library

        _patch()
        types, root, names = TT.trace_types(build, label or library.PROGRAM)
        for d, t in types.items():
            if d not in self.types:
                CT.validate(t, lambda x: types.get(x) or self.types[x])
        C = build() if circuit is None else circuit
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


def write_modules(reg: CircuitTypeTable, path: Path) -> int:
    """``modules.json.gz``: ``{schema, format, types: {digest: content}}`` (``circuit_types``' content, its digest the key),
    streamed; returns its bytes."""
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8") as f:
        f.write(json.dumps({"schema": SCHEMA + "/modules", "format": CT.FORMAT})[:-1] + ',"types":{')
        for n, d in enumerate(sorted(reg.types)):
            f.write(("," if n else "") + json.dumps(d) + ":" + reg.types[d].canonical().decode())
        f.write("}}")
    return path.stat().st_size


# --- the definition walker (templates without a Flock unit lowering, attention per T, GEMM coordinates) ---------------

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


def load_catalog(*packages: str) -> None:
    """Import every module of each catalog package but its quarantine, so its Definitions are registered."""
    import importlib
    import pkgutil

    for package in packages:
        pkg = importlib.import_module(package)
        for m in pkgutil.iter_modules(pkg.__path__):
            if m.name.startswith("quarantine"):
                continue
            try:
                importlib.import_module(f"{package}.{m.name}")
            except Exception:  # noqa: BLE001  (a registry module that needs a GPU stack or data)
                continue


def _registered(pid: str):
    """The PrimitiveDefinition ``pid`` the registry resolves (a loaded module's, `load_catalog`, or a width family's), None
    for none."""
    import verity.circuits.ops.prims  # noqa: F401
    from verity.circuits.defs import REGISTRY, PrimitiveDefinition

    p = REGISTRY.resolve(pid)
    return p if isinstance(p, PrimitiveDefinition) else None


def prims() -> dict:
    """Every registered PrimitiveDefinition by id: ``verity.circuits.ops``' and those of the catalogs loaded (`load_catalog`)."""
    import verity.circuits.ops.prims  # noqa: F401
    from verity.circuits.defs import REGISTRY, PrimitiveDefinition

    return {k: v for k, v in REGISTRY.defs.items() if isinstance(v, PrimitiveDefinition)}


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

    def __init__(self, reg: CircuitTypeTable, store: DefStore, cache: dict):
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
            from verity.circuits import library

            self.cache[key] = self.reg.type_of(build, f"tensor-core k-step unit ({pid}, total)", "tc-step", library.definition_label(pid))
            gc.collect()
        return self.cache[key]

    def prim(self, pid: str, consts: tuple):
        """(child entry without count) for primitive ``pid`` with operand constants ``consts`` (None = free)."""
        if pid in IL.TC_STEPS or pid in IL.TC_STEPS_FP8:
            return {"ref": self.tc_unit(pid)}
        if (m := re.fullmatch(r"GatherBf16x(\d+)_v1", pid)) and GATHERS == "mux":
            return self.gather(int(m.group(1)))
        p = _registered(pid)
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
            TW = IL.compositions()[IL.TOPP_WORD]
            self.cache[ck] = self.reg.type_of(lambda: TW.build(name, key), TW.NAMES[name] + (f" ({consts})" if consts else ""), "piece")
            gc.collect()
        return self.cache[ck]

    def composed(self, pid: str, V: int, p, splits) -> str:
        """A primitive too large to hold as one circuit (``TopPMaskWordx{V}_v1``: about 10^11 ANDs at V = 128256), as a tree of
        its stages (``topp_word``'s paths) whose leaves are its pieces with instance counts."""
        ck = ("composed", pid, p, splits)
        if ck in self.cache:
            return self.cache[ck]
        counts = IL.compositions()[IL.TOPP_WORD].count(V, p, splits)
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
        p = _registered(pid)
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


# --- commitment marks: the no-recompute partition (verity.circuits.word, Q_word_v1{X=16,W=32,R=no-recompute}) -------------


# --- templates and rows -------------------------------------------------------------------------------------------------

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
#: templates checked by opening the committed table, not as a circuit (calibration/census/subcircuits.json commitment_opening)
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


def _tail(reg: CircuitTypeTable, units, target, cache: dict) -> list[dict]:
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


def _unit(reg: CircuitTypeTable, units, name: str, cache: dict) -> str:
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


def template(reg: CircuitTypeTable, tname: str, tp: dict, T_range=None, cache: dict | None = None, walker: "Walker | None" = None,
             spec: str | None = None, member: str | None = None) -> dict:
    """A leaf template's record: its proof blocks under the committed cut and its tail (Flock's unit lowerings), or its
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
        from verity_flock import subcircuit_templates as TM

        sub = TM.subcircuit(SPINE_UNITS[tname], **_params(tname, tp))
        return _spine(reg, sub, base, walker, spec, f"{tname}{json.dumps(tp, sort_keys=True)}",
                      f"{tname}{json.dumps(_params(tname, tp), sort_keys=True)}", cache)
    return _walked(base, walker, spec, spec or tname)


def _spine(reg: CircuitTypeTable, sub, base: dict, walker: "Walker", spec: str | None, walked_name: str, name: str, cache: dict) -> dict:
    """A spine template's record: its Flock unit with the instance count and its tail, or, where the kernel's warps differ (no
    single unit), its Definition walked."""
    from verity_flock import lowerings

    mod = lowerings.registry("C-Flock").module(sub.template.id)
    if hasattr(mod, "width") and mod.width(int(sub.params["N"])) is None:
        rec = _walked(base, walker, spec, walked_name)
        rec["note"] = (f"at N = {sub.params['N']} the kernel's warps differ, so there is no single Flock unit "
                       f"(verity_flock.templates.{Path(mod.__file__).stem}.width); its Definition is walked instead, every "
                       f"primitive with its piece, none in a proof block")
        return rec
    low = mod.lowering(sub)
    uid = _unit(reg, low.units, f"{mod.UNIT} unit", cache)
    children = [{"ref": uid, "count": low.units.n, "c_flock_unit": True}] + _tail(reg, low.units, low.definition, cache)
    ib, ob = _io_bits(low.definition)
    return dict(base, name=name, kind="template", subcircuit=sub.id, children=children, unit=uid, in_bits=ib, out_bits=ob)


def _walked(base: dict, walker: "Walker", did: str | None, name: str) -> dict:
    if did is None:
        return dict(base, name=name, kind="not-yet-lowered", reason="its Definition is not in the program graphs' store", children=[])
    d = walker.defs[walker.walk(did)]
    return dict(base, name=name, kind="template", definition=d["id"], children=[{"def": d["id"], "count": 1}],
                in_bits=d.get("in_bits"), out_bits=d.get("out_bits"))


def _totals(reg: CircuitTypeTable, children: list[dict], defs: dict | None = None) -> dict:
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


class _Encoded:
    """A store of Definitions encoded here, in the codec's v0 form a program graph's store holds them in, for a subcircuit no
    row carries."""

    def __init__(self):
        self.ix: dict[str, dict] = {}

    def add(self, fn) -> str:
        from verity.circuits.codec import encode_specialization

        top, defs = encode_specialization(fn)
        self.ix.update(defs)
        return top

    def get(self, did: str):
        return self.ix.get(did)


def statics_key(statics: dict) -> str:
    """Instance statics as one key: canonical JSON (sorted keys, no spaces), ``{}`` for a subcircuit whose statics never vary."""
    return json.dumps(statics, sort_keys=True, separators=(",", ":"))


def subcircuit(sub, statics=({},)) -> dict:
    """One bound subcircuit's Boolean export (``verity_flock.subcircuit_templates.Subcircuit``), self-contained: a template record
    per instance statics (``variants``, keyed by :func:`statics_key`; attention one per key count T), as a row's templates are
    recorded (a spine template its Flock unit and tail, any other its Definition walked), with every definition, subcircuit
    node and circuit type (``circuit_types`` content, keyed by digest) they reach. Raises where the subcircuit has no core
    Definition (a silicon reference)."""
    reg, cache, store = CircuitTypeTable(), {}, _Encoded()
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


def _library_digest() -> str:
    from verity.circuits import library

    return library.digest()
