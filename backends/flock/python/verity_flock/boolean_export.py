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
one node, referenced with instance counts.

    python -m verity_flock.boolean_export --program-graph ROW.program.json --out DIR
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from . import fp, gf2, ir_lower as IL, unit as U
from .gf2 import Circuit, L, const_bits

SCHEMA = "verity-boolean-circuit/v1"
GATE_LIMIT = 20000
OP = {"AND": 0, "XOR": 1, "NOT": 2}
CONST0, CONST1 = -1, -2

#: traced function -> display name (the id is structural; renaming a display name changes no id)
DISPLAY = {
    "bf16_to_f32": "bf16→f32", "f32_to_bf16": "f32→bf16 (round)", "unpack": "unpack f32", "round_f32": "round",
    "specials": "specials (NaN/Inf)", "f32_mul": "f32 multiply", "f32_add": "f32 add", "f32_sub": "f32 subtract",
    "f32_fma": "f32 fused multiply-add", "f32_div_const": "f32 divide by constant", "f32_neg": "f32 negate",
    "tc_dot16": "tensor-core k-step (AmpereBF16TcDot16, total)", "_tc_operand": "tensor-core operand decode",
    "_classes": "classify (NaN/Inf)", "_finite_or_zero": "finite stand-in", "decode": "one-hot decode",
    "lookup": "table lookup", "_normalize_sig": "normalize", "shl": "shift left", "shr_sticky": "align (shift right, sticky)",
    "min_u": "min", "neg_if": "conditional negate", "select": "select", "is_zero": "zero test", "add": "integer add",
    "sub": "integer subtract", "multiply": "integer multiply", "compress_columns": "carry-save compress",
    "lzc": "leading-zero count", "shr": "shift right", "shl_window": "shift left (window)", "mux": "mux",
    "or_reduce": "OR-reduce", "and_reduce": "AND-reduce", "group_sum": "group sum (align + add + normalize)",
    "saturated": "saturation test", "decode_operand": "operand decode", "max_tree": "max tree",
}
KIND = {"multiply": "multiply", "f32_mul": "multiply", "shr_sticky": "align", "add": "add", "f32_add": "add",
        "sub": "add", "_normalize_sig": "normalize", "lzc": "normalize", "shl": "normalize", "round_f32": "round",
        "f32_to_bf16": "round", "lookup": "lookup", "decode": "lookup", "bf16_to_f32": "convert", "tc_dot16": "dot-step",
        "group_sum": "add", "compress_columns": "add", "f32_fma": "fma"}

# --- tracing -----------------------------------------------------------------------------------------------------------

class _Call:
    __slots__ = ("fn", "start", "end", "ins", "outs", "kids", "seq0", "seq1")

    def __init__(self, fn, start):
        self.fn, self.start, self.end, self.ins, self.outs, self.kids = fn, start, start, [], [], []
        self.seq0 = self.seq1 = _SEQ[0]


_T: dict = {"C": None, "stack": None}
_SEQ = [0]


class TL(L):
    """A linear form made by an XOR or NOT the lowering performed while traced: the gate, its operands, the subcircuit that
    ran it and its creation order."""

    __slots__ = ("op", "p1", "p2", "nd", "seq")


def _mk(op, p1, p2, s, c):
    t = TL(s, c)
    t.op, t.p1, t.p2 = op, p1, p2
    t.nd = _T["stack"][-1]
    t.seq = _SEQ[0]
    _SEQ[0] += 1
    return t


def _xor(self, o):
    s, c = self.s ^ o.s, self.c ^ o.c
    if _T["stack"] is None or not s:
        return L(s, c)
    if not self.s or not o.s:
        x, k = (o, self) if not self.s else (self, o)
        return x if k.c == 0 else _mk("NOT", x, None, s, c)
    return _mk("XOR", self, o, s, c)


def _inv(self):
    if _T["stack"] is None or not self.s:
        return L(self.s, self.c ^ 1)
    return _mk("NOT", self, None, self.s, self.c ^ 1)


_new_orig = Circuit._new


def _new(self, kind, data):
    w = _new_orig(self, kind, data)
    seq = self.__dict__.setdefault("seq", [])
    seq.append(_SEQ[0])
    _SEQ[0] += 1
    return w


def _bits(x, out):
    if isinstance(x, L):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _bits(y, out)
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            _bits(getattr(x, f.name), out)


def _wrap(name, fn):
    def w(*a, **k):
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
            r = fn(*a, **k)
        finally:
            st.pop()
        node.end = len(_T["C"].kind) if _T["C"] is not None else node.start
        node.seq1 = _SEQ[0]
        _bits([x for x in a if not isinstance(x, Circuit)], node.ins)
        _bits(list(k.values()), node.ins)
        _bits(r, node.outs)
        return r

    w.__wrapped__ = fn
    return w


_PATCHED = False


def _patch():
    global _PATCHED
    if _PATCHED:
        return
    originals = {}
    for mod in (gf2, fp, U):
        for name in DISPLAY:
            f = getattr(mod, name, None)
            if callable(f) and not isinstance(f, type):
                originals[id(f)] = (name, f)
    wrapped = {k: _wrap(n, f) for k, (n, f) in originals.items()}
    for mod in (gf2, fp, U, IL):
        for name in dir(mod):
            f = getattr(mod, name)
            if callable(f) and id(f) in wrapped:
                setattr(mod, name, wrapped[id(f)])
    for k, f in list(IL.PIECES.items()):
        IL.PIECES[k] = _wrap(k, f)
    L.__xor__, L.inv, Circuit._new = _xor, _inv, _new
    _PATCHED = True


def trace(build) -> tuple[Circuit, _Call]:
    """Run ``build()`` (returns the Circuit it built) with the named functions traced."""
    _patch()
    root = _Call("root", 0)
    _T.update(C=None, stack=[root])
    try:
        C = build()
    finally:
        _T["stack"] = None
    root.end = len(C.kind)
    root.seq1 = _SEQ[0]
    root.ins = [w for bits in C.inputs.values() for w in bits]
    root.outs = [w for bits in C.outputs.values() for w in bits]
    return C, root


# --- analysis: counts, structural ids, gate lists ----------------------------------------------------------------------

class Registry:
    """Node types by structural hash, across units, templates and rows."""

    def __init__(self, out: Path):
        self.nodes: dict[str, dict] = {}
        self.gates_dir = out / "gates"
        self.gates_dir.mkdir(parents=True, exist_ok=True)
        self.verified = 0

    def analyse(self, C: Circuit, root: _Call, name: str, kind: str) -> str:
        live = C.live()
        is_and = [k == "and" for k in C.kind]
        data = C.data
        memo: dict[str, _Call] = {}
        owner = [root] * len(C.kind)

        def own_range(node):
            for k in node.kids:
                own_range(k)
            for v in range(node.start, node.end):
                if owner[v] is root:
                    owner[v] = node
        for k in root.kids:
            own_range(k)
        xn: dict[int, list[int]] = {}
        seen: set[int] = set()

        def charge(f, user):
            stack = [(f, user)]
            while stack:
                x, u = stack.pop()
                if id(x) in seen:
                    continue
                seen.add(id(x))
                if isinstance(x, TL):
                    e = xn.setdefault(id(x.nd), [0, 0])
                    e[0 if x.op == "XOR" else 1] += 1
                    stack.append((x.p1, x.nd))
                    if x.p2 is not None:
                        stack.append((x.p2, x.nd))
                elif len(x.s) > 1 or (x.s and x.c):
                    e = xn.setdefault(id(u), [0, 0])
                    e[0] += len(x.s) - 1
                    e[1] += x.c
        for v in live:
            if is_and[v]:
                for f in data[v]:
                    charge(f, owner[v])
        for bits in C.outputs.values():
            for f in bits:
                charge(f, root)

        def visit(node):
            kid_ids = [visit(k) for k in node.kids]
            covered = set()
            for k in node.kids:
                covered.update(range(k.start, k.end))
            own = [v for v in range(node.start, node.end) if is_and[v] and v in live and v not in covered]
            xor, nots = xn.get(id(node), (0, 0))
            ext = {}
            for i, f in enumerate(node.ins):
                if len(f.s) == 1 and f.c == 0:
                    (v,) = tuple(f.s)
                    if not node.start <= v < node.end:
                        ext.setdefault(v, i)
            extra: dict[int, int] = {}

            def lab(v):
                if node.start <= v < node.end:
                    return (0, v - node.start)
                if v in ext:
                    return (1, ext[v])
                return (2, extra.setdefault(v, len(extra)))

            def canon(f):
                return (tuple(sorted(lab(t) for t in f.s)), f.c)

            h = hashlib.sha256()
            h.update(json.dumps([node.fn, len(node.ins), len(node.outs), xor, nots]).encode())
            for v in own:
                a, b = data[v]
                h.update(repr((v - node.start, canon(a), canon(b))).encode())
            for (kh, _), k in zip(kid_ids, node.kids):
                h.update(repr((kh, k.start - node.start)).encode())
            h.update(repr([canon(f) for f in node.outs]).encode())
            hid = "sc-" + h.hexdigest()[:16]
            n_and, n_xor, n_not = len(own), xor, nots
            children: dict[str, int] = {}
            for kh, (ka, kx, kn) in kid_ids:
                n_and, n_xor, n_not = n_and + ka, n_xor + kx, n_not + kn
                children[kh] = children.get(kh, 0) + 1
            if hid not in self.nodes:
                fn = node.fn
                self.nodes[hid] = {"id": hid, "name": _display(fn, node), "kind": KIND.get(fn, "piece" if fn in IL.PIECES or fn.endswith("_v1")
                                                                                          else "subcircuit"),
                                   "function": fn, "and": n_and, "xor": n_xor, "not": n_not, "in_bits": len(node.ins),
                                   "out_bits": len(node.outs), "children": [{"ref": c, "count": n} for c, n in children.items()],
                                   "own": {"and": len(own), "xor": xor, "not": nots}}
            memo.setdefault(hid, node)
            return hid, (n_and, n_xor, n_not)

        rid, _ = visit(root)
        rec = self.nodes[rid]
        rec["name"], rec["kind"] = name, kind
        self._gate_lists(C, root, rid, memo)
        return rid

    def _gate_lists(self, C, root, rid, memo):
        """Write the gate list of every node type that is at most GATE_LIMIT gates and whose parent is larger (or is a root),
        once; verify each against the circuit on random inputs."""
        want = set()

        def total(h):
            n = self.nodes[h]
            return n["and"] + n["xor"] + n["not"]

        def walk(h, parent_big):
            n = self.nodes[h]
            small = total(h) <= GATE_LIMIT
            if small and parent_big and total(h) > 0:
                want.add(h)
            for c in n["children"]:
                walk(c["ref"], not small)

        walk(rid, True)
        vals, bundle = None, {}
        for h in want:
            rec = self.nodes[h]
            if rec.get("gates") or h not in memo:
                continue
            node = memo[h]
            gl = gate_list(C, node)
            if vals is None:
                vals = _evaluate_all(C, 64)
            _check(gl, node, vals)
            self.verified += 1
            bundle[h] = gl
            rec["gates"] = {"file": f"gates/{rid}.json.gz", "key": h}
            rec["gate_list"] = {"and": gl["counts"]["and"], "xor": gl["counts"]["xor"], "not": gl["counts"]["not"]}
        if bundle:
            (self.gates_dir / f"{rid}.json.gz").write_bytes(gzip.compress(json.dumps(
                {"schema": SCHEMA + "/gates", "lists": bundle}, separators=(",", ":")).encode(), mtime=0))


def _display(fn, node) -> str:
    base = DISPLAY.get(fn, fn)
    return f"{base} [{len(node.ins)}→{len(node.outs)} bits]"


def gate_list(C: Circuit, node: _Call) -> dict:
    """The node's gates as an explicit AND/XOR/NOT list, in the order the lowering made them: wires 0..n_in-1 are its inputs
    (forms made outside it, its argument bits first), each gate appends one wire, -1 / -2 are constants 0 / 1; ``gates`` is
    flat ``[op, a, b]`` (op 0 AND, 1 XOR, 2 NOT with b = -1). A linear form with no recorded operations (``fp.lookup``'s table
    sums) is a chain of XORs where it is used."""
    data, kind, cseq = C.data, C.kind, C.seq
    lo, hi, s0, s1 = node.start, node.end, node.seq0, node.seq1

    def inside(x):
        if isinstance(x, TL):
            return s0 <= x.seq < s1
        return len(x.s) == 1 and x.c == 0 and lo <= next(iter(x.s)) < hi and kind[next(iter(x.s))] == "and"

    # the objects the outputs need, inside the node; everything else they reach is an input
    need_tl: dict[int, TL] = {}
    need_and: set[int] = set()
    ext: dict = {}
    stack = list(node.outs)
    while stack:
        x = stack.pop()
        if isinstance(x, TL):
            if not s0 <= x.seq < s1:
                ext.setdefault(("f", id(x)), x)
                continue
            if id(x) in need_tl:
                continue
            need_tl[id(x)] = x
            stack.append(x.p1)
            if x.p2 is not None:
                stack.append(x.p2)
            continue
        for t in x.s:
            if lo <= t < hi and kind[t] == "and":
                if t not in need_and:
                    need_and.add(t)
                    stack.extend(data[t])
            else:
                ext.setdefault(("b", t), L(frozenset((t,)), 0))
    order: dict = {}
    origin = []
    for i, f in enumerate(node.ins):
        k = ("f", id(f)) if isinstance(f, TL) and not s0 <= f.seq < s1 else (("b", next(iter(f.s))) if len(f.s) == 1 and f.c == 0 else None)
        if k is not None and k in ext and k not in order:
            order[k] = len(order)
            origin.append(["arg", i])
    for k, x in ext.items():
        if k not in order:
            order[k] = len(order)
            origin.append(["ext", k[1]] if k[0] == "b" else ["form", sorted(x.s), x.c])
    n_in = len(order)
    gates: list[int] = []
    cnt = {"and": 0, "xor": 0, "not": 0}
    nxt = [n_in]
    wire_tl: dict[int, int] = {}
    wire_and: dict[int, int] = {}
    chain: dict = {}

    def emit(op, a, b=-1):
        gates.extend((OP[op], a, b))
        cnt[op.lower()] += 1
        nxt[0] += 1
        return nxt[0] - 1

    def bit(t):
        return wire_and[t] if t in wire_and else order[("b", t)]

    def wire(x):
        if isinstance(x, TL):
            return wire_tl[id(x)] if id(x) in wire_tl else order[("f", id(x))]
        if not x.s:
            return CONST1 if x.c else CONST0
        if len(x.s) == 1 and x.c == 0:
            return bit(next(iter(x.s)))
        k = x.key()
        if k not in chain:
            ts = sorted(x.s)
            w = bit(ts[0])
            for t in ts[1:]:
                w = emit("XOR", w, bit(t))
            chain[k] = emit("NOT", w) if x.c else w
        return chain[k]

    items = sorted([(cseq[v], 0, v) for v in need_and] + [(x.seq, 1, x) for x in need_tl.values()], key=lambda e: e[0])
    for _, typ, x in items:
        if typ == 0:
            a, b = data[x]
            wire_and[x] = emit("AND", wire(a), wire(b))
        elif x.op == "XOR":
            wire_tl[id(x)] = emit("XOR", wire(x.p1), wire(x.p2))
        else:
            wire_tl[id(x)] = emit("NOT", wire(x.p1))
    outs = [wire(f) for f in node.outs]
    return {"inputs": n_in, "input_origin": origin, "outputs": outs, "gates": gates, "counts": cnt}


def _evaluate_all(C: Circuit, T: int) -> list[int]:
    rng = random.Random(101)
    mask = (1 << T) - 1
    val = [0] * len(C.kind)
    for name, bits in C.inputs.items():
        for w in bits:
            (v,) = tuple(w.s)
            val[v] = rng.getrandbits(T)
    for v, k in enumerate(C.kind):
        if k == "and":
            a, b = C.data[v]
            x = mask if a.c else 0
            for t in a.s:
                x ^= val[t]
            y = mask if b.c else 0
            for t in b.s:
                y ^= val[t]
            val[v] = x & y
    return val


def _ev(f, val, mask):
    x = mask if f.c else 0
    for t in f.s:
        x ^= val[t]
    return x


def _check(gl, node, val, T: int = 64):
    mask = (1 << T) - 1
    w = []
    for o in gl["input_origin"]:
        if o[0] == "arg":
            w.append(_ev(node.ins[o[1]], val, mask))
        elif o[0] == "ext":
            w.append(val[o[1]])
        else:
            w.append(_ev(L(frozenset(o[1]), o[2]), val, mask))
    g = gl["gates"]
    for i in range(0, len(g), 3):
        op, a, b = g[i], g[i + 1], g[i + 2]
        va = mask if a == CONST1 else 0 if a == CONST0 else w[a]
        if op == 2:
            w.append(va ^ mask)
            continue
        vb = mask if b == CONST1 else 0 if b == CONST0 else w[b]
        w.append(va & vb if op == 0 else va ^ vb)
    got = [mask if o == CONST1 else 0 if o == CONST0 else w[o] for o in gl["outputs"]]
    want = [_ev(f, val, mask) for f in node.outs]
    if got != want:
        raise AssertionError(f"gate list of {node.fn} disagrees with the circuit")


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
    """The program graphs' Definitions (``definitions/index.json`` + ``<digest[:2]>.json.gz`` shards), by id."""

    def __init__(self, pg_dir: Path):
        self.dir = Path(pg_dir) / "definitions"
        self.ix = json.loads((self.dir / "index.json").read_text())["definitions"]
        self.shards: dict[str, dict] = {}
        self.by_name: dict[str, list] = {}
        for did in self.ix:
            name, ps = parse_id(did)
            self.by_name.setdefault(name, []).append((did, ps))

    def get(self, did: str):
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
    """Every registered PrimitiveDefinition by id (verity.ml and the vLLM registry modules)."""
    global _PRIMS
    if _PRIMS is None:
        import importlib
        import pkgutil

        from verity.ir.defs import PrimitiveDefinition

        mods = ["verity.ml.prims"]
        import verity_vllm.program.registry as R
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


def _words(t) -> list[int]:
    k = t[0]
    if k == "Value":
        return [t[1]]
    if k == "Array":
        return _words(t[2]) * t[1]
    if k == "Tuple":
        return [w for x in t[1] for w in _words(x)]
    raise ValueError(f"type {t!r}")


def _const_value(fn: str):
    if fn.startswith("Const") and "[" in fn:
        return int(fn.split("[", 1)[1].split("]", 1)[0], 16)
    return None


class Walker:
    """Definitions as trees: children are sub-definitions (``call`` x1, ``batch`` / ``scan`` x n) and primitive leaves; a
    tensor-core step is the proof unit; another primitive is its ir_lower piece (lowered once per operand signature, constant
    operands a Const node gives propagated) or not yet lowered."""

    def __init__(self, reg: Registry, store: DefStore, cache: dict):
        self.reg, self.store, self.cache = reg, store, cache
        self.defs: dict[str, dict] = {}

    def tc_unit(self, pid: str) -> str:
        key = ("tc", pid)
        if key not in self.cache:
            groups, W, F = IL.TC_STEPS[pid]

            def build():
                C = Circuit()
                b = [C.input(f"in{j}", 16) for j in range(16)]
                a = [C.input(f"in{16 + j}", 16) for j in range(16)]
                acc = C.input("in32", 32)
                C.output("out0", fp.tc_dot16(C, acc, a, b, groups=groups, W=W, F=F))
                return C
            C, root = trace(build)
            self.cache[key] = self.reg.analyse(C, root, f"tensor-core k-step unit ({pid}, total)", "proof-unit")
        return self.cache[key]

    def prim(self, pid: str, consts: tuple):
        """(child entry without count) for primitive ``pid`` with operand constants ``consts`` (None = free)."""
        if pid in IL.TC_STEPS:
            return {"ref": self.tc_unit(pid), "proof_unit": True}
        p = prims().get(pid)
        if p is None:
            return {"not_yet_lowered": pid, "reason": "primitive not in the registries this export reads"}
        widths = [w for _, t in p.params for w in t.leaf_widths()]
        consts = consts if len(consts) == len(widths) else (None,) * len(widths)
        free = [w for c, w in zip(consts, widths) if c is None]
        if pid in IL.PIECES and free:
            ck = ("prim", pid, consts, tuple(widths))
            if ck not in self.cache:
                def build():
                    C = Circuit()
                    ops = [const_bits(c, w) if c is not None else C.input(f"in{j}", w) for j, (c, w) in enumerate(zip(consts, widths))]
                    C.output("out0", IL.PIECES[pid](C, *ops))
                    return C
                try:
                    C, root = trace(build)
                    self.cache[ck] = self.reg.analyse(C, root, f"{pid} (outside the units)", "piece")
                except Exception as e:  # noqa: BLE001  (the piece does not take these operands)
                    self.cache[ck] = str(e)
            got = self.cache[ck]
            if not got.startswith("sc-"):
                return {"not_yet_lowered": pid, "in_bits": sum(free), "out_bits": sum(p.ret.leaf_widths()), "reason": got}
            return {"ref": got, "proof_unit": False, "placement": "outside the committed units"}
        if not free:
            return None
        return {"not_yet_lowered": pid, "in_bits": sum(free), "out_bits": sum(p.ret.leaf_widths()),
                "reason": "ir_lower has no piece for this primitive yet"}

    def walk(self, did: str) -> str:
        rid = "def-" + hashlib.sha256(did.encode()).hexdigest()[:16]
        if rid in self.defs:
            return rid
        d = self.store.get(did)
        rec = {"id": rid, "name": did, "kind": "definition", "children": []}
        self.defs[rid] = rec
        if d is None:
            rec.update(kind="not-yet-lowered", reason="definition not in the program graphs' store")
            return rid
        if d.get("form") != "composite":
            rec["children"] = [self._leaf(did, (), 1)] if self._leaf(did, (), 1) else []
            self._total(rec)
            return rid
        nodes = d["body"]["nodes"]
        rec["in_bits"] = sum(w for prm in d["body"]["params"] for w in _words(prm[1]))
        rec["out_bits"] = sum(_words(d["body"]["ret_type"]))
        agg: dict = {}
        for n in nodes:
            fn, form = n["fn"], n["form"]
            count = n.get("n", 1) if form in ("batch", "scan") else 1
            if _const_value(fn) is not None:
                continue
            if fn in self.store.ix and (self.store.get(fn) or {}).get("form") == "composite":
                key = ("def", fn)
                agg.setdefault(key, [{"def": self.walk(fn)}, 0])[1] += count
                continue
            consts = []
            for a in n.get("args", []):
                ref = (a.get("refs") or {}).get("affine")
                c = None
                if form == "primitive" and ref and ref[0] == "n" and ref[3] == 1:
                    src = nodes[ref[1]]
                    c = _const_value(src["fn"]) if src.get("form") == "primitive" else None
                consts.extend([c] * (len(_words(a["type"])) if form == "primitive" else 1))
            key = ("prim", fn, tuple(consts) if form == "primitive" else ())
            if key not in agg:
                e = self._leaf(fn, key[2], count)
                if e is None:
                    continue
                agg[key] = [e, 0]
            agg[key][1] += count
        rec["children"] = [dict(e, count=c) for e, c in agg.values()]
        self._total(rec)
        return rid

    def _leaf(self, pid, consts, count):
        return self.prim(pid, consts)

    def _total(self, rec):
        t = {"and": 0, "xor": 0, "not": 0, "not_yet_lowered": 0}
        for c in rec["children"]:
            if "def" in c:
                s = self.defs[c["def"]]
                for k in t:
                    t[k] += s.get(k, 0) * c["count"]
                if s["kind"] == "not-yet-lowered":
                    t["not_yet_lowered"] += c["count"]
            elif "ref" in c:
                n = self.reg.nodes[c["ref"]]
                for k in ("and", "xor", "not"):
                    t[k] += n[k] * c["count"]
            else:
                t["not_yet_lowered"] += c["count"]
        rec.update(t)


# --- templates and rows -------------------------------------------------------------------------------------------------

#: program-graph template -> spine template (C-Flock's lowering module, or the GEMM coordinate as tensor-core k-steps)
SPINE = {"RoPEHead": "rope-head", "SiluMul_v1": "silu-mul", "RMSNormFusedCuda_v2": "rmsnorm-fused-cuda",
         "RMSNormTriton_v1": "rmsnorm-triton", "GemmCoordinate": "gemm-coordinate", "AttentionHead": "attention-head"}
NOT_LOWERED = {
    "Embedding_v1": "a row gather by token id (data movement from the committed table); no Boolean lowering yet",
    "EmbeddingShard_v1": "a row gather by token id from a tensor-parallel shard of the table; no Boolean lowering yet",
    "GumbelTopPTokenSelect_v1": "Gumbel noise, the top-p word and the sampling carries; no Boolean lowering yet",
}
DOT_SEMANTICS = {"AmpereBF16TcDot16_v1": "sm80.mma.m16n8k16.bf16"}
TC_UNIT_NAME = "tensor-core k-step unit (AmpereBF16TcDot16, total)"


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


def _tail(reg: Registry, units, target, cache: dict) -> list[dict]:
    """The template's gates outside its units: each (primitive, constant operands) once with its count, lowered with its
    ``ir_lower`` piece where one exists, else marked not yet lowered."""
    _prog, call = IL.standalone(target)
    by = {g.index: g for g in call}
    groups: dict = {}
    for g in units.tail:
        fam = getattr(g.prim, "family", None)
        if fam in ("Input", "Const"):
            continue
        consts = tuple(int(by[o].prim.evaluate()) if o in by and getattr(by[o].prim, "family", None) == "Const" else None
                       for o in g.operands)
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
                    C = Circuit()
                    ops = [const_bits(c, w) if c is not None else C.input(f"in{j}", w) for j, (c, w) in enumerate(zip(consts, widths))]
                    C.output("out0", IL.PIECES[pid](C, *ops))
                    return C
                try:
                    C, root = trace(build)
                    cache[ck] = reg.analyse(C, root, f"{pid} (outside the units)", "piece")
                except Exception as e:  # noqa: BLE001  (the piece does not take these operands)
                    cache[ck] = str(e)
            if not cache[ck].startswith("sc-"):
                out.append({"not_yet_lowered": pid, "count": n, "in_bits": sum(free), "out_bits": prim.ret.w, "reason": cache[ck]})
                continue
            out.append({"ref": cache[ck], "count": n, "proof_unit": False, "placement": "outside the committed units (the template's tail)"})
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
        C, root = trace(units.circuit)
        cache[key] = reg.analyse(C, root, name, "proof-unit")
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
             spec: str | None = None) -> dict:
    """A leaf template's record: its proof units under the committed cut and its tail (C-Flock's unit lowerings), or its
    Definition tree (the walker: attention per T, GEMM coordinates, every other template)."""
    cache = {} if cache is None else cache
    rid = "tpl-" + hashlib.sha256(json.dumps([tname, tp, T_range], sort_keys=True).encode()).hexdigest()[:16]
    base = {"id": rid, "template": tname, "params": tp}
    if tname.startswith(("Const32", "Const64")):
        return dict(base, name=tname, kind="constant", **{"and": 0, "xor": 0, "not": 0}, children=[])
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
                             **{k: d.get(k) for k in ("and", "xor", "not", "not_yet_lowered", "in_bits", "out_bits")}}
        top = per_t[str(T_range[1])]
        return dict(base, name=f"{tname}{json.dumps(tp, sort_keys=True)} T∈[{T_range[0]},{T_range[1]}]", kind="template",
                    children=top.get("children", []), children_at=f"T={T_range[1]}", per_key_count=per_t,
                    in_bits=top.get("in_bits"), out_bits=top.get("out_bits"))
    if tname == "GemmCoordinate":
        ids = (walker.store.find("GemmCoordinate_v2", K=str(tp["K"]), DOT=HOPPER_DOT) if tp.get("dot") == "HopperBF16WgmmaDot16_v1"
               else walker.store.find("GemmCoordinate_v1", K=str(tp["K"])))
        return _walked(base, walker, ids.get(None), f"GemmCoordinate{json.dumps(tp, sort_keys=True)}")
    if tname in SPINE_UNITS:
        from verity_numerical.bench import lowerings, templates as TM

        spine = SPINE_UNITS[tname]
        sub = TM.subcircuit(spine, **_params(tname, tp))
        mod = lowerings.registry("C-Flock").module(spine)
        if hasattr(mod, "width") and mod.width(int(sub.params["N"])) is None:
            rec = _walked(base, walker, spec, f"{tname}{json.dumps(tp, sort_keys=True)}")
            rec["note"] = (f"at N = {sub.params['N']} the kernel's warps differ, so there is no single C-Flock unit "
                           f"({mod.__name__}.width); its Definition is walked instead, every primitive with its piece, none in a proof unit")
            return rec
        low = mod.lowering(sub)
        uid = _unit(reg, low.units, f"{mod.UNIT} unit", cache)
        children = [{"ref": uid, "count": low.units.n, "proof_unit": True}] + _tail(reg, low.units, low.definition, cache)
        ib, ob = _io_bits(low.definition)
        return dict(base, name=f"{tname}{json.dumps(_params(tname, tp), sort_keys=True)}", kind="template", subcircuit=sub.id,
                    children=children, unit=uid, in_bits=ib, out_bits=ob)
    return _walked(base, walker, spec, spec or tname)


def _walked(base: dict, walker: "Walker", did: str | None, name: str) -> dict:
    if did is None:
        return dict(base, name=name, kind="not-yet-lowered", reason="its Definition is not in the program graphs' store", children=[])
    d = walker.defs[walker.walk(did)]
    return dict(base, name=name, kind="template", definition=d["id"], children=[{"def": d["id"], "count": 1}],
                in_bits=d.get("in_bits"), out_bits=d.get("out_bits"))


def _totals(reg: Registry, children: list[dict], defs: dict | None = None) -> dict:
    t = {"and": 0, "xor": 0, "not": 0, "not_yet_lowered": 0}
    for c in children:
        if "def" in c:
            d = defs[c["def"]]
            for k in t:
                t[k] += (d.get(k) or 0) * c["count"]
        elif "ref" in c:
            n = reg.nodes[c["ref"]]
            for k in ("and", "xor", "not"):
                t[k] += n[k] * c["count"]
        else:
            t["not_yet_lowered"] += c["count"]
    return t


def export_row(pg_path: Path, out: Path, reg: Registry, tpl_cache: dict, cache: dict, walker: "Walker") -> dict:
    p = json.loads(Path(pg_path).read_text())
    row = p["row"]
    nodes, used = [], []
    for g in p["groups"]:
        tname, tp = g["template"], g.get("template_params") or {}
        T_range = g.get("varying", {}).get("T") if tname in ("AttentionHead", "AttentionSoftcap_v1") else None
        key = json.dumps([tname, tp, T_range], sort_keys=True)
        if key not in tpl_cache:
            t0 = time.time()
            rec = template(reg, tname, tp, T_range, cache, walker, g.get("spec"))
            if rec["kind"] == "template":
                rec.update(_totals(reg, rec["children"], walker.defs))
            tpl_cache[key] = rec
            print(f"  template {rec['name'][:80]}: {rec['kind']} {time.time() - t0:.1f}s", flush=True)
        rec = tpl_cache[key]
        used.append(rec["id"])
        node = {"id": f"{row['row']}:{g['id']}", "name": g["module"] or "(request)", "kind": "program-node", "group": g["id"],
                "definition": g.get("spec") or g.get("definition"), "template": rec["id"], "instances": g.get("instances"),
                "calls": g.get("calls")}
        if T_range:
            n_t = T_range[1] - T_range[0] + 1
            node["varying"] = {"T": T_range}
            if g.get("instances") and g["instances"] % n_t == 0:
                node["instances_per_T"] = g["instances"] // n_t
                node["instances_per_T_rule"] = "instances / (T range size): the program graph gives the range, not the histogram"
        nodes.append(node)
    return {"schema": SCHEMA + "/row", "row": row["row"], "row_key": row["serving"]["row_key"], "model": row["serving"].get("model"),
            "program_graph": Path(pg_path).name, "nodes": nodes, "edges": p.get("edges", []), "templates": sorted(set(used))}


def _membership(reg: Registry, templates, defs: dict | None = None) -> None:
    """Every subcircuit's proof-unit membership under the committed cut: the unit roots whose trees contain it, and whether it
    also appears outside the units (a template's lowered tail)."""
    for n in reg.nodes.values():
        n["proof_units"], n["outside_units"] = [], False
    done = set()

    def mark(h, unit):
        key = (h, unit)
        if key in done:
            return
        done.add(key)
        n = reg.nodes[h]
        if unit is None:
            n["outside_units"] = True
        elif unit not in n["proof_units"]:
            n["proof_units"].append(unit)
        for c in n["children"]:
            mark(c["ref"], unit)
    for t in templates:
        variants = [t["children"]] + [v["children"] for v in (t.get("per_key_count") or {}).values() if "children" in v]
        for ch in variants:
            for c in ch:
                if "ref" in c:
                    mark(c["ref"], c["ref"] if c.get("proof_unit") else None)
    for d in (defs or {}).values():
        for c in d["children"]:
            if "ref" in c:
                mark(c["ref"], c["ref"] if c.get("proof_unit") else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verity_flock.boolean_export")
    ap.add_argument("--program-graph", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = Registry(out)
    ntpl = out / "templates.json"
    tpl_cache, cache = {}, {}
    walker = Walker(reg, DefStore(Path(a.program_graph[0]).parent), cache)
    rows = []
    for pg in a.program_graph:
        t0 = time.time()
        doc = export_row(Path(pg), out, reg, tpl_cache, cache, walker)
        f = out / (doc["row_key"] + ".boolean.json")
        f.write_text(json.dumps(doc, separators=(",", ":")))
        rows.append({"row": doc["row"], "row_key": doc["row_key"], "file": f.name, "bytes": f.stat().st_size, "nodes": len(doc["nodes"]),
                     "templates": len(doc["templates"]), "seconds": round(time.time() - t0, 1)})
        print(f"row {doc['row']}: {len(doc['nodes'])} nodes, {len(doc['templates'])} templates, {time.time() - t0:.0f}s", flush=True)
    _membership(reg, tpl_cache.values(), walker.defs)
    ntpl.write_text(json.dumps({"schema": SCHEMA + "/templates", "templates": {r["id"]: r for r in tpl_cache.values()},
                                "definitions": walker.defs}, separators=(",", ":")))
    (out / "subcircuits.json").write_text(json.dumps({"schema": SCHEMA + "/subcircuits", "gate_limit": GATE_LIMIT, "nodes": reg.nodes},
                                                     separators=(",", ":")))
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
    gates = sorted(reg.gates_dir.glob("*.json.gz"))
    index = {"schema": SCHEMA + "/index", "generator": "verity_flock.boolean_export (lane flock-ir-lowering)",
             "source": "internal/datasets/program-graphs/<row>.program.json (its groups' templates)",
             "rows": rows, "files": {"row": "<row_key>.boolean.json", "templates": "templates.json", "subcircuits": "subcircuits.json",
                                     "gates": "gates/<circuit root id>.json.gz: {lists: {subcircuit id: gate list}}"},
             "gate_list_rule": "wires 0..inputs-1 are the inputs; each gate appends one wire; -1 = 0, -2 = 1; gates = flat [op, a, b], "
                               "op 0 AND, 1 XOR, 2 NOT (b = -1)",
             "gate_limit": GATE_LIMIT, "subcircuits": len(reg.nodes), "gate_lists": len(gates),
             "gate_lists_bytes": sum(p.stat().st_size for p in gates),
             "verification": f"every gate list evaluated against its circuit (gf2 bit-sliced, 64 random vectors): {reg.verified} checked, 0 differ",
             "not_yet_lowered": nyl}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({"rows": rows, "subcircuits": len(reg.nodes), "gate_lists_verified": reg.verified}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
