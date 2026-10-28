"""`verity_flock.derive`, the prover's mirror of the verifier's `derive`: a flat unit's rows are `ir_lower.netlist`'s byte
for byte; generated reads are `Flock.Lookup.buildV2`'s (#202's digests); on hand-built and random statements (placed and
inline calls and reads, exports, nested placement, the unit's parts), the rows solved in logical order compute
`circuit_types.evaluate`, every row holds, and the logical order is topological. `derive_vectors.json` pins both orders for
the Lean `derive` to reproduce: ``python test_derive.py --write``."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

from verity.ml.circuits import library
from verity_flock import circuit_types as CT
from verity_flock import derive as D
from verity_flock import layouts as LY

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verifier"))
from test_circuit_types import _adder, _full_adder  # noqa: E402

VECTORS = HERE / "derive_vectors.json"
T = 64
MASK = (1 << T) - 1


def _sha(ws):
    return hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in ws)).hexdigest()


# --- solving derived rows -------------------------------------------------------------------------------------------------

def solve(dv: D.Derivation, ins: list[int]) -> tuple[list[int], bool]:
    """Solve the unit's rows (Δ applied) in logical order on bit-sliced inputs; return its outputs and whether every row
    holds, each read's product rows with their table-direct B side."""
    d = dv.unit
    rows = dv.block_rows()
    prod: dict[int, tuple] = {}
    for r in d.reads:
        nh = 1 << (r.n - r.lo)
        for j in range(r.k):
            for h in range(nh):
                prod[r.prod0 + j * nh + h] = (r, j, h)
    z: dict = {D.ONE: MASK}

    def ev(xs):
        v = 0
        for x in xs:
            v ^= z.get(x, 0)
        return v

    def implicit(c):
        if c not in prod:
            return 0
        r, j, h = prod[c]
        ws, v = dv.tables[r.table], 0
        for l in range(1 << r.lo):
            if (ws[(h << r.lo) + l] >> j) & 1:
                v ^= z.get(r.lo_top + l, 0)
        return v

    inputs = set(d.in_cols)
    for i, c in enumerate(d.in_cols):
        z[c] = ins[i]
    for c, a, b in dv.logical():
        if c == D.ONE or c in inputs:
            continue
        z[c] = ev(a) & (ev(b) ^ implicit(c))
    holds = all(z.get(c, 0) == ev(a) & (ev(b) ^ implicit(c)) for c, (a, b) in rows.items() if c not in inputs)
    holds &= all(z[c] == ev(rows[c][0]) & ev(rows[c][1]) for c in inputs)
    return [z.get(c, 0) for c in d.out_cols], holds


def check(types, layouts, unit, tables, seed=0):
    dv = D.Derivation(types, layouts, unit, tables)
    t = types[layouts[unit].type]
    rng = random.Random(seed)
    ins = [rng.getrandbits(T) for _ in range(t.in_bits)]
    want = CT.evaluate(types, t.digest(), ins, T, lambda s, i: tables[s][i])
    got, holds = solve(dv, ins)
    assert holds, "a derived row does not hold"
    assert got == want, "the derived rows compute something else"
    for j, lay in enumerate(layouts):
        if lay.own[0] == "derive":
            assert dv.physical(j).own == LY.own_size(types, layouts, j, dv.info.get)
    seen = {D.ONE}
    for c, a, b in dv.logical():
        assert all(x in seen or x == c and c in dv.unit.in_cols for x in a + b), f"row {c} reads a later row"
        seen.add(c)
    return dv


# --- cases ----------------------------------------------------------------------------------------------------------------

def _lay(t, rl, calls=(), reads=(), own=("derive",), ins=(), outs=()):
    return LY.BlockLayout(t.digest() if t is not None else None, rl, tuple(ins), tuple(outs), own, tuple(calls), tuple(reads))


def _wrapper(u: CT.CircuitType) -> CT.CircuitType:
    return CT.CircuitType(u.label, ((u.in_bits, 1),), ((u.out_bits, 1),), (u.digest(),), (),
                          ((CT.CALL, 0, tuple(((i,), 0) for i in range(u.in_bits))),),
                          tuple(((u.in_bits + i,), 0) for i in range(u.out_bits)))


def _reader(ws, ib, vb, gates=True) -> tuple[CT.CircuitType, str]:
    """A read of the table at its inputs' XOR with a shifted copy, then ANDs over its value and inputs."""
    sha = _sha(ws)
    items = [(CT.AND, ((0,), 0), ((1,), 1))] + [(CT.READ, 0, tuple(((b,), 0) if b else ((0, ib), 0) for b in range(ib)))]
    n = ib + 1
    outs = [((n + j,), j & 1) for j in range(vb)]
    if gates:
        items.append((CT.AND, ((n,), 0), ((n + vb - 1,), 0)))
        outs.append(((n + vb,), 0))
    t = CT.CircuitType(library.PROGRAM, ((ib, 1),), ((len(outs), 1),), (), ((sha, ib, vb),), tuple(items), tuple(outs))
    return t, sha


def hand_cases():
    """(name, types, layouts, unit, tables)."""
    fa = _full_adder()
    add4 = _adder(fa)
    w = _wrapper(add4)
    T_ = {x.digest(): x for x in (fa, add4, w)}
    L_fa = _lay(fa, 9)
    out = [("full adder, flat", {fa.digest(): fa}, [_lay(fa, None)], 0, {}),
           ("4-bit adder, calls in block slots (all inputs exported)", T_, [L_fa, _lay(add4, None, calls=[(i, 0, i) for i in range(4)])], 1, {}),
           ("4-bit adder, calls inlined", T_, [L_fa, _lay(add4, None, calls=[(i, 0, None) for i in range(4)])], 1, {}),
           ("wrapper over the 4-bit adder, nested placement", T_,
            [L_fa, _lay(add4, 12, calls=[(i, 0, i + 1) for i in range(4)]), _lay(w, None, calls=[(0, 1, 0)])], 2, {}),
           ("wrapper over the 4-bit adder, both levels inlined", T_,
            [L_fa, _lay(add4, 9, calls=[(i, 0, None) for i in range(4)]), _lay(w, None, calls=[(0, 1, None)])], 2, {})]
    rng = random.Random(11)
    ws = [rng.getrandbits(6) | 0x40 for _ in range(64)]
    rt, sha = _reader(ws, 6, 8)
    TR = {rt.digest(): rt}
    out += [("a read inline", TR, [_lay(rt, None, reads=[(1, None, 3)])], 0, {sha: ws}),
            ("a read placed in a block slot", TR, [_lay(None, 9, own=("gen", sha, 6, 8, 3)), _lay(rt, None, reads=[(1, 0, 0)])], 1, {sha: ws})]
    wr = _wrapper(rt)
    TW = {rt.digest(): rt, wr.digest(): wr}
    out += [("a read placed inside a placed callee", TW,
             [_lay(None, 10, own=("gen", sha, 6, 8, 2)), _lay(rt, 11, reads=[(1, 0, 1)]), _lay(wr, None, calls=[(0, 1, 0)])], 2, {sha: ws})]
    return out


def random_cases(n, seed):
    import layout_agree as A
    rng = random.Random(seed)
    out = []
    for i in range(n):
        tables, tlist = {}, []
        for ib, vb in ((3, 4), (4, 6), (5, 8)):
            ws = A.words_of(rng, ib, vb)
            tables[_sha(ws)] = ws
            tlist.append((_sha(ws), ib, vb))
        info = {s: LY.table_info(tables[s]) for s, _, _ in tlist}.get
        pool = [A.flat(rng, tlist) for _ in range(rng.randint(1, 3))]
        for u in pool:
            A._width[u.digest()] = u.out_bits
        caller = A.tree(rng, pool)
        A._width[caller.digest()] = caller.out_bits
        used = [t for t in pool if t.digest() in caller.calls]
        top = [caller] + ([_wrapper(caller)] if rng.random() < 0.5 else [])
        types = {t.digest(): t for t in used + top}
        order = list(dict.fromkeys(t.digest() for t in used + top))
        layouts, unit = A.make_layouts(rng, types, order, top[-1].digest(), info)
        out.append((f"random {seed}.{i}", types, layouts, unit, {s: tables[s] for s in tables}))
    return out


def fanout_case(m, r):
    callee = CT.CircuitType(library.PROGRAM, ((1, 1),), ((1, 1),), (), (), ((CT.AND, ((0,), 0), ((0,), 1)),), (((1,), 0),))
    ws = [(i * 7 + 3) & 0xF for i in range(16)]
    sha = _sha(ws)
    items = [(CT.CALL, 0, (((i,), 0),)) for i in range(m)] + [(CT.READ, 0, tuple(((i % m,), 0) for _ in range(4))) for i in range(r)]
    outs = [((m + i,), 0) for i in range(m)] + [((2 * m + 4 * i + b,), 0) for i in range(r) for b in range(4)]
    unit = CT.CircuitType(library.PROGRAM, ((m, 1),), ((len(outs), 1),), (callee.digest(),), ((sha, 4, 4),), tuple(items), tuple(outs))
    layouts = [_lay(callee, 10), _lay(None, 10, own=("gen", sha, 4, 4, 2)),
               _lay(unit, None, calls=[(i, 0, i) for i in range(m)], reads=[(m + i, 1, i) for i in range(r)])]
    return (f"fan-out {m} calls, {r} reads", {callee.digest(): callee, unit.digest(): unit}, layouts, 2, {sha: ws})


# --- tests ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", hand_cases(), ids=lambda c: c[0])
def test_hand_built_statements(case):
    _, types, layouts, unit, tables = case
    check(types, layouts, unit, tables)


def test_random_statements():
    for name, types, layouts, unit, tables in random_cases(120, 7):
        try:
            check(types, layouts, unit, tables, seed=hash(name) & 0xFFFF)
        except AssertionError as e:
            raise AssertionError(f"{name}: {e}") from None


def test_a_wide_unit():
    _, types, layouts, unit, tables = fanout_case(20, 20)
    dv = check(types, layouts, unit, tables)
    assert len(dv.unit.parts) == 40 and dv.unit.delta[0][1] == D.ONE


def _small_tables():
    """#202's `lookup_v2_vectors.small_tables()`, reproduced: the same generator, so the same tables."""
    rng = random.Random(20260928)

    def words(n, f):
        return [f(i) for i in range(1 << n)]

    return [("full-8", 6, 3, 8, words(6, lambda i: rng.getrandbits(8))),
            ("low5-const-high", 6, 2, 12, words(6, lambda i: rng.getrandbits(5) | (1 << 7) | (1 << 10))),
            ("const-bit-inside", 7, 3, 10, words(7, lambda i: (rng.getrandbits(9) & ~(1 << 4)) | (1 << 2))),
            ("all-constant", 5, 2, 16, words(5, lambda i: 0x0000A5C3)),
            ("one-word-top", 7, 4, 31, words(7, lambda i: rng.getrandbits(20) | ((1 << 30) if i == 77 else 0))),
            ("random-20", 8, 5, 31, words(8, lambda i: rng.getrandbits(20)))]


#: `rows_sha512` per small table in #202's `lookup_v2_vectors.json` (at `10d8e46b`): the verifier lane's `Flock.Lookup.rowsDigest`
LEAN_ROWS = {
    "full-8": "02199433a3386f563cd5656b18df015b5b4632478140036f1e3f57625922b051686b530a57c71faf07989e0f63094e8fa386b50c373ef01a3fccfbe5c2627ae1",
    "low5-const-high": "de1c49c45b82cfa80bd8aafa3c288d3326fefb1dbf5d84a249bb5121c7222d9561a2d3646d0cca9f3e035a95ec0bcbac3089c19711c6e1e076cc4425d0f6d195",
    "const-bit-inside": "6c15e7851edb92dc11b92291323c871093b108960dbfe82671b266bf7a644168a0bce4ce4602aad7af13da0e552ab1d21b40d751e1feace53d07a2e8cd7c7b38",
    "all-constant": "2113813d6a6f6c330cb63d4f084d7b294e99f472c4f57abc12b898b46723093ef70ebfebc51358250f77fca3c430ab38e2eaef9cd20f0d96d7c907ba35aeb45c",
    "one-word-top": "959c1450808843a4f24f5045e4895f7cdd52782b14f0973ede72c464a67d788778fc7c18327ae5b34bea15b849b617083c4461222463d021bf5fb1448f35eddf",
    "random-20": "d7157af1a3ca04e4743659f739c6252cde451f19fe75ab18bf849292e451e577fed4056941962d153a82722dde6e3544cf18ade24c2b67c1d708aaf9c1b8fd39",
}


def test_generated_reads_are_lean_buildv2():
    """Each small table of #202's vectors: this generator's rows digest is the Lean one."""
    lean = HERE.parent / "verifier" / "lookup_v2_vectors.json"
    want = ({e["label"]: e["rows_sha512"] for e in json.loads(lean.read_text())["small"]} if lean.exists() else LEAN_ROWS)
    for label, n, lo, bits, ws in _small_tables():
        d = D.gen_read(ws, _sha(ws), n, lo, bits)
        assert D.rows_digest(d, n, lo, bits) == want[label], label


def test_flat_units_are_ir_lowers_netlists():
    pytest.importorskip("verity.ml.definitions.registry.prims")
    from verity_flock import ir_lower as IL
    from verity_numerical.bench import lowerings, templates as TM

    for template, params in (("rope-head", {"D": 64}), ("silu-mul", {"I": 8192}), ("silu-mul-v2", {"I": 256})):
        low = lowerings.registry("C-Flock").module(template).lowering(TM.subcircuit(template, **params))
        U, C = low.units, low.circuit
        ins = [b for j in range(len(U.in_widths)) for b in C.inputs[f"in{j}"]]
        outs = [f for j in range(len(U.out_widths)) for f in C.outputs[f"out{j}"]]
        t = CT.from_gf2(C, ins, outs, [(1, w) for w in U.in_widths], [(1, w) for w in U.out_widths], library.PROGRAM, {})
        cut = lambda g: tuple(itertools.accumulate(len(b) for _, _, b in g))[:-1]  # noqa: E731
        lay = LY.BlockLayout(t.digest(), None, cut(low.layout.in_groups), cut(low.layout.out_groups), ("derive",), (), ())
        dv = D.Derivation({t.digest(): t}, [lay], 0, {})
        assert D.netlist(dv.unit, low.name) == IL.netlist(low.layout, low.name), template


# --- vectors --------------------------------------------------------------------------------------------------------------

def _archive(types, layouts, unit, tables):
    order, seen = [], set()

    def visit(d):
        if d in seen:
            return
        seen.add(d)
        for c in types[d].calls:
            visit(c)
        order.append(d)

    for d in types:
        visit(d)
    return {"types": [types[d].content() for d in order], "tables": {s: {"words": ws} for s, ws in tables.items()},
            "layouts": [l.to_json() for l in layouts], "unit_layout": unit}


def vector(case) -> dict:
    name, types, layouts, unit, tables = case
    dv = D.Derivation(types, layouts, unit, tables)
    h = lambda s: hashlib.sha512(s.encode()).hexdigest()  # noqa: E731
    inner = {}
    for j, lay in enumerate(layouts):
        if j != unit:
            text = D.netlist(dv.physical(j), lay.digest()[:16])
            inner[lay.digest()] = {"rows": dv.physical(j).size, "sha512": h(text)}
    unit_text = D.netlist(dv.unit, layouts[unit].digest()[:16])
    return {"name": name, "archive": _archive(types, layouts, unit, tables),
            "physical": {"layouts": inner, "unit": {"rows": dv.unit.size, "sha512": h(unit_text)},
                         "delta": h(D.delta_text(dv))},
            "logical": {"rows": len(dv.logical()), "sha512": h(D.logical_text(dv))}}


def vector_cases():
    return hand_cases() + [fanout_case(20, 20)] + random_cases(12, 2026)


def vectors() -> dict:
    return {"about": "derive vectors: each archive (flock-verify layout's format) with the SHA-512 of the rows the "
                     "prover's derive (verity_flock.derive) gives it: each inner layout's physical rows and the unit's own "
                     "region as flock-ir-unit/v2 text with READ lines (derive.netlist), the unit's parts and Δ "
                     "(derive.delta_text), and the logical list (derive.logical_text). Written by test_derive.py --write.",
            "cases": [vector(c) for c in vector_cases()]}


def test_vectors_are_current():
    assert json.loads(VECTORS.read_text()) == vectors()


if __name__ == "__main__" and "--write" in sys.argv:
    v = vectors()
    cases = ",\n ".join(json.dumps(c, separators=(",", ":")) for c in v["cases"])
    VECTORS.write_text("{\"about\": " + json.dumps(v["about"]) + ",\n \"cases\": [\n " + cases + "\n]}\n")
