"""The Lean row circuits are `pc8`'s and `pc4`'s row templates: the same computing gates in the same order, each with its
primitive, width and operands in order (an earlier gate, read plain or through core's wiring, or a constant), with the
template's parameter leaves as its input gates. `lean_rows` restates `RowCircuit.row8A`, `row8B`
(`Security/Definitions/Pouw/PearlC/RowCircuit.lean`), `row4A` and `row4B` (`Row4Circuit.lean`) run by run; the Lean proofs are
about those tables, so this test and the pinned descriptor digests are what tie them to the Programs. Unlike the tile's
check (`test_circuit_tile_lean.py`) this one reads each gate's operand order and constants from the template's own gates,
since `CallGraph` keeps neither. Each primitive's semantics is the trusted step: core's are `RowOps`' fields, PoUW's are
`RowPrims` (held to `verity.ml.protocols.pouw.circuit.words` by `test_row_prims_lean.py`), the atoms are `p.step` and `nvf4Word`, and
`F32Add_v2` is `f32Add`."""
from __future__ import annotations

import struct

import pytest

from verity.ml.circuits import Program
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.cut import WIRING, CallGraph, call_scope

pc8 = pytest.importorskip("verity.ml.protocols.pouw.circuit.pc8", reason="the row templates are `verity.ml.protocols.pouw.circuit.pc8`'s")
pc4 = pytest.importorskip("verity.ml.protocols.pouw.circuit.pc4", reason="the row templates are `verity.ml.protocols.pouw.circuit.pc4`'s")

#: each Lean `Prim` (its constructor) as the template's primitive id and width; `atom8` is the device's atom
PRIMS = {"add": ("F32Add_v2", 32), "mul": ("F32Mul_v2", 32), "div": ("F32Div_v2", 32), "fma": ("F32Fma_v2", 32),
         "sqrt": ("PouwF32SqrtRn_v1", 32), "fmaxf": ("F32Fmaxf_v1", 32), "neg": ("F32Neg_v1", 32),
         "toE4m3": ("F32ToE4m3Sat_v1", 8), "toE2m1": ("F32ToE2m1Sat_v1", 4), "e4m3ToF32": ("E4m3ToF32_v1", 32),
         "f2fpBf16": ("F2fpBf16_v1", 16), "le": ("I32Le_v1", 1), "selI32": ("SelectI32_v1", 32),
         "selBf16": ("SelectBf16_v1", 16), "band": ("BitAnd_v1", 1), "iadd": ("I32Add_v1", 32),
         "atom4": ("BlackwellNvf4OmmaDot64_v1", 32), "eTilde": ("PouwETilde_v1", 16), "bf16Neg": ("PouwBf16Neg_v1", 16),
         "sqAcc": ("PouwSqAcc_v1", 576), "sqCeil": ("PouwSqCeil_v1", 64), "sqGe": ("PouwSqGe_v1", 1)}
#: each Lean `Wire` as the template's wiring primitives, innermost first
WIRES = {"id": (), "fabs": ("F32Fabs_v1",), "bf16": ("Bf16ToF32_v1",), "bf16Fabs": ("Bf16ToF32_v1", "F32Fabs_v1")}

#: `program_digest` of the templates the Lean tables were checked against; a change to any needs the Lean re-read
DIGESTS8 = {
    ("row_a", "sm120", 128): "503976f589f901300d9efe1de563f0c521c45d883c965dfe7128f39063db5df8144f6d22fca2ae8e05ae55ea65e7db29f835a2796d4e33cbe7bda76a9ee3c1ea",
    ("row_b", "sm120", 128): "1b3d529f400664fe37780e42ac4a674008dfd651d61e6e96360cd10f9960f0408cc7d712510567a221ec00e2372275d88aceb1a97ecdd9c5185c69a8f640bd5b",
    ("row_a", "h100", 128): "fc9a8c19fdc0739dd7a44e2a68e7a2f42ac3b5f93610a1613b12a7e5765a77ed8272071d0764a7e2e280b7a36a67449ee415121b0d7ab916f6884661a4b1171c",
    ("row_b", "h100", 128): "6bca0c263b14aa9a29114a55c07ca0e774a21e04137c43898a5883ca1c445c1ec6e3eee31c533e64f13ed299f82fdad1cdc3040909421f4eb2886622d090d21c",
}
DIGESTS4 = {
    ("row_a", 64): "05776871660ac10e7e7e544ed93592eb6f58b312c9aa4cbeb8d43a2f27722b71ec8863cbe3690feb91f75d4e7b7712e5abe4068a7ce3e87fab937b27fc9df5fb",
    ("row_b", 64): "8221194fdd79643540af01ae6f4cf06032e35ed84f774277f23785842d9bd2d7a87165d32ca1072076f91936a186e1047d5c82a2db2e17e3bba55b82e4522afa",
    ("row_a", 128): "7882e13ab6e49da5da56d4c0a3008a974be80091564a32009367712db5b92fd97b66a6dc4356fc9a39e64d8d0cf9eb46e6ceba8c666795dc127924410454431e",
    ("row_b", 128): "146e9dc2971de9d4bab5844638ecc89c29d9ef45e7b13ca69a26d79bec39fd645ea34c9f748e05f7c5384243d98abce04dafdec5ee75062514c5b5162122f519",
}


def g(a: int, w: str = "id"):
    return ("g", a, w)


def c(v: int):
    return ("c", v)


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# ---- the runs both families share (`RowCircuit`'s stats) ------------------------------------------------------------------

def offsets(k: int, b: int) -> dict[str, int]:
    """`fmaAt`, `bf0At` … `rhoAt`: the stats' sections from gate `b` on."""
    n8 = k // 8
    o = {"fma": b + 2 * (k - 1)}
    o["bf0"] = o["fma"] + n8
    o["bf1"] = o["bf0"] + min(16, n8)
    o["bf2"] = o["bf1"] + min(8, n8)
    o["bf3"] = o["bf2"] + min(4, n8)
    o["bf4"] = o["bf3"] + min(2, n8)
    o["rho"] = o["bf4"] + min(1, n8)
    return o


def stats_runs(k: int, b: int, ab: str, w: str):
    """`selRun`, `fmaRun`, the five butterfly rounds and `rhoRun`: the running max of |x| (`ab` its wiring), the FMA
    partials over every 8th position (`w` their wiring), their butterfly and ρ."""
    n8, o = k // 8, offsets(k, b)
    runs = []
    runs.append([("le", [g(0, ab) if r == 0 else g(b + r - 1), g(r // 2 + 1, ab)]) if r % 2 == 0 else
                 ("selI32", [g(b + r - 1), g(r // 2 + 1, ab), g(0, ab) if r == 1 else g(b + r - 2)])
                 for r in range(2 * (k - 1))])
    runs.append([("fma", [g(8 * i, w), g(8 * i, w), c(0) if i < 32 else g(o["fma"] + i - 32)]) for i in range(n8)])
    last = lambda j: o["fma"] + j + 32 * ((n8 - 1 - j) // 32)                                       # noqa: E731
    runs.append([("add", [g(last(j)), g(last(j + 16)) if j + 16 < n8 else c(0)]) for j in range(min(16, n8))])
    for prev, at, h in (("bf0", "bf1", 8), ("bf1", "bf2", 4), ("bf2", "bf3", 2), ("bf3", "bf4", 1)):
        p = o[prev]
        runs.append([("add", [g(p + j), g(p + j + h) if j + h < n8 else c(0)]) for j in range(min(h, n8))])
    runs.append([("div", [g(o["rho"] - 1), c(f32(float(n8)))]), ("sqrt", [g(o["rho"])])])
    return runs


def chain8(code, col, gt: int, r: int):
    """`chain8`: gate `r` of a chain's 5·(K/128) gates at gate `gt`, the codes at `code t` and the other side at `col t`."""
    s = 32 * (4 * (r // 5) + r % 5)
    if r % 5 < 4:
        return ("atom8", [c(0) if r % 5 == 0 else g(gt - 1)] + [g(code(s + l)) for l in range(32)]
                + [g(col(s + l)) for l in range(32)])
    return ("add", [c(0) if r // 5 == 0 else g(gt - 5), g(gt - 1)])


def chain4(code, col, scale, gt: int, r: int):
    """`chain4`: gate `r` of a chain's K/64 NVF4 atoms at gate `gt`."""
    return ("atom4", [c(0) if r == 0 else g(gt - 1)] + [g(code(64 * r + l)) for l in range(64)]
            + [g(col(64 * r + l)) for l in range(64)] + [g(scale(4 * r + l)) for l in range(4)] + [c(56)] * 4)


# ---- pc8 ---------------------------------------------------------------------------------------------------------------

def lean_rows8(k: int, side: str):
    """`row8A K` / `row8B K`: (input widths, gates as (prim, args), outputs in return order)."""
    nin = 65 * k + 32 + (1024 if side == "b" else 0)
    widths = [32] * k + [8] * (32 + 64 * k) + [32] * (nin - 65 * k - 32)
    b, nw, n8 = nin, k // 128, k // 8
    fb0, cb0, g0 = k + 32, 33 * k + 32, 65 * k + 32
    o = offsets(k, b)
    runs = stats_runs(k, b, "fabs", "id")
    rho, sF = o["rho"] + 1, o["fma"] - 1
    al = o["rho"] + 2
    runs.append([("mul", [c(0x40800000), g(rho)]), ("add", [g(sF), g(al)]), ("fmaxf", [g(al + 1), c(0x2F800000)]),
                 ("div", [c(0x43E00000), g(al + 2)]), ("mul", [g(rho), g(al + 3)]), ("mul", [g(al + 4), c(0x3AB504F3)])])
    alpha, beta, sq = al + 3, al + 5, al + 6
    runs.append([("sqAcc", [c(0) if i == 0 else g(sq + i - 1), g(8 * i)]) for i in range(n8)])
    th = sq + n8
    runs.append([("sqCeil", [g(th - 1), c(n8)])])
    dead = th + 1
    runs.append([[("sqGe", [g(r // 3), g(th)]), ("selI32", [g(dead + r - 1), c(1), c(0)]),
                  ("iadd", [c(0) if r == 2 else g(dead + r - 3), g(dead + r - 1)])][r % 3] for r in range(3 * k)])
    ok0 = dead + 3 * k
    runs.append([("fma", [g(rho), g(alpha), c(0xBF800000)]), ("le", [c(0), g(ok0)]), ("le", [g(sF), c(0x7F7FFFFF)]),
                 ("le", [g(rho), c(0x7F7FFFFF)]), ("band", [g(ok0 + 2), g(ok0 + 3)]), ("le", [g(ok0 - 1), c(k // 64)]),
                 ("band", [g(ok0 + 1), g(ok0 + 5)]), ("band", [g(ok0 + 4), g(ok0 + 6)])])
    ok, ep = ok0 + 7, ok0 + 8
    runs.append([[("e4m3ToF32", [g(k + r // 3)]), ("mul", [g(beta), g(ep + r - 1)]), ("toE4m3", [g(ep + r - 1)])][r % 3]
                 for r in range(96)])
    ew = ep + 96
    runs.append([("e4m3ToF32", [g(ep + 3 * p + 2)]) for p in range(32)])
    cd = ew + 32
    runs.append([[("mul", [g(alpha), g(r // 3)]),
                  ("atom8", [g(cd + r - 1)] + [g(ep + 3 * p + 2) for p in range(32)]
                   + [g(fb0 + 32 * (r // 3) + p) for p in range(32)]),
                  ("toE4m3", [g(cd + r - 1)])][r % 3] for r in range(3 * k)])
    code = lambda t: cd + 3 * t + 2                                                                # noqa: E731
    post = cd + 3 * k
    if side == "a":
        ln = 5 * nw + 1
        runs.append([chain8(code, lambda t, q=r // ln: cb0 + 32 * t + q, post + r, r % ln) if r % ln < 5 * nw else
                     ("f2fpBf16", [g(post + r - 1)]) for r in range(32 * ln)])
        bw = post + 32 * ln
        runs.append([("f2fpBf16", [g(ew + p)]) for p in range(32)])
        runs.append([("selBf16", [g(ok), c(1), c(0)])])
        outs = [code(t) for t in range(k)] + [bw + p for p in range(32)] + [post + p * ln + ln - 1 for p in range(32)]
        return widths, [x for run in runs for x in run], outs + [bw + 32]
    ln = 5 * nw + 35

    def mid(r: int):
        q, s = r // ln, r % ln
        if s < 32:
            return ("fma", [g(ew + s), g(g0 + 32 * s + q), c(0) if s == 0 else g(post + r - 1)])
        if s < 32 + 5 * nw:
            return chain8(code, lambda t: 33 * k + 32 + 32 * t + q, post + r, s - 32)
        return [("neg", [g(post + r - 1)]), ("add", [g(post + r - 2 - 5 * nw), g(post + r - 1)]),
                ("f2fpBf16", [g(post + r - 1)])][s - 32 - 5 * nw]
    runs.append([mid(r) for r in range(32 * ln)])
    nb = post + 32 * ln
    runs.append([("f2fpBf16", [g(ew + r // 2)]) if r % 2 == 0 else ("bf16Neg", [g(nb + r - 1)]) for r in range(64)])
    runs.append([("selBf16", [g(ok), c(1), c(0)])])
    outs = [code(t) for t in range(k)] + [post + q * ln + ln - 1 for q in range(32)] + [nb + 2 * p + 1 for p in range(32)]
    return widths, [x for run in runs for x in run], outs + [nb + 64]


# ---- pc4 ---------------------------------------------------------------------------------------------------------------

def lean_rows4(k: int, side: str):
    """`row4A K` / `row4B K`: (input widths, gates as (prim, args), outputs in return order)."""
    nin = 65 * k + 32 + (1024 if side == "b" else 0)
    widths = [16] * k + [4] * (32 + 64 * k) + [32] * (nin - 65 * k - 32)
    b, nc, nbk = nin, k // 64, k // 16
    fb0, cb0, g0 = k + 32, 33 * k + 32, 65 * k + 32
    o = offsets(k, b)
    runs = stats_runs(k, b, "bf16Fabs", "bf16")
    rho, sF, a0 = o["rho"] + 1, o["fma"] - 1, o["rho"] + 2
    runs.append([("mul", [g(rho), c(0x3E800000)]), ("mul", [c(0x40800000), g(a0)]), ("add", [g(sF), g(a0 + 1)]),
                 ("fmaxf", [g(a0 + 2), c(0x2F800000)]), ("div", [c(0x45280000), g(a0 + 3)]), ("mul", [g(a0), g(a0 + 4)]),
                 ("mul", [g(a0 + 5), c(0x3EA920A7)]), ("toE4m3", [g(a0 + 6)]), ("le", [c(0x3C700000), g(a0 + 6)]),
                 ("le", [g(a0 + 6), c(0x7F7FFFFF)]), ("band", [g(a0 + 8), g(a0 + 9)]), ("le", [g(sF), c(0x7F7FFFFF)]),
                 ("band", [g(a0 + 11), g(a0 + 10)])])
    alpha, ae, ok, d0 = a0 + 4, a0 + 7, a0 + 12, a0 + 13
    runs.append([("mul", [g(alpha), g(r // 2, "bf16")]) if r % 2 == 0 else
                 ("atom4", [g(d0 + r - 1)] + [g(k + p) for p in range(32)] + [c(0)] * 32
                  + [g(fb0 + 32 * (r // 2) + p) for p in range(32)] + [c(0)] * 32 + [g(ae)] * 4 + [c(24)] * 4)
                 for r in range(2 * k)])
    d = lambda t: d0 + 2 * t + 1                                                                   # noqa: E731
    bk = d0 + 2 * k

    def blk(r: int):
        q, s = r // 52, r % 52
        base, t0 = bk + 52 * q, 16 * q
        if s < 15:
            return ("fmaxf", [g(d(t0), "fabs") if s == 0 else g(base + s - 1), g(d(t0 + s + 1), "fabs")])
        if s < 20:
            return [("mul", [g(base + 14), c(0x3E2AAAAB)]), ("fmaxf", [g(base + 15), c(0x3B000000)]),
                    ("toE4m3", [g(base + 16)]), ("e4m3ToF32", [g(base + 17)]), ("div", [c(0x3F800000), g(base + 18)])][s - 15]
        u = s - 20
        return ("mul", [g(d(t0 + u // 2)), g(base + 19)]) if u % 2 == 0 else ("toE2m1", [g(base + s - 1)])
    runs.append([blk(r) for r in range(52 * nbk)])
    code = lambda t: bk + 52 * (t // 16) + 21 + 2 * (t % 16)                                       # noqa: E731
    scale = lambda i: bk + 52 * i + 17                                                             # noqa: E731
    post = bk + 52 * nbk
    if side == "a":
        ln = nc + 1
        runs.append([chain4(code, lambda t, q=r // ln: cb0 + 32 * t + q, scale, post + r, r % ln) if r % ln < nc else
                     ("f2fpBf16", [g(post + r - 1)]) for r in range(32 * ln)])
        et = post + 32 * ln
        runs.append([("eTilde", [g(k + p), g(ae)]) for p in range(32)])
        runs.append([("selBf16", [g(ok), c(1), c(0)])])
        outs = ([code(t) for t in range(k)] + [scale(i) for i in range(nbk)] + [et + p for p in range(32)]
                + [post + p * ln + ln - 1 for p in range(32)])
        return widths, [x for run in runs for x in run], outs + [et + 32]
    et, mid0, ln = post, post + 32, nc + 35
    runs.append([("eTilde", [g(k + p), g(ae)]) for p in range(32)])

    def mid(r: int):
        q, s = r // ln, r % ln
        if s < 32:
            return ("fma", [g(et + s, "bf16"), g(g0 + 32 * s + q), c(0) if s == 0 else g(mid0 + r - 1)])
        if s < 32 + nc:
            return chain4(code, lambda t: cb0 + 32 * t + q, scale, mid0 + r, s - 32)
        return [("neg", [g(mid0 + r - 1)]), ("add", [g(mid0 + r - 2 - nc), g(mid0 + r - 1)]),
                ("f2fpBf16", [g(mid0 + r - 1)])][s - 32 - nc]
    runs.append([mid(r) for r in range(32 * ln)])
    nb = mid0 + 32 * ln
    runs.append([("bf16Neg", [g(et + p)]) for p in range(32)])
    runs.append([("selBf16", [g(ok), c(1), c(0)])])
    outs = ([code(t) for t in range(k)] + [scale(i) for i in range(nbk)] + [mid0 + q * ln + ln - 1 for q in range(32)]
            + [nb + p for p in range(32)])
    return widths, [x for run in runs for x in run], outs + [nb + 32]


# ---- the template's side -------------------------------------------------------------------------------------------------

def ir_table(fn):
    """The template's computing gates in order, each as (primitive id, width, operands): an operand is a parameter leaf
    or an earlier computing gate, by its index after the inputs, with the wiring it is read through (innermost first),
    or a constant's value."""
    cg, call = call_scope(fn)
    lo, n = call.offset, fn.gates
    gates = [cg.gate(lo + j) for j in range(n)]
    nin = len(CallGraph(fn).input_ids)
    idx: dict[int, int] = {}

    def arg(o: int):
        if not 0 <= o - lo < n:
            return ("g", o, ())
        gt = gates[o - lo]
        if not gt.prim.params:
            return ("c", gt.prim.evaluate())
        if gt.prim.id in WIRING:
            (inner,) = gt.operands
            a = arg(inner)
            assert a[0] == "g", "wiring reads a gate"
            return ("g", a[1], a[2] + (gt.prim.id,))
        return ("g", idx[o - lo], ())

    table = []
    for j, gt in enumerate(gates):
        if not gt.prim.params or gt.prim.id in WIRING:
            continue
        idx[j] = nin + len(table)
        table.append((gt.prim.id, gt.prim.ret.w, [arg(o) for o in gt.operands]))
    return table


def check(fn, widths, table, outs, atom: str | None = None):
    prims = dict(PRIMS, atom8=(atom, 32))
    graph = CallGraph(fn)
    cg, _ = call_scope(fn)
    base = len(widths)
    assert sorted(graph.input_ids) == list(range(base))
    assert [cg.gate(r).prim.ret.w for r in range(base)] == widths
    ir = ir_table(fn)
    assert len(ir) == len(table) == graph.n
    for i, ((pid, w, args), (kind, largs)) in enumerate(zip(ir, table)):
        want = [a if a[0] == "c" else ("g", a[1], WIRES[a[2]]) for a in largs]
        assert (pid, w, args) == (*prims[kind], want), (base + i, kind)
    assert list(graph.width) == [prims[k][1] for k, _ in table]
    reads = [set() for _ in range(graph.n)]
    for s, d in zip(graph.src, graph.dst):
        reads[d].add(base + s)
    for ix, d in graph.input_reads:
        reads[d].add(graph.input_ids[ix])
    assert reads == [{a[1] for a in args if a[0] == "g"} for _, args in table]
    assert [base + o for o in graph.outputs] == outs


CASES8 = [(dev, k) for dev in ("sm120", "h100") for k in (128, 256)] + [("sm120", 384), ("sm120", 1024)]


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("dev,k", CASES8)
def test_the_lean_row8_tables_are_the_templates(dev, k, side):
    fn = pc8.templates(k, (1, 1), dev)["row_" + side]
    check(fn, *lean_rows8(k, side), atom=pc8.ATOMS[pc8.DEVICES[dev].atom.name].id)


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("k", [64, 128, 192, 256, 1024])
def test_the_lean_row4_tables_are_the_templates(k, side):
    check(pc4.templates(k)["row_" + side], *lean_rows4(k, side))


@pytest.mark.parametrize("role,dev,k", sorted(DIGESTS8))
def test_the_checked_row8_templates_are_pinned(role, dev, k):
    assert program_digest(Program(pc8.templates(k, (1, 1), dev)[role])) == DIGESTS8[role, dev, k]


@pytest.mark.parametrize("role,k", sorted(DIGESTS4))
def test_the_checked_row4_templates_are_pinned(role, k):
    assert program_digest(Program(pc4.templates(k)[role])) == DIGESTS4[role, k]


#: the deployed depth's row templates (sm_120), each table check and digest about 20 seconds
DEPLOYED = {
    ("pc8", "row_a"): "694e71b1f6684ede5ad3e49e365dd0548f73b8ac65f2cf96a16cb9774d59eec09174fb6b1499b671a48f4e510f69d3062b6fcdebb2cd469fdc602a401ba97eb6",
    ("pc8", "row_b"): "26ab2748b86b3557171b28ae0da02e2c61f5f79221edb9e3e77f2fec34fb57954ed930040b7edc2ca45f4d04847eb97d1bcb4dacdf99b5e9b6f48c230c48e33a",
    ("pc4", "row_a"): "fc08f808658abb29a2d5e7eb566409ae39e350a3999a990964765fd4c6593ee8f4f5fdcafad9e8a7347e48592349a66b753bad4b3587d7ab745523b87b0b78c1",
    ("pc4", "row_b"): "463b39057954204873d5c5efc969682ca54bd2f78732913869decdb1973bd1d6f558125a8cd4db48c74c516892c406824ac1cbb140be01c0e7c07e50f6fff297",
}


@pytest.mark.slow
@pytest.mark.parametrize("family,role", sorted(DEPLOYED))
def test_the_deployed_rows_are_the_lean_tables(family, role):
    """Catches a change to a row template at K = 8192 that the checks at K <= 1024 and the K = 128 pins don't see, such
    as one gated on the depth."""
    side = role[-1]
    if family == "pc8":
        fn = pc8.templates(8192, (1, 1), "sm120")[role]
        check(fn, *lean_rows8(8192, side), atom=pc8.ATOMS[pc8.DEVICES["sm120"].atom.name].id)
    else:
        fn = pc4.templates(8192)[role]
        check(fn, *lean_rows4(8192, side))
    assert program_digest(Program(fn)) == DEPLOYED[family, role]
