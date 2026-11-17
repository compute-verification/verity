"""The Lean row and tile circuits are `pc4`'s row and tile templates, as `test_circuit_row_lean` and
`test_circuit_tile_lean` hold `pc8`'s, on their helpers. `lean_rows4` restates `row4A` and `row4B`
(`security/Definitions/Pouw/PearlC/Row4Circuit.lean`) run by run, and `lean_table4` restates `TileCircuit.tile4Gate`
(`Tile4Circuit.lean`, `tile4Spec K TM TN`); the Lean proofs (`tile4_correct` in `Pouw.SecurityProofs.PearlC`) are about
those tables, so this test and the pinned descriptor digests are what tie them to the Programs."""
from __future__ import annotations

import pytest

from test_circuit_row_lean import c, check, g, offsets, stats_runs
from verity.circuits import Program
from verity.circuits.codec import program_digest
from verity.circuits.cut import CallGraph, call_scope

pc4 = pytest.importorskip("verity.protocols.experimental.pearl_c4.prover.pc4", reason="the row templates are `verity.protocols.experimental.pearl_c4.prover.pc4`'s")

#: `program_digest` of the row templates the Lean tables were checked against; a change to any needs the Lean re-read
ROW_DIGESTS4 = {
    ("row_a", 64): "05776871660ac10e7e7e544ed93592eb6f58b312c9aa4cbeb8d43a2f27722b71ec8863cbe3690feb91f75d4e7b7712e5abe4068a7ce3e87fab937b27fc9df5fb",
    ("row_b", 64): "8221194fdd79643540af01ae6f4cf06032e35ed84f774277f23785842d9bd2d7a87165d32ca1072076f91936a186e1047d5c82a2db2e17e3bba55b82e4522afa",
    ("row_a", 128): "7882e13ab6e49da5da56d4c0a3008a974be80091564a32009367712db5b92fd97b66a6dc4356fc9a39e64d8d0cf9eb46e6ceba8c666795dc127924410454431e",
    ("row_b", 128): "146e9dc2971de9d4bab5844638ecc89c29d9ef45e7b13ca69a26d79bec39fd645ea34c9f748e05f7c5384243d98abce04dafdec5ee75062514c5b5162122f519",
}


def chain4(code, col, scale, gt: int, r: int):
    """`chain4`: gate `r` of a chain's K/64 NVF4 atoms at gate `gt`."""
    return ("atom4", [c(0) if r == 0 else g(gt - 1)] + [g(code(64 * r + p)) for p in range(64)]
            + [g(col(64 * r + p)) for p in range(64)] + [g(scale(4 * r + p)) for p in range(4)] + [c(56)] * 4)


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


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("k", [64, 128, 192, 256, 1024])
def test_the_lean_row4_tables_are_the_templates(k, side):
    check(pc4.templates(k)["row_" + side], *lean_rows4(k, side))


@pytest.mark.parametrize("role,k", sorted(ROW_DIGESTS4))
def test_the_checked_row4_templates_are_pinned(role, k):
    assert program_digest(Program(pc4.templates(k)[role])) == ROW_DIGESTS4[role, k]


#: the deployed depth's row templates, each table check and digest about 20 seconds
DEPLOYED_ROWS = {
    "row_a": "fc08f808658abb29a2d5e7eb566409ae39e350a3999a990964765fd4c6593ee8f4f5fdcafad9e8a7347e48592349a66b753bad4b3587d7ab745523b87b0b78c1",
    "row_b": "463b39057954204873d5c5efc969682ca54bd2f78732913869decdb1973bd1d6f558125a8cd4db48c74c516892c406824ac1cbb140be01c0e7c07e50f6fff297",
}


@pytest.mark.slow
@pytest.mark.parametrize("role", sorted(DEPLOYED_ROWS))
def test_the_deployed_rows_are_the_lean_tables(role):
    """Catches a change to a row template at K = 8192 that the checks at K <= 1024 and the K = 128 pins don't see, such
    as one gated on the depth."""
    fn = pc4.templates(8192)[role]
    check(fn, *lean_rows4(8192, role[-1]))
    assert program_digest(Program(fn)) == DEPLOYED_ROWS[role]


#: `program_digest` of the `pc4` templates the Lean table was checked against
TILE_DIGESTS4 = {
    (128, (1, 1)): "7cc96f4aa6a2b68a676404f3be86c997cb6d5889d67b4336adcb2301bb2b42dbf59011ee318ce0af96bcf2135df08607339dce9c7128b64f64e865e064749476",
    (128, (64, 64)): "81ac3d445c5518b223309462102f5a1a4381c91b2830d7a92937ecfb97e323b50ebf3fa186579bb5234075477c1a34f64616e4934173483f89bd780d8c8c9516",
}

#: the deployed tile (`pc4.TILE`) at K = 8192
DEPLOYED_TILE4 = {(8192, (64, 64)): "5d52d991c237f460928a26c00019c8ad906d0b50a95a9fa6ee5c74a95b8e0ac06d22d1e0315317b91dc87878e3f01796cd59d0d697cc3f641491a8c309dd9a19"}


def lean_table4(k: int, tm: int, tn: int):
    """`tile4Gate K TM TN`'s computed gates as (kind, reads) after its `tile4In` input gates, its input widths, and its
    outputs in the template's output-leaf order (every `c4Gate`, then every `u4Gate`, row-major)."""
    nc, row = k // 64, k + k // 16
    base, ln = (tm + tn) * row + 64 * (tm + tn), nc + 4
    widths = [4] * (tm * k) + [8] * (tm * (k // 16)) + [4] * (tn * k) + [8] * (tn * (k // 16)) + [16] * (64 * (tm + tn))
    gates, cs, us = [], [], []
    for i in range(tm):
        for j in range(tn):
            e = base + (i * tn + j) * ln
            a, b = i * k, tm * row + j * k
            sa, sb = tm * k + i * (k // 16), tm * row + tn * k + j * (k // 16)
            pa, pb = (tm + tn) * row + 64 * i, (tm + tn) * row + 64 * tm + 64 * j
            for r in range(ln):
                g = e + r
                if r < nc:
                    gates.append(("atom", ([] if r == 0 else [g - 1]) + [a + 64 * r + p for p in range(64)]
                                  + [b + 64 * r + p for p in range(64)] + [sa + 4 * r + p for p in range(4)]
                                  + [sb + 4 * r + p for p in range(4)]))
                else:
                    s = r - nc
                    gates.append(("peel", [g - 1] + [pa + 16 * s + p for p in range(16)]
                                  + [pb + 16 * s + p for p in range(16)]))
            cs.append(e + nc - 1)
            us.append(e + ln - 1)
    return base, widths, gates, cs + us


CASES4 = [(k, (1, 1)) for k in (64, 128, 1024, 8192)] + [(128, (2, 3)), (256, (3, 2)), (1024, (2, 2)), (128, (64, 64))]


@pytest.mark.parametrize("k,tile", CASES4)
def test_the_lean_tile4_table_is_the_templates_call_graph(k, tile):
    tm, tn = tile
    fn = pc4.templates(k, tile)["tile"]
    g = CallGraph(fn)
    c, _ = call_scope(fn)
    base, widths, table, outs = lean_table4(k, tm, tn)
    assert sorted(g.input_ids) == list(range(base))
    assert [c.gate(r).prim.ret.w for r in range(base)] == widths
    names = {v: n for n, v in g.prims.items()}
    kind = {"BlackwellNvf4OmmaDot64_v1": "atom", "HopperBF16WgmmaDot16_v1": "peel"}
    reads = [set() for _ in range(g.n)]
    for s, d in zip(g.src, g.dst):
        reads[d].add(base + s)
    for ix, d in g.input_reads:
        reads[d].add(g.input_ids[ix])
    assert g.n == len(table)
    assert [kind[names[g.prim[i]]] for i in range(g.n)] == [t for t, _ in table]
    assert list(g.width) == [32] * g.n
    assert reads == [set(r) for _, r in table]
    assert all(len(set(r)) == len(r) for _, r in table)
    assert [base + o for o in g.outputs] == outs


@pytest.mark.parametrize("k,tile", sorted(TILE_DIGESTS4))
def test_the_checked_tile4_templates_are_pinned(k, tile):
    assert program_digest(Program(pc4.templates(k, tile)["tile"])) == TILE_DIGESTS4[k, tile]


@pytest.mark.slow
@pytest.mark.parametrize("k,tile", sorted(DEPLOYED_TILE4))
def test_the_deployed_tile4_is_pinned(k, tile):
    """Catches a change to `pc4`'s deployed template at K = 8192 that the table check and the K = 128 pins don't see."""
    assert tile == pc4.TILE
    assert program_digest(Program(pc4.templates(k, tile)["tile"])) == DEPLOYED_TILE4[k, tile]
