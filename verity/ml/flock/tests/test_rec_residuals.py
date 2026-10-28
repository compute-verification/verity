"""`verity_flock.rec_residuals`, V*'s algebra statement: ``InnerRepCheck`` (its M0 unit, with the leaf-summed products, and its
Definition) against `gf2k.residuals` on a small S; its ports on rec-algebra's fixture session (the messages by round, the
running sums, the verifier's v with the comb), holding on both reps and failing at a changed comb or running sum; and the
statement's files, every row port a registered read, under Lean's `statement`."""
from __future__ import annotations

import dataclasses
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
from verity.ml.commitments import hm96
from verity.ml.circuits.ops.boolean import gf2k as G
from verity_flock import rec_algebra as A
from verity_flock import rec_live as RL
from verity_flock import rec_residuals as RR

HERE = Path(__file__).resolve().parent
BIN = HERE.parents[2] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"
SET = HERE / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"
INNER = json.loads((HERE / "fixtures" / "inner_claims.json").read_text())

#: three residuals over 130 + 3 elements: three products (an element, a transposed block, a sum) and an added term; no product;
#: one product
SMALL = {"nw": 130, "nv": 3, "ops": [[["w", 0]], [["wt", 1, 3], ["v", 0]], [["v", 1]], [["w", 129], ["w", 0]]],
         "res": [{"mul": [[0, 1], [2, 3], [1, 3]], "add": [["v", 2]]}, {"mul": [], "add": [["w", 129]]},
                 {"mul": [[2, 0]], "add": []}]}
#: w as three row ports: an element alone, the transposed block, an element alone
S_SMALL = {"gf": SMALL, "ports": [["m0", 1], ["m1", 128], ["m2", 1]]}


def _rows(g: random.Random, w: list[int], noise: bool = False) -> list[list[int]]:
    """w's elements as S_SMALL's row ports' words, the words after each port's elements random with ``noise``."""
    out, at = [], 0
    for _, n in S_SMALL["ports"]:
        r = RR.words(w[at:at + n], RR.port_words(n))
        if noise:
            r[8 * n:] = [g.getrandbits(16) for _ in r[8 * n:]]
        out.append(r)
        at += n
    return out


def _cases(g: random.Random, k: int) -> list[tuple[list[list[int]], list[int], list[int]]]:
    """``k`` random port sets (rows with noise after their elements, v) and their reference outputs."""
    out = []
    for _ in range(k):
        w = [g.getrandbits(128) for _ in range(SMALL["nw"])]
        v = [g.getrandbits(128) for _ in range(SMALL["nv"])]
        rows = _rows(g, w, noise=True)
        want = RR.words(G.residuals(SMALL, w, v), RR.OUT_WORDS)
        assert RR.residual_words(S_SMALL, rows, v) == want
        out.append((rows, v, want))
    return out


@pytest.mark.parametrize("op", [[["w", 130]], [["v", 3]], [["wt", 3, 0]], [["wt", 1, 128]]],
                         ids=["w-past-end", "v-past-end", "wt-past-end", "wt-bit-128"])
def test_the_unit_refuses_an_out_of_range_s(op):
    """A term past its port's end would slice short and the sums' `zip` would truncate it; `residual_forms` refuses the S."""
    bad = {**SMALL, "ops": SMALL["ops"] + [op], "res": SMALL["res"] + [{"mul": [[len(SMALL["ops"]), 0]], "add": []}]}
    C = RR.Circuit()
    W = [b for i in range(SMALL["nw"]) for b in C.input(f"w{i}", 128)]
    V = [b for i in range(SMALL["nv"]) for b in C.input(f"v{i}", 128)]
    with pytest.raises(ValueError, match="out of range"):
        RR.residual_forms(C, bad, W, V)


def test_the_leaf_operands_are_karatsubas_ands_in_order():
    """`_leaf_operands` on each side, zipped, is the sequence of operand pairs `gf2k.karatsuba` ANDs, so the unit's gates and
    their order are the ones the product's recursion lays down."""
    g = random.Random(3)
    form = lambda: RR.L(frozenset(g.sample(range(400), g.randrange(4))), g.getrandbits(1))
    for n in (1, 2, 8, 128):
        a, b = [form() for _ in range(n)], [form() for _ in range(n)]
        rec = RR._Pairs()
        G.karatsuba(rec, a, b)
        key = lambda pairs: [((x.s, x.c), (y.s, y.c)) for x, y in pairs]
        assert key(zip(RR._leaf_operands(a), RR._leaf_operands(b))) == key(rec.pairs)
        assert len(rec.pairs) == 3 ** (n.bit_length() - 1)


@pytest.mark.slow
def test_the_unit_sums_each_residuals_products_leaf_by_leaf():
    """The unit (through its M0 layout) equals the reference on random ports, the words after each port's elements unread: one
    Karatsuba (2,187 ANDs) a product, and one copy row a leaf for the residual of several products."""
    low = RR.lowering(S_SMALL, "rec-residuals/small")
    assert sum(1 for k in low.circuit.kind if k == "and") == 4 * RR.LEAVES + RR.LEAVES
    cases = _cases(random.Random(7), 3)
    flat = [sum(rows, []) + RR.words(v, RR.v_words(S_SMALL)) for rows, v, _ in cases]
    got, _cut, ok = low.evaluate(np.array(flat, np.uint64))
    assert int(ok) == 0b111
    assert [got[i].astype(np.uint16).tolist() for i in range(3)] == [want for _, _, want in cases]
    assert any(cases[0][2][:8]) and not any(cases[0][2][24:])


@pytest.mark.slow
def test_the_definition_computes_the_reference():
    """``InnerRepCheck{S_SMALL}`` evaluated bit-sliced from its descriptor, on the unit's cases."""
    from verity.ml.circuits.evaluation import evaluate_batch

    cases = _cases(random.Random(7), 3)
    bits = lambda ws: np.array([[(w >> j) & 1 for j in range(16)] for w in ws], dtype=np.uint8)
    ports = [np.array([bits(rows[i]) for rows, _, _ in cases]) for i in range(3)]
    ports.append(np.array([bits(RR.words(v, RR.v_words(S_SMALL))) for _, v, _ in cases]))
    res = np.asarray(evaluate_batch(RR.definition(S_SMALL), *ports)).reshape(3, RR.OUT_WORDS, 16)
    word = lambda w: int(sum(int(b) << j for j, b in enumerate(w)))
    assert [[word(w) for w in res[t]] for t in range(3)] == [want for _, _, want in cases]


@pytest.fixture(scope="module")
def reps():
    assert (SET / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/hidden-outputs`")
    record = json.loads((SET / "records-cpu" / "honest.json").read_text())["record"]
    out = []
    for r, d in enumerate(INNER["reps"]):
        sh = A.Shape(INNER["m"], INNER["k_log"], 2 + len(d["extras"]))
        p = A.decode(sh, (SET / "records-cpu" / f"honest.rep{r}.proof").read_bytes())
        rounds = A.rep_rounds(record, d["stream"])
        extras = [A.Claim(G.of_hex(e["value"]), tuple(map(G.of_hex, e["skip"])), tuple(map(G.of_hex, e["x"]))) for e in d["extras"]]
        v, info = A.verifier_rows(sh, rounds, extras, [G.of_hex(x) for x in d["comb"]])
        [st] = [s for s in record["streams"] if s["stream"] == d["stream"]]
        frees = [f for f in (RL.split_round(bytes.fromhex(rd["msg"]), k == 0).free for k, rd in enumerate(st["rounds"])) if f]
        acc0 = [0x1234 + r, 0x5678 + r]
        cs = A.consistency(sh, p, info)
        rows = [RL.message_rows(f)[0] for f in frees] + [RR.words(acc0, 64), RR.words([acc0[0] ^ cs[0], acc0[1] ^ cs[1]], 64)]
        out.append((sh, rows, v))
    return out


def test_the_ports_are_the_rounds_and_the_running_sums(reps):
    """`spec`'s row ports are the rep's absorbing rounds (`rec_live.message_rows`' rows, as ``c_k`` commits them) and the
    running sums, and S.gf is the algebra's structure."""
    sh, rows, _ = reps[0]
    S = RR.spec(sh)
    assert S["gf"] == A.structure(sh)[0]
    assert [nm for nm, _ in S["ports"]] == [f"m{i}" for i in range(len(A.rounds(sh)))] + ["acc0", "acc1"]
    assert [n for _, n in S["ports"]] == [len(r) for r in A.rounds(sh)] + [2, 2]
    assert [len(r) for r in rows] == [RR.port_words(n) for _, n in S["ports"]]


def test_a_real_rep_holds_and_fails_at_a_changed_comb_or_running_sum(reps):
    """Both reps of the fixture's honest session: every residual zero on the rounds' rows, the running sums and the verifier's
    v; the comb changed in v moves the lincheck alone, and the last running sum changed moves Ligerito's final check alone."""
    for sh, rows, v in reps:
        S = RR.spec(sh)
        assert RR.residual_words(S, rows, v) == [0] * RR.OUT_WORDS
    sh, rows, v = reps[0]
    S = RR.spec(sh)
    nz = lambda out: [i for i in range(len(S["gf"]["res"])) if any(out[8 * i:8 * i + 8])]
    v2 = list(v)
    v2[-64] ^= 1
    assert nz(RR.residual_words(S, rows, v2)) == [1]
    rows2 = [list(r) for r in rows]
    rows2[-1][0] ^= 1
    assert nz(RR.residual_words(S, rows2, v)) == [2 + sh.claims]


def _holo(sh, rows, v, h):
    """A rep as a holo shape's: h's row inserted at `rec_algebra.h_round`, and the public comb in v the fold less h, so that
    comb + h is the fixture's fold."""
    hs = dataclasses.replace(sh, holo=True)
    i = A.h_round(hs)
    return hs, rows[:i] + [RR.words(h, RR.port_words(64))] + rows[i:], v[:-64] + [c ^ x for c, x in zip(v[-64:], h, strict=True)]


def test_a_holo_rep_reads_h_as_a_row_and_refuses_a_wrong_one(reps):
    """With hidden unit circuits the prover sends h, the hidden comb's fold, as its own round after z_partial (row port
    ``m<i>``, `rec_algebra.h_round`, the row `rec_sparse.RecSparse` reads too), and L4′ checks Σ (comb[l] + h[l]) z_partial[l]:
    both fixture reps hold with the comb split into a public part and h, and a wrong h moves the lincheck alone."""
    g = random.Random(4)
    for sh, rows, v in reps:
        h = [g.getrandbits(128) for _ in range(64)]
        hs, rows_h, v_h = _holo(sh, rows, v, h)
        S = RR.spec(hs)
        assert S["ports"][A.h_round(hs)] == [f"m{A.h_round(hs)}", 64]
        assert RR.residual_words(S, rows_h, v_h) == [0] * RR.OUT_WORDS
        assert RR.residual_words(RR.spec(sh), rows, v_h) != [0] * RR.OUT_WORDS
        nz = lambda out: [i for i in range(len(S["gf"]["res"])) if any(out[8 * i:8 * i + 8])]
        bad = [list(r) for r in rows_h]
        bad[A.h_round(hs)][0] ^= 1
        assert nz(RR.residual_words(S, bad, v_h)) == [1]


@pytest.mark.parametrize("shape", [(25, 22, 8), (22, 10, 2), (35, 24, 8)], ids=str)
def test_h_adds_no_product_to_the_algebra(shape):
    """A holo shape's S has the same products as the plain one, so InnerRepCheck's ANDs (2,187 a product) are unchanged: h is
    a w term XORed into the 64 comb operands (``[v comb_l, w h_l]``), and w gains its 64 elements."""
    sh = A.Shape(*shape)
    S, H = RR.spec(sh)["gf"], RR.spec(dataclasses.replace(sh, holo=True))["gf"]
    assert [r["mul"] for r in H["res"]] == [r["mul"] for r in S["res"]] and H["nv"] == S["nv"] and H["nw"] == S["nw"] + 64
    combs = [op for op in H["ops"] if len(op) == 2 and op[0][0] == "v" and op[0][1] >= H["nv"] - 64]
    assert len(combs) == 64 and all(op[1][0] == "w" for op in combs)
    if shape == (25, 22, 8):
        assert RR.LEAVES * sum(len(r["mul"]) for r in H["res"]) == 2704 * 2187 == 5_913_648


def _holo_s(S, p, c0):
    """Lean's `Flock.RecAlgebra.holoS p c0`: S with h's 64 elements inserted into w at p, and comb slot c0 + l's operand
    gaining w[p + l]."""
    shift = lambda i: i if i < p else i + 64

    def term(t):
        if t[0] == "w":
            return [["w", shift(t[1])]]
        if t[0] == "wt":
            return [["wt", shift(t[1]), t[2]]]
        return [t, ["w", p + t[1] - c0]] if c0 <= t[1] < c0 + 64 else [t]

    flat = lambda ts: [x for t in ts for x in term(t)]
    return {"nw": S["nw"] + 64, "nv": S["nv"], "ops": [flat(op) for op in S["ops"]],
            "res": [{"mul": r["mul"], "add": flat(r["add"])} for r in S["res"]]}


@pytest.mark.parametrize("shape", [(25, 22, 8), (22, 10, 2), (35, 24, 8)], ids=str)
def test_the_holo_structure_is_holoS_of_the_plain_one(shape):
    """L4′'s Lean (`lincheck_of_residuals_holo`, `VBridge/Algebra/Holo.lean`) takes a holo shape's S to be `holoS p c0` of
    the plain one, p being h's first w slot and c0 the comb's first v slot, with every ring switch's 128 w slots on one
    side of p (`HoloOk`)."""
    sh = A.Shape(*shape)
    (S, _), (H, lay) = A.structure(sh), A.structure(dataclasses.replace(sh, holo=True))
    p = lay.index[("h", 0)]
    assert [lay.index[("h", l)] for l in range(64)] == list(range(p, p + 64))
    assert p == 128 + 2 * (sh.m - 6) + 2 + 2 * (sh.k_log - 6) + 64
    assert H == _holo_s(S, p, S["nv"] - 64)
    wts = [t for op in S["ops"] for t in op if t[0] == "wt"] + [t for r in S["res"] for t in r["add"] if t[0] == "wt"]
    assert wts and all(p <= t[1] or t[1] + 128 <= p for t in wts)


def test_the_parts_are_the_residuals_within_the_coin_schedule(reps):
    """`parts` cuts the algebra into statements of at most MAX_REGIONS regions and MAX_SLOTS sha512x3 slots that each read
    their residuals' row ports and v elements: each part's residuals are the whole's on the fixture's reps (zero on the
    honest ones, the changed comb moving the part of the lincheck alone). At K = 4096's m = 35 the whole has 113 regions (111
    row ports, v and out), so two parts; without the slot bound the first would take 171 slots, which put its block at
    circuit.K_MAX (run r20261006-093343-89f7)."""
    for sh, rows, v in reps:
        S = RR.spec(sh)
        whole = RR.residual_words(S, rows, v)
        by = dict(zip([nm for nm, _ in S["ports"]], rows))
        ps = RR.parts(S)
        assert sorted(i for p in ps for i in p.res) == list(range(len(S["gf"]["res"])))
        for p in ps:
            assert RR.regions(p.S["ports"]) <= RR.MAX_REGIONS and len(p.res) <= RR.OUT_WORDS // RR.F_WORDS
            assert RR.slots(p.S["ports"], p.S["gf"]["nv"]) <= RR.MAX_SLOTS
            got = RR.residual_words(p.S, [by[nm] for nm, _ in p.S["ports"]], [v[i] for i in p.v])
            assert got[:8 * len(p.res)] == [x for i in p.res for x in whole[8 * i:8 * i + 8]] == [0] * (8 * len(p.res))
    sh, rows, v = reps[0]
    S = RR.spec(sh)
    by = dict(zip([nm for nm, _ in S["ports"]], rows))
    v2 = list(v)
    v2[-64] ^= 1
    moved = [p.res for p in RR.parts(S) if any(RR.residual_words(p.S, [by[nm] for nm, _ in p.S["ports"]], [v2[i] for i in p.v]))]
    assert len(moved) == 1 and 1 in moved[0]
    big = RR.spec(A.Shape(35, 24, 8))
    assert RR.regions(big["ports"]) == 113 and len(RR.parts(big)) == 2
    assert all(RR.slots(p.S["ports"], p.S["gf"]["nv"]) <= RR.MAX_SLOTS for p in RR.parts(big))
    assert [RR.slots(p.S["ports"], p.S["gf"]["nv"]) for p in RR.parts(big, max_slots=10 ** 9)] == [171, 40]


def test_an_algebra_of_more_residuals_than_one_statement_holds_is_cut_into_parts():
    """A whole algebra may have more than ``OUT_WORDS / 8`` residuals (a universal unit's inner session has 17, run
    r20261009-065336-108b); `parts` cuts it into statements that each hold at most that many, and a statement alone still may
    not."""
    many = dict(SMALL, res=[{"mul": [], "add": [["w", 129]]}] * 17)
    S = {"gf": many, "ports": S_SMALL["ports"]}
    with pytest.raises(ValueError, match="17 residuals"):
        RR._check(S)
    assert [p.res for p in RR.parts(S)] == [list(range(16)), [16]]


@pytest.mark.parametrize("salt_len", [192, 0], ids=["hm96-leaves", "plain-leaves"])
def test_the_universal_units_inner_algebra_is_two_parts_within_the_caps(salt_len):
    """At the universal unit's inner shape (m = 26, k_log = 23, 12 claims) the algebra's 17 residuals cut into two parts, of
    156 and 159 sha512x3 slots and 53 and 46 regions, whichever leaves its Merkle scheme has; its last residual alone reads
    130 slots, which a MAX_SLOTS of 128 refused."""
    S = RR.spec(A.Shape(26, 23, 12, salt_len=salt_len))
    ps = RR.parts(S)
    shape = lambda p: (p.res, RR.slots(p.S["ports"], p.S["gf"]["nv"]), RR.regions(p.S["ports"]))
    assert [shape(p) for p in ps] == [(list(range(13)), 156, 53), (list(range(13, 17)), 159, 46)]
    for p in ps:
        _, slots, regions = shape(p)
        assert slots <= RR.MAX_SLOTS and regions <= RR.MAX_REGIONS and len(p.res) <= RR.OUT_WORDS // RR.F_WORDS
    assert [shape(p)[1] for p in RR.parts({"gf": {**S["gf"], "res": S["gf"]["res"][16:]}, "ports": S["ports"]})] == [130]
    with pytest.raises(ValueError, match="residual 16 alone reads more than 64 regions or 128 sha512x3 slots"):
        RR.parts(S, max_slots=128, max_lone_slots=128)


def test_a_residual_wider_than_max_slots_stands_alone_within_max_lone_slots():
    """llama32-1b/b1/i256's 2^29 bucket samples inner sessions of 16 claims (Gumbel's 7 regions, note
    proofs/20261010T2112Z-finding-c3-claims-per-shape), whose batched-T residual alone reads 172 sha512x3 slots: it is a part
    of its own, and composes at k_log 26, under circuit.K_MAX (each part's net built in run r20261010-212155-f5ab). Shapes
    whose residuals each fit MAX_SLOTS cut as they did under MAX_SLOTS alone; a uniform 172 would have moved (34, 26, 8)'s
    first part to 172 slots and a 2^27 block."""
    shape = lambda p: (len(p.res), RR.slots(p.S["ports"], p.S["gf"]["nv"]), RR.regions(p.S["ports"]))
    S = RR.spec(A.Shape(35, 27, 16))
    ps = RR.parts(S)
    assert [shape(p) for p in ps] == [(11, 140, 64), (9, 115, 62), (1, 172, 19)] and ps[2].res == [20]
    assert RR.MAX_SLOTS < 172 <= RR.MAX_LONE_SLOTS
    with pytest.raises(ValueError, match="residual 20 alone reads more than 64 regions or 160 sha512x3 slots"):
        RR.parts(S, max_lone_slots=RR.MAX_SLOTS)
    S8 = RR.spec(A.Shape(34, 26, 8))
    assert [shape(p) for p in RR.parts(S8)] == [shape(p) for p in RR.parts(S8, max_lone_slots=0)] == [(10, 129, 61), (3, 123, 63)]


@pytest.mark.slow
def test_the_widest_lone_residual_max_lone_slots_admits_is_twenty_claims():
    """The batched T alone reads 215 slots at 20 claims, a part that composes at k_log 27 (= circuit.K_MAX) beside a 2^23 unit
    (run r20261010-223054-2894), and 226 at 21 claims, just past MAX_LONE_SLOTS, which refuses it. Slow for the two specs alone
    (about 10 s); the quick test above pins the 16 claims a pinned shape needs."""
    ps = RR.parts(RR.spec(A.Shape(35, 27, 20)))
    assert [(len(p.res), RR.slots(p.S["ports"], p.S["gf"]["nv"]), RR.regions(p.S["ports"])) for p in ps][-1] == (1, 215, 23)
    assert 215 <= RR.MAX_LONE_SLOTS < 226
    with pytest.raises(ValueError, match="residual 25 alone reads more than 64 regions or 223 sha512x3 slots"):
        RR.parts(RR.spec(A.Shape(35, 27, 21)))


def _holding(g: random.Random) -> tuple[list[int], list[int]]:
    """Ports on which SMALL's residuals are zero: the added terms set to the residuals they cancel."""
    w = [g.getrandbits(128) for _ in range(SMALL["nw"])]
    v = [g.getrandbits(128) for _ in range(SMALL["nv"])]
    w[129], v[1] = 0, 0
    v[2] ^= G.residuals(SMALL, w, v)[0]
    assert G.residuals(SMALL, w, v) == [0, 0, 0]
    return w, v


@pytest.mark.slow
def test_the_statement_reads_every_row_and_v_from_a_registration(tmp_path):
    """Two instances, one holding and one whose first residual is not zero (v's added constant changed). Each row port reads
    row i of a two-row registration in instance i, the verifier's own entry built from the rows' ``b || c`` alone
    (`commits_entry`), as V* builds a round's from its two ``c_k``; v is row i of the value the verifier registers itself
    (`verifier_value`, a function of the two v's alone), so the statement has no public input; and, when Lean is built, its
    `statement` under the reads file. Under a name of V*'s parts (`rec-residuals/…`) the toy S is refused: such a unit must be
    the verifier's own part net (PROTOCOL.md §16.10)."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    g = random.Random(11)
    (w0, v0), (w1, v1) = _holding(g), _holding(g)
    v1[2] ^= 1
    program = "ab" * 64
    rows0, rows1 = _rows(g, w0), _rows(g, w1)
    regs, own = {}, {}
    for i, (nm, _) in enumerate(S_SMALL["ports"]):
        salts = [hm96.SHA512.fresh_salt() for _ in range(2)]
        regs[nm] = (RG.Registered.commit(program, f"rec-c{i}", [rows0[i], rows1[i]], salts), [0, 1])
        bcs = RL.commit_rows([rows0[i], rows1[i]], salts)
        own[nm] = RR.commits_entry(program, f"rec-c{i}", RR.port_words(S_SMALL["ports"][i][1]), bcs)
        assert own[nm] == regs[nm][0].entry()
    st = RR.statement(S_SMALL, "toy-residuals/small", 2)
    s = RR.write(st, regs, [v0, v1], tmp_path, own=own)
    assert s["holds"] == [True, False] and s["nonzero_residuals"] == [[], [0]] and s["unit_agrees"] == [True, True]
    meta = json.loads((tmp_path / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert [p[0] for p in meta["ports"]] == ["m0", "m1", "m2", "v", "out"]
    assert [p[1] for p in meta["ports"]][3] == RR.v_words(S_SMALL) == 64
    assert meta["per_vu"]["sha512x3"] == RR.slots(S_SMALL["ports"], SMALL["nv"])
    assert "public_inputs" not in meta and not (tmp_path / "public-inputs.bin").exists()
    reads = json.loads((tmp_path / "registered.json").read_text())["ports"]
    assert sorted(reads) == ["m0", "m1", "m2", "out", "v"] and all(reads[nm]["reads"] == [0, 1] for nm in reads)
    assert reads["out"]["root"] == s["zero_root"] and all(reads[nm]["root"] == own[nm]["root"] for nm in own)
    vreg = RR.verifier_value(st.prog, RR.V_VALUE, [RR.words(v, RR.v_words(S_SMALL)) for v in (v0, v1)])
    assert reads["v"] == {**vreg.entry(), "word_bits": 16, "reads": [0, 1]}
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")

    def statement(circuit):
        return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(circuit),
                               "--public", str(tmp_path / "pub.bin"),
                               "--partition", str(tmp_path / "partition.json"), "--program", str(tmp_path / "program.json"),
                               "--registered", str(tmp_path / "registered.json")], capture_output=True, text=True, check=False)

    p = statement(tmp_path / "circuit.txt")
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["registered"] == {nm: {"value": e["value"], "root": e["root"]} for nm, e in reads.items()}
    text = (tmp_path / "circuit.txt").read_text()
    head = text.index("\n", text.index("CIRCUIT unit ")) + 1
    end = text.index("\n", head)
    fields = text[head:end].split(" ")
    assert fields[1] == "toy-residuals/small"
    (tmp_path / "renamed.txt").write_text(text[:head] + " ".join([fields[0], "rec-residuals/small", *fields[2:]]) + text[end:])
    p = statement(tmp_path / "renamed.txt")
    why = "unit rec-residuals/small is not rec-residuals/m<m>-k<k_log>-c<claims>-p<j>"
    assert p.returncode == 1 and why in p.stdout, p.stdout + p.stderr


def _with_holo(text: str, holo: bool) -> str:
    """``text`` with META's `RR.HOLO_KEY` as `RR.statement` writes it for ``holo`` (present only when true)."""
    head, meta, rest = text.split("\n", 2)
    m = {k: v for k, v in json.loads(meta[5:]).items() if k != RR.HOLO_KEY}
    if holo:
        m[RR.HOLO_KEY] = True
    return f"{head}\nMETA {json.dumps(m, sort_keys=True, separators=(',', ':'))}\n{rest}"


@pytest.mark.slow
def test_the_statements_holo_selects_the_one_net_the_verifier_holds_a_part_to(tmp_path):
    """At Shape(22, 10, 2), whose one part reads h when holo: the verifier holds the unit to the net META's `HOLO_KEY`
    selects, so a holo statement on the plain part and a plain statement on the holo part are refused by name, and each
    statement that says its part's shape passes the residuals check (stopping at the partition object it is not given).
    Composing a part takes about two minutes; no cheaper test reaches `RecResiduals.check` with a real part's unit."""
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    name = "rec-residuals/m22-k10-c2-p0"
    (tmp_path / "pub.bin").write_bytes(b"")
    texts = {holo: RR.statement(RR.parts(RR.spec(A.Shape(22, 10, 2, holo=holo)))[0].S, name, 2, holo=holo).text
             for holo in (False, True)}
    assert RR.HOLO_KEY not in json.loads(texts[False].split("\n", 2)[1][5:])
    assert json.loads(texts[True].split("\n", 2)[1][5:])[RR.HOLO_KEY] is True and _with_holo(texts[True], True) == texts[True]

    def statement(text):
        (tmp_path / "circuit.txt").write_text(text)
        p = subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(tmp_path / "circuit.txt"),
                            "--public", str(tmp_path / "pub.bin")], capture_output=True, text=True, check=False)
        return p.returncode, p.stdout + p.stderr

    bits = {False: 244736, True: 252928}
    for part_holo in (False, True):
        for says in (False, True):
            rc, out = statement(_with_holo(texts[part_holo], says))
            if says == part_holo:
                assert rc == 1 and "the verifier's partition object is required" in out, out
            else:
                kind = "holo" if says else "plain"
                why = (f"unit {name} has {bits[part_holo]} input and 2048 output bits, not its {kind} part's {bits[says]} and 2048 "
                       f"(META {RR.HOLO_KEY}: {str(says).lower()})")
                assert rc == 1 and why in out, out
