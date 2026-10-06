"""`verity_flock.rec_residuals`, V*'s algebra statement: ``InnerRepCheck`` (its M0 unit, with the leaf-summed products, and its
Definition) against `gf2k.residuals` on a small S; its ports on rec-algebra's fixture session (the messages by round, the
running sums, the verifier's v with the comb), holding on both reps and failing at a changed comb or running sum; and the
statement's files, every row port a registered read, under Lean's `statement`."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
from verity.primitives.commitments import hm96
from verity_catalog.definitions.boolean import gf2k as G
from verity_flock import rec_algebra as A
from verity_flock import rec_live as RL
from verity_flock import rec_residuals as RR

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "verifier" / "lean" / ".lake" / "build" / "bin" / "flock-verify"
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
    from verity.primitives.circuits.evaluation import evaluate_batch

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
        f"{SET} is not fetched: `uv run research data fetch-fixtures backends/flock/tests/fixtures/hidden-outputs`")
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
    `statement` under the reads file."""
    from verity.protocols.verification.sampled_proofs.one_stage import registered as RG

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
    st = RR.statement(S_SMALL, "rec-residuals/small", 2)
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
    p = subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(tmp_path / "circuit.txt"),
                        "--public", str(tmp_path / "pub.bin"),
                        "--partition", str(tmp_path / "partition.json"), "--program", str(tmp_path / "program.json"),
                        "--registered", str(tmp_path / "registered.json")], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["registered"] == {nm: {"value": e["value"], "root": e["root"]} for nm, e in reads.items()}
