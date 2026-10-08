"""`verity_flock.rec_algebra`: the inner verifier's algebraic checks as residuals over F, on the flat GEMM selftest's honest
session (the fetched fixture `hidden-outputs/flat/gemm-coordinate`: m = 25, k_log = 22, 8 claims). The reference's residuals
are zero on both reps and its intermediates are the Lean verifier's (`fixtures/inner_claims.json`, which
`tests/lean/InnerClaims.lean` dumps); the circuit's structure, verifier rows and witness give the reference's residuals,
honest and with a message changed, through `GfResiduals`'s reference and through the IR evaluator; and a changed message
moves a residual off zero exactly when the Lean verifier rejects the session, for the check that rejects it. And
`Gf128Mul_v1` lowers to the rows rec-lean's Karatsuba lemma is about (`Recursion/Karatsuba.lean`).

The set is a fetched fixture (`fixtures/artifacts.json`), which `suites.py` fetches before the suite runs, so a missing set
fails these tests rather than skipping them."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gf2k as G
from verity.ml.circuits.boolean.forms import Circuit
from verity_flock import ir_lower as IL
from verity_flock import rec_algebra as A

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SET = HERE / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"
INNER = json.loads((HERE / "fixtures" / "inner_claims.json").read_text())
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: the set was recorded under f5df3bbc's identity, which the unversioned name no longer is (test_lean_hidden_outputs.py)
STATEMENT = "verity/flock-circuit@f5df3bbc+seed-injection"
#: the fixture's unit reads two input groups, a form the soundness proof does not cover (verifier PROTOCOL.md §16.15), so
#: `verify` refuses every session of it once every check has passed: this verdict is a pass, as `agree.py --lean-refuses`
#: judges it
PASSED = "refused: a unit reading 2 input groups: a statement form the soundness proof does not cover"
FLIP = 1 << 17

#: a message element changed (by FLIP), and the residuals that move off zero when T is the inner verifier's own: 0 is the
#: zerocheck, 1 the lincheck, 2 + k claim k's ring switch, 10 and 11 Ligerito's final check (K's c0, c1), 12 the absorbed T
#: against the batched sum. Ligerito's last message (27) enters no check.
MUTATIONS = {
    ("ab", 9): [0],
    ("g1", 3): [0],
    ("qinf", 2): [1],
    ("zp", 5): [1, 2],
    ("rs", 2, 77): [4, 10, 11],
    ("msg", 4, 1, 0): [10, 11],
    ("ood", 2): [10, 11],
    ("yr", 3): [10, 11],
    ("msg", 26, 0, 1): [11],
    ("msg", 26, 1, 1): [10, 11],
    ("msg", 27, 0, 0): [],
}
#: where V*'s statement differs, T being the absorbed one: a changed ring switch moves its own residual and T's, not the final
#: check, which uses the absorbed T as the inner verifier uses its own
ABSORBED = {("rs", 2, 77): [4, 12]}


@dataclass
class Rep:
    shape: A.Shape
    data: bytes
    proof: A.Proof
    rounds: list
    extras: list
    comb: list
    dump: dict


def _claim(e: dict) -> A.Claim:
    return A.Claim(G.of_hex(e["value"]), tuple(map(G.of_hex, e["skip"])), tuple(map(G.of_hex, e["x"])))


@pytest.fixture(scope="module")
def session():
    assert (SET / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/hidden-outputs`")
    record = json.loads((SET / "records-cpu" / "honest.json").read_text())["record"]
    reps = []
    for r, d in enumerate(INNER["reps"]):
        sh = A.Shape(INNER["m"], INNER["k_log"], 2 + len(d["extras"]))
        data = (SET / "records-cpu" / f"honest.rep{r}.proof").read_bytes()
        reps.append(Rep(sh, data, A.decode(sh, data), A.rep_rounds(record, d["stream"]), [_claim(e) for e in d["extras"]],
                        [G.of_hex(x) for x in d["comb"]], d))
    return record, reps


def _value(p: A.Proof, k: tuple) -> int:
    if k[0] in ("ab", "c", "zp", "ood", "yr"):
        return getattr(p, k[0])[k[1]]
    if k[0] in ("g1", "ginf"):
        return p.zc[k[1]][k[0] == "ginf"]
    if k[0] in ("q1", "qinf"):
        return p.lc[k[1]][k[0] == "qinf"]
    if k[0] == "rs":
        return p.rs[k[1]][k[2]]
    if k[0] == "msg":
        return p.msgs[k[1]][k[2]][k[3]]
    return getattr(p, k[0])


def _mutated(rep: Rep, k: tuple) -> tuple[bytes, bytes, bytes]:
    """The rep's proof with message element ``k`` changed by FLIP, and the element's bytes before and after: every element
    of the session's proofs is 16 bytes found once in the proof."""
    old = G.to_bytes(_value(rep.proof, k))
    new = G.to_bytes(_value(rep.proof, k) ^ FLIP)
    assert rep.data.count(old) == 1, k
    return rep.data.replace(old, new), old, new


def _nonzero(xs) -> list[int]:
    return [i for i, x in enumerate(xs) if x]


#: SHA-512 of Gf128Mul's rows in the recursion's order (p0, p2, then the sums' product), each [256 + k, a's variables, a's
#: constant, b's variables, b's constant] over a = variables 0..127 and b = 128..255, as compact JSON
KARATSUBA_ROWS = ("e6b8a513843b112aefe5b265144cd01747a443ffb42eba34e37e640bd604c816"
                  "f82db680ff40c5d48e76304d5a7a5df406bb94309540a2eec5cb07f435333634")
#: SHA-512 of [rows, outputs] in an order-free form: the rows [a's variables, a's constant, b's variables, b's constant] sorted
#: (lists compared lexicographically), the k-th sorted row numbered 256 + k, and the 128 outputs [variables, constant]
KARATSUBA_CANONICAL = ("76a54ecc5f8f375228c8f1faf56353d46d3735c4abbcfe1041d3f5d09308e920"
                       "a7d2ff462046ba8e4e15c71bb6b51a10b3708b167c992950f43c0cbbaf7b2629")


def _rows_and_outputs(C: Circuit, out: list) -> tuple[list, list]:
    ands = [v for v in range(len(C.kind)) if C.kind[v] == "and"]
    return ([[v, sorted(C.data[v][0].s), C.data[v][0].c, sorted(C.data[v][1].s), C.data[v][1].c] for v in ands],
            [[sorted(o.s), o.c] for o in out])


def _canonical(rows: list, outs: list) -> list:
    order = sorted(rows, key=lambda r: r[1:])
    num = {r[0]: 256 + k for k, r in enumerate(order)} | {i: i for i in range(256)}
    return [[r[1:] for r in order], [[sorted(num[v] for v in o[0]), o[1]] for o in outs]]


def _sha512_json(x) -> str:
    return hashlib.sha512(json.dumps(x, separators=(",", ":")).encode()).hexdigest()


def test_gf128mul_lowers_to_the_rows_of_the_karatsuba_lemma():
    """The recursion (`gf2k.mul_forms`) gives rec-lean's row list, pinned on both sides by one digest; C-Flock's lowering of
    `Gf128Mul_v1` gives the same rows and outputs in the IR's canonical gate order, so it is pinned order-free."""
    C = Circuit()
    rows, outs = _rows_and_outputs(C, G.mul_forms(C, C.input("a", 128), C.input("b", 128)))
    assert len(rows) == 3 ** 7 and _sha512_json(rows) == KARATSUBA_ROWS
    C2 = Circuit()
    a, b = C2.input("a", 128), C2.input("b", 128)
    low = IL.lower(C2, bind(G.Gf128Mul), [[x] for x in a] + [[x] for x in b])
    rows2, outs2 = _rows_and_outputs(C2, [w[0] for w in low])
    assert _sha512_json(_canonical(rows, outs)) == _sha512_json(_canonical(rows2, outs2)) == KARATSUBA_CANONICAL


def test_the_dump_is_current():
    stale = [f for f, h in INNER["inputs"].items() if hashlib.sha256((REPO / f).read_bytes()).hexdigest() != h]
    assert not stale, f"changed since the dump was generated, rerun verity/ml/flock/tests/lean/generate_inner_claims.sh: {stale}"


def test_the_honest_reps_pass_every_check_at_the_lean_verifiers_values(session):
    _, reps = session
    h = G.of_hex
    for rep in reps:
        out, info = A.residuals(rep.shape, rep.proof, rep.rounds, rep.comb, rep.extras)
        assert len(out) == 5 + rep.shape.claims and _nonzero(out) == []
        zc, lc = rep.dump["zerocheck"], rep.dump["lincheck"]
        assert (info["z"], info["rho"], info["r"], info["v_c"]) == (h(zc["z"]), [h(x) for x in zc["rho"]],
                                                                    [h(x) for x in zc["r"]], h(zc["v_c"]))
        assert (rep.proof.fa, rep.proof.fb) == (h(zc["v_a"]), h(zc["v_b"]))
        assert (info["w"], info["r_prime"], info["r_in"]) == (h(lc["w"]), h(lc["r_prime"]), [h(x) for x in lc["r_in"]])
        assert info["T"] == h(rep.dump["T"]) and info["gamma"] == [h(x) for x in rep.dump["gamma"]]


def _ports(rep: Rep, seed: int) -> tuple[dict, A.Layout, list[int], list[int]]:
    """The structure, its layout, and its v and w ports on the honest rep, the running sums starting at a random value."""
    S, lay = A.structure(rep.shape)
    g = random.Random(seed)
    V, info = A.verifier_rows(rep.shape, rep.rounds, rep.extras, rep.comb)
    W = A.witness(rep.shape, rep.proof, rep.rounds, rep.extras, acc0=(g.getrandbits(128), g.getrandbits(128)), info=info)
    return S, lay, V, W


@pytest.fixture(scope="module")
def ports(session):
    return _ports(session[1][0], seed=0)


def test_the_structure_is_a_function_of_the_shape(session, ports):
    """Both reps' coins give one structure (`verifier_rows` checks it against `structure`), and its ports are laid out as the
    module states: the message elements by round and the running sums; then per residual its coefficients and constant, and
    the comb."""
    _, reps = session
    S, lay, V0, _ = ports
    V1, _ = A.verifier_rows(reps[1].shape, reps[1].rounds, reps[1].extras, reps[1].comb)
    assert len(V0) == len(V1) == S["nv"] == len(lay.v)
    assert S["nw"] == len(lay.w) and lay.w == [x for r in A.rounds(reps[0].shape) for x in r] + A.ACC_NAMES
    assert lay.v[-64:] == [("comb", l) for l in range(64)] and V0[-64:] == reps[0].comb
    assert len(S["res"]) == 5 + reps[0].shape.claims
    read = {t[1] for op in S["ops"] for t in op if t[0] == "w"} | {t[1] for r in S["res"] for t in r["add"] if t[0] == "w"}
    assert not {lay.index[("msg", 27, h, c)] for h in range(2) for c in range(2)} & read


def test_the_rounds_are_the_transcripts(session):
    """`rounds` names each rep's absorbed elements as its transcript's rounds carry them: per round that absorbs any, the free
    elements of the round's message (`rec_live.split_round`, what `c_k` commits) are the witness's values at the round's names,
    in order, T included where the prover absorbs it."""
    from verity_flock import rec_live as RL

    record, reps = session
    for rep in reps:
        S, lay = A.structure(rep.shape)
        V, info = A.verifier_rows(rep.shape, rep.rounds, rep.extras, rep.comb)
        val = dict(zip(lay.w, A.witness(rep.shape, rep.proof, rep.rounds, rep.extras, info=info)))
        [st] = [s for s in record["streams"] if s["stream"] == rep.dump["stream"]]
        frees = [RL.split_round(bytes.fromhex(rd["msg"]), k == 0).free for k, rd in enumerate(st["rounds"])]
        names = A.rounds(rep.shape)
        assert [[val[k] for k in nm] for nm in names] == [f for f in frees if f]


def test_the_running_sums_stand_for_the_consistency_sums(session, ports):
    """The final check's coefficients on the consistency sums (κ) are functions of the coins alone, not of the claims; the
    rep's running sums differ by the openings' share of the final check (`consistency`), at any start; and a running sum
    changed moves the final check's component it carries, alone."""
    _, reps = session
    rep = reps[0]
    S, lay, V, W = ports
    g = random.Random(2)
    other = [A.Claim(g.getrandbits(128), tuple(g.getrandbits(128) for _ in e.skip), tuple(g.getrandbits(128) for _ in e.x))
             for e in rep.extras]
    _, info = A.verifier_rows(rep.shape, rep.rounds, rep.extras, rep.comb)
    _, info2 = A.verifier_rows(rep.shape, rep.rounds, other, rep.comb)
    assert info["kappa"] == info2["kappa"]
    acc = [W[lay.index[k]] for k in A.ACC_NAMES]
    assert [acc[0] ^ acc[2], acc[1] ^ acc[3]] == A.consistency(rep.shape, rep.proof, info) != [0, 0]
    fin = 2 + rep.shape.claims
    for r in range(2):
        W2 = list(W)
        W2[lay.index[("acc1", r)]] ^= FLIP
        assert _nonzero(G.residuals(S, W2, V)) == [fin + r]


@pytest.mark.parametrize("k", [("zp", 5), ("rs", 2, 77), ("msg", 26, 0, 1)], ids=str)
def test_the_circuit_gives_the_references_residuals(session, ports, k):
    """With T the absorbed one, as V*'s witness carries it."""
    rep = session[1][0]
    S, lay, V, W = ports
    assert _nonzero(G.residuals(S, W, V)) == []
    data, _, _ = _mutated(rep, k)
    want, _ = A.residuals(rep.shape, A.decode(rep.shape, data), rep.rounds, rep.comb, rep.extras, t=W[lay.index[A.T_NAME]])
    W2 = list(W)
    W2[lay.index[k]] ^= FLIP
    assert G.residuals(S, W2, V) == want and _nonzero(want) == ABSORBED.get(k, MUTATIONS[k])


def _bits(xs) -> np.ndarray:
    return np.array([[(x >> i) & 1 for i in range(128)] for x in xs], dtype=np.uint8)


def _val(bs) -> int:
    return sum(int(b) << i for i, b in enumerate(bs))


def test_the_lincheck_through_the_ir_evaluator(ports):
    """The lincheck's residual alone (98 products, 1.5M gates) on the IR evaluator: zero on the honest rep, the reference's
    value on a changed z_partial."""
    S, lay, V, W = ports
    sub = dict(S, res=[S["res"][1]])
    W2 = list(W)
    W2[lay.index[("zp", 5)]] ^= FLIP
    out = evaluate_batch(bind(G.GfResiduals, S=sub), np.stack([_bits(W), _bits(W2)]), np.stack([_bits(V)] * 2))
    assert [_val(out[0, 0]), _val(out[1, 0])] == [0, G.residuals(sub, W2, V)[0]] and _val(out[1, 0]) != 0


@pytest.mark.slow
def test_every_check_through_the_ir_evaluator(session):
    """The algebra whole (2,704 products) on the IR evaluator, both reps: zero on the honest witness, and on a changed message
    the residuals the reference gives. It catches a fault in a residual the lincheck test does not build (the zerocheck's
    product of messages, the transposed ring-switch blocks, Ligerito's K-valued check and its running sums); about 20 s."""
    _, reps = session
    for rep in reps:
        S, lay, V, W = _ports(rep, seed=1)
        W2 = list(W)
        W2[lay.index[("rs", 2, 77)]] ^= FLIP
        out = evaluate_batch(bind(G.GfResiduals, S=S), np.stack([_bits(W), _bits(W2)]), np.stack([_bits(V)] * 2))
        assert [_val(x) for x in out[0]] == [0] * len(S["res"])
        assert [_val(x) for x in out[1]] == G.residuals(S, W2, V) and _nonzero(G.residuals(S, W2, V)) == ABSORBED[("rs", 2, 77)]


@pytest.mark.slow
def test_every_mutation_moves_the_residuals_the_table_names(session):
    """Every entry of MUTATIONS on the reference (each a decode and a run of the inner verifier's algebra, about 1 s): the
    table the Lean agreement test reads."""
    _, reps = session
    rep = reps[0]
    for k, want in MUTATIONS.items():
        data, _, _ = _mutated(rep, k)
        out, _ = A.residuals(rep.shape, A.decode(rep.shape, data), rep.rounds, rep.comb, rep.extras)
        assert _nonzero(out) == want, k


def _reason(i: int, claims: int) -> str:
    """The Lean verifier's reason for the first failing check, when residual ``i`` is the first nonzero one."""
    if i == 0:
        return "zerocheck: "
    if i == 1:
        return "lincheck: "
    if i < 2 + claims:
        return f"opening: ring-switch claim {i - 2} "
    return "ligerito: final check failed"


def _write_session(d: Path, record: dict, proofs: list[bytes]) -> None:
    d.mkdir()
    (d / "session.json").write_text(json.dumps(record))
    for r, b in enumerate(proofs):
        (d / f"circuit.rep{r}.bin").write_bytes(b)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_lean_verifier_rejects_exactly_the_mutations_with_a_nonzero_residual(session, tmp_path):
    """Each entry of MUTATIONS as a session (the proof's bytes and the round that retained them, with their digests), judged
    by the Lean verifier of record: exactly the sessions whose residuals are all zero (the honest one, and a changed last
    Ligerito message) pass every check, so their verdict is the form's refusal (`PASSED`), and it rejects each other one at
    the check of its first nonzero residual. It catches a
    reference that is zero where the inner verifier rejects, or that sees a check fail where the inner verifier sees none,
    which tests against the reference alone cannot; about 18 s a session, four at a time."""
    record, reps = session
    rep = reps[0]
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    cases = {"honest": []}
    _write_session(tmp_path / "honest", record, [r.data for r in reps])
    for n, (k, want) in enumerate(MUTATIONS.items()):
        data, old, new = _mutated(rep, k)
        rec = copy.deepcopy(record)
        [st] = [s for s in rec["streams"] if s["stream"] == "circuit/rep0"]
        [rd] = [r for r in st["rounds"] if r.get("msg") and old in bytes.fromhex(r["msg"])]
        msg = bytes.fromhex(rd["msg"])
        assert msg.count(old) == 1
        msg = msg.replace(old, new)
        rd["msg"], rd["msg_sha512"] = msg.hex(), hashlib.sha512(msg).hexdigest()
        [ps] = [p for p in rec["proof_sha512"] if p["stream"] == "circuit/rep0"]
        ps["sha512"] = hashlib.sha512(data).hexdigest()
        _write_session(tmp_path / f"m{n}", rec, [data, reps[1].data])
        cases[f"m{n}"] = want
    stage = SET / "stage"
    verdicts, names = {}, list(cases)
    for i in range(0, len(names), 4):
        procs = {c: subprocess.Popen([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(stage / "circuit.txt"),
                                      "--public", str(stage / "pub-4.bin"), "--session", str(tmp_path / c)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for c in names[i:i + 4]}
        for c, p in procs.items():
            out, _ = p.communicate()
            [v] = [json.loads(ln[8:]) for ln in out.splitlines() if ln.startswith("VERDICT ")]
            verdicts[c] = v
    for c, want in cases.items():
        assert not verdicts[c]["accepted"], (c, want)
        assert (verdicts[c]["why"] == PASSED) == (want == []), (c, want, verdicts[c]["why"])
        if want:
            assert verdicts[c]["why"].startswith(_reason(want[0], rep.shape.claims)), (c, want, verdicts[c]["why"])
