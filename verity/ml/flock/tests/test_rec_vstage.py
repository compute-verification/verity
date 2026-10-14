"""`verity_flock.rec_vstage`, V*'s statements over an inner session, on rec-algebra's fixture session replayed through the rec
proxy (each round's salted tops, ``c_k`` and coins for the verifier; its tops, row and salts for the prover). The
verifier's side from the public record alone (the table's shape, the fold points, every round's value rebuilt from its two
``c_k``, the openings' coefficients) against the Lean verifier's values (`fixtures/inner_claims.json`); the prover's running
sums giving the algebra exactly the consistency sums of the openings, so both reps hold; and each forgery (a message, a running
sum, the comb) failing the relation that reads it. Staging the circuits is the pod's (``pod/85-rec-reprice.sh STEP=vstage``);
staging with one salts file registers one chain; the slow test
patches a small statement's ``inst.bin`` as the forged message does. ``VERITY_FLOCK_INNER_FOLD=1`` also runs `InnerFold.lean`
interpreted (about 15 minutes) and holds its comb and extra claims to InnerClaims' dump."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from verity.ml.circuits.ops.boolean import gf2k as G
from verity_flock import rec_algebra as A
from verity_flock import rec_live as RL
from verity_flock import rec_open as R
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR
from verity_flock import rec_vstage as S
from verity_flock import rec_vstar as V

HERE = Path(__file__).resolve().parent
SET = HERE / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"
INNER = json.loads((HERE / "fixtures" / "inner_claims.json").read_text())
LEAN = HERE.parents[2] / "core" / "service" / "flock"


def _fold() -> dict:
    """``InnerFold.lean``'s output on the fixture, from InnerClaims' dump of the same values (the opt-in test holds them equal)."""
    return {"m": INNER["m"], "k_log": INNER["k_log"], "extras": INNER["reps"][0]["extras"],
            "reps": [{"comb": d["comb"]} for d in INNER["reps"]]}


@pytest.fixture(scope="module")
def inner(tmp_path_factory) -> Path:
    """The fixture's honest session replayed through the firewall (`rec_live.Proxy`, seeded salts), whose coin server returns
    the recorded coins, as `rec_live.Proxy.write` lays it out. The session ran on seeded coins: its Hello is replayed without
    the coin scheme."""
    assert (SET / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/hidden-outputs`")
    rec = json.loads((SET / "records-cpu" / "honest.json").read_text())["record"]
    g, pending = random.Random(5), bytearray()

    def rand(n: int) -> bytes:
        out = bytes(pending[:n])
        del pending[:n]
        assert len(out) == n, "a coin the record does not hold"
        return out

    coins = lambda hexes: b"".join(RL.f128_bytes(int(c, 16)) for c in hexes)
    schedule = RL.schedule_for({"dense_m": 25, "k_log": 22, "nbl": 3, "regions": 3})
    proxy = RL.Proxy(RL.CoinServer(rand), lambda: g.randbytes(192), schedule=schedule)
    send = lambda *req: RL.decode_resp(proxy.handle(RL.encode_req(*req)))
    hello = json.loads(rec["hello"])
    del hello["coins"]
    lk = rec["link"]
    pending += coins(x for p in lk["points"] for x in p)
    assert send(RL.REQ_HELLO, json.dumps(hello))[0] == RL.RESP_OK
    assert send(RL.REQ_COMMIT, bytes.fromhex(lk["root_f"]), [(t, bytes.fromhex(v)) for t, v in lk["root_b"].items()], [])[0] == RL.RESP_COINS
    assert send(RL.REQ_LINK, coins(lk["y"]))[0] == RL.RESP_OK
    for st in rec["streams"]:
        assert send(RL.REQ_OPEN, st["stream"], st["parent"] or "", st["label"].encode(), st["at"])[0] == RL.RESP_OK
        for rd in st["rounds"]:
            pending += coins(rd["coins"])
            assert send(RL.REQ_ROUND, st["stream"], bytes.fromhex(rd["msg"]), rd["n"]) == (RL.RESP_COINS, [int(c, 16) for c in rd["coins"]])
    for r in (0, 1):
        assert send(RL.REQ_PROOF, "circuit", r, (SET / "records-cpu" / f"honest.rep{r}.proof").read_bytes())[0] == RL.RESP_OK
    assert send(RL.REQ_FINISH)[0] == RL.RESP_VERDICT
    d = tmp_path_factory.mktemp("inner")
    proxy.write(d)
    return d


@pytest.fixture(scope="module")
def side(inner):
    """Both sides' values: the verifier's (`rec_vstage.verifier`), the openings and their coefficients, the chain, the rounds'
    registrations and the verifier's entries."""
    public, private, proofs = V.load(inner)
    vf = S.verifier(public, _fold())
    qs = V.queries(public, private, proofs)
    coefs = S.coefficients(vf, qs)
    program = O.session_program((inner / "public.json").read_bytes())
    mrs = S.message_rounds(public, private, vf.sh)
    regs, own = S.messages(program, mrs)
    return vf, qs, coefs, O.Chain.build(program, qs, coefs), mrs, regs, own, proofs


def _ports(vf, chain, mrs, rep: int, acc: O.Chain | None = None) -> list[list[int]]:
    """A rep's row ports as the algebra reads them: its rounds' rows, then the chain's rows either side of its openings."""
    a0, a1 = (acc or chain).rep_ends(rep)
    rows = (acc or chain).reg.rows
    return [mr["rows"][rep] for mr in mrs] + [list(rows[a0]), list(rows[a1])]


def test_the_record_holds_every_top_salted_and_the_openings_climb_to_them(inner):
    """Replayed through the firewall, the coin server's record holds no tree top in the clear (level 0's ``root_B``, each
    later level's cap: one ``b || c`` a node), V*'s opened tops are the proofs' caps, and every opening climbs to its node."""
    public, private, proofs = V.load(inner)
    text = (inner / "public.json").read_text()
    for r, b in proofs.items():
        caps = [lv.cap for lv in V.decode(b)["levels"]]
        assert [list(t.nodes) for t in V.tops(public, private, f"circuit/rep{r}")] == caps
        assert not any(n.hex() in text for c in caps for n in c)
    assert [len(bcs) for bcs in public["link"]["roots"].values()] == [len(V.decode(proofs[0])["levels"][0].cap)]
    assert all(V.opens(q) for q in V.queries(public, private, proofs))


def test_the_shape_and_the_fold_points_are_the_verifiers(inner):
    """The table's shape from the circuit's META and the instance count, as Lean's setup derives it, and each rep's fold
    points from its coins alone: the Lean verifier's (its zerocheck's z and ρ, its lincheck's point)."""
    meta, n = S.meta(SET / "stage" / "circuit.txt"), S.instances(SET / "stage" / "pub-4.bin")
    sh = S.shape(meta, n)
    assert (sh.m, sh.k_log) == (INNER["m"], INNER["k_log"])
    pts = S.points(json.loads((inner / "public.json").read_text()), meta, n)
    rec = json.loads((SET / "records-cpu" / "honest.json").read_text())["record"]
    assert pts["link"] == rec["link"]["points"] and len(pts["reps"]) == 2
    for p, d in zip(pts["reps"], INNER["reps"], strict=True):
        assert p["rho_in"] == d["zerocheck"]["rho"][:sh.k_log - 6] and p["ts"][::-1] == d["lincheck"]["r_in"]
        assert p["lw"] == [G.hex128(x) for x in A.skip_weights(G.of_hex(d["zerocheck"]["z"]))]
    with pytest.raises(ValueError, match="shape"):
        S.verifier(json.loads((inner / "public.json").read_text()), _fold(), {**pts, "m": pts["m"] + 1})


def test_the_verifier_rebuilds_each_rounds_value_from_its_two_c_k(inner, side):
    """A port a round that absorbs elements, its two rows (rep 0's, rep 1's) under the salts the prover committed in the inner
    session; the verifier's entry from the ``c_k`` alone is the registration's, and rows that are not the ``c_k``'s are
    refused."""
    vf, _qs, _c, _ch, mrs, regs, own, _p = side
    assert list(regs) == [nm for nm, _ in RR.spec(vf.sh)["ports"][:-2]] == [f"m{i}" for i in range(len(A.rounds(vf.sh)))]
    assert all(own[nm] == regs[nm].entry() for nm in regs)
    bad = [dict(mr) for mr in mrs]
    bad[3] = {**bad[3], "rows": [[bad[3]["rows"][0][0] ^ 1, *bad[3]["rows"][0][1:]], bad[3]["rows"][1]]}
    with pytest.raises(ValueError, match="round 3"):
        S.messages("ab" * 64, bad)


def test_the_running_sums_are_the_openings_and_both_reps_hold(side):
    """Every opening's position is the verifier's, each rep's two chain rows differ by exactly the rep's consistency sums
    (`rec_algebra.consistency` at the verifier's κ), and every residual of both reps is zero on the rounds' rows, the chain's
    rows and the verifier's v."""
    vf, qs, _c, chain, mrs, _r, _o, proofs = side
    assert len(qs) == chain.reg.leaves - 1 and chain.rep_ends(0)[1] == chain.rep_ends(1)[0]
    S_ = RR.spec(vf.sh)
    for rep in range(2):
        p = A.decode(vf.sh, proofs[rep])
        a0, a1 = chain.rep_ends(rep)
        cs = A.consistency(vf.sh, p, vf.reps[rep].info)
        assert [x ^ y for x, y in zip(chain.sums(a0), chain.sums(a1))] == cs != [0, 0]
        assert RR.residual_words(S_, _ports(vf, chain, mrs, rep), vf.reps[rep].v) == [0] * RR.OUT_WORDS


def test_each_forgery_fails_the_relation_that_reads_it(side):
    """The forged message's row is not its ``c_k``'s and moves the algebra; the forged running sum breaks the two openings
    either side of it and both reps' final checks; the forged comb moves rep 0's lincheck alone."""
    vf, qs, coefs, chain, mrs, _r, _o, _p = side
    S_ = RR.spec(vf.sh)
    claims = vf.sh.claims
    nz = lambda out: [i for i in range(len(S_["gf"]["res"])) if any(out[8 * i:8 * i + 8])]
    rows = _ports(vf, chain, mrs, 0)
    rows[0] = [rows[0][0] ^ 1, *rows[0][1:]]
    assert RL.commit_rows([rows[0]], [mrs[0]["salts"][0]]) != [mrs[0]["commits"][0]]
    assert nz(RR.residual_words(S_, rows, vf.reps[0].v)) != []
    p_last = chain.rep_ends(0)[1] - 1
    forged = S.forge_chain(chain, p_last + 1)
    assert forged.reg.salts == chain.reg.salts and forged.reg.root != chain.reg.root
    by = {(q.rep, q.level, q.j): q for q in qs}
    broken = [p for k, p in sorted(forged.at.items(), key=lambda kv: kv[1])
              if list(forged.reg.rows[p + 1]) != R.acc_ref(forged.reg.rows[p], coefs[k], R.words(by[k].row))]
    assert broken == [p_last, p_last + 1]
    assert nz(RR.residual_words(S_, _ports(vf, chain, mrs, 0, forged), vf.reps[0].v)) == [2 + claims]
    assert nz(RR.residual_words(S_, _ports(vf, chain, mrs, 1, forged), vf.reps[1].v)) == [2 + claims]
    v = list(vf.reps[0].v)
    v[-64] ^= 1
    assert nz(RR.residual_words(S_, _ports(vf, chain, mrs, 0), v)) == [1]


def test_stagings_with_one_salts_file_read_one_chain(inner, tmp_path):
    """The pod stages each statement in its own process: the running sums' salts, drawn once and kept in the salts file, give
    every staging the same chain, and a file of another length is refused."""
    from verity.ml.commitments import hm96

    p = tmp_path / "acc-salts.bin"
    a = S.stage(inner, _fold(), tmp_path / "a", levels=[], algebra=False, salts=p)
    b = S.stage(inner, _fold(), tmp_path / "b", levels=[], algebra=False, salts=p)
    assert a["acc_root"] == b["acc_root"] and p.stat().st_size == a["chain_rows"] * hm96.SHA512.salt_bytes
    with pytest.raises(ValueError, match="salts"):
        S.acc_salts(p, a["chain_rows"] + 1)


def test_the_shared_rows_list_the_algebras_row_ports_and_rs_tops_and_refuse_two_registrations(tmp_path):
    """`shared_rows` (`--rec-rows`, which `RecVStar.checkShared` reads by value): the algebra's row ports but `v` and `out`, and
    each of R's levels' `out` under its value `rec-top-stacked-L<l>`. Two parts registering one port differently, two entries
    of one value at two roots, and an R level whose `out` is not one of R's tops are refused."""
    def e(value, root="aa", reads=(0,)):
        return {"value": value, "root": root, "leaves": 4, "words": 64, "word_bits": 16, "domain": "dd", "reads": list(reads)}

    def staged(name, ports):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "registered.json").write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": ports}))
        return tmp_path / name

    p0 = staged("p0", {"m0": e("rec-c0"), "acc0": e(O.ACC_VALUE, "bb"), "v": e("rec-v"), "out": e("rec-zero")})
    p1 = staged("p1", {"m0": e("rec-c0"), "acc1": e(O.ACC_VALUE, "bb", (1,)), "v": e("rec-v", "cc")})
    top = staged("L0", {"out": e("rec-top-stacked-L0", "ee"), "acc": e(O.ACC_VALUE, "bb", (5,))})
    staged("open", {"claim": e(O.CLAIM_VALUE, "ff")})
    assert S.shared_rows([p0, p1], [top])["ports"] == {"m0": e("rec-c0"), "acc0": e(O.ACC_VALUE, "bb"),
                                                       "acc1": e(O.ACC_VALUE, "bb", (1,)),
                                                       "claim": e(O.CLAIM_VALUE, "ff"),
                                                       "rec-top-stacked-L0": e("rec-top-stacked-L0", "ee")}
    with pytest.raises(ValueError, match="its port m0 is not the other statements' m0"):
        S.shared_rows([p0, staged("p2", {"m0": e("rec-c0", "ab")})])
    with pytest.raises(ValueError, match="register rec-acc twice"):
        S.shared_rows([p0, staged("p3", {"acc1": e(O.ACC_VALUE, "ab", (1,))})])
    with pytest.raises(ValueError, match="its out reads rec-top-L0"):
        S.shared_rows([p0], [staged("L0x", {"out": e("rec-top-L0")})])


@pytest.mark.slow
def test_the_forged_message_changes_one_bit_of_the_provers_row_alone(tmp_path):
    """`forge_row` on a two-instance statement's ``inst.bin``: one bit of the named row moves, its header, its other rows, its
    salts and every ``b || c`` stay, and a word the layout does not put there is refused."""
    from test_rec_residuals import S_SMALL, _holding, _rows
    from verity.ml.commitments import hm96
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    g = random.Random(13)
    (w0, v0), (w1, v1) = _holding(g), _holding(g)
    rows0, rows1 = _rows(g, w0), _rows(g, w1)
    regs = {nm: (RG.Registered.commit("ab" * 64, f"rec-c{i}", [rows0[i], rows1[i]], [hm96.SHA512.fresh_salt() for _ in range(2)]), [0, 1])
            for i, (nm, _) in enumerate(S_SMALL["ports"])}
    st = RR.statement(S_SMALL, "rec-residuals/small", 2)
    RR.write(st, regs, [v0, v1], tmp_path, evaluate=False)
    honest = (tmp_path / "inst.bin").read_bytes()
    S.forge_row(tmp_path / "inst.bin", st.text, "m1", 1, 5, rows1[1][5])
    forged = (tmp_path / "inst.bin").read_bytes()
    diff = [i for i, (a, b) in enumerate(zip(honest, forged, strict=True)) if a != b]
    nl = honest.index(b"\n") + 1
    words = [p[1] for p in json.loads(st.text.split("\n", 2)[1][5:])["ports"]]
    assert diff == [nl + 2 * (2 * words[0] + words[1] + 5)] and honest[diff[0]] ^ forged[diff[0]] == 1
    with pytest.raises(AssertionError, match="layout"):
        S.forge_row(tmp_path / "inst.bin", st.text, "m1", 1, 5, rows1[1][5])


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_FLOCK_INNER_FOLD"), reason="interprets the Lean verifier's setup on the fixture "
                    "(about 15 minutes): set VERITY_FLOCK_INNER_FOLD=1")
def test_inner_fold_gives_inner_claims_comb_and_extras(inner, tmp_path):
    """`InnerFold.lean` at `points`' POINTS, from the statement and the coins alone, gives the comb of each rep and the extra
    claims InnerClaims dumped while running the whole inner verifier."""
    if shutil.which("lake") is None:
        pytest.skip("lake is not installed")
    meta, n = S.meta(SET / "stage" / "circuit.txt"), S.instances(SET / "stage" / "pub-4.bin")
    (tmp_path / "points.json").write_text(json.dumps(S.points(json.loads((inner / "public.json").read_text()), meta, n)))
    p = subprocess.run(["lake", "env", "lean", "--run", str(HERE / "lean" / "InnerFold.lean"),
                        "verity/flock-circuit@f5df3bbc+seed-injection", str(SET / "stage" / "circuit.txt"),
                        str(SET / "stage" / "pub-4.bin"), str(tmp_path / "points.json")],
                       cwd=LEAN, capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout)
    assert (out["m"], out["k_log"]) == (INNER["m"], INNER["k_log"])
    assert out["extras"] == INNER["reps"][0]["extras"] == INNER["reps"][1]["extras"]
    assert [r["comb"] for r in out["reps"]] == [d["comb"] for d in INNER["reps"]]


def test_inner_folds_plain_leaf_scheme_is_the_staged_one():
    """`InnerFold.lean --plain-leaves` pins the META ``leaf_scheme`` a recursive session's inner statement is staged with."""
    from verity_flock import circuit as FC

    src = (HERE / "lean" / "InnerFold.lean").read_text()
    body = src[src.index("def leafSchemePlain : Json :="):].split("\n\n", 1)[0]
    got = dict(re.findall(r'\("(\w+)", \.str "([^"]*)"\)', body))
    got.update({k: int(v) for k, v in re.findall(r'\("(\w+)", toJson \((\d+) : Nat\)\)', body)})
    assert got == FC.LEAF_SCHEME_PLAIN
    assert 'merkleLeaf := "flock-leaf/sha512-unsalted"' in src and FC.LEAF_SCHEME_PLAIN["id"] == "flock-leaf/sha512-unsalted"
