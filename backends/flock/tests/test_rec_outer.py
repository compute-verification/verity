"""`verity_flock.rec_outer`, V*'s outer statements: a level of inner openings staged as one statement of `RecOpen` instances
whose top nodes the prover registers under the gateway's salts (the verifier's entry rebuilt from their ``b || c``) and whose
running sums the prover registers as one chain. Staged here at the smallest shape (8 lanes, one sibling: a synthetic tree of
four plain leaves, its top two nodes) with three honest openings and one whose sibling is another node's. Composing the
unit's circuit takes about a minute, so that test is slow; it catches what `test_rec_open` can't: the statement's ports,
program, partition, public inputs and registered rows (the tops, the running sums and, with `link_rows`, the opened rows) as
the Lean verifier's `statement` reads them (when it is built), and an opening that does not climb to its top node found
before anything is proved."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from test_rec_open import _tree
from verity_flock import circuit as FC
from verity_flock import rec_live as RL
from verity_flock import rec_open as R
from verity_flock import rec_outer as O
from verity_flock import rec_vstar as V

BIN = Path(__file__).resolve().parents[1] / "verifier" / "lean" / ".lake" / "build" / "bin" / "flock-verify"
PROGRAM = O.session_program(b"a test session")


def _queries(level: int = 0):
    rows, layers = _tree(2, 8)
    q = lambda rep, j, p, sib: V.Query(rep, level, j, p, 2, 1, layers[1][p >> 1], rows[p], None, [sib])
    return [q(0, 0, 1, layers[0][0]), q(0, 1, 2, layers[0][3]), q(1, 0, 3, layers[0][2]), q(1, 1, 0, layers[0][3])], layers


def _top(nodes: list[bytes], seed: int) -> V.Top:
    g = random.Random(seed)
    salts = [g.randbytes(V.SALT) for _ in nodes]
    return V.Top(tuple(nodes), tuple(salts), tuple(RL.commit_top(b"".join(nodes), salts)))


def _coefs(qs) -> dict:
    g = random.Random(3)
    kappa = [[[g.getrandbits(128) for _ in range(8)] for _ in range(2)] for _ in range(2)]
    ea = [g.getrandbits(128) for _ in range(2)]
    return {(q.rep, q.level, q.j): O.coefficients(kappa, ea, q.level, q.j) for q in qs}


def test_the_chain_runs_rep_major_and_each_row_adds_its_openings_terms():
    qs, _ = _queries()
    coefs = _coefs(qs)
    ch = O.Chain.build(PROGRAM, qs, coefs)
    assert O.chain_order(qs) == [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)] and ch.reg.leaves == 5
    assert ch.rep_ends(0) == (0, 2) and ch.rep_ends(1) == (2, 4) and ch.reg.rows[0] == (0,) * R.ACC_WORDS
    by = {(q.rep, q.level, q.j): q for q in qs}
    for k, p in ch.at.items():
        assert list(ch.reg.rows[p + 1]) == R.acc_ref(ch.reg.rows[p], coefs[k], R.words(by[k].row))
    assert ch.reg.entry()["value"] == O.ACC_VALUE and ch.reg.program == PROGRAM


def test_a_levels_tops_are_one_value_the_verifier_rebuilds_from_their_b_c():
    """Level 0's top, both reps' one tree, is one run of rows; a later level's are rep 0's then rep 1's. The verifier's entry
    from the ``b || c`` alone is the prover's registration's, each opening reads its top node's row, and a top whose nodes
    are not the ones its ``b || c`` commit is refused."""
    qs, layers = _queries()
    shared = _top(layers[1], 1)
    t0 = O.TopValue.build(PROGRAM, 0, {0: shared, 1: shared})
    assert t0.first == {0: 0, 1: 0} and t0.own == t0.reg.entry() and t0.own["value"] == "rec-top-L0" and t0.own["leaves"] == 2
    assert [t0.position(q) for q in qs] == [0, 1, 1, 0]
    assert [list(t0.reg.rows[t0.position(q)]) for q in qs] == [R.expected(q.cap_node) for q in qs]
    t1 = O.TopValue.build(PROGRAM, 1, {0: _top(layers[1], 2), 1: _top(layers[1], 3)})
    assert t1.first == {0: 0, 1: 2} and t1.own["leaves"] == 4 and t1.own == t1.reg.entry()
    assert [t1.position(q) for q in qs] == [0, 1, 3, 2]
    bad = V.Top(shared.nodes[::-1], shared.salts, shared.bcs)
    with pytest.raises(ValueError, match="gateway committed"):
        O.TopValue.build(PROGRAM, 0, {0: bad, 1: bad})


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    qs, layers = _queries(level=1)
    (level,) = O.levels(qs)
    coefs = _coefs(qs)
    tops = O.TopValue.build(PROGRAM, 1, {0: _top(layers[1], 2), 1: _top(layers[1], 3)})
    d = tmp_path_factory.mktemp("rec-outer")
    return d, O.stage(level, d, O.Chain.build(PROGRAM, qs, coefs), coefs, tops, link_rows=True), tops


@pytest.mark.slow
def test_a_level_is_one_template_over_both_reps(staged):
    d, s, tops = staged
    assert s["template"] == "RecOpen_v3{LANES=8,H=1}" and s["instances"] == 4 and s["compressions"] == 4
    assert s["opened"] == 3 and s["not_opened"] == [(1, 1)] and s["summed"] == 4
    own = json.loads((d / "registered.json").read_text())["ports"]
    assert own["out"]["reads"] == [0, 1, 3, 2] and own["out"]["root"] == s["registered_root"] == tops.own["root"]
    assert own["out"]["value"] == s["value"] == "rec-top-L1" and own["out"]["leaves"] == s["top_rows"] == 4
    assert own["row"]["reads"] == [0, 1, 2, 3] and own["row"]["root"] == s["rows_root"] != s["registered_root"]
    assert own["acc"]["reads"] == [0, 1, 2, 3] and own["acc_out"]["reads"] == [1, 2, 3, 4]
    assert own["acc"]["root"] == own["acc_out"]["root"] == s["acc_root"] and own["acc"]["leaves"] == 5
    head = json.loads((d / "pub.bin").read_bytes().split(b"\n", 1)[0])
    assert {k: v["value"] for k, v in head["registered"].items()} == {"row": "rec-rows-L1", "out": "rec-top-L1",
                                                                     "acc": O.ACC_VALUE, "acc_out": O.ACC_VALUE}
    assert head["registered"]["out"]["positions"] == [0, 1, 3, 2]
    meta = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert list(meta["program_digests"]) == [s["template"]] and meta["program_sha512"] == s["program_sha512"]
    assert [p[0] for p in meta["ports"]] == ["row", "aux", "acc", "out", "acc_out"]
    assert [(p["port"], p["words"]) for p in meta["public_inputs"]] == [("coef", R.coef_words(8)), ("dirs", R.DIRS_WORDS)]
    assert meta["leaf_scheme"] == FC.LEAF_SCHEME
    with pytest.raises(ValueError, match="one template"):
        O.levels([*_queries()[0], V.Query(0, 0, 2, 0, 3, 1, b"", [0] * 8, None, [b"", b""])])


@pytest.mark.slow
def test_the_top_forgeries_move_the_provers_output_row_or_its_salt_alone(staged, tmp_path):
    """`rec_vstage.forge_top` on the staged level: ``top`` on the opening whose sibling is another node's (it climbs to that
    node, which becomes its output row), ``top-salt`` on an honest one. Only the prover's ``inst.bin`` moves, by that row or
    that salt; the verifier's ``pub.bin`` holds the ``b || c`` the gateway committed for each opening's top node, which the
    forged row (or salt) does not open."""
    from verity_flock import rec_vstage as S

    d, _s, tops = staged
    (level,) = O.levels(_queries(level=1)[0])
    text = (d / "circuit.txt").read_text()
    honest, pub = (d / "inst.bin").read_bytes(), (d / "pub.bin").read_bytes()
    nl = pub.index(b"\n") + 1
    shared = json.loads(pub[:nl])["shared_rows"]
    ports = [p[0] for p in json.loads(text.split("\n", 2)[1][5:])["ports"]]
    out_at = nl + 128 * sum(shared[p] for p in ports[:ports.index("out")])
    bc = lambda i: pub[out_at + 128 * i:out_at + 128 * (i + 1)]
    for i, q in enumerate(level["queries"]):
        at = tops.position(q)
        assert [bc(i)] == RL.commit_rows([list(tops.reg.rows[at])], [tops.reg.salts[at]])
    for kind, i in (("top", 3), ("top-salt", 0)):
        f = tmp_path / kind
        shutil.copytree(d, f)
        rec = S.forge_top(f, level, tops, kind, i)
        forged = (f / "inst.bin").read_bytes()
        assert (f / "pub.bin").read_bytes() == pub
        diff = [k for k, (a, b) in enumerate(zip(honest, forged, strict=True)) if a != b]
        row_at, salt_at = S._offsets(honest, text, "out", i)
        salt = tops.reg.salts[rec["top_row"]]
        if kind == "top":
            q = level["queries"][i]
            assert row_at <= diff[0] and diff[-1] < row_at + 2 * R.OUT_WORDS
            new = [int.from_bytes(forged[row_at + 2 * w:row_at + 2 * w + 2], "little") for w in range(R.OUT_WORDS)]
            row, aux = R.pack(q.row, q.sibs)
            climbed = R.open_ref(8, 1, row, aux, [0] * R.ACC_WORDS, [0] * R.coef_words(8), R.dirs_words(q.pos, 1))[0]
            assert climbed == new == R.expected(bytes.fromhex(rec["node"])) != R.expected(q.cap_node)
            assert RL.commit_rows([new], [salt]) != [bc(i)]
        else:
            assert salt_at <= diff[0] and diff[-1] < salt_at + V.SALT
            assert forged[salt_at:salt_at + V.SALT] != salt
            assert RL.commit_rows([list(tops.reg.rows[rec["top_row"]])], [forged[salt_at:salt_at + V.SALT]]) != [bc(i)]
    with pytest.raises(ValueError, match="committed top node"):
        S.forge_top(tmp_path / "top-salt", level, tops, "top", 0)


def test_a_level_of_salted_leaves_is_refused(tmp_path):
    qs, layers = _queries()
    qs = [V.Query(**{**qs[0].__dict__, "salt": bytes(V.SALT)}), *qs[1:]]
    (level,) = O.levels(qs)
    tops = O.TopValue.build(PROGRAM, 0, {0: _top(layers[1], 1), 1: _top(layers[1], 1)})
    with pytest.raises(ValueError, match="plain SHA-512 leaves"):
        O.stage(level, tmp_path, O.Chain.build(PROGRAM, qs, _coefs(qs)), _coefs(qs), tops)


@pytest.mark.slow
@pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")
def test_the_lean_verifier_reads_the_statement_under_the_verifiers_own_files(staged):
    d, _, _ = staged
    p = subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                        "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
                        "--registered", str(d / "registered.json"), "--public-inputs", str(d / "public-inputs.bin")],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["partition_check"]["units"] == "derived"
    own = json.loads((d / "registered.json").read_text())["ports"]
    assert line["registered"] == {k: {"value": v["value"], "root": v["root"]} for k, v in own.items()}
