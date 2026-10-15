"""`verity_flock.rec_outer`, V*'s outer statements: a level of inner openings staged as one statement of `RecOpen` instances
whose top nodes the prover registers under the firewall's salts (the verifier's entry rebuilt from their ``b || c``) and whose
running sums the prover registers as one chain. Staged here at the smallest shape (8 lanes, one sibling: a synthetic tree of
four plain leaves, its top two nodes) with three honest openings and one whose sibling is another node's. Composing the
unit's circuit takes about a minute, so that test is slow; it catches what `test_rec_open` can't: the statement's ports,
program, partition and registered rows (the verifier's ``coef`` and ``dirs``, the tops, the running sums and, with
`link_rows`, the opened rows), with no public input, as the Lean verifier's `statement` reads them (when it is built), and an
opening that does not climb to its top node found before anything is proved."""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import shutil
import subprocess
from pathlib import Path

import keyed
import pytest
from test_rec_open import _tree
from verity_flock import circuit as FC
from verity_flock import rec_live as RL
from verity_flock import rec_open as R
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR
from verity_flock import rec_vstar as V

BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"
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
    with pytest.raises(ValueError, match="firewall committed"):
        O.TopValue.build(PROGRAM, 0, {0: bad, 1: bad})


def test_a_value_the_verifier_registers_is_a_function_of_its_rows_alone():
    """`rec_residuals.verifier_value`: the same program, name and rows give the same registration (rows, salts, root), so the
    verifier's entry and the prover's table rows and salts agree without the verifier sending anything; another name, another
    program or one bit of a row gives another root, and no salt repeats."""
    qs, _ = _queries()
    coefs = _coefs(qs)
    rows = [coefs[(q.rep, q.level, q.j)] for q in qs]
    a, b = (RR.verifier_value(PROGRAM, O.COEF_VALUE, rows) for _ in range(2))
    assert (a.rows, a.salts, a.root) == (b.rows, b.salts, b.root) and a.entry()["value"] == O.COEF_VALUE
    assert len(set(a.salts)) == len(rows) and a.salts == tuple(RR.verifier_salts(PROGRAM, O.COEF_VALUE, len(rows)))
    flipped = [[rows[0][0] ^ 1, *rows[0][1:]], *rows[1:]]
    others = [RR.verifier_value(PROGRAM, O.DIRS_VALUE, rows), RR.verifier_value("another program", O.COEF_VALUE, rows),
              RR.verifier_value(PROGRAM, O.COEF_VALUE, flipped)]
    assert len({a.root, *(o.root for o in others)}) == 4
    assert others[2].salts == a.salts


@pytest.fixture(scope="module")
def staged(once):
    """Level 1 staged once on the machine (`once`, here and in `test_rec_vstar_checks`), with what `rec_outer.stage` said."""
    qs, layers = _queries(level=1)
    (level,) = O.levels(qs)
    coefs = _coefs(qs)
    tops = O.TopValue.build(PROGRAM, 1, {0: _top(layers[1], 2), 1: _top(layers[1], 3)})

    def make(d):
        s = O.stage(level, d / "level", O.Chain.build(PROGRAM, qs, coefs), coefs, tops, link_rows=True)
        (d / "stage.pickle").write_bytes(pickle.dumps(s))

    d = once("rec-outer-staged", make, lambda: keyed.key(code=[__name__]))
    return d / "level", pickle.loads((d / "stage.pickle").read_bytes()), tops, level, coefs


@pytest.mark.slow
def test_a_level_is_one_template_over_both_reps(staged):
    d, s, tops, level, coefs = staged
    assert s["template"] == "RecOpen_v4{LANES=8,H=1}" and s["instances"] == 4 and s["compressions"] == 4
    assert s["opened"] == 3 and s["not_opened"] == [(1, 1)] and s["summed"] == 4
    own = json.loads((d / "registered.json").read_text())["ports"]
    qs = level["queries"]
    for port, value, rows in (("coef", O.COEF_VALUE, [coefs[(q.rep, q.level, q.j)] for q in qs]),
                              ("dirs", O.DIRS_VALUE, [R.dirs_words(q.pos, 1) for q in qs])):
        mine = RR.verifier_value(s["program_sha512"], value, rows).entry()
        assert own[port] == {**mine, "word_bits": 16, "reads": [0, 1, 2, 3]} and own[port]["root"] == s[f"{port}_root"]
    assert own["out"]["reads"] == [0, 1, 3, 2] and own["out"]["root"] == s["registered_root"] == tops.own["root"]
    assert own["out"]["value"] == s["value"] == "rec-top-L1" and own["out"]["leaves"] == s["top_rows"] == 4
    assert own["row"]["reads"] == [0, 1, 2, 3] and own["row"]["root"] == s["rows_root"] != s["registered_root"]
    assert own["acc"]["reads"] == [0, 1, 2, 3] and own["acc_out"]["reads"] == [1, 2, 3, 4]
    assert own["acc"]["root"] == own["acc_out"]["root"] == s["acc_root"] and own["acc"]["leaves"] == 5
    head = json.loads((d / "pub.bin").read_bytes().split(b"\n", 1)[0])
    assert {k: v["value"] for k, v in head["registered"].items()} == {
        "row": "rec-rows-L1", "out": "rec-top-L1", "acc": O.ACC_VALUE, "acc_out": O.ACC_VALUE, "coef": O.COEF_VALUE,
        "dirs": O.DIRS_VALUE}
    assert head["registered"]["out"]["positions"] == [0, 1, 3, 2]
    assert head["registered"]["coef"]["positions"] == head["registered"]["dirs"]["positions"] == [0, 1, 2, 3]
    meta = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert list(meta["program_digests"]) == [s["template"]] and meta["program_sha512"] == s["program_sha512"]
    assert [p[0] for p in meta["ports"]] == ["row", "aux", "acc", "coef", "dirs", "out", "acc_out"]
    assert [p[1] for p in meta["ports"]][3:5] == [R.coef_words(8), R.DIRS_WORDS] == [128, 64]
    assert "public_inputs" not in meta and "leaves_public" not in meta and "public_inputs" not in head
    assert not (d / "public-inputs.bin").exists()
    assert meta["leaf_scheme"] == FC.LEAF_SCHEME
    with pytest.raises(ValueError, match="one template"):
        O.levels([*_queries()[0], V.Query(0, 0, 2, 0, 3, 1, b"", [0] * 8, None, [b"", b""])])


@pytest.mark.slow
def test_the_top_forgeries_move_the_provers_output_row_or_its_salt_alone(staged, tmp_path):
    """`rec_vstage.forge_top` on the staged level: ``top`` on the opening whose sibling is another node's (it climbs to that
    node, which becomes its output row), ``top-salt`` on an honest one. Only the prover's ``inst.bin`` moves, by that row or
    that salt; the verifier's ``pub.bin`` holds the ``b || c`` the firewall committed for each opening's top node, which the
    forged row (or salt) does not open."""
    from verity_flock import rec_vstage as S

    d, _s, tops, _level, _coefs_ = staged
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


@pytest.mark.slow
def test_a_forged_coef_or_dirs_row_does_not_open_the_verifiers_registration(staged, tmp_path):
    """``coef`` and ``dirs`` forged as `rec_vstage.stage` forges them (`rec_vstage.forge_row`, bit 0 of instance 0's word 0):
    one bit of the prover's row moves in ``inst.bin`` alone, under the verifier's public salt; ``pub.bin`` keeps the ``b || c``
    of the verifier's own row, which the forged row does not open."""
    from verity_flock import rec_vstage as S

    d, s, _tops, level, coefs = staged
    text = (d / "circuit.txt").read_text()
    honest, pub = (d / "inst.bin").read_bytes(), (d / "pub.bin").read_bytes()
    nl = pub.index(b"\n") + 1
    shared = json.loads(pub[:nl])["shared_rows"]
    ports = [p[0] for p in json.loads(text.split("\n", 2)[1][5:])["ports"]]
    q = level["queries"][0]
    for port, value, row in (("coef", O.COEF_VALUE, list(coefs[(q.rep, q.level, q.j)])),
                             ("dirs", O.DIRS_VALUE, R.dirs_words(q.pos, 1))):
        f = tmp_path / port
        shutil.copytree(d, f)
        S.forge_row(f / "inst.bin", text, port, 0, 0, row[0])
        forged = (f / "inst.bin").read_bytes()
        assert (f / "pub.bin").read_bytes() == pub
        row_at, salt_at = S._offsets(honest, text, port, 0)
        assert [k for k, (a, b) in enumerate(zip(honest, forged, strict=True)) if a != b] == [row_at]
        salt = RR.verifier_salts(s["program_sha512"], value, len(level["queries"]))[0]
        assert forged[salt_at:salt_at + V.SALT] == salt
        at = nl + 128 * sum(shared[p] for p in ports[:ports.index(port)])
        assert RL.commit_rows([row], [salt]) == [pub[at:at + 128]]
        assert RL.commit_rows([[row[0] ^ 1, *row[1:]]], [salt]) != [pub[at:at + 128]]


@pytest.mark.slow
def test_a_level_without_a_climb_reads_its_leaves_as_its_tops(tmp_path):
    """H = 0 (a level whose cap is its leaves, as m = 23's level 2): ``aux`` is one row the unit does not read, and each
    opening's ``out`` is its leaf, read against the registered top at its position. Three honest openings and one whose row
    is another leaf's."""
    rows, layers = _tree(1, 8)
    q = lambda rep, j, p, row: V.Query(rep, 0, j, p, 1, 1, layers[0][p], row, None, [])
    qs = [q(0, 0, 0, rows[0]), q(0, 1, 1, rows[1]), q(1, 0, 1, rows[1]), q(1, 1, 0, rows[1])]
    (level,) = O.levels(qs)
    coefs = _coefs(qs)
    shared = _top(layers[0], 5)
    tops = O.TopValue.build(PROGRAM, 0, {0: shared, 1: shared})
    assert [tops.position(x) for x in qs] == [0, 1, 1, 0]
    s = O.stage(level, tmp_path, O.Chain.build(PROGRAM, qs, coefs), coefs, tops)
    assert s["template"] == "RecOpen_v4{LANES=8,H=0}" and s["instances"] == 4 and s["compressions"] == 2
    assert s["opened"] == 3 and s["not_opened"] == [(1, 1)] and s["summed"] == 4
    meta = json.loads((tmp_path / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert [p[:2] for p in meta["ports"]][:5] == [["row", 64], ["aux", 64], ["acc", 64], ["coef", 128], ["dirs", 64]]


def test_a_level_of_salted_leaves_is_refused(tmp_path):
    qs, layers = _queries()
    qs = [V.Query(**{**qs[0].__dict__, "salt": bytes(V.SALT)}), *qs[1:]]
    (level,) = O.levels(qs)
    tops = O.TopValue.build(PROGRAM, 0, {0: _top(layers[1], 1), 1: _top(layers[1], 1)})
    with pytest.raises(ValueError, match="plain SHA-512 leaves"):
        O.stage(level, tmp_path, O.Chain.build(PROGRAM, qs, _coefs(qs)), _coefs(qs), tops)


def _verify(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                           "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"), "--program",
                           str(d / "program.json"), "--registered", str(d / "registered.json")],
                          capture_output=True, text=True, check=False)


def _statement(d: Path) -> dict:
    """The Lean verifier's `statement` on a staged level under the verifier's own files: its first line."""
    p = _verify(d)
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["partition_check"]["units"] == "derived"
    return line


@pytest.mark.slow
@pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")
def test_the_lean_verifier_reads_the_statement_under_the_verifiers_own_files(staged):
    d = staged[0]
    line = _statement(d)
    own = json.loads((d / "registered.json").read_text())["ports"]
    assert line["registered"] == {k: {"value": v["value"], "root": v["root"]} for k, v in own.items()}


@pytest.mark.slow
@pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")
def test_the_lean_verifier_reads_rs_level_under_its_own_name_as_the_opening(staged, tmp_path):
    """One of R's levels names the opening unit `rec-open-stacked/l<LANES>-h<H>` (V*'s `recOpenStackedName`), and the Lean
    verifier holds it to the opening's net under that name (`Flock.RecOpen.check`), as it holds a `rec-open/` unit. The
    staged level with its unit renamed (its public file pinning the renamed file) is read under the verifier's own files;
    with the unit's first AND row changed as well, it is refused, naming the row."""
    d, name = tmp_path / "s", "rec-open-stacked/l8-h1"
    shutil.copytree(staged[0], d)
    lines = (d / "circuit.txt").read_text().split("\n")
    at = next(k for k, line in enumerate(lines) if line.startswith("CIRCUIT unit ")) + 1
    fields = lines[at].split(" ")
    assert fields[1] == "rec-open/l8-h1", fields[:2]
    lines[at] = " ".join([fields[0], name, *fields[2:]])
    head, rest = (staged[0] / "pub.bin").read_bytes().split(b"\n", 1)
    head = json.loads(head)

    def write():
        text = "\n".join(lines)
        (d / "circuit.txt").write_text(text)
        head["circuit_sha512"] = FC.digest(text)
        head["units"]["classes"] = [FC.class_digest(text)] * len(head["units"]["classes"])
        (d / "pub.bin").write_bytes(json.dumps(head).encode() + b"\n" + rest)

    write()
    _statement(d)
    row = next(q for q, line in enumerate(lines[at + 1:]) if line not in (f"1 {q} 1 {q}", "0 0"))
    lines[at + 1 + row] = "1 0 1 1"
    write()
    p = _verify(d)
    assert p.returncode == 1 and f"not the verifier's {name} (row {row} differs)" in p.stdout, p.stdout + p.stderr


@pytest.mark.slow
@pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")
def test_the_lean_verifier_refuses_outputs_not_laid_as_vstars(staged, tmp_path):
    """`RecOpen.check` holds a `rec-open/` unit's outputs to V*'s layout (`Flock.RecOpen.outsOk`): `out` then `acc_out`,
    64 words each, one output row each, every copy returning words 0..127 in order. The staged level with one copy's
    `leaves_out` permuted passes the rows' and input leaves' checks and is refused there, before the net is built."""
    d = tmp_path / "s"
    shutil.copytree(staged[0], d)
    lines = (d / "circuit.txt").read_text().split("\n")
    meta = json.loads(lines[1][5:])
    meta["leaves_out"][0][0], meta["leaves_out"][0][1] = meta["leaves_out"][0][1], meta["leaves_out"][0][0]
    lines[1] = "META " + json.dumps(meta)
    text = "\n".join(lines)
    (d / "circuit.txt").write_text(text)
    head, rest = (staged[0] / "pub.bin").read_bytes().split(b"\n", 1)
    head = json.loads(head)
    head["circuit_sha512"] = FC.digest(text)
    head["units"]["classes"] = [FC.class_digest(text)] * len(head["units"]["classes"])
    (d / "pub.bin").write_bytes(json.dumps(head).encode() + b"\n" + rest)
    p = _verify(d)
    assert p.returncode == 1 and "outputs are not out then acc_out" in p.stdout, p.stdout + p.stderr


def _sparse(H: int, c: int, positions: list[int], seed: int):
    """A tree of plain leaves, `H` levels below a top of `2^c` nodes, known at `positions` and random elsewhere: the rows
    there, each one's siblings bottom up, and the top's nodes."""
    g = random.Random(seed)
    rows = {p: [g.getrandbits(128) for _ in range(8)] for p in set(positions)}
    nodes, sibs = {p: V.leaf(r) for p, r in rows.items()}, {p: [] for p in rows}
    node = lambda i: nodes.setdefault(i, g.randbytes(64))
    for h in range(H):
        for p in sorted(rows):
            sibs[p].append(node((p >> h) ^ 1))
        nodes = {i >> 1: hashlib.sha512(node(i & ~1) + node(i | 1)).digest() for i in sorted(nodes)}
    return rows, sibs, [node(i) for i in range(1 << c)]


@pytest.mark.slow
@pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")
@pytest.mark.parametrize("H, positions", [(0, [0, 1, 1, 0]), (17, [5, 1 << 16 | 3, 3 << 16 | 1 << 9, 1 << 17 | 6])])
def test_the_lean_verifier_reads_a_level_at_either_edge_of_the_climb(H, positions, tmp_path):
    """`RecOpen_v4` climbs 0 to 32 levels, and the Lean verifier takes a `rec-open/` unit only as its own net and layout at
    the unit's height (`Flock.RecOpen`). A level staged at H = 0 (``aux`` one row the unit does not read) and at 17 (the
    first height whose side bits are ``dirs``' word 1, set at two of its four positions) is read under the verifier's own
    files. Staging H = 17 takes about two minutes; no cheaper test compares the verifier's net with ``rec_open``'s past
    16 or at 0."""
    rows, sibs, top = _sparse(H, 1, positions, H)
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    qs = [V.Query(rep, 0, j, p, H + 1, 1, top[p >> H], rows[p], None, sibs[p]) for (rep, j), p in zip(keys, positions)]
    (level,) = O.levels(qs)
    coefs = _coefs(qs)
    shared = _top(top, 5)
    s = O.stage(level, tmp_path, O.Chain.build(PROGRAM, qs, coefs), coefs,
                O.TopValue.build(PROGRAM, 0, {0: shared, 1: shared}))
    assert s["template"] == f"RecOpen_v4{{LANES=8,H={H}}}" and s["opened"] == s["summed"] == 4
    _statement(tmp_path)
