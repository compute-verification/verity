"""Registered reads (PROTOCOL.md §16.10, `protocols/sampled_proofs/README.md` "Registered values"): a statement whose port reads
rows of a registered value (RoPE's `cs`, rows of the program's `cs_table`, committed once at registration) opens every
table row against the verifier's own registered root, at the position the verifier's own program reads, with the units
derived from that program, or the Lean verifier refuses it. The statements are staged here over a hidden-output RoPE d64
circuit (§16.13, the default `verity/flock-circuit`) composed for the `RegisteredRoPE` program and its `Q_template_instance`
v0 partition; the reads come from `reads` on that program (`statement`: setup, no proof; `verify`: setup of each table,
with no proofs). Upstream's proved session over the public-output statement (`FIXTURE`, vectors.json's registered-reads
set) is held to the same reads. Composing the 56 MB circuit takes about 30 s and each call sets it up (about 8 s), so the
module is slow; it catches what the Python reference (`protocols/sampled_proofs/tests/test_registered.py`) can't: the
executable verifier's own refusals, on both statements' setup paths."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.commitments import hm96
from verity.ir.codec import canonical_json, encode_program
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import coll_concat
from verity.ir.types import Array
from verity_flock import circuit as FC
from verity_sampled_proofs.one_stage import partition as P, registered as G

LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
VERIFIER = LEAN.parent
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
FIXTURE = "art:bf70d8eb"
STATEMENT = "verity/flock-circuit"
UPSTREAM_STATEMENT = "verity/flock-circuit@967b8d06"
VALUE = "cs_table"
#: the program's: the row of `cs_table` each of its 8 RoPEHead Calls reads
POSITIONS = (11, 5, 3, 1, 0, 9, 8, 4)
TABLE_ROWS = 12

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def _program() -> tuple[Program, dict]:
    pytest.importorskip("verity_numerical")
    from verity_numerical.bench import templates as TM

    template = TM.rope_head(64).definition()
    (_, xt), (_, ct) = template.signature()[0]
    ret = template.signature()[1]

    def sig(S):
        return (("x", Array(len(POSITIONS), xt)), (VALUE, Array(TABLE_ROWS, ct))), Array(len(POSITIONS), ret)

    def body(B, S, x, cs):
        return coll_concat(sig(S)[1], [B.call(template, x[i], cs[p]) for i, p in enumerate(POSITIONS)])

    return Program(bind(CompositeDefinition("RegisteredRoPE", 0, (), sig, body, register=False))), P.template_instance_query(template)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The hidden-output circuit bound to the program and its partition, the registered cs table, the verifier's own reads
    file from its program, and the statement's bindings."""
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    d = tmp_path_factory.mktemp("registered")
    program, query = _program()
    part = P.partition(program, query)
    s = InputSet.open(generate(TM.rope_head(64), len(POSITIONS), 20260928, d / "set")["path"])
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit, partition=P.digest(part), program=P.program_sha512(program),
                      statics=FC.set_statics(s))
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    (d / "partition.json").write_text(json.dumps(part))
    reads = G.reads(program, query, {VALUE: 64})
    assert reads == {"cs": {"value": VALUE, "rows": list(POSITIONS)}}
    rng = np.random.default_rng(20261001)
    reg = G.Registered.commit(P.program_sha512(program), VALUE, rng.integers(0, 1 << 16, (TABLE_ROWS, 64)).tolist())
    own = G.reads_file(program, query, {VALUE: 64}, {VALUE: reg.root.hex()})
    (d / "own.json").write_text(json.dumps(own))
    return SimpleNamespace(dir=d, text=text, reg=reg, own=own, program=program, x=rng.integers(0, 1 << 16, (8, 64)),
                           bound=["--partition", str(d / "partition.json"), "--program", str(d / "program.json")])


def _stage(s, name: str, registered=True, salts=None, entry=None, extra=None) -> Path:
    positions, refs = G.table(POSITIONS)
    rows, reg_salts, e = s.reg.statement(positions)
    sub = json.loads(s.text.split("\n", 2)[1][5:]).get("subcircuit", "rope-head/d64/neox-bf16")
    inputs = SimpleNamespace(name="rope-registered", subcircuit=SimpleNamespace(id=sub))
    reg = {"cs": entry or e, **(extra or {})} if registered else None
    out = s.dir / f"{name}.bin"
    fresh = [hm96.SHA512.fresh_salt() for _ in range(8)]
    FC.write(s.text, inputs, 0, 8, None, out, None, list(range(8)), share_rows=True, outs={"out": np.zeros((8, 64), np.uint16)},
             tables={"x": s.x, "cs": np.asarray(rows)}, refs={"x": list(range(8)), "cs": refs},
             salts={"x": fresh, "cs": salts or reg_salts, "out": [hm96.SHA512.fresh_salt() for _ in range(8)]}, registered=reg)
    return out


def _statement(s, public: Path, own: Path | None, bound: list[str] | None = None) -> tuple[int, str]:
    args = [str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(s.dir / "circuit.txt"), "--public", str(public)]
    p = subprocess.run(args + (s.bound if bound is None else bound) + (["--registered", str(own)] if own else []),
                       capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def _own(s, name: str, edit) -> Path:
    doc = json.loads(json.dumps(s.own))
    edit(doc["ports"])
    (s.dir / f"{name}.json").write_text(json.dumps(doc))
    return s.dir / f"{name}.json"


def test_an_honest_registered_statement_is_accepted_and_reports_its_roots(staged):
    code, out = _statement(staged, _stage(staged, "honest"), staged.dir / "own.json")
    assert code == 0, out
    line = json.loads(out.splitlines()[0])
    assert line["registered"] == {"cs": {"value": VALUE, "root": staged.reg.root.hex()}}
    assert line["partition_check"]["units"] == "derived"
    assert json.loads(line["identity"])["outputs"].startswith("hidden"), "the statement hides its outputs (§16.13)"


def test_registered_reads_need_the_verifiers_own_roots(staged):
    """A statement that names registered reads, checked with no registered roots, is refused (its rows would open nothing)."""
    code, out = _statement(staged, _stage(staged, "no-own"), None)
    assert code == 1 and "the verifier holds no registered roots" in out


def test_registered_reads_need_the_units_derived(staged):
    """A read's position is the verifier's for a unit, and which unit an instance is comes from its label: without the
    verifier's program the units are taken as stated, and registered reads are refused."""
    code, out = _statement(staged, _stage(staged, "as-stated"), staged.dir / "own.json", staged.bound[:2])
    assert code == 1 and "units are taken as stated" in out, out


def test_a_registered_value_committed_afresh_is_refused(staged):
    """The verifier reads cs as registered: a statement that names no registered entry for it (its rows under fresh salts,
    with no path), or whose cs rows carry fresh salts under the registered paths, is refused."""
    code, out = _statement(staged, _stage(staged, "plain", registered=False), staged.dir / "own.json")
    assert code == 1 and "names no registered rows" in out
    fresh = [hm96.SHA512.fresh_salt() for _ in G.table(POSITIONS)[0]]
    code, out = _statement(staged, _stage(staged, "fresh", salts=fresh), staged.dir / "own.json")
    assert code == 1 and f"does not open {VALUE}'s registered root at position {POSITIONS[0]}" in out


def test_each_table_row_opens_the_registered_root_at_its_position(staged):
    positions, _ = G.table(POSITIONS)
    _, _, e = staged.reg.statement(positions)
    swapped = {**e, "paths": [e["paths"][1], e["paths"][0], *e["paths"][2:]]}
    moved = {**e, "positions": [positions[0] - 1, *positions[1:]]}
    short = {**e, "paths": [e["paths"][0][:-128], *e["paths"][1:]]}
    for name, entry, want in (("swapped", swapped, "table row 0 does not open"), ("moved", moved, "table row 0 does not open"),
                              ("short", short, "table row 0 does not open"), ("value", {**e, "value": "rope.other"}, f"reads {VALUE}, not")):
        code, out = _statement(staged, _stage(staged, name, entry=entry), staged.dir / "own.json")
        assert code == 1 and want in out, (name, out)


def test_each_instance_reads_the_position_its_unit_reads_in_the_verifiers_program(staged):
    """A table row that opens the registered root is still refused when an instance reads it in place of the row its unit
    reads in the verifier's program."""
    public = _stage(staged, "honest-2")
    code, out = _statement(staged, public, _own(staged, "own-moved", lambda ps: ps["cs"]["reads"].__setitem__(3, 10)))
    assert code == 1 and f"unit 3 reads position {POSITIONS[3]}, and the verifier's program reads 10" in out
    code, out = _statement(staged, public, _own(staged, "own-none", lambda ps: ps["cs"]["reads"].__setitem__(4, None)))
    assert code == 1 and f"reads no row of {VALUE} in unit 4" in out


def test_the_ports_are_exactly_the_verifiers(staged):
    public = _stage(staged, "honest-3")
    code, out = _statement(staged, public, _own(staged, "own-root", lambda ps: ps["cs"].__setitem__("root", "00" * 64)))
    assert code == 1 and f"does not open {VALUE}'s registered root" in out
    code, out = _statement(staged, public, _own(staged, "own-x", lambda ps: ps.__setitem__("x", {**ps["cs"], "value": "x"})))
    assert code == 1 and "commits port x's rows of x afresh" in out
    code, out = _statement(staged, public, _own(staged, "own-y", lambda ps: ps.__setitem__("y", ps["cs"])))
    assert code == 1 and "y is not an input port" in out
    x_entry = {"value": "x", "positions": list(range(8)), "paths": [""] * 8}
    code, out = _statement(staged, _stage(staged, "extra-x", extra={"x": x_entry}), staged.dir / "own.json")
    assert code == 1 and "names port x registered, and the verifier's registered reads do not" in out
    bad = staged.dir / "bad.json"
    cs = {k: v for k, v in staged.own["ports"]["cs"].items() if k not in ("words", "word_bits")}
    for doc, want in (({"format": "verity/registered-reads/v1", "ports": {}}, "not a verity/registered-reads/v0 file"),
                      ({**staged.own, "ports": {}}, "no port"),
                      ({**staged.own, "ports": {"cs": {**cs, "words": 64, "word_bits": 8}}}, "leaves of 16 or 32 bits"),
                      ({**staged.own, "ports": {"cs": {**cs, "word_bits": 32}}}, "a row's words when they are 32"),
                      ({**staged.own, "ports": {"cs": {**cs, "words": 0}}}, "at least one to a row")):
        bad.write_text(json.dumps(doc))
        code, out = _statement(staged, public, bad)
        assert code == 2 and want in out, out


def test_the_ports_rows_are_the_registered_rows(staged):
    """The reads file's row width (`words` leaves of `word_bits`, each leaf `word_bits / 16` u16 words, padded to whole
    SHA-512 blocks) must be the circuit port's: cs's rows are 64 u16 words, one block, which 64 16-bit leaves fill and so
    do 32 32-bit ones; 64 32-bit leaves or 65 16-bit ones are two blocks, and are refused. A file that names no width has
    the port's own 16-bit rows (the fixture's, `test_a_session_verified_with_registered_reads_is_held_to_them`)."""
    assert staged.own["ports"]["cs"]["words"] == 64 and staged.own["ports"]["cs"]["word_bits"] == 16
    public = _stage(staged, "honest-6")
    for name, width, want in (("own-32x32", {"words": 32, "word_bits": 32}, None),
                              ("own-64x32", {"words": 64, "word_bits": 32}, "rows of 64 32-bit leaves"),
                              ("own-65x16", {"words": 65}, "rows of 65 16-bit leaves")):
        code, out = _statement(staged, public, _own(staged, name, lambda ps, width=width: ps["cs"].update(width)))
        if want is None:
            assert code == 0, (name, out)
        else:
            assert code == 1 and f"port cs's rows are 64 u16 words, not {VALUE}'s registered {want}" in out, (name, out)


def test_an_output_row_is_never_a_registered_value(staged):
    """An output row is its instance's own (§16.13): reads that name one are refused, before any row is opened."""
    code, out = _statement(staged, _stage(staged, "honest-4"), _own(staged, "own-out", lambda ps: ps.__setitem__("out", ps["cs"])))
    assert code == 1 and "out is an output row, its instance's own" in out, out


def test_each_table_of_a_session_is_held_to_the_registered_reads(staged):
    """`verify` sets up every table's statement through the same check (§16.12): with two tables, table 1's drawn statement
    (units 4-7) is refused when the verifier's program moves unit 5's read, and under the honest reads setup passes and the
    session fails only on its missing proofs."""
    sess = staged.dir / "session"
    sess.mkdir(exist_ok=True)
    (sess / "session.json").write_text("{}")
    public = _stage(staged, "honest-5")

    def why(own: Path) -> str:
        p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(staged.dir / "circuit.txt"), "--public",
                            str(public), *staged.bound, "--registered", str(own), "--session-tables", "2", "--session", str(sess)],
                           capture_output=True, text=True, check=False)
        (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        assert line["accepted"] is False
        return line["why"]

    assert why(staged.dir / "own.json").startswith("session files: "), "setup passes: the session has no proofs"
    moved = _own(staged, "own-moved-5", lambda ps: ps["cs"]["reads"].__setitem__(5, 10))
    assert why(moved) == f"setup: registered reads: port cs: unit 5 reads position {POSITIONS[5]}, and the verifier's program reads 10"


@pytest.fixture(scope="module")
def rope(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return root


def test_a_session_verified_with_registered_reads_is_held_to_them(rope):
    """`verify` builds every table's statement through the same check on the public-output statement too: the fixture's
    honest session, proved by upstream over a registered statement, is accepted under its own reads file and refused under
    one that moves every position."""
    program, _ = _program()
    assert canonical_json(encode_program(program)) == (rope / "stage" / "program.json").read_bytes()
    bound = ["--partition", str(rope / "stage" / "partition.json"), "--program", str(rope / "stage" / "program.json")]

    def verify(session: str, own: Path) -> dict:
        p = subprocess.run([str(BIN), "verify", "--statement", f"{UPSTREAM_STATEMENT}+seed-injection", "--circuit",
                            str(rope / "stage" / "circuit.txt"), "--public", str(rope / "registered" / "pub-8.bin"), *bound,
                            "--registered", str(own), "--session", str(rope / "vectors" / session)],
                           capture_output=True, text=True, check=False)
        (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        return line

    ok = verify("honest", rope / "registered" / "own.json")
    assert ok["accepted"] is True and ok["partition"]["units"] == "derived", ok
    root = json.loads((rope / "registered" / "own.json").read_text())["ports"]["cs"]["root"]
    assert ok["registered"] == {"cs": {"value": VALUE, "root": root}}
    moved = verify("honest", rope / "vectors" / "wrong-position" / "registered.json")
    assert moved["accepted"] is False and "reads position" in moved["why"], moved
