"""Registered reads (PROTOCOL.md §16.10, `verity/core/protocols/sampled_proofs/README.md` "Registered values"): a statement whose port reads
rows of a registered value (RoPE's `cs`, rows of the program's `cs_table`, committed once at registration) opens every
table row against the verifier's own registered root, at the position the verifier's own program reads, with the units
derived from that program, or the Lean verifier refuses it. The statements are staged here over a hidden-output RoPE d64
circuit (§16.13, the default `verity/flock-circuit`) composed for the `RegisteredRoPE` program and its `Q_template_instance`
v0 partition; the reads come from `reads` on that program (`statement`: setup, no proof; `verify`: setup of each table,
with no proofs). Upstream's proved session over the public-output statement (`FIXTURE`, vectors.json's registered-reads
set) is held to the same reads. Composing the 56 MB circuit takes about 30 s and each call sets it up (about 8 s), so the
module is slow; it catches what the Python reference (`verity/core/protocols/sampled_proofs/tests/test_registered.py`) can't: the
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

from verity.ml.commitments import hm96
from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array
from verity_flock import circuit as FC
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
FIXTURE = "art:bf70d8eb"
STATEMENT = "verity/flock-circuit"
UPSTREAM_STATEMENT = "verity/flock-circuit@967b8d06"
VALUE = "cs_table"
#: the program's: the row of `cs_table` each of its 8 RoPEHead Calls reads
POSITIONS = (11, 5, 3, 1, 0, 9, 8, 4)
TABLE_ROWS = 12
#: a registered value whose rows are the units' outputs (PoUW's checked words): the row each unit's `out` must be, which
#: the verifier's own reads file states (its program has no parameter that names it)
OUT_VALUE = "rope_out"
OUT_POSITIONS = (2, 7, 0, 9, 4, 4, 1, 6)
OUT_ROWS = 10

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
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit, partition=P.digest(part), program=P.program_digest(program),
                      statics=FC.set_statics(s))
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    (d / "partition.json").write_text(json.dumps(part))
    reads = G.reads(program, query, {VALUE: 64})
    assert reads == {"cs": {"value": VALUE, "rows": list(POSITIONS)}}
    rng = np.random.default_rng(20261001)
    reg = G.Registered.commit(P.program_digest(program), VALUE, rng.integers(0, 1 << 16, (TABLE_ROWS, 64)).tolist())
    own = G.reads_file(program, query, {VALUE: 64}, {VALUE: reg.root.hex()})
    (d / "own.json").write_text(json.dumps(own))
    x = rng.integers(0, 1 << 16, (8, 64))
    out = G.Registered.commit(P.program_digest(program), OUT_VALUE,
                              np.random.default_rng(20261003).integers(0, 1 << 16, (OUT_ROWS, 64)).tolist())
    own_out = {**own, "ports": {**own["ports"], "out": {**out.entry(), "word_bits": 16, "reads": list(OUT_POSITIONS)}}}
    (d / "own-out.json").write_text(json.dumps(own_out))
    return SimpleNamespace(dir=d, text=text, reg=reg, own=own, out=out, own_out=own_out, program=program, x=x,
                           bound=["--partition", str(d / "partition.json"), "--program", str(d / "program.json")])


def _stage(s, name: str, registered=True, salts=None, entry=None, extra=None, outs=None, out_salts=None, cs=None) -> Path:
    positions, refs = G.table(POSITIONS)
    rows, reg_salts, e = s.reg.statement(positions)
    rows = rows if cs is None else cs
    sub = json.loads(s.text.split("\n", 2)[1][5:]).get("subcircuit", "rope-head/d64/neox-bf16")
    inputs = SimpleNamespace(name="rope-registered", subcircuit=SimpleNamespace(id=sub))
    reg = {"cs": entry or e, **(extra or {})} if registered else extra
    out = s.dir / f"{name}.bin"
    fresh = [hm96.SHA512.fresh_salt() for _ in range(8)]
    FC.write(s.text, inputs, 0, 8, None, out, None, list(range(8)), share_rows=True,
             outs={"out": np.zeros((8, 64), np.uint16) if outs is None else np.asarray(outs, np.uint16)},
             tables={"x": s.x, "cs": np.asarray(rows)}, refs={"x": list(range(8)), "cs": refs},
             salts={"x": fresh, "cs": salts or reg_salts, "out": out_salts or [hm96.SHA512.fresh_salt() for _ in range(8)]},
             registered=reg)
    return out


def _stage_out(s, name: str, positions=OUT_POSITIONS, outs=None, entry=None) -> Path:
    """The honest statement with its output row registered too: each instance's output the registered `rope_out` row at its
    unit's position, committed under that row's registered salt, the header naming each row's position and path."""
    rows, salts, e = s.out.statement(positions)
    return _stage(s, name, extra={"out": entry or e}, outs=rows if outs is None else outs, out_salts=salts)


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


def test_a_table_of_two_registrations_rows_opens_no_one_root(staged):
    """A table whose rows are two registrations' of the value (two row tables of a window, each row under its own salt with
    a true path to its own root) is refused whichever one root the verifier holds: a port is read against one registered
    root, so a prover that builds a chunk over two row tables can't have it verified (`tables.registered_reads`)."""
    positions, _ = G.table(POSITIONS)
    other = G.Registered.commit(P.program_digest(staged.program), VALUE,
                                np.random.default_rng(20261024).integers(0, 1 << 16, (TABLE_ROWS, 64)).tolist())
    (rows_a, salts_a, e_a), (rows_b, salts_b, e_b) = staged.reg.statement(positions), other.statement(positions)
    half = len(positions) // 2
    mix = lambda a, b: list(a[:half]) + list(b[half:])  # noqa: E731
    public = _stage(staged, "two-tables", cs=mix(rows_a, rows_b), salts=mix(salts_a, salts_b),
                    entry={**e_a, "paths": mix(e_a["paths"], e_b["paths"])})
    code, out = _statement(staged, public, staged.dir / "own.json")
    assert code == 1 and f"table row {half} does not open {VALUE}'s registered root at position {positions[half]}" in out, out
    code, out = _statement(staged, public, _own(staged, "own-other-root", lambda ps: ps["cs"].__setitem__("root", other.root.hex())))
    assert code == 1 and f"table row 0 does not open {VALUE}'s registered root at position {positions[0]}" in out, out


def test_each_instance_reads_the_position_its_unit_reads_in_the_verifiers_program(staged):
    """A table row that opens the registered root is still refused when an instance reads it in place of the row its unit
    reads in the verifier's program."""
    public = _stage(staged, "honest-2")
    code, out = _statement(staged, public, _own(staged, "own-moved", lambda ps: ps["cs"]["reads"].__setitem__(3, 10)))
    assert code == 1 and f"unit 3 reads position {POSITIONS[3]}, and the verifier's program reads 10" in out
    code, out = _statement(staged, public, _own(staged, "own-none", lambda ps: ps["cs"]["reads"].__setitem__(4, None)))
    assert code == 1 and f"reads no row of {VALUE} in unit 4" in out


def test_a_unit_past_a_ports_reads_reads_none(staged):
    """A unit past a port's ``reads`` reads none, as a null there does (`badInstance`'s ``reads[u]?``), so a reads file
    trimmed of its trailing nulls (`h1window.table_reads_file`'s) verifies as the untrimmed one: the honest statement is
    accepted alike under reads padded with nulls past the population, and a statement whose last unit's read is null or
    absent is refused alike."""
    public = _stage(staged, "honest-trim")

    def verdict(name, edit):
        code, out = _statement(staged, public, _own(staged, name, edit))
        return code, [line for line in out.splitlines() if " ms" not in line]

    trimmed = verdict("own-trimmed", lambda ps: None)
    assert trimmed[0] == 0 and verdict("own-padded", lambda ps: ps["cs"]["reads"].extend([None] * 5)) == trimmed
    null = verdict("own-last-null", lambda ps: ps["cs"]["reads"].__setitem__(7, None))
    assert null[0] == 1 and any(f"reads no row of {VALUE} in unit 7" in line for line in null[1]), null
    assert verdict("own-last-absent", lambda ps: ps["cs"]["reads"].pop()) == null


def test_the_ports_are_exactly_the_verifiers(staged):
    public = _stage(staged, "honest-3")
    code, out = _statement(staged, public, _own(staged, "own-root", lambda ps: ps["cs"].__setitem__("root", "00" * 64)))
    assert code == 1 and f"does not open {VALUE}'s registered root" in out
    code, out = _statement(staged, public, _own(staged, "own-x", lambda ps: ps.__setitem__("x", {**ps["cs"], "value": "x"})))
    assert code == 1 and "commits port x's rows of x afresh" in out
    code, out = _statement(staged, public, _own(staged, "own-y", lambda ps: ps.__setitem__("y", ps["cs"])))
    assert code == 1 and "y is not a row of the circuit (an input port, or an output port's row)" in out
    x_entry = {"value": "x", "positions": list(range(8)), "paths": [""] * 8}
    code, out = _statement(staged, _stage(staged, "extra-x", extra={"x": x_entry}), staged.dir / "own.json")
    assert code == 1 and "names port x registered, and the verifier's registered reads do not" in out
    bad = staged.dir / "bad.json"
    cs = {k: v for k, v in staged.own["ports"]["cs"].items() if k not in ("words", "word_bits")}
    for doc, want in (({"format": "verity/registered-reads/v2", "ports": {}}, "not a verity/registered-reads/v1 file"),
                      ({**staged.own, "ports": {"cs": {**cs, "reads_from": 0}}}, "reads_from is a verity/registered-reads/v1 key"),
                      ({**staged.own, "format": G.READS_V1, "ports": {"cs": {**cs, "reads_at": 0}}}, "and no other"),
                      ({**staged.own, "format": G.READS_V1, "window": 0}, "and no other"),
                      ({**staged.own, "ports": {}}, "no port"),
                      ({**staged.own, "ports": {"cs": {**cs, "words": 64, "word_bits": 8}}}, "leaves of 16 or 32 bits"),
                      ({**staged.own, "ports": {"cs": {**cs, "word_bits": 32}}}, "a row's words when they are 32"),
                      ({**staged.own, "ports": {"cs": {**cs, "words": 0}}}, "at least one to a row")):
        bad.write_text(json.dumps(doc))
        code, out = _statement(staged, public, bad)
        assert code == 2 and want in out, out


def test_a_v1_files_reads_start_at_its_first_unit(staged):
    """A `verity/registered-reads/v1` port's `reads` start at unit `reads_from`: from unit 0 they are v0's, and from unit 1
    unit 0 reads nothing, so its instance is refused."""
    public = _stage(staged, "honest-v1")
    v1 = staged.dir / "own-v1.json"
    v1.write_text(json.dumps({**staged.own, "format": G.READS_V1}))
    code, out = _statement(staged, public, v1)
    assert code == 0, out
    cs = staged.own["ports"]["cs"]
    later = staged.dir / "own-v1-later.json"
    later.write_text(json.dumps({**staged.own, "format": G.READS_V1,
                                 "ports": {"cs": {**cs, G.READS_FROM: 1, "reads": cs["reads"][1:]}}}))
    code, out = _statement(staged, public, later)
    assert code == 1 and f"reads no row of {VALUE} in unit 0" in out, out


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
            assert code == 1 and f"port cs's rows are 1024 bits (64 u16 words), not {VALUE}'s registered {want}" in out, (name, out)


def test_a_row_v1_port_is_refused_a_bit_row_registration(staged):
    """cs is a `hm96-sha512/row/v1` port: a reads file that names its value's rows in another format (`row_schema`) is
    refused before any row is opened, since its root's leaves are framed under that format."""
    public = _stage(staged, "honest-v2")
    for schema in (G.SCHEMA_V2, G.SCHEMA_SEG):
        code, out = _statement(staged, public, _own(staged, f"own-{schema[-6:].replace('/', '-')}",
                                                    lambda ps, schema=schema: ps["cs"].update(row_schema=schema)))
        assert code == 1 and (f"port cs is a {G.SCHEMA} row, and {VALUE}'s registered rows are {schema} rows") in out, out


def test_a_registered_output_row_is_accepted_and_reports_its_root(staged):
    """An output row (§16.13) is a registered port too: each instance's output row, committed under the registered row's
    salt, opens the registered root at the position the verifier's reads file names for its unit."""
    code, out = _statement(staged, _stage_out(staged, "out-honest"), staged.dir / "own-out.json")
    assert code == 0, out
    line = json.loads(out.splitlines()[0])
    assert line["registered"] == {"cs": {"value": VALUE, "root": staged.reg.root.hex()},
                                  "out": {"value": OUT_VALUE, "root": staged.out.root.hex()}}


def test_a_registered_output_row_must_be_the_registered_row(staged):
    """An instance whose output is not the registered row (its salt the registered one), or whose row is committed under a
    fresh salt, does not open the registered root; nor does a row with another row's path, or at another position."""
    rows, _, e = staged.out.statement(OUT_POSITIONS)
    wrong = np.asarray(rows, np.uint16)
    wrong[3, 5] ^= 1
    code, out = _statement(staged, _stage_out(staged, "out-wrong", outs=wrong), staged.dir / "own-out.json")
    assert code == 1 and f"port out: table row 3 does not open {OUT_VALUE}'s registered root at position {OUT_POSITIONS[3]}" in out, out
    fresh = _stage(staged, "out-fresh", extra={"out": e}, outs=rows)
    code, out = _statement(staged, fresh, staged.dir / "own-out.json")
    assert code == 1 and f"port out: table row 0 does not open {OUT_VALUE}'s registered root at position {OUT_POSITIONS[0]}" in out, out
    swapped = {**e, "paths": [e["paths"][1], e["paths"][0], *e["paths"][2:]]}
    moved = {**e, "positions": [OUT_POSITIONS[0] + 1, *OUT_POSITIONS[1:]]}
    for name, entry in (("out-swapped", swapped), ("out-moved", moved)):
        code, out = _statement(staged, _stage_out(staged, name, entry=entry), staged.dir / "own-out.json")
        assert code == 1 and "port out: table row 0 does not open" in out, (name, out)


def test_a_registered_output_row_is_at_its_units_position(staged):
    """A row that opens the registered root at its own position is refused when the verifier's reads put the unit at
    another: unit 2's output is the registered row 3, and the verifier reads row 0 for it."""
    at = (*OUT_POSITIONS[:2], 3, *OUT_POSITIONS[3:])
    code, out = _statement(staged, _stage_out(staged, "out-elsewhere", positions=at), staged.dir / "own-out.json")
    assert code == 1 and f"port out: unit 2 reads position 3, and the verifier's program reads {OUT_POSITIONS[2]}" in out, out


def test_a_registered_output_row_is_on_both_sides(staged):
    """The statement and the verifier name the same registered ports: an output row the statement names registered and the
    verifier's file does not is refused, and so is one the verifier reads as registered and the statement commits afresh."""
    code, out = _statement(staged, _stage_out(staged, "out-one-side"), staged.dir / "own.json")
    assert code == 1 and "names port out registered, and the verifier's registered reads do not" in out, out
    code, out = _statement(staged, _stage(staged, "out-afresh"), staged.dir / "own-out.json")
    assert code == 1 and f"commits port out's rows of {OUT_VALUE} afresh" in out, out


def test_each_table_of_a_session_is_held_to_the_registered_reads(staged):
    """`verify` sets up every table's statement through the same check (§16.12): with two tables, table 1's drawn statement
    (units 4-7) is refused when the verifier's program moves unit 5's read, and under the honest reads setup passes, so the
    verifier goes on to read the (empty) session's proofs, which it names by the tables it set up."""
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

    honest = why(staged.dir / "own.json")
    assert honest.startswith("session files: ") and "circuit-00.rep0.bin" in honest, honest
    moved = _own(staged, "own-moved-5", lambda ps: ps["cs"]["reads"].__setitem__(5, 10))
    assert why(moved) == f"setup: registered reads: port cs: unit 5 reads position {POSITIONS[5]}, and the verifier's program reads 10"


@pytest.mark.parametrize("value", ["rec-top-L0", "rec-top-stacked-L0"])
def test_a_drawn_session_draws_every_unit_reading_a_rounds_commitment_in_full(staged, value):
    """U4 (§16.10, `Registered.checkFull`): a statement whose registered output row is a round's commitment (level 0's
    committed tops `rec-top-L0`, or R's `rec-top-stacked-L0`; a `rec-c<k>` meets `checkPubRows` first) is refused at setup
    when the record's draw does not draw every unit reading it in full, with one table and with two (`buildSession`'s call:
    `verify` sets up every hm96 statement through `Stmt.setupTables`, and `buildStmt`'s hm96 path serves `statement`, which
    has no draw). The full draw passes U4: `rec-top-L0` then meets `checkTopRows` and R's tops `checkPubRows`, since the
    verifier holds no inner transcript to check them against. U4's Lean test (`Flock.Test.Full`) checks the function; this
    checks that `verify` calls it."""
    c = G.Registered.commit(P.program_digest(staged.program), value,
                            np.random.default_rng(20261010).integers(0, 1 << 16, (OUT_ROWS, 64)).tolist())
    rows, salts, e = c.statement(OUT_POSITIONS)
    public = _stage(staged, value, extra={"out": e}, outs=rows, out_salts=salts)
    own = staged.dir / f"own-{value}.json"
    own.write_text(json.dumps({**staged.own, "ports": {**staged.own["ports"],
                                                       "out": {**c.entry(), "word_bits": 16, "reads": list(OUT_POSITIONS)}}}))

    def why(k: int, tables: int) -> str:
        sess = staged.dir / f"session-{value}-{k}-{tables}"
        sess.mkdir(exist_ok=True)
        draw = {"law": "subset", "population": len(OUT_POSITIONS), "k": k, "units": list(range(k))}
        (sess / "session.json").write_text(json.dumps({"unit_draw": draw}))
        p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(staged.dir / "circuit.txt"), "--public",
                            str(public), *staged.bound, "--registered", str(own),
                            *(["--session-tables", str(tables)] if tables > 1 else []), "--session", str(sess)],
                           capture_output=True, text=True, check=False)
        (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        assert line["accepted"] is False
        return line["why"]

    full = "setup: rec-tops: " if value == "rec-top-L0" else "setup: rec-pub: "
    for tables in (1, 2):
        assert why(len(OUT_POSITIONS), tables).startswith(full), tables
        assert why(2, tables) == (f"setup: U4: unit 0 reads {value}, a round's commitment (port out), and the draw does not draw "
                                  "it in full"), tables


#: the inner session's link points, two of two coins (`f128_hex`)
LINK_POINTS = [[f"{2 * i + j + 1:032x}" for j in range(2)] for i in range(2)]


def _coin(x: int, k: int) -> list[str]:
    """Rep `x`'s round `k`'s one coin."""
    return [f"{0x100 + 2 * x + k:032x}"]


def _inner_session(d: Path, points=LINK_POINTS) -> list[str]:
    """An inner session of two reps of two rounds each, one coin a round (`_coin`), whose record holds the link points
    `points`, and stand-ins for the inner statement: a circuit header whose META binds nothing, an empty public file and
    partition. Only check 5, which no statement here meets, reads the inner statement."""
    (d / "session").mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text("inner\nMETA {}\n")
    (d / "pub.bin").write_bytes(b"")
    (d / "partition.json").write_text("{}")
    streams = [{"stream": f"circuit/rep{x}", "rounds": [{"k": k, "coins": _coin(x, k)} for k in range(2)]} for x in (0, 1)]
    (d / "session" / "session.json").write_text(json.dumps({"link": {"points": points}, "streams": streams}))
    return ["--rec-inner-circuit", str(d / "circuit.txt"), "--rec-inner-public", str(d / "pub.bin"), "--rec-inner-partition",
            str(d / "partition.json"), "--rec-inner-session", str(d / "session")]


def _hexes(bs: list[bytes]) -> list[str]:
    return [b.hex() for b in bs]


def _inner_transcript(path: Path, root: list[bytes], tops: list[list[bytes]], points=LINK_POINTS) -> Path:
    """The coin server's record of that session: Hello's one table, its `root_B` (`root`, a `b ‖ c` a node), rep `x`'s round 0
    absorbing one `c_k` and its round 1 committing its level-1 top `tops[x]`, and the link points, flat."""
    rounds = [r for x in (0, 1) for r in (
        {"stream": f"circuit/rep{x}", "k": 0, "tops": [], "level0": 0, "commits": _hexes([bytes([0xC0 + x]) * 128]), "n": 1,
         "coins": _coin(x, 0)},
        {"stream": f"circuit/rep{x}", "k": 1, "tops": [_hexes(tops[x])], "level0": 0, "commits": [], "n": 1,
         "coins": _coin(x, 1)})]
    path.write_text(json.dumps({
        "kind": "rec/inner-transcript/v1", "hello": {"tables": ["circuit"], "link": {"m": 2, "points": 2}},
        "link": {"roots": {"circuit": _hexes(root)}, "points": [c for p in points for c in p], "y": None},
        "rounds": rounds, "end": "finish"}))
    return path


@pytest.mark.parametrize("level", [0, 1])
def test_a_rounds_tops_are_the_coin_servers(staged, tmp_path, level):
    """`RecVStar.checkTopRows`: a statement whose `out` reads the circuit table's level-`l` tops `rec-top-L<l>` (V*'s levels'
    `out`) has each table row, at the registered position `x` it opens, the `b ‖ c` the coin server recorded there: at level
    0 node `x` of the table's `root_B`, above it rep 0's level-`l` top and then rep 1's (`rec_outer.TopValue`). Here the
    registered rows are the session's level-`l` tops: its `root_B`, or rep 0's top of six nodes and rep 1's of four. With
    the inner transcript recording them the full draw passes setup (to the session's proof files, which it has none of); a
    transcript whose `root_B`, rep 0's top or rep 1's top differs at one node a row reads is refused at that row, and an
    inner session record whose link points are not the transcript's is refused before setup (`RecVStar.innerLink`). This
    is the one test of `root_B` on a statement: V*'s level 0 at the rec-coins fixture is 436 openings of 64 lanes, about
    30 times level 3's circuit."""
    value = f"rec-top-L{level}"
    c = G.Registered.commit(P.program_digest(staged.program), value,
                            np.random.default_rng(20261018).integers(0, 1 << 16, (OUT_ROWS, 64)).tolist())
    rows, salts, e = c.statement(OUT_POSITIONS)
    public = _stage(staged, f"tops-{level}", extra={"out": e}, outs=rows, out_salts=salts)
    own = staged.dir / f"own-tops-{level}.json"
    own.write_text(json.dumps({**staged.own, "ports": {**staged.own["ports"],
                                                       "out": {**c.entry(), "word_bits": 16, "reads": list(OUT_POSITIONS)}}}))
    sess = staged.dir / f"session-tops-{level}"
    sess.mkdir(exist_ok=True)
    draw = {"law": "subset", "population": len(OUT_POSITIONS), "k": len(OUT_POSITIONS), "units": list(range(len(OUT_POSITIONS)))}
    (sess / "session.json").write_text(json.dumps({"unit_draw": draw}))
    with (staged.dir / "circuit.txt").open() as f:
        unit = f.readline().split()[1]
    bcs, other = list(c.bcs), [bytes([0xA0 + i]) * 128 for i in range(OUT_ROWS)]
    root, tops = (bcs, [other[:6], other[6:]]) if level == 0 else (other, [bcs[:6], bcs[6:]])

    def verify(root, tops, points=LINK_POINTS) -> subprocess.CompletedProcess:
        return subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(staged.dir / "circuit.txt"),
                               "--public", str(public), *staged.bound, "--registered", str(own),
                               *_inner_session(tmp_path / "inner", points), "--rec-inner-transcript",
                               str(_inner_transcript(tmp_path / "transcript.json", root, tops)), "--session", str(sess)],
                              capture_output=True, text=True, check=False)

    def why(root, tops) -> str:
        p = verify(root, tops)
        lines = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        assert len(lines) == 1 and lines[0]["accepted"] is False, (p.returncode, p.stderr)
        return lines[0]["why"]

    def off(xs: list[bytes], i: int) -> list[bytes]:
        return [bytes([b[0] ^ 1]) + b[1:] if j == i else b for j, b in enumerate(xs)]

    assert why(root, tops).startswith("session files: "), why(root, tops)
    # row r of the table reads position OUT_POSITIONS[r]: 2, 7, 0, 9, 4, 4, 1, 6
    if level == 0:
        assert why(off(root, 9), tops) == (f"setup: rec-tops: {unit}: port out's table row 3 is not node 9 of circuit's root_B, "
                                           "which the coin server committed")
    else:
        assert why(root, [off(tops[0], 2), tops[1]]) == (
            f"setup: rec-tops: {unit}: port out's table row 0 is not node 2 of rep 0's level 1 top, which the coin server committed")
        assert why(root, [tops[0], off(tops[1], 1)]) == (
            f"setup: rec-tops: {unit}: port out's table row 1 is not node 1 of rep 1's level 1 top, which the coin server committed")
    p = verify(root, tops, [LINK_POINTS[0], [LINK_POINTS[1][0], f"{0xFF:032x}"]])
    assert (p.returncode, p.stderr.strip()) == (2, "rec inner transcript: the inner session's record's link points are not the "
                                                   "ones the coin server drew (link.points): check 5 would take the extra "
                                                   "claims at points the prover chose")


@pytest.fixture(scope="module")
def rope(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch_sessions(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return root


def test_a_session_verified_with_registered_reads_is_held_to_them(rope):
    """`verify` builds every table's statement through the same check on the public-output statement too: the fixture's
    honest session, proved by upstream over a registered statement, sets up under its own reads file, where it meets the
    refusal of its form, a shared-row public file (§16.15), which only a statement that sets up meets, and is refused at setup
    under one that moves every position. Upstream's seed-injection build has no zk patch, so the session's plain files are
    upstream's form, without the `zk_level0 = None` tag the deployed prover writes, and are read under `--upstream-plain`
    (§8.1)."""
    program, _ = _program()
    assert canonical_json(encode_program(program)) == (rope / "stage" / "program.json").read_bytes()
    bound = ["--partition", str(rope / "stage" / "partition.json"), "--program", str(rope / "stage" / "program.json")]

    def verify(session: str, own: Path) -> dict:
        p = subprocess.run([str(BIN), "verify", "--statement", f"{UPSTREAM_STATEMENT}+seed-injection", "--circuit",
                            str(rope / "stage" / "circuit.txt"), "--public", str(rope / "registered" / "pub-8.bin"), *bound,
                            "--registered", str(own), "--upstream-plain", "--session", str(rope / "vectors" / session)],
                           capture_output=True, text=True, check=False)
        (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        return line

    ok = verify("honest", rope / "registered" / "own.json")
    assert ok["accepted"] is False and ok["why"] == ("refused: a shared-row public file (header shared_rows): a statement form "
                                                     "the soundness proof does not cover"), ok
    root = json.loads((rope / "registered" / "own.json").read_text())["ports"]["cs"]["root"]
    assert ok["registered"] == {"cs": {"value": VALUE, "root": root}}
    moved = verify("honest", rope / "vectors" / "wrong-position" / "registered.json")
    assert moved["accepted"] is False and "reads position" in moved["why"], moved
