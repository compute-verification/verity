"""Registered reads (PROTOCOL.md §16.10, `protocols/sampled_proofs/README.md` "Registered values"): a statement whose port reads
rows of a registered value (RoPE's `cs`, rows of the program's `cs_table`, committed once at registration) opens every
table row against the verifier's own registered root, at the position the verifier's own program reads, with the units
derived from that program, or the Lean verifier refuses it. The statements are staged here over the RoPE d64 circuit of
`FIXTURE` (vectors.json's registered-reads set), bound to its `RegisteredRoPE` program and `Q_template_instance` v0
partition; the reads come from `reads` on that program (`statement`: setup, no proof). Each call sets up that 56 MB
circuit (about 8 s), so the module is slow; it catches what the Python reference (`protocols/sampled_proofs/
tests/test_registered.py`) can't: the executable verifier's own refusals."""
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
STATEMENT = "verity/flock-circuit@967b8d06"
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
def rope(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return root


@pytest.fixture(scope="module")
def staged(rope, tmp_path_factory):
    """The registered cs table, the verifier's own reads file from its program, and the statement's bindings."""
    d = tmp_path_factory.mktemp("registered")
    program, query = _program()
    assert canonical_json(encode_program(program)) == (rope / "stage" / "program.json").read_bytes()
    reads = G.reads(program, query, {VALUE: 64})
    assert reads == {"cs": {"value": VALUE, "rows": list(POSITIONS)}}
    rng = np.random.default_rng(20261001)
    reg = G.Registered.commit(P.program_sha512(program), VALUE, rng.integers(0, 1 << 16, (TABLE_ROWS, 64)).tolist())
    own = G.reads_file(program, query, {VALUE: 64}, {VALUE: reg.root.hex()})
    (d / "own.json").write_text(json.dumps(own))
    return SimpleNamespace(dir=d, text=(rope / "stage" / "circuit.txt").read_text(), reg=reg, own=own,
                           x=rng.integers(0, 1 << 16, (8, 64)), bound=["--partition", str(rope / "stage" / "partition.json"),
                                                                     "--program", str(rope / "stage" / "program.json")])


def _stage(s, name: str, registered=True, salts=None, entry=None, extra=None) -> Path:
    positions, refs = G.table(POSITIONS)
    rows, reg_salts, e = s.reg.statement(positions)
    header = s.text.split("\n", 2)[1]
    sub = json.loads(header[5:]).get("subcircuit", "rope-head/d64/neox-bf16")
    inputs = SimpleNamespace(name="rope-registered", content_digest="00" * 32, subcircuit=SimpleNamespace(id=sub))
    reg = {"cs": entry or e, **(extra or {})} if registered else None
    out = s.dir / f"{name}.bin"
    FC.write(s.text, inputs, 0, 8, None, out, None, list(range(8)), share_rows=True, outs={"out": np.zeros((8, 64), np.uint16)},
             tables={"x": s.x, "cs": np.asarray(rows)}, refs={"x": list(range(8)), "cs": refs},
             salts={"x": [hm96.SHA512.fresh_salt() for _ in range(8)], "cs": salts or reg_salts}, registered=reg)
    return out


def _statement(rope, s, public: Path, own: Path | None, bound: list[str] | None = None) -> tuple[int, str]:
    args = [str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(rope / "stage" / "circuit.txt"), "--public", str(public)]
    p = subprocess.run(args + (s.bound if bound is None else bound) + (["--registered", str(own)] if own else []),
                       capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def _own(s, name: str, edit) -> Path:
    doc = json.loads(json.dumps(s.own))
    edit(doc["ports"])
    (s.dir / f"{name}.json").write_text(json.dumps(doc))
    return s.dir / f"{name}.json"


def test_an_honest_registered_statement_is_accepted_and_reports_its_roots(rope, staged):
    code, out = _statement(rope, staged, _stage(staged, "honest"), staged.dir / "own.json")
    assert code == 0, out
    line = json.loads(out.splitlines()[0])
    assert line["registered"] == {"cs": {"value": VALUE, "root": staged.reg.root.hex()}}
    assert line["partition"]["units"] == "derived"


def test_registered_reads_need_the_verifiers_own_roots(rope, staged):
    """A statement that names registered reads, checked with no registered roots, is refused (its rows would open nothing)."""
    code, out = _statement(rope, staged, _stage(staged, "no-own"), None)
    assert code == 1 and "the verifier holds no registered roots" in out


def test_registered_reads_need_the_units_derived(rope, staged):
    """A read's position is the verifier's for a unit, and which unit an instance is comes from its label: without the
    verifier's program the units are taken as stated, and registered reads are refused."""
    code, out = _statement(rope, staged, _stage(staged, "as-stated"), staged.dir / "own.json", staged.bound[:2])
    assert code == 1 and "units are taken as stated" in out, out


def test_a_registered_value_committed_afresh_is_refused(rope, staged):
    """The verifier reads cs as registered: a statement that names no registered entry for it (its rows under fresh salts,
    with no path), or whose cs rows carry fresh salts under the registered paths, is refused."""
    code, out = _statement(rope, staged, _stage(staged, "plain", registered=False), staged.dir / "own.json")
    assert code == 1 and "names no registered rows" in out
    fresh = [hm96.SHA512.fresh_salt() for _ in G.table(POSITIONS)[0]]
    code, out = _statement(rope, staged, _stage(staged, "fresh", salts=fresh), staged.dir / "own.json")
    assert code == 1 and f"does not open {VALUE}'s registered root at position {POSITIONS[0]}" in out


def test_each_table_row_opens_the_registered_root_at_its_position(rope, staged):
    positions, _ = G.table(POSITIONS)
    _, _, e = staged.reg.statement(positions)
    swapped = {**e, "paths": [e["paths"][1], e["paths"][0], *e["paths"][2:]]}
    moved = {**e, "positions": [positions[0] - 1, *positions[1:]]}
    short = {**e, "paths": [e["paths"][0][:-128], *e["paths"][1:]]}
    for name, entry, want in (("swapped", swapped, "table row 0 does not open"), ("moved", moved, "table row 0 does not open"),
                              ("short", short, "table row 0 does not open"), ("value", {**e, "value": "rope.other"}, f"reads {VALUE}, not")):
        code, out = _statement(rope, staged, _stage(staged, name, entry=entry), staged.dir / "own.json")
        assert code == 1 and want in out, (name, out)


def test_each_instance_reads_the_position_its_unit_reads_in_the_verifiers_program(rope, staged):
    """A table row that opens the registered root is still refused when an instance reads it in place of the row its unit
    reads in the verifier's program."""
    public = _stage(staged, "honest-2")
    code, out = _statement(rope, staged, public, _own(staged, "own-moved", lambda ps: ps["cs"]["reads"].__setitem__(3, 10)))
    assert code == 1 and f"unit 3 reads position {POSITIONS[3]}, and the verifier's program reads 10" in out
    code, out = _statement(rope, staged, public, _own(staged, "own-none", lambda ps: ps["cs"]["reads"].__setitem__(4, None)))
    assert code == 1 and f"reads no row of {VALUE} in unit 4" in out


def test_the_ports_are_exactly_the_verifiers(rope, staged):
    public = _stage(staged, "honest-3")
    code, out = _statement(rope, staged, public, _own(staged, "own-root", lambda ps: ps["cs"].__setitem__("root", "00" * 64)))
    assert code == 1 and f"does not open {VALUE}'s registered root" in out
    code, out = _statement(rope, staged, public, _own(staged, "own-x", lambda ps: ps.__setitem__("x", {**ps["cs"], "value": "x"})))
    assert code == 1 and "commits port x's rows of x afresh" in out
    code, out = _statement(rope, staged, public, _own(staged, "own-y", lambda ps: ps.__setitem__("y", ps["cs"])))
    assert code == 1 and "y is not an input port" in out
    x_entry = {"value": "x", "positions": list(range(8)), "paths": [""] * 8}
    code, out = _statement(rope, staged, _stage(staged, "extra-x", extra={"x": x_entry}), staged.dir / "own.json")
    assert code == 1 and "names port x registered, and the verifier's registered reads do not" in out
    bad = staged.dir / "bad.json"
    for doc, want in (({"format": "verity/registered-reads/v1", "ports": {}}, "not a verity/registered-reads/v0 file"),
                      ({**staged.own, "ports": {}}, "no port")):
        bad.write_text(json.dumps(doc))
        code, out = _statement(rope, staged, public, bad)
        assert code == 2 and want in out, out


def test_a_session_verified_with_registered_reads_is_held_to_them(rope, staged):
    """`verify` builds every table's statement through the same check: the fixture's honest session, proved by upstream over
    a registered statement, is accepted under its own reads file and refused under one that moves every position."""
    def verify(session: str, own: Path) -> dict:
        p = subprocess.run([str(BIN), "verify", "--statement", f"{STATEMENT}+seed-injection", "--circuit",
                            str(rope / "stage" / "circuit.txt"), "--public", str(rope / "registered" / "pub-8.bin"), *staged.bound,
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
