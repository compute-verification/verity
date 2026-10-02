"""Registered reads (PROTOCOL.md §16.10, `protocols/sampled_proofs/README.md` "Registered values"): a statement whose port reads
rows of a registered value (RoPE's `cs`, a table committed once at registration) opens every table row against the
verifier's own registered root, at the position the verifier's own program reads, or the Lean verifier refuses it. The
statements are staged here over the RoPE d64 circuit of `art:425f6860` (`statement`: setup, no proof). Each call sets up
that 56 MB circuit (about 8 s), so the module is slow; it catches what the Python reference (`protocols/sampled_proofs/
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
from verity_flock import circuit as FC
from verity_sampled_proofs.one_stage import registered as G

LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
VERIFIER = LEAN.parent
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
FIXTURE = "art:425f6860"
STATEMENT = "verity/flock-circuit@967b8d06"
#: the row of the registered cs table each of the 8 units reads, as the verifier's program fixes it
READS = [5, 5, 0, 11, 3, 5, 7, 0]
TABLE_ROWS = 12

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


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
    """The registered cs table, an honest registered statement, and the verifier's own reads file."""
    d = tmp_path_factory.mktemp("registered")
    text = (rope / "stage" / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    rng = np.random.default_rng(20261001)
    reg = G.Registered.commit(meta["program_sha512"], "rope.cs", rng.integers(0, 1 << 16, (TABLE_ROWS, 64)).tolist())
    own = G.reads_file([reg.entry()], {"cs": {"value": "rope.cs", "rows": list(READS)}})
    (d / "own.json").write_text(json.dumps(own))
    return SimpleNamespace(dir=d, text=text, reg=reg, own=own, x=rng.integers(0, 1 << 16, (8, 64)))


def _stage(s, name: str, registered=True, salts=None, entry=None, extra=None) -> Path:
    positions, refs = G.table(READS)
    rows, reg_salts, e = s.reg.statement(positions)
    header = s.text.split("\n", 2)[1]
    sub = json.loads(header[5:]).get("subcircuit", "rope-head/d64/neox-bf16")
    inputs = SimpleNamespace(name="rope-registered", content_digest="00" * 32, subcircuit=SimpleNamespace(id=sub))
    reg = {"cs": entry or e, **(extra or {})} if registered else None
    out = s.dir / f"{name}.bin"
    FC.write(s.text, inputs, 0, 8, None, out, share_rows=True, outs={"out": np.zeros((8, 64), np.uint16)},
             tables={"x": s.x, "cs": np.asarray(rows)}, refs={"x": list(range(8)), "cs": refs},
             salts={"x": [hm96.SHA512.fresh_salt() for _ in range(8)], "cs": salts or reg_salts}, registered=reg)
    return out


def _statement(rope, public: Path, own: Path | None) -> tuple[int, str]:
    args = [str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(rope / "stage" / "circuit.txt"), "--public", str(public)]
    p = subprocess.run(args + (["--registered", str(own)] if own else []), capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def _own(s, name: str, edit) -> Path:
    doc = json.loads(json.dumps(s.own))
    edit(doc["ports"])
    (s.dir / f"{name}.json").write_text(json.dumps(doc))
    return s.dir / f"{name}.json"


def test_an_honest_registered_statement_is_accepted_and_reports_its_roots(rope, staged):
    code, out = _statement(rope, _stage(staged, "honest"), staged.dir / "own.json")
    assert code == 0, out
    assert json.loads(out.splitlines()[0])["registered"] == {"cs": {"value": "rope.cs", "root": staged.reg.root.hex()}}


def test_registered_reads_need_the_verifiers_own_roots(rope, staged):
    """A statement that names registered reads, checked with no registered roots, is refused (its rows would open nothing)."""
    code, out = _statement(rope, _stage(staged, "no-own"), None)
    assert code == 1 and "the verifier holds no registered roots" in out


def test_a_registered_value_committed_afresh_is_refused(rope, staged):
    """The verifier reads cs as registered: a statement that names no registered entry for it (its rows under fresh salts,
    with no path), or whose cs rows carry fresh salts under the registered paths, is refused; so is the fixture's own
    per-instance statement."""
    code, out = _statement(rope, _stage(staged, "plain", registered=False), staged.dir / "own.json")
    assert code == 1 and "names no registered rows" in out
    fresh = [hm96.SHA512.fresh_salt() for _ in G.table(READS)[0]]
    code, out = _statement(rope, _stage(staged, "fresh", salts=fresh), staged.dir / "own.json")
    assert code == 1 and "does not open rope.cs's registered root at position 5" in out
    code, out = _statement(rope, rope / "stage" / "pub-8.bin", staged.dir / "own.json")
    assert code == 1 and "names no registered rows" in out


def test_each_table_row_opens_the_registered_root_at_its_position(rope, staged):
    positions, _ = G.table(READS)
    _, _, e = staged.reg.statement(positions)
    swapped = {**e, "paths": [e["paths"][1], e["paths"][0], *e["paths"][2:]]}
    moved = {**e, "positions": [positions[0] + 1, *positions[1:]]}
    short = {**e, "paths": [e["paths"][0][:-128], *e["paths"][1:]]}
    for name, entry, want in (("swapped", swapped, "table row 0 does not open"), ("moved", moved, "table row 0 does not open"),
                              ("short", short, "table row 0 does not open"), ("value", {**e, "value": "rope.other"}, "reads rope.cs, not")):
        code, out = _statement(rope, _stage(staged, name, entry=entry), staged.dir / "own.json")
        assert code == 1 and want in out, (name, out)


def test_each_instance_reads_the_position_its_unit_reads_in_the_verifiers_program(rope, staged):
    """A table row that opens the registered root is still refused when an instance reads it in place of the row its unit
    reads in the verifier's program."""
    public = _stage(staged, "honest-2")
    code, out = _statement(rope, public, _own(staged, "own-moved", lambda ps: ps["cs"]["reads"].__setitem__(3, 10)))
    assert code == 1 and "unit 3 reads position 11, and the verifier's program reads 10" in out
    code, out = _statement(rope, public, _own(staged, "own-none", lambda ps: ps["cs"]["reads"].__setitem__(4, None)))
    assert code == 1 and "reads no row of rope.cs in unit 4" in out


def test_the_ports_are_exactly_the_verifiers(rope, staged):
    public = _stage(staged, "honest-3")
    code, out = _statement(rope, public, _own(staged, "own-root", lambda ps: ps["cs"].__setitem__("root", "00" * 64)))
    assert code == 1 and "does not open rope.cs's registered root" in out
    code, out = _statement(rope, public, _own(staged, "own-x", lambda ps: ps.__setitem__("x", {**ps["cs"], "value": "rope.x"})))
    assert code == 1 and "commits port x's rows of rope.x afresh" in out
    code, out = _statement(rope, public, _own(staged, "own-y", lambda ps: ps.__setitem__("y", ps["cs"])))
    assert code == 1 and "y is not an input port" in out
    x_entry = {"value": "rope.x", "positions": list(range(8)), "paths": [""] * 8}
    code, out = _statement(rope, _stage(staged, "extra-x", extra={"x": x_entry}), staged.dir / "own.json")
    assert code == 1 and "names port x registered, and the verifier's registered reads do not" in out
    bad = staged.dir / "bad.json"
    for doc, want in (({"format": "verity/registered-reads/v1", "ports": {}}, "not a verity/registered-reads/v0 file"),
                      ({**staged.own, "ports": {}}, "no port")):
        bad.write_text(json.dumps(doc))
        code, out = _statement(rope, public, bad)
        assert code == 2 and want in out, out


def test_a_session_verified_with_registered_reads_is_held_to_them(rope, staged):
    """`verify` builds every table's statement through the same check: the fixture's recorded session, proved over a
    statement with no registered reads (accepted without `--registered`, `test_lean_session_tables.py`), is refused once
    the verifier reads cs as registered."""
    session = sorted(p for p in (rope / "sessions" / "j2").iterdir() if p.is_dir())[0]
    p = subprocess.run([str(BIN), "verify", "--statement", "verity/flock-circuit@967b8d06+seed-injection", "--circuit",
                        str(rope / "stage" / "circuit.txt"), "--public", str(rope / "stage" / "pub-8.bin"), "--coins", "seed",
                        "--session-tables", "2", "--registered", str(staged.dir / "own.json"), "--session", str(session)],
                       capture_output=True, text=True, check=False)
    (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    assert p.returncode == 1 and line["accepted"] is False and "names no registered rows" in line["why"], line
    assert line["registered"] == {"cs": {"value": "rope.cs", "root": staged.reg.root.hex()}}
