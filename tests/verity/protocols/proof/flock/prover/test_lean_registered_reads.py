"""Registered reads (`Flock.HmRow`, `verity/protocols/proof/spot/verifier/one_stage/registered.py`): a statement whose port reads
rows of a registered value (RoPE's `cs`, rows of the program's `cs_table`, committed once at registration) opens every
table row against the verifier's own registered root, at the position the verifier's own program reads, with the units
derived from that program, or the Lean verifier refuses it. The statements are staged here over a hidden-output RoPE d64
circuit (`HmOut`, the default `verity/flock-circuit`) composed for the `RegisteredRoPE` program and its partition object of
one block; the verifier's file is Flock's `Registered.FORMAT` (`registered.READS_V2`), whose positions C gives (`statement`: setup, no proof; `verify`: setup of each table,
with no proofs). Upstream's proved session over the public-output statement (`FIXTURE`,
`pod/97-registered-reads.sh`) is held to the same reads. Composing the 56 MB circuit takes about 30 s and each call sets it up (about 8 s), so the
module is slow; it catches what the Python reference (`tests/verity/protocols/proof/spot/verifier/test_registered.py`) can't: the
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
from verity.circuits import partition_object as PO
from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array
from verity_flock import circuit as FC
from verity.protocols.proof.spot.verifier.one_stage import partition as P, registered as G
from verity.commitments import registered_value as RV

LEAN = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proof" / "flock" / "verifier"
VERIFIER = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proof" / "flock" / "prover" / "agreement"
BIN = LEAN / ".lake" / "build" / "bin" / "verity-verify-proof"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "registered-reads"
STATEMENT = "verity/flock-circuit"
UPSTREAM_STATEMENT = "verity/flock-circuit@967b8d06"
VALUE = "cs_table"
#: the program's: the row of `cs_table` each of its 8 RoPEHead Calls reads
POSITIONS = (11, 5, 3, 1, 0, 9, 8, 4)
TABLE_ROWS = 12
#: the units' outputs, committed once in the template's output field's tree (`Flock.Registered.fieldTree`), row u unit u's
OUT_VALUE = "RoPEHead_v1{D=64}/o0"
OUT_POSITIONS = tuple(range(len(POSITIONS)))
OUT_ROWS = len(POSITIONS)

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def _program(registered: tuple[str, ...] = ("x", VALUE)) -> tuple[Program, dict]:
    """The verifier's own program and query: `registered` its registered inputs, the root nodes whose rows its registered
    reads derive (`Flock.Registered.regRow`)."""
    from verity_flock import subcircuit_templates as TM

    template = TM.rope_head(64).definition()
    (_, xt), (_, ct) = template.signature()[0]
    ret = template.signature()[1]

    def sig(S):
        return (("x", Array(len(POSITIONS), xt)), (VALUE, Array(TABLE_ROWS, ct))), Array(len(POSITIONS), ret)

    def body(B, S, x, cs):
        return coll_concat(sig(S)[1], [B.call(template, x[i], cs[p]) for i, p in enumerate(POSITIONS)])

    return (Program(bind(CompositeDefinition("RegisteredRoPE", 0, (), sig, body, register=False)), registered=registered),
            P.template_instance_query(template))


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    return stage(tmp_path_factory.mktemp("registered"))


def stage(d: Path) -> SimpleNamespace:
    """The hidden-output circuit bound to the program and its partition, the registered cs table, x rows and output
    rows, the verifier's own reads file (its program's reads of cs and x, and the output rows), and the statement's
    bindings, under `d`."""
    from verity_flock import subcircuit_templates as TM
    from synthetic_sets import synthetic_set
    from verity_flock.input_sets import InputSet

    subprocess.run(["lake", "build", "verity-verify-proof"], cwd=LEAN, check=True, capture_output=True)
    program, query = _program()
    part = PO.build(program, query, [[(0, len(POSITIONS))]])
    s = InputSet.open(synthetic_set(TM.rope_head(64), len(POSITIONS), 20260928, d / "set")["path"])
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit, partition=P.digest(part), program=P.program_digest(program),
                      statics=FC.set_statics(s))
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    (d / "partition.json").write_text(json.dumps(part))
    assert G.reads(program, query, {VALUE: 64})["cs"] == {"value": VALUE, "rows": list(POSITIONS)}
    rng = np.random.default_rng(20261001)
    reg = RV.Registered.commit(P.program_digest(program), VALUE, rng.integers(0, 1 << 16, (TABLE_ROWS, 64)).tolist())
    x = rng.integers(0, 1 << 16, (8, 64))
    xreg = RV.Registered.commit(P.program_digest(program), "x", x.tolist())
    out = RV.Registered.commit(P.program_digest(program), OUT_VALUE,
                              np.random.default_rng(20261003).integers(0, 1 << 16, (OUT_ROWS, 64)).tolist())
    own = G.reads_file(program, query, {VALUE: 64, "x": 64}, {VALUE: reg.root.hex(), "x": xreg.root.hex()}, positions=False)
    own = {**own, "ports": {**own["ports"], "out": {**out.entry(), "word_bits": 16}}}
    (d / "own.json").write_text(json.dumps(own))
    s = SimpleNamespace(dir=d, text=text, reg=reg, xreg=xreg, own=own, out=out, program=program, x=x,
                        part=part, template=query["params"]["template"])
    s.bound = ["--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
               "--interface", str(_interface(s, "interface"))]
    return s


def _interface(s, name: str, reads=("cs",), verdicts=(), program: str | None = None) -> Path:
    """The window's registered circuit record (`verity/window-circuit/v1`) as `RegBound.declaredOf` reads it: its program
    the partition object's (or `program`), declaring registered, for the statement's template, the ports `reads` (the
    program's registered read of `cs`, as `G.reads` finds it) and the openings' verdicts `verdicts` (an output, as an
    embedding read's ``o1``)."""
    reg = {"params": {}, "tables": [], "reads": [{"reader": s.template, "port": p} for p in reads], "roots": [],
           "openings": [{"reader": s.template, "verdict": v} for v in verdicts]}
    (s.dir / f"{name}.json").write_text(json.dumps({"format": "verity/window-circuit/v1",
                                                     "program": program or s.part["program"], "registered": reg}))
    return s.dir / f"{name}.json"


def _stage(s, name: str, registered=True, salts=None, entry=None, extra=None, outs=None, out_salts=None, cs=None,
           afresh=(), at=POSITIONS) -> Path:
    """The statement, every port registered (`cs` at the program's positions, `x` row i instance i's, `out` the
    `rope_out` rows at `OUT_POSITIONS`), but the ports in `afresh`, whose rows it commits under fresh salts and names no
    registered entry for; `registered=False` names none but `extra`; `at`: the row of `cs` each instance reads."""
    positions, refs = G.table(at)
    rows, reg_salts, e = s.reg.statement(positions)
    rows = rows if cs is None else cs
    _, x_salts, x_e = s.xreg.statement(range(8))
    out_rows, reg_out_salts, out_e = s.out.statement(OUT_POSITIONS)
    sub = json.loads(s.text.split("\n", 2)[1][5:]).get("subcircuit", "rope-head/d64/neox-bf16")
    inputs = SimpleNamespace(name="rope-registered", subcircuit=SimpleNamespace(id=sub))
    reg = {k: v for k, v in {"cs": entry or e, "x": x_e, "out": out_e}.items() if k not in afresh}
    reg = {**reg, **(extra or {})} if registered else extra
    path = s.dir / f"{name}.bin"
    fresh = lambda: [hm96.SHA512.fresh_salt() for _ in range(8)]  # noqa: E731
    FC.write(s.text, inputs, 0, 8, None, path, list(range(8)), share_rows=True,
             outs={"out": np.asarray(out_rows if outs is None else outs, np.uint16)},
             tables={"x": s.x, "cs": np.asarray(rows)}, refs={"x": list(range(8)), "cs": refs},
             salts={"x": fresh() if "x" in afresh else x_salts, "cs": salts or reg_salts,
                    "out": out_salts or (fresh() if "out" in afresh else reg_out_salts)},
             registered=reg)
    return path


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
    assert line["registered"] == {"cs": {"value": VALUE, "root": staged.reg.root.hex()},
                                  "x": {"value": "x", "root": staged.xreg.root.hex()},
                                  "out": {"value": OUT_VALUE, "root": staged.out.root.hex()}}
    assert line["partition_check"]["units"] == "derived"
    assert json.loads(line["identity"])["outputs"].startswith("hidden"), "the statement hides its outputs (`HmOut`)"


def test_registered_reads_need_the_verifiers_own_roots(staged):
    """A statement that names registered reads, checked with no registered roots, is refused (its rows would open nothing)."""
    code, out = _statement(staged, _stage(staged, "no-own"), None)
    assert code == 1 and "the verifier holds no registered roots" in out


def test_registered_reads_need_the_units_derived(staged):
    """A read's position is derived from C, the verifier's program: without it registered reads are refused."""
    code, out = _statement(staged, _stage(staged, "as-stated"), staged.dir / "own.json", staged.bound[:2])
    assert code == 1 and "a port's reads are derived from C, the verifier's program (--program)" in out, out


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
    other = RV.Registered.commit(P.program_digest(staged.program), VALUE,
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
    reads in the verifier's program, C: instance 3 reads row 10, its path true, where C reads row 1."""
    at = (*POSITIONS[:3], 10, *POSITIONS[4:])
    code, out = _statement(staged, _stage(staged, "moved-3", at=at), staged.dir / "own.json")
    assert code == 1 and f"unit 3 reads position 10, and the verifier's program reads {POSITIONS[3]}" in out, out


def test_the_ports_are_exactly_the_verifiers(staged):
    public = _stage(staged, "honest-3")
    code, out = _statement(staged, public, _own(staged, "own-root", lambda ps: ps["cs"].__setitem__("root", "00" * 64)))
    assert code == 1 and f"does not open {VALUE}'s registered root" in out
    code, out = _statement(staged, _stage(staged, "x-afresh", afresh=("x",)), staged.dir / "own.json")
    assert code == 1 and "commits port x's rows of x afresh" in out
    code, out = _statement(staged, public, _own(staged, "own-y", lambda ps: ps.__setitem__("y", ps["cs"])))
    assert code == 1 and "parameter y of root node 2: RoPEHead_v1{D=64} has no parameter y" in out, out
    code, out = _statement(staged, public, _own(staged, "own-no-x", lambda ps: ps.pop("x")))
    assert code == 1 and "names port x registered, and the verifier's registered reads do not" in out
    bad = staged.dir / "bad.json"
    cs = {k: v for k, v in staged.own["ports"]["cs"].items() if k not in ("words", "word_bits")}
    for doc, want in (({"format": G.READS_V1, "ports": {}}, f"not a {G.READS_V2} file"),
                      ({**staged.own, "ports": {"cs": {**cs, "reads": list(POSITIONS)}}}, "no position: C and the partition give them"),
                      ({**staged.own, "ports": {"cs": {**cs, "reads_at": 0}}}, "and no other"),
                      ({**staged.own, "window": 0}, "and no other"),
                      ({**staged.own, "ports": {}}, "no port"),
                      ({**staged.own, "ports": {"cs": {**cs, "words": 64, "word_bits": 8}}}, "leaves of 16 or 32 bits"),
                      ({**staged.own, "ports": {"cs": {**cs, "word_bits": 32}}}, "a row's words when they are 32"),
                      ({**staged.own, "ports": {"cs": {**cs, "words": 0}}}, "at least one to a row")):
        bad.write_text(json.dumps(doc))
        code, out = _statement(staged, public, bad)
        assert code == 2 and want in out, out


def test_every_port_of_the_circuit_is_a_registered_port(staged):
    """`Registered.checkComplete`, which `--registered` runs when the file binds no port (`RegBound.check`): a statement
    that commits a port's rows afresh is refused when the verifier's registered reads leave that port out too (each side
    agreeing, `check` alone would pass), an input port (`x`) or an output row (`out`), so no output's cone reads a row only
    the public file states."""
    for port in ("x", "out"):
        public = _stage(staged, f"{port}-unregistered", afresh=(port,))
        code, out = _statement(staged, public, _own(staged, f"own-no-{port}", lambda ps, port=port: ps.pop(port)))
        assert code == 1 and (f"registered reads (complete): the statement's port {port} is no port of the verifier's "
                              "registered reads") in out, (port, out)


def _bound(s, name: str, drop: tuple[str, ...], public: Path, bound: tuple[str, ...], root: str | None = None) -> Path:
    """The verifier's reads without the ports `drop`, binding the ports `bound` (`G.BOUND`) at the public header's
    frame-v3-sha512 root and domain of their rows (the window's registration's, which the audit holds the header's to
    before any coin), or at `root`."""
    fv = json.loads(public.read_bytes().split(b"\n", 1)[0])["frame_v3"]
    doc = {**s.own, "ports": {p: e for p, e in s.own["ports"].items() if p not in drop},
           **({G.BOUND: {p: {"root": root or fv["roots"][p], "domain": fv["domain_ids"][p]} for p in bound}} if bound else {})}
    (s.dir / f"{name}.json").write_text(json.dumps(doc))
    return s.dir / f"{name}.json"


def test_a_port_the_reads_leave_out_is_bound_to_the_windows_committed_root(staged):
    """`Flock.RegBound`, under the window's registered circuit record declaring only `cs` registered (`staged.bound`'s
    `--interface`): a statement that commits its output rows afresh passes when the verifier's file binds `out` at the
    frame-v3-sha512 root and domain of its rows, and is refused when it binds it at another root, binds a registered
    port too, or leaves it neither registered nor bound."""
    public = _stage(staged, "out-bound", afresh=("out",))
    code, out = _statement(staged, public, _bound(staged, "own-out-bound", ("out",), public, ("out",)))
    assert code == 0, out
    code, out = _statement(staged, public, _bound(staged, "own-out-other", ("out",), public, ("out",), "00" * 64))
    assert code == 1 and ("registered reads (bound): port out's frame-v3-sha512 root or domain in the public file is not the "
                          "window's committed one") in out, out
    code, out = _statement(staged, public, _bound(staged, "own-out-cs-bound", ("out",), public, ("out", "cs")))
    assert code == 1 and "registered reads (bound): port cs is a registered port and bound" in out, out
    code, out = _statement(staged, public, _bound(staged, "own-out-neither", ("out",), public, ()))
    assert code == 1 and ("registered reads (complete): the statement's port out is no port of the verifier's "
                          "registered reads") in out, out


def test_a_port_the_interface_declares_registered_is_never_bound(staged):
    """`RegBound.check` reads which ports are registered from the window's registered circuit record (`--interface`,
    `RegBound.declaredOf`), not from the reads file: where the record declares `out` registered for the statement's
    template (an opening's verdict, as an embedding read's ``o1``), a statement that commits `out` afresh is refused
    under a reads file that binds it, and under one that names it neither registered nor bound, which the record
    declaring only `cs` accepts. A file that binds ports is refused without the record, and with another program's."""
    both = _stage(staged, "out-afresh-declared", afresh=("out",))
    out_bound = _bound(staged, "own-out-bound-declared", ("out",), both, ("out",))
    out_neither = _bound(staged, "own-out-neither-declared", ("out",), both, ())
    declared = [*staged.bound[:4], "--interface", str(_interface(staged, "interface-out", verdicts=("out",)))]
    code, out = _statement(staged, both, out_bound, declared)
    assert code == 1 and ("registered reads (declared): port out is one the window's interface declares registered: it "
                          "reads at a registered root, never a window's") in out, out
    code, out = _statement(staged, both, out_neither, declared)
    assert code == 1 and ("registered reads (complete): the statement's port out is no port of the verifier's registered "
                          "reads") in out, out
    code, out = _statement(staged, both, out_bound)
    assert code == 0, out
    code, out = _statement(staged, both, out_bound, staged.bound[:4])
    assert code == 1 and "the verifier holds no registered interface (--interface)" in out, out
    other = [*staged.bound[:4], "--interface", str(_interface(staged, "interface-other", program="00" * 64))]
    code, out = _statement(staged, both, out_bound, other)
    assert code == 2 and "the record's program is not the partition object's: it is another window's interface" in out, out


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
        assert code == 1 and (f"port cs is a {RV.SCHEMA} row, and {VALUE}'s registered rows are {schema} rows") in out, out


def test_a_registered_output_row_must_be_the_registered_row(staged):
    """An output row (§16.13) is a registered port like an input's (the honest statement's `out`): an instance whose output
    is not the registered row (its salt the registered one), or whose row is committed under a fresh salt, does not open
    the registered root; nor does a row with another row's path, or at another position."""
    rows, _, e = staged.out.statement(OUT_POSITIONS)
    wrong = np.asarray(rows, np.uint16)
    wrong[3, 5] ^= 1
    code, out = _statement(staged, _stage_out(staged, "out-wrong", outs=wrong), staged.dir / "own.json")
    assert code == 1 and f"port out: table row 3 does not open {OUT_VALUE}'s registered root at position {OUT_POSITIONS[3]}" in out, out
    fresh = _stage(staged, "out-fresh", out_salts=[hm96.SHA512.fresh_salt() for _ in range(8)])
    code, out = _statement(staged, fresh, staged.dir / "own.json")
    assert code == 1 and f"port out: table row 0 does not open {OUT_VALUE}'s registered root at position {OUT_POSITIONS[0]}" in out, out
    swapped = {**e, "paths": [e["paths"][1], e["paths"][0], *e["paths"][2:]]}
    moved = {**e, "positions": [OUT_POSITIONS[0] + 1, *OUT_POSITIONS[1:]]}
    for name, entry in (("out-swapped", swapped), ("out-moved", moved)):
        code, out = _statement(staged, _stage_out(staged, name, entry=entry), staged.dir / "own.json")
        assert code == 1 and "port out: table row 0 does not open" in out, (name, out)


def test_a_registered_output_row_is_at_its_units_position(staged):
    """A row that opens the registered root at its own position is refused when C puts the unit at another: unit 2's output
    is the registered row 3, and its leaf in the field's tree is 2."""
    at = (*OUT_POSITIONS[:2], 3, *OUT_POSITIONS[3:])
    code, out = _statement(staged, _stage_out(staged, "out-elsewhere", positions=at), staged.dir / "own.json")
    assert code == 1 and f"port out: unit 2 reads position 3, and the verifier's program reads {OUT_POSITIONS[2]}" in out, out


def test_a_registered_output_row_is_on_both_sides(staged):
    """The statement and the verifier name the same registered ports: an output row the statement names registered and the
    verifier's file does not is refused, and so is one the verifier reads as registered and the statement commits afresh."""
    code, out = _statement(staged, _stage_out(staged, "out-one-side"), _own(staged, "own-no-out", lambda ps: ps.pop("out")))
    assert code == 1 and "names port out registered, and the verifier's registered reads do not" in out, out
    code, out = _statement(staged, _stage(staged, "out-afresh", afresh=("out",)), staged.dir / "own.json")
    assert code == 1 and f"commits port out's rows of {OUT_VALUE} afresh" in out, out


def test_each_table_of_a_session_is_held_to_the_registered_reads(staged):
    """`verify` sets up every table's statement through the same check (`Stmt.setupTables`): with two tables, table 1's sampled statement
    (units 4-7) is refused when unit 5 reads another row than C's, and the honest statement's setup passes, so the
    verifier goes on to read the (empty) session's proofs, which it names by the tables it set up."""
    sess = staged.dir / "session"
    sess.mkdir(exist_ok=True)
    (sess / "session.json").write_text("{}")
    public = _stage(staged, "honest-5")

    def why(public: Path) -> str:
        p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(staged.dir / "circuit.txt"), "--public",
                            str(public), *staged.bound, "--registered", str(staged.dir / "own.json"), "--session-tables", "2",
                            "--session", str(sess)],
                           capture_output=True, text=True, check=False)
        (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        assert line["accepted"] is False
        return line["why"]

    honest = why(public)
    assert honest.startswith("session files: ") and "circuit-00.rep0.bin" in honest, honest
    moved = _stage(staged, "moved-5", at=(*POSITIONS[:5], 10, *POSITIONS[6:]))
    assert why(moved) == f"setup: registered reads: port cs: unit 5 reads position 10, and the verifier's program reads {POSITIONS[5]}"


@pytest.fixture(scope="module")
def rope(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    assert (FIXTURE / "stage" / "program.json").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures {FIXTURE.relative_to(FIXTURE.parents[7])}`")
    root = ci.canonical_tree(FIXTURE, tmp_path_factory.mktemp("store") / "registered-reads")
    subprocess.run(["lake", "build", "verity-verify-proof"], cwd=LEAN, check=True, capture_output=True)
    return root


def test_a_session_verified_with_registered_reads_is_held_to_them(rope, tmp_path):
    """`verify` builds every table's statement through the same check on the public-output statement too: the fixture's
    honest session, proved by upstream over a program whose `cs_table` is a run input of C (no registered input), is
    refused at setup under its own reads file in Flock's `Registered.FORMAT`, which names `cs` as `cs_table`: C commits a
    run input's rows once, in `input/n<w>`. Re-registering it would change its public file, and so its proof, which
    upstream's build made. Upstream's seed-injection build has no zk patch, so the session's plain files are upstream's
    form, without the `zk_level0 = None` tag the deployed prover writes, which `verify` refuses and `agree
    --upstream-plain` reads (§8.1): upstream, reading no registered values, accepts the session, and the Lean verifier's
    checks refuse it at setup."""
    program, _ = _program(registered=())
    assert canonical_json(encode_program(program)) == (rope / "stage" / "program.json").read_bytes()
    bound = ["--partition", str(rope / "stage" / "partition.json"), "--program", str(rope / "stage" / "program.json")]
    session = str(rope / "vectors" / "honest")
    upstream = tmp_path / "upstream-verdicts.json"
    upstream.write_text(json.dumps({session: True}))
    doc = json.loads((rope / "registered" / "own.json").read_text())
    own = tmp_path / "own-v2.json"
    own.write_text(json.dumps({"format": G.READS_V2, "ports": {k: {f: v for f, v in e.items() if f not in ("reads", RV.READS_FROM)}
                                                               for k, e in doc["ports"].items()}}))
    p = subprocess.run([str(BIN), "agree", "--statement", f"{UPSTREAM_STATEMENT}+seed-injection", "--circuit",
                        str(rope / "stage" / "circuit.txt"), "--public", str(rope / "registered" / "pub-8.bin"), *bound,
                        "--registered", str(own), "--upstream-plain", "--upstream-verdicts", str(upstream),
                        "--session", session], capture_output=True, text=True, check=False)
    (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("AGREEMENT ")]
    assert "accepted" not in line and line["upstream"] is True and line["agree"] is False and p.returncode == 1, line
    assert line["check"].startswith("setup: registered reads: port cs: C's node ") and "is a run input" in line["check"], line
