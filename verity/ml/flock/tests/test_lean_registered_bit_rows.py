"""Registered reads of bit rows: a statement whose ports are `hm96-sha512/row/v2` and `hm96-sha512/row-seg/v1` rows reads
registered values committed in each port's own format (`registered.Registered.commit`'s `schema`), and the Lean verifier
opens each table row under its port's leaf schema, refusing a registration in another format, under another format's
root or of another width. The statement is `test_lean_registered_reads.py`'s RoPE d64 one, composed with every row a bit
row of its words' bits (`compose`'s `port_bits`) and cs segmented, and both input ports registered: x (unit i reads row
i) as a `row/v2` value and cs (`cs_table`) as a `row-seg/v1` one. A second circuit has both ports `row/v2` and reads
two rows of one value a unit, one at each port (`one_value`), as H1's tile reads R rows of A. Composing a circuit takes
about 30 s and each call sets it up, so the module is slow; it catches what `tests/test_registered.py` can't: the
executable verifier's own checks of each format, and of several rows of one value a unit."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.commitments import hm96
from verity.ml.circuits.codec import canonical_json, encode_program
from verity_flock import circuit as FC
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G

from test_lean_registered_reads import BIN, LEAN, POSITIONS, STATEMENT, TABLE_ROWS, VALUE, _program

N = len(POSITIONS)
WORDS = 64
SCHEMAS = {"x": G.SCHEMA_V2, VALUE: G.SCHEMA_SEG}
#: the one value both input ports read rows of (`one_value`)
ROWS = "rope.rows"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The bit-row circuit bound to the program and its partition, x and cs_table registered in their ports' formats, the
    verifier's own reads file and the honest statement."""
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    d = tmp_path_factory.mktemp("registered-bit-rows")
    program, query = _program()
    part = P.partition(program, query)
    s = InputSet.open(generate(TM.rope_head(64), N, 20260928, d / "set")["path"])
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit, partition=P.digest(part), program=P.program_digest(program),
                      statics=FC.set_statics(s), port_bits=[16 * WORDS, 16 * WORDS], seg_ports=frozenset({"cs"}))
    meta = json.loads(text.split("\n", 2)[1][5:])
    assert [tuple(r) for r in meta["row_schemas"][:2]] == [(G.SCHEMA_V2, 1024), (G.SCHEMA_SEG, 1024)]
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    (d / "partition.json").write_text(json.dumps(part))
    values = {VALUE: WORDS, "x": WORDS}
    assert G.reads(program, query, values) == {"cs": {"value": VALUE, "rows": list(POSITIONS)},
                                                "x": {"value": "x", "rows": list(range(N))}}
    rng = np.random.default_rng(20261020)
    prog = P.program_digest(program)
    regs = {v: G.Registered.commit(prog, v, rng.integers(0, 1 << 16, (rows, WORDS)).tolist(), schema=SCHEMAS[v])
            for v, rows in ((VALUE, TABLE_ROWS), ("x", N))}
    roots = {v: r.root.hex() for v, r in regs.items()}
    own = G.reads_file(program, query, values, roots, schemas=SCHEMAS)
    assert own["ports"]["x"]["row_schema"] == G.SCHEMA_V2 and own["ports"]["cs"]["row_schema"] == G.SCHEMA_SEG
    (d / "own.json").write_text(json.dumps(own))
    st = SimpleNamespace(dir=d, text=text, regs=regs, own=own, bound=["--partition", str(d / "partition.json"),
                                                                      "--program", str(d / "program.json")])
    st.public = _stage(st, "honest")
    return st


@pytest.fixture(scope="module")
def one_value(tmp_path_factory):
    """Two rows of one registered value a unit (H1's A, a tile's R rows at ports `a0 .. a{R-1}`): the circuit with x and
    cs both `row/v2` rows, and one value `ROWS` of N + TABLE_ROWS rows whose row u unit u reads at x and row
    N + POSITIONS[u] at cs, each port its own registered port of the value; the verifier's reads file and the honest
    statement."""
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    d = tmp_path_factory.mktemp("registered-rows-of-one-value")
    program, query = _program()
    part = P.partition(program, query)
    s = InputSet.open(generate(TM.rope_head(64), N, 20260928, d / "set")["path"])
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit, partition=P.digest(part), program=P.program_digest(program),
                      statics=FC.set_statics(s), port_bits=[16 * WORDS, 16 * WORDS])
    assert [tuple(r) for r in json.loads(text.split("\n", 2)[1][5:])["row_schemas"][:2]] == [(G.SCHEMA_V2, 1024)] * 2
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    (d / "partition.json").write_text(json.dumps(part))
    rng = np.random.default_rng(20261021)
    prog, n = P.program_digest(program), N + TABLE_ROWS
    regs = {name: G.Registered.commit(prog, value, rng.integers(0, 1 << 16, (n, WORDS)).tolist(), schema=G.SCHEMA_V2)
            for name, value in (("rows", ROWS), ("again", ROWS), ("other", "other"))}
    reads = {"x": list(range(N)), "cs": [N + p for p in POSITIONS]}
    reg = regs["rows"]
    port = {"value": ROWS, "root": reg.root.hex(), "leaves": n, "words": WORDS, "word_bits": 16,
            "domain": reg.domain.domain_id.hex(), "row_schema": G.SCHEMA_V2}
    own = {"format": G.READS, "ports": {p: {**port, "reads": r} for p, r in reads.items()}}
    (d / "own.json").write_text(json.dumps(own))
    st = SimpleNamespace(dir=d, text=text, regs=regs, reads=reads, own=own,
                         bound=["--partition", str(d / "partition.json"), "--program", str(d / "program.json")])
    st.public = _write(st, "honest", {p: (reg, r) for p, r in reads.items()})
    return st


def _stage(s, name: str, regs: dict | None = None) -> Path:
    """The statement whose x and cs tables are the registered rows each unit reads, under their registered salts and paths
    (`regs`: these registrations in place of the fixture's)."""
    regs = regs or s.regs
    return _write(s, name, {"x": (regs["x"], range(N)), "cs": (regs[VALUE], POSITIONS)})


def _write(s, name: str, reads: dict) -> Path:
    """The statement whose table of each port is the rows of `reads[port]`'s registration at its positions, each unit
    reading its own, under their registered salts and paths."""
    tabs, salts, entries, refs = {}, {}, {}, {}
    for port, (reg, rows_read) in reads.items():
        positions, refs[port] = G.table(rows_read)
        rows, salts[port], entries[port] = reg.statement(positions)
        tabs[port] = np.asarray(rows, np.uint16)
    sub = json.loads(s.text.split("\n", 2)[1][5:]).get("subcircuit", "rope-head/d64/neox-bf16")
    inputs = SimpleNamespace(name="rope-registered-bit-rows", subcircuit=SimpleNamespace(id=sub))
    out = s.dir / f"{name}.bin"
    FC.write(s.text, inputs, 0, N, None, out, None, list(range(N)), share_rows=True,
             outs={"out": np.zeros((N, WORDS), np.uint16)}, tables=tabs, refs=refs,
             salts={**salts, "out": [hm96.SHA512.fresh_salt() for _ in range(N)]}, registered=entries)
    return out


def _statement(s, own: Path, public: Path | None = None) -> tuple[int, str]:
    args = [str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(s.dir / "circuit.txt"),
            "--public", str(public or s.public)]
    p = subprocess.run(args + s.bound + ["--registered", str(own)], capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def _own(s, name: str, edit) -> Path:
    doc = json.loads(json.dumps(s.own))
    edit(doc["ports"])
    (s.dir / f"{name}.json").write_text(json.dumps(doc))
    return s.dir / f"{name}.json"


def test_bit_row_registered_statement_is_accepted(staged):
    code, out = _statement(staged, staged.dir / "own.json")
    assert code == 0, out
    line = json.loads(out.splitlines()[0])
    assert line["registered"] == {"cs": {"value": VALUE, "root": staged.regs[VALUE].root.hex()},
                                  "x": {"value": "x", "root": staged.regs["x"].root.hex()}}
    assert line["partition_check"]["units"] == "derived"


def test_a_registration_in_another_format_is_refused(staged):
    """Each port is checked in its own format: a reads file naming x's rows `row/v1` (no `row_schema`) or `row-seg/v1`, or
    cs's `row/v2`, is refused before any row is opened."""
    for port, schema, value in (("x", None, "x"), ("x", G.SCHEMA_SEG, "x"), ("cs", G.SCHEMA_V2, VALUE), ("cs", None, VALUE)):
        def edit(ps, port=port, schema=schema):
            ps[port].pop("row_schema")
            ps[port].update({} if schema is None else {"row_schema": schema})
        code, out = _statement(staged, _own(staged, f"own-{port}-{schema and schema[-6:].replace('/', '-')}", edit))
        want = f"port {port} is a {SCHEMAS[value]} row, and {value}'s registered rows are {schema or G.SCHEMA} rows"
        assert code == 1 and want in out, (port, schema, out)


def test_a_root_of_another_format_or_value_is_refused(staged):
    """The rows' leaves are framed under their port's schema: the root of the same rows and salts under another format's
    leaves (the reads file naming the port's format) opens nothing, nor does another value's root."""
    for port, value in (("x", "x"), ("cs", VALUE)):
        reg = staged.regs[value]
        for schema in G.SCHEMAS:
            if schema == reg.schema:
                continue
            other = G.Registered.commit(reg.program, value, reg.rows, reg.salts, reg.word_bits, schema)
            code, out = _statement(staged, _own(staged, f"own-root-{port}-{schema[-6:].replace('/', '-')}",
                                                lambda ps, port=port, other=other: ps[port].update(root=other.root.hex())))
            assert code == 1 and f"port {port}: table row 0 does not open {value}'s registered root" in out, (port, out)
    code, out = _statement(staged, _own(staged, "own-root-swapped", lambda ps: ps["x"].update(root=ps["cs"]["root"])))
    assert code == 1 and "port x: table row 0 does not open x's registered root" in out, out


def test_a_bit_row_registration_of_another_width_is_refused(staged):
    """A bit row is exactly its registered leaves' bits, `words * word_bits`: x's and cs's rows are 1024 bits, which 64
    16-bit leaves fill and so do 32 32-bit ones; 63 or 65 16-bit leaves, or 64 32-bit ones, are refused (no whole-block
    padding, unlike a `row/v1` row)."""
    for name, width, want in (("own-w32x32", {"words": 32, "word_bits": 32}, None),
                              ("own-w63", {"words": 63}, "rows of 63 16-bit leaves"),
                              ("own-w65", {"words": 65}, "rows of 65 16-bit leaves"),
                              ("own-w64x32", {"words": 64, "word_bits": 32}, "rows of 64 32-bit leaves")):
        for port, value in (("x", "x"), ("cs", VALUE)):
            code, out = _statement(staged, _own(staged, f"{name}-{port}", lambda ps, port=port, width=width: ps[port].update(width)))
            if want is None:
                assert code == 0, (name, port, out)
            else:
                assert code == 1 and f"port {port}'s rows are 1024 bits (64 u16 words), not {value}'s registered {want}" in out, \
                    (name, port, out)


def test_two_rows_of_one_value_a_unit_are_accepted(one_value):
    """Each unit reads two rows of the one value, row u at x and row N + POSITIONS[u] at cs: each port opens the value's
    root at its own positions."""
    code, out = _statement(one_value, one_value.dir / "own.json")
    assert code == 0, out
    root = one_value.regs["rows"].root.hex()
    assert json.loads(out.splitlines()[0])["registered"] == {p: {"value": ROWS, "root": root} for p in ("x", "cs")}


def test_a_row_at_the_wrong_index_is_refused(one_value):
    """A row of the value at another index than the one the verifier's program reads for the unit at that port: the
    verifier's reads shifted by one row, or the two ports' reads swapped (the honest statement, each of its rows still
    opening the root at its stated position). The verifier checks its ports in name order, cs first."""
    r = one_value.reads
    for name, edit, want in (
            ("own-shifted", lambda ps: ps["x"].update(reads=r["x"][1:] + r["x"][:1]),
             "port x: unit 0 reads position 0, and the verifier's program reads 1"),
            ("own-swapped", lambda ps: (ps["x"].update(reads=r["cs"]), ps["cs"].update(reads=r["x"])),
             f"port cs: unit 0 reads position {N + POSITIONS[0]}, and the verifier's program reads 0")):
        code, out = _statement(one_value, _own(one_value, name, edit))
        assert code == 1 and want in out, (name, out)
    swapped = _write(one_value, "swapped", {"x": (one_value.regs["rows"], r["cs"]), "cs": (one_value.regs["rows"], r["x"])})
    code, out = _statement(one_value, one_value.dir / "own.json", swapped)
    assert code == 1 and f"port cs: unit 0 reads position 0, and the verifier's program reads {N + POSITIONS[0]}" in out, out


def test_a_row_of_another_value_is_refused(one_value):
    """cs's rows from another commitment: another tree under the value's own name and domain opens another root, and
    another value's rows are named as that value's."""
    rows, again, other = (one_value.regs[k] for k in ("rows", "again", "other"))
    for name, reg, want in (("again", again, f"port cs: table row 0 does not open {ROWS}'s registered root at position "
                                                f"{N + POSITIONS[0]}"),
                            ("other", other, f"port cs reads {ROWS}, not another value")):
        public = _write(one_value, f"cs-{name}", {"x": (rows, one_value.reads["x"]), "cs": (reg, one_value.reads["cs"])})
        code, out = _statement(one_value, one_value.dir / "own.json", public)
        assert code == 1 and want in out, (name, out)


def test_a_reads_file_naming_a_port_twice_is_refused(one_value):
    """A JSON parser keeps one of two equal keys: a file naming cs twice, the first under another root, would read as the
    honest file. flock-verify refuses the text, naming the key, as it does a port's key named twice: exit 2, a verifier
    input it can't read, as any malformed `--registered` file."""
    good = {p: json.dumps(e) for p, e in one_value.own["ports"].items()}
    bad = json.dumps({**one_value.own["ports"]["cs"], "root": one_value.regs["again"].root.hex()})
    for name, ports, key in (("cs-twice", f'"cs": {bad}, "cs": {good["cs"]}, "x": {good["x"]}', '"cs"'),
                             ("root-twice", f'"cs": {good["cs"][:-1]}, "root": "{"00" * 64}"}}, "x": {good["x"]}', '"root"')):
        own = one_value.dir / f"{name}.json"
        own.write_text(f'{{"format": "{G.READS}", "ports": {{{ports}}}}}')
        assert json.loads(own.read_text())["ports"].keys() == {"cs", "x"}
        code, out = _statement(one_value, own)
        assert code == 2 and f"registered reads: an object of the file names the key {key} twice" in out, (name, out)
    code, out = _statement(one_value, one_value.dir / "own.json")
    assert code == 0, out


def test_a_circuit_naming_a_port_twice_is_refused(one_value, monkeypatch):
    """A registered read names its port by name, so under `--registered` a circuit naming a port twice is refused: x and cs
    are both 64-word `row/v2` rows, and the circuit with cs renamed x (its META alone; the gates as they are) is refused by
    the statement writer and, staged with the honest port list, by flock-verify, whose hidden-output parse refuses a row
    named twice (`HmOut.checkRows`) before the registered check's own refusal (`Registered.namesOnce`)."""
    head, line, rest = one_value.text.split("\n", 2)
    ports = json.loads(line[5:])["ports"]
    old = next(t for t in (json.dumps(ports), json.dumps(ports, separators=(",", ":"))) if t in line)
    twice = json.dumps([["x" if p[0] == "cs" else p[0], *p[1:]] for p in ports], separators=(", ", ": ") if ", " in old else (",", ":"))
    text = "\n".join([head, line.replace(old, twice, 1), rest])
    d = one_value.dir / "twice"
    d.mkdir()
    (d / "circuit.txt").write_text(text)
    st = SimpleNamespace(**{**vars(one_value), "dir": d, "text": text})
    want = "registered reads: the circuit names port x twice, and a registered read names one port"
    with pytest.raises(ValueError, match=want):
        _write(st, "writer", {p: (one_value.regs["rows"], r) for p, r in one_value.reads.items()})
    honest = json.loads(one_value.text.split("\n", 2)[1][5:])
    real = FC.write
    monkeypatch.setattr(FC, "write", lambda *a, **k: real(*a, **{**k, "meta": honest}))
    public = _write(st, "staged", {p: (one_value.regs["rows"], r) for p, r in one_value.reads.items()})
    code, out = _statement(st, one_value.dir / "own.json", public)
    assert code == 1 and out.startswith("setup: the rows are not the input rows") and out.endswith("each name once"), out
