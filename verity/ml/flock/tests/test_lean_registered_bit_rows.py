"""Registered reads of bit rows: a statement whose ports are `hm96-sha512/row/v2` and `hm96-sha512/row-seg/v1` rows reads
registered values committed in each port's own format (`registered.Registered.commit`'s `schema`), and the Lean verifier
opens each table row under its port's leaf schema, refusing a registration in another format, under another format's
root or of another width. The statement is `test_lean_registered_reads.py`'s RoPE d64 one, composed with every row a bit
row of its words' bits (`compose`'s `port_bits`) and cs segmented, and both input ports registered: x (unit i reads row
i) as a `row/v2` value and cs (`cs_table`) as a `row-seg/v1` one. Composing the circuit takes about 30 s and each call
sets it up, so the module is slow; it catches what `tests/test_registered.py` can't: the executable verifier's own
checks of each format."""
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


def _stage(s, name: str, regs: dict | None = None) -> Path:
    """The statement whose x and cs tables are the registered rows each unit reads, under their registered salts and paths
    (`regs`: these registrations in place of the fixture's)."""
    regs = regs or s.regs
    tabs, salts, entries, refs = {}, {}, {}, {}
    for port, value, reads in (("x", "x", range(N)), ("cs", VALUE, POSITIONS)):
        positions, refs[port] = G.table(reads)
        rows, salts[port], entries[port] = regs[value].statement(positions)
        tabs[port] = np.asarray(rows, np.uint16)
    sub = json.loads(s.text.split("\n", 2)[1][5:]).get("subcircuit", "rope-head/d64/neox-bf16")
    inputs = SimpleNamespace(name="rope-registered-bit-rows", subcircuit=SimpleNamespace(id=sub))
    out = s.dir / f"{name}.bin"
    FC.write(s.text, inputs, 0, N, None, out, None, list(range(N)), share_rows=True,
             outs={"out": np.zeros((N, WORDS), np.uint16)}, tables=tabs, refs=refs,
             salts={**salts, "out": [hm96.SHA512.fresh_salt() for _ in range(N)]}, registered=entries)
    return out


def _statement(s, own: Path) -> tuple[int, str]:
    args = [str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(s.dir / "circuit.txt"), "--public", str(s.public)]
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
