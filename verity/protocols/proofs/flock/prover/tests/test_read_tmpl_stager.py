"""A table read's structural unit is staged on the verifier's template (`read_tmpl`), never on `struct_lower`'s lowering of
the Python Definition, whose Definitions and columns `Flock.ReadNet.check` refuses. The template is ``flock-read-tmpl V H``'s
text parsed, once a shape; it is the unit of the templated statement `UnitStager` writes (its template section) and the unit a
round evaluates, on ports that are its port groups (other ports are refused). A stand-in executable runs the plumbing
without Lean. Where the tree builds the verifier's own executable, a round of honest and wrong reads stages on its template,
each output the IR reference's."""
from __future__ import annotations

import hashlib
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_class_statement_unit_stager import TOY, TOY_ROWS
from test_tmpl_statement import _tmpl
from verity.circuits.boolean import BIT
from verity.circuits.boolean.forms import add
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import bind
from verity.circuits.types import Array
from verity.commitments import row_table
from verity.commitments.gates import embedding as E
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, bit_row
from verity.commitments.windows import interface as IF
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import read_tmpl as RT
from verity_flock import tmpl as T

LEAN = Path(__file__).resolve().parents[2] / "verifier"
MODE = "iv/hybrid/4"
#: a unit whose parameters are a row each, in order: the ports a verifier's template reads (``w`` a row past one compression)
PAIR = bind(definition("ReadTmplStagerPair", 1, lambda C, a, b, w: add(C, a, b, width=16, carry_out=True) + [C.AND(w[0], w[2047])],
                       (("a", Array(16, BIT)), ("b", Array(16, BIT)), ("w", Array(2048, BIT))), Array(18, BIT), register=False))
PAIR_ROWS = [("a", V2, 16, ["a"]), ("b", V2, 16, ["b"]), ("w", V2, 2048, ["w"])]


@pytest.fixture
def fresh():
    RT.template.cache_clear()
    yield
    RT.template.cache_clear()


def _stand_in(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "flock-read-tmpl"
    exe.write_text(f"#!/bin/sh\necho \"$@\" >> {tmp_path / 'args'}\n{body}\n")
    exe.chmod(0o755)
    return exe


def test_the_read_template_is_the_executables_text_parsed_once_a_shape(tmp_path, monkeypatch, fresh):
    """`read_tmpl.template` runs the executable ``V H`` once a shape and parses its text; `read_tmpl.of` asks it for a bound
    ``EmbeddingRead_v1`` at its V and H and for nothing else. A refusal is the executable's own words, and a missing
    executable says how to build it."""
    text = T.write(_tmpl())
    (tmp_path / "t.txt").write_text(text)
    monkeypatch.setattr(RT, "EXE", str(_stand_in(tmp_path, f"cat {tmp_path / 't.txt'}")))
    assert T.write(RT.template(7, 3)) == text
    assert RT.of(E.read(7, 3)) is RT.template(7, 3) and RT.of(TOY) is None and RT.of(PAIR) is None
    assert (tmp_path / "args").read_text() == "7 3\n"
    monkeypatch.setattr(RT, "EXE", str(_stand_in(tmp_path, "echo 'flock-read-tmpl: the check refuses V = 1, H = 1' >&2; exit 1")))
    with pytest.raises(ValueError, match="the check refuses V = 1, H = 1"):
        RT.template(1, 1)
    monkeypatch.setattr(RT, "EXE", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="lake build flock-read-tmpl"):
        RT.template(2, 1)


def test_the_executable_is_the_one_named_else_the_trees_build_else_the_paths(tmp_path, monkeypatch):
    """`read_tmpl.executable` is `EXE` where it is set, else the tree's build (`BUILT`) where it is built, else
    ``flock-read-tmpl`` on the path; with none it says how to build it. `read_tmpl.digest` is its bytes' SHA-512."""
    dirs = {d: tmp_path / d for d in ("named", "built", "path", "empty")}
    for d in dirs.values():
        d.mkdir()
    named, built, on_path = (_stand_in(dirs[d], f"echo {d}") for d in ("named", "built", "path"))
    monkeypatch.setattr(RT, "EXE", None)
    monkeypatch.setattr(RT, "BUILT", dirs["empty"] / "flock-read-tmpl")
    monkeypatch.setenv("PATH", str(dirs["path"]))
    assert RT.executable() == str(on_path)
    monkeypatch.setattr(RT, "BUILT", built)
    assert RT.executable() == str(built)
    monkeypatch.setattr(RT, "EXE", str(named))
    assert RT.executable() == str(named)
    assert RT.digest() == hashlib.sha512(named.read_bytes()).hexdigest() != hashlib.sha512(built.read_bytes()).hexdigest()
    monkeypatch.setattr(RT, "EXE", None)
    monkeypatch.setattr(RT, "BUILT", dirs["empty"] / "flock-read-tmpl")
    monkeypatch.setenv("PATH", str(dirs["empty"]))
    with pytest.raises(FileNotFoundError, match="lake build flock-read-tmpl"):
        RT.executable()


@pytest.mark.slow
def test_a_structural_unit_the_verifier_has_a_template_for_is_staged_on_it(tmp_path, monkeypatch):
    """Where `read_tmpl.of` gives a template, the structural stager's unit is that template, never `struct_lower`'s: its
    template section is its text, and a round's outputs are its evaluation, which the stage holds to the IR reference. On ports
    that are not its port groups (a parameter split across rows) it is refused before anything is written."""
    ports = CS.unit_ports(PAIR, "unit/pair", [(r, ps) for r, _s, _b, ps in PAIR_ROWS])[0]
    t, _lw, _top = CS.SL.lower(PAIR, name="unit/pair", ports=ports)
    given = T.parse(T.write(t))
    monkeypatch.setattr(RT, "of", lambda d: given)
    st = CS.UnitStager(PAIR, "unit/pair", PAIR_ROWS, [(V2, (0,))], layout=MODE, structural=True)
    assert st.low.template is given and FC.template_of(st.text) == T.write(given)
    rng = np.random.default_rng(7)
    rows = {r: rng.integers(0, 1 << 16, (6, bits // 16), dtype=np.uint16) for r, _s, bits, _ps in PAIR_ROWS}
    got = st.stage(rows, tmp_path / "round", b"k" * 32)
    a, b, w = rows["a"][:, 0].astype(int), rows["b"][:, 0].astype(int), rows["w"].astype(int)
    assert [int(o[0]) | int(o[1]) << 16 for o in got["outs"]["o0"]] == list(a + b | (w[:, 0] & w[:, 127] >> 15 & 1) << 17)
    with pytest.raises(ValueError, match="give each parameter a row of its own, in order"):
        CS.UnitStager(TOY, "unit/toy", TOY_ROWS, [(V2, (0,))], layout=MODE, structural=True)


@pytest.fixture(scope="module")
def verifiers(tmp_path_factory):
    """The verifier's ``flock-read-tmpl``, built from this tree's Lean where it has the executable."""
    if not (LEAN / "FlockReadTmpl.lean").exists():
        pytest.skip("this tree's Lean has no flock-read-tmpl")
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-read-tmpl"], cwd=LEAN, check=True, capture_output=True)
    return RT.BUILT


def _reads(V: int, H: int) -> list[tuple]:
    """Every token of a table of V random rows read honestly, then a read under another salt and one at a token past V."""
    rng = random.Random(f"read-tmpl-stager-{V}-{H}")
    dom = CommitmentDomain(rng.randbytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
    rows = [[rng.randrange(1 << 16) for _ in range(H)] for _ in range(V)]
    salts = [rng.randbytes(E.SALT_BYTES) for _ in range(V)]
    table = row_table.commit(dom, [bit_row((r, 16)) for r in rows], salts)

    def path(t: int) -> list[bytes]:
        p = table.path(t)
        return [p[k:k + 64] for k in range(0, len(p), 64)]

    honest = [(t, table.root, dom.domain_id, rows[t], salts[t], path(t)) for t in range(V)]
    return honest + [(0, table.root, dom.domain_id, rows[0], rng.randbytes(E.SALT_BYTES), path(0)),
                     (V, table.root, dom.domain_id, rows[0], salts[0], path(0))]


@pytest.mark.slow
def test_a_read_unit_is_the_verifiers_template_and_stages_a_round_of_reads(verifiers, tmp_path, monkeypatch, fresh):
    """A structural ``EmbeddingRead_v1{V = 5, H = 2}`` on the rows serving commits (`interface.unit_rows`): its unit is the
    template ``flock-read-tmpl 5 2`` writes, the tree's build found with nothing named, byte for byte its template section, and a
    round of honest reads, a read under another salt and one past V stages on it, each read's row and verdict the IR
    reference's (`gates.embedding.reference`)."""
    V, H = 5, 2
    monkeypatch.setattr(RT, "EXE", None)
    assert RT.executable() == str(verifiers)
    text = subprocess.run([str(verifiers), str(V), str(H)], capture_output=True, text=True, check=True).stdout
    in_rows, out_rows = IF.unit_rows(E.read(V, H), None)
    st = CS.UnitStager(E.read(V, H), "unit/read", in_rows, out_rows, layout=MODE, structural=True)
    assert FC.template_of(st.text) == text and T.write(st.low.template) == text
    cases = _reads(V, H)
    rows = {r: np.stack([np.packbits(np.array(E.arguments(V, *c)[g], np.uint8), bitorder="little").view("<u2") for c in cases])
            for g, (r, *_rest) in enumerate(in_rows)}
    got = st.stage(rows, tmp_path / "round", b"k" * 32)
    want = [E.reference(V, *c) for c in cases]
    assert [list(map(int, w)) for w in got["outs"]["o0"]] == [h for h, _ok in want]
    assert [int(w[0]) for w in got["outs"]["o1"]] == [ok for _h, ok in want] == [1] * V + [0, 1]
