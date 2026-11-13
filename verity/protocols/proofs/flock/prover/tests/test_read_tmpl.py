"""`flock-read-tmpl`, the verifier's `Flock.ReadNet.readTmpl` compiled, writes a template that computes
`gates.embedding.reference`: its text, evaluated a Definition at a time (`tmpl.evaluate`, never unrolled), gives each read's
row and verdict, on every honest read of a small table and on each read with one thing wrong. So the template the prover
stages, which is the one `Flock.ReadNet.check` accepts, computes the read the Python reference states.

It is slow because the template holds an 8,192-bit row segment's Definition at every V and H. That Definition is what a
smaller test can't reach. Two segments (H = 513) take both segment Definitions and the outer hash over them.

`flock-read-tmpl check V H CIRCUIT` decides what the read's binding takes of a circuit file: that it parses as a templated
statement, its unit's template passes `Flock.ReadNet.check V H` and its outputs are its unit's. It refuses a usage error, a
file it can't read, one that isn't a templated statement, and one whose unit is another template. Opt-in
(``FLOCK_READ_SERVED``), it accepts each served read's circuit file at its own (V, H) and refuses it at another."""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import settings
from verity.commitments import row_table as RT
from verity.commitments.gates import embedding as E
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, merkle_depth
from verity.commitments.rowleaf import bit_row
from verity_flock import tmpl as T

LEAN = Path(__file__).resolve().parents[2] / "verifier"
EXE = LEAN / ".lake/build/bin/flock-read-tmpl"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
#: (V, H): depth 1 and one segment; depth 3 with a token bound that isn't a power of two; two segments
SHAPES = [(2, 1), (5, 2), (3, 513)]
USAGE = "usage: flock-read-tmpl V H | flock-read-tmpl check V H CIRCUIT"
#: the served reads' circuit files by name, each with its SHA-512, its art id and the (V, H) it reads at, and the
#: (file, V, H) the check must refuse
SERVED = json.loads((Path(__file__).parent / "fixtures" / "served_reads.json").read_text())
SERVED_DIR = settings.served_reads()
served = pytest.mark.skipif(SERVED_DIR is None,
                            reason="reads the served reads' circuit files, about 340 MB each, from the directory "
                                   "FLOCK_READ_SERVED names; a check takes about 90 s and 12 GB")


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build", "flock-read-tmpl"], cwd=LEAN, check=True, capture_output=True)


def read_tmpl(V: int, H: int) -> subprocess.CompletedProcess:
    return subprocess.run([str(EXE), str(V), str(H)], capture_output=True, text=True)


def check(V: int, H: int, circuit: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(EXE), "check", str(V), str(H), str(circuit)], capture_output=True, text=True)


def _table(rng: random.Random, V: int, H: int):
    """A table of V random rows of H bf16 words, one token a row, under a registered value's dense domain."""
    dom = CommitmentDomain(rng.randbytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
    rows = [[rng.randrange(1 << 16) for _ in range(H)] for _ in range(V)]
    salts = [rng.randbytes(E.SALT_BYTES) for _ in range(V)]
    return dom, rows, salts, RT.commit(dom, [bit_row((r, 16)) for r in rows], salts)


def _cases(rng: random.Random, V: int, H: int) -> list[tuple]:
    """Every token read honestly, then each read with one thing wrong (another token, a row bit, the salt, the root, the
    domain id, a sibling), and tokens outside [0, V)."""
    dom, rows, salts, table = _table(rng, V, H)
    D = merkle_depth(V)

    def path(rank: int) -> list[bytes]:
        p = table.path(rank)
        return [p[k:k + 64] for k in range(0, len(p), 64)]

    out = []
    for t in range(V):
        args = (t, table.root, dom.domain_id, rows[t], salts[t], path(t))
        bad_row = list(rows[t])
        bad_row[rng.randrange(H)] ^= 1 << rng.randrange(16)
        bad_path = list(args[5])
        bad_path[rng.randrange(D)] = rng.randbytes(64)
        out += [args, ((t + 1) % V, *args[1:]), (t, args[1], args[2], bad_row, *args[4:]),
                (*args[:4], rng.randbytes(E.SALT_BYTES), args[5]), (t, rng.randbytes(64), *args[2:]),
                (t, args[1], rng.randbytes(64), *args[3:]), (*args[:5], bad_path)]
    for tok in (V, (1 << 32) - 1, -2, 1 << 31):
        out.append((tok, table.root, dom.domain_id, rows[0], salts[0], path(0)))
    return out


@needs_lake
@pytest.mark.slow
@pytest.mark.parametrize("V,H", SHAPES, ids=[f"V{V}-H{H}" for V, H in SHAPES])
def test_the_template_computes_the_read(built, V, H):
    out = read_tmpl(V, H)
    assert out.returncode == 0, out.stderr
    t = T.parse(out.stdout)
    assert T.ordered(t)["ok"]
    cases = _cases(random.Random(f"read-tmpl-{V}-{H}"), V, H)
    bits = np.array([[b for arg in E.arguments(V, *c) for b in arg] for c in cases], np.uint8)
    got = T.unslice_lanes(T.evaluate(t, T.slice_lanes(bits)), len(cases))
    # the output group is the H row words, then a word whose low bit is ok
    assert got.shape == (len(cases), 16 * H + 16)
    want = [E.reference(V, *c) for c in cases]
    assert any(ok for _, ok in want) and not all(ok for _, ok in want)
    for g, (h, ok) in zip(got, want):
        assert [int(sum(int(b) << j for j, b in enumerate(g[16 * w:16 * w + 16]))) for w in range(H)] == h
        assert int(g[16 * H]) == ok


@needs_lake
@pytest.mark.slow
@pytest.mark.parametrize("V,H", [(1, 1), (2, 0), (1 << 32, 1)])
def test_the_check_refuses_a_read_out_of_range(built, V, H):
    out = read_tmpl(V, H)
    assert out.returncode == 1
    assert out.stderr.strip() == f"flock-read-tmpl: the check refuses V = {V}, H = {H}"


@needs_lake
@pytest.mark.parametrize("args", [[], ["2"], ["2", "x"], ["check", "2", "1"], ["check", "x", "1", "c"], ["2", "1", "c"]])
def test_the_command_refuses_a_usage_error(built, args):
    out = subprocess.run([str(EXE), *args], capture_output=True, text=True)
    assert (out.returncode, out.stdout, out.stderr.strip()) == (2, "", USAGE)


@needs_lake
def test_the_check_refuses_a_file_it_cant_read(built, tmp_path):
    out = check(2, 1, tmp_path / "missing.txt")
    assert out.returncode == 2
    assert out.stderr.startswith(f"flock-read-tmpl: {tmp_path / 'missing.txt'}: ")


@needs_lake
def test_the_check_refuses_a_file_that_isnt_a_templated_statement(built, tmp_path):
    (tmp_path / "circuit.txt").write_text("flock-circuit\n{}\n")
    out = check(2, 1, tmp_path / "circuit.txt")
    assert out.returncode == 1
    assert out.stderr.startswith(f"flock-read-tmpl: {tmp_path / 'circuit.txt'}: it doesn't parse as verity/flock-circuit/tmpl: ")


@needs_lake
@pytest.mark.slow
def test_the_check_refuses_a_templated_statement_of_another_unit(built, tmp_path):
    """The Boolean toy's templated statement parses as one, so it is its unit's template that the check refuses."""
    from test_class_statement_unit_stager import _toy
    from verity_flock import class_statement as CS

    args, rows, _kw = _toy(0)
    CS.UnitStager(*args, layout="iv/hybrid/4").stage(rows, tmp_path, b"k" * 32)
    out = check(2, 1, tmp_path / "circuit.txt")
    assert out.returncode == 1
    assert re.fullmatch(rf"flock-read-tmpl: {re.escape(str(tmp_path / 'circuit.txt'))}: the check refuses its template "
                        r".+ at V = 2, H = 1", out.stderr.strip()), out.stderr


def _served(name: str) -> Path:
    f = SERVED["files"][name]
    path = SERVED_DIR / name
    assert path.is_file(), f"{path}: fetch it with `research data fetch {f['art']} --path circuit.txt`"
    h = hashlib.sha512()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    assert h.hexdigest() == f["sha512"], f"{path} is not {f['art']}"
    return path


@needs_lake
@served
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SERVED["files"]))
def test_a_served_read_holds_the_read_template(built, name):
    f = SERVED["files"][name]
    out = check(f["V"], f["H"], _served(name))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"ok EmbeddingRead_v1(V={f['V']},H={f['H']})"


@needs_lake
@served
@pytest.mark.slow
@pytest.mark.parametrize("name,V,H", [tuple(r) for r in SERVED["refused"]], ids=[f"{n}-V{V}-H{H}" for n, V, H in SERVED["refused"]])
def test_a_served_read_is_refused_at_another_shape(built, name, V, H):
    f, path = SERVED["files"][name], _served(name)
    out = check(V, H, path)
    assert out.returncode == 1
    assert out.stderr.strip() == (f"flock-read-tmpl: {path}: the check refuses its template "
                                  f"EmbeddingRead_v1(V={f['V']},H={f['H']}) at V = {V}, H = {H}")
