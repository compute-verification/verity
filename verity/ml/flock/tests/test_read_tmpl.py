"""`flock-read-tmpl`, the verifier's `Flock.ReadNet.readTmpl` compiled, writes a template that computes
`gates.embedding.reference`: its text, evaluated a Definition at a time (`tmpl.evaluate`, never unrolled), gives each read's
row and verdict, on every honest read of a small table and on each read with one thing wrong. So the template the prover
stages, which is the one `Flock.ReadNet.check` accepts, computes the read the Python reference states.

It is slow because the template holds an 8,192-bit row segment's Definition at every V and H. That Definition is what a
smaller test can't reach. Two segments (H = 513) take both segment Definitions and the outer hash over them."""
from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from verity.ml.commitments import row_table as RT
from verity.ml.commitments.gates import embedding as E
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth
from verity.ml.commitments.rowleaf import bit_row
from verity_flock import tmpl as T

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
EXE = LEAN / ".lake/build/bin/flock-read-tmpl"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
#: (V, H): depth 1 and one segment; depth 3 with a token bound that isn't a power of two; two segments
SHAPES = [(2, 1), (5, 2), (3, 513)]


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build", "flock-read-tmpl"], cwd=LEAN, check=True, capture_output=True)


def read_tmpl(V: int, H: int) -> subprocess.CompletedProcess:
    return subprocess.run([str(EXE), str(V), str(H)], capture_output=True, text=True)


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
    assert got.shape == (len(cases), 16 * H + 1)
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
