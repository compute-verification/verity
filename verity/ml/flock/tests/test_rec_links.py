"""The link table (`verity_flock.rec_links`) and its reference check: the prover's rows, their padding to the public grid and
their commitment as advice, and the mirror held to `Flock.RecLinks.check` (``flock-rec-links``) on every vector, honest
and refused, each refusal code at least once."""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_links as R

from verity.ml.commitments import advice

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-rec-links"


def test_a_link_row_is_its_kind_and_four_u32s_then_zeros():
    row = R.encode(R.Link(1, 2, 70000, 4))
    assert row == struct.pack("<5I", R.KIND_LINK, 1, 2, 70000, 4) + bytes(R.ROW_BYTES - R.FIELD_BYTES)
    assert R.decode(row) == R.Link(1, 2, 70000, 4)
    with pytest.raises(ValueError):
        R.encode(R.Link(2 ** 32, 0, 0, 0))


@pytest.mark.parametrize("n, rows", [(1, 1), (15, 15), (16, 16), (17, 18), (33, 36), (100, 104)])
def test_the_table_is_its_links_then_fillers_on_the_grid(n, rows):
    links = [R.Link(u + 1, 0, u, 0) for u in range(n)]
    t = R.table(links)
    assert len(t) == rows == R.grid(n)
    assert [R.decode(r) for r in t] == links + [None] * (rows - n)
    with pytest.raises(ValueError):
        R.table([])


def test_every_vector_has_its_verdict_and_every_refusal_is_exercised():
    verdicts = [R.verdict(c).split() for c in R.vectors()["cases"]]
    for name, kind, what in verdicts:
        if name.startswith("honest-"):
            assert kind == "ok", name
        else:
            assert (kind, name.startswith(what)) == ("refused", True), name
    assert {what for _, kind, what in verdicts if kind == "refused"} == set(R.CODES)


def test_the_committed_table_opens_row_by_row_and_reveals_only_its_count_and_root():
    links = [R.Link(u + 1, 0, u, 0) for u in range(17)]
    reg = R.register(links)
    n, bits = len(reg.rows), 8 * R.ROW_BYTES
    assert n == R.grid(17) == 18
    for i in range(n):
        o = reg.open(i)
        assert advice.verify(reg.root, n, bits, R.BINDING, o)
        assert R.decode(o.row) == (links[i] if i < 17 else None)
    forged = advice.Opening(3, R.encode(R.Link(9, 0, 2, 0)), reg.salts[3], reg.open(3).path)
    assert not advice.verify(reg.root, n, bits, R.BINDING, forged)
    assert R.register(links).root != reg.root


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_lean_check_agrees_with_the_mirror(tmp_path):
    subprocess.run(["lake", "build", "flock-rec-links"], cwd=LEAN, check=True, capture_output=True)
    v = R.vectors()
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(v))
    out = subprocess.run([str(BIN), str(path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [R.verdict(c) for c in v["cases"]]
