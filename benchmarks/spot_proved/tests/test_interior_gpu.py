"""`interior_gpu`'s commit is `verity.ml.windows.rows.commit_ports`'s, byte for byte: every port's root and every row's
``b || c`` under ``hm96-sha512/row/v1``, salted by `rows.salts` under the window's key at the port's index. Its CPU twin is
held to it here; its device, where a GPU is."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.windows import rows as CS

HERE = Path(__file__).resolve().parents[1]
#: rows and words a port: one row, odd and even widths, rows across a block boundary, counts that pad their trees
SHAPES = {"a": (1, 1), "b": (2, 7), "c": (3, 8), "d": (5, 63), "e": (8, 64), "f": (33, 65), "g": (17, 200)}


def _module(monkeypatch):
    spec = importlib.util.spec_from_file_location("spot_interior_gpu", HERE / "interior_gpu.py")
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, m)
    spec.loader.exec_module(m)
    return m


def _case(seed: int):
    rng = np.random.default_rng(seed)
    key = rng.bytes(32)
    index = {p: int(i) for p, i in zip(SHAPES, rng.integers(1, 1 << 40, len(SHAPES)))}
    index["a"] = 0
    domains = {p: CommitmentDomain(rng.bytes(64), -1, RangeIndexedDomain(0, n), hash="sha512") for p, (n, _) in SHAPES.items()}
    rows = {p: rng.integers(0, 1 << 16, (n, w), dtype=np.uint16) for p, (n, w) in SHAPES.items()}
    want = CS.commit_ports(domains, rows, {}, {p: CS.salts(key, index[p], SHAPES[p][0]) for p in SHAPES})
    return key, index, domains, rows, want


def test_the_cpu_twin_commits_as_commit_ports(monkeypatch) -> None:
    ig = _module(monkeypatch)
    key, index, domains, rows, want = _case(1)
    assert ig.commit_ports(domains, rows, key, index, device=False) == want


def test_rows_added_a_step_at_a_time_commit_as_the_whole_port(monkeypatch) -> None:
    ig = _module(monkeypatch)
    key, index, domains, rows, want = _case(2)
    it = ig.Interior({p: (domains[p], w) for p, (_, w) in SHAPES.items()}, key, index, device=False)
    for lo in range(0, max(n for n, _ in SHAPES.values()), 4):
        for p, r in rows.items():
            if lo < len(r):
                it.add(p, lo, r[lo:lo + 4])
        it.flush()
    assert it.finish() == want


def test_another_key_or_index_changes_every_string(monkeypatch) -> None:
    ig = _module(monkeypatch)
    key, index, domains, rows, want = _case(3)
    for k, i in ((bytes(32), index), (key, {**index, "g": index["g"] ^ 1 << 33})):
        roots, strings = ig.commit_ports(domains, rows, k, i, device=False)
        changed = [p for p in SHAPES if strings[p] != want[1][p]]
        assert changed == (list(SHAPES) if k != key else ["g"]) and all(roots[p] != want[0][p] for p in changed)


def test_the_device_commits_as_commit_ports(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ig = _module(monkeypatch)
    key, index, domains, rows, want = _case(4)
    dev = {p: torch.from_numpy(r.view(np.int16)).cuda() for p, r in rows.items()}
    assert ig.commit_ports(domains, dev, key, index) == want
