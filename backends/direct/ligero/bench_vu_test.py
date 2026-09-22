"""``bench_vu``'s vectorised loader (``vu.marshal_ids``) builds exactly the arrays the tuple path (``prove_vus``) builds,
including padding and recycled tier ids; the store declaration parses ``--dump-reps``.  CPU only, no fixtures: a small
synthetic ``VuTier`` in memory (the operand arrays of ``bench-instances/v1`` are built, not committed).

    uv run pytest backends/direct/ligero/bench_vu_test.py
"""
from __future__ import annotations

from array import array

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_numerical.bench import instances

from .chain import Layout
from .vu import STEP, STEPS, _pad_unit, _vu_data, marshal_ids, vu_units


def _tier(n: int, seed: int = 3) -> instances.VuTier:
    rng = np.random.default_rng(seed)
    x = array("H", rng.integers(0, 2**16, n * STEP * STEPS, dtype=np.uint16).tolist())
    w = array("H", rng.integers(0, 2**16, n * STEP * STEPS, dtype=np.uint16).tolist())
    y = array("H", rng.integers(0, 2**16, n, dtype=np.uint16).tolist())
    acc = array("I", rng.integers(0, 2**32, n * STEPS, dtype=np.uint32).tolist())
    return instances.VuTier("vu-k1536", n, STEP * STEPS, x, w, y, acc, [[i] for i in range(n)], ["id"])


def _tuple_path(vus, lay: Layout):
    """The arrays ``ChainRunner.prove_vus`` marshals before its timer (copied, not imported: that is the contract)."""
    units, y16 = [], torch.zeros(lay.l, dtype=torch.int64)
    for v, (x, w, accs, y) in enumerate(vus):
        units += vu_units(x, w, accs)
        y16[STEPS * v + STEPS - 1] = y
    while len(units) < lay.l:
        units.append(_pad_unit())
    a = np.asarray([u[1] for u in units], dtype=np.int64)
    b = np.asarray([u[2] for u in units], dtype=np.int64)
    c_in, y_out = np.asarray([u[0] for u in units], dtype=np.int64), np.asarray([u[3] for u in units], dtype=np.int64)
    return a, b, c_in, y_out, y16


@pytest.mark.parametrize("n_vus,l", [(1, 256), (2, 256), (5, 512), (10, 1024)])
def test_marshal_ids_matches_tuple_path(n_vus, l):
    vu = _tier(7)
    ids = [i % vu.n for i in range(11, 11 + n_vus)]          # crosses the tier boundary: recycled ids, as bench_vu does
    lay = Layout(l, n_vus, STEPS)
    got = marshal_ids(vu, ids, lay)
    want = _tuple_path([_vu_data(vu, i) for i in ids], lay)
    for g, w, name in zip(got, want, ("a", "b", "c_in", "y_out", "y16")):
        g, w = np.asarray(g), np.asarray(w)
        assert g.dtype == np.int64 and g.shape == w.shape, name
        assert np.array_equal(g, w), name
    a, b, c_in, y_out, y16 = got
    assert a.shape == (l, STEP) and c_in.shape == (l,) and (a[n_vus * STEPS:] == 0).all() and (y_out[n_vus * STEPS:] == 0).all()
    assert (c_in[::STEPS][:n_vus] == 0).all()                 # every VU starts from c_0 = +0
    assert int(y16[STEPS - 1]) == vu.y[ids[0]] and int((y16 != 0).sum()) <= n_vus


def test_marshal_ids_rejects_layout_mismatch():
    vu = _tier(3)
    with pytest.raises(ValueError):
        marshal_ids(vu, [0, 1], Layout(256, 3, STEPS))
    with pytest.raises(ValueError):
        marshal_ids(vu, [0, 1, 2, 0], Layout(256, 4, STEPS))   # 4 x 96 > 256 columns


def test_tool_parse_dump_reps():
    pytest.importorskip("research.store.tool")            # tools/research (the store declaration's only dependency)
    from .tool import parse

    argv = ["python", "backends/direct/ligero/bench_result.py", "bench-vu", "--zk", "--mode", "fiat-shamir", "--batch", "131072",
            "--total-vus", "1048576", "--reps", "3", "--dump-reps", "1", "--verifier", "/workspace/bin/ligero-verify", "--threads", "16"]
    out = parse(argv)
    assert out["cmd"] == "bench-vu" and out["dump_reps"] == 1 and out["total_vus"] == 1048576 and out["batch"] == 131072
    assert "_unknown" not in out
    assert "dump_reps" not in parse(["python", "-m", "backends.direct.ligero.run", "bench", "--batch", "4096"])
