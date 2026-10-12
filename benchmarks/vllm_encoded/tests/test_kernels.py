import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kernels as K


def sketched(nbs: int, nchunks: int, sb: int, t: int, seed: int) -> K.TensorLinear:
    """A TensorLinear's sketch geometry over a random store on the CPU (no library: only `sketch_vectors` reads it)."""
    tlib = object.__new__(K.TensorLib)
    tlib.sb, tlib.sketch = sb, {"t": t}
    lin = object.__new__(K.TensorLinear)
    lin.tlib, lin.n, lin.nchunks = tlib, 16 * nbs, nchunks
    g = torch.Generator().manual_seed(seed)
    lin.store = torch.randint(0, 256, (nbs * nchunks * sb,), generator=g, dtype=torch.int64).to(torch.uint8).view(
        torch.int16)
    return lin


@pytest.mark.parametrize("nbs,nchunks,sb,t", [(5, 2, 8256, 2), (7, 1, 8192, 3), (3, 3, 600, 1)])
def test_sketch_vectors_are_sketchings(nbs, nchunks, sb, t):
    skt, lin = K._sketching(), sketched(nbs, nchunks, sb, t, nbs * sb + t)
    for groups in sorted({1, 2, 3, nbs}):
        ref = skt.vectors(lin.bytes.numpy(), nbs, nchunks, sb, groups, t)
        assert ref.shape[0] == 16 * groups * nchunks * skt.tiles_per_cta(nbs, groups, t)
        vs = list(range(ref.shape[0]))
        assert np.array_equal(lin.sketch_vectors(vs, groups), ref)


@pytest.mark.parametrize("n,kd,t", [(10240, 8192, 8), (8192, 28672, 8), (57344, 8192, 4), (16, 512, 8)])
def test_tag_vectors_bound_every_launch(n, kd, t):
    skt, tlib = K._sketching(), object.__new__(K.TensorLib)
    tlib.sb, tlib.sketch = 8256, {"t": t}
    nbs, nch = n // 16, kd // 512
    most = max(16 * groups * nch * skt.tiles_per_cta(nbs, groups, t) for groups in range(1, nbs + 1))
    assert tlib.tag_vectors(n, kd) == most


class StepLib:
    """A TensorLib's in-step commit without the library: `vectors` and `steps` are the shapes' own, `commit` is recorded."""

    vectors = K.TensorLib.vectors

    def __init__(self, capacity: int) -> None:
        self.sketch, self._vectors, self.commits = {"t": 8}, {}, []
        self.so = type("so", (), {"wt_sketch_vectors": staticmethod(lambda n, kd, g, t: n // 16 * g + kd)})
        self.step = {"vectors": capacity, "at": 0, "roots": {}}

    def steps(self, m, n, kd, cfg):
        return m + 1, None

    def commit(self, nvec, m):
        self.commits.append((m, nvec))


def test_a_step_writes_its_tags_one_launch_after_another(monkeypatch):
    shapes = [(32, 512), (16, 1024), (48, 512)]
    tlib, lins, seen = StepLib(10_000), [], []
    for n, kd in shapes:
        lin = object.__new__(K.TensorLinear)
        lin.tlib, lin.n, lin.kd, lin.configs, lin.at, lin.first, lin.last = tlib, n, kd, {}, None, False, False
        lins.append(lin)
    lins[0].first = lins[-1].last = True
    monkeypatch.setattr(K.Linear, "forward", lambda self, x8, xs: seen.append(self.at) or x8)
    monkeypatch.setattr(K.TensorLinear, "default", lambda self: K.Config(4))
    for m in (1, 4, 1):
        for lin in lins:
            lin.forward(torch.zeros(m, lin.kd), None)
    per = {m: [n // 16 * (m + 1) + kd for n, kd in shapes] for m in (1, 4)}
    want = [list(np.cumsum([0] + per[m][:-1])) for m in (1, 4, 1)]
    assert seen == [a for w in want for a in w]
    assert tlib.commits == [(m, sum(per[m])) for m in (1, 4, 1)]
    assert all(lin.at is None for lin in lins)
    monkeypatch.setattr(K.Linear, "forward", lambda self, x8, xs: x8)
    for lin in lins:
        lin.forward(torch.zeros(K.MAX_ROWS + 1, lin.kd), None)
    assert len(tlib.commits) == 3


def test_a_step_refuses_to_overflow_its_buffer(monkeypatch):
    tlib = StepLib(40)
    lin = object.__new__(K.TensorLinear)
    lin.tlib, lin.n, lin.kd, lin.configs, lin.at, lin.first, lin.last = tlib, 32, 512, {}, None, True, True
    monkeypatch.setattr(K.Linear, "forward", lambda self, x8, xs: x8)
    monkeypatch.setattr(K.TensorLinear, "default", lambda self: K.Config(4))
    with pytest.raises(RuntimeError, match="overflow"):
        lin.forward(torch.zeros(1, 512), None)
    assert tlib.commits == []
