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


@pytest.mark.parametrize("tw,s,warps,t", [(86, 96, 12, 8), (43, 192, 8, 8), (43, 192, 12, 32), (128, 64, 16, 8)])
@pytest.mark.parametrize("sketch", [None, 16, 64])
def test_tensor_configs_fit_the_fused_kernel(tw, s, warps, t, sketch):
    """Every config the tensor arm launches passes `wt_fused_sketch`'s bounds and its shared memory at M = 32
    (`fused_smem_bytes<4, k_s>` with the signs in global memory), for p768, p1536 (8 warps at T = 8, and 12 at T = 32)
    and p512; the sketched ones include the tensor-core codec's p1536 picks, 8 x 8 and 10 x 9."""
    tlib = object.__new__(K.TensorLib)
    tlib.sb, tlib.task_words, tlib.fused_warps, tlib.fused_span = tw * s, tw, warps, (tw + 30) // 32 + 1
    tlib.sketch = {"k": sketch, "t": t} if sketch else None
    for tagged in (False, True):
        cfgs = tlib.configs(tagged)
        assert cfgs
        if sketch == 16 and warps == 12 and tw == 43:
            assert {(8, 8), (10, 9)} <= {(d, st) for d, st, _, _ in cfgs}
        for d, st, tg, ks in cfgs:
            assert ks == (sketch or 0) and (tg == -1 if sketch or not tagged else tg >= 0)
            assert d >= tlib.fused_span and d + 2 + max(tg, 0) <= warps and st >= -(-32 * d // tw)
            assert not ks or st < 2 * t
            assert st * tlib.sb + 4 * 8 * 528 + 16 * ((5 * st + 1) // 2) + 16 + 3 * 64 * ks <= K.SMEM_PER_BLOCK


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


@pytest.mark.parametrize("bits,nbs,nchunks", [(512, 8, 2), (512, 5, 3), (256, 4, 1)])
def test_wide_vectors_are_wide_sketchs(bits, nbs, nchunks):
    """serve.py's per-vector cut of a wide store (a tile's T records) is `wide_sketch.vectors` over the whole store."""
    import serve

    ws, f = K._wide_sketch(), K._wide().field(bits)
    wlib = object.__new__(K.WideLib)
    wlib.lane_words, wlib.limbs, wlib.field = f.lane_words(), f.limbs, f
    wlib.sketch = {"t": ws.T, "n": ws.vector_bytes(f)}
    lin = object.__new__(K.WideLinear)
    lin.wlib, lin.n, lin.nchunks = wlib, 16 * nbs, nchunks
    g = torch.Generator().manual_seed(bits + nbs)
    lin.store = torch.randint(-(1 << 31), (1 << 31) - 1, (nbs * nchunks * f.lane_words() * f.limbs * 32,), generator=g,
                              dtype=torch.int64).to(torch.int32)
    ref = ws.vectors(lin.store.numpy().astype("<i4").tobytes(), 16 * nbs, 512 * nchunks, f)
    vs = list(range(ref.shape[0]))
    assert ref.shape[0] == 16 * ws.tiles(16 * nbs, 512 * nchunks)
    assert np.array_equal(serve.wide_vectors(lin, vs), ref)


class Log(list):
    def stream(self, name):
        log = self

        class Stream:
            cuda_stream = name

            def wait_event(self, e):
                log.append(("wait", name, e.name))
        return Stream()

    def event(self):
        log, n = self, sum(1 for x in self if x[0] == "event")
        self.append(("event", n))

        class Event:
            name = n

            def record(self, s):
                log.append(("record", s.cuda_stream, n))
        return Event()


class Ptr:
    def __init__(self, p: int) -> None:
        self.p = p

    def data_ptr(self) -> int:
        return self.p


class WideStepLib:
    """A WideLib's in-step commit without the library: the ring's slots are pointers 100 and 200, and `leaves`, `fold`
    (when the step folds) and `tree` are logged with their streams."""

    def __init__(self, log: Log, capacity: int, fold: bool) -> None:
        self.log, self.sketch = log, {"t": 4}
        self.step = {"ring": [Ptr(100), Ptr(200)], "vectors": capacity, "at": 0, "slot": 0, "pending": None,
                     "side": log.stream("side"), "folds": log.stream("folds"), "fold": fold, "unjoined": None,
                     "roots": {}}

    def leaves(self, tags, at, nvec, s):
        self.log.append(("leaves", s, tags, at, nvec))

    def fold(self, tags, at, nvec, s):
        if self.step["fold"]:
            self.log.append(("fold", s, at, nvec))

    def tree(self, s):
        self.log.append(("tree", s))


def wide_step(monkeypatch, blocks: list[int], capacity: int, fold: bool = False, seal: tuple[int, ...] = ()):
    log = Log()
    w, lins = WideStepLib(log, capacity, fold), []
    for i, nv in enumerate(blocks):
        lin = object.__new__(K.WideLinear)
        lin.wlib, lin.blocks, lin.at, lin.first, lin.last, lin.fork = w, nv, None, False, False, None
        lin.seal = i in seal
        lins.append(lin)
    lins[0].first = lins[-1].last = True
    cur = log.stream("cur")
    monkeypatch.setattr(K.torch.cuda, "current_stream", lambda: cur)
    monkeypatch.setattr(K.torch.cuda, "Event", log.event)
    monkeypatch.setattr(K.Linear, "forward", lambda self, x8, xs: log.append(
        ("gemm", self.at, w.step["ring"][w.step["slot"]].data_ptr())) or x8)
    return log, w, lins


def test_a_wide_step_hashes_each_linears_leaves_beside_the_next_gemm(monkeypatch):
    blocks = [32, 48, 16]
    log, w, lins = wide_step(monkeypatch, blocks, 96)
    for m in (1, 4):
        del log[:]
        for lin in lins:
            lin.forward(torch.zeros(m, 8), None)
        e = [x for x in log if x[0] != "event"]
        f1, f2 = lins[1].fork.name, lins[2].fork.name
        j1, j2 = lins[1].join.name, lins[2].join.name
        assert e == [("gemm", 0, 100),
                     ("record", "cur", f1), ("wait", "side", f1), ("leaves", "side", 100, 0, 32), ("record", "side", j1),
                     ("gemm", 32, 200), ("wait", "cur", j1),
                     ("record", "cur", f2), ("wait", "side", f2), ("leaves", "side", 200, 32, 48), ("record", "side", j2),
                     ("gemm", 80, 100), ("wait", "cur", j2),
                     ("leaves", "cur", 100, 80, 16), ("tree", "cur")]
        assert w.step["roots"][m] == 96 and w.step["pending"] is None
        assert all(lin.at is None for lin in lins)
    monkeypatch.setattr(K.Linear, "forward", lambda self, x8, xs: x8)
    del log[:]
    for lin in lins:
        lin.forward(torch.zeros(K.MAX_ROWS + 1, 8), None)
    assert log == []


def test_a_folding_wide_step_folds_each_linears_leaves_on_a_third_stream_joined_at_a_seal(monkeypatch):
    """The fold waits only for its linear's leaves, so the GEMM after next has its ring slot once they are hashed; the
    folds join the GEMM's stream at a sealed linear (the last before attention) and at the step's last."""
    log, w, lins = wide_step(monkeypatch, [32, 48, 16, 64, 32], 192, fold=True, seal=(1,))

    def hashed(i, tags, at, nv):
        f, j, d = (getattr(lins[i], a).name for a in ("fork", "join", "folded"))
        return [("record", "cur", f), ("wait", "side", f), ("leaves", "side", tags, at, nv), ("record", "side", j),
                ("wait", "folds", j), ("fold", "folds", at, nv), ("record", "folds", d)]

    for m in (1, 8):
        del log[:]
        for lin in lins:
            lin.forward(torch.zeros(m, 8), None)
        e = [x for x in log if x[0] != "event"]
        j = [getattr(lin, "join", None) and lin.join.name for lin in lins]
        d = [getattr(lin, "folded", None) and lin.folded.name for lin in lins]
        assert e == ([("gemm", 0, 100)]
                     + hashed(1, 100, 0, 32) + [("gemm", 32, 200), ("wait", "cur", j[1]), ("wait", "cur", d[1])]
                     + hashed(2, 200, 32, 48) + [("gemm", 80, 100), ("wait", "cur", j[2])]
                     + hashed(3, 100, 80, 16) + [("gemm", 96, 200), ("wait", "cur", j[3])]
                     + hashed(4, 200, 96, 64) + [("gemm", 160, 100), ("wait", "cur", j[4]),
                                                 ("leaves", "cur", 100, 160, 32), ("fold", "cur", 160, 32),
                                                 ("wait", "cur", d[4]), ("tree", "cur")])
        assert w.step["roots"][m] == 192 and w.step["unjoined"] is None


def test_a_wide_step_refuses_to_overflow_its_commit(monkeypatch):
    log, _, lins = wide_step(monkeypatch, [32, 48], 64)
    lins[0].forward(torch.zeros(1, 8), None)
    with pytest.raises(RuntimeError, match="overflow"):
        lins[1].forward(torch.zeros(1, 8), None)
    assert not any(x[0] in ("leaves", "tree") for x in log)


def test_a_step_refuses_to_overflow_its_buffer(monkeypatch):
    tlib = StepLib(40)
    lin = object.__new__(K.TensorLinear)
    lin.tlib, lin.n, lin.kd, lin.configs, lin.at, lin.first, lin.last = tlib, 32, 512, {}, None, True, True
    monkeypatch.setattr(K.Linear, "forward", lambda self, x8, xs: x8)
    monkeypatch.setattr(K.TensorLinear, "default", lambda self: K.Config(4))
    with pytest.raises(RuntimeError, match="overflow"):
        lin.forward(torch.zeros(1, 512), None)
    assert tlib.commits == []
