"""The batch path at the tiny shape on the CPU: kernels' per-slot commit (the host twin) gives the reference's roots,
micro-batches and dropped tables change no root, the forward stays at most `LAG` groups ahead of a slow commit, rows at
their own positions are the lockstep rows, a drawn slot opens by replaying its request alone, the network engine serves what
lockstep serves, the torch softmax is the Definitions', the map is circuits' FAMILIES table at the frozen shape and each
unit's commit unit is its output rows, and a subcircuit joins the registration and the sample."""
from __future__ import annotations

import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch

from e2e_h200 import attention as A
from e2e_h200.commit import Committer, _layout, _slots, fixture_salt, fixture_secrets
from e2e_h200.driver import Driver, Params
from e2e_h200.engine import Engine
from e2e_h200.opening import Openings, commit_slot_parallel, replay_slot
from e2e_h200.plan import Map
from e2e_h200.shape import FROZEN, MINI, TINY
from e2e_h200.store import GeneratorStore
from e2e_h200.units import Family, Replayer
from e2e_h200.weights import Weights

pytestmark = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")),
                                reason="the slots commit's host twin needs a C++ compiler")

PROMPTS = np.array([list(b"hi"), list(b"ok"), list(b"yo")])
STEPS = 4


def committer(shape=TINY):
    return Committer("slots", *fixture_secrets("test"), shape, "cpu", fixture_salt("test"))


@pytest.fixture(scope="module")
def params():
    return Params(TINY, GeneratorStore(Weights(TINY)))


@pytest.fixture(scope="module")
def kept(params):
    c = committer()
    d = Driver(TINY, params, 3, c, keep=True)
    return c, d, d.run(PROMPTS, STEPS)


def test_slots_roots_are_the_references(kept):
    c, d, results = kept
    S = c.S
    for st, r in zip(d.steps, results, strict=True):
        roots = tuple(S.commit_slot(c.context, c.layout, r.salt_key, b, c.slot_rows(st, b)).root for b in range(3))
        assert roots == r.slot_roots
        assert S.step_tree(c.context, c.layout, r.step, roots).root == r.root


def test_micro_batches_and_dropped_tables_change_no_root(params, kept):
    d = Driver(TINY, params, 3, committer(), keep=False, micro_batch=2)
    assert [r.root for r in d.run(PROMPTS, STEPS)] == [r.root for r in kept[2]] and not d.steps


def _slow_commit(monkeypatch, c, lag):
    """``c`` with the slowest side stream: it reads a group's tables only when the forward waits on that group's event.
    Returns what it saw: the most groups added and unread when the forward hands in the next, the most bytes of tables
    held, and each group's bytes."""
    held: dict[int, int] = {}
    seen = {"lead": 0, "bytes": 0, "sizes": []}

    class Slow(c.G.StepCommitter):
        def add(self, tensors):
            seen["lead"] = max(seen["lead"], len(held))
            super().add(tensors)
            k = len(seen["sizes"])
            seen["sizes"].append(sum(np.asarray(t).nbytes for t in tensors))
            held[k] = seen["sizes"][k]
            seen["bytes"] = max(seen["bytes"], sum(held.values()))
            return k

        def finish(self):
            held.clear()
            return super().finish()

    def wait(k):
        for j in [j for j in held if j <= k]:
            del held[j]

    monkeypatch.setattr(c.G, "StepCommitter", Slow)
    monkeypatch.setattr(c, "_wait", wait)
    monkeypatch.setattr("e2e_h200.commit.LAG", lag)
    return seen


#: TINY with 3 layers: 5 groups, so a bound of 2 binds
DEEP = replace(TINY, name="tiny-h32-l3", layers=3)


@pytest.fixture(scope="module")
def deep():
    p = Params(DEEP, GeneratorStore(Weights(DEEP)))
    return p, [r.root for r in Driver(DEEP, p, 3, committer(DEEP), keep=False).run(PROMPTS, 2)]


@pytest.mark.parametrize("lag", [1, 2])
def test_the_forward_stays_at_most_lag_groups_ahead_of_a_slow_commit(deep, monkeypatch, lag):
    params, roots = deep
    c = committer(DEEP)
    seen = _slow_commit(monkeypatch, c, lag)
    assert [r.root for r in Driver(DEEP, params, 3, c, keep=False).run(PROMPTS, 2)] == roots
    groups = len(c.groups)
    sizes = seen["sizes"][:groups]
    assert seen["lead"] == lag < groups - 1
    assert seen["bytes"] <= max(sum(sizes[i:i + lag + 1]) for i in range(groups)) < sum(sizes)


def test_without_the_bound_a_slow_commit_holds_the_whole_step(deep, monkeypatch):
    c = committer(DEEP)
    seen = _slow_commit(monkeypatch, c, 10 ** 9)
    Driver(DEEP, deep[0], 3, c, keep=False).run(PROMPTS, 1)
    assert seen["lead"] == len(c.groups) - 1 and seen["bytes"] == sum(seen["sizes"])


def test_rows_at_their_own_positions_are_the_lockstep_rows(params, kept):
    """Slot 0 runs alone at first, slot 2 joins two steps late: every slot's rows at each position are lockstep's."""
    _, d0, results = kept
    d = Driver(TINY, params, 3, committer(), keep=True)
    toks, fed = {s: [int(PROMPTS[s, 0])] for s in range(3)}, {s: 0 for s in range(3)}
    rows = {}
    for live in ([0], [0, 1], [0, 1, 2], [0, 1, 2], [1, 2], [2]):
        res = d.step_slots(np.array(live), np.array([fed[s] for s in live]), np.array([toks[s][-1] for s in live]))
        for b, s in enumerate(live):
            rows[(s, fed[s])] = {op: c[b] for op, (c, _) in d.steps[-1].tables.items()}
            fed[s] += 1
            nxt = int(PROMPTS[s, fed[s]]) if fed[s] < PROMPTS.shape[1] else int(res.tokens_out[b])
            toks[s].append(nxt)
    assert set(rows) == {(s, t) for s in range(3) for t in range(STEPS)}
    for (s, t), got in rows.items():
        for op, (c, _) in d0.steps[t].tables.items():
            assert np.array_equal(got[op], c[s]), (s, t, op)


@pytest.mark.parametrize("shape", [TINY, MINI], ids=lambda s: s.name)
def test_a_drawn_slot_opens_by_replaying_its_request(params, shape):
    """A unit of every family, sampled from a slot replayed alone, opens against the step's root as its one commit unit."""
    params = params if shape is TINY else Params(shape, GeneratorStore(Weights(shape)))
    c = committer(shape)
    d = Driver(shape, params, 3, c, keep=False, tamper=None)
    results = d.run(PROMPTS, STEPS)
    op = replay_slot(shape, params, c, results, PROMPTS, 1, STEPS - 1, "cpu")
    assert all(op.roots_equal) and len(op.roots_equal) == STEPS
    units = Replayer(shape, params.store.weights, op.driver.steps, op.tokens).draw(STEPS - 1, 1, b"t")
    opn = Openings(c, results)
    assert opn.reference_equal(op, STEPS - 1)
    docs = [opn.open(op, STEPS - 1, u) for u in units]
    assert {u.family for u in units} == set(Map(shape).units())
    assert all(u.ok for u in units) and all(x["verified"] for x in docs)
    bad = Driver(shape, params, 3, c, keep=False, tamper="l0.wv").run(PROMPTS, STEPS)
    op = replay_slot(shape, params, c, bad, PROMPTS, 1, STEPS - 1, "cpu")
    assert not any(op.roots_equal)
    assert not Openings(c, bad).open(op, 0, units[0])["verified"]


def test_the_parallel_slot_commit_is_the_references(params, kept):
    """`commit_slot_parallel` gives `commit_slot`'s strings, root and paths, and openings through it verify."""
    c, d, results = kept
    S, st, r = c.S, d.steps[STEPS - 1], results[STEPS - 1]
    rows = c.slot_rows(st, 2)
    ref = S.commit_slot(c.context, c.layout, r.salt_key, 2, rows)
    par = commit_slot_parallel(S, c.context, c.layout, r.salt_key, 2, rows, 3)
    assert par.strings == ref.strings and par.root == ref.root == r.slot_roots[2]
    assert all(par.tree.open([u]) == ref.tree.open([u]) for u in (0, c.layout.unit_count // 2, c.layout.unit_count - 1))
    op = replay_slot(TINY, params, c, results, PROMPTS, 1, STEPS - 1, "cpu")
    units = Replayer(TINY, params.store.weights, op.driver.steps, op.tokens).draw(STEPS - 1, 1, b"t")
    opn = Openings(c, results, workers=3)
    assert opn.reference_equal(op, STEPS - 1) and all(opn.open(op, STEPS - 1, u)["verified"] for u in units)


def test_the_engine_serves_what_lockstep_serves(kept):
    """Two requests through the endpoint's engine, the second arriving three calls late: each one's ids are the lockstep's."""
    _, _, results = kept
    want = [list(PROMPTS[s]) + [int(r.tokens_out[s]) for r in results[PROMPTS.shape[1] - 1:]] for s in range(2)]
    e = Engine(TINY, device="cpu", pool=4)
    seqs = {0: [int(t) for t in PROMPTS[0]]}
    for call in range(STEPS + 2):
        if call == 3:
            seqs[1] = [int(t) for t in PROMPTS[1]]
        live = [s for s in seqs if len(seqs[s]) < STEPS + 1]
        for s, t in zip(live, e.next_tokens([seqs[s] for s in live]), strict=True):
            seqs[s].append(t)
    assert [seqs[s][:len(want[s])] for s in range(2)] == want
    assert e.summary()["steps"] > 0


def test_softmax_t_is_the_definitions():
    rng = np.random.default_rng(5)
    x = rng.integers(0, 1 << 32, 4096, dtype=np.uint64).astype(np.uint32)
    x[:6] = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xC3000000]
    got = A.ex2_t(torch.from_numpy(x.astype(np.int64)), A.ex2_table(torch.device("cpu"))).numpy().astype(np.uint32)
    assert np.array_equal(got, A.MU.mufu_ex2_words(x, A.MU.table("ex2")))
    sw = A.bf16_of_f32((rng.standard_normal((16, 4, 32)) * 30).astype(np.float32))
    lens = rng.integers(1, 33, 16)
    p = A.softmax_t(torch.from_numpy(sw.view(np.int16)).view(torch.bfloat16), torch.from_numpy(lens))
    p = p.view(torch.int16).numpy().view(np.uint16)
    for b, n in enumerate(lens):
        assert np.array_equal(p[b, :, :n], A.softmax(sw[b:b + 1, :, :n]).probs[0]) and not p[b, :, n:].any()


#: circuits' FAMILIES table at the frozen shape: a slot's units by key, the embedding's and head's included
FAMILIES = {
    "EmbeddingCoordinate_v1{V=256}": 16_384,
    "AllGather_v1{WORLD=128,N=1}": 40_066,
    "RmsSumSqChunks_v1{C=4096,K=128}": 420,
    "RmsNormOutSums_v1{C=128,S=128,G=2,N=16384,EPS=1e-05}": 13_440,
    "RowAbsMax_v1{N=16384,RELU=false}": 157,
    "QuantChunkS8_v1{N=8192,RELU=false}": 314,
    "ScaledMmS8CoordinateRows_v1{K=16384,R=2,DOT=BlackwellS8ImmaDot32_v1}": 5_118_720,
    "AllGather_v1{WORLD=64,N=1}": 13_416,
    "QkSoftmaxRowPad_v1{T=32,D=64}": 13_312,
    "PvRow_v1{T=32,K=32,D=64,DOT=HopperBF16WgmmaDot16_v1}": 13_312,
    "AllGather_v1{WORLD=2,N=64}": 6_656,
    "Bf16AddRow_v2{N=128}": 13_312,
    "RowAbsMax_v1{N=16384,RELU=true}": 156,
    "RowAbsMax_v1{N=16387,RELU=true}": 52,
    "QuantChunkS8_v1{N=8192,RELU=true}": 416,
    "ScaledMmS8CoordinateRows_v1{K=65536,R=8,DOT=BlackwellS8ImmaDot32_v1}": 851_968,
    "Argmax_v2{V=256}": 1,
}


def test_the_map_is_circuits_families_table_at_the_frozen_shape():
    m = Map(FROZEN)
    got: dict[str, int] = {}
    for g in m.groups():
        for k, n in m.counts(g).items():
            got[k] = got.get(k, 0) + n
    assert got == FAMILIES
    S, _ = _slots()
    lay, _ = _layout(FROZEN)
    assert lay.unit_count == sum(FAMILIES.values()) == 6_102_102 and lay.row_count == 6_102_832
    assert max(u.bits for u in m.units().values()) <= S.UNIT_MAX_BITS


@pytest.mark.parametrize("shape", [TINY, MINI, FROZEN], ids=lambda s: s.name)
def test_each_units_commit_unit_is_its_output_rows(shape):
    """Unit i of every family commits as one commit unit (`Committer.unit_of`) whose rows are its output bytes in its table's
    row, and a quantizer chunk's the scale too; the layout's commit units are the map's units, in its order."""
    S, _ = _slots()
    lay, bases = _layout(shape)
    m = Map(shape)
    u0 = 0
    for gi, entries in enumerate(m.groups()):
        src = 0
        for t, us in entries:
            for u in us:
                assert bases[u.name] == u0 and lay.unit_base[gi] <= u0
                for i in sorted({0, u.count - 1}):
                    rows = [[(p.source, p.offset, p.nbytes) for p in r.parts] for r in lay.units[u0 + i]]
                    assert rows == [[(src, u.offset + i * u.stride, u.nbytes)]] + ([[(src + 1, 0, 4)]] if u.scale else []), \
                        (u.name, i)
                u0 += u.count
            src += 2 if t.quant else 1
    assert u0 == lay.unit_count


class Toy:
    """A subcircuit whose one family re-reads each slot's argmax: what a plugged-in module gives the registration."""
    name = "toy"

    def __init__(self, **kw):
        self.served = 0

    def serve(self, driver, result):
        self.served += 1

    def families(self, replayer, step):
        def replay(b, _):
            w = replayer.u16(step, "argmax", b)
            return [w], w, w.copy()
        return [Family("sub:toy.argmax", "Toy_v1{}", "fixture", 1, replay)]

    def manifest(self):
        return {"template": "Toy_v1{}"}


def test_a_subcircuit_joins_the_registration_and_the_draw(params):
    toy = Toy()
    d = Driver(TINY, params, 3, Committer("reference", *fixture_secrets("test")), subcircuits=[toy])
    results = d.run(PROMPTS, 2)
    rp = Replayer(TINY, params.store.weights, d.steps, [r.tokens_in for r in results], [toy])
    assert toy.served == 2
    assert {u.family for u in rp.draw(1, 1, b"t")} >= {"sub:toy.argmax"}
    assert [b["name"] for b in rp.registration(1)["blocks"]] == ["model", "toy"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_seq_sum_is_the_hosts():
    from e2e_h200 import ops as O
    g = torch.Generator().manual_seed(11)
    x = torch.randn(300, 4099, generator=g) * torch.exp2(torch.randint(-140, 40, (300, 4099), generator=g).float())
    x[0], x[1, 1::2], x[2] = 1e-45, -x[1, 0::2][:2049], torch.tensor([3.0e38, -3.0e38] * 2050)[:4099]
    host = O.seq_sum(x)
    assert O._seq_sum_kernel() is not None
    assert torch.equal(O.seq_sum(x.cuda()).cpu().view(torch.int32), host.view(torch.int32))
