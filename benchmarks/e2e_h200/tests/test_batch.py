"""The batch path at ALLP-IFACE's tiny shape on the CPU: kernels' per-slot commit (the host twin) gives the reference's roots,
micro-batches and dropped tables change no root, rows at their own positions are the lockstep rows, a drawn slot opens by
replaying its request alone, the network engine serves what lockstep serves, the torch softmax is the Definitions', the
layout's unit map covers each unit's bytes, and a subcircuit joins the registration and the draw."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch

from e2e_h200 import attention as A
from e2e_h200.commit import Committer, fixture_salt, fixture_secrets, tables
from e2e_h200.driver import Driver, Params
from e2e_h200.engine import Engine
from e2e_h200.opening import Openings, replay_slot
from e2e_h200.shape import FROZEN, TINY
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


def test_a_drawn_slot_opens_by_replaying_its_request(params):
    c = committer()
    d = Driver(TINY, params, 3, c, keep=False, tamper=None)
    results = d.run(PROMPTS, STEPS)
    op = replay_slot(TINY, params, c, results, PROMPTS, 1, STEPS - 1, "cpu")
    assert all(op.roots_equal) and len(op.roots_equal) == STEPS
    units = Replayer(TINY, params.store.weights, op.driver.steps, op.tokens).draw(STEPS - 1, 1, b"t")
    opn = Openings(c, results)
    assert opn.reference_equal(op, STEPS - 1)
    docs = [opn.open(op, STEPS - 1, u) for u in units]
    assert units and all(u.ok for u in units) and all(x["verified"] for x in docs)
    bad = Driver(TINY, params, 3, c, keep=False, tamper="l0.wv").run(PROMPTS, STEPS)
    op = replay_slot(TINY, params, c, bad, PROMPTS, 1, STEPS - 1, "cpu")
    assert not any(op.roots_equal)
    assert not Openings(c, bad).open(op, 0, units[0])["verified"]


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


def test_the_unit_map_covers_each_units_bytes():
    c = committer(FROZEN)
    S, lay = c.S, c.layout
    flat = [t for g in tables(FROZEN) for t in g]
    src = {}
    for gi, g in enumerate(tables(FROZEN)):
        i = 0
        for t in g:
            src[t.op] = (gi, i)
            i += 2 if t.quant else 1
    for op, lo, hi, scale in (("l3.wup", 2 * 40000, 2 * 40001, False), ("l3.down_quant", 0, FROZEN.mlp, True),
                              ("l0.attn_quant", 0, FROZEN.width, True), ("head", 0, 2 * FROZEN.vocab, False),
                              ("l51.scores", 2 * 255 * 32, 2 * 255 * 32 + 2, False), ("embed", 0, 2 * FROZEN.width, False)):
        gi, s0 = src[op]
        covered = set()
        for u in c.units_of(op, lo, hi, scale):
            assert lay.unit_base[gi] <= u < lay.unit_base[gi] + len(lay.groups[gi].units)
            for row in lay.units[u]:
                for p in row.parts:
                    covered |= {(p.source - s0, o) for o in range(p.offset, p.offset + p.nbytes)}
        assert {(0, o) for o in range(lo, hi)} <= covered and (not scale or (1, 0) in covered), op
    assert len(flat) == 2 + 17 * FROZEN.layers + 3 and S.UNIT_MAX_BITS == 262_144


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
        return {"template": "Toy_v1{}", "issue": "CMP-126"}


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
