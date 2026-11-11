"""PoUW's per-call units on the host twin (`e2e_h200.percall:python`) at a one-layer shape whose linears are whole 32 x 32
tiles: every served linear call of the step is a J10-R call with every unit's receipt committed, a unit's z is the served
call's s32 sums (a sum that differs is caught), the step's PoUW root binds the receipts, and serving is unchanged."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from e2e_h200 import percall as P
from e2e_h200.commit import Committer, fixture_salt, fixture_secrets
from e2e_h200.driver import Driver, Params
from e2e_h200.shape import Shape
from e2e_h200.slots import Request, SlotKV, SlotShape
from e2e_h200.store import GeneratorStore
from e2e_h200.weights import Weights

pytestmark = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")),
                                reason="the engine's host twin needs a C++ compiler")

#: width 64, 1 layer, 1 head of 64, MLP 128: every linear's N and K a multiple of 32, K >= 64
SHAPE = Shape("percall-w64-l1", 64, 1, 1, 64, 1, 128, 256, 8, 64)
PROMPTS = np.random.default_rng(0).integers(32, 127, (64, 1))


@pytest.fixture(scope="module")
def params():
    return Params(SHAPE, GeneratorStore(Weights(SHAPE)))


def serve(params, sub=None, batch=32, micro_batch=32):
    d = Driver(SHAPE, params, batch, Committer("slots", *fixture_secrets("t"), SHAPE, "cpu", fixture_salt("t")),
               micro_batch=micro_batch, keep=False, subcircuits=[sub] if sub else [])
    return d.run(PROMPTS[:batch], 1)[0]


def test_every_served_linear_call_is_a_call_and_every_z_the_served_sums(params):
    sub = P.python(SHAPE, run_id="t")
    res = serve(params, sub, batch=64)
    doc = sub.steps[0]
    assert doc["calls"] == 2 * 7 and sub.mismatches == 0
    assert doc["units"] == doc["z_checked_units"] == 2 * (2 + 2 + 2 + 2 + 4 + 2 + 8)
    assert doc["first_position"] == 1 << P.POS_SHIFT
    assert res.root == serve(params, batch=64).root


def test_the_serve_path_runs_the_units_on_every_call_of_every_step(params):
    sub = P.python(SHAPE, run_id="t")
    d = Driver(SHAPE, params, 32, Committer("reference", *fixture_secrets("t")), subcircuits=[sub])
    kv = SlotKV(SlotShape(32, SHAPE.layers, SHAPE.context, SHAPE.head_dim, prompt=1, generate=1), "cpu")
    d.serve(kv, [Request(i, bytes([int(PROMPTS[i, 0])])) for i in range(32)])
    assert len(sub.steps) == len(d.results) >= 1 and sub.mismatches == 0
    assert all(s["calls"] == 7 and s["units"] == 22 for s in sub.steps)


class Skewed(P.PerCall):
    """Hands the units one served sum off by one: z no longer the served accumulator."""

    def call(self, driver, st, op, lo, codes, w, acc):
        if op == "l0.wup":
            acc = acc.clone()
            acc[5, 70] += 1
        super().call(driver, st, op, lo, codes, w, acc)


def test_a_z_that_is_not_the_served_sums_is_caught(params):
    sub = Skewed(SHAPE, run_id="t")
    serve(params, sub)
    assert sub.steps[0]["z_mismatches"] == 1 and sub.mismatches == 1


class Flipped(P.PerCall):
    def _commit(self, c, receipts, acc):
        if c.op == "head":
            receipts = receipts.copy()
            receipts[3, 7] ^= 1
        super()._commit(c, receipts, acc)


@pytest.mark.slow
def test_the_steps_root_binds_every_receipt(params):
    a, b, c = P.python(SHAPE, run_id="t"), P.python(SHAPE, run_id="t"), Flipped(SHAPE, run_id="t")
    for sub in (a, b, c):
        serve(params, sub)
    assert a.steps[0]["root"] == b.steps[0]["root"] != c.steps[0]["root"]
    assert a.manifest()["credit"]["eps"] is None and len(a.manifest()["credit"]["blockers"]) == 2
