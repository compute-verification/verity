"""The served step's `Program` (`e2e_h200.program`): evaluated node by node on a driven step's inputs, its absent K and V rows
NaNs, every family's output is the driver's committed table, and it returns the step's token and K and V rows."""
from __future__ import annotations

import numpy as np
import pytest

from e2e_h200 import program as PR
from e2e_h200.commit import Committer, fixture_secrets
from e2e_h200.driver import Driver
from e2e_h200.shape import MINI, PV_K, TINY
from e2e_h200.store import GeneratorStore
from e2e_h200.units import Replayer
from e2e_h200.weights import Weights
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided
from e2e_h200 import served_weights as SW

BATCH, STEPS = 2, 4
#: an absent K or V row's words: a NaN, which reaches the step's output unless the padded units mask keys n and past
ABSENT = 0x7FC0
PROMPTS = np.array([list(b"hi"), list(b"ok")])


def bits(a: np.ndarray) -> np.ndarray:
    """Little-endian words as their bits, least significant first: a leaf a bit, in the Program's leaf order."""
    return np.unpackbits(np.ascontiguousarray(a).view(np.uint8), bitorder="little")


def _resolve(refs, vals: dict) -> np.ndarray:
    if isinstance(refs, Affine):
        return vals[(refs.space, refs.idx)][refs.base:refs.base + refs.count]
    if isinstance(refs, Strided):
        j = np.arange(refs.count, dtype=np.int64)
        return vals[(refs.space, refs.idx)][refs.base + (j // refs.inner) * refs.outer_stride + (j % refs.inner) * refs.inner_stride]
    if isinstance(refs, Concat):
        return np.concatenate([_resolve(p, vals) for p in refs.parts]) if refs.parts else np.zeros(0, np.uint8)
    if isinstance(refs, Explicit):
        return np.array([vals[(s, i)][k] for s, i, k in refs.items], np.uint8)
    raise TypeError(type(refs).__name__)


def interpret(prog, args: list[np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Every named node's output bits and the return's, the root body evaluated node by node (`evaluate_batch`)."""
    body = prog.fn.body
    vals = {("p", i): a for i, a in enumerate(args)}
    named = {}
    for k, node in enumerate(body.nodes):
        if node.form == "batch":
            ins = [_resolve(a.refs, vals) if ax is None else _resolve(a.refs, vals).reshape(node.n, -1)
                   for a, ax in zip(node.args, node.axes)]
            ins = [np.broadcast_to(x, (node.n, x.size)) if ax is None else x for x, ax in zip(ins, node.axes)]
        else:
            ins = [_resolve(a.refs, vals)[None] for a in node.args]
        out = evaluate_batch(node.fn, *ins).reshape(-1).astype(np.uint8)
        vals[("n", k)] = out
        if node.name:
            named[node.name] = out
    return named, _resolve(body.ret, vals)


def drive(shape, fused=frozenset()):
    w = Weights(shape)
    d = Driver(shape, GeneratorStore(w), BATCH, Committer("reference", *fixture_secrets("test"), fused=fused), "cpu",
               clock=lambda: 0.0)
    results = d.run(PROMPTS, STEPS)
    return w, d, Replayer(shape, w, d.steps, [r.tokens_in for r in results], fused=fused)


def inputs(shape, w: Weights, rp: Replayer, step: int, b: int) -> list[np.ndarray]:
    st = rp.steps[step]
    t = int(st.positions[b])
    out = [bits(w.array(x.name)) for x in SW.tensors(shape)]
    out += [bits(np.array([rp.tokens[step][b]], np.uint16)), bits(np.array([t + 1], np.uint16)),
            bits(np.array([t], np.int32))]
    kv = [rp.kv(step, i, b) for i in range(shape.layers)]
    for j in (0, 1):
        rows = np.full((shape.layers, PV_K, shape.head_dim), ABSENT, np.uint16)
        for i, pair in enumerate(kv):
            rows[i, :len(pair[j])] = pair[j]
        out.append(bits(rows))
    return out


@pytest.fixture(scope="module", params=[(TINY, frozenset()), (MINI, frozenset()), (MINI, frozenset({"norm", "pv", "residual"}))],
                ids=lambda p: "-".join([p[0].name, *(["fused", *sorted(p[1])] if p[1] else [])]))
def served(request):
    shape, fused = request.param
    w, d, rp = drive(shape, fused)
    return shape, w, rp, PR.program(shape, fused)


def test_every_family_is_the_drivers_table_and_the_return_is_the_steps_token_and_kv(served):
    shape, w, rp, prog = served
    step = STEPS - 1
    tables = rp.steps[step].tables
    table_of = {u.name: u.table for g in rp.map.groups() for _, us in g for u in us}
    names = set(table_of) | set(table_of.values())
    for b in range(BATCH):
        named, ret = interpret(prog, inputs(shape, w, rp, step, b))
        checked = 0
        for name, out in named.items():
            if name.endswith("_quant"):
                codes = np.ascontiguousarray(tables[name][0][b]).view(np.uint8)
                chunk = rp.map.f.qchunk(codes.size)
                rows = out.reshape(codes.size // chunk, -1)
                assert rows.shape[1] in (8 * chunk, 8 * chunk + 32), name
                assert np.array_equal(np.packbits(rows[:, :8 * chunk].reshape(-1), bitorder="little"), codes), name
                if rows.shape[1] > 8 * chunk:
                    assert np.array_equal(np.packbits(rows[0, 8 * chunk:], bitorder="little"),
                                          np.asarray(tables[name][1][b], np.float32).reshape(1).view(np.uint8)), name
                checked += 1
            elif name.endswith("_quant.scale"):
                word = np.asarray(tables[name.removesuffix(".scale")][1][b], np.float32).reshape(1).view(np.uint8)
                assert all(np.array_equal(np.packbits(r, bitorder="little"), word) for r in out.reshape(-1, 32)), name
                checked += 1
            elif name.endswith("_amax.part") or name.endswith("_amax"):
                table = np.ascontiguousarray(tables[name.removesuffix(".part")][0][b]).view(np.uint8)
                got = np.packbits(out, bitorder="little")
                lo = 0 if name.endswith(".part") else table.size - 2
                assert np.array_equal(got, table[lo:lo + got.size]), name
                checked += 1
            elif name == "pos":
                t = int(rp.steps[step].positions[b])
                assert np.array_equal(np.packbits(out, bitorder="little"),
                                      np.ascontiguousarray(w.array("pos")[:, t]).astype(np.uint16).view(np.uint8)), name
                checked += 1
            elif name in names:
                assert np.array_equal(np.packbits(out, bitorder="little"),
                                      np.ascontiguousarray(tables[table_of.get(name, name)][0][b]).view(np.uint8)), name
                checked += 1
        assert checked == len(named)
        L, d = shape.layers, shape.head_dim
        tok, ks, vs = ret[:16], ret[16:16 + 16 * L * d], ret[16 + 16 * L * d:]
        assert np.packbits(tok, bitorder="little").view(np.uint16)[0] == \
            np.ascontiguousarray(tables["argmax"][0][b]).view(np.uint16)[0]
        for i in range(L):
            for got, op in ((ks, "wk"), (vs, "wv")):
                assert np.array_equal(np.packbits(got[16 * d * i:16 * d * (i + 1)], bitorder="little"),
                                      np.ascontiguousarray(tables[f"l{i}.{op}"][0][b]).view(np.uint8))


def test_its_root_parameters_are_the_weights_by_name_then_the_run_inputs(served):
    shape, _w, _rp, prog = served
    names = [n for n, _ in prog.fn.signature()[0]]
    assert names == [t.name for t in SW.tensors(shape)] + ["token", "n", "p", "keys", "values"]
    assert SW.reader(shape, prog).digest == program_digest(prog)
