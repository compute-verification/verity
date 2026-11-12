"""The value test of `circuit_properties.work`'s default reading, the circuit alone with nothing stated: on small
attention steps, evaluated word by word (`serve3_reference.materialize`), no two MACs a product it charges reads one
pair of values beyond the share a sequence's start leaves (`Work.keep`), where the batch repeats tokens within its rows.
Two pairs count as one where both operands agree to a relative `TOL` = 2^-16 in either order (`distinct`), so a pair
of values a step rounds a hair apart is no distinct pair.

Each step is `test_program_reading.loop`'s scan of three iterations of a training step under AdamW at random weights,
and the third iteration's products are the ones compared. The batch repeats a token within each row and meets
`SHARED_PREFIX`'s premise: past each row's first position, no two positions of the batch hold equal words. At a batch that
breaks it in each of its three ways (`BROKEN`, a row of one token among them), the reading does charge MACs that read
one pair twice (`test_a_batch_that_breaks_the_premise_repeats_pairs_the_reading_charges`). The steps:

- `sdpa`: an embedding, two layers of one-head causal attention (`scaled_dot_product_attention`) and a head.
- `after_mask`: the same with the mask an explicit constant added to the scores, and a product of the masked scores by
  a weight right after the add, which the reading may not charge: the add leaves two positions of one token equal
  wherever the mask's words are 0, so that product reads one pair at both.
- `rope`: `sdpa` with rotary tables, inputs the Program doesn't state.
- `relu`: `sdpa` with a ReLU MLP (`clamp_min`) after the second layer, whose down-projection the reading may not charge:
  two of its inputs' negative words give one 0, so an op with a specific argument does not give a specific value by
  that alone.
- `lf`: the lf model's step at its smallest size.

A dense step on a batch of values (`test_a_value_batchs_rows_may_repeat`) is charged nothing from the circuit alone:
two of its rows may be equal, as two here are, and every product of its rows by a shared weight then reads one pair
twice.

In every step the first layer's P·V reads repeated pairs, where two keys of one token hold one value row, and is left
out.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from test_program_reading import LF, WK, loop

from verity.definitions.numerics.serve3_reference import materialize
from verity_vllm.program.frontend.train_step import AdamW, step_program

V, D, T, L = 8, 4, 4, 2
K0 = 65536.0
#: two rows of T + 1 tokens, each repeating a token, the loss reading the first T
BATCH = torch.tensor([[1, 2, 1, 3, 5], [4, 1, 4, 6, 7]])
#: the relative distance within which two operand values count as one
TOL = 2.0 ** -16
#: batches that break the premise, each by its first row but the last: the other row is one the held batch could have
BROKEN = {"a run of one token first": [[1, 1, 2, 4, 5], [6, 7, 0, 5, 1]],
          "one token": [[3, 3, 3, 3, 3], [6, 7, 0, 5, 1]],
          "prefixes in proportion": [[1, 2, 1, 2, 5], [6, 7, 0, 5, 1]],
          "two rows sharing two tokens first": [[1, 2, 3, 4, 5], [1, 2, 6, 7, 0]]}
THIRD = "Iteration_v1#2"


def _rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attn(nn.Module):
    def __init__(self, after_mask: bool = False, rope: bool = False, relu: bool = False):
        super().__init__()
        self.embed = nn.Parameter(torch.empty(V, D))
        self.ws = nn.ModuleList(nn.Linear(D, D, bias=False) for _ in range(4 * L))
        self.head = nn.Linear(D, V, bias=False)
        self.after_mask, self.rope, self.relu = after_mask, rope, relu
        if relu:
            self.up, self.down = nn.Linear(D, 2 * D, bias=False), nn.Linear(2 * D, D, bias=False)
        if after_mask:
            self.u = nn.Parameter(torch.empty(T, T))
        if rope:
            ang = torch.outer(torch.arange(T, dtype=torch.float32), 1.0 / (100 ** (torch.arange(0, D, 2) / D)))
            self.register_buffer("cos", ang.cos(), persistent=False)
            self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, idx):
        b, t = idx.shape
        x = F.embedding(idx, self.embed)
        for i in range(L):
            wq, wk, wv, wo = self.ws[4 * i:4 * i + 4]
            q, k, v = (w(x).view(b, t, 1, D).transpose(1, 2) for w in (wq, wk, wv))
            if self.rope:
                q, k = _rope(q, self.cos[:t], self.sin[:t]), _rope(k, self.cos[:t], self.sin[:t])
            if self.after_mask:
                mask = torch.where(torch.ones(t, t, dtype=torch.bool, device=q.device).tril(), 0.0, -1e4)
                s = q @ k.transpose(-1, -2) / math.sqrt(D) + mask
                a = s.softmax(-1) @ v
                x = x + 1e-4 * (s @ self.u).sum(-1).transpose(1, 2)
            else:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + wo(a.transpose(1, 2).reshape(b, t, D))
        if self.relu:
            x = x + self.down(self.up(x).clamp_min(0.0))
        return self.head(x)


def _square(model, x):
    return model(x[:, :-1]).square().sum()


CASES = {"sdpa": lambda: Attn(), "after_mask": lambda: Attn(after_mask=True), "rope": lambda: Attn(rope=True),
         "relu": lambda: Attn(relu=True), "lf": lambda: LF.LM(LF.Size("mini", D, 2, 1, 8), ctx=T, vocab=V)}


def _bits(x: torch.Tensor) -> list[int]:
    x = x.detach().contiguous().reshape(-1)
    if x.dtype == torch.float32:
        return x.view(torch.int32).numpy().astype(np.uint32).tolist()
    return [int(v) & ((1 << 64) - 1) for v in x.tolist()]


def evaluated(make, batch: torch.Tensor = BATCH, loss=_square, n: int = 3):
    """``make()``'s step on ``batch`` under ``loss`` as `loop`'s scan of ``n`` iterations, and every gate's value at
    random weights."""
    opt = AdamW(lr=0.05)
    with torch.device("meta"):
        meta = make()
    step = step_program(meta, loss, opt, torch.empty_like(batch, device="meta"))
    prog = loop(step, n)
    bufs = {"buf_" + name: b for name, b in make().named_buffers()}
    g = torch.Generator().manual_seed(0)
    P = len(step.theta)
    ins = []
    for pi, (name, _t) in enumerate(prog.fn.signature()[0]):
        leaves = prog.input_coll(pi).type.leaves
        if pi < P:
            ins.append(_bits(torch.randn(leaves, generator=g) * 0.7))
        elif pi < 3 * P:
            ins.append([0] * leaves)
        elif pi < 3 * P + 3:
            key = AdamW.SCALARS[pi - 3 * P]
            ins.append(_bits(torch.tensor([opt.scalars(s + 1)[key] for s in range(n)], dtype=torch.float32)))
        elif pi == 3 * P + 3:
            ins.append(_bits(torch.stack([batch] * n)))
        else:
            ins.append(_bits(bufs[name]))
    return step, prog, materialize(prog, ins)


def pairs(prog, tab) -> dict:
    """By the step's product node: the reason it is left out ("" where charged) and each of its third iteration's MACs'
    pair of operand values."""
    by: dict = defaultdict(lambda: ["", []])
    for call, site, reason in WK.calls(prog, K0):
        if THIRD not in call.parent.path():
            continue
        got = [tuple(sorted((tab[g.operands[0]], tab[g.operands[1]]))) for g in call if g.prim.name == "F32Fma"]
        assert len(got) == site.macs
        by[site.node][0] = reason
        by[site.node][1].extend(got)
    return by


def _close(x: np.ndarray, y) -> np.ndarray:
    return (x == y) | (np.abs(x - y) <= TOL * np.maximum(np.abs(x), np.abs(y)))


def distinct(ps: list) -> int:
    """How many of ``ps``' pairs of words are distinct at `TOL`: each, in turn, unless both its values are close to
    an earlier distinct pair's, in either order."""
    a = np.array(sorted(set(ps)), dtype=np.uint32).reshape(-1, 2).view(np.float32).astype(np.float64)
    reps, m = np.empty_like(a), 0
    for v in a:
        r = reps[:m]
        if not ((_close(r[:, 0], v[0]) & _close(r[:, 1], v[1])) | (_close(r[:, 0], v[1]) & _close(r[:, 1], v[0]))).any():
            reps[m], m = v, m + 1
    return m


def repeats(ps: list) -> int:
    return len(ps) - distinct(ps)


def fed_by(step, node: int, test) -> bool:
    """Whether every node product ``node``'s x reads passes ``test``."""
    R = WK.Reader(step, K0)
    src = {i for *_ab, s, i, _lo, _hi in R.parts(step.fn.body.nodes[node].args[0].refs) if s == "n"}
    return bool(src) and all(test(step.fn.body.nodes[i]) for i in src)


def first_pv(step, by) -> int:
    """The first attention's P·V: the first product whose x reads only a softmax's rows."""
    return first_after(step, by, r"(SoftmaxRef|SelectF32)_v\d+")


def first_after(step, by, op: str, product: str = "") -> int:
    """The first product, of a Definition starting ``product``, whose x reads only ``op``'s outputs."""
    nodes = step.fn.body.nodes
    return min(k for k in by if nodes[k].fn.id.startswith(product) and fed_by(step, k, lambda nd: re.match(op, nd.fn.id)))


def first_after_mask(step, by) -> int:
    """The first product whose x reads only an add of a constant."""
    R, nodes = WK.Reader(step, K0), step.fn.body.nodes

    def add_const(nd):
        return nd.fn.id.startswith("F32Add") and any(
            all(nodes[i].fn.id.startswith("Const") for *_ab, s, i, _lo, _hi in R.parts(a.refs) if s == "n")
            for a in nd.args)
    return min(k for k in by if fed_by(step, k, add_const))


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES)
def test_what_the_circuit_alone_charges_reads_distinct_pairs(case):
    """Evaluates the step's three iterations word by word (seconds a case): what it catches is a rule of the default
    reading that calls a value specific where two positions of one token hold one value, so that a product it charges
    reads one pair of values at both, which no token-indexed MAC count may hold twice."""
    step, prog, tab = evaluated(CASES[case])
    keep = WK.read(prog, K0).keep
    by = pairs(prog, tab)
    charged = [ps for reason, ps in by.values() if not reason]
    assert charged
    for ps in charged:
        assert distinct(ps) >= keep * len(ps)
    pv = first_pv(step, by)
    assert by[pv][0] and repeats(by[pv][1])
    if case == "after_mask":
        am = first_after_mask(step, by)
        assert by[am][0] == "gathered" and repeats(by[am][1])
    if case == "relu":
        down = first_after(step, by, r"SelectF32_v\d+", f"LinearRef_v1{{K={2 * D},N={D},")
        assert by[down][0] and repeats(by[down][1])


@pytest.mark.slow
def test_a_value_batchs_rows_may_repeat():
    """A float batch whose first and third rows are equal, through two linear layers: what it catches is a reading
    that takes a value batch's rows as distinct, so that it charges a product whose members read one pair of values
    at the two equal rows."""
    x = torch.randn(4, D, generator=torch.Generator().manual_seed(1))
    x[2] = x[0]

    def make():
        return nn.Sequential(nn.Linear(D, 2 * D, bias=False), nn.SiLU(), nn.Linear(2 * D, D, bias=False))

    step, prog, tab = evaluated(make, x, lambda model, b: model(b).square().sum())
    by = pairs(prog, tab)
    assert all(reason for reason, _ps in by.values())
    down = first_after(step, by, r"(F32Mul|SiluRef)_v\d+", f"LinearRef_v1{{K={2 * D},N={D},")
    assert by[down][0] == "gathered" and repeats(by[down][1])


def repeating(case: str, batch: torch.Tensor) -> set:
    """The products the reading charges on ``case``'s step at ``batch`` whose MACs repeat a pair."""
    _step, prog, tab = evaluated(CASES[case], batch)
    return {k for k, (reason, ps) in pairs(prog, tab).items() if not reason and repeats(ps)}


@pytest.mark.slow
@pytest.mark.parametrize("case,broken", [
    ("lf", "a run of one token first"), ("lf", "one token"), ("lf", "two rows sharing two tokens first"),
    ("sdpa", "prefixes in proportion")])
def test_a_batch_that_breaks_the_premise_repeats_pairs_the_reading_charges(case, broken):
    """The premise is a limit, held as a fact: at a batch that breaks it, the reading charges products whose MACs read
    one pair twice past the ones that repeat at the held batch (the first position's, which `keep` leaves out). Where
    a positional table reaches the rows, as lf's rope tables do, the prefixes in proportion are no break."""
    held = repeating(case, BATCH)
    assert repeating(case, torch.tensor(BROKEN[broken])) - held
    if case == "lf":
        assert not repeating(case, torch.tensor(BROKEN["prefixes in proportion"])) - held
