"""The value test of `circuit_properties.work`'s default reading, the circuit alone with nothing stated: on small
attention steps, evaluated word by word (`serve3_reference.materialize`), no two MACs a product it charges reads one
pair of values beyond the share a sequence's start leaves (`Work.keep`), where the batch repeats tokens within its rows.

Each step is `test_program_reading.loop`'s scan of three iterations of a training step under AdamW at random weights,
and the third iteration's products are the ones compared. The batch repeats a token within each row and starts no row
with a run of one token, nor with a prefix that repeats a shorter one: such a batch makes two positions' values equal
at every layer, which no reading of the circuit can tell (`work`'s limits). The steps:

- `sdpa`: an embedding, two layers of one-head causal attention (`scaled_dot_product_attention`) and a head.
- `after_mask`: the same with the mask an explicit constant added to the scores, and a product of the masked scores by
  a weight right after the add, which no reading may charge: the add alone leaves two positions of one token one row
  shifted, so that product reads one pair at both.
- `rope`: `sdpa` with rotary tables, inputs the Program doesn't state.
- `relu`: `sdpa` with a ReLU MLP (`clamp_min`) after the second layer, whose down-projection no reading may charge:
  two of its inputs' negative words give one 0, so an op with a specific argument does not give a specific value by that alone.
- `lf`: the lf model's step at its smallest size.

In every step the first layer's P·V reads repeated pairs, where two keys of one token hold one value row, and is left
out. With the rope tables stated positional (`work.Layout`), the reading charges more, all of it the same invariant's.
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
ROPE = WK.Layout("batch", *BATCH.shape, ("buf_cos", "buf_sin"))
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


def evaluated(make, n: int = 3):
    """``make()``'s step on `BATCH` as `loop`'s scan of ``n`` iterations, and every gate's value at random weights."""
    opt = AdamW(lr=0.05)
    with torch.device("meta"):
        meta = make()
    step = step_program(meta, _square, opt, torch.empty(*BATCH.shape, dtype=torch.long, device="meta"))
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
            ins.append(_bits(torch.stack([BATCH] * n)))
        else:
            ins.append(_bits(bufs[name]))
    return step, prog, materialize(prog, ins)


def pairs(prog, tab, layout=None) -> dict:
    """By the step's product node: the reason it is left out ("" where charged) and each of its third iteration's MACs'
    pair of operand values."""
    by: dict = defaultdict(lambda: ["", []])
    for call, site, reason in WK.calls(prog, K0, layout=layout):
        if THIRD not in call.parent.path():
            continue
        got = [tuple(sorted((tab[g.operands[0]], tab[g.operands[1]]))) for g in call if g.prim.name == "F32Fma"]
        assert len(got) == site.macs
        by[site.node][0] = reason
        by[site.node][1].extend(got)
    return by


def repeats(ps: list) -> int:
    return len(ps) - len(set(ps))


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
        assert len(set(ps)) >= keep * len(ps)
    pv = first_pv(step, by)
    assert by[pv][0] and repeats(by[pv][1])
    if case == "after_mask":
        am = first_after_mask(step, by)
        assert by[am][0] == "gathered" and repeats(by[am][1])
    if case == "relu":
        down = first_after(step, by, r"SelectF32_v\d+", f"LinearRef_v1{{K={2 * D},N={D},")
        assert by[down][0] and repeats(by[down][1])
    if case in ("rope", "lf"):
        rope = pairs(prog, tab, ROPE)
        mine = {k for k, (reason, _ps) in by.items() if not reason}
        theirs = {k for k, (reason, _ps) in rope.items() if not reason}
        assert mine < theirs
        for k in theirs:
            assert len(set(rope[k][1])) >= keep * len(rope[k][1])
