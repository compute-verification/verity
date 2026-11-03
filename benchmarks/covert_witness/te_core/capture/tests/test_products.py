"""te_core.capture.products: a step's matrix products as te_core.io_bound reads them. On test_script.py's program at
H = 256 (past the K at which io_bound's GEMM rate rule changes, 3^1.5·√(k/32), 235 at the console's k), the step
recorded from the start (te_core.capture.script), whose weights the measured step reads through the versions the
previous step's update wrote, has the rows of the same step captured by capture_step, and those are io_bound.rows_of's."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as tF
from torch import nn

from te_core import formalism, io_bound, policy
from te_core.capture import capture_step, products, script

PROGRAM = Path(__file__).parent / "recorded" / "islands" / "toy" / "train.py"
V, H, B, CTX = 16, 256, 4, 8
SHAPES = {"emb": [V, H], "fc": [H, H], "head": [V, H], "frozen": [V, H]}


class Toy(nn.Module):
    """The program's model: an embedding, one tanh layer and an untied head."""

    def __init__(self):
        super().__init__()
        self.emb, self.fc, self.head = (nn.Parameter(torch.randn(*SHAPES[n])) for n in ("emb", "fc", "head"))

    def forward(self, x):
        return tF.linear(torch.tanh(tF.linear(tF.embedding(x, self.emb), self.fc)), self.head)


def comparable(block: dict) -> list:
    return sorted(json.dumps(r, sort_keys=True) for r in block["rows"])


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "STREAM_TOKENS", 10 ** 5)
    (tmp_path / "toy.start.json").write_text(json.dumps({"config": {"vocab": V, "d_model": H},
                                                         "tensors": {n: ["float32", s] for n, s in SHAPES.items()}}))
    return script.record(PROGRAM, ["--batch", str(B), "--update", "inplace"], step=1, starts=tmp_path,
                         wall=[0.1] * 8, step_tokens=B * CTX)


def test_a_recorded_steps_rows_are_the_captured_steps_and_io_bounds(recorded):
    """The forward's head product is fresh and its fc product an expansion (io_bound's rule pays a lookup at the token
    ids at no rate); in the backward three products are fresh and one an expansion. The trace keeps the rows with the
    capture's version, and the order is unknown."""
    k = policy.CONSOLE[1]
    model = Toy()
    ref = capture_step(model, torch.optim.SGD(model.parameters(), lr=1e-3),
                       lambda m, b: tF.cross_entropy(m(b[0]).flatten(0, 1), b[1].flatten()),
                       (torch.zeros(B, CTX, dtype=torch.long), torch.zeros(B, CTX, dtype=torch.long)))
    got, want = products.rows(recorded, k), products.rows(ref, k)
    assert comparable(got) == comparable(want)
    assert recorded.formalism["products"] == got and (got["capture"], got["order"]) == (products.CAPTURE, "unknown")
    assert got["gates"] == recorded.formalism["gates"] and want["gates"] == io_bound.step_gates(ref, "bfloat16")
    dims = ("batch", "m", "k", "n")
    by_dims = {(r.phase, *(r.dims[x] for x in dims)): r for r in io_bound.rows_of(ref, "bfloat16", k)}
    assert len(by_dims) == len(got["rows"])
    kinds = []
    for r in got["rows"]:
        o = by_dims[(r["phase"], *(r["dims"][x] for x in dims))]
        assert formalism.product_kind(r) == o.kind and r["count"] == 1
        assert (tuple(r["widths"]), tuple(r["rates"])) == (o.widths, o.rates)
        assert r["k_whole"] == (o.k_whole if o.k_whole != float("inf") else None)
        kinds.append((r["phase"], o.kind))
    assert sorted(kinds) == sorted([("forward", "fresh"), ("forward", "expansion"), ("backward", "expansion")]
                                   + [("backward", "fresh")] * 3)


class Fan(nn.Module):
    """Three trained layers; the second's output reaches the third's operand through one column repeated across the
    row (`fan`), through a row norm (`norm`), or directly."""

    def __init__(self, d: int, via: str) -> None:
        super().__init__()
        self.w1, self.w2, self.w3 = (nn.Parameter(torch.randn(d, d)) for _ in range(3))
        self.via = via

    def forward(self, x):
        h = (x @ self.w1) @ self.w2
        if self.via == "fan":
            h = h[:, :1].expand(-1, h.shape[1])
        elif self.via == "norm":
            h = h / h.norm(dim=-1, keepdim=True)
        return h @ self.w3


@pytest.mark.parametrize("via, broken", (("direct", []), ("norm", []),
                                         ("fan", [{"op": "expand", "fan_out": 16.0, "count": 1}])))
def test_p_match_fails_where_one_entry_is_handed_to_several_on_the_way_into_an_operand(via, broken):
    """A product's output carried into the next product's operand one to one (directly, or through a row norm whose
    statistic is charged nothing and whose broadcast its reduction pays for) keeps P-match, and the step's whole-making
    system reads the class (K = 16) feeding the third layer's operand; one column repeated across its row hands each
    entry made whole to 16, so P-match fails at the expand and the fresh-products term is unmeasured with the reason."""
    torch.manual_seed(0)
    m = Fan(16, via)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: tF.mse_loss(mm(b[0]), b[1]),
                     (torch.randn(6, 16), torch.randn(6, 16)))
    k = policy.CONSOLE[1]
    got = products.rows(g, k)
    assert got["p_match"] == broken
    third = [r for r in got["rows"] if r["phase"] == "forward" and all(r["affected"]) and r["fed"]]
    assert bool(third) == (not broken) and all([16, 1.0] in r["fed"] for r in third)
    p = formalism.products_of([(1, g)], k)
    assert p["p_match"] == broken and p["system"] is not None
    m_ = formalism.Measured(parts=(formalism.Part(gates=got["gates"], tokens=6, token_gates=got["gates"],
                                                  affected_gates=got["gates"]),), tau_bits=0.0, a_bits=0.0, u=0.0,
                            u_in=0.0, delta_bits=0.0, products=p)
    term, why, _ = formalism.fresh_products(m_, *policy.CONSOLE)
    assert (term is None and "breaks P-match" in why and "expand (fan-out 16" in why) if broken else why is None


class MatVec(nn.Module):
    """A layer plus a bias made by a matrix-vector product of its weight and a trained vector (aten.mv)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.w, self.v = nn.Parameter(torch.randn(d, d)), nn.Parameter(torch.randn(d))

    def forward(self, x):
        return x @ self.w + torch.mv(self.w, self.v)


def test_a_bf16_steps_matrix_vector_products_are_priced_as_gemms():
    """aten.mv is a GEMM by its name (te_core.opnames' GEMM_LIKE), as capture.structure counts it, so a bf16 step
    with torch.mv is priced at the bf16 GEMM's Definition through the capture's rows and io_bound, never as an
    elementwise op, whose price has two Definitions that a GEMM's can't take."""
    from te_core import gates

    torch.manual_seed(0)
    m = MatVec(16).to(torch.bfloat16)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: tF.mse_loss(mm(b[0]), b[1]),
                     (torch.empty(6, 16, dtype=torch.bfloat16), torch.empty(6, 16, dtype=torch.bfloat16)))
    got = products.rows(g, policy.CONSOLE[1])
    mv = [r for r in got["rows"] if r["op"] == "mv"]
    assert mv and all(r["gates_per_mac"] == gates.per_mac("gemm/bf16") for r in mv)
    assert {r.op for r in io_bound.rows_of(g, "bfloat16")} == {"aten.mm.default", "aten.mv.default"}
    assert got["gates"] == io_bound.step_gates(g, "bfloat16") > 0
