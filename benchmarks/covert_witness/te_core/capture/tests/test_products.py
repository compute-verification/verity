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
