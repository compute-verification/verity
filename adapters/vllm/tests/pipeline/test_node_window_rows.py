"""verity_vllm.pipeline.node_window_rows: a rank's node window committed (`commit_rank_window`), on two toy ranks of one
request: a ``Gemm_v2{K=16,N=16}`` chain of two layers, two decode steps and two requests a rank, each layer's ``w`` a
registered read of the rank's weights, the prompt's rows a boundary read and, at rank 0, the served ids too (as
`node_window_reads.boundary_reads` declares them), joined as `node_window_join.join` joins rank windows. A test's values
stand in for `served_values`: each unit's rows by the chain's wiring."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.ops import gemm as G
from verity.commitments.periods import interface as IF
from verity.protocols.resource_accounting.compute.pouw.prover.serving import registered_commit

from verity_vllm.pipeline import node_window_rows as NWR
from verity_vllm.pipeline import window_replay as WRP
from verity_vllm.program.predict import node_window_join as NJ

GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
GID = descriptor_id(GEMM)
REQUESTS, STEPS, LAYERS = 2, 2, 2
LAYER = REQUESTS * STEPS
UNITS = LAYERS * LAYER
WORLD, CAP = 2, 2
DESIGN = {"law": "stratified", "k": 2}
TP_ROOT = "ab" * 64
KEYS = ("run", "population", "partition_digest", "registration_digest", "seconds", "window")
WIRES = (IF.Wire(GID, "x", GID, "o0", at=LAYER, radix=(STEPS, REQUESTS), base=0, step=(REQUESTS, 1)),
         IF.Wire(GID, "x", GID, "o0", at=REQUESTS, radix=(STEPS - 1, REQUESTS), base=LAYER, step=(REQUESTS, 1)))
REGISTERED = IF.Registered({"w": LAYERS * 256}, reads=(IF.Read(GID, "w", "w", 0, (LAYERS, LAYER), 0, (256, 0)),))


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def rank(r: int) -> NJ.Rank:
    """Rank ``r``'s window: units in (layer, step, request) order, layer 1 reading layer 0's output at its step and
    request, layer 0 at step s + 1 reading layer 1's at step s, the first step's ``x`` the prompt's (a boundary read)
    and, at rank 0, the last units' outputs the served ids."""
    reads = [IF.Read(GID, "x", "prompt", 0, (REQUESTS,), 0, (16,))]
    if r == 0:
        reads.append(IF.Read(GID, "o0", "served", UNITS - REQUESTS, (REQUESTS,), 0, (16,)))
    circuit = IF.Circuit(f"toy/rank{r}", ((GEMM, UNITS),), wires=WIRES, registered=REGISTERED,
                         boundary_reads=tuple(reads),
                         order=IF.ordered({GID: UNITS}, WIRES, {GID: range(0, UNITS, REQUESTS)}))
    spans = {GID: tuple((a, REQUESTS, 0, (a // REQUESTS) % STEPS) for a in range(0, UNITS, REQUESTS))}
    return NJ.Rank(r, circuit, (), spans, STEPS, {})


class Values:
    """Each unit's rows (`node_window_values.Check`'s ``inputs`` and ``outputs``), from rank ``r``'s made-up weights and
    prompt through the chain."""

    def __init__(self, r: int):
        rng = np.random.default_rng(r)
        self.w = np.stack([_bf16(rng.standard_normal((16, 16)) / 4).reshape(256) for _ in range(LAYERS)])
        x, y = {(0, 0, q): _bf16(rng.standard_normal(16)) for q in range(REQUESTS)}, {}
        for s in range(STEPS):
            for layer in range(LAYERS):
                for q in range(REQUESTS):
                    if (layer, s) != (0, 0):
                        x[layer, s, q] = y[0, s, q] if layer else y[1, s - 1, q]
                    y[layer, s, q] = np.asarray(evaluate_batch(GEMM, x[layer, s, q][None], self.w[layer][None],
                                                               kernel="reference"), dtype=np.uint16).reshape(16)
        order = [(layer, s, q) for layer in range(LAYERS) for s in range(STEPS) for q in range(REQUESTS)]
        self.x, self.y = np.stack([x[k] for k in order]), np.stack([y[k] for k in order])
        self.ws = self.w[[k[0] for k in order]]

    def inputs(self, tid: str, units):
        u = np.asarray(units)
        return [self.x[u], self.ws[u]]

    def outputs(self, tid: str, units):
        return {"o0": self.y[np.asarray(units)]}


def weights(r: int) -> dict:
    return {"w": registered_commit("test/node-window", "w", Values(r).w, bytes(32), r, keep_rows=True)}


def commit(r: int, out, **edit):
    args = {"rank": rank(r), "values": Values(r), "run": f"{TP_ROOT}/rank{r}", "law": DESIGN, "cap": CAP,
            "registered": weights(r), "out": out, **edit}
    return NWR.commit_rank_window(args.pop("rank"), args.pop("values"), **args)


@pytest.fixture(scope="module")
def stated():
    ranks = [rank(r) for r in range(WORLD)]
    joined = NJ.join(ranks)
    st = IF.ranks_of({"ranks": NJ.ranks_of(ranks, joined, CAP)})
    IF.check_ranks(joined.circuit.layouts, st, joined.circuit.wires, joined.circuit.registered)
    return st


def test_each_rank_commits_the_window_the_join_names(stated, tmp_path):
    """Each rank's window, committed and written under its run, returns `tp.serving_rows.write`'s five keys and
    ``window``, the H of the ``circuit.json`` it wrote (`window_replay.window_digest`), which is that rank's entry in
    the joined record's ranks (`interface.ranks_of`); the two ranks' differ, rank 0's naming the served ids. The written
    ``window.json`` and ``partition.json`` are the period's: its run, population and partition digest. Every member
    reads ``w`` by reference (the rank's registered value's rows) and ``x`` from its writers but the first, whose ``x``
    is the prompt's, held."""
    got = {r: commit(r, tmp_path / f"rank{r}") for r in range(WORLD)}
    for r, res in got.items():
        d = tmp_path / f"rank{r}"
        assert tuple(res) == KEYS
        assert res["window"] == WRP.window_digest(json.loads((d / "circuit.json").read_bytes())) == stated.of[r].window
        assert (res["run"], res["population"]) == (f"{TP_ROOT}/rank{r}", UNITS)
        win, part = (json.loads((d / f).read_text()) for f in ("window.json", "partition.json"))
        assert (win["run"], win["population"]) == (res["run"], UNITS)
        assert win["registration_digest"] == res["registration_digest"]
        assert win["partition_digest"] == part["partition_digest"] == res["partition_digest"]
        assert part["law"] == DESIGN and isinstance(res["seconds"], dict)
        assert [m.get("value_rows") for m in win["members"]] == [["w"]] * (UNITS // CAP)
        assert [m.get("wired_rows") for m in win["members"]] == [None] + [["x"]] * (UNITS // CAP - 1)
    assert got[0]["window"] != got[1]["window"]


@pytest.mark.parametrize("edit, name", [
    ({"rank": replace(rank(1), circuit=replace(rank(1).circuit, boundary_reads=(),
                                               ingress=(IF.Ingress(GID, "x", 0, REQUESTS, 0),)))}, "ingress"),
    ({"cap": None}, "cap"),
    ({"cap": 0}, "cap"),
    ({"registered": {}}, "registered"),
    ({"registered": {**weights(1), "v": weights(1)["w"]}}, "registered"),
    ({"registered": {"w": registered_commit("test/node-window", "w", Values(1).w.reshape(LAYERS * 2, 128), bytes(32), 1,
                                            keep_rows=True)}}, "registered"),
], ids=["ingress", "no-cap", "cap-0", "registered-missing", "registered-extra", "registered-shape"])
def test_refused_by_name_before_anything_is_written(edit, name, tmp_path):
    """A rank window that declares an ingress (its prompt is a boundary read), a cap that is not a positive count, and
    registered values that are not the ones its reads read (one missing, one it doesn't register, one of another row
    shape) are refused (ValueError) by name, and nothing is written under ``out``."""
    out = tmp_path / "rank1"
    with pytest.raises(ValueError, match=rf"^rank 1: {name}"):
        commit(1, out, **edit)
    assert not out.exists()
