"""verity_vllm.program.predict.dsa past INDEX_TOPK: a row of more than INDEX_TOPK keys reads its top-k order as the Build's
`vllm_dsa` rule does, from a field of the Program's `run` input, and the Program returns the bit that the order is the top-k set
in ascending index order (the canonical row, which an engine under `VERITY_DSA_TOPK=canonical` serves).

GLM-5.3's checkpoint is cut to its first 4 layers (three full indexers, then one that shares the third's top-k) and its
INDEX_TOPK to 4, so a few rows reach past it without the 2,048 keys of the served one."""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from verity.ml.circuits.refs import runs
from verity.ml.circuits.types import Array, Tuple as TupleT, Value
from verity.ml.definitions.registry.dsa_topk import waksman_apply, waksman_switches
from verity_vllm.check.batch import own
from verity_vllm.engine.profiles.engine_classes import DSA_TOPK_ENV
from verity_vllm.program import workload as W
from verity_vllm.program.predict.batch import workload_labels
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.common import Unexpressible
from verity_vllm.program.predict.dsa import run_bits, run_field
from verity_vllm.program.predict.node import build_node_program
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.score import request_component
from tests.program.predict.test_node import _first_layers
from tests.program.predict.test_predict import GLM53, _row_phi
from tests.program.predict.test_steps import GREEDY, HF_CONFIG_DIR, _class_g, _session

TOPK = 4
I32 = Value(32)
BIT = Value(1)


def _short(hf: dict) -> dict:
    hf.update(num_hidden_layers=4, indexer_types=hf["indexer_types"][:4], mlp_layer_types=hf["mlp_layer_types"][:4], index_topk=TOPK)
    return hf


def _canonical(engine: dict) -> dict:
    return {**engine, "env": {**engine.get("env", {}), **DSA_TOPK_ENV}}


@pytest.fixture
def phi(tmp_path) -> Phi:
    p = _first_layers(_row_phi(GLM53, "GLM53", tmp_path), 4)
    assert p.hf_config["indexer_types"] == ["full"] * 3 + ["shared"]
    return Phi.from_json({**p.to_json(), "model": {**p.model, "hf_config": _short(p.hf_config)}, "engine": _canonical(p.engine)})


def _families(r) -> Counter:
    return Counter(n.fn.id.split("{", 1)[0] for n in r.program.fn.body.nodes)


def _calls(r, family: str) -> list:
    return [n for n in r.program.fn.body.nodes if n.fn.id.startswith(family + "{")]


def _order(L: int, p: int) -> list[int]:
    """An order of TOPK of the row's p + 1 keys, a different one per (layer, position): `run_bits` routes any order, the served
    canonical one among them."""
    return list(dict.fromkeys([(p - k * (L + 1)) % (p + 1) for k in range(TOPK)] + list(range(p + 1))))[:TOPK]


def test_a_row_past_index_topk_reads_its_order_from_run_and_returns_its_check(phi):
    """A step of 6 rows: rows 4 and 5 have 5 and 6 keys. Each full indexer layer reads one `run` field per such row, in first-read
    order, routes the row indices through it (DsaWaksmanOrder) and checks them against its logits (DsaTopKLongRow); every layer,
    the shared one included, reads the TOPK rows DsaWaksmanRows routes at those settings, and the sparse MLA reads TOPK rows."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    params, ret = r.program.fn.signature()
    names = [n for n, _ in params]
    assert names[-2:] == ["run", "weights"] and names.count("run") == 1
    fields = params[-2][1].fields
    assert [f for f, _ in fields] == [run_field(L, p) for L in range(3) for p in (4, 5)]
    assert [t for _, t in fields] == [Array(waksman_switches(p + 1, TOPK), BIT) for _ in range(3) for p in (4, 5)]
    assert ret == TupleT(Array(1, I32), Array(6, BIT))
    assert (r.checks["output"], r.checks["accept"], len(r.checks["of"])) == (1, 1, 6)
    assert r.run["name"] == "run" and [f["name"] for f in r.run["fields"]] == [f for f, _ in fields]
    fam = _families(r)
    assert (fam["DsaWaksmanOrder_v1"], fam["DsaTopKLongRow_v1"], fam["DsaWaksmanRows_v1"], fam["DsaTopKShortRow_v1"]) == (6, 6, 8, 12)
    assert fam["DsaTopKOrder_v1"] == 0
    assert fam["DsaMqaLogits_v3"] == 18
    assert [n.fn.bindings["T"] for n in _calls(r, "SparseMlaSplit_v1")] == [min(i + 1, TOPK) for i in range(6)] * 4
    src = r.program.fn.bindings["SRC"]["inputs"]
    assert src[0] == [6] and len(src) == len(params) - 2, "the SRC inputs are the wrapper's, without `run` and the weights"


def test_the_shared_layer_reads_the_last_indexer_layers_settings(phi):
    """Layer 3 (`shared`) selects no top-k of its own: its DsaWaksmanRows read layer 2's `run` field at each row, as the Build's
    IndexShare layer reads the last indexer layer's selection."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    at = len(r.program.fn.signature()[0]) - 2
    offset = {f["name"]: f["offset"] for f in r.run["fields"]}
    read = [list(runs(n.args[0].refs)) for n in _calls(r, "DsaWaksmanRows_v1")]
    assert all(len(x) == 1 and x[0][:2] == ("p", at) for x in read), "each DsaWaksmanRows reads one field of `run`"
    assert [x[0][2] for x in read] == [offset[run_field(L, p)] for L in (0, 1, 2, 2) for p in (4, 5)]


def test_a_row_past_index_topk_under_an_engine_without_the_canonical_top_k_is_refused(phi):
    """An engine whose stated environment lacks `VERITY_DSA_TOPK=canonical` serves the kernels' own order past INDEX_TOPK, which
    DsaTopKLongRow refuses: its row past INDEX_TOPK is refused by name. A registration that states no environment builds it, and a
    step within INDEX_TOPK builds under any."""
    def at(env: dict | None) -> Phi:
        eng = {k: v for k, v in phi.engine.items() if k != "env"}
        return Phi.from_json({**phi.to_json(), "engine": eng if env is None else {**eng, "env": env}})
    with pytest.raises(Unexpressible, match=re.escape("a row past INDEX_TOPK (4) keys under an engine whose environment sets "
                                                      "VERITY_DSA_TOPK=None, not {'VERITY_DSA_TOPK': 'canonical'}")):
        build_program(at({"VLLM_BATCH_INVARIANT": "0"}), "step", M=5, pos0=0, with_desc=False)
    assert build_program(at(None), "step", M=5, pos0=0, with_desc=False).run is not None
    assert build_program(at({"VLLM_BATCH_INVARIANT": "0"}), "step", M=TOPK, pos0=0, with_desc=False).run is None


def test_no_row_past_index_topk_has_no_run_input(phi):
    r = build_program(phi, "step", M=TOPK, pos0=0, with_desc=False)
    params, ret = r.program.fn.signature()
    assert "run" not in [n for n, _ in params] and ret == Array(1, I32) and r.run is None and r.checks is None
    assert _families(r)["DsaWaksmanOrder_v1"] == 0


def test_run_bits_route_each_served_order(phi):
    """`run_bits` puts each field's Waksman settings at its offset: through them the network routes rows 0..P to the order
    the served indexer wrote for that row. An order missing, or one of other than INDEX_TOPK slots, is refused by name."""
    r = build_program(phi, "request", LP=6, T=1, with_desc=False)
    orders = {(L, p): _order(L, p) for L in range(3) for p in (4, 5, 6)}
    assert sorted(orders) == sorted((int(a), int(b)) for a, b in (re.findall(r"\d+", f["name"]) for f in r.run["fields"]))
    bits = run_bits(r.run, orders, TOPK)
    assert len(bits) == sum(f["leaves"] for f in r.run["fields"]) and set(bits) <= {0, 1}
    for f in r.run["fields"]:
        L, p = (int(x) for x in re.findall(r"\d+", f["name"]))
        assert waksman_apply(bits[f["offset"]:f["offset"] + f["leaves"]], list(range(p + 1)), TOPK) == orders[(L, p)]
    with pytest.raises(KeyError, match=re.escape("no served top-k order for layer 2's row at position 6")):
        run_bits(r.run, {k: v for k, v in orders.items() if k != (2, 6)}, TOPK)
    with pytest.raises(ValueError, match=re.escape("an order of 3 slots, not INDEX_TOPK 4")):
        run_bits(r.run, {**orders, (0, 4): orders[(0, 4)][:3]}, TOPK)


def test_a_node_program_past_index_topk_is_refused(phi):
    with pytest.raises(Unexpressible, match=re.escape("a node Program whose rows read more than INDEX_TOPK keys")):
        build_node_program(phi, "step", M=6, pos0=0, with_desc=False)


@pytest.mark.slow
def test_gp01_composes_rank_programs_that_read_run_and_holds_it_to_its_request(phi):
    """At TP8 a request past INDEX_TOPK has `run` before its weights and one that isn't doesn't, so the weights sit at different
    positions; GP-01 composes them, `run` request-local, and R1's labels hold each request's `run` to it."""
    comps = [request_component(build_program(phi, "request", LP=lp, T=0)) for lp in (5, 3)]
    assert "run" in comps[0].groups and "run" not in comps[1].groups
    wl = W.WorkloadProgram([W.WorkloadRequest(f"r{i}", i, lp, 1, 0, c, T=0) for i, (lp, c) in enumerate(zip((5, 3), comps))], lag=1)
    assert not wl.shared_positions_uniform
    params, ret = wl.fn.signature()
    assert ret == TupleT(TupleT(Array(1, I32), Array(3, BIT)), Array(1, I32))
    labels, _ = workload_labels(wl)
    held = {n: lab for (n, _), lab in zip(params, labels) if ".run" in n}
    assert held and all(n.startswith("request[r0].run") for n in held) and set(held.values()) == {own(0)}


@pytest.mark.slow
def test_class_g_compiles_a_prompt_past_index_topk_at_its_launch_context(tmp_path):
    """Class G (about 15 s): a prompt of 10 tokens prefilled in two chunks beside a short request, then decoded, its rows past
    INDEX_TOPK from position 4. The step compiler builds each request's Program at its launch context (R1, R2 and R3 hold), the
    long one reading `run` and returning its checks, and that Program is `build_program`'s at the unit's launch batch and chunks.
    The fast tests build each Program alone; this one holds the batch rules to a Program that reads `run`."""
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    from verity_vllm.program.predict.steps import Registered, compile_steps
    reg = _class_g(tmp_path, model={"hf_config": hf})
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    advice, ingress = _session([(10, 3, {**GREEDY, "max_tokens": 3}), (3, 2, {**GREEDY, "max_tokens": 2})],
                               [[(0, 0, 6)], [(0, 6, 4), (1, 0, 3)], [(0, 10, 1), (1, 3, 1)], [(0, 11, 1)]])
    c = compile_steps(reg, advice, ingress)
    long, short = c.requests
    assert (long.LP, long.T, long.chunks, short.LP, short.chunks) == (10, 2, (6, 4), 3, ())
    sigs = sorted(("run" in [n for n, _ in ps], str(ret)) for ps, ret in (p.fn.signature() for p in c.programs.values()))
    assert sigs == [(False, str(Array(2, I32))), (True, str(TupleT(Array(3, I32), Array(24, BIT))))]
    alone = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": long.policy}), "request", LP=10, T=2, dead_free=True,
                          batch=long.batch(), chunks=long.chunks)
    assert alone.digest in c.programs
