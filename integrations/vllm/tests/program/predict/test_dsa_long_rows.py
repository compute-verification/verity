"""verity_vllm.program.predict.dsa past INDEX_TOPK: a row of more than INDEX_TOPK keys selects its top-k as the Build's `vllm_dsa`
rule does, in the circuit: `DsaTopKMask` marks the logits' top-k set, `DsaButterflySettings` sets the concentrator for it, and
sparse MLA reads the rows `DsaButterflyRows` routes at those settings, in ascending index order (the canonical row, which an engine
under `VERITY_DSA_TOPK=canonical` serves).  The Program has no `run` input and no checks.

GLM-5.3's checkpoint is cut to its first 4 layers (three full indexers, then one that shares the third's top-k) and its
INDEX_TOPK to 4, so a few rows reach past it without the 2,048 keys of the served one; one test keeps the served INDEX_TOPK and
reads its rows' history as an Input."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from verity.ml.circuits.refs import runs
from verity.ml.circuits.types import Array, Tuple as TupleT, Value
from verity.ml.definitions.registry.dsa_topk import waksman_apply, waksman_switches
from verity_vllm.check.batch import own
from verity_vllm.engine.profiles.engine_classes import DSA_TOPK_ENV
from verity_vllm.program import workload as W
from verity_vllm.program.predict import score as S
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
#: what a row past INDEX_TOPK calls that a row within it doesn't
LONG = ("DsaTopKMask_v1", "DsaButterflySettings_v1", "DsaButterflyRows_v1")
#: the relation form's calls, which no Program makes
RELATION = ("DsaWaksmanOrder_v1", "DsaTopKLongRow_v1", "DsaWaksmanRows_v1", "DsaTopKOrder_v1")


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


def _families(nodes) -> Counter:
    return Counter(n.fn.id.split("{", 1)[0] for n in nodes)


def _calls(r, family: str) -> list[tuple[int, object]]:
    return [(i, n) for i, n in enumerate(r.program.fn.body.nodes) if n.fn.id.startswith(family + "{")]


def _reads(node, arg: int) -> list[int]:
    """The nodes whose outputs argument `arg` of `node` reads."""
    return sorted({x[1] for x in runs(node.args[arg].refs) if x[0] == "n"})


def _order(L: int, p: int) -> list[int]:
    """An order of TOPK of the row's p + 1 keys, a different one per (layer, position): `run_bits` routes any order."""
    return list(dict.fromkeys([(p - k * (L + 1)) % (p + 1) for k in range(TOPK)] + list(range(p + 1))))[:TOPK]


def test_a_row_past_index_topk_selects_its_top_k_from_its_logits(phi):
    """A step of 6 rows: rows 4 and 5 have 5 and 6 keys.  Each full indexer layer marks each such row's top-k set from that row's
    logits (DsaTopKMask) and computes the concentrator's settings from the mask (DsaButterflySettings); every layer, the shared
    one included, reads the TOPK rows DsaButterflyRows routes at those settings, and the sparse MLA reads TOPK rows.  No `run`
    input, no checks: the Program returns the step's sample alone."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    params, ret = r.program.fn.signature()
    names = [n for n, _ in params]
    assert "run" not in names and names[-1] == "weights" and ret == Array(1, I32) and r.checks is None
    fam = _families(r.program.fn.body.nodes)
    assert [fam[f] for f in LONG] == [6, 6, 8] and fam["DsaTopKShortRow_v1"] == 12 and fam["DsaMqaLogits_v3"] == 18
    assert [fam[f] for f in RELATION] == [0] * len(RELATION)
    nodes = r.program.fn.body.nodes
    for i, n in _calls(r, "DsaTopKMask_v1"):
        (lg,) = _reads(n, 0)
        assert nodes[lg].fn.id == f"DsaMqaLogits_v3{{N={n.fn.bindings['N']},H=32,D=128}}", "the mask reads the row's own logits"
        assert nodes[i + 1].fn.id.startswith("DsaButterflySettings_v1{") and _reads(nodes[i + 1], 0) == [i]
    assert [n.fn.bindings["T"] for n in r.program.fn.body.nodes if n.fn.id.startswith("SparseMlaSplit_v1{")] == [
        min(i + 1, TOPK) for i in range(6)] * 4
    src = r.program.fn.bindings["SRC"]["inputs"]
    assert src[0] == [6] and len(src) == len(params) - 1, "the SRC inputs are the wrapper's, without the weights"


def test_the_shared_layer_reads_the_last_indexer_layers_settings(phi):
    """Layer 3 (`shared`) selects no top-k of its own: its DsaButterflyRows read the settings layer 2 computed at each row, as the
    Build's IndexShare layer reads the last indexer layer's selection."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    settings = [i for i, _ in _calls(r, "DsaButterflySettings_v1")]
    read = [_reads(n, 0) for _, n in _calls(r, "DsaButterflyRows_v1")]
    assert read == [[settings[k]] for k in (0, 1, 2, 3, 4, 5, 4, 5)]


def test_a_row_past_index_topk_under_an_engine_without_the_canonical_top_k_is_refused(phi):
    """An engine whose stated environment lacks `VERITY_DSA_TOPK=canonical` serves the kernels' own order past INDEX_TOPK, not the
    ascending row DsaButterflyRows reads: its row past INDEX_TOPK is refused by name.  A registration that states no environment
    builds it, and a step within INDEX_TOPK builds under any."""
    def at(env: dict | None) -> Phi:
        eng = {k: v for k, v in phi.engine.items() if k != "env"}
        return Phi.from_json({**phi.to_json(), "engine": eng if env is None else {**eng, "env": env}})
    with pytest.raises(Unexpressible, match=re.escape("a row past INDEX_TOPK (4) keys under an engine whose environment sets "
                                                      "VERITY_DSA_TOPK=None, not {'VERITY_DSA_TOPK': 'canonical'}")):
        build_program(at({"VLLM_BATCH_INVARIANT": "0"}), "step", M=5, pos0=0, with_desc=False)
    assert _families(build_program(at(None), "step", M=5, pos0=0, with_desc=False).program.fn.body.nodes)["DsaTopKMask_v1"] == 3
    assert _families(build_program(at({"VLLM_BATCH_INVARIANT": "0"}), "step", M=TOPK, pos0=0,
                                   with_desc=False).program.fn.body.nodes)["DsaTopKMask_v1"] == 0


def test_no_row_past_index_topk_calls_the_mask_form(phi):
    r = build_program(phi, "step", M=TOPK, pos0=0, with_desc=False)
    params, ret = r.program.fn.signature()
    assert ret == Array(1, I32) and r.checks is None and [n for n, _ in params][-1] == "weights"
    assert [_families(r.program.fn.body.nodes)[f] for f in LONG] == [0, 0, 0]


@pytest.fixture
def served(tmp_path) -> Phi:
    p = _first_layers(_row_phi(GLM53, "GLM53", tmp_path), 4)
    assert p.hf_config["index_topk"] == 2048
    return Phi.from_json({**p.to_json(), "engine": _canonical(p.engine)})


@pytest.mark.parametrize("C", [2047, 2049, 32767])
def test_a_decode_row_at_the_served_index_topk_selects_its_top_k_at_its_own_keys(served, C):
    """At the served INDEX_TOPK (2,048), a request a Commit window carries in at decode C reads its C positions as an Input and
    its decode row has C + 1 keys: 2,048 stay within INDEX_TOPK, and 2,050 or 32,768 select the top 2,048 of the row's own
    logits in the mask form at N = C + 1.  Every layer's sparse MLA reads 2,048 rows."""
    r = build_program(served, "request", LP=C, T=1, C=C, with_desc=False)
    n, nodes = C + 1, r.program.fn.body.nodes
    fam = _families(nodes)
    assert "run" not in [x for x, _ in r.program.fn.signature()[0]] and r.checks is None
    assert [m.fn.bindings["T"] for m in nodes if m.fn.id.startswith("SparseMlaSplit_v1{")] == [2048] * 4
    if n <= 2048:
        assert [fam[f] for f in LONG] == [0, 0, 0] and fam["DsaTopKShortRow_v1"] == 3
        return
    assert [fam[f] for f in LONG] == [3, 3, 4] and fam["DsaTopKShortRow_v1"] == 0
    assert {m.fn.id for f in LONG for _, m in _calls(r, f)} == {
        f"DsaTopKMask_v1{{N={n},TOPK=2048}}", f"DsaButterflySettings_v1{{N={n},K=2048}}", f"DsaButterflyRows_v1{{N={n},K=2048,W=576,E=16}}"}
    for _, m in _calls(r, "DsaTopKMask_v1"):
        (lg,) = _reads(m, 0)
        assert nodes[lg].fn.id.startswith("DsaMqaLogits_") and nodes[lg].fn.bindings["N"] == n


def test_run_bits_route_each_served_order():
    """`run_bits` (the Commit's and the replay's `run` layout) puts each field's Waksman settings at its offset: through them the
    network routes rows 0..P to the order the served indexer wrote for that row.  An order missing, or one of other than
    INDEX_TOPK slots, is refused by name."""
    keys = [(L, p) for L in range(3) for p in (4, 5, 6)]
    fields, at = [], 0
    for L, p in keys:
        fields.append({"name": run_field(L, p), "offset": at, "leaves": waksman_switches(p + 1, TOPK)})
        at += fields[-1]["leaves"]
    layout, orders = {"fields": fields}, {k: _order(*k) for k in keys}
    bits = run_bits(layout, orders, TOPK)
    assert len(bits) == at and set(bits) <= {0, 1}
    for f, (L, p) in zip(fields, keys):
        assert waksman_apply(bits[f["offset"]:f["offset"] + f["leaves"]], list(range(p + 1)), TOPK) == orders[(L, p)]
    with pytest.raises(KeyError, match=re.escape("no served top-k order for layer 2's row at position 6")):
        run_bits(layout, {k: v for k, v in orders.items() if k != (2, 6)}, TOPK)
    with pytest.raises(ValueError, match=re.escape("an order of 3 slots, not INDEX_TOPK 4")):
        run_bits(layout, {**orders, (0, 4): orders[(0, 4)][:3]}, TOPK)


def test_a_node_program_past_index_topk_selects_each_ranks_top_k(phi):
    """At TP8 the node Program joins the ranks' forwards, each selecting its rows past INDEX_TOPK as its rank Program does."""
    r = build_node_program(phi, "step", M=6, pos0=0, with_desc=False)
    fam = _families(r.node_fn.body.nodes)
    assert r.world == 8 and [fam[f] for f in LONG] == [6 * 8, 6 * 8, 8 * 8] and r.program.fn.signature()[1] == Array(8, I32)


@pytest.mark.slow
def test_gp01_composes_a_rank_program_past_index_topk_beside_one_within_it(phi):
    """At TP8 GP-01 composes a request past INDEX_TOPK with one within it, and R1's labels hold every request-local Input of each
    to its own request: the long one's selection reads its own logits alone.  What no cheaper test can: that the composition and
    R1 take a rank Program that selects past INDEX_TOPK."""
    canon: dict = {}
    rs = [build_program(phi, "request", LP=lp, T=0) for lp in (5, 3)]
    assert [_families(r.program.fn.body.nodes)["DsaTopKMask_v1"] for r in rs] == [3, 0]
    comps = [request_component(r, canon) for r in rs]
    assert not any("run" in c.groups for c in comps)
    wl = W.WorkloadProgram([W.WorkloadRequest(f"r{i}", i, lp, 1, 0, c, T=0) for i, (lp, c) in enumerate(zip((5, 3), comps))], lag=1)
    params, ret = wl.fn.signature()
    assert ret == TupleT(Array(1, I32), Array(1, I32))
    labels, _ = workload_labels(wl)
    for i in (0, 1):
        held = {n: lab for (n, _), lab in zip(params, labels) if n.startswith(f"request[r{i}].")}
        assert held and set(held.values()) == {own(i)}, held


@pytest.mark.slow
def test_class_g_compiles_a_prompt_past_index_topk_at_its_launch_context(tmp_path):
    """Class G (about 15 s): a prompt of 10 tokens prefilled in two chunks beside a short request, then decoded, its rows past
    INDEX_TOPK from position 4.  The step compiler builds each request's Program at its launch context (R1, R2 and R3 hold), the
    long one selecting its rows past INDEX_TOPK, and that Program is `build_program`'s at the unit's launch batch and chunks.
    The fast tests build each Program alone; this one holds the batch rules to a Program that selects past INDEX_TOPK."""
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    from verity_vllm.program.predict.steps import Registered, compile_steps
    reg = _class_g(tmp_path, model={"hf_config": hf})
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    advice, ingress = _session([(10, 3, {**GREEDY, "max_tokens": 3}), (3, 2, {**GREEDY, "max_tokens": 2})],
                               [[(0, 0, 6)], [(0, 6, 4), (1, 0, 3)], [(0, 10, 1), (1, 3, 1)], [(0, 11, 1)]])
    c = compile_steps(reg, advice, ingress)
    long, short = c.requests
    assert (long.LP, long.T, long.chunks, short.LP, short.chunks) == (10, 2, (6, 4), 3, ())
    sigs = sorted((_families(p.fn.body.nodes)["DsaTopKMask_v1"], str(p.fn.signature()[1])) for p in c.programs.values())
    assert sigs == [(0, str(Array(2, I32))), (3 * 8, str(Array(3, I32)))], "rows 4..11 of the long one, at each indexer layer"
    alone = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": long.policy}), "request", LP=10, T=2, dead_free=True,
                          batch=long.batch(), chunks=long.chunks)
    assert alone.digest in c.programs


def _step(*rows: tuple[int, int, int, int]):
    from verity_vllm.pipeline import window_advice as WA
    return WA.StepAdvice(tuple(WA.Row(i, "decode" if n == 1 and s >= lp else "prefill", s, n, 0, False, ()) for i, lp, s, n in rows))


@pytest.mark.slow
def test_class_g_carries_a_request_past_index_topk_with_its_indexer_history(tmp_path):
    """Class G (about 40 s): A (a 3-token prompt, 5 outputs: its rows past INDEX_TOPK from position 4) and B (2 and 3, within it)
    whole in one window are, call for call, their Programs across three: the first leaves A at position 6 and B at 3, the second
    runs B's last row with A carried through it idle, and the third runs A's last.  A's rows read the indexer's keys and scales
    past INDEX_TOPK, so the edges carry them for A alone: the first window's A returns its samples, then every layer's latent rows,
    index-K slots and scales so far, out-ports `edge.kv`, `edge.ik`, `edge.iks` and `edge.last`, while B's index-K history stays
    its own unread Inputs; the idle window passes A's fields through to its own edge, and the third reads them as A's history,
    `edge.ik` and `edge.iks` one field each.  Each window's Program selects the top-k of the rows it computes past INDEX_TOPK, and
    the advice's side (`pipeline.edge.fields`) holds each port's members as each window's Program returns them.  What no cheaper
    test can: that a window's edges carry a request past INDEX_TOPK whole, through a window it idles in."""
    from verity.ml.circuits.types import field_types
    from verity_vllm.pipeline import edge as E
    from verity_vllm.pipeline import window_advice as WA
    from verity_vllm.program.predict.steps import Registered, compile_steps
    from verity_vllm.program.workload import EDGE_PORTS
    from tests.program.predict.test_steps import CACHE_ON
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    reg = _class_g(tmp_path, engine=CACHE_ON, model={"hf_config": hf})
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    a, b = WA.RequestAdvice("a", 3, 5, "length"), WA.RequestAdvice("b", 2, 3, "length")
    steps = [_step((0, 3, 0, 3), (1, 2, 0, 2)), _step((0, 3, 3, 1), (1, 2, 2, 1)), _step((0, 3, 4, 1)), _step((0, 3, 5, 1)),
             _step((1, 2, 3, 1)), _step((0, 3, 6, 1))]
    carried_a = WA.Carried(0, 6, 4, ((0, 6),))
    out = WA.Advice((a, b), tuple(steps[:4]), (0,) * 6)
    idle = WA.Advice((a, b), (steps[4],), (0,), (carried_a, WA.Carried(0, 3, 2, ((0, 3),))))
    into = WA.Advice((a,), (steps[5],), (0,), (carried_a,))
    assert WA.follows(out, idle) == [(0, 0), (1, 1)] and WA.follows(idle, into) == [(0, 0)]
    ingress = {"a": {**GREEDY, "max_tokens": 5, "prompt": [1, 2, 3]}, "b": {**GREEDY, "max_tokens": 3, "prompt": [4, 5]}}
    cs = [compile_steps(reg, adv, {q.id: ingress[q.id] for q in adv.requests})
          for adv in (WA.Advice((a, b), tuple(steps), (0,) * 8), out, idle, into)]
    assert all(c.batch_rule is None for c in cs) and cs[2].idle == (0,)
    prog = [{u.request: c.programs[u.digest] for u in c.requests if u is not None} for c in cs]
    for r in (0, 1):
        assert [n.fn.id for n in prog[0][r].fn.body.nodes] == [n.fn.id for p in prog[1:] if r in p for n in p[r].fn.body.nodes]
    assert [u.edge for u in cs[1].requests] == [("prefix", "prefix_ik", "prefix_iks"), ("prefix",)]
    ret = field_types(prog[1][0].fn.signature()[1])
    assert ret[0] == Array(4, I32) and ret[1:] == (Array(4, Array(6, Array(576, Value(16)))), Array(3, Array(6, Array(128, Value(8)))),
                                                   Array(3, Array(6, Value(32))))
    wls = [c.workload for c in cs]
    assert field_types(wls[1].fn.signature()[1])[0] == Array(4, I32)
    assert wls[1].edge.outs == wls[2].edge.ins and [[g for g, _, _ in gs] for _, gs in wls[2].edge.ins] == [
        ["prefix", "prefix_ik", "prefix_iks", "last_token"], ["prefix", "last_token"]]
    assert wls[2].edge.outs == wls[2].edge.ins[:1] == wls[3].edge.ins, "the idle window hands A's fields on as they came in"
    for adv, wl in ((out, wls[1]), (idle, wls[2]), (into, wls[3])):
        assert E.fields(adv, TOPK) == {p: tuple(i for i, gs in (wl.edge.outs if wl.edge else ()) if g in [h for h, _, _ in gs])
                                       for p, g in EDGE_PORTS}, "the advice's side holds the fields the Program's edge returns"
    rids = {w: [r.request_id for r in wls[w].requests] for w in (2, 3)}
    for w, held in ((2, [2, 1, 1, 2]), (3, [1, 1, 1, 1])):
        params = dict(wls[w].fn.signature()[0])
        assert wls[w].ports == ("edge.kv", "edge.ik", "edge.iks", "edge.last")
        assert [len(field_types(params[p])) for p in wls[w].ports] == held
        assert [t.leaves for t in field_types(params["edge.ik"])] == [ret[2].leaves]
    names = [n for n, _ in wls[2].fn.signature()[0] if n.startswith("request[") and ".peers_" not in n]
    assert names == [f"request[{rids[2][0]}].{g}" for g in ("prompt", "prefix_ik", "prefix_iks")], "B's index-K history is its own"
    assert [n for n, _ in wls[3].fn.signature()[0] if n.startswith("request[") and ".peers_" not in n] == [f"request[{rids[3][0]}].prompt"]
    for p, pos in zip(prog, ({4, 5, 6}, {4, 5}, set(), {6})):
        assert _families(p[0].fn.body.nodes)["DsaTopKMask_v1"] == 3 * len(pos) if 0 in p else not pos
        assert 1 not in p or _families(p[1].fn.body.nodes)["DsaTopKMask_v1"] == 0


@pytest.mark.slow
def test_a_class_g_chunked_prompt_past_index_topk_traced_build_is_its_compiled_program(tmp_path):
    """The prompt above (10 tokens in chunks of 6 and 4 beside a short request, then two decodes; rows past INDEX_TOPK from
    position 4), compiled at rank 0 under the cut checkpoint, and its trace: the command `trace_commands` gives it, run on the
    pinned vLLM with the cut config.json in the hub and no CUDA device visible, as the row driver runs a Build.  `compare` finds the
    two digests equal.  What no cheaper test can: that the Build's `vllm_dsa` rule, on vLLM's own graph, traces the mask form the
    predictor writes past INDEX_TOPK.  It needs the pinned vLLM (a pod's venv), so it skips without it."""
    pytest.importorskip("vllm")
    from verity_vllm.program.predict.steps import Registered, compare, compile_steps, trace_commands
    hub = tmp_path / "hub"
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    reg = _class_g(hub, model={"hf_config": hf})
    cp = Path(S.hf_config_path(str(hub), reg.phi.model["repo"], reg.phi.model["revision"]))
    assert cp.is_file(), "the config.json `_class_g` wrote, which the trace loads"
    cp.write_text(json.dumps(hf))
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    advice, ingress = _session([(10, 3, {**GREEDY, "max_tokens": 3, "ignore_eos": True}), (3, 2, {**GREEDY, "max_tokens": 2, "ignore_eos": True})],
                               [[(0, 0, 6)], [(0, 6, 4), (1, 0, 3)], [(0, 10, 1), (1, 3, 1)], [(0, 11, 1)]])
    c = compile_steps(reg, advice, {"q0": ingress["q0"]}, partial=True)
    u = c.requests[0]
    assert (u.LP, u.T, u.chunks) == (10, 2, (6, 4))
    assert _families(c.programs[u.digest].fn.body.nodes)["DsaTopKMask_v1"] == 3 * 8
    s = c.summary()
    (cmd,) = trace_commands(s, reg, str(tmp_path / "traces"))
    assert json.loads(cmd[cmd.index("--launch-context") + 1])["chunks"] == [6, 4]
    subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", *cmd[1:]], check=True,
                   env={**os.environ, "HF_HUB_CACHE": str(hub), "CUDA_VISIBLE_DEVICES": ""})
    got = compare(s, str(tmp_path / "traces"))
    assert (got["equal"], got["mismatches"]) == (1, 0), got["units"]
