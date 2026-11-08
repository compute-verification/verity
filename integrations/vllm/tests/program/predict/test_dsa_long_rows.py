"""verity_vllm.program.predict.dsa past INDEX_TOPK: a row of more than INDEX_TOPK keys selects its top-k in the mask form, as
the Build's `vllm_dsa` rule does, with no `run` input and no checks: at the row's class (`vllm_dsa.long_row_keys`, its keys
rounded up to a multiple of INDEX_TOPK, the logits past its own `PAD_SCORE`) `DsaTopKMask_v1` of its logits, the settings
`DsaButterflySettings_v1` computes from that mask, and the TOPK history rows `DsaButterflyRows_v1` routes at them
(`DsaButterflyRowsPad_v1` below the class, the history and its pad two arguments), the top-k set's in ascending index order (the canonical row, which an engine under `VERITY_DSA_TOPK=canonical` serves).

GLM-5.3's checkpoint is cut to its first 4 layers (three full indexers, then one that shares the third's top-k) and its
INDEX_TOPK to 4, so a few rows reach past it without the 2,048 keys of the served one."""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import runs
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.registry import dsa_topk_mask as M
from verity.ml.definitions.registry.dsa_topk import topk_set
from verity_vllm.engine.profiles.engine_classes import CLASS_G_SERVED, DSA_TOPK_ENV
from verity_vllm.program.frontend.rules.vllm_dsa import PAD_SCORE, long_row_keys
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.common import Unexpressible
from verity_vllm.program.predict.node import build_node_program
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered, compile_steps
from tests.program.predict.test_node import _first_layers
from tests.program.predict.test_predict import GLM53, _row_phi
from tests.program.predict.test_steps import GREEDY, HF_CONFIG_DIR, _class_g, _session

TOPK = 4
I32 = Value(32)
BIT = Value(1)
MASK_FORM = ("DsaTopKMask_v1", "DsaButterflySettings_v1", "DsaButterflyRowsPad_v1")
RUN_FORM = ("DsaWaksmanOrder_v1", "DsaTopKLongRow_v1", "DsaWaksmanRows_v1")
CTX131K = os.environ.get("VERITY_VLLM_CTX131K", "")


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


def _no_run(r) -> bool:
    params, _ = r.program.fn.signature()
    return "run" not in [n for n, _ in params] and r.run is None and r.checks is None


def test_a_row_past_index_topk_selects_its_top_k_in_the_mask_form(phi):
    """A step of 6 rows: rows 4 and 5 have 5 and 6 keys, both of class 8. Each full indexer layer marks each such row's top-k
    set (DsaTopKMask) and sets the concentrator from it (DsaButterflySettings); every layer, the shared one included, reads the
    TOPK rows the concentrator routes at those settings over the row's history padded to 8 (DsaButterflyRowsPad), and the sparse
    MLA reads TOPK rows. The Program reads no `run` input and returns no checks."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    assert _no_run(r) and r.program.fn.signature()[1] == Array(1, I32)
    fam = _families(r)
    assert [fam[f] for f in MASK_FORM] == [6, 6, 8] and fam["DsaTopKShortRow_v1"] == 12 and fam["DsaMqaLogits_v3"] == 18
    assert not any(fam[f] for f in RUN_FORM)
    assert {(n.fn.bindings["N"], n.fn.bindings["TOPK"]) for n in _calls(r, "DsaTopKMask_v1")} == {(8, TOPK)}
    assert not fam["DsaButterflyRows_v1"]
    assert {tuple(n.fn.bindings[k] for k in "TNKW") for n in _calls(r, "DsaButterflyRowsPad_v1")} == {(5, 8, TOPK, 576), (6, 8, TOPK, 576)}
    assert [n.fn.bindings["T"] for n in _calls(r, "SparseMlaSplit_v1")] == [min(i + 1, TOPK) for i in range(6)] * 4


def test_a_long_rows_keys_past_its_own_score_the_pad(phi):
    """Row 4's mask reads its 5 logits, then the first 3 words of the Program's one block of PAD_SCORE (a batch of the constant,
    TOPK - 1 words), one run of references; its rows read its 5 history rows, one argument, and the first 3 x 576 words of the
    one block of zero words, another. The Program holds those two blocks and no other batch of a constant."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    nodes = r.program.fn.body.nodes
    first = _calls(r, "DsaTopKMask_v1")[0]
    logits, pad = runs(first.args[0].refs)
    assert logits[0] == "n" and nodes[logits[1]].fn.id.startswith("DsaMqaLogits_v3{N=5,") and logits[2:] == (0, 5)
    assert (nodes[pad[1]].form, nodes[pad[1]].fn.id, nodes[pad[1]].n, pad[2:]) == ("batch", f"Const32[0x{PAD_SCORE:08x}]_v1", TOPK - 1, (0, 3))
    rows = _calls(r, "DsaButterflyRowsPad_v1")[0]
    hist, (zero,) = rows.args[1], runs(rows.args[2].refs)
    assert rows.fn.bindings["T"] == 5 and hist.type.n == 5 and not any(nodes[x[1]].form == "batch" for x in runs(hist.refs))
    assert (nodes[zero[1]].form, nodes[zero[1]].fn.id, nodes[zero[1]].n, zero[2:]) == ("batch", "Const16[0x0000]_v1", (TOPK - 1) * 576, (0, 3 * 576))
    assert [i for i, n in enumerate(nodes) if n.form == "batch" and n.fn.id.startswith("Const")] == sorted({pad[1], zero[1]})


def test_the_shared_layer_reads_the_last_indexer_layers_settings(phi):
    """Layer 3 (`shared`) selects no top-k of its own: its DsaButterflyRowsPad read layer 2's settings at each row, as the
    Build's IndexShare layer reads the last indexer layer's selection."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    nodes = r.program.fn.body.nodes
    settings = [i for i, n in enumerate(nodes) if n.fn.id.startswith("DsaButterflySettings_v1{")]
    read = [list(runs(n.args[0].refs)) for n in _calls(r, "DsaButterflyRowsPad_v1")]
    assert all(len(x) == 1 and x[0][0] == "n" for x in read), "each DsaButterflyRowsPad reads one settings call's output"
    assert [x[0][1] for x in read] == [settings[2 * L + k] for L in (0, 1, 2, 2) for k in (0, 1)]


def test_a_long_rows_history_is_written_as_the_row_befores_and_its_own_row(phi):
    """The descriptor writes row 5's history (6 rows) as an alias to row 4's (5 rows) plus row 5's own two parts (its kv_c and its
    k_pe), in every layer: a row costs the descriptor O(1) at any length. With the pad in the same argument, no earlier argument
    was a prefix of it, and each row's history was written whole, O(rows^2) a sequence."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=True)
    body = r.desc["definitions"][r.desc["root"]]["body"]["nodes"]
    pads = [k for k, n in enumerate(body) if str(r.desc["callees"][n["fn"]]).startswith("DsaButterflyRowsPad_v1{")]
    assert len(pads) == 8
    for a, b in zip(pads[::2], pads[1::2]):
        hist = body[b]["args"][1]["refs"]["concat"]
        assert hist[0] == {"alias": [a, 1]} and len(hist) == 3, hist


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 13, 16])
def test_a_rows_class_selects_its_own_top_k_set_in_ascending_order(n):
    """At the row's class N, the logits past its own scored PAD_SCORE and its history rows past its own zero: DsaTopKMask marks
    the row's own top-TOPK set, on rows that tie (PAD_SCORE's word among them), and DsaButterflyRows at the settings
    DsaButterflySettings computes from that mask reads the set's rows in ascending index order, never a padded one."""
    N, rng = long_row_keys(n, TOPK), random.Random(n)
    assert N % TOPK == 0 and n <= N < n + TOPK
    pool = [0x00000000, 0x80000000, 0x3F800000, PAD_SCORE, 0x7FC00000, 0xFF800000]
    rows = [[rng.choice(pool) if rng.random() < 0.5 else rng.getrandbits(32) for _ in range(n)] for _ in range(12)]
    rows += [[PAD_SCORE] * n, [0x3F800000] * n]
    masks = np.asarray(E.evaluate_batch(bind(M.DsaTopKMask, N=N, TOPK=TOPK), [s + [PAD_SCORE] * (N - n) for s in rows]))
    for s, m in zip(rows, masks.reshape(len(rows), N).tolist()):
        want = sorted(topk_set(s, TOPK))
        assert [j for j, b in enumerate(m) if b] == want
        hist = [[j + 1, 7 * j] for j in range(n)] + [[0, 0]] * (N - n)
        assert M.butterfly_apply(M.butterfly_settings(m, TOPK), hist, TOPK, dead=[0, 0]) == [hist[j] for j in want]


def test_a_row_past_index_topk_under_an_engine_without_the_canonical_top_k_is_refused(phi):
    """An engine whose stated environment lacks `VERITY_DSA_TOPK=canonical` serves the kernels' own order past INDEX_TOPK, not
    the ascending one the mask form reads its rows in: its row past INDEX_TOPK is refused by name. A registration that states no
    environment builds it, and a step within INDEX_TOPK builds under any."""
    def at(env: dict | None) -> Phi:
        eng = {k: v for k, v in phi.engine.items() if k != "env"}
        return Phi.from_json({**phi.to_json(), "engine": eng if env is None else {**eng, "env": env}})
    with pytest.raises(Unexpressible, match=re.escape("a row past INDEX_TOPK (4) keys under an engine whose environment sets "
                                                      "VERITY_DSA_TOPK=None, not {'VERITY_DSA_TOPK': 'canonical'}")):
        build_program(at({"VLLM_BATCH_INVARIANT": "0"}), "step", M=5, pos0=0, with_desc=False)
    assert _families(build_program(at(None), "step", M=5, pos0=0, with_desc=False))["DsaTopKMask_v1"] == 3
    within = build_program(at({"VLLM_BATCH_INVARIANT": "0"}), "step", M=TOPK, pos0=0, with_desc=False)
    assert _no_run(within) and not any(_families(within)[f] for f in MASK_FORM)


@pytest.mark.slow
def test_a_node_program_past_index_topk_reads_no_run(phi):
    """At TP8 each rank's rows past INDEX_TOPK select their top-k from that rank's own logits, so a node Program joins them with
    no `run` input (about 10 s)."""
    r = build_node_program(phi, "step", M=6, pos0=0, with_desc=False)
    assert "run" not in [n for n, _ in r.program.fn.signature()[0]]


@pytest.mark.slow
def test_class_g_compiles_a_prompt_past_index_topk_at_its_launch_context(tmp_path):
    """Class G (about 15 s): a prompt of 10 tokens prefilled in two chunks beside a short request, then decoded, its rows past
    INDEX_TOPK from position 4. The step compiler builds each request's Program at its launch context (R1, R2 and R3 hold), the
    long one selecting its rows' top-k in the mask form with no `run` input, and that Program is `build_program`'s at the unit's
    launch batch and chunks. The fast tests build each Program alone; this one holds the batch rules to a Program past INDEX_TOPK."""
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    reg = _class_g(tmp_path, model={"hf_config": hf})
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    advice, ingress = _session([(10, 3, {**GREEDY, "max_tokens": 3}), (3, 2, {**GREEDY, "max_tokens": 2})],
                               [[(0, 0, 6)], [(0, 6, 4), (1, 0, 3)], [(0, 10, 1), (1, 3, 1)], [(0, 11, 1)]])
    c = compile_steps(reg, advice, ingress)
    long, short = c.requests
    assert (long.LP, long.T, long.chunks, short.LP, short.chunks) == (10, 2, (6, 4), 3, ())
    sigs = sorted(("run" in [n for n, _ in ps], str(ret)) for ps, ret in (p.fn.signature() for p in c.programs.values()))
    assert sigs == [(False, str(Array(2, I32))), (False, str(Array(3, I32)))]
    alone = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": long.policy}), "request", LP=10, T=2, dead_free=True,
                          batch=long.batch(), chunks=long.chunks)
    assert alone.digest in c.programs and _families(alone)["DsaTopKMask_v1"] == 3 * 7


def _step(*rows: tuple[int, int, int, int]):
    from verity_vllm.pipeline import window_advice as WA
    return WA.StepAdvice(tuple(WA.Row(i, "decode" if n == 1 and s >= lp else "prefill", s, n, 0, False, ()) for i, lp, s, n in rows))


@pytest.mark.slow
def test_class_g_carries_a_request_past_index_topk_with_its_indexer_history(tmp_path):
    """Class G (about 40 s): A (a 3-token prompt, 5 outputs: its rows past INDEX_TOPK from position 4) and B (2 and 3, within it)
    whole in one window are, call for call, their Programs across three: the first leaves A at position 6 and B at 3, the second
    runs B's last row with A carried through it idle, and the third runs A's last. A's rows read the indexer's keys and scales
    past INDEX_TOPK, so the edges carry them for A alone: the first window's A returns its samples, then every layer's latent rows, index-K slots and scales so far, out-ports `edge.kv`, `edge.ik`, `edge.iks` and `edge.last`,
    while B's index-K history stays its own unread Inputs; the idle window passes A's fields through to its own edge, and the
    third reads them as A's history, `edge.ik` and `edge.iks` one field each. No window reads a `run` input, and the advice's side (`pipeline.edge.fields`) holds each port's members as each window's Program returns them. What
    no cheaper test can: that a window's edges carry a request past INDEX_TOPK whole, through a window it idles in."""
    from verity.ml.circuits.types import field_types
    from verity_vllm.pipeline import edge as E
    from verity_vllm.pipeline import window_advice as WA
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
    assert ret[1:] == (Array(4, Array(6, Array(576, Value(16)))), Array(3, Array(6, Array(128, Value(8)))), Array(3, Array(6, Value(32))))
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
    assert [n for n, _ in wls[3].fn.signature()[0] if n.startswith("request[") and ".peers_" not in n] == [
        f"request[{rids[3][0]}].prompt"]
    assert not any("run" in dict(q.fn.signature()[0]) for p in prog for q in p.values())


@pytest.mark.slow
@pytest.mark.skipif(not CTX131K, reason="hours and hundreds of GB at the served 131,072 positions: VERITY_VLLM_CTX131K=LAYERS[:KEYS] "
                                        "compiles GLM-5.3's first LAYERS layers (at least its 3 dense ones) for one request of KEYS "
                                        "positions (default: the served max_model_len)")
def test_class_g_compiles_one_request_at_its_served_max_model_len(tmp_path):
    """Class G at its served limits (`CLASS_G_SERVED`): one greedy request of KEYS prompt tokens (the served max_model_len unless
    stated) and one output, which its prefill's last row samples (no decode position: T 0), its prompt prefilled alone in chunks
    of the long-prefill threshold, its last row of KEYS keys. The step compiler plans it and builds its request Program on the meta device, every row past
    INDEX_TOPK (2,048) in the mask form at its class: no `run` input and no checks, one DsaTopKMask per full indexer layer and row
    past INDEX_TOPK, at most KEYS / INDEX_TOPK classes. What no cheaper test can: that a request at the served limit compiles, at
    the real INDEX_TOPK; a run of it measures its time and memory. GLM-5.3's 78 layers don't fit one machine unrolled at 131,072
    positions, so LAYERS cuts the checkpoint (3: its dense full-indexer layers; 4: then a MoE layer sharing the third's top-k)."""
    layers, _, keys = CTX131K.partition(":")
    KEYS, CH = int(keys or CLASS_G_SERVED["max_model_len"]), CLASS_G_SERVED["long_prefill_token_threshold"]
    assert KEYS <= CLASS_G_SERVED["max_model_len"]
    reg = _class_g(tmp_path, layers=int(layers), engine=dict(CLASS_G_SERVED))
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    hf, LP = reg.phi.hf_config, KEYS
    topk, full = int(hf["index_topk"]), hf["indexer_types"][:int(layers)].count("full")
    advice, ingress = _session([(LP, 1, {**GREEDY, "max_tokens": 1})], [[(0, s, min(CH, LP - s))] for s in range(0, LP, CH)])
    c = compile_steps(reg, advice, ingress)
    (unit,) = c.requests
    (prog,) = c.programs.values()
    assert (unit.LP, unit.T) == (LP, 0) and "run" not in dict(prog.fn.signature()[0]) and prog.fn.signature()[1] == Array(1, I32)
    masks = Counter(n.fn.bindings["N"] for n in prog.fn.body.nodes if n.fn.id.startswith("DsaTopKMask_v1{"))
    assert sum(masks.values()) == full * max(LP - topk, 0)
    assert set(masks) == set(range(2 * topk, long_row_keys(LP, topk) + 1, topk)) if LP > topk else not masks
