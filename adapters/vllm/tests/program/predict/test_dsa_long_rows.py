"""verity_vllm.program.predict.dsa past INDEX_TOPK: a row of more than INDEX_TOPK keys selects its top-k as the Build's `vllm_dsa`
rule does, in the circuit: `DsaTopKMask` marks the logits' top-k set, `DsaButterflySettings` sets the concentrator for it, and
sparse MLA reads the rows `DsaButterflyRows` routes at those settings, in ascending index order (the canonical row, which an engine
under `VERITY_DSA_TOPK=canonical` serves).  The Program has no `run` input, and its top-k adds no check: a request's one check
is its length rule (`stop_rule`).  A row is selected at its class (`vllm_dsa.long_row_keys`: its keys rounded up to a multiple
of INDEX_TOPK, the logits past its own `PAD_SCORE`), and below it read through `DsaButterflyRowsPad`, the history and its pad
two arguments.

GLM-5.3's checkpoint is cut to its first 4 layers (three full indexers, then one that shares the third's top-k) and its
INDEX_TOPK to 4, so a few rows reach past it without the 2,048 keys of the served one; one test keeps the served INDEX_TOPK and
reads its rows' history as an Input."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.defs import bind
from verity.circuits.refs import runs
from verity.circuits.types import Array, Tuple as TupleT, Value
from verity.definitions.registry import dsa_topk_mask as M
from verity.definitions.registry.dsa_topk import topk_set, waksman_apply, waksman_switches
from verity_vllm.check.batch import own
from verity_vllm.engine.profiles.engine_classes import CLASS_G_SERVED, DSA_TOPK_ENV
from verity_vllm.program.frontend.rules.vllm_dsa import PAD_SCORE, long_row_keys
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
#: what a row past INDEX_TOPK calls that a row within it doesn't (the rows at its class, or below it with its pad)
LONG = ("DsaTopKMask_v1", "DsaButterflySettings_v1", "DsaButterflyRows_v1", "DsaButterflyRowsPad_v1")
#: the relation form's calls, which no Program makes
RELATION = ("DsaWaksmanOrder_v1", "DsaTopKLongRow_v1", "DsaWaksmanRows_v1")
#: the mask class G's batched form calls, a row's own keys among its block's (`vllm_dsa.long_row_blocks`)
COUNTED = "DsaTopKMaskCounted_v1"
CTX131K = os.environ.get("VERITY_VLLM_CTX131K", "")
#: the max_model_len of the launch that test serves at
CTX = 131072


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


def _members(nodes, family: str) -> int:
    """The calls of `family` among `nodes`, a batch's once per member."""
    return sum(n.n if n.form == "batch" else 1 for n in nodes if n.fn.id.startswith(family + "{"))


def _calls(r, family: str) -> list[tuple[int, object]]:
    return [(i, n) for i, n in enumerate(r.program.fn.body.nodes) if n.fn.id.startswith(family + "{")]


def _reads(node, arg: int) -> list[int]:
    """The nodes whose outputs argument `arg` of `node` reads."""
    return sorted({x[1] for x in runs(node.args[arg].refs) if x[0] == "n"})


def _logits_and_pad(nodes, mask) -> tuple[int, ...]:
    """The logits call a DsaTopKMask reads, then the pad block it reads past them, if any."""
    return tuple(sorted(_reads(mask, 0), key=lambda k: not nodes[k].fn.id.startswith("DsaMqaLogits_v3{")))


def _order(L: int, p: int) -> list[int]:
    """An order of TOPK of the row's p + 1 keys, a different one per (layer, position): `run_bits` routes any order."""
    return list(dict.fromkeys([(p - k * (L + 1)) % (p + 1) for k in range(TOPK)] + list(range(p + 1))))[:TOPK]


def test_a_row_past_index_topk_selects_its_top_k_from_its_logits(phi):
    """A step of 6 rows: rows 4 and 5 have 5 and 6 keys, both of class 8.  Each full indexer layer marks each such row's top-k set
    from that row's logits and PAD_SCORE past them (DsaTopKMask) and computes the concentrator's settings from the mask
    (DsaButterflySettings); every layer, the shared one included, reads the TOPK rows the concentrator routes at those settings over
    the row's history padded to 8 (DsaButterflyRowsPad), and the sparse MLA reads TOPK rows.  No `run` input, no checks: the
    Program returns the step's sample alone."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    params, ret = r.program.fn.signature()
    names = [n for n, _ in params]
    assert "run" not in names and names[-1] == "weights" and ret == Array(1, I32) and r.checks is None
    fam = _families(r.program.fn.body.nodes)
    assert [fam[f] for f in LONG] == [6, 6, 0, 8] and fam["DsaTopKShortRow_v1"] == 12 and fam["DsaMqaLogits_v3"] == 18
    assert [fam[f] for f in RELATION] == [0] * len(RELATION)
    nodes = r.program.fn.body.nodes
    for i, n in _calls(r, "DsaTopKMask_v1"):
        lg, pad = _logits_and_pad(nodes, n)
        keys = nodes[lg].fn.bindings["N"]
        assert nodes[lg].fn.id == f"DsaMqaLogits_v3{{N={keys},H=32,D=128}}", "the mask reads the row's own logits"
        assert n.fn.bindings["N"] == long_row_keys(keys, TOPK) == 8 and nodes[pad].fn.id == f"Const32[0x{PAD_SCORE:08x}]_v1"
        assert nodes[i + 1].fn.id.startswith("DsaButterflySettings_v1{") and _reads(nodes[i + 1], 0) == [i]
    assert {tuple(n.fn.bindings[k] for k in "TNKW") for _, n in _calls(r, "DsaButterflyRowsPad_v1")} == {(5, 8, TOPK, 576), (6, 8, TOPK, 576)}
    assert [n.fn.bindings["T"] for n in r.program.fn.body.nodes if n.fn.id.startswith("SparseMlaSplit_v1{")] == [
        min(i + 1, TOPK) for i in range(6)] * 4
    src = r.program.fn.bindings["SRC"]["inputs"]
    assert src[0] == [6] and len(src) == len(params) - 1, "the SRC inputs are the wrapper's, without the weights"


def test_the_shared_layer_reads_the_last_indexer_layers_settings(phi):
    """Layer 3 (`shared`) selects no top-k of its own: its DsaButterflyRowsPad read the settings layer 2 computed at each row, as
    the Build's IndexShare layer reads the last indexer layer's selection."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    settings = [i for i, _ in _calls(r, "DsaButterflySettings_v1")]
    read = [_reads(n, 0) for _, n in _calls(r, "DsaButterflyRowsPad_v1")]
    assert read == [[settings[k]] for k in (0, 1, 2, 3, 4, 5, 4, 5)]


def test_a_long_rows_keys_past_its_own_score_the_pad(phi):
    """Row 4's mask reads its 5 logits, then the first 3 words of the Program's one block of PAD_SCORE (a batch of the constant,
    TOPK - 1 words), one run of references; its rows read its 5 history rows, one argument, and the first 3 x 576 words of the
    one block of zero words, another. The Program holds those two blocks and no other batch of a constant."""
    r = build_program(phi, "step", M=6, pos0=0, with_desc=False)
    nodes = r.program.fn.body.nodes
    (_, first), *_ = _calls(r, "DsaTopKMask_v1")
    logits, pad = runs(first.args[0].refs)
    assert logits[0] == "n" and nodes[logits[1]].fn.id.startswith("DsaMqaLogits_v3{N=5,") and logits[2:] == (0, 5)
    assert (nodes[pad[1]].form, nodes[pad[1]].fn.id, nodes[pad[1]].n, pad[2:]) == ("batch", f"Const32[0x{PAD_SCORE:08x}]_v1", TOPK - 1, (0, 3))
    (_, rows), *_ = _calls(r, "DsaButterflyRowsPad_v1")
    hist, (zero,) = rows.args[1], runs(rows.args[2].refs)
    assert rows.fn.bindings["T"] == 5 and hist.type.n == 5 and not any(nodes[x[1]].form == "batch" for x in runs(hist.refs))
    assert (nodes[zero[1]].form, nodes[zero[1]].fn.id, nodes[zero[1]].n, zero[2:]) == ("batch", "Const16[0x0000]_v1", (TOPK - 1) * 576, (0, 3 * 576))
    assert [i for i, n in enumerate(nodes) if n.form == "batch" and n.fn.id.startswith("Const")] == sorted({pad[1], zero[1]})


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
    assert [_families(r.program.fn.body.nodes)[f] for f in LONG] == [0] * len(LONG)


@pytest.fixture
def served(tmp_path) -> Phi:
    p = _first_layers(_row_phi(GLM53, "GLM53", tmp_path), 4)
    assert p.hf_config["index_topk"] == 2048
    return Phi.from_json({**p.to_json(), "engine": _canonical(p.engine)})


@pytest.mark.parametrize("C", [2047, 2049, 32767])
def test_a_decode_row_at_the_served_index_topk_selects_its_top_k_at_its_class(served, C):
    """At the served INDEX_TOPK (2,048), a request a Commit window carries in at decode C reads its C positions as an Input and
    its decode row has C + 1 keys: 2,048 stay within INDEX_TOPK, and 2,050 or 32,768 select the top 2,048 of the row's own
    logits in the mask form at its class: 4,096 for 2,050 (its rows through DsaButterflyRowsPad), 32,768 for 32,768.  Every
    layer's sparse MLA reads 2,048 rows."""
    r = build_program(served, "request", LP=C, T=1, C=C, with_desc=False)
    n, nodes = C + 1, r.program.fn.body.nodes
    fam = _families(nodes)
    assert "run" not in [x for x, _ in r.program.fn.signature()[0]] and r.checks is None
    assert [m.fn.bindings["T"] for m in nodes if m.fn.id.startswith("SparseMlaSplit_v1{")] == [2048] * 4
    if n <= 2048:
        assert [fam[f] for f in LONG] == [0] * len(LONG) and fam["DsaTopKShortRow_v1"] == 3
        return
    N = long_row_keys(n, 2048)
    assert [fam[f] for f in LONG] == ([3, 3, 0, 4] if N > n else [3, 3, 4, 0]) and fam["DsaTopKShortRow_v1"] == 0
    rows = f"DsaButterflyRowsPad_v1{{T={n},N={N},K=2048,W=576,E=16}}" if N > n else f"DsaButterflyRows_v1{{N={N},K=2048,W=576,E=16}}"
    assert {m.fn.id for f in LONG for _, m in _calls(r, f)} == {f"DsaTopKMask_v1{{N={N},TOPK=2048}}", f"DsaButterflySettings_v1{{N={N},K=2048}}", rows}
    for _, m in _calls(r, "DsaTopKMask_v1"):
        lg, *pad = _logits_and_pad(nodes, m)
        assert nodes[lg].fn.id.startswith("DsaMqaLogits_") and nodes[lg].fn.bindings["N"] == n and len(pad) == (N > n)


def _batched(phi: Phi, LP: int, T: int, table_reads: bool):
    batch = [[[LP, 0]]] + [[[1, LP + t]] for t in range(T)]
    return build_program(phi, "request", LP=LP, T=T, dead_free=True, batched=True, batch=batch, table_reads=table_reads)


def _wiring(r, sel: str, rows: tuple[str, ...]) -> list[tuple[int, list[int]]]:
    """Each read of a row's history in order: the keys it reads, and which of `sel`'s calls (by their order) its positions are."""
    at = {i: j for j, (i, _) in enumerate(_calls(r, sel))}
    reads = sorted(((i, n) for f in rows for i, n in _calls(r, f)), key=lambda x: x[0])

    def keys(b):
        return b["V"] if "V" in b else b.get("T", b.get("N"))   # the reads', the rows pad's, the rows'
    return [(keys(n.fn.bindings), [at[k] for k in _reads(n, 0)]) for _, n in reads]


def test_table_reads_move_only_a_request_with_rows_past_index_topk(phi):
    """`table_reads` (`Registered.dsa_table_reads`): a request whose rows stay within INDEX_TOPK is the same Program either way,
    digest for digest.  One past it (a prompt of 6 and two decodes) reads each such row's history by table: a `DsaTopKIndex` of
    its counted mask in place of each settings, and a `DsaHistoryReads` at the block's keys in place of each read of the rows
    network, at the same selection (the shared layer's at the last indexer layer's), without the rows' zero pad; everything
    else, the counted masks and their logits' pad included, as before."""
    assert _batched(phi, 3, 1, False).digest == _batched(phi, 3, 1, True).digest
    off, on = _batched(phi, 6, 2, False), _batched(phi, 6, 2, True)
    assert off.digest != on.digest
    a, b = _families(off.program.fn.body.nodes), _families(on.program.fn.body.nodes)
    net, table, zero = ("DsaButterflySettings_v1", "DsaButterflyRows_v1", "DsaButterflyRowsPad_v1"), ("DsaTopKIndex_v1", "DsaHistoryReads_v1"), "Const16[0x0000]_v1"
    assert [b[f] for f in net] == [0, 0, 0] and a[COUNTED] == b[COUNTED] > 0
    assert [b[f] for f in table] == [a[net[0]], a[net[1]] + a[net[2]]] and a[net[2]] > 0
    assert {f: n for f, n in a.items() if f not in (*net, zero)} == {f: n for f, n in b.items() if f not in (*table, zero)}
    assert _wiring(on, table[0], table[1:]) == _wiring(off, net[0], net[1:]), "each row read at the same keys and selection"

    def zeros(r):
        return [n.n for n in r.program.fn.body.nodes if n.form == "batch" and n.fn.id == zero]
    assert len(zeros(off)) == len(zeros(on)) + 1, "the rows network's zero pad is gone"


def test_table_reads_are_the_batched_forms(phi):
    with pytest.raises(ValueError, match="table_reads is the batched form's"):
        build_program(phi, "request", LP=6, T=1, table_reads=True)


def test_a_registration_with_table_reads_builds_and_traces_its_request_programs_by_table(tmp_path):
    """`Registered.dsa_table_reads`: class G's registration builds a request Program past INDEX_TOPK in the table form
    (`request_program`) and traces it with `--dsa-table-reads` after `--batched`; off (the default), neither.  A registration
    with no batched form (a batch-invariant target), or with the prefix cache on, refuses it by name."""
    from verity_vllm.program.predict.steps import Registered, plan_steps, request_program, trace_commands
    from tests.program.predict.test_steps import _registered
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    reg = _class_g(tmp_path, model={"hf_config": hf})
    off = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    on = Registered.of(off.phi, 7, dsa_table_reads=True)
    assert not off.dsa_table_reads
    advice, ingress = _session([(6, 2, {**GREEDY, "max_tokens": 2})], [[(0, 0, 6)], [(0, 6, 1)]])
    (u,) = plan_steps(on, advice, ingress).requests
    a, b = (request_program(r, u).program.fn.body.nodes for r in (off, on))
    assert _members(b, "DsaTopKIndex_v1") == _members(a, "DsaButterflySettings_v1") > 0
    assert _members(a, "DsaTopKIndex_v1") == _members(b, "DsaButterflySettings_v1") == 0
    summary = {"requests": [{"LP": u.LP, "T": u.T, "C": u.C, "policy": u.policy, "launch": u.batch(), "chunks": list(u.chunks),
                             "digest": "d", "stop": u.stop}]}
    (x,), (y,) = trace_commands(summary, off, "/t"), trace_commands(summary, on, "/t")
    assert x[-1] == "--batched" and y == [*x, "--dsa-table-reads"]
    assert _registered().phi.target_profile().batch_invariant
    with pytest.raises(ValueError, match="dsa_table_reads is class G's batched form's"):
        Registered.of(_registered().phi, 7, dsa_table_reads=True)
    cached = Phi.from_json({**off.phi.to_json(), "engine": {**off.phi.engine, "enable_prefix_caching": True}})
    assert not Registered.of(cached, 7).dsa_table_reads
    with pytest.raises(ValueError, match="dsa_table_reads takes the prefix cache off"):
        Registered.of(cached, 7, dsa_table_reads=True)


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
    assert r.world == 8 and [fam[f] for f in LONG] == [6 * 8, 6 * 8, 0, 8 * 8] and r.program.fn.signature()[1] == Array(8, I32)


@pytest.mark.slow
def test_gp01_composes_a_rank_program_past_index_topk_beside_one_within_it(phi):
    """At TP8 GP-01 composes a request past INDEX_TOPK with one within it, and R1's labels hold every request-local Input of each
    to its own request: the long one's selection reads its own logits alone.  What no cheaper test can: that the composition and
    R1 take a rank Program that selects past INDEX_TOPK."""
    rs = [build_program(phi, "request", LP=lp, T=0) for lp in (5, 3)]
    assert [_families(r.program.fn.body.nodes)["DsaTopKMask_v1"] for r in rs] == [3, 0]
    comps = [request_component(r) for r in rs]
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
    long one selecting its rows past INDEX_TOPK, and that Program is `build_program`'s at the unit's launch batch and chunks, batched.
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
    assert not any(_families(p.fn.body.nodes)["DsaTopKMask_v1"] for p in c.programs.values()), "the batched form's mask is counted"
    sigs = sorted((_members(p.fn.body.nodes, COUNTED), str(p.fn.signature()[1])) for p in c.programs.values())
    rule = Array(1, Value(1))
    assert sigs == [(0, str(TupleT(Array(2, I32), rule))), (3 * 8, str(TupleT(Array(3, I32), rule)))], \
        "rows 4..11 of the long one, at each indexer layer; each Program's length rule last"
    alone = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": long.policy}), "request", LP=10, T=2, dead_free=True,
                          batched=True, batch=long.batch(), chunks=long.chunks, stop=long.stop)
    assert alone.digest in c.programs


def _step(*rows: tuple[int, int, int, int]):
    from verity_vllm.pipeline import window_advice as WA
    return WA.StepAdvice(tuple(WA.Row(i, "decode" if n == 1 and s >= lp else "prefill", s, n, 0, False, ()) for i, lp, s, n in rows))


@pytest.mark.slow
def test_class_g_carries_a_request_past_index_topk_with_its_indexer_history(tmp_path):
    """Class G (about 40 s): A (a 3-token prompt, 5 outputs: its rows past INDEX_TOPK from position 4) and B (2 and 3, within it)
    whole in one window are, call for call, their Programs across three, each with its own pad constants and length rule: the first leaves A at position 6 and B at 3, the second
    runs B's last row with A carried through it idle, and the third runs A's last.  A's rows read the indexer's keys and scales
    past INDEX_TOPK, so the edges carry them for A alone: the first window's A returns its samples, then every layer's latent rows,
    index-K slots and scales so far, out-ports `edge.kv`, `edge.ik`, `edge.iks` and `edge.last`, while B's index-K history stays
    its own unread Inputs; the idle window passes A's fields through to its own edge, and the third reads them as A's history,
    `edge.ik` and `edge.iks` one field each.  Each window's Program selects the top-k of the rows it computes past INDEX_TOPK, and
    the advice's side (`pipeline.edge.fields`) holds each port's members as each window's Program returns them.  What no cheaper
    test can: that a window's edges carry a request past INDEX_TOPK whole, through a window it idles in."""
    from verity.circuits.types import field_types
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

    def calls(p) -> list[str]:
        ids = [n.fn.id for n in p.fn.body.nodes if not n.fn.id.startswith("Const")]
        assert ids[-1].startswith("StopCheck_v1"), "each window's Program ends in its own length rule"
        return ids[:-1]

    for r in (0, 1):
        assert calls(prog[0][r]) == [i for p in prog[1:] if r in p for i in calls(p[r])]
    assert [u.edge for u in cs[1].requests] == [("prefix", "prefix_ik", "prefix_iks"), ("prefix",)]
    ret = field_types(prog[1][0].fn.signature()[1])
    assert ret[:2] == (Array(4, I32), Array(1, Value(1))) and ret[2:] == (
        Array(4, Array(6, Array(576, Value(16)))), Array(3, Array(6, Array(128, Value(8)))), Array(3, Array(6, Value(32))))
    wls = [c.workload for c in cs]
    assert field_types(wls[1].fn.signature()[1])[0] == TupleT(*ret[:2])
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
        assert [t.leaves for t in field_types(params["edge.ik"])] == [ret[3].leaves]
    names = [n for n, _ in wls[2].fn.signature()[0] if n.startswith("request[") and ".peers_" not in n]
    assert names == [f"request[{rids[2][0]}].{g}" for g in ("prompt", "prefix_ik", "prefix_iks")], "B's index-K history is its own"
    assert [n for n, _ in wls[3].fn.signature()[0] if n.startswith("request[") and ".peers_" not in n] == [f"request[{rids[3][0]}].prompt"]
    for p, pos in zip(prog, ({4, 5, 6}, {4, 5}, set(), {6})):
        assert _members(p[0].fn.body.nodes, COUNTED) == 3 * len(pos) if 0 in p else not pos
        assert 1 not in p or _members(p[1].fn.body.nodes, COUNTED) == 0


@pytest.mark.slow
def test_class_g_compiles_a_prompt_whose_prefill_spans_two_window_edges(tmp_path):
    """Class G as served, the prefix cache off, its long-prefill threshold cut to 4 (about 6 s): a 10-token prompt prefilled 4, 4
    and 2 rows a step, then decoded twice, its rows past INDEX_TOPK from position 4, in three Commit windows whose two edges fall
    inside the prompt.  Each window compiles: the first leaves the request mid-prompt with its latent rows, index-K slots and
    scales as out-ports, the second carries it in at 4 positions and out at 8, the third carries it in and finishes it, and the
    three windows' calls are the whole window's, each window selecting the top-k of the rows it computes past INDEX_TOPK.  A
    request left mid-prompt has no `edge.last`.  What no cheaper test can: that a served class G prefill, the 131,072-token
    prompt's shape, spans window edges in the compiler, which `engine.class_admission` refuses only because the served Commit
    doesn't commit the edge."""
    from verity.circuits.types import field_types
    from verity_vllm.pipeline import edge as E
    from verity_vllm.pipeline import window_advice as WA
    from verity_vllm.program.predict.steps import Registered, compile_steps
    from verity_vllm.program.workload import EDGE_PORTS
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    reg = _class_g(tmp_path, engine={"long_prefill_token_threshold": 4, "max_num_batched_tokens": 16}, model={"hf_config": hf})
    assert reg.phi.engine["enable_prefix_caching"] is False
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    a = WA.RequestAdvice("a", 10, 3, "length")
    steps = [_step((0, 10, s, n)) for s, n in ((0, 4), (4, 4), (8, 2), (10, 1), (11, 1))]
    windows = (WA.Advice((a,), tuple(steps[:1]), ()), WA.Advice((a,), tuple(steps[1:2]), (), (WA.Carried(0, 4, 0, ((0, 4),)),)),
               WA.Advice((a,), tuple(steps[2:]), (0,) * 3, (WA.Carried(0, 8, 0, ((0, 8),)),)))
    assert WA.follows(*windows[:2]) == WA.follows(*windows[1:]) == [(0, 0)]
    ingress = {"a": {**GREEDY, "max_tokens": 3, "prompt": list(range(1, 11))}}
    cs = [compile_steps(reg, adv, ingress) for adv in (WA.Advice((a,), tuple(steps), (0,) * 3), *windows)]
    units = [u for c in cs for u in c.requests]
    assert [(u.LP, u.C, u.carried_in, u.carried_out, u.edge) for u in units] == [
        (10, 0, False, False, ()), (10, 0, False, True, ("prefix", "prefix_ik", "prefix_iks")),
        (10, 4, True, True, ("prefix", "prefix_ik", "prefix_iks")), (10, 8, True, False, ())]
    assert units[0].chunks == (4, 4, 2) and units[0].T == units[3].T == 2

    def calls(c) -> list[str]:
        (p,) = c.programs.values()
        return [n.fn.id for n in p.fn.body.nodes if not n.fn.id.startswith(("Const", "StopCheck_v1"))]

    assert calls(cs[0]) == calls(cs[1]) + calls(cs[2]) + calls(cs[3])
    assert [_members(next(iter(c.programs.values())).fn.body.nodes, COUNTED) for c in cs] == [3 * 8, 0, 3 * 4, 3 * 4]
    wls = [c.workload for c in cs]
    assert wls[1].edge.outs == wls[2].edge.ins and wls[2].edge.outs == wls[3].edge.ins and not wls[3].edge.outs
    assert [[g for g, _, _ in gs] for _, gs in wls[2].edge.ins] == [["prefix", "prefix_ik", "prefix_iks"]]
    assert wls[2].ports == wls[3].ports == ("edge.kv", "edge.ik", "edge.iks")
    assert [[t.leaves for t in field_types(dict(wls[w].fn.signature()[0])["edge.kv"])] for w in (2, 3)] == [
        [4 * 4 * 576], [4 * 8 * 576]], "every layer's latent rows so far"
    for adv, wl in zip(windows, wls[1:]):
        assert E.fields(adv, TOPK) == {p: tuple(i for i, gs in (wl.edge.outs if wl.edge else ()) if g in [h for h, _, _ in gs])
                                       for p, g in EDGE_PORTS}, "the advice's side holds the fields the Program's edge returns"


@pytest.mark.slow
@pytest.mark.parametrize("table_reads", [False, True], ids=["rows-network", "table-reads"])
def test_a_class_g_chunked_prompt_past_index_topk_traced_build_is_its_compiled_program(tmp_path, table_reads):
    """The prompt above (10 tokens in chunks of 6 and 4 beside a short request, then two decodes; rows past INDEX_TOPK from
    position 4), compiled at rank 0 under the cut checkpoint, and its trace: the command `trace_commands` gives it, run on the
    pinned vLLM with the cut config.json in the hub and no CUDA device visible, as the row driver runs a Build.  `compare` finds the
    two digests equal, by the rows network and under the registration's table reads (`--dsa-table-reads`).  What no cheaper test
    can: that the Build's `vllm_dsa` rule, on vLLM's own graph, traces the mask form the predictor writes past INDEX_TOPK.  It
    needs the pinned vLLM (a pod's venv), so it skips without it."""
    pytest.importorskip("vllm")
    from verity_vllm.program.predict.steps import Registered, compare, compile_steps, trace_commands
    hub = tmp_path / "hub"
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    reg = _class_g(hub, model={"hf_config": hf})
    cp = Path(S.hf_config_path(str(hub), reg.phi.model["repo"], reg.phi.model["revision"]))
    assert cp.is_file(), "the config.json `_class_g` wrote, which the trace loads"
    cp.write_text(json.dumps(hf))
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7,
                        dsa_table_reads=table_reads)
    advice, ingress = _session([(10, 3, {**GREEDY, "max_tokens": 3, "ignore_eos": True}), (3, 2, {**GREEDY, "max_tokens": 2, "ignore_eos": True})],
                               [[(0, 0, 6)], [(0, 6, 4), (1, 0, 3)], [(0, 10, 1), (1, 3, 1)], [(0, 11, 1)]])
    c = compile_steps(reg, advice, {"q0": ingress["q0"]}, partial=True)
    u = c.requests[0]
    assert (u.LP, u.T, u.chunks) == (10, 2, (6, 4))
    assert _members(c.programs[u.digest].fn.body.nodes, COUNTED) == 3 * 8
    s = c.summary()
    (cmd,) = trace_commands(s, reg, str(tmp_path / "traces"))
    assert json.loads(cmd[cmd.index("--launch-context") + 1])["chunks"] == [6, 4] and ("--dsa-table-reads" in cmd) == table_reads
    assert _members(c.programs[u.digest].fn.body.nodes, "DsaTopKIndex_v1") == (3 * 8 if table_reads else 0)
    subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", *cmd[1:]], check=True,
                   env={**os.environ, "HF_HUB_CACHE": str(hub), "CUDA_VISIBLE_DEVICES": ""})
    got = compare(s, str(tmp_path / "traces"))
    assert (got["equal"], got["mismatches"]) == (1, 0), got["units"]


@pytest.mark.slow
@pytest.mark.skipif(not CTX131K, reason="hours and hundreds of GB at the served 131,072 positions: VERITY_VLLM_CTX131K=LAYERS[:KEYS] "
                                        "compiles GLM-5.3's first LAYERS layers (at least its 3 dense ones) for one request of KEYS "
                                        "positions (default: the served max_model_len)")
def test_class_g_compiles_one_request_at_its_served_max_model_len(tmp_path):
    """Class G at its served limits (`CLASS_G_SERVED`) under a launch of 131,072 positions (`CTX`): one greedy request of KEYS
    prompt tokens (the served max_model_len unless
    stated) and one output, which its prefill's last row samples (no decode position: T 0), its prompt prefilled alone in chunks
    of the long-prefill threshold, its last row of KEYS keys. The step compiler plans it and builds its request Program on the meta device, every row past
    INDEX_TOPK (2,048) in the mask form at its class: no `run` input, its one check its length rule, one DsaTopKMaskCounted per full indexer layer and row
    past INDEX_TOPK (in the last layer only its last chunk's: the rest are dead units, `prune`), at most KEYS / INDEX_TOPK classes. What no cheaper test can: that a request at the served limit compiles, at
    the real INDEX_TOPK; a run of it measures its time and memory. GLM-5.3's 78 layers don't fit one machine unrolled at 131,072
    positions, so LAYERS cuts the checkpoint (3: its dense full-indexer layers; 4: then a MoE layer sharing the third's top-k)."""
    layers, _, keys = CTX131K.partition(":")
    KEYS, CH = int(keys or CTX), CLASS_G_SERVED["long_prefill_token_threshold"]
    assert KEYS <= CTX
    from verity_vllm.program.predict.steps import Registered, compile_steps
    reg = _class_g(tmp_path, layers=int(layers), engine={**CLASS_G_SERVED, "max_model_len": CTX})
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    hf, LP = reg.phi.hf_config, KEYS
    topk, full = int(hf["index_topk"]), hf["indexer_types"][:int(layers)].count("full")
    advice, ingress = _session([(LP, 1, {**GREEDY, "max_tokens": 1})], [[(0, s, min(CH, LP - s))] for s in range(0, LP, CH)])
    c = compile_steps(reg, advice, ingress)
    (unit,) = c.requests
    (prog,) = c.programs.values()
    assert (unit.LP, unit.T) == (LP, 0) and "run" not in dict(prog.fn.signature()[0])
    assert prog.fn.signature()[1] == TupleT(Array(1, I32), Array(1, BIT)), "its sample, then its length rule"
    masks = Counter()
    for n in prog.fn.body.nodes:
        if n.fn.id.startswith(COUNTED + "{"):
            masks[n.fn.bindings["N"]] += n.n if n.form == "batch" else 1
    long, last = max(LP - topk, 0), LP - CH * ((LP - 1) // CH)
    dead = long - min(last, long) if hf["indexer_types"][int(layers) - 1] == "full" else 0
    assert sum(masks.values()) == full * long - dead
    assert set(masks) == set(range(2 * topk, long_row_keys(LP, topk) + 1, topk)) if LP > topk else not masks
