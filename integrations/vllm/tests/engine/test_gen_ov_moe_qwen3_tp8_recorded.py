"""The Qwen3-MoE fold scope and the TP8 collectives' peer reads on RECORDED events: Qwen3-235B-A22B BF16 at TP8.

`fixtures/qwen3_235b_a22b_tp8_moe_l47.rank{0,7}.jsonl.gz` are layer 47 of ranks 0 and 7 of the 235B Match of record (B1 i256 o32, RTX PRO
6000, vLLM d9105ea80, batch-invariant eager; run r20261003-163309-0f1d), cut by `tests/engine/moe_scope_cut.py --logits` (each
`moe_scope_cut` Note names the source log and its sha256): the RunHeader, the layer's weight Roots and the lm_head shard's, and for the
prefill step (256 rows) and the first decode step (1 row) the scope, the block's all-reduce right after it, and the step's
`logits_processor` nodes (the lm_head shard's GEMM and the logits all-gather).  `fixtures/QWEN3_235B_A22B.config.json` is the
checkpoint's config.json at the recorded revision (its sha256 is the header's `config_sha256`); the rank profile is derived from it.
  * each rank folds the scope as the TP2 cut does (`test_gen_ov_moe_qwen3_recorded.py`): the router's gate GEMM ahead of the router,
    writing its logits, the experts, the activation (the SiluMul version the profile binds) and the routed sum; after the scope the
    block's `AllReduce_v2{WORLD=8,N=H}` over the routed sum; then the lm_head shard and the logits' `AllGather_v1{WORLD=8,N=V/8}`;
  * `fold` over the cut (the hidden state the cut does not compute read as stand-ins, everything else the fold's own) gives every
    collective one `peers` field per peer rank, in rank order at adjacent offsets; a one-row collective (the decode all-reduce, both
    logits all-gathers) reads its 7 peers as ONE run across those fields, the prefill all-reduce one run per field and row;
  * `Prog` canonicalises each recorded read as one ("PEER", ordinal, ..) per peer in rank order around the rank's own partial, which is
    how a rank Program reading one `peers_<k>` Input per peer reads it, and the two compare equal; without the split at field boundaries
    the one-row reads do not.
Not covered: the Build's TP8 rank Program itself (here a Program of that form over the same fields), the collective ordinals of a whole
step (the cut numbers only its own collectives), and the cross-rank leg.
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
import re
from pathlib import Path

import pytest

from tests.engine.moe_scope_cut import cut, write
from verity.ml.circuits.refs import Affine, runs
from verity_vllm.check.match import program_compare as PC
from verity_vllm.engine.profiles import generic
from verity_vllm.observe.events import Note
from verity_vllm.observe.fold import fold as FD
from verity_vllm.observe.fold.memory import Unresolved
from verity_vllm.observe.fold.resolve_log import resolve_log
from verity_vllm.observe.fold.resolver import Compute, Group, Transparent
from verity_vllm.observe.log import read_log
from verity_vllm.program import compact as K

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = FIXTURES / "QWEN3_235B_A22B.config.json"
REPO, REVISION = "Qwen/Qwen3-235B-A22B", "8efa61729e24bd65b1d152b5ab5409052aa80e65"
WORLD, LAYER = 8, 47
MODULE = f"model.layers.{LAYER}.mlp.experts"
RANKS = (0, 7)
SOURCE_SHA256 = {0: "bea53d712ea6577e477cfeed29ad04709e1d8327cf98cdcdcf79dd3ad422258a",
                 7: "60f1d811f6e00b83453e845f984952af12945a38bb84a03705ef5a1b1676da55"}
#: engine step -> (rows M, whether the scope aligns: the naive block assignment skips moe_align_block_size for M <= 2)
STEPS = {0: (256, True), 1: (1, False)}
DOT = "HopperBF16WgmmaDot16_v1"


def fixture(rank: int) -> Path:
    return FIXTURES / f"qwen3_235b_a22b_tp8_moe_l{LAYER}.rank{rank}.jsonl.gz"


@functools.lru_cache(maxsize=None)
def profile():
    cfg = json.loads(CONFIG.read_text())
    # this base's `generic.rank_hf_config` refuses 4 key/value heads over 8 ranks; vLLM gives each rank one of them (num_kv_head_replicas),
    # and 8 heads over 8 ranks is the same one head per rank, so the rank configuration and profile are the same
    return generic.build_profile("QWEN3_235B_A22B", WORLD, hf_config={**cfg, "num_key_value_heads": WORLD}, repo=REPO, revision=REVISION)


C = profile().config
H, I, E, TOPK, V = C["H"], C["I"], C["NUM_EXPERTS"], C["TOPK"], C["V"]


@functools.lru_cache(maxsize=None)
def _resolved(rank: int):
    header, results, roots, notes = resolve_log(fixture(rank), profile())
    return header, {sr.step: sr for sr in results}, roots, notes


@pytest.mark.parametrize("rank", RANKS)
def test_fixture_is_the_recorded_layer_scope_of_the_tp8_capture(rank):
    header, steps, roots, notes = _resolved(rank)
    assert (C["WORLD"], H, I, E, TOPK, V) == (WORLD, 4096, 1536 // WORLD, 128, 8, 151936 // WORLD)
    assert header.profile["case"] == "QWEN3_235B_A22B" and header.profile["model"]["revision"] == REVISION
    assert header.profile["model"]["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert header.profile["vllm"]["commit"] == "d9105ea80" and header.profile["parallelism"]["tensor_parallel_size"] == WORLD
    assert header.profile["gpu"]["compute_capability"] == "12.0" and header.profile["engine_flags"]["VLLM_BATCH_INVARIANT"] == "1"
    assert (header.notes["tp2"]["rank"], header.notes["tp2"]["world_size"]) == (rank, WORLD)
    (note,) = notes
    assert note.text == "moe_scope_cut" and note.data["sha256"] == SOURCE_SHA256[rank] and note.data["origin"].endswith(f"log.rank{rank}.jsonl.gz")
    assert (note.data["layer"], note.data["steps"], note.data["logits"]) == (LAYER, [0, 1], True)
    assert sorted(r.name for rs in roots.values() for r in rs) == ["lm_head.weight", f"{MODULE}.routed_experts.w13_weight",
                                                                   f"{MODULE}.routed_experts.w2_weight", f"model.layers.{LAYER}.mlp.gate.weight"]
    assert {s: (sr.phase_hint, sr.rows) for s, sr in steps.items()} == {0: ("prefill", 256), 1: ("decode", 1)}


@pytest.mark.parametrize("rank", RANKS)
def test_the_fixture_is_a_fixed_point_of_the_cutter(tmp_path, rank):
    events = list(read_log(fixture(rank)))
    note = next(e for e in events if isinstance(e, Note))
    write(cut(events, LAYER, set(STEPS), {k: note.data[k] for k in ("origin", "sha256")}, logits=True), tmp_path / "recut.jsonl.gz")
    assert gzip.decompress((tmp_path / "recut.jsonl.gz").read_bytes()) == gzip.decompress(fixture(rank).read_bytes())


@pytest.mark.parametrize("step", sorted(STEPS))
@pytest.mark.parametrize("rank", RANKS)
def test_recorded_tp8_scope_resolves_the_gate_gemm_ahead_of_the_router_and_the_rank_s_collectives(rank, step):
    _, steps, roots, _ = _resolved(rank)
    sr = steps[step]
    M, aligned = STEPS[step]
    assert sr.error is None and not sr.unsupported, (sr.error, [u.reason for u in sr.unsupported])
    assert sr.gemm_target["target"] == "blackwell_consumer" and sr.gemm_target["compute_capability"] == [12, 0]
    group, ar, *logits = sr.resolutions
    assert isinstance(group, Group) and isinstance(ar, Compute)
    computes = [p for p in group.parts if isinstance(p, Compute)]
    gemm, router, up, silu, down, msum = computes
    assert [c.spec.id for c in (gemm, router, up, down, msum)] == [
        f"Gemm_v2{{K={H},N={E},DOT={DOT}}}", f"MoeRouterTopKOrderedNorm_v1{{E={E},TOPK={TOPK},VPT=8}}",
        f"MoeExpertGemm_v2{{E={E},K={H},N={2 * I},DOT={DOT}}}", f"MoeExpertGemmW_v2{{E={E},K={I},N={H},DOT={DOT}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]
    assert re.fullmatch(rf"SiluMul_v\d+\{{I={I}\}}", silu.spec.id), silu.spec.id
    assert [c.rows for c in computes] == [M, M, M * TOPK, M * TOPK, M * TOPK, M]
    assert len([p for p in group.parts if isinstance(p, Transparent) and any("moe_align_block_size" in k for k in p.kernels)]) == int(aligned)

    gate_w = next(r for rs in roots.values() for r in rs if r.name == f"model.layers.{LAYER}.mlp.gate.weight")
    assert gemm.kernels == ("matmul_kernel_persistent",) and gemm.hint == f"model.layers.{LAYER}.mlp.gate/{gate_w.name}"
    assert gemm.operands[0].view == up.operands[0].view and gemm.operands[0].view.shape == (M, H)
    assert gemm.operands[1].view == gate_w.view and gate_w.view.shape == (E, H)        # the router is replicated: all E experts
    assert gemm.outputs[0].view == router.operands[0].view and gemm.outputs[0].view.shape == (M, E)
    assert group.parts.index(gemm) < group.parts.index(router)
    assert silu.operands[0].view == up.outputs[0].view and down.operands[0].view == silu.outputs[0].view

    # the block's all-reduce over the rank's routed sum, issued after the scope
    assert ar.spec.id == f"AllReduce_v2{{WORLD={WORLD},N={H}}}" and ar.rows == M and ar.pattern == "tp.collective"
    (own,) = ar.operands
    assert own.kind == "all_reduce" and own.site == MODULE and own.own == msum.outputs[0].view

    # the lm_head shard (one logits row at both steps) and the logits all-gather over it
    lm, ag = [r for r in logits if isinstance(r, Compute)]
    assert all(isinstance(r, Transparent) for r in logits if r is not lm and r is not ag)
    lm_w = next(r for rs in roots.values() for r in rs if r.name == "lm_head.weight")
    assert lm.spec.id == f"Gemm_v2{{K={H},N={V},DOT={DOT}}}" and lm.rows == 1 and lm.operands[1].view == lm_w.view
    assert ag.spec.id == f"AllGather_v1{{WORLD={WORLD},N={V}}}" and ag.rows == 1
    (own,) = ag.operands
    assert own.kind == "all_gather" and own.site == "logits_processor" and own.own == lm.outputs[0].view


# --- the fold's peer reads of the recorded collectives, and how `Prog` canonicalises them ------------------------------------------------

class _CutFold(FD._Fold):
    """`fold` over a cut: a read the cut cannot answer (state computed by the layers outside it) gets a stand-in run; every other read,
    every write, every collective and every peers field is the fold's own."""

    def _mem(self, what, fn, *args):
        try:
            return super()._mem(what, fn, *args)
        except Unresolved:
            if what != "read":
                raise
            self.live_in.add((self.cur_step, args[0].alloc))
            return Affine("p", -1, 0, args[0].numel)


@functools.lru_cache(maxsize=None)
def _folded(rank: int):
    f = _CutFold(profile(), False)
    f.live_in = set()
    return f, f.run(read_log(fixture(rank)), with_program=False)


#: the cut's collectives in issue order: (step, k, kind, site, rows, N)
COLLECTIVES = [(0, 0, "all_reduce", MODULE, 256, H), (0, 1, "all_gather", "logits_processor", 1, V),
               (1, 0, "all_reduce", MODULE, 1, H), (1, 1, "all_gather", "logits_processor", 1, V)]


@pytest.mark.parametrize("rank", RANKS)
def test_the_fold_gives_each_recorded_collective_one_peers_field_per_peer_rank(rank):
    f, res = _folded(rank)
    _, steps, _, _ = _resolved(rank)
    # the cut has no sampler: its SampleEnd and the decode row's StepLink are all that is missing
    assert {e["class"] for e in res.errors} == {"SampleEnd", "TokenWiring"}, res.errors
    # the stand-ins: the scope's [M,H] hidden states and the lm_head's input row, which the layers outside the cut compute
    gate = {s: next(p for p in sr.resolutions[0].parts if isinstance(p, Compute)) for s, sr in steps.items()}
    lm = {s: next(r for r in sr.resolutions if isinstance(r, Compute) and r.spec.id.startswith("Gemm_v2")) for s, sr in steps.items()}
    assert f.live_in == {(s, c.operands[0].view.alloc) for d in (gate, lm) for s, c in d.items()}
    assert res.tp == {"rank": rank, "world": WORLD}
    assert [(c["step"], c["k"], c["kind"], c["site"], c["rows"], c["N"]) for c in res.collectives] == COLLECTIVES
    (layout,) = res.peers_layout
    peers = [r for r in range(WORLD) if r != rank]
    want, off = [], 0
    for step, k, _kind, _site, rows, N in COLLECTIVES:
        for pr in peers:
            want.append((f"s{step}_k{k}_r{pr}", off, len(want), rows))
            off += rows * N
    assert [(x["name"], x["offset"], x["ordinal"], x["rows"]) for x in layout["fields"]] == want and layout["leaves"] == off


def _rows(rank: int) -> list[tuple[dict, list[tuple]]]:
    """(instance meta, operand runs) of the first and the last row of every recorded collective (one row for the one-row ones)."""
    _f, res = _folded(rank)
    out = []
    for m in res.instances:
        if m.get("collective") and (m["row"] == 0 or m["row"] == m["rows"] - 1):
            (arg,) = res.body.nodes[m["node"]].args
            out.append((m, runs(arg.refs)))
    return out


@pytest.mark.parametrize("rank", RANKS)
def test_a_one_row_collective_reads_its_peers_as_one_run_across_the_fields(rank):
    _f, res = _folded(rank)
    (layout,) = res.peers_layout
    flds = {x["name"]: x for x in layout["fields"]}
    assert [(m["step"], m["collective"]["k"], m["row"]) for m, _ in _rows(rank)] == [(0, 0, 0), (0, 0, 255), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    for m, rs in _rows(rank):
        c = m["collective"]
        N, j = c["N"], c["row_in_request"]
        base = [flds[f"s{c['step']}_k{c['k']}_r{pr}"]["offset"] for pr in range(WORLD) if pr != rank]
        own = [r for r in rs if r[0] == "n"]
        peer = [r for r in rs if r[0] == "p"]
        assert len(own) == 1 and own[0][3] - own[0][2] == N and rs.index(own[0]) == (0 if rank == 0 else len(rs) - 1)
        if m["rows"] == 1:      # 7 adjacent fields of one row each: the fold reads them as ONE run
            assert peer == [("p", layout["param"], base[0], base[-1] + N)], (m["spec"], rs)
        else:                   # 7 fields of M rows each: row j of every field, one run per field
            assert peer == [("p", layout["param"], b + j * N, b + (j + 1) * N) for b in base], (m["spec"], rs)


OWN, PEERS0 = 0, 1


def _prog(label: str, spec: str, own_spec: str, N: int, inputs: dict, peer_fields: dict, operand: list[tuple], compact: bool) -> PC.Prog:
    """One collective row as `Prog` holds it: the peers Input node(s), the own partial's producer, the collective reading them."""
    P = PC.Prog.__new__(PC.Prog)
    P.label, P.digest = label, f"digest-{label}"
    P.inputs = inputs
    P.tok_node, P.seed_nodes, P.seed_index = -1, set(), {}
    P.weight_fields, P.param_names, P.oracle_rows, P.n_nodes, P.nodes = {}, [], None, 0, None
    P.supplied_fields, P.supplied_owner = {}, {}
    P.peer_fields, P.peer_index = peer_fields, {n: 0 for n in peer_fields}
    prod = max(inputs) + 1
    inst = [{"k": prod, "spec": own_spec, "form": "call", "args": [], "out": f"bf16[{N}]"},
            {"k": prod + 1, "spec": spec, "form": "call", "args": [[run if run[1] != OWN else ("n", prod, run[2], run[3]) for run in operand]],
             "out": f"bf16[{N}]"}]
    P.compact = compact
    P.inst = [dict(r, args=[K.from_runs(a) if compact else list(a) for a in r["args"]]) for r in inst]
    P.k2i = {r["k"]: i for i, r in enumerate(P.inst)}
    P._bind_splits()
    P._segment()
    return P


def _sides(rank: int, m: dict, rs: list[tuple], layout: dict, compact: bool) -> tuple[PC.Prog, PC.Prog, list[tuple]]:
    """The fold's row (ONE peers Input with the fields `peers_layout` states, `load_oracle`'s form) and a rank Program's (one Input per
    `peers_<k>` field, read in rank order around its own partial), with the canonical operand both must have."""
    c = m["collective"]
    N, own_spec = c["N"], "MoeSum_v1" if c["kind"] == "all_reduce" else "Gemm_v2"
    mine = [x for x in layout["fields"] if x["name"].startswith(f"s{c['step']}_k{c['k']}_")]
    j = c["row_in_request"]
    to_n = [(("n", OWN, r[2], r[3]) if r[0] == "n" else ("n", PEERS0, r[2], r[3])) for r in rs]
    F = _prog(f"fold{rank}", m["spec"], own_spec, N, {PEERS0: ("Input16_v1", layout["leaves"])},
              {PEERS0: [(x["offset"], x["ordinal"]) for x in layout["fields"]]}, to_n, compact)
    ins = {PEERS0 + i: ("Input16_v1", x["rows"] * N) for i, x in enumerate(mine)}
    reads = [("n", PEERS0 + i, j * N, (j + 1) * N) for i in range(len(mine))]
    A = _prog(f"build{rank}", m["spec"], own_spec, N, ins, {PEERS0 + i: [(0, x["ordinal"])] for i, x in enumerate(mine)},
              reads[:rank] + [("n", OWN, 0, N)] + reads[rank:], compact)
    want = [("PEER", x["ordinal"], j * N, (j + 1) * N) for x in mine]
    want.insert(rank, ("N", 0, 0, 0, N))
    return A, F, want


def _canon(P: PC.Prog) -> list[tuple]:
    got = P.canon(1)[0]
    return K.expand_mixed(got) if P.compact else got


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("rank", RANKS)
def test_recorded_peer_reads_canonicalise_per_field_and_compare_with_the_rank_program_s(rank, compact):
    _f, res = _folded(rank)
    (layout,) = res.peers_layout
    for m, rs in _rows(rank):
        A, F, want = _sides(rank, m, rs, layout, compact)
        assert _canon(F) == _canon(A) == want, (m["spec"], m["step"], m["row"], _canon(F))
        r = PC.compare_steps(A, F)
        assert r["totals"]["operands_different"] == 0 and r["canonical_sha256"]["equal"], (m["spec"], m["step"], r["steps"][0].get("wiring"))


@pytest.mark.parametrize("rank", RANKS)
def test_without_the_field_boundary_split_the_one_row_reads_differ(monkeypatch, rank):
    """The recorded reads are the case the split exists for: unsplit, the one-row run is ("PEER?", ..) and the compare fails; the prefill
    all-reduce (one run per field) compares equal either way."""
    _f, res = _folded(rank)
    (layout,) = res.peers_layout
    monkeypatch.setattr(PC.Prog, "_split_peer", lambda self, run: [run])
    for m, rs in _rows(rank):
        A, F, want = _sides(rank, m, rs, layout, False)
        r = PC.compare_steps(A, F)
        if m["rows"] == 1:
            assert any(t[0] == "PEER?" for t in _canon(F)) and not r["canonical_sha256"]["equal"], (m["spec"], m["step"])
        else:
            assert _canon(F) == want and r["canonical_sha256"]["equal"], (m["spec"], m["row"])
