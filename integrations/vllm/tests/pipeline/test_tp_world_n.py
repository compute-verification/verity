"""Tensor parallelism at world N (synthetic world 3, offline): the world-parametric Definitions, their identities (world 2
keeps `AllReduce2_v1{N}` / `AllGather2_v1{N}` -- every recorded rank Program's digest unchanged), the registry-derived family selectors
the consumers use instead of spelled prefixes, the N-ary collective bus (one peer input per (collective, peer rank), rank order), the
`tp_links.json` v1 -> v2 reader, the three-rank collective linkage, and the three-rank cross-rank check / commit binding."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import b1_tp2 as TP2, prims as P
from verity_vllm.program.frontend import collective_kinds


# --- helpers --------------------------------------------------------------------------------------------------------------

def _eval(prog: Program, inputs: list[list[int]]) -> list[int]:
    Cc = prog.circuit
    t: dict[int, int] = {}
    for pi, vals in enumerate(inputs):
        coll = prog.input_coll(pi)
        assert coll.type.leaves == len(vals)
        for j, v in enumerate(vals):
            t[resolve(Cc.root_scope, coll.refs[j])] = v
    evaluate_region(Cc, Cc[0:prog.gates], t)
    return [t[g] for g in Cc.root_outputs()]


def _bf16_words(a) -> list[int]:
    return [int(x) >> 16 for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def _rand_words(rng, N: int) -> list[int]:
    return _bf16_words(rng.standard_normal(N) * rng.choice([1e-3, 1.0, 64.0, 2.0 ** 20], N))


# --- the Definitions and their identities ---------------------------------------------------------------------------------

def test_world_2_selectors_keep_the_recorded_identities():
    """The family NAME is part of a Definition's identity (`Name_vN{statics}`), so world 2 MUST still bind the `*2` composites."""
    assert TP2.all_reduce(2, 2048).id == "AllReduce2_v1{N=2048}" == bind(TP2.AllReduce2, N=2048).id
    assert TP2.all_gather(2, 512).id == "AllGather2_v1{N=512}" == bind(TP2.AllGather2, N=512).id
    assert TP2.all_reduce(3, 2048).id == "AllReduce_v2{WORLD=3,N=2048}"
    assert TP2.all_gather(4, 512).id == "AllGather_v1{WORLD=4,N=512}"
    for bad in (0, 1):
        with pytest.raises(ValueError):
            TP2.all_reduce(bad, 8)
        with pytest.raises(ValueError):
            TP2.all_gather(bad, 8)


def _chain(parts: list[list[int]], order) -> list[int]:
    acc = list(parts[order[0]])
    for r in order[1:]:
        acc = [P.Bf16Add.evaluate(a, b) for a, b in zip(acc, parts[r])]
    return acc


def test_allreduce_order_is_the_leaf_to_root_tree_chain():
    assert TP2.allreduce_order(2) == (1, 0)
    assert TP2.allreduce_order(4) == (3, 2, 1, 0)


@pytest.mark.parametrize("W,N", [(3, 16), (4, 5), (3, 1), (4, 256)])
def test_allreduce_world_n_is_the_descending_rank_bf16_chain(W, N):
    sp = TP2.all_reduce(W, N)
    params, ret = sp.signature()
    assert [n for n, _ in params] == ["parts"] and params[0][1].leaves == W * N and ret.leaves == N
    rng = np.random.default_rng(20260923 + W * 100 + N)
    parts = [_rand_words(rng, N) for _ in range(W)]
    out = _eval(Program(sp), [sum(parts, [])])
    # acc = parts[W-1]; acc = Bf16Add(acc, parts[r]) for r = W-2..0, one RNE rounding per hop
    assert out == _chain(parts, range(W - 1, -1, -1))
    if N >= 256:
        # per-hop rounding makes the order observable: the ascending chain (AllReduce_v1's) differs on this sample
        assert out != _chain(parts, range(W))


def test_allreduce_world_2_parametric_body_equals_allreduce2_bitwise():
    """The parametric body at WORLD=2 is one Bf16Add per element == `AllReduce2_v1` (the identity world 2 keeps)."""
    N = 64
    rng = np.random.default_rng(7)
    x0, x1 = _rand_words(rng, N), _rand_words(rng, N)
    assert _eval(Program(bind(TP2.AllReduce, WORLD=2, N=N)), [x0 + x1]) == _eval(Program(bind(TP2.AllReduce2, N=N)), [x0 + x1])


@pytest.mark.parametrize("W,N", [(3, 8), (4, 3)])
def test_allgather_world_n_is_the_rank_order_concatenation(W, N):
    sp = TP2.all_gather(W, N)
    params, ret = sp.signature()
    assert params[0][1].leaves == W * N and ret.leaves == W * N
    rng = np.random.default_rng(11)
    parts = [_rand_words(rng, N) for _ in range(W)]
    assert _eval(Program(sp), [sum(parts, [])]) == sum(parts, [])
    assert sp.gates == 0


def test_registry_derived_family_selectors():
    """Consumers (the v2 query, sampled_replay, tp/worker, tp/commit) ask the registry, never a spelled prefix."""
    assert collective_kinds.collective_kind("AllReduce2_v1{N=8}") == "all_reduce" == collective_kinds.collective_kind("AllReduce_v2{WORLD=3,N=8}")
    assert collective_kinds.collective_kind("AllGather2_v1") == "all_gather" == collective_kinds.collective_kind("AllGather_v1{WORLD=4,N=8}")
    assert collective_kinds.collective_kind("Gemm_v1{K=1,N=2}") is None and collective_kinds.collective_kind("") is None
    assert collective_kinds.cross_rank_families(2) == {"AllReduce2_v1", "AllGather2_v1", "EmbeddingShard_v1"}
    assert collective_kinds.cross_rank_families(3) == {"AllReduce_v2", "AllGather_v1", "EmbeddingShard_v1"}
    from verity_vllm.check.match import xrank_collectives as XC
    from verity_vllm.query import required as v1_bridge
    assert v1_bridge.collective_kind("AllGather_v1{WORLD=3,N=8}") == "all_gather"
    assert XC.collective_family("all_reduce", 2) == "AllReduce2_v1" and XC.collective_family("all_reduce", 3) == "AllReduce_v2"
    assert XC.collective_family("all_gather", 4) == "AllGather_v1"


# --- the N-ary collective bus (peer inputs per (collective, peer rank), rank order) -------------------------------------------

def test_collective_bus_world_3_peer_layout_and_rank_order():
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.export_ops import CollectiveBus, ensure_ops
    ensure_ops()
    W, rank = 3, 1
    probe = CollectiveBus(rank=rank, world=W, mode="probe")
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    probe.all_reduce(x)
    probe.all_gather(x)
    assert probe.peer_ranks == [0, 2] and [r["kind"] for r in probe.record] == ["all_reduce", "all_gather"]
    peers = probe.example_peers(device="cpu")
    assert len(peers) == 2 * (W - 1)                      # world-1 peer inputs per collective, flat index k*(world-1) + j
    # replay: peers carry rank-tagged constants; the own partial is placed at index `rank` of `parts`
    own = torch.full((2, 4), 10.0, dtype=torch.bfloat16)
    p0 = torch.full((2, 4), 1.0, dtype=torch.bfloat16)
    p2 = torch.full((2, 4), 100.0, dtype=torch.bfloat16)
    bus = CollectiveBus(rank=rank, world=W, mode="trace", peers=[p0, p2, p0, p2], record=list(probe.record))
    ar = bus.all_reduce(own)
    assert torch.equal(ar, torch.full((2, 4), 111.0, dtype=torch.bfloat16))
    ag = bus.all_gather(own)
    assert ag.shape == (2, 12) and ag[0, :4].tolist() == [1.0] * 4 and ag[0, 4:8].tolist() == [10.0] * 4 and ag[0, 8:].tolist() == [100.0] * 4


def _derive_world8_collective(monkeypatch, *, fused: bool, custom: bool):
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.derive import derive_report
    from verity_vllm.program.frontend.export_ops import ensure_ops
    from verity_vllm.program.frontend.rules import vllm_bindings as VB
    from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset

    ensure_ops()
    monkeypatch.setitem(VB._OBSERVED, "allreduce_settings",
                        {"custom_all_reduce": custom, "symmetric_memory": True, "flashinfer": True})

    class Collective(torch.nn.Module):
        def forward(self, *args):
            parts = list(args[:8])
            if fused:
                return torch.ops.verity_tp.fused_allreduce_rms_norm(parts, args[8], args[9], 1e-5)
            return torch.ops.verity_tp.all_reduce(parts)

    rows = tuple(torch.zeros(1, 6144, dtype=torch.bfloat16) for _ in range(8))
    inputs = rows + ((torch.zeros(1, 6144, dtype=torch.bfloat16), torch.ones(6144, dtype=torch.bfloat16)) if fused else ())
    return derive_report(Collective(), inputs, mode="profile", profile="b1-eager-v3", rules=vllm_ruleset(),
                         name="world8-collective", check_registry=True, rank=(8, 0))


@pytest.mark.parametrize("fused,kind", [
    (False, "AllReduceMnnvl_v1{WORLD=8,N=6144,SHOT=1}"),
    (True, "AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144,SHOT=1,CLUSTER=8,EPS=1e-05,FMA=True}"),
])
@pytest.mark.parametrize("custom", [True, False], ids=["defaults", "custom-off"])
def test_world_8_collective_rules_bind_captured_default_and_custom_off_routes(monkeypatch, fused, kind, custom):
    """Both captured switch settings bind through `tp_allreduce.covered`; no route is admitted merely because its shape matches."""
    derived = _derive_world8_collective(monkeypatch, fused=fused, custom=custom)
    assert derived.ok, derived.unsupported
    assert [n.fn.id for n in derived.program.fn.body.nodes] == [kind]


#: the TP4 capture's switches (vLLM's batch invariance, NCCL over SHM) and GPU
TP4_ENV = {"VLLM_BATCH_INVARIANT": "1", "NCCL_P2P_DISABLE": "1"}
RTX_PRO_6000 = ((12, 0), 188)


def _derive_world4_all_reduce(monkeypatch, *, env: dict, gpu: tuple, M: int, N: int = 8192):
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.derive import derive_report
    from verity_vllm.program.frontend.export_ops import ensure_ops
    from verity_vllm.program.frontend.rules import vllm_bindings as VB
    from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity.ml.definitions.registry.tp_allreduce import settings_of

    ensure_ops()
    monkeypatch.setitem(VB._OBSERVED, "allreduce_settings", settings_of(True, env))
    monkeypatch.setitem(VB._OBSERVED, "target_profile", TargetProfile(compute_capability=gpu[0], num_sms=gpu[1]))

    class Collective(torch.nn.Module):
        def forward(self, *parts):
            return torch.ops.verity_tp.all_reduce(list(parts))

    rows = tuple(torch.zeros(M, N, dtype=torch.bfloat16) for _ in range(4))
    return derive_report(Collective(), rows, mode="profile", profile="b1-eager-v3", rules=vllm_ruleset(),
                         name="world4-collective", check_registry=True, rank=(4, 0))


def test_world_4_all_reduce_binds_nccl_s_tree_on_the_rtx_pro_6000_over_shm(monkeypatch):
    """The TP4 capture's route (batch-invariant pynccl over SHM on 4 RTX PRO 6000) is one `AllReduceNcclTree_v1` call over the
    flattened [M * N] partials, down the capture's chain (M = 256, the entry's top, is `test_tp_allreduce`'s binding test)."""
    derived = _derive_world4_all_reduce(monkeypatch, env=TP4_ENV, gpu=RTX_PRO_6000, M=1)
    assert derived.ok, derived.unsupported
    assert [n.fn.id for n in derived.program.fn.body.nodes] == ["AllReduceNcclTree_v1{WORLD=4,SIZE=8192,CHAIN=(0,1,2,3)}"]


@pytest.mark.parametrize("env,gpu,M,N", [(TP4_ENV, ((9, 0), 132), 1, 8192), ({"VLLM_BATCH_INVARIANT": "1"}, RTX_PRO_6000, 1, 8192),
                                         ({"NCCL_P2P_DISABLE": "1"}, RTX_PRO_6000, 1, 8192), (TP4_ENV, RTX_PRO_6000, 257, 8192),
                                         (TP4_ENV, RTX_PRO_6000, 1, 4096)],
                         ids=["h200", "p2p-on", "not-batch-invariant", "M-257", "N-4096"])
def test_world_4_all_reduce_off_the_captured_route_is_refused_by_name(monkeypatch, env, gpu, M, N):
    derived = _derive_world4_all_reduce(monkeypatch, env=env, gpu=gpu, M=M, N=N)
    assert not derived.ok
    assert "numerics_unregistered" in repr(derived.unsupported) and "world=4" in repr(derived.unsupported), derived.unsupported


# --- tp_links.json: v2 schema at world N, v1 files read as v2 ----------------------------------------------------------------

def test_read_tp_links_upgrades_v1_and_accepts_v2(tmp_path):
    from verity_vllm.pipeline.build import read_tp_links
    v1 = {"schema": "cb-a/tp_links/v1", "world": 2, "rank": 0, "links": [{"k": 0, "kind": "all_reduce", "peer_input": "peers_0", "peer_rank": 1}]}
    (tmp_path / "v1.json").write_text(json.dumps(v1))
    d = read_tp_links(str(tmp_path / "v1.json"))
    assert d["peer_ranks"] == [1] and d["links"][0]["peers"] == [{"rank": 1, "input": "peers_0", "operand": "part_rank1", "collective_k": 0}]
    v2 = {"schema": "cb-a/tp_links/v2", "world": 3, "rank": 1, "peer_ranks": [0, 2],
          "links": [{"k": 0, "kind": "all_reduce", "own_operand": "part_rank1",
                     "peers": [{"rank": 0, "input": "peers_0", "operand": "part_rank0", "collective_k": 0}, {"rank": 2, "input": "peers_1", "operand": "part_rank2", "collective_k": 0}]}]}
    (tmp_path / "v2.json").write_text(json.dumps(v2))
    d = read_tp_links(str(tmp_path / "v2.json"))
    assert d["peer_ranks"] == [0, 2] and [p["rank"] for p in d["links"][0]["peers"]] == [0, 2]


# --- three-rank collective linkage over the capture logs -----------------------------------------------------------------------

def _rank_events(rank: int, world: int, *, ar_shape=(4, 8), steps=(0,)):
    from verity_vllm.observe.events import DispatchEnter, DispatchExit, RunHeader, StorageAlloc, TritonLaunch, ViewDesc

    def _v(a, shape, dtype="bfloat16"):
        strides = tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape)))
        return ViewDesc(a, 0, tuple(shape), strides, dtype, "cuda:0")

    stream = 1000 + rank
    hdr = RunHeader("verity-capture/log/v1", "2026-09-23T00:00:00Z", "pid", {"parallelism": {"tensor_parallel_size": world}}, {"repo": "x"},
                    {"requests": []}, notes={"tp2": {"rank": rank, "world_size": world}})
    evs = [hdr]
    eid, alloc = 10, 100
    for step in steps:
        a_out, a_res, a_g, a_gres = alloc, alloc + 1, alloc + 2, alloc + 3
        alloc += 10
        evs += [
            StorageAlloc(a_out, 0x100 + a_out, 4 * 8 * 2, "cuda:0", 0, "op"),
            TritonLaunch(eid, None, step, "forward", "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent", "sha",
                         {"a_ptr": _v(50, (4, 4)), "b_ptr": _v(1, (8, 4)), "c_ptr": _v(a_out, ar_shape)}, {}, [1], 4, 3, stream, "model.layers.0.self_attn.o_proj"),
            DispatchEnter(eid + 1, None, step, "forward", "vllm.all_reduce.default", "vllm::all_reduce(Tensor tensor, str group_name) -> Tensor",
                          [_v(a_out, ar_shape), "tp:0"], {}, stream, "model.layers.0.self_attn.o_proj"),
            StorageAlloc(a_res, 0x100 + a_res, 4 * 8 * 2, "cuda:0", 0, "op"),
            DispatchExit(eid + 1, _v(a_res, ar_shape)),
            StorageAlloc(a_g, 0x100 + a_g, 1 * 8 * 2, "cuda:0", 0, "op"),
            TritonLaunch(eid + 2, None, step, "forward", "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent", "sha",
                         {"a_ptr": _v(63, (1, 8)), "b_ptr": _v(1, (8, 8)), "c_ptr": _v(a_g, (1, 8))}, {}, [1], 4, 3, stream, "lm_head"),
            DispatchEnter(eid + 3, None, step, "forward", "vllm.all_gather.default", "vllm::all_gather(Tensor tensor, int dim, int world_size, str group_name) -> Tensor",
                          [_v(a_g, (1, 8)), -1, world, "tp:0"], {}, stream, "logits_processor"),
            StorageAlloc(a_gres, 0x100 + a_gres, 1 * 8 * world * 2, "cuda:0", 0, "op"),
            DispatchExit(eid + 3, _v(a_gres, (1, 8 * world))),
        ]
        eid += 4
    return evs


def test_link_three_synthetic_ranks():
    from tests.engine.collectives import collective_link as CL
    W = 3
    sh = {r: CL.extract_rank_shard(_rank_events(r, W), rank=r) for r in range(W)}
    for r in range(W):
        assert sh[r]["world_size"] == W and sh[r]["rows_by_kind"] == {"all_reduce": 1, "all_gather": 1}
    link = CL.link_shards(sh)
    assert link["ranks"] == [0, 1, 2] and link["world_size"] == 3 and link["by_status"] == {"linked": 2}
    ar, ag = link["links"]
    assert ar["definition"]["id"] == "AllReduce_v2{WORLD=3,N=32}" and set(ar["required_writes"]) == {"0", "1", "2"}
    assert ar["definition"]["module"] == "verity.ml.definitions.registry.b1_tp2"
    assert ag["definition"]["id"] == "AllGather_v1{WORLD=3,N=8}"
    # world 2 of the same synthetic run keeps the recorded gather identity
    sh2 = {r: CL.extract_rank_shard(_rank_events(r, 2), rank=r) for r in range(2)}
    assert CL.link_shards(sh2)["links"][1]["definition"]["id"] == "AllGather2_v1{N=8}"
    # a missing rank is a named status, never a silent two-rank link
    part = CL.link_shards({r: sh[r] for r in (0, 2)})
    assert part["by_status"] == {"missing-rank": 2} and part["world_size"] == 3


# --- three-rank cross-rank check (rank_match) --------------------------------------------------------------------------------

def _colls(rank: int, world: int, *, N: int = 8, steps: int = 2):
    """One rank's fold `collectives` for `rank_match.cross_rank_check`: per step one all_reduce (k 0) and one all_gather (k 1), one peers
    field per (collective, peer rank) as `fold._collective_prepare` lays them out (world 2: `s<s>_k<k>`; world > 2: `s<s>_k<k>_r<peer>`)."""
    peers = [r for r in range(world) if r != rank]
    out = []
    for s in range(steps):
        for k, kind, site in ((0, "all_reduce", "model.layers.0.self_attn.o_proj"), (1, "all_gather", "lm_head")):
            fields = [{"request": "r0", "param": 1, "field": (f"s{s}_k{k}" if world == 2 else f"s{s}_k{k}_r{p}"), "ordinal": 2 * s + k, "rows": 1, "offset": 0, "peer_rank": p}
                      for p in peers]
            out.append({"step": s, "k": k, "kind": kind, "site": site, "rows": 1, "N": N, "rank": rank, "world": world, "event": 10 * s + k,
                        "own_view": {"alloc": 100 + k}, "fields": fields})
    return out


def test_cross_rank_check_world_3_passes_and_names_a_missing_rank():
    from verity_vllm.check.match import rank_match as RMx
    W = 3
    res = RMx.cross_rank_check({r: _colls(r, W) for r in range(W)})
    assert res["world"] == 3 and res["ranks"] == [0, 1, 2]
    failed = [c for c in res["checks"] if c["name"] in ("count", "order", "op", "segmentation", "site", "shard_linkage") and not c["pass"]]
    assert not failed, failed
    # ranks 0 and 2 only: not a world -- refused by name, never folded into a two-rank check
    bad = RMx.cross_rank_check({0: _colls(0, W), 2: _colls(2, W)})
    assert bad["verdict"] == "FAIL" and bad["checks"][0]["name"] == "ranks"
    # a rank that issued one collective fewer is named
    short = {r: _colls(r, W) for r in range(W)}
    short[2] = short[2][:-1]
    res2 = RMx.cross_rank_check(short)
    cnt = next(c for c in res2["checks"] if c["name"] == "count")
    assert not cnt["pass"] and any("rank" in f for f in cnt["fails"])


# --- worlds 2, 4 and 8 ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("world_args,W", [(["--tp", "2"], 2), (["--tp", "4"], 4), (["--world", "8"], 8), (["--tp", "8", "--world", "8"], 8)])
def test_rank_commit_takes_worlds_2_4_and_8(world_args, W, tmp_path):
    from verity_vllm.pipeline import cli
    args = cli.parse("tp-commit", ["--workload", str(tmp_path / "wl.json"), "--out", str(tmp_path / "out"), *world_args])
    assert (args.tp if args.world is None else args.world) == W


@pytest.mark.parametrize("W", [2, 4, 8])
def test_config_run_row_takes_worlds_2_4_and_8(W, monkeypatch):
    from verity_vllm.pipeline import cli, row as ROW, row_tp
    seen = {}

    class Rec:
        def __init__(self, **kw):
            seen.update(kw)

        def run(self):
            return 0

    monkeypatch.setattr(row_tp, "TpRow", Rec)
    o = cli.parse("row", ["run", f"qwen3-14b__bf16__rtxpro6000__tp{W}__b1__i256__o32__mixed__greedy__bi-eager", "QWEN3_14B", "Qwen/Qwen3-14B",
                          "a" * 40, "--config-run", "1", "--replay-k", "460", "--world", str(W)])
    assert ROW.run_row(o, "build,commit") == 0 and seen["world"] == W and seen["commit"].config_run == "1"


@pytest.mark.parametrize("W", [2, 4, 8])
@pytest.mark.parametrize("kind", ["all_reduce", "all_gather"])
def test_collective_record_names_own_and_peer_operands_at_worlds_2_4_and_8(W, kind):
    """World 2's binary ops keep their record's keys; a world-N op takes its `parts` as one list in rank order, and its record names
    the rank's own operand and every peer's input."""
    from types import SimpleNamespace
    import torch
    from verity_vllm.program.frontend import emit as E
    for rank in range(W):
        g = torch.fx.Graph()
        parts = [g.placeholder(f"p{r}") for r in range(W)]
        peer_in = {parts[p]: f"in.{kind}.k0.r{p}" for p in range(W) if p != rank}
        ctx = SimpleNamespace(is_input_storage=lambda a: peer_in.get(a))
        rec = E._collective_record(ctx, SimpleNamespace(args=tuple(parts) if W == 2 else (parts,)), kind, 3)
        assert rec["kind"] == kind and rec["occurrence"] == 3 and rec["own_operand"] == rank and rec["operand_ranks"] == list(range(W))
        if W == 2:
            assert list(rec) == ["kind", "occurrence", "operand_ranks", "own_operand", "peer_operand", "peer_input"]
            assert rec["peer_operand"] == 1 - rank and rec["peer_input"] == f"in.{kind}.k0.r{1 - rank}"
        else:
            assert rec["peer_operand"] is None and rec["peer_operands"] == [p for p in range(W) if p != rank]
            assert rec["peer_inputs"] == [f"in.{kind}.k0.r{p}" for p in range(W) if p != rank]


@pytest.mark.parametrize("W", [2, 4, 8])
def test_own_partial_rows_are_the_ranks_part_at_worlds_2_4_and_8(W):
    """A collective row's operand is `world` equal parts in rank order: rank r's own rows are its part's (N leaves for a per-row
    collective, M * N for a flat one), each named by the node producing it, not only the first (rank 0) or the last (rank W-1) run."""
    from verity_vllm.check.match import rank_match as RMx
    N, spec = 7, lambda k: f"spec{k}"
    runs = [("n" if r % 2 else "p", r, 0, N) for r in range(W)]
    for rank in range(W):
        assert RMx._own_rows(runs, {"rank": rank, "world": W, "N": N}, spec) == ([f"spec{rank}"] if rank % 2 else ["<not-a-node>"])
    merged = [("n", 10, 0, 2 * N)] + [("p", 1, 0, N) for r in range(2, W)]
    assert RMx._own_rows(merged, {"rank": 0, "world": W, "N": N}, spec) == ["spec10"]
    assert RMx._own_rows(merged, {"rank": 1, "world": W, "N": N}, spec) == ["spec10"]
    M = 3                       # flat: part r is rank r's M rows; rank 1's rows produced by nodes 20, 21, 22, the others peers fields
    flat = [("p", 1, 0, M * N)] + [("n", 20 + i, 0, N) for i in range(M)] + [("p", 1, M * N, (W - 1) * M * N)] if W > 2 else \
        [("p", 1, 0, M * N)] + [("n", 20 + i, 0, N) for i in range(M)]
    assert RMx._own_rows(flat, {"rank": 1, "world": W, "N": N}, spec) == ["spec20", "spec21", "spec22"]
    assert RMx._own_rows(flat, {"rank": 0, "world": W, "N": N}, spec) == ["<not-a-node>"] * M


def test_a_fused_norm_recorder_row_is_the_collective_its_path_folds_to_and_the_unfused_ones_are_the_commit_s():
    """The recorder files each `fused_allreduce_rms_norm` call as `fused_norm.KIND` over `stack([partial, residual])`: on its unfused path
    the fold's `all_reduce` over the [M, N] partial, on its fused path the fold's fused collective.  The provenance check leaves the unfused
    calls to the Commit's replay, so they count with the checked ones."""
    from verity_vllm.check.match import rank_match as RMx
    from verity_vllm.engine.collectives import fused_norm as FN
    W, steps = 3, 2
    colls = {r: _colls(r, W, steps=steps) for r in range(W)}

    def rec(r, path):
        out = []
        for c in colls[r]:
            fused = c["kind"] == "all_reduce"
            out.append({"step": c["step"], "k": c["k"], "kind": FN.KIND if fused else c["kind"], "module": c["site"],
                        "shape": [2, 1, 8] if fused else [1, 8], **({"path": path} if fused else {}),
                        "out_shape": [1, 8 * (W if c["kind"] == "all_gather" else 1)], "part_sha256": f"p{r}{c['step']}{c['k']}",
                        "out_sha256": f"o{c['step']}{c['k']}"})
        return out

    common = steps * 2
    prov = {"pass": True, "collectives_checked": common - steps, "fused_norm_unfused_left_to_commit": steps, "recorder_sha256": {str(r): f"h{r}" for r in range(W)},
            "tensors_digested": 2 * W * common, "n_digest_mismatches": 0}
    shas = {r: f"h{r}" for r in range(W)}
    res = RMx.cross_rank_check(colls, recorder={r: rec(r, FN.UNFUSED) for r in range(W)}, provenance=prov, recorder_sha256=shas)
    assert {ch["name"]: ch["pass"] for ch in res["checks"]}["recorder"] and res["verdict"] == "PASS", res["failed"]
    fused = RMx.cross_rank_check(colls, recorder={r: rec(r, FN.FUSED) for r in range(W)})
    fails = next(ch for ch in fused["checks"] if ch["name"] == "recorder")["fails"]
    assert fails and all("folds as fused_allreduce_rms_norm" in f for f in fails), fails
    short = RMx.cross_rank_check(colls, recorder={r: rec(r, FN.UNFUSED) for r in range(W)}, provenance={**prov, "fused_norm_unfused_left_to_commit": 0},
                                 recorder_sha256=shas)
    assert "provenance" in short["failed"]


def test_a_flat_all_reduce_counts_one_producer_per_own_row():
    from verity_vllm.check.match import rank_match as RMx
    W, M = 3, 4
    colls = {r: [dict(c, rows=M, flat=True, fields=[dict(f, rows=M) for f in c["fields"]]) for c in _colls(r, W, steps=1) if c["kind"] == "all_reduce"] for r in range(W)}
    ok = {r: {(0, 0): ["DeepGemmFp8Linear_v1{K=2048,N=6144}"] * M} for r in range(W)}
    assert "producer" not in RMx.cross_rank_check(colls, producers=ok)["failed"]
    one = {r: {(0, 0): ["DeepGemmFp8Linear_v1{K=2048,N=6144}"]} for r in range(W)}
    assert "producer" in RMx.cross_rank_check(colls, producers=one)["failed"]


@pytest.mark.parametrize("ops,admitted", [
    ((("MoeSum_v1{TOPK=8,H=8}", 0), ("DeepGemmFp8Linear_v1{K=2048,N=8}", 0)), True),     # GLM-5.3's MoE block: routed + shared partials
    ((("MoeSum_v1{TOPK=8,H=8}", 0), ("FlashInferFp8Linear_v1{K=256,N=8}", 0)), True),    # the decode steps' shared expert (FlashInfer at small M)
    ((("MoeSum_v1{TOPK=8,H=8}", 0), ("RMSNormCudaVector8_v1{N=8}", 0)), False),         # a partial plus the residual stream: no partial
    ((("MoeSum_v1{TOPK=8,H=8}", 0), ("DeepGemmFp8Linear_v1{K=2048,N=8}", 1)), False),   # an operand of another step
])
def test_a_residual_add_is_a_partial_producer_only_over_partials_of_its_rank_and_step(tmp_path, ops, admitted):
    """`ResidualAdd_v1` producing an own partial is admitted only when every operand row is an admitted producer's of the same step."""
    from verity_vllm.check.match import rank_match as RMx
    N, W, rank = 8, 3, 1
    rows = [{"node": i, "step": st, "spec": sp, "operand_refs": []} for i, (sp, st) in enumerate(ops)]
    rows.append({"node": 2, "step": 0, "spec": f"ResidualAdd_v1{{N={N}}}", "operand_refs": [[["n", 0, 0, N]], [["n", 1, 0, N]]]})
    rows.append({"node": 3, "step": 0, "spec": f"AllReduceMnnvl_v1{{WORLD={W},N={N},SHOT=1}}", "rows": 1,
                 "collective": {"kind": "all_reduce", "step": 0, "k": 0, "rank": rank, "world": W, "N": N},
                 "operand_refs": [[["p", 1, 0, N], ["n", 2, 0, N], ["p", 1, N, 2 * N]]]})
    (tmp_path / "instances.jsonl").write_text("\n".join(json.dumps(r) for r in [{"schema": "x"}] + rows) + "\n")
    (name,) = RMx.own_producers_from_record(tmp_path / "instances.jsonl")[(0, 0)]
    assert name.startswith(f"ResidualAdd_v1{{N={N}}}<-") and RMx._admitted(name) is admitted, name
    assert not RMx._admitted(f"ResidualAdd_v1{{N={N}}}<-MoeSum_v1|<not-a-node>") and not RMx._admitted("ResidualAdd_v1{N=8}")


@pytest.mark.parametrize("own,admitted", [
    ("MoeSum_v1{TOPK=6,H=8}", True),
    ("ResidualAdd_v1{N=8}<-DeepGemmFp8Linear_v1{K=384,N=8}|MoeSum_v1{TOPK=6,H=8}", True),   # the routed sum plus the shared expert's down-proj
    ("ResidualAdd_v1{N=8}<-DeepGemmFp8Linear_v1{K=384,N=8}|Gemm_v1{K=384,N=8}", False),     # two linears and no routed sum
    ("DeepGemmFp8Linear_v1{K=384,N=8}", False),                                             # the shared expert alone
])
def test_a_moe_sites_partial_is_a_routed_sum_to_which_the_shared_experts_partial_may_be_added(own, admitted):
    """A MoE runner that reduces its shared expert's output with the routed sum (DeepSeek-V4-Pro's `MoERunner` at TP8) issues the
    all-reduce of their `ResidualAdd_v1`: the links check admits it at the MoE site, and refuses a partial with no routed sum."""
    from verity_vllm.check.match import rank_match as RMx
    W, N, site = 3, 8, "model.layers.1.ffn.experts.runner"
    colls = {r: [dict(c, site=site) for c in _colls(r, W, steps=1) if c["kind"] == "all_reduce"] for r in range(W)}
    links = {r: [{"k": 0, "peer_collective_k": 0, "peers": [{"rank": p, "collective_k": 0} for p in range(W) if p != r], "kind": "all_reduce",
                  "shape": [1, N], "module": site, "module_path": f"{site} (MoERunner)"}] for r in range(W)}
    res = RMx.cross_rank_check(colls, producers={r: {(0, 0): [own]} for r in range(W)}, links=links)
    assert ("links" not in res["failed"]) is admitted, next(c for c in res["checks"] if c["name"] == "links")["fails"]
