"""The MoE align source (`verity_vllm/acquire/sources/moe_align_source.py`) on CPU: DeepSeek-V4-Pro's Marlin experts call the
module-global `moe_align_block_size` of experts/marlin_moe.py, emulated here as the pinned kernel writes its outputs (the pairs taking
their atomicAdd slots in a random order, what it leaves unwritten holding a pair index).

It catches a recomputed rank, start, blk or par that is not what `MoeAlignRanks_v1` / `MoeAlignBlocks_v1` compute over the GPU's
schedule (or that does not place each pair where the kernel did); a sorted_token_ids, expert_ids or num_tokens_post_padded the
recompute does not hold to (a pair moved, duplicated or dropped, a padding slot overwritten, a block's expert or the padded count
off); an unwritten slot read as a placement; values committed under labels the binder does not bind to the align rows of DeepSeek-V4's
manifest by port; an expert_map or a second call in one module accepted; and the blocks' call boundaries left unclaimed."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from tests.acquire.test_glm53_sources import Committer
from tests.program.test_marlin_moe import _vllm_align
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import moe_align_source as MA
from verity_vllm.acquire.sources import oracle_witness as OW
from verity_vllm.acquire.sources import taps
from verity.ml.definitions.registry import marlin_moe as MM
from verity_vllm.query.manifest.bind import Binder, source_ports


def kernel(ids: list[int], E: int, BM: int, order, *, unwritten: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sorted_token_ids, expert_ids, num_tokens_post_padded) as `moe_align_block_size` returns them, the pairs taking their slots in
    `order`; the slots past num_tokens_post_padded, which the kernel never writes (`torch.empty`), hold `unwritten`."""
    P = len(ids)
    pos, npp = _vllm_align(ids, E, BM, order)
    L = P + E * (BM - 1) if P >= E else min(P * BM, P + E * (BM - 1))
    sorted_ids = torch.full((L,), unwritten, dtype=torch.int32)
    sorted_ids[:npp] = P
    for p, q in enumerate(pos):
        if q >= 0:
            sorted_ids[q] = p
    expert_ids = torch.full((-(-L // BM),), unwritten, dtype=torch.int32)
    off = 0
    for e in range(E):
        n = -(-sum(1 for i in ids if i == e) // BM)
        expert_ids[off:off + n] = e
        off += n
    return sorted_ids, expert_ids, torch.tensor([npp], dtype=torch.int32)


def words(t: torch.Tensor) -> list[int]:
    return [int(x) & MM.M32 for x in t.view(torch.int32).reshape(-1).tolist()]


CASES = [(4, 1, 8), (5, 12, 8), (6, 23, 48), (3, 30, 4), (8, 6, 16)]


@pytest.mark.parametrize("E,P,BM", CASES)
def test_values_are_the_definitions_over_the_gpu_schedule(E: int, P: int, BM: int) -> None:
    """In every atomic order: the ranks are `MoeAlignRanks_v1`'s over the GPU's sorted positions and over themselves as keys (the key
    the replay binds), start its second port, (blk, par) `MoeAlignBlocks_v1`'s over them, and each pair of [0, E) sits where the kernel
    placed it; experts outside [0, E) (ignored by the kernel) included."""
    rng = np.random.default_rng(P * 31 + E)
    for trial in range(5):
        ids = rng.choice([0, 0, 0, E - 1, E - 1, 1, -1, E, 7 - 2**31], P).tolist() if trial % 3 == 2 else rng.integers(0, E, P).tolist()
        order = rng.permutation(P)
        sorted_ids, expert_ids, ntpp = kernel(ids, E, BM, order, unwritten=int(rng.integers(0, P)))
        rank, start, blk, par, ok = MA.align_values(torch.tensor(ids, dtype=torch.int32).reshape(-1, 1), sorted_ids, expert_ids, ntpp, E, BM)
        assert bool(ok), (ids, order)
        pos, npp = _vllm_align(ids, E, BM, order)
        u = [i & MM.M32 for i in ids]
        ranks = bind(MM.MoeAlignRanks, E=E, P=P, BM=BM)
        assert list(evaluate(ranks, u, words(rank), list(range(P)))) == words(rank) + words(start)
        gpu = evaluate(ranks, u, [q & MM.M32 if q >= 0 else r for q, r in zip(pos, words(rank))], list(range(P)))
        assert list(gpu) == words(rank) + words(start)
        assert list(evaluate(bind(MM.MoeAlignBlocks, P=P, BM=BM), u, words(rank), words(start))) == words(blk) + words(par)
        assert [b * BM + r % BM for b, r, q in zip(blk.tolist(), rank.tolist(), pos) if q >= 0] == [q for q in pos if q >= 0]
        assert int(par) * BM == npp


def _tamper(sorted_ids, expert_ids, ntpp, ids, E, BM, how):
    s, x, n = sorted_ids.clone(), expert_ids.clone(), ntpp.clone()
    P, npp = len(ids), int(ntpp)
    placed = [q for q in range(npp) if int(s[q]) < P]
    pads = [q for q in range(npp) if int(s[q]) == P]
    if how == "swap":                 # two pairs of different experts trade slots
        a = placed[0]
        b = next(q for q in placed if ids[int(s[q])] != ids[int(s[a])])
        s[a], s[b] = s[b].clone(), s[a].clone()
    elif how == "dup":                # a padding slot holds a placed pair again
        s[pads[0]] = s[placed[0]]
    elif how == "drop":               # a placed pair's slot holds the padding value
        s[placed[-1]] = P
    elif how == "pad":                # a padding slot holds something else
        s[pads[0]] = P + 3
    elif how == "expert":
        x[0] = (int(x[0]) + 1) % E
    elif how == "count":
        n += BM
    return s, x, n


@pytest.mark.parametrize("how", ["swap", "dup", "drop", "pad", "expert", "count"])
def test_outputs_the_kernel_did_not_write_are_refused(how: str) -> None:
    E, P, BM = 4, 13, 4
    rng = np.random.default_rng(7)
    ids = [0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 1, 2]
    sorted_ids, expert_ids, ntpp = kernel(ids, E, BM, rng.permutation(P))
    topk = torch.tensor(ids, dtype=torch.int32).reshape(-1, 1)
    assert bool(MA.align_values(topk, sorted_ids, expert_ids, ntpp, E, BM)[-1])
    assert not bool(MA.align_values(topk, *_tamper(sorted_ids, expert_ids, ntpp, ids, E, BM, how), E, BM)[-1])


# ---------------------------------------------------------------- the source on a rank's model


E_, BM_, TOPK, M = 6, 4, 2, 5


class FusedMoE(torch.nn.Module):
    """An MoE layer's experts as `fused_marlin_moe` reaches the align: the module-global `moe_align_block_size` of experts/marlin_moe.py,
    looked up at call time, with the pinned arguments."""

    def __init__(self, ids: torch.Tensor) -> None:
        super().__init__()
        self.ids, self.top_k, self.expert_map, self.calls = ids, TOPK, None, 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.calls):
            sys.modules[MA.MARLIN_MODULE].moe_align_block_size(self.ids, BM_, E_, self.expert_map, ignore_invalid_experts=True)
        return x


def model_of(ids: list[torch.Tensor]) -> torch.nn.Module:
    root, inner = torch.nn.Module(), torch.nn.Module()
    layers = torch.nn.ModuleList()
    for t in ids:
        layer, ffn = torch.nn.Module(), torch.nn.Module()
        ffn.experts = FusedMoE(t)
        layer.ffn = ffn
        layers.append(layer)
    inner.layers = layers
    root.model = inner
    return root


def fake_marlin(monkeypatch, *, tamper: str | None = None, seed: int = 0) -> None:
    """The kernel, its atomic order sampled from ``seed`` and the call's pair count: two runs of other seeds schedule apart."""
    def moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False, ignore_invalid_experts=False):
        ids = topk_ids.reshape(-1).tolist()
        out = kernel(ids, num_experts, block_size, np.random.default_rng(len(ids) + seed).permutation(len(ids)), unwritten=0)
        return _tamper(*out, ids, num_experts, block_size, tamper) if tamper else out

    m = types.ModuleType(MA.MARLIN_MODULE)
    m.moe_align_block_size = moe_align_block_size
    monkeypatch.setitem(sys.modules, MA.MARLIN_MODULE, m)


def align_rows(layer: int, P: int, step: int = 0) -> list[dict]:
    """Layer `layer`'s align rows at `step` as DeepSeek-V4-Pro's manifest of record spells them, and the module's partial beside them."""
    ex = f"model.layers.{layer}.ffn.experts"
    ranks, blocks = f"MoeAlignRanks_v1{{E={E_},P={P},BM={BM_}}}", f"MoeAlignBlocks_v1{{P={P},BM={BM_}}}"
    base = {"engine_step": step, "step": step, "request_id": "r0", "invocation": 0, "rank": 0, "dtype": "f32"}
    return [{**base, "family": "instance_outputs", "op_path": ex, "output_member": "rank.0", "spec": ranks, "port": "0", "position": [10 + layer, 0]},
            {**base, "family": "instance_outputs", "op_path": ex, "output_member": "rank.1", "spec": ranks, "port": "1", "position": [10 + layer, 1]},
            {**base, "family": "call_boundaries", "op_path": f"{ex}/marlin_moe_align_{layer}", "output_member": "0", "spec": blocks, "port": "0",
             "position": [11 + layer, 0]},
            {**base, "family": "call_boundaries", "op_path": f"{ex}/marlin_moe_align_{layer}", "output_member": "1", "spec": blocks, "port": "1",
             "position": [11 + layer, 1]},
            {**base, "family": "tp_rank_partials", "op_path": ex, "output_member": "part_rank0", "spec": "ResidualAdd_v1", "port": "out",
             "position": [12 + layer, 0], "dtype": "bf16"}]


def ids_of(seed: int) -> torch.Tensor:
    return torch.tensor(np.random.default_rng(seed).integers(0, E_, (M, TOPK)), dtype=torch.int32)


def run_step(src, c, model) -> None:
    src.on_step_begin()
    model.model.layers[0].ffn.experts(torch.zeros(1))
    model.model.layers[1].ffn.experts(torch.zeros(1))
    src.on_step_end(c._cur)


def test_the_source_commits_the_schedule_and_binds_it_by_port(monkeypatch) -> None:
    fake_marlin(monkeypatch)
    ids = [ids_of(1), ids_of(2)]
    model, c = model_of(ids), Committer()
    src = MA.MoeAlignSource(c, model)
    src.install()
    try:
        run_step(src, c, model)
    finally:
        src.uninstall()
    assert not c.errors, c.errors
    names = [n for _k, n in c.log]
    assert names == [f"model.layers.{k}.ffn.experts/{m}" for k in (0, 1) for m in MA.MEMBERS]
    assert all(c.acquired[n].dtype == torch.float32 for n in names)
    for k, t in enumerate(ids):
        flat = t.reshape(-1).tolist()
        want = MA.align_values(t, *kernel(flat, E_, BM_, np.random.default_rng(len(flat)).permutation(len(flat))), E_, BM_)
        for m, w in zip(MA.MEMBERS, want):
            assert words(c.acquired[f"model.layers.{k}.ffn.experts/{m}"]) == words(w)
    P = M * TOPK
    got = Binder([r for k in (0, 1) for r in align_rows(k, P)], rank=0, ports=source_ports([src])).bind(0, names)
    assert got == [(f"model.layers.{k}.ffn.experts{sub}", member, 0, "port") for k in (0, 1)
                   for sub, member in (("", "rank.0"), ("", "rank.1"), (f"/marlin_moe_align_{k}", "0"), (f"/marlin_moe_align_{k}", "1"))]
    assert src.stats["calls"] == 2 and src.stats["pairs"] == 2 * P
    assert src.extra_step_bytes({}, model) == {"per_token_bytes": 3 * 4 * 2 * TOPK, "per_seq_bytes": 4 * 2, "tensors_per_step": 8}


@pytest.mark.parametrize("tamper", ["swap", "count"])
def test_a_kernel_output_off_the_schedule_is_a_collector_error(monkeypatch, tamper: str) -> None:
    fake_marlin(monkeypatch, tamper=tamper)
    model, c = model_of([ids_of(3), ids_of(4)]), Committer()
    src = MA.MoeAlignSource(c, model)
    src.install()
    try:
        run_step(src, c, model)
    finally:
        src.uninstall()
    assert len(c.errors) == 2 and all("moe_align_block_size's outputs are not the kernel's" in e for e in c.errors)
    assert "model.layers.0.ffn.experts at engine step 0" in c.errors[0]


def test_an_expert_map_or_a_second_call_is_refused(monkeypatch) -> None:
    fake_marlin(monkeypatch)
    model, c = model_of([ids_of(5), ids_of(6)]), Committer()
    model.model.layers[0].ffn.experts.expert_map = torch.arange(E_)
    model.model.layers[1].ffn.experts.calls = 2
    src = MA.MoeAlignSource(c, model)
    src.install()
    try:
        run_step(src, c, model)
        sys.modules[MA.MARLIN_MODULE].moe_align_block_size(ids_of(7), BM_, E_, None)
    finally:
        src.uninstall()
    assert any("model.layers.0.ffn.experts passed an expert_map" in e for e in c.errors)
    assert any("model.layers.1.ffn.experts called moe_align_block_size 2 times" in e for e in c.errors)
    assert any("outside every MoE module" in e for e in c.errors)
    assert not any(n.startswith("model.layers.0.") for _k, n in c.log)


def test_the_blocks_call_boundaries_are_claimed_and_attached_on_a_rank() -> None:
    rows = align_rows(3, M * TOPK)
    assert [CB.claimed(r) for r in rows] == [None, None, "moe_align", "moe_align", None]
    man = {"ranks": 8, "identities": rows}
    assert "moe_align" in taps.attached(man) and CB.unattached(man, taps.attached(man)) == []
    assert "moe_align" not in taps.attached({"ranks": 1, "identities": rows})
    assert not CB.manifest_sites(man)


def test_the_match_capture_records_the_witness_the_commit_commits(monkeypatch, tmp_path) -> None:
    """The capture's witness rows are the digests of the very bytes the Commit's source commits for the same calls, at each call's
    position among the step's collectives, with the source's work inside the observer's quiet block; a dense model records none."""
    import contextlib
    import json

    fake_marlin(monkeypatch)
    ids = [ids_of(1), ids_of(2)]
    model, c = model_of(ids), Committer()
    src = MA.MoeAlignSource(c, model)
    src.install()
    try:
        run_step(src, c, model)
    finally:
        src.uninstall()
    pos, quiet, real = [0], [], MA.align_values

    @contextlib.contextmanager
    def hide():
        quiet.append(True)
        yield
        quiet.pop()

    def align_values(*a):
        assert quiet, "the recompute ran where the observer sees it"
        return real(*a)

    monkeypatch.setattr(MA, "align_values", align_values)
    w = MA.capture_witness(model, 5, lambda: pos[0], hide)
    for step in (0, 1):
        w.step(step)
        for k in (0, 1):
            pos[0] = 3 * k + 2                   # the collectives the rank issued before this layer's align call
            model.model.layers[k].ffn.experts(torch.zeros(1))
    got = w.close(tmp_path)
    rows = [json.loads(ln) for ln in (tmp_path / OW.witness_file(5)).read_text().splitlines()]
    assert got == {"calls": 4, "rows": 16, "schedule_calls": 4, "errors_n": 0, "errors": []} and len(rows) == 16
    want = {f"model.layers.{k}.ffn.experts/{m}": OW.digest(c.acquired[f"model.layers.{k}.ffn.experts/{m}"].numpy().tobytes())
            for k in (0, 1) for m in MA.MEMBERS}
    assert [(r["step"], r["k"], f"{r['module']}/{r['member']}") for r in rows] == [
        (s, 3 * k + 2, f"model.layers.{k}.ffn.experts/{m}") for s in (0, 1) for k in (0, 1) for m in MA.MEMBERS]
    assert all(r["rank"] == 5 and r["sha256"] == want[f"{r['module']}/{r['member']}"] for r in rows)
    sched = torch.load(tmp_path / OW.schedule_file(5), weights_only=True)
    assert sched["rank"] == 5 and [(x["step"], x["module"]) for x in sched["calls"]] == [
        (s, f"model.layers.{k}.ffn.experts") for s in (0, 1) for k in (0, 1)]
    for x in sched["calls"]:
        assert x["ids"].tolist() == ids[int(x["module"].split(".")[2])].reshape(-1).tolist()
        assert x["rank"].view(torch.float32).numpy().tobytes() == c.acquired[f"{x['module']}/align.rank"].numpy().tobytes()
    assert MA.capture_witness(torch.nn.Linear(2, 2), 0, lambda: 0, hide) is None


# ---------------------------------------------------------------- the Commit replays the capture's schedule


@pytest.mark.parametrize("E,P,BM", CASES)
def test_the_replayed_output_is_the_kernels_for_the_recorded_order(E: int, P: int, BM: int) -> None:
    """Two atomic orders of one call: rewritten to the other order's recorded ranks, this order's sorted_token_ids is the kernel's for
    that order slot for slot, `align_values` holds it to the kernel's outputs and recomputes the recorded ranks, and the slots past
    num_tokens_post_padded are untouched; experts outside [0, E) (ignored by the kernel) included."""
    rng = np.random.default_rng(P * 17 + E)
    for trial in range(5):
        ids = rng.choice([0, 0, 0, E - 1, E - 1, 1, -1, E, 7 - 2**31], P).tolist() if trial % 3 == 2 else rng.integers(0, E, P).tolist()
        topk = torch.tensor(ids, dtype=torch.int32).reshape(-1, 1)
        mine, theirs = kernel(ids, E, BM, rng.permutation(P), unwritten=7), kernel(ids, E, BM, rng.permutation(P), unwritten=3)
        cap_rank = MA.align_values(topk, *theirs, E, BM)[0]
        sorted_ids = mine[0].clone()
        same, placed = MA.replay_schedule(topk, sorted_ids, E, BM, topk.reshape(-1), cap_rank)
        assert bool(same) and bool(placed)
        npp = int(mine[2])
        assert sorted_ids[:npp].tolist() == theirs[0][:npp].tolist() and sorted_ids[npp:].tolist() == mine[0][npp:].tolist()
        rank, *_rest, ok = MA.align_values(topk, sorted_ids, mine[1], mine[2], E, BM)
        assert bool(ok) and rank.tolist() == cap_rank.tolist()


def test_a_call_routed_otherwise_or_a_record_that_misplaces_a_pair_is_not_replayed() -> None:
    E, P, BM = 4, 13, 4
    ids = [0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 1, 2]
    topk = torch.tensor(ids, dtype=torch.int32).reshape(-1, 1)
    rng = np.random.default_rng(3)
    mine, theirs = kernel(ids, E, BM, rng.permutation(P)), kernel(ids, E, BM, rng.permutation(P))
    cap_rank = MA.align_values(topk, *theirs, E, BM)[0]
    other = topk.reshape(-1).clone()
    other[[0, 7]] = other[[7, 0]]
    dup = cap_rank.clone()
    dup[1] = dup[0]
    past = cap_rank.clone()
    past[7] = 2                                                      # expert 2 has two pairs
    for cap_ids, rank, want in ((other, cap_rank, (False, True)), (topk.reshape(-1), dup, (True, False)),
                                (topk.reshape(-1), past, (True, False)), (topk.reshape(-1)[:-1], cap_rank[:-1], (False, False))):
        sorted_ids = mine[0].clone()
        got = MA.replay_schedule(topk, sorted_ids, E, BM, cap_ids, rank)
        assert (bool(got[0]), bool(got[1])) == want and sorted_ids.tolist() == mine[0].tolist()


def _capture(monkeypatch, tmp_path, ids: list[torch.Tensor], steps: int = 2) -> tuple[dict, list[dict]]:
    """A Match capture of `model_of(ids)` at rank 0 under the kernel's atomic order of seed 0: (its witness rows by (step, module, member), its
    schedule file's calls)."""
    import contextlib
    import json

    fake_marlin(monkeypatch, seed=0)
    w = MA.capture_witness(model_of(ids), 0, lambda: 0, contextlib.nullcontext)
    model = w.src.modules
    for s in range(steps):
        w.step(s)
        for _n, m in model:
            m(torch.zeros(1))
    w.close(tmp_path)
    rows = [json.loads(ln) for ln in (tmp_path / OW.witness_file(0)).read_text().splitlines()]
    return {(r["step"], r["module"], r["member"]): r["sha256"] for r in rows}, torch.load(tmp_path / OW.schedule_file(0), weights_only=True)["calls"]


def _commit(src, c, model, steps: int = 2) -> dict:
    """The Commit's committed witness digests by (step, module, member) over ``steps`` steps."""
    got = {}
    for s in range(steps):
        c._cur.step = s
        run_step(src, c, model)
        got |= {(s, n.rsplit("/", 1)[0], n.rsplit("/", 1)[1]): OW.digest(c.acquired[n].numpy().tobytes())
                for n in c.acquired if n.rsplit("/", 1)[1] in MA.MEMBERS}
    return got


def test_the_commit_replays_the_captures_schedule_and_commits_the_captures_witness(monkeypatch, tmp_path) -> None:
    """The Commit's kernel hands out other slots than the capture's (seed 1 against 0), so unarmed its witness differs from the capture's;
    armed with the capture's schedule file, every call is replayed and every committed member is the capture's, byte for byte.  A new
    run (the record after a warm-up) counts only its own calls."""
    ids = [ids_of(1), ids_of(2)]
    cap, _calls = _capture(monkeypatch, tmp_path, ids)
    fake_marlin(monkeypatch, seed=1)
    model, c = model_of(ids), Committer()
    plain = MA.MoeAlignSource(c, model)
    plain.install()
    try:
        assert _commit(plain, c, model) != cap
    finally:
        plain.uninstall()
    assert MA.replay_stats([plain]) is None
    c = Committer()
    src = MA.MoeAlignSource(c, model)
    assert MA.arm_replay([object(), src], tmp_path, 0) == {"armed": True, "file": str(tmp_path / OW.schedule_file(0)), "calls": 4}
    src.install()
    try:
        assert _commit(src, c, model) == cap
    finally:
        src.uninstall()
    assert not c.errors, c.errors
    assert MA.replay_stats([src]) == {"replayed": 4, "routing_differs": 0, "missing": 0, "routing_differs_first": []}
    src.install()
    src.uninstall()
    assert MA.replay_stats([src])["replayed"] == 0


def test_a_call_routed_otherwise_runs_its_own_schedule_and_is_counted(monkeypatch, tmp_path) -> None:
    """Layer 1 routes otherwise in the Commit than in the capture: its calls keep the kernel's own order (their witness differs, the
    oracle names it) and are counted with where they are; layer 0 is replayed.  A capture without a schedule file arms nothing."""
    _cap, _calls = _capture(monkeypatch, tmp_path, [ids_of(1), ids_of(2)])
    fake_marlin(monkeypatch, seed=1)
    model, c = model_of([ids_of(1), ids_of(9)]), Committer()
    src = MA.MoeAlignSource(c, model)
    MA.arm_replay([src], tmp_path, 0)
    src.install()
    try:
        _commit(src, c, model)
    finally:
        src.uninstall()
    assert not c.errors, c.errors
    assert MA.replay_stats([src]) == {"replayed": 2, "routing_differs": 2, "missing": 0,
                                      "routing_differs_first": [[0, "model.layers.1.ffn.experts"], [1, "model.layers.1.ffn.experts"]]}
    fresh = MA.MoeAlignSource(Committer(), model)
    got = MA.arm_replay([fresh], tmp_path / "none", 0)
    assert got is not None and got["armed"] is False and "predates the schedule record" in got["why"]
    assert fresh.replay is None and MA.replay_stats([fresh]) is None and MA.arm_replay([], tmp_path, 0) is None


def test_the_rank_worker_arms_the_replay_once_before_the_committer_installs(monkeypatch, tmp_path) -> None:
    """`tp2_commit_install` arms the rank's align source with the capture's schedule before its committer installs: once, at the
    warm-up's install, and the record's install reuses it.  A Commit without a capture arms nothing."""
    from verity_vllm.pipeline.tp import rank_worker as RW

    _capture(monkeypatch, tmp_path, [ids_of(1), ids_of(2)])
    events: list = []
    arm = MA.MoeAlignSource.arm
    monkeypatch.setattr(MA.MoeAlignSource, "arm", lambda self, calls: (events.append(("arm", len(calls))), arm(self, calls))[1])

    def worker(capture_dir: str | None) -> tuple[object, MA.MoeAlignSource]:
        src = MA.MoeAlignSource(Committer(), model_of([ids_of(1), ids_of(2)]))
        com = types.SimpleNamespace(sources=[object(), src], reset=lambda: None,
                                    install=lambda: events.append(("install", src.replay is not None)))
        ext = object.__new__(RW.TP2CaptureWorkerExtension)
        ext._tp2_commit = {"com": com, "match_capture_dir": capture_dir, "rank": 0}
        return ext, src

    ext, _src = worker(str(tmp_path))
    want = {"armed": True, "file": str(tmp_path / OW.schedule_file(0)), "calls": 4}
    assert [ext.tp2_commit_install()["schedule_replay"] for _ in range(2)] == [want, want]
    assert events == [("arm", 4), ("install", True), ("install", True)]
    events.clear()
    ext, src = worker(None)
    assert ext.tp2_commit_install()["schedule_replay"] is None and src.replay is None and events == [("install", False)]
