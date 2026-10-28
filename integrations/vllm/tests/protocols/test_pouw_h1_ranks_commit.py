"""`protocol_options.pouw_h1_ranks` with ``served`` and ``commit`` (ROOTS1 across tensor-parallel ranks): the
Commit runs on every rank, each rank's tap ships what it acquired, and the holder's one `pouw_h1_served.Served` and
`pouw_h1_commit.CommitTap` register each step's A table of K over both ranks' calls and its AROW over both ranks' sites, so
core's ROOTS1 steps (`pouw_h1_gpu.core_verdict`, the GPU run's check) tie the window. One rank's call handed other codes
than its Commit acquired is refused at ``arow``, and a rank whose input ids aren't the first rank's leaves the window
untied. The ranks are stand-ins: each worker an `acquire.h1_commit.RankCommit` over two linears on the int8 arm with z/y records
(`test_pouw_h1_commit.Arm`), or a linear and a MoE layer's routed experts (`MoEModel`), its Commit `acquire.h1_commit.attach`'s
on the rows-sha512 committer's face (`test_pouw_h1_commit.Committer`), each step's ids at its step-input port. On the MoE each rank's expert sites are renamed to the window's names of them
(``…experts.0.down@tp1``), so each step's AROW of the experts' K spans both ranks' expert calls."""
from __future__ import annotations

import hashlib
import importlib
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
# ruff: noqa: E402  (the imports below need torch, so they follow its importorskip)
from torch import nn

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw.coins import Challenger
from verity_vllm.acquire import h1_commit as HC
from verity_vllm.engine import int8_experts as X
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import commit as CT
from verity_vllm.protocols.pouw.h1 import hooks as PHK
from verity_vllm.protocols.pouw.h1 import ranks as PHR
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw.h1 import served as SV

from tests.protocols import pouw_h1_gpu as G
from tests.protocols import pouw_h1_ranks_gpu as RG
from tests.protocols.test_pouw_h1_commit import (
    EXPERTS, FIRST, I, K, MOE, PROMPTS, Arm, Committer, Linear, MoE, _ids, vllm_config)
from tests.protocols.test_pouw_h1_ranks import Executor, _rounds, _so
from tests.protocols.test_pouw_h1_served import _unit

ROWS, STEPS = 32, 3
KEY = hashlib.sha512(b"key").digest()[:32]
#: step 0 admits a and b, step 1 decodes one row of each, step 2 is past the last served one
SCHEDULE = (_so(*((q, p, 4) for q, p in PROMPTS.items())), _so(running={"a": 1, "b": 1}))


class Model(nn.Module):
    """One rank's shards: two linears on their arms, ``edit`` theirs (counting the rank's calls)."""

    def __init__(self, seed, edit=None):
        super().__init__()
        calls = [0]
        self.model = nn.Module()
        self.model.embed_tokens = nn.Module()
        self.model.layers = nn.ModuleList([Linear(64, seed, Arm(edit, calls)), Linear(32, seed + 1, Arm(edit, calls))])


class MoEModel(nn.Module):
    """One rank's shards of a dense linear and a MoE layer of `EXPERTS` experts (`test_pouw_h1_commit.MoE`, top-1 routed
    by its ``ROUTES``, as every rank routes): its H1 linears the linear then each expert's gate_up and down, 7 a rank."""

    def __init__(self, seed, edit=None):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Module()
        self.model.layers = nn.ModuleList([Linear(64, seed, Arm(edit)), MoE(seed + 1)])


class Rank(HC.RankCommit):
    """A worker of a TP engine: its (pp, tp) rank and its model's two shards, named as vLLM names them."""

    def __init__(self, tp, edit=None, moe=False):
        self.rank, self.model = (0, tp), (MoEModel if moe else Model)(10 * tp + 1, edit)
        self.layers = PHK.layers(self.model, (Linear,))
        self.model_runner, self.vllm_config = SimpleNamespace(decode_query_len=1), vllm_config()

    def _h1rank_layers(self):
        return self.model, self.layers

    def _h1rank_rank(self):
        return self.rank


def _window(monkeypatch, edits=(None, None), ids=lambda tp, s, x: x, served=None, moe=False, negative=None):
    """A served window at TP 2 with the Commit on each rank: the holder and its ranks. ``edits`` makes each rank's codes H1
    is handed of what its arm quantized; ``ids`` the input ids rank tp's Commit acquires at step s of the honest ones;
    ``moe`` serves `MoEModel`'s ranks; ``negative`` (a `pouw_h1_ranks_gpu.KINDS` kind) is the GPU harness's edit on rank
    1 (`pouw_h1_ranks_gpu._w_edit`), the site it zeroed in ``holder.edited``."""
    monkeypatch.setattr(HC, "committer", lambda engine, run_id: Committer(engine))
    workers = [Rank(1, edits[1], moe), Rank(0, edits[0], moe)]
    executor = Executor(workers)
    served = served or SV.Served(PH.TILE, _unit(), window=hashlib.sha512(b"w").digest())
    holder = PHR.install(None, executor=executor, rows=ROWS, steps=STEPS, coins=Challenger(0, random.Random(0).randbytes),
                         rounds=_rounds(), key=KEY, entry="t", admission=RQ.ROUND0, served=served,
                         commit={"cp": {"repo": "r", "revision": "v"}, "run_id": "t"})
    if negative is not None:
        executor.collective_rpc(RG._w_edit, kwargs={"kind": negative, "tp": 1})
    run, g = holder.run, torch.Generator().manual_seed(9)
    for s, so in enumerate(SCHEDULE):
        executor.execute_model(so)
        honest, xs = _ids(run, s), [(torch.randn(len(run.owners[s]), K, generator=g) * 3).bfloat16() for _ in range(2)]
        for w in sorted(workers, key=lambda w: w.rank):
            com = w._h1rank["commit"][0]
            src, port = com.sources
            com._cur = SimpleNamespace(step=FIRST + s)
            src.on_step_begin()
            got = ids(w.rank[1], s, honest)
            com._plan_shape = (len(got), len(set(run.owners[s])))
            port._pre(None, (torch.tensor(got), torch.arange(len(got))), {})
            for layer, x in zip(w.model.model.layers, xs):
                layer(x)
            com._cur = None
    if negative is not None:
        holder.edited = [x for x in executor.collective_rpc(RG._w_unedit) if x is not None]
    holder.finish()
    return holder, workers


def _verdict(holder):
    run = holder.run
    return G.core_verdict({"commitment": dict(run.commits), "calls": run.manifest(),
                           "descriptor": SV.descriptor(run.served.program).hex()})


def _reasons(run):
    """The served calls' refusals were the window's coins the auditor host's: the in-process coins' refusal comes first
    (`pouw_h1.H1Run.gated`), and the stand-in's coins are in-process."""
    run.coin_source = PH.CHALLENGER
    return {g.get("reason") for g in run.gated().values()}


def test_each_steps_a_table_spans_both_ranks_and_is_the_commits_arow(monkeypatch):
    """The honest window: every call of both ranks is in its step's A table of K = 64, rank 0's calls then rank 1's, and
    the Commit's AROW of each (step, K) is the rows its ranks' Commits acquired at those sites, the window's own tree; each
    round's input ids are the first rank's; core's ROOTS1 steps tie the window, and its calls are refused for their in-process coins only."""
    holder, workers = _window(monkeypatch)
    run, tap = holder.run, holder.run.served.tap
    assert run.static() and run.modules == {0: "model.layers.0@tp0", 1: "model.layers.1@tp0", 2: "model.layers.0@tp1",
                                            3: "model.layers.1@tp1"}
    assert run.details["roots1"] == {"tied": True, "reason": None, "values": 2 * STEPS, "own_trees": 0,
                                     "steps": {"0": FIRST, "1": FIRST + 1}}
    for s in range(2):
        assert tap.where[CR.commit_rows(s, K)] == {
            "round": s, "calls": [4 * s + w for w in range(4)], "steps": [FIRST + s] * 4,
            "sites": [CT.site(run.modules[w]) for w in range(4)], "rows": [len(run.owners[s])] * 4}
    assert tap.round_ids[0] == [8, 9, 5, 6, 7] and not tap.refused
    core = _verdict(holder)
    assert core["tied"] and core["commit_values"] == 2 * STEPS, core
    assert {g["reason"] for g in run.gated().values()} == {PH.IN_PROCESS} and _reasons(run) == {None}
    assert sorted(holder.committed) == [0, 2] and all(w._h1rank is None for w in workers)
    assert all((c["commit_scheme"], c["run_root"]) == (CT.SCHEME, "00" * CT.ROOT_BYTES) for c in holder.committed.values())
    assert all(w.model.model.layers[0].quant_method.observe.__func__ is Arm.observe for w in workers)


def _zero_first(i, xq):
    return torch.zeros_like(xq) if i == 0 else xq


def test_a_zero_on_one_ranks_call_is_refused_at_arow(monkeypatch):
    """Rank 1's first call handed all-zero codes, with every rank's Commit honest: the window's A table of step 0 and
    K = 64 holds the zero rows where its AROW holds what rank 1's Commit acquired, so the Commit registers a tree of its
    own, the serving process credits nothing (`pouw_h1_commit.UNTIED`) and core refuses the window at ``arow``."""
    holder, _ = _window(monkeypatch, edits=(None, _zero_first))
    run = holder.run
    assert run.details["roots1"]["reason"].startswith("roots1: round 0's A table h1/a/64@0 of calls [0, 1, 2, 3] is not "
                                                      "the Commit's row-table/0/k64")
    assert run.served.tap.own == 1 and _reasons(run) == {CT.UNTIED}
    v = G.negative_verdict({"core": _verdict(holder), "roots1": run.details["roots1"], "same_tokens": True})
    assert v["refused"] and v["arow"][1].startswith("round 0's A table of K = 64 h1/a/64@0 is not the Commit's "), v


def test_a_ranks_input_ids_off_the_first_ranks_leave_the_window_untied(monkeypatch):
    """Rank 1's Commit acquiring other input ids at step 1 than rank 0's: the window's ids are the first rank's, and the
    tap refuses, so the serving process credits nothing."""
    holder, _ = _window(monkeypatch, ids=lambda tp, s, x: [*x[:1], 99, *x[2:]] if (tp, s) == (1, 1) else x)
    reason = holder.run.details["roots1"]["reason"]
    assert reason == f"roots1: the Commit's tap refused: engine step {FIRST + 1}: a rank's input ids are not the first rank's"
    assert _reasons(holder.run) == {CT.UNTIED}


def test_install_refuses_a_periods_weights_across_ranks_and_a_commit_without_served(monkeypatch):
    """At TP > 1 the window registers its own ``b/<K>``: the period's `Weights` (or a registry) are refused, and so is the
    Commit without ``served``, whose tables its values tie to."""
    executor = Executor([Rank(0), Rank(1)])
    kw = {"executor": executor, "rows": ROWS, "steps": STEPS, "coins": Challenger(0, random.Random(0).randbytes),
          "rounds": _rounds(), "key": KEY, "entry": "t"}
    with pytest.raises(ValueError, match="registers its own b/<K>"):
        PHR.install(None, served=SV.Served(weights=SimpleNamespace()), **kw)
    with pytest.raises(ValueError, match="it needs served"):
        PHR.install(None, commit={}, **kw)


def test_the_commit_extension_names_the_rank_with_the_commit():
    """`pouw_h1_ranks.COMMIT_EXTENSION`, the string vLLM's workers import as ``worker_extension_cls``, is
    `acquire.h1_commit.RankCommit`, which ``protocol_options`` can't import (``acquire`` imports it)."""
    module, _, name = PHR.COMMIT_EXTENSION.rpartition(".")
    assert getattr(importlib.import_module(module), name) is HC.RankCommit
    assert issubclass(HC.RankCommit, PHR.RankExtension) and not hasattr(PHR.RankExtension, "h1rank_commit")


def _rank_experts(projection, tp):
    return [f"{MOE}.{e}.{projection}@tp{tp}" for e in range(EXPERTS)]


def test_a_moes_expert_calls_on_both_ranks_are_in_their_steps_a_tables_and_the_commits_arow(monkeypatch):
    """A MoE at TP 2: each rank's routed expert sites (no modules, `int8_experts.ExpertSite`) are named by their rank in
    the window (``…experts.0.down@tp1``) and the rank's tap ships what its Commit acquired at them under those names, so
    each step's AROW of the experts' K = I is both ranks' downs, rank 0's then rank 1's, an idle expert's zero rows
    included; core's ROOTS1 steps tie the window with the expert calls in it. Each rank's release takes its Commit's
    wrapper off an expert's ``apply`` before H1's beneath it, so neither stays on."""
    holder, workers = _window(monkeypatch, moe=True)
    run, tap = holder.run, holder.run.served.tap
    assert [run.modules[w] for w in range(7, 14)] == ["model.layers.0@tp1", *(f"{MOE}.{e}.{p}@tp1" for e in range(EXPERTS)
                                                                             for p in X.PROJECTIONS)]
    assert run.details["roots1"] == {"tied": True, "reason": None, "values": 3 * STEPS, "own_trees": 0,
                                     "steps": {"0": FIRST, "1": FIRST + 1}}
    for s, rows in ((0, [2, 3, 0]), (1, [0, 1, 1])):
        assert tap.where[CR.commit_rows(s, I)] == {
            "round": s, "calls": [14 * s + w for w in (2, 4, 6, 9, 11, 13)], "steps": [FIRST + s] * 6,
            "sites": [CT.site(n) for tp in (0, 1) for n in _rank_experts(X.DOWN, tp)], "rows": rows * 2}
        assert tap.where[CR.commit_rows(s, K)]["calls"] == [14 * s + w for w in (0, 1, 3, 5, 7, 8, 10, 12)]
        assert tap.where[CR.commit_rows(s, K)]["sites"][4:] == [CT.site(n) for n in ["model.layers.0@tp1",
                                                                                     *_rank_experts(X.GATE_UP, 1)]]
    assert not tap.refused and not tap._x
    assert RG.expert_calls(run) == {"calls": 12 * STEPS, "with_rows": 16, "ranks": ["tp0", "tp1"]}
    core = _verdict(holder)
    assert core["tied"] and core["commit_values"] == 3 * STEPS, core
    assert _reasons(run) == {None}
    sites = [m.quant_method for w in workers for m in w.layers if isinstance(m, X.ExpertSite)]
    assert len(sites) == 12 and not any({"apply", "observe", "zy_rows"} & set(vars(q)) for q in sites)


def test_a_zero_on_one_expert_call_on_rank_1_is_refused_at_arow(monkeypatch):
    """The GPU harness's ``a-zero-expert`` on rank 1 (`pouw_h1_ranks_gpu._w_edit`): H1 handed all-zero codes at rank 1's
    first routed expert down call with rows (expert 0's, step 0, call 9), every Commit honest. The window's A table of
    step 0 and K = I holds the zero rows where its AROW holds what rank 1's Commit acquired at ``…experts.0.down@tp1``,
    so the serving process credits nothing and core refuses the window at ``arow``, as the harness's verdict reads it."""
    holder, _ = _window(monkeypatch, moe=True, negative=RG.EXPERT_NEGATIVE)
    run = holder.run
    assert holder.edited == [{"tp": 1, "at": f"{MOE}.0.down"}]
    assert run.details["roots1"]["reason"].startswith(
        f"roots1: round 0's A table h1/a/{I}@0 of calls [2, 4, 6, 9, 11, 13] is not the Commit's row-table/0/k{I} ")
    assert run.served.tap.own == 1 and _reasons(run) == {CT.UNTIED}
    v = G.negative_verdict({"core": _verdict(holder), "roots1": run.details["roots1"], "edited": holder.edited[0]["at"]})
    assert v["refused"] and v["arow"][1].startswith(f"round 0's A table of K = {I} h1/a/{I}@0 is not the Commit's "), v


def test_a_rank_whose_commit_acquires_a_site_the_window_cannot_name_is_refused(monkeypatch):
    """A rank's expert sites the window has no name for would reach the holder's tap under their rank-local names, which
    no call reads and every rank shares: `acquire.h1_commit.RankCommit.h1rank_commit` refuses the rank's Commit."""
    named = pouw._named
    monkeypatch.setattr(pouw, "_named", lambda model, layers: {
        w: n for w, n in named(model, layers).items() if not isinstance(layers[w], X.ExpertSite)})
    with pytest.raises(ValueError, match=r"acquires 6 sites the window has no name for, \['model.layers.1.experts.0.down'"):
        _window(monkeypatch, moe=True)
