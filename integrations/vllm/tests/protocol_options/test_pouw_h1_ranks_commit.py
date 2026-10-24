"""`protocol_options.pouw_h1_ranks` with ``served`` and ``commit`` (ROOTS1 across tensor-parallel ranks, VLLM-81 item 2): the
Commit runs on every rank, each rank's tap ships what it acquired, and the holder's one `pouw_h1_served.Served` and
`pouw_h1_commit.CommitTap` register each step's A table of K over both ranks' calls and its AROW over both ranks' sites, so
core's ROOTS1 steps (`pouw_h1_gpu.core_verdict`, the GPU run's check) tie the window. One rank's call handed other codes
than its Commit acquired is refused at ``arow``, and a rank whose input ids aren't the first rank's leaves the window
untied. The ranks are stand-ins: each worker a `RankExtension` over two linears on the int8 arm with z/y records
(`test_pouw_h1_commit.Arm`), its committer the int8 source's face (`acquire.h1_commit.source`)."""
from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity_vllm.acquire import h1_commit as HC  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_commit as CT  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_ranks as PHR  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_requests as RQ  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_served as SV  # noqa: E402

from tests.protocol_options import pouw_h1_gpu as G  # noqa: E402
from tests.protocol_options.test_pouw_h1_commit import EMBED, FIRST, K, PROMPTS, Arm, Linear, _ids  # noqa: E402
from tests.protocol_options.test_pouw_h1_ranks import Executor, _rounds, _so  # noqa: E402
from tests.protocol_options.test_pouw_h1_served import _unit  # noqa: E402

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


class Committer:
    """The committer's face the int8 source and the tap use, and what `RankExtension.h1rank_release` finalizes."""

    name = "stand-in"

    def __init__(self, engine):
        self.engine, self._cur, self.errors, self.src, self.closed = engine, None, [], None, False

    def _acquire(self, name, t, job):
        pass

    def finalize(self):
        return SimpleNamespace(run_root=bytes(64), extra={"expected_tensors": 0, "committed_tensors": 0, "errors_n": 0})

    def close(self):
        self.src.uninstall()
        self.closed = True


def _attach(engine, layers, tap=None, **committer):
    """`acquire.h1_commit.attach` on the stand-in committer: the int8 source at each linear's site and the rank's tap."""
    com = Committer(engine)
    com.src = HC.source(com, engine.model, layers)
    tap.embed = EMBED
    tap.tap(com)
    com.src.install()
    return com, com.src, tap


class Rank(PHR.RankExtension):
    """A worker of a TP engine: its (pp, tp) rank and its model's two shards, named as vLLM names them."""

    def __init__(self, tp, edit=None):
        self.rank, self.model = (0, tp), Model(10 * tp + 1, edit)
        self.layers = list(self.model.model.layers)
        self.model_runner, self.vllm_config = SimpleNamespace(decode_query_len=1), None

    def _h1rank_layers(self):
        return self.model, self.layers

    def _h1rank_rank(self):
        return self.rank


def _window(monkeypatch, edits=(None, None), ids=lambda tp, s, x: x, served=None):
    """A served window at TP 2 with the Commit on each rank: the holder and its ranks. ``edits`` makes each rank's codes H1
    is handed of what its arm quantized; ``ids`` the input ids rank tp's Commit acquires at step s of the honest ones."""
    monkeypatch.setattr(HC, "attach", _attach)
    workers = [Rank(1, edits[1]), Rank(0, edits[0])]
    executor = Executor(workers)
    served = served or SV.Served(PH.TILE, _unit(), window=hashlib.sha512(b"w").digest())
    holder = PHR.install(None, executor=executor, rows=ROWS, steps=STEPS, coins=Challenger(0, random.Random(0).randbytes),
                         rounds=_rounds(), key=KEY, entry="t", admission=RQ.ROUND0, served=served,
                         commit={"cp": {"repo": "r", "revision": "v"}, "run_id": "t"})
    run, g = holder.run, torch.Generator().manual_seed(9)
    for s, so in enumerate(SCHEDULE):
        executor.execute_model(so)
        honest, xs = _ids(run, s), [(torch.randn(len(run.owners[s]), K, generator=g) * 3).bfloat16() for _ in range(2)]
        for w in sorted(workers, key=lambda w: w.rank):
            com = w._h1rank["commit"][0]
            com._cur = SimpleNamespace(step=FIRST + s)
            com.src.on_step_begin()
            com._acquire(f"{EMBED}/in", torch.tensor(ids(w.rank[1], s, honest)), com._cur)
            for layer, x in zip(w.layers, xs):
                layer(x)
            com._cur = None
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
