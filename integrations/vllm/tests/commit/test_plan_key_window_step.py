"""A served window's step stages what its plan key predicts when a source commits by engine step.

A served window restarts its engine steps at 0, and the int8 source stages a linear's Int8RowQuant_v1 (x_q, s_a) only at the
engine steps its manifest names (a bench manifest's first 32). Keyed by the geometry alone, a lone request whose batch-mates had
finished met one key at a window step inside those steps in one window and past them in the next: the second step followed a
plan of the quantizers' bytes and staged none, which bounded staging fails closed on (a hit step keeps no host copy to re-hash
from), and the plan it relearned failed the first window's step when its schedule came again. The plan key now binds the names
the step-keyed sources commit at the step (`NativeCollectCommitter.step_targets`).

The committer runs on the CPU as in `test_native_collect_plan_cache`, one commitment run per window, with the real int8 source
quantizing one linear's rows each step. A step whose key has a plan must stage that plan's bytes, compared as the bounded worker
compares them (padded to the chunk).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from verity_vllm.acquire.sources import int8_source as I8
from verity_vllm.commit.committer import native_host as nh
from verity_vllm.engine import int8_ops

from tests.commit import test_native_collect_plan_cache as PC

LIN = "model.layers.0.mlp.down_proj"
K = 512                                         # a lone request's x_q outgrows the chunk padding the worker compares at
SITES = 4                                       # the manifest's engine steps with an Int8RowQuant_v1 site: 0..3

#: per window, per step, the scheduled requests (id, KV length, new tokens). The first window's pair prefills at step 0 and b
#: finishes after step 1, so a decodes alone at step 2. The second window's pair arrives two steps in, behind a request that
#: finishes first, so d finishes after step 3 and c decodes alone at step 4, past the sites, at a's geometry of step 2. The
#: third window is the first's schedule again.
FIRST = [[("a", 0, 8), ("b", 0, 8)], [("a", 8, 1), ("b", 8, 1)], [("a", 9, 1)], [("a", 10, 1)]]
SECOND = [[("e", 0, 5)], [("e", 5, 1)], [("c", 0, 8), ("d", 0, 8)], [("c", 8, 1), ("d", 8, 1)], [("c", 9, 1)]]
WINDOWS = [FIRST, SECOND, FIRST]


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8, K, dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.full((8, 1), 1e-3), requires_grad=False)

    def forward(self, x):
        return int8_ops.linear(x, self.weight, self.weight_scale)[2]


def _model() -> torch.nn.Module:
    m = torch.nn.Module()
    m.model = torch.nn.Module()
    m.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    m.model.layers[0].mlp = torch.nn.Module()
    m.model.layers[0].mlp.down_proj = _Linear()
    return m


def _manifest() -> dict:
    """A bench manifest's Int8RowQuant_v1 call boundaries of the linear: one per engine step, its fx node counting on."""
    return {"ranks": 1, "identities": [{"family": "call_boundaries", "op_path": f"{LIN}/row_quant_{e}", "output_member": m, "step": e,
                                        "engine_step": e, "spec": I8.QUANT_SPEC, "request_id": "r0"}
                                       for e in range(SITES) for m in ("0", "1")]}


class _Runner(PC._Runner):
    """The model step: the linear over the step's rows (its row_quant acquired by the int8 source at a site), then the logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def execute_model(self, scheduler_output, *a, **kw):
        n = scheduler_output.total_num_scheduled_tokens
        self.model.model.layers[0].mlp.down_proj((torch.arange(n * K) % 251 - 125).reshape(n, K).bfloat16())
        self.committer._acquire("logits", torch.zeros(n, 40), self.committer._cur)


def _sched(reqs: list[tuple[str, int, int]]) -> SimpleNamespace:
    cached = [(r, c) for r, c, _ in reqs if c]
    return SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id=r, num_computed_tokens=0) for r, c, _ in reqs if not c],
                           scheduled_cached_reqs=SimpleNamespace(req_ids=[r for r, _ in cached], num_computed_tokens=[c for _, c in cached]),
                           num_scheduled_tokens={r: n for r, _, n in reqs}, total_num_scheduled_tokens=sum(n for _, _, n in reqs))


def _committer(monkeypatch):
    monkeypatch.setattr(PC, "DTYPES", (*PC.DTYPES, torch.int8))
    com = PC.committer(monkeypatch)
    com.runner = _Runner(_model())
    com.runner.committer = com
    I8.attach(com, True, _manifest(), (), model=com.runner.model)
    return com


def _probe(com) -> list[list[int]]:
    """Per step, [the bytes of the plan its key has (-1: none, a learning step), the bytes it staged]."""
    nc, out = com._nc, []
    begin, end = nc.begin_step, nc.end_step

    def begin_step(key, *a):
        out.append([nc.plan_used(key)])
        return begin(key, *a)

    def end_step():
        r = end()
        out[-1].append(nc.byte_cursor)
        return r

    nc.begin_step, nc.end_step = begin_step, end_step
    return out


def _serve(com, windows) -> list[tuple[int, int, int, int]]:
    """Each window as one commitment run; the (window, step, plan bytes, staged bytes) of every hit step that left its plan."""
    steps = _probe(com)
    at = []
    for w, sched in enumerate(windows):
        com.install()
        for s, reqs in enumerate(sched):
            com.runner.input_batch = SimpleNamespace(req_ids=[r for r, _, _ in reqs], num_reqs=len(reqs), num_scheduled_tokens=None)
            com.runner.execute_model(_sched(reqs))
            com._end_step()
            com._commit_step(com._q.get_nowait())
            at.append((w, s))
        com.finalize()
        com.uninstall()
        com.reset()
    pad = nh.GPU_CHUNK_BYTES
    return [(w, s, plan, used) for (w, s), (plan, used) in zip(at, steps) if plan >= 0 and -(-plan // pad) != -(-used // pad)]


def test_a_geometry_key_alone_hits_the_plan_of_another_window_step_after_a_request_finishes(monkeypatch):
    """The key as it was: the second window's lone request, past the sites, hits the plan the first window's lone request learned
    with the quantizers' bytes, and the plan it relearns is then missed by the first window's schedule run again."""
    com = _committer(monkeypatch)
    com.step_targets = lambda step: b""
    q = K + 4                                                 # x_q and s_a of one row
    assert _serve(com, WINDOWS) == [(1, 4, q + 160, 160), (2, 2, 160, q + 160)]


def test_every_hit_step_stages_its_plan_when_the_key_binds_the_step_keyed_sources(monkeypatch):
    com = _committer(monkeypatch)
    assert _serve(com, WINDOWS) == [] and com.errors == []
    src = com.sources[0]
    assert src.stats["quant_tensors"] == 2 * sum(min(len(sched), SITES) for sched in WINDOWS)
    assert len({com.step_targets(e) for e in range(SITES)}) == SITES and com.step_targets(SITES) == b""   # the fx nodes count on


def test_a_steps_digest_is_kept_only_while_the_sources_are_the_same(monkeypatch):
    com = _committer(monkeypatch)
    first, other = com.step_targets(0), com.step_targets(1)
    com.attach_source(SimpleNamespace(targets_at=lambda step: {"logits_processor/div": 1} if step == 0 else {}))
    assert com.step_targets(0) not in (first, b"") and com.step_targets(1) == other
