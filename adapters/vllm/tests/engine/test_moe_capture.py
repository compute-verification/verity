"""`engine.moe_capture` on stand-ins for vLLM's V2 model runner, its MoE modules and their launch sites, on CPU torch
with made-up values.

At each step that schedules a target, each named module's router ids and the operands and outputs of its two launches
are kept for the target's rows alone, copied right after each launch (the stand-in block reuses its workspace and
clears every tensor when it returns, as the next layer would), and `moe_capture.read` gives them back from the record
`write` wrote. A step that launches at no Python site, experts without ``A_scale``, a step laid out in another order
than the one read, draft tokens, a rank whose pieces are not the trace's and a changed file are refused. The wraps are
patches of `verity_vllm.engine.hooks`, and `moe_capture.uninstall` takes them off."""
from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests import fake_vllm as stand_in
from verity_vllm.engine import hooks
from verity_vllm.engine import moe_capture as MC

E, TOPK, H, MID, G = 6, 2, 8, 4, 4
TILE13 = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
TILE2 = {**TILE13, "BLOCK_SIZE_N": 128}
MODULES = ("model.layers.0.mlp.experts", "model.layers.1.mlp.experts")
RUNNER_MODULE = "tests.engine.stand_in_model_runner"
#: the replay's step trace: rows (request, first position, tokens) in the runner's order. Request ``a`` is the target:
#: two prefill chunks and a decode, at other rows each step; step 3 holds no row of it
STEPS: list[dict[str, Any]] = [
    {"step": 0, "rows": [["b", 0, 2], ["a", 0, 3]]},
    {"step": 1, "rows": [["b", 2, 1], ["a", 3, 2]]},
    {"step": 2, "rows": [["a", 5, 1], ["b", 3, 1]]},
    {"step": 3, "rows": [["b", 4, 1]]}]
PIECES = [[0, 3], [3, 2], [5, 1]]
#: the rows of ``a`` in each of its steps' launches, and the rows the stand-in runner pads a step's launches with
ROWS = [(2, 5), (1, 3), (0, 1)]
PAD = [0, 2, 0, 0]


def batch_order(tokens: dict[str, int], drafts: dict[str, list[int]], decode_query_len: int) -> list[str]:
    """The V2 runner's: decodes first, then by rows."""
    return sorted(tokens, key=lambda r: (tokens[r] != decode_query_len, tokens[r]))


def scheduled(step: dict[str, Any], drafts: dict[str, list[int]] | None = None) -> SimpleNamespace:
    """A step's `SchedulerOutput`: a request at position 0 is new, any other cached."""
    rows = step["rows"]
    new = [SimpleNamespace(req_id=r, num_computed_tokens=0) for r, start, _ in rows if start == 0]
    cached = [(r, start) for r, start, _ in rows if start]
    return SimpleNamespace(num_scheduled_tokens={r: n for r, _, n in rows}, total_num_scheduled_tokens=sum(n for *_, n in rows),
                           scheduled_new_reqs=new, scheduled_spec_decode_tokens=drafts or {},
                           scheduled_cached_reqs=SimpleNamespace(req_ids=[r for r, _ in cached],
                                                                 num_computed_tokens=[s for _, s in cached]))


class Vllm:
    """The stand-in's launch sites: the launcher at both of its modules and the grouped router, each writing made-up
    values from one seeded generator."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.gen = torch.Generator().manual_seed(20)
        fake = stand_in.install(monkeypatch, extra=[":".join(MC.ROUTER_SITE)])
        self.fm, self.tm, self.router = fake[MC.FUSED_MOE_SITE[0]], fake[MC.TRITON_MOE_SITE[0]], fake[MC.ROUTER_SITE[0]]
        self.fm.invoke_fused_moe_triton_kernel = self.tm.invoke_fused_moe_triton_kernel = self.kernel
        fake.set(":".join(MC.ROUTER_SITE), self.grouped_topk)
        mod = types.ModuleType(RUNNER_MODULE)
        setattr(mod, "sort_batch_req_ids", batch_order)
        monkeypatch.setitem(sys.modules, RUNNER_MODULE, mod)
        self.runner_module = mod

    def bits(self, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        """Made-up values of ``dtype``: every bit pattern of its words."""
        return torch.randint(0, 256, (*shape, torch.empty((), dtype=dtype).element_size()), dtype=torch.uint8,
                             generator=self.gen).view(dtype).reshape(shape)

    def kernel(self, A: Any, B: Any, C: Any, A_scale: Any, B_scale: Any, topk_weights: Any, sorted_token_ids: Any,
               expert_ids: Any, num_tokens_post_padded: Any, mul_routed_weight: bool, top_k: int, config: Any) -> None:
        C.copy_(self.bits(*C.shape, dtype=C.dtype))

    def grouped_topk(self, hidden_states: Any, gating_output: Any, topk: int, **k: Any) -> tuple[torch.Tensor, torch.Tensor]:
        m = gating_output.shape[0]
        return (self.bits(m, topk, dtype=torch.float32),
                torch.randint(0, gating_output.shape[1], (m, topk), dtype=torch.int32, generator=self.gen))


class Experts(torch.nn.Module):
    """A block-FP8 MoE block as the pinned vLLM launches it: the grouped router, the w13 launch, the w2 launch on the
    same workspace. ``truth`` holds a clone of each tensor right after its launch, a step; ``flat`` launches by keyword
    at the functional path's module with pair-major outputs and weights, the default by position at the experts'."""

    def __init__(self, v: Vllm, flat: bool) -> None:
        super().__init__()
        self.v, self.flat = v, flat
        self.truth: list[dict[str, torch.Tensor]] = []
        #: "graph": the kernels run with no Python launch site; "bf16": the launches carry no A_scale
        self.path = "served"

    def forward(self, m: int) -> None:
        v, e4m3 = self.v, torch.float8_e4m3fn
        weights, ids = v.router.grouped_topk(hidden_states=None, gating_output=torch.empty(m, E), topk=TOPK, renormalize=True)
        a1, s1 = v.bits(m, H, dtype=e4m3), v.bits(m, H // G, dtype=torch.float32)
        a2, s2 = v.bits(m * TOPK, MID, dtype=e4m3), v.bits(m * TOPK, MID // G, dtype=torch.float32)
        work = torch.empty(m * TOPK * H, dtype=torch.bfloat16)        # cache1 and cache3 are one buffer: 2 * MID == H
        up = work.view(m, TOPK, 2 * MID)
        down, w = (work.view(m * TOPK, H), weights.reshape(-1)) if self.flat else (work.view(m, TOPK, H), weights)
        if self.path == "graph":
            v.kernel(a1, None, up, s1, None, None, None, None, None, False, TOPK, TILE13)
            v.kernel(a2, None, down, s2, None, w, None, None, None, True, 1, TILE2)
            return
        scales = (None, None) if self.path == "bf16" else (s1, s2)
        if self.flat:
            v.fm.invoke_fused_moe_triton_kernel(a1, None, up, A_scale=scales[0], B_scale=None, topk_weights=None,
                                                sorted_token_ids=None, expert_ids=None, num_tokens_post_padded=None,
                                                mul_routed_weight=False, top_k=TOPK, config=TILE13)
        else:
            v.tm.invoke_fused_moe_triton_kernel(a1, None, up, scales[0], None, None, None, None, None, False, TOPK, TILE13)
        got = {"router_ids": ids.clone(), "in_q": a1.clone(), "in_scale": s1.clone(), "up": up.clone()}
        if self.flat:
            v.fm.invoke_fused_moe_triton_kernel(a2, None, down, A_scale=scales[1], B_scale=None, topk_weights=w,
                                                sorted_token_ids=None, expert_ids=None, num_tokens_post_padded=None,
                                                mul_routed_weight=True, top_k=1, config=TILE2)
        else:
            v.tm.invoke_fused_moe_triton_kernel(a2, None, down, scales[1], None, w, None, None, None, True, 1, TILE2)
        self.truth.append({**got, "act_q": a2.clone(), "act_scale": s2.clone(), "topk_weights": weights.clone(),
                           "down": work.view(m, TOPK, H).clone()})
        for t in (ids, weights, a1, s1, a2, s2, work):
            t.view(torch.uint8).zero_()


class Model(torch.nn.Module):
    def __init__(self, v: Vllm) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for flat in (False, True):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = Experts(v, flat)
            self.model.layers.append(layer)

    def forward(self, m: int) -> None:
        for layer in self.model.layers:
            layer.mlp.experts(m)


class Runner:
    """The V2 runner as the capture reads it: its module's batch order, its input batch after the forward."""

    decode_query_len = 1

    def __init__(self, v: Vllm) -> None:
        self.model, self.order, self.steps = Model(v), batch_order, 0

    def get_model(self) -> Model:
        return self.model

    def execute_model(self, so: SimpleNamespace, dummy_run: bool = False) -> None:
        tokens = so.num_scheduled_tokens
        ids = self.order(tokens, {}, self.decode_query_len)
        start = {r.req_id: r.num_computed_tokens for r in so.scheduled_new_reqs}
        start.update(zip(so.scheduled_cached_reqs.req_ids, so.scheduled_cached_reqs.num_computed_tokens))
        self.execute_model_state = SimpleNamespace(input_batch=SimpleNamespace(
            req_ids=[*ids, "pad"], num_reqs=len(ids), query_start_loc_np=np.cumsum([0, *(tokens[r] for r in ids), 9]),
            num_computed_tokens_np=np.array([start[r] for r in ids] + [0])))
        self.model(sum(tokens.values()) + PAD[self.steps % len(PAD)])
        self.steps += 1


Runner.__module__ = RUNNER_MODULE


class Workers:
    """A test's workers; the captures left on them come off with the test, newest first."""

    def __init__(self, v: Vllm) -> None:
        self.v = v
        self.made: list[SimpleNamespace] = []

    def __call__(self, rank: int = 0) -> SimpleNamespace:
        self.made.append(SimpleNamespace(rank=rank, model_runner=Runner(self.v)))
        return self.made[-1]


@pytest.fixture
def v(monkeypatch: pytest.MonkeyPatch) -> Vllm:
    return Vllm(monkeypatch)


@pytest.fixture
def workers(v: Vllm) -> Iterator[Workers]:
    before = hooks.live()
    made = Workers(v)
    yield made
    for w in reversed(made.made):
        if id(w.model_runner) in MC._ON:
            MC.uninstall(w)
    assert hooks.live() == before and not MC._ON


def experts(worker: SimpleNamespace, name: str) -> Experts:
    return dict(worker.model_runner.get_model().named_modules())[name]


def run(worker: SimpleNamespace, steps: list[dict[str, Any]] = STEPS) -> None:
    for s in steps:
        worker.model_runner.execute_model(scheduled(s))


def words(t: torch.Tensor) -> np.ndarray:
    """A tensor's words as the capture keeps them, by another road: its bytes, reread at its element size."""
    return np.frombuffer(t.contiguous().view(torch.uint8).numpy().tobytes(), dtype=f"<u{t.element_size()}").reshape(t.shape)


def expected(truth: list[dict[str, torch.Tensor]], field: str, shape: tuple[int, ...]) -> np.ndarray:
    """Request ``a``'s rows of ``field`` over its three steps, a token's of ``shape``, from what the stand-in block
    launched (a pair-major tensor's rows by token)."""
    return np.concatenate([words(truth[s][field]).reshape(-1, *shape)[lo:hi] for s, (lo, hi) in enumerate(ROWS)])


def test_every_launch_of_the_targets_rows_is_kept_and_read_back(workers: Workers, tmp_path) -> None:
    ranks = [workers(0), workers(1)]
    assert [MC.install(w, ["a", "c"], list(MODULES)) for w in ranks] == [0, 1]
    assert len(hooks.live(MC.TAG)) == 6
    for w in ranks:
        run(w)
    takes = [MC.take(w, str(tmp_path / "ranks")) for w in ranks]
    assert [MC.uninstall(w) for w in reversed(ranks)] == [1, 0] and not hooks.live(MC.TAG)
    spans = MC.spans(takes, STEPS)
    assert list(spans) == ["a"] and spans["a"]["pieces"] == PIECES and spans["a"]["steps"] == [0, 1, 2]
    path = MC.write(tmp_path / "ranks", spans)
    assert path == tmp_path / "ranks" / MC.CAPTURE and sorted(json.loads(path.read_text())["requests"]["a"]["ranks"]) == ["0", "1"]
    shapes = {"in_q": (H,), "in_scale": (H // G,), "router_ids": (TOPK,), "up": (TOPK, 2 * MID), "act_q": (TOPK, MID),
              "act_scale": (TOPK, MID // G), "topk_weights": (TOPK,), "down": (TOPK, H)}
    assert tuple(shapes) == MC.KEPT
    for rank, w in enumerate(ranks):
        for name in MODULES:
            got = MC.read(tmp_path / "ranks", "a", rank, name)
            truth = experts(w, name).truth
            assert len(truth) == len(STEPS)
            for f in MC.KEPT:
                assert got[f].shape == (6, *shapes[f]) and got[f].any()
                assert np.array_equal(got[f], expected(truth, f, shapes[f])), (rank, name, f)
            assert got["pieces"] == PIECES and got["steps"] == [0, 1, 2] and got["tiles"] == [[TILE13, TILE2]] * 3
            assert got["of"] == {"in_q": "float8_e4m3fn", "in_scale": "float32", "router_ids": "int32", "up": "bfloat16",
                                 "act_q": "float8_e4m3fn", "act_scale": "float32", "topk_weights": "float32",
                                 "down": "bfloat16"}
    # a file is the request's fields in `KEPT`'s order, nothing else: 6 tokens of 8 + 8 + 8 + 32 + 8 + 8 + 8 + 32 bytes
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["ranks/capture.json", "ranks/rank0/0/0.bin", "ranks/rank0/0/1.bin", "ranks/rank1/0/0.bin",
                     "ranks/rank1/0/1.bin"]
    assert {(tmp_path / f).stat().st_size for f in files[1:]} == {6 * 112}
    first = MC.read(tmp_path / "ranks", "a", 0, MODULES[0])
    assert (tmp_path / "ranks/rank0/0/0.bin").read_bytes()[:6 * H] == first["in_q"].tobytes()
    assert not np.array_equal(first["down"], MC.read(tmp_path / "ranks", "a", 1, MODULES[0])["down"])


def test_a_step_without_a_target_and_a_dummy_run_keep_nothing(workers: Workers, tmp_path) -> None:
    w = workers()
    MC.install(w, ["a"], [MODULES[0]])
    w.model_runner.execute_model(scheduled(STEPS[3]))
    w.model_runner.execute_model(scheduled(STEPS[0]), dummy_run=True)
    assert MC.take(w, str(tmp_path)) == {"rank": 0, "requests": {}} and not list(tmp_path.rglob("*.bin"))
    assert len(experts(w, MODULES[0]).truth) == 2


def test_a_step_that_launches_at_no_python_site_is_refused(workers: Workers) -> None:
    w = workers()
    MC.install(w, ["a"], list(MODULES))
    experts(w, MODULES[1]).path = "graph"
    with pytest.raises(TypeError, match=r"model\.layers\.1\.mlp\.experts gave no in_q, in_scale, up, act_q, .*CUDA graph"):
        run(w, STEPS[:1])


def test_experts_without_a_scale_are_refused(workers: Workers) -> None:
    w = workers()
    MC.install(w, ["a"], [MODULES[0]])
    experts(w, MODULES[0]).path = "bf16"
    with pytest.raises(TypeError, match="w13 launch carries no A_scale"):
        run(w, STEPS[:1])


def test_a_step_laid_out_in_another_order_than_the_one_read_is_refused(workers: Workers, v: Vllm,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    w = workers()
    MC.install(w, ["a"], [MODULES[0]])
    monkeypatch.setattr(v.runner_module, "sort_batch_req_ids", lambda tokens, drafts, n: sorted(tokens))
    with pytest.raises(ValueError, match=r"laid its requests out at .*\{'a': \(2, 5, 0\)\}.*read its launches at \{'a': \(0, 3, 0\)\}"):
        run(w, STEPS[:1])


def test_draft_tokens_are_refused(workers: Workers) -> None:
    w = workers()
    MC.install(w, ["a"], [MODULES[0]])
    with pytest.raises(TypeError, match="draft tokens"):
        w.model_runner.execute_model(scheduled(STEPS[0], drafts={"a": [7]}))


def test_install_refuses_what_it_cannot_capture(workers: Workers, v: Vllm, monkeypatch: pytest.MonkeyPatch) -> None:
    w = workers()
    with pytest.raises(KeyError, match=r"no module named model\.layers\.9\.mlp\.experts"):
        MC.install(w, ["a"], ["model.layers.9.mlp.experts"])
    with pytest.raises(ValueError, match="no request or no module"):
        MC.install(w, [], [MODULES[0]])
    with monkeypatch.context() as m:
        m.delattr(v.runner_module, "sort_batch_req_ids")
        with pytest.raises(TypeError, match="has no sort_batch_req_ids"):
            MC.install(w, ["a"], [MODULES[0]])
    with monkeypatch.context() as m:
        m.delattr(v.router, "grouped_topk")
        with pytest.raises(TypeError, match="grouped_topk_router holds no grouped_topk"):
            MC.install(w, ["a"], [MODULES[0]])
    assert not hooks.live(MC.TAG) and not experts(w, MODULES[0])._forward_hooks
    MC.install(w, ["a"], [MODULES[0]])
    with pytest.raises(ValueError, match="already on this worker's runner"):
        MC.install(w, ["a"], [MODULES[0]])
    MC.uninstall(w)
    with pytest.raises(TypeError, match="no capture is on this worker's runner"):
        MC.take(w, "unused")


def test_a_rank_whose_pieces_are_not_the_traces_is_refused(workers: Workers, tmp_path) -> None:
    w = workers()
    MC.install(w, ["a"], [MODULES[0]])
    run(w)
    take = MC.take(w, str(tmp_path))
    assert MC.spans([take], STEPS)["a"]["ranks"]["0"] == take["requests"]["a"]["modules"]
    with pytest.raises(ValueError, match=r"request a: rank 0 captured .*\[\[0, 3\], \[3, 2\], \[5, 1\]\].*ran it at \[\[0, 3\], \[3, 2\]\]"):
        MC.spans([take], STEPS[:2])
    with pytest.raises(ValueError, match="ran it at"):
        MC.spans([take], [*STEPS[:2], {"step": 2, "rows": [["a", 5, 2]]}])


def test_read_refuses_a_changed_file_and_what_the_record_lacks(workers: Workers, tmp_path) -> None:
    w = workers()
    MC.install(w, ["a"], [MODULES[0]])
    run(w)
    MC.write(tmp_path, MC.spans([MC.take(w, str(tmp_path))], STEPS))
    for request, rank, name, lacks in (("b", 0, MODULES[0], "no request b"), ("a", 1, MODULES[0], "at rank 1"),
                                       ("a", 0, MODULES[1], r"no model\.layers\.1\.mlp\.experts at rank 0")):
        with pytest.raises(KeyError, match=lacks):
            MC.read(tmp_path, request, rank, name)
    record = tmp_path / MC.CAPTURE
    doc = json.loads(record.read_text())
    entry = doc["requests"]["a"]["ranks"]["0"][MODULES[0]]
    entry["fields"]["down"]["shape"] = [TOPK, H + 1]
    record.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="672 bytes, and its fields' shapes say 696"):
        MC.read(tmp_path, "a", 0, MODULES[0])
    del entry["fields"]["down"]
    record.write_text(json.dumps(doc))
    with pytest.raises(KeyError, match="has the fields"):
        MC.read(tmp_path, "a", 0, MODULES[0])
    file = tmp_path / entry["file"]
    file.write_bytes(bytes([file.read_bytes()[0] ^ 1]) + file.read_bytes()[1:])
    with pytest.raises(ValueError, match="not the bytes the record's sha256 names"):
        MC.read(tmp_path, "a", 0, MODULES[0])


def test_launch_arg_reads_a_launch_by_position_and_by_keyword() -> None:
    a = ("scale", None, "weights", None, None, None, False, 8, {"BLOCK_SIZE_M": 16})
    assert [MC.launch_arg(a, {}, n) for n in MC.LAUNCH_ARGS] == ["scale", "weights", False, 8, {"BLOCK_SIZE_M": 16}]
    assert MC.launch_arg(a[:1], {"mul_routed_weight": True}, "mul_routed_weight") is True
    assert MC.launch_arg((), {}, "A_scale") is None and MC.launch_arg(a[:6], {}, "mul_routed_weight") is None
