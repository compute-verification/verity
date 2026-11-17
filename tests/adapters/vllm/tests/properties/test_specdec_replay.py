"""The speculative-decoding stand-in without a GPU (the GPU run is `tests/properties/specdec_replay_gpu.py --out ...` on one
card): its report over records made here, its arms in their order with a failed phase recorded, its serve and replay loop
(`drive`) on a batch queue of two, where every step scheduled must leave its record, and, where vLLM is installed, that loop on
vLLM's own scheduler under vLLM's own engine-core loop with drafts on, the model stood in by a hash: a replay given the
trace's records alone repeats every step and token of the serve."""
from __future__ import annotations

import hashlib
import json
import queue
import random
import sys
import types
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from tests.properties import specdec_replay_gpu as X
from verity_vllm.engine.step_trace import StepTrace


def _step(i: int, rows: list[list[Any]], emitted: dict[str, int], new: tuple[str, ...] = (), hit: dict[str, int] | None = None) -> dict:
    return {"kind": "step", "step": i, "T": sum(r[2] for r in rows), "order": "runner", "rows": rows, "new": list(new),
            "resumed": [], "cache_hit": hit or {}, "preempted": [], "emitted": emitted, "finished": {}}


#: one request under three draft tokens: a prefill from a 4-token hit, a row before any draft, then rows of 1 + 3 tokens that
#: keep 1, 3, 0 and 2 drafts. The row after the first of them was scheduled while it ran, so the trace puts it 2 tokens ahead;
#: so was the last row, which async scheduling ran for a request that had finished: it emits nothing and counts in no total
DRAFTS = {"steps": [_step(0, [["a", 4, 6]], {"a": 1}, new=("a",), hit={"a": 4}), _step(1, [["a", 10, 1]], {"a": 1}),
                    _step(2, [["a", 11, 4]], {"a": 2}), _step(3, [["a", 15, 4]], {"a": 4}), _step(4, [["a", 17, 4]], {"a": 1}),
                    _step(5, [["a", 18, 4]], {"a": 3}), _step(6, [["a", 22, 4]], {})],
          "drafted": {"2": {"a": 3}, "3": {"a": 3}, "4": {"a": 3}, "5": {"a": 3}, "6": {"a": 3}}, "tokens": {"a": [5, 6, 7]}}


def test_the_report_counts_the_drafts_kept_and_where_the_trace_puts_each_row():
    got = X.acceptance(DRAFTS)
    assert (got["rows_with_drafts"], got["draft_tokens"], got["accepted"]) == (4, 12, 6)
    assert got["rows_by_accepted"] == {"1": 1, "3": 1, "0": 1, "2": 1}
    assert got["first_position"] == {"true": 4, "ahead_by_rejected": 2, "other": 0}
    assert (got["tokens_not_one_more_than_drafts"], got["emitted_outside_one_to_all"]) == (0, 0)
    assert got["cache_hits"] == {"a": 4} and got["distinct_tokens"] == 3


def test_the_report_counts_a_row_that_breaks_the_records_reading_of_a_draft():
    bad = {"steps": [_step(0, [["a", 0, 8]], {"a": 1}, new=("a",)), _step(1, [["a", 8, 3]], {"a": 5}), _step(2, [["a", 14, 4]], {"a": 1})],
           "drafted": {"1": {"a": 3}, "2": {"a": 3}}, "tokens": {}}
    got = X.acceptance(bad)
    assert (got["tokens_not_one_more_than_drafts"], got["emitted_outside_one_to_all"]) == (1, 1)
    assert got["first_position"] == {"true": 1, "ahead_by_rejected": 0, "other": 1}


def test_rows_without_drafts_are_all_at_their_true_position():
    plain = {"steps": [_step(0, [["a", 0, 8], ["b", 0, 8]], {}, new=("a", "b")), _step(1, [["a", 8, 2], ["b", 8, 1]], {"a": 1}),
                       _step(2, [["a", 10, 1], ["b", 9, 1]], {"a": 1, "b": 1})], "drafted": {"0": {}, "1": {}, "2": {}}, "tokens": {}}
    got = X.acceptance(plain)
    assert got["first_position"] == {"true": 4, "ahead_by_rejected": 0, "other": 0} and got["rows_with_drafts"] == 0


def test_the_arms_run_in_their_order_and_a_failed_phase_is_recorded_while_the_rest_go_on(tmp_path, monkeypatch, capsys):
    calls = []

    def phase(argv, env, stdout, stderr):
        a = dict(zip(argv[2::2], argv[3::2], strict=True))
        calls.append((a["--weights"], a["--arm"], a["--phase"]))
        if calls[-1] == ("dummy", "mtp1", "serve"):
            stdout.write("boom\n")
            stdout.flush()
            return types.SimpleNamespace(returncode=3)
        doc = {**DRAFTS, "config": {}, "redrawn": None, "sampling": {"a": {"temperature": 0.0}}, "empty_schedules": 0,
               "arrived_after_an_empty_schedule": {}}
        (Path(a["--dir"]) / f"{a['--phase']}.json").write_text(json.dumps({**doc, "drafted": {}} if a["--arm"] == "off" else doc))
        stdout.write("fine\n")
        stdout.flush()
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(X.subprocess, "run", phase)
    monkeypatch.setattr(sys, "argv", ["specdec_replay_gpu.py", "--out", str(tmp_path)])
    assert X.main() == 1
    report = json.loads((tmp_path / "specdec_replay.json").read_text())
    assert [name.replace("-", ":") for name in report["runs"]] == X.RUNS.split(",")
    assert calls[0] == ("scaled", "off", "serve") and ("dummy", "mtp1", "replay") not in calls
    assert report["ok"] is False and report["runs"]["dummy-mtp1"]["serve"] == {"rc": 3, "log_tail": ["boom"]}
    assert report["runs"]["scaled-mtp5"]["replay_equal"]["equal"] is True
    against = {name for name, e in report["runs"].items() if "tokens_against_off" in e}
    assert against == {"scaled-mtp5", "scaled-mtp1", "scaled-mtp3", "scaled-synthetic3", "dummy-synthetic3"}
    assert "FAILED" in capsys.readouterr().out


# ---- the runner's loop on a batch queue of two, with no vLLM ------------------------------------------------------------------

def _hash(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha512(repr(parts).encode()).digest()[:8], "big")


class Queued:
    """What `drive` steps and `StepTrace` wraps, with the batch queue of two that vLLM's engine core keeps under async scheduling
    (``EngineCore.step_with_batch_queue``): a call schedules a step while the one before it is in flight, and processes the
    oldest step's output once the queue is full or nothing is left to schedule. A request is a prompt row, then rows of one
    token and ``drafts`` draft tokens, of which a hash of the request and its length keeps some. The scheduler counts a row in
    flight as one token, as vLLM's does, so a row that keeps a draft can end its request with the next row already scheduled:
    that row runs and emits nothing."""

    PROMPT = 8

    def __init__(self, drafts: int, limits: dict[str, int]) -> None:
        self.drafts, self.limits = drafts, limits
        self.requests: dict[str, Any] = {}
        self.ended = False                      # a request ended since the last schedule, which frees it
        self.queue: deque = deque()
        self.tokens: dict[str, list[int]] = {}

    def add(self, rid: str) -> None:
        self.tokens[rid] = []
        self.add_request(types.SimpleNamespace(request_id=rid, num_prompt_tokens=self.PROMPT, sampling_params=None, computed=0, flying=0))

    def add_request(self, request: Any) -> None:
        self.requests[request.request_id] = request

    def finish_requests(self, request_ids: Any, finished_status: Any) -> list[Any]:
        return []

    def schedule(self) -> Any:
        self.ended = False
        new, cached, rows, drafts = [], [], {}, {}
        for rid, q in self.requests.items():
            if not q.computed:
                new.append(types.SimpleNamespace(req_id=rid, num_computed_tokens=0))
                rows[rid] = q.num_prompt_tokens
            elif len(self.tokens[rid]) + q.flying < self.limits[rid]:
                cached.append((rid, q.computed))
                rows[rid], drafts[rid] = 1 + self.drafts, [-1] * self.drafts
            else:
                continue                        # held back: its rows in flight reach its limit if each emits its one token
            q.computed += rows[rid]
            q.flying += 1
        return types.SimpleNamespace(
            scheduled_new_reqs=new, num_scheduled_tokens=rows, total_num_scheduled_tokens=sum(rows.values()),
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=[r for r, _ in cached], num_computed_tokens=[n for _, n in cached]),
            scheduled_spec_decode_tokens=drafts, preempted_req_ids=set())

    def update_from_output(self, so: Any, mro: Any) -> dict[int, Any]:
        outs = []
        for rid in mro.req_ids:
            q, have = self.requests.get(rid), self.tokens[rid]
            if q is None:
                continue                        # a row of a request that ended at the step before
            q.flying -= 1
            drafts = len(so.scheduled_spec_decode_tokens.get(rid, ()))
            kept = min(_hash(rid, len(have), "kept") % (drafts + 1), self.limits[rid] - len(have) - 1)
            q.computed -= drafts - kept
            new = [1000 + _hash(rid, len(have) + i) % 40000 for i in range(1 + kept)]
            have += new
            ended = len(have) >= self.limits[rid]
            if ended:
                del self.requests[rid]
                self.ended = True
            outs.append(types.SimpleNamespace(request_id=rid, new_token_ids=new, finish_reason=1 if ended else None, stop_reason=None))
        return {0: types.SimpleNamespace(outputs=outs)}

    def has_unfinished_requests(self) -> bool:
        return bool(self.requests)

    def step(self) -> list[Any]:
        if self.requests or self.ended:
            so = self.schedule()
            self.queue.appendleft(so)
            if len(self.queue) < 2 and (so.total_num_scheduled_tokens or self.requests):
                return []
        elif not self.queue:
            return []
        so = self.queue.pop()
        outs = self.update_from_output(so, types.SimpleNamespace(req_ids=list(so.num_scheduled_tokens)))
        return [types.SimpleNamespace(finished=True, request_id=o.request_id, outputs=[types.SimpleNamespace(token_ids=self.tokens[o.request_id])])
                for o in outs[0].outputs if o.finish_reason is not None]


def _queued(eng: Queued, phase: str, due: list[dict[str, Any]]) -> dict[str, Any]:
    trace = StepTrace().install(eng)
    doc = X.drive(eng, trace, phase, due, eng.add)
    trace.uninstall()
    return json.loads(json.dumps({**doc, "records": trace.records, "sampling": dict.fromkeys(eng.limits, {"temperature": 0.0})}))


#: a second request arrives once the first has ended, so the engine is idle with requests still due
TWO = [{"id": "first", "iteration": 0}, {"id": "next", "iteration": 10 ** 6}]


@pytest.mark.parametrize("drafts", [0, 3])
def test_every_step_the_engine_scheduled_has_its_record_when_the_runner_stops(drafts):
    """Under a draft the scheduler schedules a request's next row before the output that ends it is back. vLLM's own loop
    steps while its batch queue holds a step, so that row's output is processed and the trace records the step; a runner that
    stops when no request is unfinished leaves the last such step scheduled and without its record."""
    after_the_end = []
    for limit in range(2, 14):
        limits = {"first": limit, "next": 15 - limit}
        serve = _queued(Queued(drafts, limits), "serve", TWO)
        replay = _queued(Queued(drafts, limits), "replay", X.arrivals(serve["records"]))
        for doc in (serve, replay):
            assert [s["step"] for s in doc["steps"]] == sorted(map(int, doc["drafted"])), (limit, doc["drafted"])
        assert {rid: len(ids) for rid, ids in serve["tokens"].items()} == limits
        assert X.compare(serve, replay)["equal"], limit
        after_the_end.append(not serve["steps"][-1]["emitted"])
    # only under a draft does a serve end on a row scheduled before its request's last output, and not every one does
    assert (any(after_the_end), all(after_the_end)) == (bool(drafts), False), after_the_end


def test_an_engine_that_never_processes_a_scheduled_steps_output_stops_the_runner_by_name():
    """The engine keeps the step in its queue and processes nothing once no request is left. Its scheduler output stays alive:
    the trace keys a step in flight by its scheduler output's ``id``, which a freed one would hand to the next."""
    class Stalled(Queued):
        def step(self) -> list[Any]:
            return super().step() if self.requests else []

    served = [_queued(Queued(3, {"first": n}), "serve", TWO[:1])["steps"] for n in range(2, 14)]
    limit, last = next((n, steps[-1]["step"]) for n, steps in enumerate(served, 2) if not steps[-1]["emitted"])
    with pytest.raises(SystemExit, match=rf"step {last} was scheduled and two engine steps with no request left processed no output"):
        _queued(Stalled(3, {"first": limit}), "serve", TWO[:1])


# ---- vLLM's scheduler under vLLM's engine-core loop, with no model -----------------------------------------------------------

#: a config vLLM builds its model config from; no weight, tokenizer or model code is read
MODEL = {"architectures": ["OPTForCausalLM"], "model_type": "opt", "vocab_size": 50272, "hidden_size": 768, "word_embed_proj_dim": 768,
         "ffn_dim": 3072, "num_attention_heads": 12, "num_hidden_layers": 12, "max_position_embeddings": 8192, "bos_token_id": 2,
         "eos_token_id": 2, "pad_token_id": 1, "do_layer_norm_before": True, "torch_dtype": "float16"}
#: the served engine's settings, as a step trace's header names them
SERVE = {"max_model_len": 2048, "max_num_seqs": 4, "max_num_batched_tokens": 256, "enable_chunked_prefill": True,
         "enable_prefix_caching": True, "async_scheduling": True, "block_size": 16, "num_gpu_blocks": 4096}


def _draft(k: int) -> dict[str, Any]:
    return {"speculative": {"method": "mtp", "num_speculative_tokens": k}} if k else {}


class Runner:
    """The model runner's side of vLLM's executor with no model, from what the scheduler sends it alone, as vLLM's own tests of
    its async scheduler stand it in: its own count of each request's tokens, a sample once a step reaches their end, and under
    a draft as many of the step's draft tokens kept as a hash of the request, its position and the step's size says. A token
    is such a hash too, so a step batched otherwise keeps and emits otherwise."""

    def __init__(self) -> None:
        self.view: dict[str, list[int]] = {}                    # request -> [tokens it holds, tokens computed]
        self.out: Any = None

    def execute_model(self, so: Any, non_block: bool = False) -> Future:
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

        for rid in (*(so.preempted_req_ids or ()), *(so.finished_req_ids or ())):
            self.view.pop(rid, None)
        for r in so.scheduled_new_reqs:
            ids = r.prefill_token_ids if r.prefill_token_ids is not None else r.prompt_token_ids
            self.view[r.req_id] = [len(ids), r.num_computed_tokens]
        done: Future = Future()
        if not so.total_num_scheduled_tokens:
            done.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
            return done
        sampled: list[list[int]]
        size, sampled = so.total_num_scheduled_tokens, []
        for rid, n in so.num_scheduled_tokens.items():
            view = self.view.get(rid)
            if view is None or view[1] + n < view[0]:
                if view is not None:
                    view[1] += n
                sampled.append([])
                continue
            first, drafts = view[1], len(so.scheduled_spec_decode_tokens.get(rid, ()))
            kept = _hash(rid, first, size, "kept") % (drafts + 1)
            sampled.append([1000 + _hash(rid, first + i, size) % 40000 for i in range(1 + kept)])
            view[1] = first + n - (drafts - kept)
            view[0] = view[1] + 1
        ids = list(so.num_scheduled_tokens)
        self.out = ModelRunnerOutput(req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)}, sampled_token_ids=sampled)
        done.set_result(None)
        return done

    def sample_tokens(self, grammar_output: Any, non_block: bool = False) -> Future:
        done: Future = Future()
        done.set_result(self.out)
        return done

    def take_draft_token_ids(self) -> None:
        return None


class Engine:
    """vLLM's scheduler for ``cfg`` (a step trace header's config) under vLLM's engine-core loop, `Runner` its model: what
    `drive` steps. The loop is ``EngineCore.step_with_batch_queue`` and ``post_step`` themselves, on an engine core given only
    what they read.

    The speculative config is built as vLLM's own scheduler tests build one (a method that needs no draft model), then given
    the header's settings before the scheduler is: the scheduler reads its method only through the config's own predicates."""

    def __init__(self, cfg: dict[str, Any], where: Path) -> None:
        import torch
        from vllm.config import (
            CacheConfig,
            DeviceConfig,
            ModelConfig,
            ParallelConfig,
            SchedulerConfig,
            SpeculativeConfig,
            VllmConfig,
        )
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
        )
        from vllm.v1.structured_output import StructuredOutputManager

        where.mkdir(parents=True, exist_ok=True)
        (where / "config.json").write_text(json.dumps(MODEL))
        model = ModelConfig(model=str(where), dtype="float16", seed=0, skip_tokenizer_init=True, max_model_len=cfg["max_model_len"])
        sched = SchedulerConfig(max_num_seqs=cfg["max_num_seqs"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
                                max_model_len=cfg["max_model_len"], enable_chunked_prefill=cfg["enable_chunked_prefill"],
                                long_prefill_token_threshold=cfg.get("long_prefill_token_threshold", 0),
                                async_scheduling=cfg["async_scheduling"], is_encoder_decoder=model.is_encoder_decoder)
        cache = CacheConfig(block_size=cfg["block_size"], cache_dtype=cfg.get("kv_cache_dtype", "auto"),
                            enable_prefix_caching=cfg["enable_prefix_caching"])
        spec = cfg.get("speculative")
        draft = None
        if spec:
            k = spec["num_speculative_tokens"]
            draft = SpeculativeConfig(model="ngram", method="ngram_gpu", num_speculative_tokens=k, prompt_lookup_max=k, prompt_lookup_min=1)
        config = VllmConfig(scheduler_config=sched, model_config=model, cache_config=cache, parallel_config=ParallelConfig(),
                            speculative_config=draft, device_config=DeviceConfig(device="cpu"))
        for name, value in (spec or {}).items():
            setattr(draft, name, value)
        block = cfg["block_size"]
        kv = KVCacheConfig(num_blocks=cfg["num_gpu_blocks"], kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(
            ["layer"], FullAttentionSpec(block_size=block, num_kv_heads=1, head_size=1, dtype=torch.float32))])
        cache.num_gpu_blocks = cfg["num_gpu_blocks"]
        register_all_kvcache_specs(config)
        self.scheduler = sched.get_scheduler_cls()(vllm_config=config, kv_cache_config=kv, block_size=block, log_stats=False,
                                                   structured_output_manager=StructuredOutputManager(config))
        self.scheduler.use_v2_model_runner = True
        hash_fn = get_hash_fn_by_name(cache.prefix_caching_hash_algo)
        init_none_hash(hash_fn)
        self.hasher = get_request_block_hasher(block, hash_fn)
        core = self.core = object.__new__(EngineCore)
        core.vllm_config, core.scheduler, core.model_executor, core.log_stats = config, self.scheduler, Runner(), False
        core.batch_queue_size = config.max_concurrent_batches
        assert core.batch_queue_size == 2, "the loop under test is the one async scheduling runs on one stage"
        core.batch_queue = deque(maxlen=core.batch_queue_size)
        core.is_ec_consumer, core.is_pooling_model, core.aborts_queue = True, False, queue.Queue()
        core.use_spec_decode = core.check_for_draft_tokens = draft is not None
        core.async_scheduling = sched.async_scheduling
        self.open: set[str] = set()
        self.tokens: dict[str, list[int]] = {}

    def add_request(self, rid: str, prompt_ids: list[int], max_tokens: int) -> None:
        from vllm import SamplingParams
        from vllm.v1.request import Request

        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, seed=0, detokenize=False)
        self.scheduler.add_request(Request(request_id=rid, prompt_token_ids=list(prompt_ids), sampling_params=params,
                                           pooling_params=None, arrival_time=0.0, block_hasher=self.hasher))
        self.open.add(rid)
        self.tokens[rid] = []

    def has_unfinished_requests(self) -> bool:
        return bool(self.open)

    def step(self) -> list[Any]:
        outs, model_executed = self.core.step_with_batch_queue()
        self.core.post_step(model_executed=model_executed)
        done = []
        for eco in (outs or {}).values():
            for o in eco.outputs:
                self.tokens[o.request_id] += o.new_token_ids
                if o.finish_reason is not None:
                    self.open.discard(o.request_id)
                    done.append(types.SimpleNamespace(finished=True, request_id=o.request_id,
                                                      outputs=[types.SimpleNamespace(token_ids=self.tokens[o.request_id])]))
        return done


def _run(cfg: dict[str, Any], where: Path, phase: str, due: list[dict[str, Any]], reqs: dict[str, tuple[Any, int]]) -> dict[str, Any]:
    """One engine for ``cfg`` stepped by the stand-in's `drive`. ``reqs``: each request's prompt (its tokens, or a function of
    the tokens served so far) and its token limit. The records are what a trace file would hold: they went through JSON."""
    eng = Engine(cfg, where)
    trace = StepTrace().install(eng.scheduler)

    def add(rid: str) -> None:
        prompt, limit = reqs[rid]
        eng.add_request(rid, prompt(eng.tokens) if callable(prompt) else prompt, limit)

    doc = X.drive(eng, trace, phase, due, add)
    trace.uninstall()
    assert [s["step"] for s in doc["steps"]] == sorted(map(int, doc["drafted"])), "a step the scheduler scheduled has no record"
    return json.loads(json.dumps({**doc, "records": trace.records, "sampling": dict.fromkeys(reqs, {"temperature": 0.0})}))


def _replay(serve: dict[str, Any], where: Path, reqs: dict[str, tuple[Any, int]], early: bool = False) -> dict[str, Any]:
    """The replay of ``serve`` from its records: an engine built from the header, each request put back where its arrival
    says. ``early``: as a replay that reads an arrival's step alone would, with no word of the empty schedule before it."""
    arrive = X.arrivals(serve["records"])
    try:
        replay = _run(serve["records"][0]["config"], where, "replay", [{**a, "after_empty": 0} for a in arrive] if early else arrive, reqs)
    except SystemExit as stuck:
        return {"equal": False, "stuck": str(stuck)}
    return X.compare(serve, replay)


def _load() -> dict[str, tuple[list[int], int]]:
    """Ten requests, short enough that one is often alone at its token limit when the next arrives; every third opens with the
    same 96 tokens, six cache blocks."""
    rng = random.Random(7)
    shared = [rng.randrange(1000, 40000) for _ in range(96)]
    out = {}
    for i in range(10):
        own = [rng.randrange(1000, 40000) for _ in range(rng.choice((8, 40, 90, 200)))]
        out[f"r{i:02d}"] = (shared + own if i % 3 == 2 else own, rng.choice((6, 12, 20)))
    return out


@pytest.mark.parametrize("k", [0, 1, 5])
def test_a_replay_from_the_trace_alone_repeats_vllms_scheduler_under_drafts(k, tmp_path, capsys):
    """Served at every spacing of arrivals from one engine iteration to twelve. Under a draft a row scheduled while the row
    before it ran is recorded ahead of its true position by what that row rejected, and nowhere else."""
    pytest.importorskip("vllm")
    reqs = _load()
    seen: dict[str, Any]
    seen = {"after_empty": 0, "early_differs": 0, "hits": 0, "empty": 0, "steps": 0, "first": {"true": 0, "ahead_by_rejected": 0, "other": 0},
            "kept": {}, "after_the_end": 0}
    for gap in range(1, 13):
        serve = _run({**SERVE, **_draft(k)}, tmp_path / "model", "serve", [{"id": rid, "iteration": i * gap} for i, rid in enumerate(reqs)], reqs)
        header = serve["records"][0]["config"]
        assert header.get("speculative", {}).get("num_speculative_tokens") == (k or None)
        assert header.get("speculative", {}).get("method") == ("mtp" if k else None)
        same = _replay(serve, tmp_path / "model", reqs)
        assert same["equal"], (gap, same)
        report = X.acceptance(serve)
        assert (report["tokens_not_one_more_than_drafts"], report["emitted_outside_one_to_all"]) == (0, 0), (gap, report)
        assert bool(report["draft_tokens"]) == bool(k)
        seen["after_empty"] += len(serve["arrived_after_an_empty_schedule"])
        seen["early_differs"] += not _replay(serve, tmp_path / "model", reqs, early=True)["equal"]
        seen["hits"] += len(report["cache_hits"])
        seen["empty"] += serve["empty_schedules"]
        seen["steps"] += len(serve["steps"])
        seen["after_the_end"] += not serve["steps"][-1]["emitted"]
        for name, n in report["first_position"].items():
            seen["first"][name] += n
        for name, n in report["rows_by_accepted"].items():
            seen["kept"][name] = seen["kept"].get(name, 0) + n
    with capsys.disabled():
        print(f"\nspecdec load k={k}: {json.dumps(seen, sort_keys=True)}")
    assert seen["hits"], "no request started from a prefix-cache hit"
    assert seen["first"]["other"] == 0, seen
    assert bool(seen["first"]["ahead_by_rejected"]) == bool(k), seen
    if not k:
        assert seen["after_empty"], "no request arrived after an empty schedule"


@pytest.mark.parametrize("k", [0, 1, 5])
def test_a_follow_up_is_put_back_after_the_step_that_finished_the_request_it_goes_on_from(k, tmp_path, capsys):
    """A request at its token limit is held back until its last step's output is back, so that schedule is empty; a second
    request whose prompt is the first's prompt and output and more arrives once the first has finished, and hits every block
    the first filled. Put back by its step alone it is scheduled before that output, and misses the block it completed.
    Without a draft every limit ends so, and the limits run over two blocks; under a draft only a limit the first reaches
    one token short of does."""
    pytest.importorskip("vllm")
    rng = random.Random(11)
    prompt, more = ([rng.randrange(1000, 40000) for _ in range(n)] for n in (40, 24))
    seen: dict[str, Any] = {"after_empty": 0, "early_differs": [], "hits": {}}
    for limit in range(8, 40):
        serve = _run({**SERVE, **_draft(k)}, tmp_path / "model", "serve", [{"id": "first", "iteration": 0}, {"id": "next", "iteration": 10 ** 6}],
                     {"first": (prompt, limit), "next": (lambda tokens: prompt + tokens["first"] + more, 4)})
        assert len(serve["tokens"]["first"]) == limit
        reqs = {"first": (prompt, limit), "next": (prompt + serve["tokens"]["first"] + more, 4)}
        same = _replay(serve, tmp_path / "model", reqs)
        assert same["equal"], (limit, same)
        hit = X.acceptance(serve)["cache_hits"]
        assert set(hit) == {"next"} and hit["next"] % SERVE["block_size"] == 0, (limit, hit)
        seen["hits"][limit] = hit["next"]
        seen["after_empty"] += "next" in serve["arrived_after_an_empty_schedule"]
        if not _replay(serve, tmp_path / "model", reqs, early=True)["equal"]:
            seen["early_differs"].append(limit)
    with capsys.disabled():
        print(f"\nspecdec follow-up k={k}: {json.dumps(seen, sort_keys=True)}")
    if not k:
        assert seen["after_empty"] == 32, "a follow-up arrived with no empty schedule before it"
        assert seen["early_differs"], "a replay that reads the arrival's step alone repeated every serve"
