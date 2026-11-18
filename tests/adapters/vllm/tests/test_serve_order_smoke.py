"""The serve's whole order of events with each declared scheme attached, alone and with each scheme an application lists
beside it, on a processor: one driver (`serve`) runs serve-openai's own functions over stand-ins for vLLM's classes
(`tests.fake_vllm`), its committer and Pearl-C's device code, so every event comes from the emitter the serve runs.
Each negative control breaks one stand-in and the smoke fails by that reason.

What a processor can't run stays a stand-in, each emitting by hand where the serve's emitter needs a card (`HAND`):
build_engine's answer (it reads the row's checkpoint manifest), the native committer's `ValueProduced` and the engine's
input batch. Not driven: the window's seal (``--seal``, a sealer process) and the plugin's attach (`scheme_plugin`). A
scheme a processor can't attach fails with its reason unless `EXCEPTED` names what it lacks and what covers it instead."""

from __future__ import annotations

import itertools
import json
import sys
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests import fake_vllm
from tests.pipeline import predicted
from tests.pipeline.test_serve_learn import Bounded, parse
from tests.pipeline.test_serve_openai import Tokenizer
from verity import hooks as H
from verity.channel import Refused
from verity.protocols.compose import schemes as declarations
from verity_vllm import messages as M

REPO = Path(__file__).resolve().parents[4]
#: the requests the fixed load sends, each of ``IDS`` prompt ids and ``STEPS`` output tokens: under ``--window-requests``
#: and ``--window-steps`` of ``SHAPE``, two windows
REQUESTS, IDS, STEPS = 8, 8, 2
PAGES, SLOTS, N, K = 40, 16, 64, 256
#: serve-openai's options as the H100 smoke's serve gives them, at the fixed load's size
SHAPE = ("--protocol", "plain", "--max-num-seqs", str(REQUESTS // 2), "--max-num-batched-tokens", "512", "--window-steps",
         str(STEPS), "--max-model-len", "64")
#: the settings each scheme needs beyond what `serve_openai.listed` holds, as an application's configuration gives them
SETTINGS = {"pearl-plus": {"kernel": "stand-in"}, "network-attester": {"bucket_us": 1000, "sat_us": 100, "free_us": 10}}
#: the stages where the stand-in emits by hand what the serve's emitter emits on a card, each a place it can drift
HAND = {"declared": "engine.build.build_engine's ex.answer(H.Declared(...)): it reads the row's checkpoint manifest",
        "values": "commit/committer/native_host.py's ValueProduced, from the native committer's module hooks",
        "batch": "vLLM's input batch (ib, ids) the committer reads from execute_model's argument"}
#: schemes a processor can't attach, each with what it lacks and what covers it instead
EXCEPTED = {"wipe": "the card's facts (device, links, disk, dram, streamer, idle_mib, ...) a processor has none of; "
                    "tests/verity/protocols/resource_accounting/memory/pous/test_wipe_served.py covers its attach and "
                    "release over simulated facts"}
ROOT = bytes(64)


def applications() -> list[list[str]]:
    """Each application's ``protocols`` list that names a declared scheme, its declared schemes in its order."""
    got = []
    for p in sorted((REPO / "applications").glob("*.toml")):
        names = [n for n in tomllib.loads(p.read_text()).get("protocols", []) if n in declarations.DECLARATIONS]
        if names:
            got.append(names)
    return got


def needs(name: str) -> list[str]:
    """``name`` after every scheme it depends on (its declaration's ``depends``), each once, those first."""
    depends = declarations.get(name).depends
    return [*dict.fromkeys(d for n in (depends if isinstance(depends, tuple) else ()) for d in needs(n)), name]


def listings() -> list[list[str]]:
    """Each declared scheme alone, then each pair an application lists together, each with what it depends on: in the
    application's order where one lists them all."""
    alone = [needs(n) for n in sorted(declarations.DECLARATIONS)]
    paired = set()
    for names in applications():
        for a, b in itertools.combinations(names, 2):
            want = {*needs(a), *needs(b)}
            paired.add(tuple(n for n in names if n in want) if want <= set(names) else (*needs(a), *needs(b)))
    return [*alone, *(list(dict.fromkeys(p)) for p in sorted(paired) if list(dict.fromkeys(p)) not in alone)]


# ---------------------------------------------------------------------------------------------------- vLLM's classes

class Block:
    def __init__(self, i: int) -> None:
        self.block_id, self.ref_cnt, self.is_null = i, 0, i == 0


class FreeQueue:
    def __init__(self, blocks: list[Block]) -> None:
        self.q = list(blocks)

    def remove(self, b: Block) -> None:
        self.q.remove(b)

    def append_n(self, bs: list[Block]) -> None:
        self.q.extend(bs)


class BlockPool:
    def __init__(self) -> None:
        self.blocks = [Block(i) for i in range(PAGES)]
        self.free_block_queue = FreeQueue(self.blocks[1:])

    def get_new_blocks(self, n: int) -> list[Block]:
        got = [self.free_block_queue.q.pop(0) for _ in range(n)]
        for b in got:
            b.ref_cnt = 1
        return got

    def free_blocks(self, ordered: Any) -> None:
        for b in ordered:
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free_block_queue.q.insert(0, b)

    def touch(self, blocks: list[Block]) -> None:
        for b in blocks:
            b.ref_cnt += 1


class KVCacheManager:
    def __init__(self, pool: BlockPool) -> None:
        self.block_pool = pool
        self.stm = SimpleNamespace(block_size=SLOTS, req_to_blocks={})
        self.coordinator = SimpleNamespace(single_type_managers=[self.stm])

    def allocate_slots(self, request: Any, blocks: int) -> list[Block]:
        got = self.block_pool.get_new_blocks(blocks)
        self.stm.req_to_blocks.setdefault(request.request_id, []).extend(got)
        return got

    def free(self, request: Any) -> None:
        self.block_pool.free_blocks(reversed(self.stm.req_to_blocks.pop(request.request_id, [])))


class Scheduler:
    """vLLM's V1 scheduler, FCFS: a request's whole prompt at its first step, a token at each after; its blocks allocated
    while scheduling and freed when its output finishes it."""

    def __init__(self, manager: KVCacheManager) -> None:
        self.kv_cache_manager = manager
        self.requests: dict[str, Any] = {}
        self.waiting: list[Any] = []
        self.running: list[Any] = []
        self.finished_req_ids: set[str] = set()
        self.scheduler_config = SimpleNamespace(policy="fcfs")
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))

    def add_request(self, request: Any) -> None:
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def schedule(self) -> Any:
        self.running, self.waiting = self.running + self.waiting, []
        tokens = {}
        for r in self.running:
            n = len(r.prompt_token_ids) - r.num_computed_tokens if not r.out else 1
            have = len(self.kv_cache_manager.stm.req_to_blocks.get(r.request_id, ()))
            if need := -(-(r.num_computed_tokens + n) // SLOTS) - have:
                self.kv_cache_manager.allocate_slots(r, need)
            r.num_computed_tokens += n
            tokens[r.request_id] = n
        return SimpleNamespace(num_scheduled_tokens=tokens, total_num_scheduled_tokens=sum(tokens.values()),
                               scheduled_spec_decode_tokens={})

    def update_from_output(self, so: Any, sampled: dict[str, int]) -> dict[int, Any]:
        outs = []
        for rid in so.num_scheduled_tokens:
            r = self.requests[rid]
            r.out.append(sampled[rid])
            done = len(r.out) >= r.max_tokens
            outs.append(SimpleNamespace(request_id=rid, new_token_ids=[sampled[rid]],
                                        finish_reason="length" if done else None, stop_reason=None))
            if done:
                self._free(r)
        return {0: SimpleNamespace(outputs=outs)}

    def finish_requests(self, request_ids: Any, status: Any) -> None:
        for rid in [request_ids] if isinstance(request_ids, str) else list(request_ids or self.requests):
            if (r := self.requests.get(rid)) is not None:
                self._free(r)

    def _free(self, r: Any) -> None:
        self.running = [x for x in self.running if x is not r]
        self.waiting = [x for x in self.waiting if x is not r]
        del self.requests[r.request_id]
        self.kv_cache_manager.free(r)


class LinearBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(N, K, dtype=torch.bfloat16), requires_grad=False)
        self.bias = None
        self.quant_method = Unquantized()


class Unquantized:
    def apply(self, layer: Any, x: Any, bias: Any = None) -> Any:
        return torch.nn.functional.linear(x, layer.weight, bias)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = LinearBase()

    def forward(self, ids: list[int]) -> Any:
        x = torch.zeros(len(ids), K, dtype=torch.bfloat16)
        x[:, 0] = torch.tensor(ids, dtype=torch.bfloat16)
        return self.proj.quant_method.apply(self.proj, x)


class GPUModelRunner:
    """Model Runner V2: its model once loaded, its KV caches once bound, a forward over the step's ids."""

    def __init__(self) -> None:
        self.cache_config = SimpleNamespace(block_size=SLOTS)
        self.kernel_block_sizes = [SLOTS]
        self.sampler = SimpleNamespace(sampling_states=SimpleNamespace(seeds=SimpleNamespace(gpu=torch.zeros(64, dtype=torch.int64))))

    def load_model(self) -> None:
        self.model = Model()

    def initialize_kv_cache(self, kv_cache_config: Any, is_profiling: bool = False) -> None:
        self.kv_caches = [torch.zeros(2, kv_cache_config.num_blocks, SLOTS, K, dtype=torch.uint8)]

    def execute_model(self, so: Any) -> Any:
        return self.model(so.ids)

    def sample(self, ib: Any, sampled: list[int]) -> list[int]:
        return sampled


class LLMEngine:
    """vLLM's in-process ``LLMEngine``, its core holding the scheduler: a step schedules, runs the model and updates
    from its output, as vLLM's engine core does; a request's internal id is ``<id>-<8 hex>``."""

    def __init__(self, runner: GPUModelRunner, sched: Scheduler) -> None:
        self.runner, self.sched, self.n, self.ids = runner, sched, 0, {}
        self.engine_core = SimpleNamespace(engine_core=SimpleNamespace(scheduler=sched))
        self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: 32000))

    def add_request(self, request_id: str, prompt: dict, params: Any) -> None:
        self.n += 1
        slot = self.n % 64
        self.runner.sampler.sampling_states.seeds.gpu[slot] = params["seed"]
        internal = f"{request_id}-{self.n:08x}"
        self.ids[internal] = request_id
        ids = list(prompt["prompt_token_ids"])
        self.sched.add_request(SimpleNamespace(request_id=internal, prompt_token_ids=ids, num_prompt_tokens=len(ids),
                                               out=[], max_tokens=params["max_tokens"], num_computed_tokens=0, slot=slot))

    def abort_request(self, ids: list[str]) -> None:
        self.sched.finish_requests([i for i, e in self.ids.items() if e in ids], "aborted")

    def has_unfinished_requests(self) -> bool:
        return bool(self.sched.requests)

    def step(self) -> list[Any]:
        so = self.sched.schedule()
        live = [self.sched.requests[r] for r in so.num_scheduled_tokens]
        ids, qsl, starts = [], [0], []
        for r, n in zip(live, so.num_scheduled_tokens.values()):
            start = r.num_computed_tokens - n
            ids += (r.prompt_token_ids + r.out)[start:start + n]
            qsl.append(len(ids))
            starts.append(start)
        so.ids = ids
        so.ib = SimpleNamespace(num_reqs=len(live), req_ids=[r.request_id for r in live], query_start_loc_np=np.array(qsl),
                                idx_mapping=torch.tensor([r.slot for r in live], dtype=torch.int32),
                                positions=torch.tensor([p for s, a, b in zip(starts, qsl, qsl[1:]) for p in range(s, s + b - a)]),
                                logits_indices=torch.tensor(qsl[1:]) - 1, num_computed_tokens_np=np.array(starts),
                                prefill_len_np=np.array([len(r.prompt_token_ids) for r in live]))
        if live:
            self.runner.execute_model(so)
        sampled = [(sum(ids[qsl[i]:qsl[i + 1]]) + 31 * len(r.out)) % 32000 for i, r in enumerate(live)]
        if live:
            self.runner.sample(so.ib, sampled)
        prompts = {r.request_id: r.prompt_token_ids for r in live}
        outs = self.sched.update_from_output(so, dict(zip(so.num_scheduled_tokens, sampled)))
        return [SimpleNamespace(request_id=self.ids[o.request_id], finished=o.finish_reason is not None,
                                prompt_token_ids=prompts[o.request_id],
                                outputs=[SimpleNamespace(token_ids=o.new_token_ids, text=f" w{o.new_token_ids[0]}",
                                                         finish_reason=o.finish_reason, stop_reason=None)])
                for o in outs[0].outputs]


VLLM = {"vllm.v1.core.block_pool:BlockPool": BlockPool, "vllm.v1.core.kv_cache_manager:KVCacheManager": KVCacheManager,
        "vllm.v1.core.sched.scheduler:Scheduler": Scheduler, "vllm.v1.worker.gpu.model_runner:GPUModelRunner": GPUModelRunner,
        "vllm.model_executor.layers.linear:LinearBase": LinearBase}


class Committer(Bounded):
    """The serve's committer under bounded staging (`test_serve_learn.Bounded`), its module hook on the model's linear
    emitting `ValueProduced` as the native committer's does (`HAND["values"]`)."""

    def install(self) -> None:
        super().install()
        from verity_vllm import messages

        def hook(module: Any, args: Any, output: Any) -> None:
            if (ex := messages.current()).wants("values"):
                ex.emit(H.ValueProduced("model.proj", "producer_output", 0, output, ex.step))

        self.handle = self.runner.model.proj.register_forward_hook(hook)

    def uninstall(self) -> None:
        self.handle.remove()
        super().uninstall()


# ---------------------------------------------------------------------------------------------- Pearl-C's device code

class DeviceRun:
    """Pearl-C's device run off a card: its passes counted, each block's output its input."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.commits, self.details, self.passes = {"coins": {}}, {"passes": []}, 0
        self.cur = SimpleNamespace(matmuls=[])

    def begin(self) -> None:
        pass

    def end_pass(self, step: bool = False) -> None:
        self.passes += 1

    def finish(self) -> None:
        pass

    def gemm(self, key: Any, x: Any, weight: Any, bias: Any) -> Any:
        return x


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pearl-C's device code (`verity_torch.pouw`'s run, weight roots and step limit) as stand-ins and its kernel
    loading no cubin, with nothing handed to PoUW and `verity_torch.pouw` not imported, so only the serve's own call
    hands its device run in."""
    import verity_torch
    from verity.protocols.resource_accounting.compute.pouw.prover.serving import tensors
    from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import kernel
    from verity_torch.pouw.pearl_c import passes, run_device

    monkeypatch.setattr(run_device, "PearlCDeviceRun", DeviceRun)
    monkeypatch.setattr(run_device, "weight_roots", lambda k, weights, scheme, *, weight_salts: {w: ROOT for w in weights})
    monkeypatch.setattr(passes, "step_tokens", lambda run, calls: 512)
    monkeypatch.setattr(kernel.Kernel, "load", classmethod(lambda cls, directory, cubin=None: SimpleNamespace()))
    monkeypatch.setattr(tensors, "_pearl_c", None)
    monkeypatch.delitem(sys.modules, "verity_torch.pouw")
    monkeypatch.delattr(verity_torch, "pouw")


# ---------------------------------------------------------------------------------------------------------- the driver

@contextmanager
def heard() -> Iterator[list[Any]]:
    """Every event emitted on the exchange in scope, in order, recorded without subscribing (a subscriber would make an
    emitter build events no scheme reads)."""
    got: list[Any] = []
    emit = M.Exchange.emit

    def recording(self: Any, event: Any) -> None:
        got.append(event)
        emit(self, event)

    M.Exchange.emit = recording
    try:
        yield got
    finally:
        M.Exchange.emit = emit


def serve(names: list[str], out: Path, monkeypatch: pytest.MonkeyPatch,
          broken: Callable[[str, Any], None] = lambda stage, it: None) -> tuple[list[Any], Any]:
    """serve-openai's run, stage by stage, over the stand-ins with ``names`` listed in order: the attach and the build
    (`serve_openai.build`), the learn-only warm-up (`serve_learn.warmed`), the fixed load's two windows through the engine
    loop (`serve_openai.main`'s `CommitWindows` and `EngineLoop`) and the shutdown's reports (`write_reports`); the events
    in their order and the exchange. ``broken(stage, it)`` breaks a stand-in at each stage, for a negative control."""
    from verity_vllm.engine import build as engine_build
    from verity_vllm.pipeline import serve_commit as sc
    from verity_vllm.pipeline import serve_learn as sl
    from verity_vllm.pipeline import serve_openai as so
    from verity_vllm.pipeline import serve_protocol as proto

    fake = fake_vllm.install(monkeypatch)
    for key, cls in VLLM.items():
        fake.set(key, cls)
    broken("vllm", fake)
    monkeypatch.chdir(out)
    M.close()
    doc = {n: SETTINGS.get(n, {}) for n in names}
    (row := out / "row").mkdir()
    opts = parse(*SHAPE, "--window-requests", str(REQUESTS // 2), "--protocols", ",".join(names),
                 "--protocols-config", json.dumps(doc), "--out", str(out / "serve"), "--precision", "bfloat16",
                 "--required-manifest", str(predicted.row(row)))
    environ = {"CARD_SKU": "stand-in", "CUDA_HOME": "/stand-in/cuda"}
    broken("options", environ)
    with monkeypatch.context() as env:
        for k in ("CARD_SKU", "CUDA_HOME"):
            env.delenv(k, raising=False)
        for k, v in environ.items():
            env.setenv(k, v)
        so.preflight(opts, which=lambda _: "/stand-in/ninja", environ=environ)
    with heard() as events:
        found = so.listed(opts, out)
        ex = M.build()
        M.attach_listed(ex, found, gate=None, ledger=None, private=str(out / "private"), process=H.ENGINE)
        broken("attached", ex)
        fixed = sorted({r.ranges for _, r in ex.requests if isinstance(r, H.FixRanges)})
        steps = {r.value for _, r in ex.requests if isinstance(r, H.RequireProperty) and r.property == "async_steps"}
        ex.answer(H.Declared("", {"model": "m", "execution": None, "eager": True, "world": 1, "dtype": "bfloat16",
                                  "quantization": None, "prefix_cache": False, "kernels": None,
                                  "async_steps": steps.pop() if len(steps) == 1 else None},
                             tuple(ex.attached), dict(ex.attached)))

        def make() -> Any:
            runner = GPUModelRunner()
            runner.load_model()
            runner.initialize_kv_cache(SimpleNamespace(num_blocks=fixed[0] if fixed else PAGES))
            return SimpleNamespace(vllm=SimpleNamespace(llm_engine=LLMEngine(runner, Scheduler(KVCacheManager(BlockPool())))))

        llm, _hook, off = engine_build.hooked(make)
        engine = llm.vllm.llm_engine
        com = Committer(engine.runner, opts.learning.learn_host_budget_mb)
        broken("built", engine)
        try:
            sl.warmed(com, llm, opts)
            broken("warmed", engine)
            windows = sc.CommitWindows(com, out, run_id="smoke", window_steps=opts.window_steps,
                                       eos_ids=[Tokenizer.eos_token_id], clock=lambda: 0.0, wall=lambda: 0.0,
                                       request_cap=opts.window_requests)
            loop = so.EngineLoop(engine, windows, cap=opts.window_requests, clock=lambda: 0.0)
            server = so.Server(proto=proto.PlainProtocol(True), served_name="m", max_model_len=opts.max_model_len,
                               tokenizer=Tokenizer(), defaults={}, submit=loop.submit, abort=loop.abort, clock=lambda: 7)
            for i in range(REQUESTS):
                body = {"prompt": [100 + i] * IDS, "max_tokens": STEPS, "min_tokens": STEPS, "ignore_eos": True,
                        "temperature": 0, "stream": True}
                loop.submit(server.admit(body, lambda _d: None)[1])
            loop.drain("the fixed load is served")
            loop.run()
            assert loop.error is None and windows.summary()["passed"], (loop.error, windows.summary())
            so.write_reports(M.current(), out)
        finally:
            off()
    return events, ex


# ------------------------------------------------------------------------------------------------------------ the cases

def unheard(events: list[Any], ex: Any) -> dict[str, list[str]]:
    """For each hook an attached scheme subscribes, the kinds of event on it that the serve never emitted."""
    return {h: missing for h in H.HOOKS if ex.subscribers[h]
            if (missing := sorted({c.__name__ for c in H.EVENTS if c.hook == h} - {type(e).__name__ for e in events}))}


@pytest.mark.parametrize("names", listings(), ids=lambda names: "+".join(names))
def test_the_serves_order_with_each_listed_scheme_attached(names, device, tmp_path, monkeypatch):
    if why := next((EXCEPTED[n] for n in names if n in EXCEPTED), None):
        pytest.skip(why)
    if unlisted := [n for n in names if not isinstance(declarations.get(n).depends, tuple)]:
        with pytest.raises(Refused, match=f"{unlisted[0]}'s declaration does not say which schemes it depends on"):
            serve(names, tmp_path, monkeypatch)
        return
    events, ex = serve(names, tmp_path, monkeypatch)
    assert not stray(events), f"events of no hook's type: {stray(events)}"
    assert not unheard(events, ex)
    phases = [type(e).__name__ for e in events if e.hook == "steps" and type(e).__name__.startswith("Phase")]
    assert phases == (["PhaseOpened", "PhaseClosed", "PhaseCommitted"] * 2 if ex.subscribers["steps"] else [])


def stray(events: list[Any]) -> list[str]:
    """The events whose type is no hook's (`H.EVENTS`)."""
    return [type(e).__name__ for e in events if type(e) not in H.EVENTS]


class Misfiled(H.RangeWritten):
    """A memory event emitted on the steps hook."""

    hook = "steps"


def breaking(stage: str, act: Callable[[Any], None]) -> Callable[[str, Any], None]:
    return lambda s, it: act(it) if s == stage else None


def misfiled(engine: Any) -> None:
    schedule = engine.sched.schedule

    def scheduled() -> Any:
        so = schedule()
        from verity_vllm import messages
        messages.current().emit(Misfiled(0, 0, 1, messages.current().next_step))
        return so
    engine.sched.schedule = scheduled


def unhanded(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    from verity.protocols.resource_accounting.compute.pouw.prover.serving import tensors
    return lambda _: monkeypatch.setattr(tensors, "use_pearl_c", lambda *a, **k: None)


def heard_warm_up(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    @contextmanager
    def heard_all(self: Any) -> Iterator[None]:
        yield
    return lambda _: monkeypatch.setattr(M.Exchange, "unheard", heard_all)


def second_phase_unopened(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    emit = M.Exchange.emit

    def dropping(self: Any, event: Any) -> None:
        if not (isinstance(event, H.PhaseOpened) and event.phase == 1):
            emit(self, event)
    return lambda _: monkeypatch.setattr(M.Exchange, "emit", dropping)


#: each stage's broken stand-in, the scheme it is run with, and the reason the smoke fails by
NEGATIVE = {
    "an event of the wrong type on a hook": ("built", ["pearl-plus"], lambda mp: misfiled, AssertionError, "Misfiled"),
    "no device run handed in": ("options", ["pearl-plus"], unhanded, NotImplementedError,
                                "no Pearl-C device run handed in"),
    "the warm-up heard, a step outside any phase before the first": ("built", ["pearl-plus"], heard_warm_up, RuntimeError,
                                                                      "after steps outside any phase"),
    "a step outside any phase after one closed": ("warmed", ["pearl-plus"], second_phase_unopened, RuntimeError,
                                                   "phase 1 closes, and the open phase is None"),
    "CARD_SKU unset": ("options", ["pearl-plus"], lambda mp: lambda env: env.pop("CARD_SKU"), Exception, "CARD_SKU"),
}


@pytest.mark.parametrize("control", list(NEGATIVE))
def test_each_broken_stand_in_fails_the_smoke_by_its_reason(control, device, tmp_path, monkeypatch):
    stage, names, act, error, reason = NEGATIVE[control]
    with pytest.raises(error, match=reason):
        events, _ = serve(names, tmp_path, monkeypatch, breaking(stage, act(monkeypatch)))
        assert not stray(events), f"events of no hook's type: {stray(events)}"
