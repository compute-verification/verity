"""serve-openai's ``--admit release-order`` (`serve_admit.ReleaseOrder`) on CPU: requests reach the engine in the network certifier's
release order (``x-verity-release``) whatever order the frontend reads their bodies in, a missing earlier release holds the
later ones, and a release that never comes (the network certifier dropped it) or is refused holds them only until its skip, never for
good. The gate's skip is driven by hand (an injected ``schedule``), so nothing here waits on the clock.

And a step's rows (`serve_admit.keep_scheduler_rows`): Model Runner V2's own order puts a later-admitted shorter
prompt's first row ahead of an earlier one's (endpoint 2's windows 220 and 237); with the scheduler's order kept, a step's
new rows are in admission order, decodes first. Kept only where no attention backend needs the decodes first."""
from __future__ import annotations

import json
import socket
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.pipeline.test_serve_openai import CERTIFIER_BODY, Committer, Engine, Runner, Serving, Tokenizer, records
from verity_vllm.engine import hooks
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_admit as sa
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline.serve_openai import SSE_DONE
from verity_vllm.protocols.pouw.h1 import requests as RQ


class Timers:
    """The gate's ``schedule``: each armed skip is kept, and `fire` runs the last one armed."""

    def __init__(self) -> None:
        self.armed: list[tuple[float, Any]] = []

    def __call__(self, delay: float, fn: Any) -> None:
        self.armed.append((delay, fn))

    def fire(self) -> None:
        self.armed[-1][1]()


def gate(hold_s: float = 5.0) -> tuple[sa.ReleaseOrder, list[str], list[str], Timers]:
    fwd, aborted, timers = [], [], Timers()
    return sa.ReleaseOrder(lambda a: fwd.append(a.engine_id), aborted.append, hold_s=hold_s, schedule=timers), fwd, aborted, timers


def arrive(g: sa.ReleaseOrder, n: int | None, run: str | None = None) -> SimpleNamespace:
    headers = {} if n is None else {sa.RELEASE_HEADER: str(n)}
    if run is not None:
        headers[sa.RUN_HEADER] = run
    adm = SimpleNamespace(engine_id=f"r{n}", release=g.key(headers))
    g.submit(adm)
    return adm


# ---------------------------------------------------------------------------------------------------------------- the gate

def test_out_of_order_arrivals_reach_the_engine_in_release_order():
    g, fwd, _, timers = gate()
    for n in (2, 1, 3):
        arrive(g, n)
    assert fwd == [] and len(g.held) == 3
    arrive(g, 0)
    assert fwd == ["r0", "r1", "r2", "r3"] and not g.held
    arrive(g, 4)
    assert fwd[-1] == "r4" and g.summary()["ordered"] == 5 and g.summary()["skipped"] == 0
    assert all(d == 5.0 for d, _ in timers.armed)


def test_a_missing_earlier_release_holds_every_later_one_until_it_comes():
    g, fwd, _, _ = gate()
    arrive(g, 0)
    for n in (3, 2):
        arrive(g, n)
    assert fwd == ["r0"]
    arrive(g, 1)
    assert fwd == ["r0", "r1", "r2", "r3"]


def test_a_release_the_warden_dropped_holds_the_rest_only_until_its_skip():
    g, fwd, _, timers = gate()
    arrive(g, 0)
    arrive(g, 2)
    arrive(g, 3)
    assert fwd == ["r0"] and len(timers.armed) == 1
    timers.fire()                                    # 1 never came: skipped, 2 and 3 go on
    assert fwd == ["r0", "r2", "r3"] and g.next == 4 and not g.held
    arrive(g, 1)                                     # it came after all: in at once, counted late
    assert fwd[-1] == "r1" and g.summary()["late"] == 1 and g.summary()["skipped"] == 1
    with pytest.raises(ValueError, match="came twice"):
        g.key({sa.RELEASE_HEADER: "1"})


def test_a_stale_skip_does_nothing():
    g, fwd, _, timers = gate()
    arrive(g, 1)
    first = timers.armed[-1][1]
    arrive(g, 0)                                     # the gap closed before its skip fired
    first()
    assert fwd == ["r0", "r1"] and g.summary()["skipped"] == 0


def test_a_refused_or_aborted_release_passes_its_turn():
    g, fwd, aborted, _ = gate()
    arrive(g, 1)
    g.passed(g.key({sa.RELEASE_HEADER: "0"}))        # 0 was refused (a 400) before it was submitted
    assert fwd == ["r1"]
    held = arrive(g, 3)
    g.abort(held.engine_id)                          # 3's client went away while it waited on 2
    arrive(g, 2)
    assert fwd == ["r1", "r2"] and aborted == []
    g.abort("r2")                                    # past the gate: the engine loop's abort
    assert aborted == ["r2"]
    arrive(g, 4)
    assert fwd[-1] == "r4"


def test_unnumbered_malformed_repeated_and_new_runs():
    g, fwd, _, _ = gate()
    arrive(g, None)
    assert fwd == ["rNone"] and g.summary()["unordered"] == 1
    for bad in ("-1", "x", "1.5", ""):
        with pytest.raises(ValueError, match="non-negative integer"):
            g.key({sa.RELEASE_HEADER: bad})
    arrive(g, 1, run="a")
    with pytest.raises(ValueError, match="came twice"):
        g.key({sa.RELEASE_HEADER: "1", sa.RUN_HEADER: "a"})
    arrive(g, 0, run="b")                            # a new run: what run a still held goes on, and b starts at 0
    assert fwd == ["rNone", "r1", "r0"] and g.run == "b" and g.next == 1
    arrive(g, 7, run="a")                            # a's, after b began: unordered
    assert fwd[-1] == "r7" and g.summary()["stale"] == 1 and g.summary()["runs"] == 2


# ---------------------------------------------------------------------------------------------------------------- a step's rows

RUNNER_MODULE = "tests.pipeline.fake_model_runner_v2"


def runner_module() -> types.ModuleType:
    """Model Runner V2's module as the step reads it (vllm/v1/worker/gpu/model_runner.py at d9105ea8): its runner's
    ``gather_batch_req_state`` lays the step out by the module's ``sort_batch_req_ids``, looked up when the step runs."""
    mod = types.ModuleType(RUNNER_MODULE)
    mod.sort_batch_req_ids = RQ.sort_batch_req_ids

    class GPUModelRunner:
        decode_query_len = 1
        attn_groups = [[SimpleNamespace(metadata_builders=[FlashAttentionMetadataBuilder()])]]

        def gather_batch_req_state(self, so: Any) -> list[str]:
            return mod.sort_batch_req_ids(so.num_scheduled_tokens, so.scheduled_spec_decode_tokens, self.decode_query_len)

    GPUModelRunner.__module__ = RUNNER_MODULE
    mod.GPUModelRunner = GPUModelRunner
    return mod


class FlashAttentionMetadataBuilder:
    reorder_batch_threshold = None


class FlashInferMetadataBuilder:
    reorder_batch_threshold = 1


@pytest.fixture
def v2():
    mod = runner_module()
    sys.modules[RUNNER_MODULE] = mod
    yield mod
    del sys.modules[RUNNER_MODULE]


def fcfs_step(running: dict[str, int], new: dict[str, int], drafts: dict[str, list[int]] | None = None) -> SimpleNamespace:
    """A `SchedulerOutput` as vLLM's FCFS scheduler lists it: its running requests in running order, then the ones it
    newly schedules in waiting-queue order, the admission order; each with its scheduled rows."""
    rows = {**running, **new}
    return SimpleNamespace(num_scheduled_tokens=rows, scheduled_spec_decode_tokens=drafts or {},
                           scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new], total_num_scheduled_tokens=sum(rows.values()))


FCFS = SimpleNamespace(scheduler_config=SimpleNamespace(policy="fcfs"))
DECODES = {f"running.{i}": 1 for i in range(5)}
#: endpoint 2 (prefix caching off), window 237: varied.00035 (128 prompt tokens), then varied.00036
#: (81), released and admitted in that order, both first scheduled in the window's step 0
W237 = fcfs_step({}, {"varied.00035": 128, "varied.00036": 81})
#: window 220: varied.00008/9/10 (86, 239 and 230 prompt tokens), released and admitted in that order, all three in step 0
W220 = fcfs_step({}, {"varied.00008": 86, "varied.00009": 239, "varied.00010": 230})
#: window 237's two beside five running decodes
W237_DECODING = fcfs_step(DECODES, W237.num_scheduled_tokens)


def test_model_runner_v2_lays_co_admitted_prefills_out_by_length(v2):
    runner = v2.GPUModelRunner()
    assert runner.gather_batch_req_state(W237) == ["varied.00036", "varied.00035"]
    assert runner.gather_batch_req_state(W220) == ["varied.00008", "varied.00010", "varied.00009"]
    assert runner.gather_batch_req_state(W237_DECODING) == [*DECODES, "varied.00036", "varied.00035"]


def test_release_order_keeps_the_schedulers_rows(v2):
    runner, vllms = v2.GPUModelRunner(), v2.sort_batch_req_ids
    rows = sa.keep_scheduler_rows(runner, FCFS)
    try:
        assert hooks.live("admit-rows") and v2.sort_batch_req_ids is sa.scheduler_order
        assert runner.gather_batch_req_state(W237) == ["varied.00035", "varied.00036"]
        assert runner.gather_batch_req_state(W220) == ["varied.00008", "varied.00009", "varied.00010"]
        assert runner.gather_batch_req_state(W237_DECODING) == [*DECODES, "varied.00035", "varied.00036"]
        # H1's batch order reads the runner module's sort, so its tables follow the same order
        assert [r for r, _ in RQ.batch_order(runner, W237_DECODING)] == [*DECODES, "varied.00035", "varied.00036"]
    finally:
        rows.uninstall()
    assert v2.sort_batch_req_ids is vllms and not hooks.live("admit-rows")
    assert runner.gather_batch_req_state(W237) == ["varied.00036", "varied.00035"]


def test_a_new_row_of_one_token_stays_behind_the_earlier_admitted_prefill(v2):
    # the token budget leaves the later-admitted request one row: vLLM puts it among the decodes, ahead of the earlier one
    step = fcfs_step({"running.0": 1, "running.1": 1}, {"early": 2045, "late": 1})
    runner = v2.GPUModelRunner()
    assert runner.gather_batch_req_state(step) == ["running.0", "running.1", "late", "early"]
    with sa.keep_scheduler_rows(runner, FCFS):
        assert runner.gather_batch_req_state(step) == ["running.0", "running.1", "early", "late"]


def test_requests_verifying_drafts_stay_first(v2):
    step = fcfs_step({"running.0": 1, "verify": 3}, {"new": 40}, drafts={"verify": [7, 8]})
    with sa.keep_scheduler_rows(v2.GPUModelRunner(), FCFS):
        assert v2.GPUModelRunner().gather_batch_req_state(step) == ["verify", "running.0", "new"]


def test_keeping_the_schedulers_rows_is_refused_without_its_premises(v2):
    v1_runner = type("GPUModelRunner", (), {"__module__": __name__})()
    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER"):
        sa.keep_scheduler_rows(v1_runner, FCFS)
    for sched in (SimpleNamespace(scheduler_config=SimpleNamespace(policy="priority")), SimpleNamespace()):
        with pytest.raises(RuntimeError, match="FCFS"):
            sa.keep_scheduler_rows(v2.GPUModelRunner(), sched)
    unbuilt = v2.GPUModelRunner()
    unbuilt.attn_groups = None
    with pytest.raises(RuntimeError, match="attn_groups"):
        sa.keep_scheduler_rows(unbuilt, FCFS)
    splitting = v2.GPUModelRunner()
    splitting.attn_groups = [[SimpleNamespace(metadata_builders=[FlashAttentionMetadataBuilder()])],
                             [SimpleNamespace(metadata_builders=[FlashInferMetadataBuilder()])]]
    with pytest.raises(RuntimeError, match="FlashInferMetadataBuilder .*decodes first"):
        sa.keep_scheduler_rows(splitting, FCFS)
    assert not hooks.live("admit-rows")


def test_the_pinned_runner_orders_rows_as_its_mirror_and_takes_the_schedulers_order():
    """On a pod (vLLM installed): vLLM's own ``sort_batch_req_ids`` inverts windows 220 and 237 as the mirror does, and
    `keep_scheduler_rows` replaces it in Model Runner V2's module."""
    mr = pytest.importorskip("vllm.v1.worker.gpu.model_runner")
    for step in (W237, W220):
        assert mr.sort_batch_req_ids(step.num_scheduled_tokens, {}, 1) == RQ.sort_batch_req_ids(step.num_scheduled_tokens, {}, 1)
    runner = type("GPUModelRunner", (), {"__module__": mr.__name__, "attn_groups": []})()
    with sa.keep_scheduler_rows(runner, FCFS):
        assert mr.sort_batch_req_ids(W237.num_scheduled_tokens, {}, 1) == list(W237.num_scheduled_tokens)


# ---------------------------------------------------------------------------------------------------------------- over HTTP

class OrderedEngine(Engine):
    def __init__(self, runner: Runner) -> None:
        super().__init__(runner)
        self.added: list[list[int]] = []

    def add_request(self, request_id: str, prompt: dict, params: Any) -> None:
        self.added.append(list(prompt["prompt_token_ids"]))
        super().add_request(request_id, prompt, params)


class OrderedServing:
    """`test_serve_openai.Serving` under ``--admit release-order``: `held` counts the requests the gate has taken."""

    def __init__(self, out: Path) -> None:
        self.runner, self.timers = Runner(), Timers()
        self.windows = sc.CommitWindows(Committer(self.runner), out, run_id="t", window_steps=64, eos_ids=[Tokenizer.eos_token_id],
                                        clock=lambda: 0.0, wall=lambda: 0.0)
        self.engine = OrderedEngine(self.runner)
        self.loop = so.EngineLoop(self.engine, self.windows)
        self.order = sa.ReleaseOrder(self.loop.submit, self.loop.abort, hold_s=5.0, schedule=self.timers)
        self.server = so.Server(proto=so.PlainProtocol(True), served_name="m", max_model_len=512, tokenizer=Tokenizer(),
                                defaults={}, submit=self.loop.submit, abort=self.loop.abort, clock=lambda: 7, order=self.order)
        self.taken, self.gate_submit = threading.Semaphore(0), self.server.submit
        self.server.submit = self._submit
        self.http = so.HttpThread(self.server, "127.0.0.1", 0)

    def _submit(self, adm: Any) -> None:
        self.gate_submit(adm)
        self.taken.release()

    def __enter__(self) -> OrderedServing:
        self.engine_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.engine_thread.start()
        self.http.start()
        self.http.ready.wait()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.loop.stop()
        self.engine_thread.join()
        self.http.shutdown()

    def on_loop(self, fn: Any) -> None:
        """Run ``fn`` on the HTTP server's event loop, where the gate lives, and wait for it."""
        done = threading.Event()
        self.http._loop.call_soon_threadsafe(lambda: (fn(), done.set()))
        done.wait()


def raw_request(body: dict, release: int | None) -> bytes:
    data = json.dumps(body).encode()
    head = b"POST /v1/completions HTTP/1.1\r\nhost: s\r\ncontent-type: application/json\r\n"
    if release is not None:
        head += b"x-verity-release: %d\r\n" % release
    return head + b"content-length: %d\r\n\r\n" % len(data) + data


def connect(s: OrderedServing) -> socket.socket:
    return socket.create_connection(s.http.address)


def read_all(c: socket.socket) -> bytes:
    out = b""
    while chunk := c.recv(1 << 16):
        out += chunk
    c.close()
    return out


def body_of(i: int) -> dict:
    return dict(CERTIFIER_BODY, prompt=[1, 100 + i], max_tokens=2, min_tokens=2)


def test_bodies_read_out_of_order_reach_the_engine_in_release_order(tmp_path):
    with OrderedServing(tmp_path) as s:
        conns = {i: connect(s) for i in range(4)}
        first = raw_request(body_of(0), 0)
        conns[0].sendall(first[:-5])                 # release 0's body is still being read
        for i in (3, 1, 2):
            conns[i].sendall(raw_request(body_of(i), i))
        for _ in range(3):
            s.taken.acquire()
        assert s.engine.added == [] and sorted(s.order.held) == [1, 2, 3]
        conns[0].sendall(first[-5:])
        raws = {i: read_all(c) for i, c in conns.items()}
    assert s.engine.added == [[1, 100 + i] for i in range(4)]
    for raw in raws.values():
        assert raw.startswith(b"HTTP/1.1 200") and raw.endswith(b"0\r\n\r\n")
    reqs = sorted(records(tmp_path, "requests.jsonl"), key=lambda r: r["admitted"])
    assert [r["release"] for r in reqs] == [["", i] for i in range(4)] and [r["admitted"] for r in reqs] == [0, 1, 2, 3]
    assert all(r["check"]["passed"] for r in reqs) and s.order.summary()["ordered"] == 4


def test_a_dropped_release_does_not_hold_the_server_for_good(tmp_path):
    with OrderedServing(tmp_path) as s:
        conns = {i: connect(s) for i in (1, 2)}      # release 0 never reaches the frontend
        for i, c in conns.items():
            c.sendall(raw_request(body_of(i), i))
        s.taken.acquire()
        s.taken.acquire()
        assert s.engine.added == [] and len(s.timers.armed) == 1
        s.on_loop(s.timers.fire)
        raws = [read_all(c) for c in conns.values()]
    assert s.engine.added == [[1, 101], [1, 102]] and all(r.startswith(b"HTTP/1.1 200") for r in raws)
    assert s.order.summary()["skipped_unseen"] == [0]


def test_a_refused_release_and_an_unnumbered_request_hold_nothing(tmp_path):
    with OrderedServing(tmp_path) as s:
        later = connect(s)
        later.sendall(raw_request(body_of(1), 1))
        s.taken.acquire()
        refused = connect(s)
        refused.sendall(raw_request(dict(body_of(0), model="other"), 0))
        bad, plain = read_all(refused), connect(s)
        plain.sendall(raw_request(body_of(9), None))
        raws = [read_all(later), read_all(plain)]
        dup = connect(s)
        dup.sendall(raw_request(body_of(5), 1))
        dup_raw = read_all(dup)
    assert bad.startswith(b"HTTP/1.1 404") and all(r.startswith(b"HTTP/1.1 200") for r in raws)
    assert sorted(s.engine.added) == [[1, 101], [1, 109]] and s.order.summary()["skipped"] == 0
    assert dup_raw.startswith(b"HTTP/1.1 400") and b"came twice" in dup_raw


def test_the_flag_defaults_off_and_takes_release_order():
    base = ["--model", "m", "--revision", "r", "--no-commit"]
    assert cli.parse("serve-openai", base, environ={}).admit == "arrival"
    opts = cli.parse("serve-openai", [*base, "--admit", "release-order", "--admit-hold-ms", "2500"], environ={})
    assert (opts.admit, opts.admit_hold_ms) == ("release-order", 2500)
    with pytest.raises(SystemExit):
        cli.parse("serve-openai", [*base, "--admit", "boundary"], environ={})
    with pytest.raises(cli.UsageError, match="admit-hold-ms"):
        so.preflight(cli.parse("serve-openai", [*base, "--admit-hold-ms", "0"], environ={}), which=lambda _: None)


def test_without_the_flag_a_release_header_changes_nothing(tmp_path):
    with Serving(tmp_path, commit=False) as s:
        c = socket.create_connection(s.http.address)
        c.sendall(raw_request(CERTIFIER_BODY, 3))
        raw = read_all(c)
    assert raw.startswith(b"HTTP/1.1 200") and SSE_DONE in raw
