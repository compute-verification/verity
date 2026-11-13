"""`verity-vllm serve-openai` under ``--admit tick`` with a Commit, when the watermark names a release lost: what the
Commit, the scheduler's trace and each window's advice hold of its request, and that the run still encodes under the
certified tick. The stand-in server of `test_serve_advice` (its HTTP side, engine loop and Commit as `main` runs them),
the gate over its advice trace as `main` puts it, serves one run whose watermark the test sends as the network certifier
writes it (`release.watermark_chunk`): a release's request is in the engine when its bucket's line names it lost, and
another's comes after the line that names it. The first is answered and aborted with a Commit record of an abort after 0
ids in no window, which passes; the scheduler never sees it, so the trace, every window's advice and the certified record
(which holds only requests with an output) are without it. The second is refused at the door and reaches nothing. The
advice v1 string of the run under the tick (`advice_code.codec`), whose replay admits the record's requests at their
release buckets, is the length terms alone, and decodes and compiles to the steps the engine ran."""
from __future__ import annotations

import socket
import threading
from dataclasses import replace
from typing import Any

import pytest
from verity.protocols.certifier.verifier import release as REL

from tests.pipeline.test_serve_admit_tick import RUN, TICK, read_all, request
from tests.pipeline.test_serve_advice import REG, C, SchedEngine, Server, _body
from tests.pipeline.test_serve_openai import Tokenizer, records
from verity_vllm.engine import step_trace as ST
from verity_vllm.engine.engine_driver import external_id
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import serve_admit as sa
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.pipeline.advice_code import codec as V1
from verity_vllm.pipeline.advice_code import schedule as SC
from verity_vllm.program.predict.steps import Registered, compile_steps

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

#: the stand-in scheduler's configuration (`SchedEngine`'s `FakeScheduler`), as the replay reads it: no prefix cache, and a
#: pool no request waits on
ENGINE = SC.Engine(max_num_seqs=4, max_num_batched_tokens=256, max_model_len=512, num_gpu_blocks=1024)
CODE = V1.Registered(ENGINE, (Tokenizer.eos_token_id,), row_order="scheduling", T=2,
                     tick=SC.Tick(TICK.bucket_us, TICK.sat_us, TICK.free_us))
LOST, LATE = _body(8, 7, 4, "na-lost"), _body(9, 6, 3, "na-late")
#: release number -> (its bucket, its body): the releases the watermark names lost are 0, bucket 1's, and 2, bucket 2's
KEPT = {1: (1, _body(0, 8, 4, "na-0")), 3: (3, _body(4, 9, 6, "na-4")), 4: (3, _body(5, 9, 6, "na-5"))}


class TickEngine(SchedEngine):
    """`SchedEngine` as vLLM's ``LLMEngine`` is under the gate: a request is unfinished from its ``add_request`` (the output
    processor holds it) whether or not the gate has let the scheduler see it, until its last output or its abort; and a
    step of a scheduler with no request runs nothing."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.unfinished: set[str] = set()

    def add_request(self, request_id: str, prompt: dict, params: dict) -> None:
        self.unfinished.add(request_id)
        super().add_request(request_id, prompt, params)

    def abort_request(self, ids: list[str]) -> None:
        self.unfinished.difference_update(ids)
        super().abort_request(ids)

    def has_unfinished_requests(self) -> bool:
        return bool(self.unfinished)

    def step(self) -> list[Any]:
        outs = super().step() if self.sched.has_requests() else []
        self.unfinished.difference_update(o.request_id for o in outs if o.finished)
        return outs


class TickServer(Server):
    """`Server` under ``--admit tick``, as `serve_openai.main` wires it: the gate over the advice trace's wrappers, the
    engine loop waiting on it, and the frontend's `TickAdmit`. ``added`` counts the requests the engine loop has added,
    each in the engine (through the gate) and then in the Commit, and ``lines`` the watermark lines the frontend has
    read."""

    engine_of = TickEngine

    def order(self) -> sa.TickAdmit:
        self.gate = self.loop.gate = sa.TickGate(TICK, hold_s=60.0, clock=lambda: 0.0)
        self.loop.sched = self.engine.sched
        self.ticked = sa.install(self.loop.sched, self.gate, key=external_id)
        self.added, self.lines = threading.Semaphore(0), threading.Semaphore(0)
        admitted = self.windows.admitted

        def counted(adm: Any) -> None:
            admitted(adm)
            self.added.release()

        def take(rec: dict[str, Any]) -> None:
            self.loop.tick(rec)
            if "t" in rec:
                self.lines.release()

        self.windows.admitted = counted
        return sa.TickAdmit(self.submit, self.loop.abort, take)

    def __exit__(self, *exc: Any) -> None:
        self.loop.stop()
        self.engine_thread.join()
        self.ticked.uninstall()                          # before the trace's wrappers, which it sits on
        super().__exit__(*exc)


def test_a_run_with_a_lost_release_commits_it_as_an_abort_after_0_and_encodes_under_the_tick_without_it(tmp_path):
    why = b"x-verity-release: %d of run 'ab' was lost on the way to the server, as the watermark says"
    s = TickServer(tmp_path, window_steps=3)
    with s:
        wm = socket.create_connection(s.http.address)
        wm.sendall(REL.watermark_head(bytes.fromhex(RUN)) + REL.watermark_chunk(0, 0))
        c = [socket.create_connection(s.http.address) for _ in range(5)]
        for n, body in ((0, LOST), (1, KEPT[1][1])):
            c[n].sendall(request(body, n, 1))
            s.added.acquire()
        # both are in the engine and have a Commit ledger, and the gate holds them: bucket 1 hasn't closed
        assert len(s.windows.ledgers) == len(s.engine.unfinished) == 2 and not s.engine.sched.requests
        # bucket 2's line too: na-0's steps take T past bucket 1's close, where the gate waits for the watermark
        wm.sendall(REL.watermark_chunk(1, 2, [0]) + REL.watermark_chunk(2, 3, [2]))
        for _ in range(3):
            s.lines.acquire()
        lost, served = read_all(c[0]), [read_all(c[1])]
        c[2].sendall(request(LATE, 2, 2))
        refused = read_all(c[2])
        for n in (3, 4):
            c[n].sendall(request(KEPT[n][1], n, 3))
            s.added.acquire()
        wm.sendall(REL.watermark_chunk(3, 5) + REL.WATERMARK_END)
        served += [read_all(c[3]), read_all(c[4])]
        end = read_all(wm)
    assert lost.startswith(b"HTTP/1.1 200") and why % 0 in lost and b'"finish_reason"' not in lost
    assert refused.startswith(b"HTTP/1.1 400") and why % 2 in refused
    assert end.startswith(b"HTTP/1.1 200") and all(a.startswith(b"HTTP/1.1 200") and a.count(b'"finish_reason":"length"') == 1 for a in served)
    g = s.gate.summary()
    assert (g["admitted"], g["lost"], g["dropped"], g["aborted"], g["skipped"], g["late"], g["opened"]) == (3, 2, 1, 0, 0, 0, None)

    # the Commit: the dropped request's ledger ends as an abort after 0 ids in no window, and passes; the refused one left nothing
    reqs = {r["user"]: r for r in records(tmp_path, "requests.jsonl")}
    assert sorted(reqs) == ["na-0", "na-4", "na-5", "na-lost"] and len(s.admitted) == len(s.windows.arrivals) == 4
    q, e = reqs["na-lost"], reqs["na-lost"]["engine_id"]
    assert (q["finish"], q["streamed_ids"], q["windows"]) == ({"reason": "abort", "stop_reason": None, "k": 0, "by": "client"}, [], [])
    assert all(r["check"]["passed"] for r in reqs.values()) and s.windows.summary()["passed"], q["check"]
    wins = [w for w in records(tmp_path, "windows.jsonl") if not w.get("empty")]
    assert wins and all(e not in w["requests"] for w in wins)

    # the scheduler never saw it: no arrival, no row and no finish of it in the trace, and no window's advice holds it
    recs = ST.load(tmp_path / ST.SERVED)
    assert not any(r.get("request") == e for r in recs) and not any(row[0] == e for r in ST.steps(recs) for row in r["rows"])
    rep = ADV.write(tmp_path, compiler=C)
    assert rep["refused"] == {} and rep["open"] == [] and sorted(map(int, rep["windows"])) == [w["window"] for w in wins]
    bodies, last = {b["user"]: b for _, b in KEPT.values()}, None
    for w in sorted(map(int, rep["windows"])):
        a = WA.decode((tmp_path / "advice" / f"w{w}" / "private" / "advice.bin").read_bytes())
        if last is not None:
            WA.follows(last, a)
        last = a
        assert {x.id for x in a.requests} <= set(bodies)
        compile_steps(REG, WA.encode(a), {x.id: bodies[x.id] for x in a.requests})

    # a certified record holds a request by its output ids, so it can't hold the dropped one. The record of the rest, as
    # the test stands in for the network certifier's (each request at its release bucket, counted from the first, and
    # its outputs released the bucket after): the tick's replay of it is the steps the engine ran, to the gate's own T
    with pytest.raises(V1.AdviceCodeError, match="request na-lost: a prompt, an output and max_tokens of at least one"):
        V1.Request("na-lost", tuple(q["prompt_ids"]), LOST["max_tokens"], tuple(q["streamed_ids"]), 0, ())
    first = min(b for b, _ in KEPT.values())
    record = V1.Record(tuple(V1.Request(body["user"], tuple(body["prompt"]), body["max_tokens"], tuple(reqs[body["user"]]["streamed_ids"]),
                                        b - first, (b - first + 1,) * body["max_tokens"]) for b, body in KEPT.values()))
    user = {r["engine_id"]: u for u, r in reqs.items()}
    ran = [[(user[i], start, n) for i, start, n in st["rows"]] for st in ST.steps(recs)]
    rp, decisions, _ = V1.replay_record(record, CODE, ENGINE, V1.release_order(record), lambda i, k: True)
    assert [[(record.requests[i].id, start, n) for i, start, n in st.rows] for st in rp.steps] == ran and decisions == []
    assert rp.clock[-1] + first * TICK.bucket_us == s.gate.T
    advice = V1.decode_bits([], record, replace(CODE, send_lengths=False))
    data = V1.encode(advice, record, CODE)
    assert sum(V1.bits_by_window(data, record, CODE).values()) == sum(b["max_tokens"].bit_length() + 1 for b in bodies.values())
    assert V1.decode(data, record, CODE) == advice
    assert [[(advice.requests[x.request].id, x.start, x.n) for x in st.rows] for st in advice.steps] == ran
    compiled = compile_steps(Registered.of(REG.phi, 7, code=CODE), data, bodies, record=record)
    assert [[(compiled.requests[r.request].id, r.start, r.n) for r in st] for st in compiled.steps] == ran
