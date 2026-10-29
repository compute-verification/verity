"""`verity-vllm serve-openai` records what `verity-vllm advice` reads, and the advice of each of its Commit windows compiles back to
the steps it served: a stand-in server (its HTTP side and engine loop as `main` runs them, the scheduler `tests.engine.fake_scheduler`
under the committer stand-in of `test_serve_openai`) serves two waves of requests, the engine idle between them, so in at least
two Commit windows; requests.jsonl
carries each body's `user`, request_users.jsonl its engine id and `user` alone, advice_trace.jsonl the scheduler's steps and
each window's open and close, and each window's advice,
committed under w<k>/, compiles (`predict.steps.compile_steps`) to the rows the trace recorded, each request at the shape it ran.
At windows of 3 steps a request runs in two windows or more: the next window carries it at the state the last one left
(`window_advice`), each window's advice binds and rebuilds, and the windows' rows place its Program's positions once each; a row
that isn't where the carry says, or a carry that a refused window never left, is refused."""
from __future__ import annotations

import json
import re
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.engine.fake_scheduler import FakeRequest, FakeScheduler, runner_order
from tests.pipeline.test_serve_openai import Committer, Runner, Tokenizer, records, serve_concurrently
from verity.core.protocols.certifier import advice as I1
from verity_vllm.engine import step_trace as ST
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_protocol as proto
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.pipeline.serve_admit import Delta
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered, compile_steps

C = ADV.registered_compiler("test-compiler", "ab" * 64)
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
FINISH = {0: "stop", 1: "length"}
REG = Registered.of(Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188}),
                    default_seed=7)


class SchedEngine:
    """vLLM's ``LLMEngine`` as the loop drives it, its decisions the stand-in scheduler's: each ``step`` schedules (whole prompts at
    a request's first step, else one decode), runs the scheduled rows through the runner in Model Runner V2's order under vLLM's
    internal ids ``<engine id>-<8 hex>``, and returns each emitted token as a DELTA output under the engine id. The client of a prompt
    in ``abort_after`` goes away once it has that many outputs (``client_goes``, called in the step that emitted the last, or as the
    engine takes the request for 0)."""

    def __init__(self, runner: Runner, abort_after: dict[tuple[int, ...], int] | None = None) -> None:
        self.runner, self.sched, self.n, self.slot, self.engine_id = runner, FakeScheduler(budget=256, max_seqs=4), 0, {}, {}
        self.abort_after, self.client_goes = dict(abort_after or {}), lambda engine_id: None

    def add_request(self, request_id: str, prompt: dict, params: dict) -> None:
        self.n += 1
        rid, ids = f"{request_id}-{self.n:08x}", list(prompt["prompt_token_ids"])
        self.slot[rid], self.engine_id[rid] = self.n % 64, request_id
        self.runner.sampler.sampling_states.seeds.gpu[self.n % 64] = params["seed"]
        outputs = [3 + (sum(ids) + 7 * k) % 200 for k in range(params["max_tokens"])]
        self.sched.add_request(FakeRequest(rid, ids, outputs, params["max_tokens"], sampling_params=SimpleNamespace(**params)))
        if self.abort_after.get(tuple(ids)) == 0:
            self.client_goes(request_id)

    def abort_request(self, ids: list[str]) -> None:
        self.sched.finish_requests([r for r, e in self.engine_id.items() if e in ids], "FINISHED_ABORTED")

    def has_unfinished_requests(self) -> bool:
        return self.sched.has_requests()

    def step(self) -> list[Any]:
        so_ = self.sched.schedule()
        nst = so_.num_scheduled_tokens
        order = runner_order(nst)
        ids, qsl, pos, starts = [], [0], [], []
        for rid in order:
            q = self.sched.requests[rid]
            start = q.num_computed_tokens - nst[rid]
            ids += (q.prompt_token_ids + q.emitted)[start:start + nst[rid]]
            qsl.append(len(ids))
            pos += range(start, start + nst[rid])
            starts.append(start)
        ib = SimpleNamespace(num_reqs=len(order), req_ids=order, query_start_loc_np=torch.tensor(qsl).numpy(),
                             idx_mapping=torch.tensor([self.slot[r] for r in order], dtype=torch.int32), positions=torch.tensor(pos),
                             logits_indices=torch.tensor(qsl[1:]) - 1, num_computed_tokens_np=torch.tensor(starts).numpy(),
                             prefill_len_np=torch.tensor([len(self.sched.requests[r].prompt_token_ids) for r in order]).numpy())
        self.runner.execute_model(SimpleNamespace(ib=ib, ids=ids))
        mro = self.sched.execute(so_)
        self.runner.sample(ib, [t[0] for t in mro.sampled_token_ids])
        outs = []
        for o in self.sched.update_from_output(so_, mro)[0].outputs:
            q = self.sched.requests[o.request_id]
            outs.append(SimpleNamespace(request_id=self.engine_id[o.request_id], finished=o.finish_reason is not None,
                                        prompt_token_ids=q.prompt_token_ids,
                                        outputs=[SimpleNamespace(token_ids=list(o.new_token_ids), text=f" w{o.new_token_ids[0]}",
                                                                 finish_reason=FINISH.get(o.finish_reason), stop_reason=None)]))
            if o.finish_reason is None and self.abort_after.get(tuple(q.prompt_token_ids)) == len(q.emitted):
                self.client_goes(self.engine_id[o.request_id])
        return outs


class Server:
    """`serve_openai.main`'s wiring over the stand-ins: the advice trace on the scheduler for the server's lifetime, the Commit's
    windows marking it."""

    def __init__(self, out: Path, window_steps: int = 64, abort_after: dict[tuple[int, ...], int] | None = None) -> None:
        self.runner = Runner()
        self.engine = SchedEngine(self.runner, abort_after)
        self.engine.client_goes = self.client_goes
        self.trace = sc.advice_trace(out, self.engine.sched)
        self.windows = sc.CommitWindows(Committer(self.runner), out, run_id="t", window_steps=window_steps, eos_ids=[Tokenizer.eos_token_id],
                                        clock=lambda: 0.0, wall=lambda: 0.0, marks=self.trace.window)
        self.loop, self.admitted = so.EngineLoop(self.engine, self.windows), {}
        self.server = so.Server(proto=proto.PlainProtocol(True), served_name="m", max_model_len=512, tokenizer=Tokenizer(), defaults={},
                                submit=self.submit, abort=self.loop.abort, clock=lambda: 7)
        self.http = so.HttpThread(self.server, "127.0.0.1", 0)
        self.engine_thread = threading.Thread(target=self.loop.run, daemon=True)

    def __enter__(self) -> Server:
        self.engine_thread.start()
        self.http.start()
        self.http.ready.wait()
        host, port = self.http.address
        self.url = f"http://{host}:{port}"
        return self

    def __exit__(self, *exc: Any) -> None:
        self.loop.stop()
        self.engine_thread.join()
        self.http.shutdown()
        self.trace.uninstall()

    def submit(self, adm: Any) -> None:
        self.admitted[adm.engine_id] = adm
        self.loop.submit(adm)

    def client_goes(self, engine_id: str) -> None:
        """The request's client goes away: its abort is queued, so the loop takes it before the next step, as it takes a
        disconnect's, and its stream ends."""
        self.loop.abort(engine_id)
        self.admitted[engine_id].deliver(Delta([], "", error="the client went away"))

    def post(self, path: str, body: dict) -> tuple[int, bytes]:
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()


def _body(i: int, lp: int, k: int, user: str | None) -> dict:
    b = {"model": "m", "prompt": [(31 * i + 7 * j + 1) % 256 for j in range(lp)], "max_tokens": k, "stream": True, "temperature": 0,
         "top_p": 1}
    return b if user is None else {**b, "user": user}


WAVES = [[_body(0, 8, 4, "na-0"), _body(1, 5, 3, "na-1"), _body(2, 12, 5, "na-2"), _body(3, 3, 2, None)],
         [_body(4, 9, 6, "na-4"), _body(5, 9, 6, "na-5")]]


def _served(tmp_path: Path, window_steps: int = 64) -> Path:
    with Server(tmp_path, window_steps) as s:
        for wave in WAVES:
            serve_concurrently(s, wave)
    return tmp_path


def test_the_server_records_each_requests_user_and_the_scheduler_trace_with_its_windows(tmp_path):
    out = _served(tmp_path)
    reqs = records(out, "requests.jsonl")
    assert sorted(r["user"] or "-" for r in reqs) == sorted(b.get("user", "-") for w in WAVES for b in w)
    assert all(r["check"]["passed"] for r in reqs)
    recs = ST.load(out / ST.SERVED)
    marks = [(r["window"], r["event"]) for r in recs if r["kind"] == "window"]
    wins = [w["window"] for w in records(out, "windows.jsonl") if not w.get("empty")]
    assert len(wins) >= len(WAVES) and all((w, "open") in marks and (w, "close") in marks for w in wins)
    assert {r["request"] for r in recs if r["kind"] == "arrive"} == {r["engine_id"] for r in reqs}
    assert records(out, "step_trace.jsonl")[0]["step_root"]          # the committer's trace is still its own file


def test_request_users_is_requests_jsonls_engine_id_and_user_alone_and_is_mirrored(tmp_path):
    out = _served(tmp_path)
    reqs, users = records(out, "requests.jsonl"), records(out, sc.REQUEST_USERS)
    assert len(users) == sum(map(len, WAVES)) and all(set(u) == {"engine_id", "user"} for u in users)
    assert [(u["engine_id"], u["user"]) for u in users] == [(r["engine_id"], r["user"]) for r in reqs]
    assert None in {u["user"] for u in users}                      # a body with no `user` is recorded as such, not dropped
    sc.Mirror(out, tmp_path / "OUT")()
    assert (tmp_path / "OUT" / sc.REQUEST_USERS).read_bytes() == (out / sc.REQUEST_USERS).read_bytes()


def test_each_commit_windows_advice_is_committed_and_compiles_to_the_steps_it_served(tmp_path):
    out = _served(tmp_path)
    rep = ADV.write(out, compiler=C)
    commits = {w["window"]: w for w in records(out, "windows.jsonl") if not w.get("empty")}
    assert rep["refused"] == {} and rep["open"] == [] and sorted(map(int, rep["windows"])) == sorted(commits)
    index = json.loads((out / "advice" / "windows.json").read_text())
    bodies = {b.get("user"): b for w in WAVES for b in w}
    reqs = {r["engine_id"]: r for r in records(out, "requests.jsonl")}
    split, _ = ADV.windows_of(ST.load(out / ST.SERVED))
    for w, entry in index["windows"].items():
        d = out / "advice" / f"w{w}"
        reg = json.loads((d / "registration.json").read_text())
        assert reg["window"] == int(w) and reg["commit"] == {"run_id": commits[int(w)]["run_id"], "run_root": commits[int(w)]["run_root"]}
        assert entry["root"] == reg["binding"]["advice"]["root"] and entry["registration"] == f"w{w}/registration.json"
        data = (d / "private" / "advice.bin").read_bytes()
        assert data.hex() not in (d / "registration.json").read_text()
        advice = I1.decode(data)
        ingress = {q.id: bodies.get(q.id) or {**bodies[None], "user": q.id} for q in advice.requests}
        c = compile_steps(REG, data, ingress)
        steps = ST.steps(split[int(w)])
        order = dict.fromkeys(row[0] for st in steps for row in st["rows"])
        index_of = {reqs[e]["user"] or reqs[e]["http_id"]: i for i, e in enumerate(order)}
        engine_of = {reqs[e]["user"] or reqs[e]["http_id"]: e for e in reqs}
        assert [q.id for q in advice.requests] == list(index_of)
        assert [[(r.request, r.start, r.n) for r in s] for s in c.steps] == \
            [[(index_of[reqs[e]["user"] or reqs[e]["http_id"]], start, n) for e, start, n in st["rows"]] for st in steps]
        for u in c.requests:
            body = ingress[u.id]
            assert (u.LP, u.T) == (len(body["prompt"]), body["max_tokens"] - 1)
            assert len(reqs[engine_of[u.id]]["streamed_ids"]) == body["max_tokens"]


def test_advice_leaves_a_window_still_open_for_a_later_call(tmp_path):
    out = _served(tmp_path)
    recs = ST.load(out / ST.SERVED)
    last = max(r["window"] for r in recs if r["kind"] == "window")
    cut = next(i for i, r in enumerate(recs) if r["kind"] == "window" and r["window"] == last and r["event"] == "close")
    (tmp_path / "mid").mkdir()
    (tmp_path / "mid" / ST.SERVED).write_text("".join(json.dumps(r) + "\n" for r in recs[:cut]))
    rep = ADV.write(tmp_path / "mid", tmp_path / "mid" / "advice", io=out / "requests.jsonl", compiler=C)
    assert rep["open"] == [last] and str(last) not in rep["windows"] and rep["windows"]


def test_a_request_split_across_commit_windows_is_carried_and_each_windows_advice_binds_rebuilds_and_compiles(tmp_path):
    out = _served(tmp_path, window_steps=3)
    rep = ADV.write(out, compiler=C)
    assert rep["refused"] == {} and rep["open"] == []
    reqs = {r["user"] or r["http_id"]: r for r in records(out, "requests.jsonl")}
    assert len(reqs["na-4"]["windows"]) > 1 and len(reqs["na-5"]["windows"]) > 1   # 6 outputs take 6 steps, past one 3-step window
    bodies = {b.get("user"): b for w in WAVES for b in w}
    last, placed, digests, carried_in = None, {}, {}, 0
    for w in sorted(map(int, rep["windows"])):
        d = out / "advice" / f"w{w}"
        data = (d / "private" / "advice.bin").read_bytes()
        a = WA.decode(data)
        assert WA.encode(a) == data
        salts = [bytes.fromhex(s) for s in json.loads((d / "private" / "opening.json").read_text())["salts"]]
        root = AW.commit_advice(C, data, salts).root.hex()
        assert json.loads((d / "registration.json").read_text())["binding"] == AW.binding(C, root, WA.total_bits(a))
        if last is not None:
            WA.follows(last, a)
        last, carried_in = a, carried_in + len(a.carried)
        c = compile_steps(REG, data, {q.id: bodies.get(q.id) or {**bodies[None], "user": q.id} for q in a.requests})
        assert c.summary()["carry"]["in"] == list(range(len(a.carried))) and c.idle == ()
        for u in c.requests:
            digests.setdefault(u.id, set()).add(u.digest)
        for rows in c.steps:
            for r in rows:
                placed.setdefault(a.requests[r.request].id, []).append((r.start, r.n, r.samples))
    assert carried_in >= 2
    for user, rows in placed.items():
        body = bodies.get(user) or bodies[None]
        LP, T = len(body["prompt"]), body["max_tokens"] - 1
        assert len(digests[user]) == 1                   # one Program, whichever window a row runs in
        assert [p for start, n, _ in sorted(rows) for p in range(start, start + n)] == list(range(LP + T))
        assert sum(k for *_, k in rows) == body["max_tokens"]


def _split(tmp_path: Path) -> tuple[Path, list[dict], str, list[int]]:
    out = _served(tmp_path, window_steps=3)
    req = next(r for r in records(out, "requests.jsonl") if r["user"] == "na-4")
    return out, ST.load(out / ST.SERVED), req["engine_id"], req["windows"]


def _rows_in(recs: list[dict], engine_id: str, window: int) -> list[list]:
    lo, hi = (next(r["before_step"] for r in recs if r["kind"] == "window" and r["window"] == window and r["event"] == e)
              for e in ("open", "close"))
    return [row for st in sorted((r for r in recs if r["kind"] == "step"), key=lambda r: r["step"]) if lo <= st["step"] < hi
            for row in st["rows"] if row[0] == engine_id]


def _rewritten(tmp_path: Path, out: Path, recs: list[dict]) -> dict:
    (tmp_path / "edited").mkdir()
    (tmp_path / "edited" / ST.SERVED).write_text("".join(json.dumps(r) + "\n" for r in recs))
    return ADV.write(tmp_path / "edited", tmp_path / "edited" / "advice", io=out / "requests.jsonl", compiler=C)


def test_a_carried_request_whose_next_window_row_isnt_at_its_carried_position_is_refused(tmp_path):
    out, recs, e, (k1, k2, *_) = _split(tmp_path)
    row = _rows_in(recs, e, k2)[0]
    at, row[1] = row[1], row[1] + 1
    rep = _rewritten(tmp_path, out, recs)
    assert str(k1) in rep["windows"] and str(k2) not in rep["windows"]
    assert f"request {e}'s row starts at {at + 1}, not where its last row ended ({at})" in rep["refused"][str(k2)]


def test_a_request_the_last_window_didnt_carry_is_refused_where_it_starts_past_its_cache_hit(tmp_path):
    out, recs, e, (k1, k2, *_) = _split(tmp_path)
    _rows_in(recs, e, k1)[0][1] = 1                       # the window that admits it is refused, so nothing carries it on
    rep = _rewritten(tmp_path, out, recs)
    assert str(k1) in rep["refused"] and str(k2) in rep["refused"]
    m = re.fullmatch(rf"step \d+: request (\S+) is first scheduled at position (\d+), not at its cache hit 0 \(window {k1} before it "
                     rf"is refused, so it opens on nothing carried\)", rep["refused"][str(k2)])
    spanning = {r["engine_id"] for r in records(out, "requests.jsonl") if {k1, k2} <= set(r["windows"])}
    assert m and m[1] in spanning and int(m[2]) == _rows_in(recs, m[1], k2)[0][1] > 0   # na-4, or na-5 if it runs first in k2


def test_a_window_whose_request_has_no_record_yet_is_pending_while_the_server_runs_and_refused_once_it_has_stopped(tmp_path):
    """The server writes a request's Commit record at the close of the window it finishes in, after the closes before it, so
    while it serves, the window na-4 is first scheduled in (3-step windows; it runs on into the next) can have no record of it.
    That window's advice is pending, not refused: nothing is written for it, ``main`` exits `PENDING`, and a later call takes
    it, as does the next window's, which opens on what it leaves. Once summary.json says the server stopped with every record
    written (``final``), the same missing record is refused, as it was before."""
    out, _, e, (k1, k2, *_) = _split(tmp_path)
    mid = tmp_path / "mid"
    mid.mkdir()
    for f in (ST.SERVED, "windows.jsonl"):
        (mid / f).write_text((out / f).read_text())
    (mid / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records(out, "requests.jsonl") if r["engine_id"] != e))
    why = f"request {e} of window {k1} has no Commit record yet, and the server is still running"
    for final in (None, False):
        if final is not None:
            (mid / "summary.json").write_text(json.dumps({"final": final}))
        rep = ADV.write(mid, compiler=C, window=k1)
        assert rep["pending"] == {str(k1): why} and rep["reports"] == {} and not (mid / "advice").exists()
        assert ADV.write(mid, compiler=C, window=k2)["pending"] == {str(k2): f"{why} (window {k2} opens on what window {k1} leaves)"}
        opts = ADV.Options(run=str(mid), window=k1, compiler_name="test-compiler", compiler_digest="ab" * 64)
        assert ADV.main(opts) == ADV.PENDING and not (mid / "advice").exists()
    (mid / "summary.json").write_text(json.dumps({"final": True}))
    rep = ADV.write(mid, compiler=C, window=k1)
    assert rep["pending"] == {} and rep["refused"] == {str(k1): f"request {e} is in the trace but not in the Commit record"}
    assert json.loads((mid / "advice" / "windows.json").read_text())["refused"] == rep["refused"]
    (mid / "requests.jsonl").write_text((out / "requests.jsonl").read_text())
    (mid / "summary.json").unlink()
    rep = ADV.write(mid, compiler=C, window=k1)
    assert rep["pending"] == {} and rep["refused"] == {} and list(rep["windows"]) == [str(k1)]


GONE, OTHER, K_GONE = _body(6, 9, 12, "na-ab"), _body(7, 5, 9, "na-7"), 5


def _aborted(tmp_path: Path) -> tuple[Server, dict]:
    """na-ab's client goes away after 5 of its 12 outputs, in 3-step windows beside na-7; the server and na-ab's record."""
    with Server(tmp_path, window_steps=3, abort_after={tuple(GONE["prompt"]): K_GONE}) as s:
        serve_concurrently(s, [GONE, OTHER])
    return s, next(r for r in records(tmp_path, "requests.jsonl") if r["user"] == "na-ab")


def test_a_request_its_client_aborts_passes_as_an_abort_after_its_k_streamed_ids_and_fails_if_they_differ(tmp_path):
    """The Commit record's check of an aborted request: its finish is ``abort`` with the k ids streamed before it, and it passes,
    since those are the committed ids' prefix; the same record against a stream that differs at one id, or that says it was
    aborted after more ids than it streamed, fails by name."""
    s, q = _aborted(tmp_path)
    assert q["finish"] == {"reason": "abort", "stop_reason": None, "k": K_GONE, "by": "client"} and q["check"]["passed"], q["check"]
    assert q["check"]["streamed_ids"] == K_GONE and q["check"]["problems"] == []
    assert s.windows.summary()["passed"]

    def check(streamed: list[int], k: int = K_GONE) -> dict:
        return s.windows.check(sc.Ledger(q["http_id"], q["engine_id"], q["prompt_ids"], q["ingress"], q["user"], streamed=streamed,
                                         finish={"reason": "abort", "stop_reason": None, "k": k}, finished=True))

    assert check(list(q["streamed_ids"]))["passed"]
    bad = list(q["streamed_ids"])
    bad[2] += 1
    got = check(bad)
    P = len(q["prompt_ids"])
    assert not got["passed"] and any(p.startswith(f"samples: 1 position(s) differ, first {P + 1}: ") for p in got["problems"])
    short = check(list(q["streamed_ids"])[:-1], k=K_GONE)
    assert not short["passed"] and f"finish: aborted after {K_GONE} ids, {K_GONE - 1} streamed" in short["problems"]
    cut = check(list(q["streamed_ids"])[:3], k=3)       # two committed outputs it never streamed: more than a step in flight's
    assert not cut["passed"] and cut["problems"] == [
        "finish: aborted after 3 ids, and 2 committed samples run past them, more than the one of the step in flight"]


def test_a_request_running_when_the_server_stops_is_an_abort_by_the_server_and_passes_as_its_streamed_prefix(tmp_path):
    """The server stops while na-ab streams, after 5 of its 12 outputs: the loop's exit ends it, so its record's abort is the
    server's, not its client's, and its 5 streamed ids are still the committed ones' prefix."""
    with Server(tmp_path, abort_after={tuple(GONE["prompt"]): K_GONE}) as s:
        s.engine.client_goes = lambda engine_id: s.loop.stop()
        serve_concurrently(s, [GONE])
    (q,) = records(tmp_path, "requests.jsonl")
    assert q["finish"] == {"reason": "abort", "stop_reason": None, "k": K_GONE, "by": "server"}
    assert q["streamed_ids"] and len(q["streamed_ids"]) == K_GONE and q["check"]["passed"], q["check"]
    with pytest.raises(ValueError, match="an abort is by the client or the server, not 'proxy'"):
        s.windows.aborted(q["engine_id"], "proxy")


def test_a_request_its_client_aborts_is_freed_after_its_step_and_every_window_it_ran_in_says_so_and_compiles(tmp_path):
    """The client of na-ab goes away after 5 of its 12 outputs: the loop aborts it before the next step, the trace records the
    abort after the step that emitted the 5th, and no row of it follows. Its prompt and 4 decodes take 5 steps, so 3-step windows
    split it: the window that leaves it carries it on with its record's finish, the abort, and each window
    that holds it is version 2, follows the last, and compiles, its Program the one of a request that kept 5 outputs."""
    gone, other, k = GONE, OTHER, K_GONE
    _, q = _aborted(tmp_path)
    recs = ST.load(tmp_path / ST.SERVED)
    (fin,) = [r for r in recs if r["kind"] == "finish" and r["request"] == q["engine_id"]]
    rows = [st["step"] for st in recs if st["kind"] == "step" for row in st["rows"] if row[0] == q["engine_id"]]
    assert len(q["streamed_ids"]) == k and len(q["windows"]) > 1
    assert (fin["reason"], fin["outside"], fin["after_step"]) == ("abort", True, max(rows)) and len(rows) == k
    rep = ADV.write(tmp_path, compiler=C)
    assert rep["refused"] == {} and rep["open"] == []
    bodies, last, placed, held = {"na-ab": gone, "na-7": other}, None, [], []
    for w in sorted(map(int, rep["windows"])):
        data = (tmp_path / "advice" / f"w{w}" / "private" / "advice.bin").read_bytes()
        a = WA.decode(data)
        if last is not None:
            WA.follows(last, a)
        last = a
        if "na-ab" in (ids := [x.id for x in a.requests]):
            held.append(w)
            assert a.version == WA.ABORTS and a.requests[ids.index("na-ab")] == WA.RequestAdvice("na-ab", 9, k, "abort")
        c = compile_steps(REG, data, {x.id: bodies[x.id] for x in a.requests})
        assert all((u.LP, u.T) == (9, k - 1) for u in c.requests if u.id == "na-ab")
        placed += [(r.start, r.n, r.samples) for rows_ in c.steps for r in rows_ if a.requests[r.request].id == "na-ab"]
    assert held == q["windows"]
    assert [p for start, n, _ in sorted(placed) for p in range(start, start + n)] == list(range(9 + k - 1))
    assert sum(x for *_, x in placed) == k


QUEUED = _body(8, 7, 4, "na-q")


def test_a_request_its_client_aborts_before_it_is_ever_scheduled_is_in_no_windows_advice_and_passes_as_an_abort_after_0(tmp_path):
    """na-q's client goes away as the engine takes it, so the loop aborts it before the next step: the trace holds its arrival and
    the abort and no row of it, and its Commit record, an abort after 0 ids in no window, passes. Every window's advice is written
    without it, follows the last and compiles; so does the advice of the whole trace taken as one window, which holds its arrival
    and drops it. The second wave closes a window after the abort, whenever na-q came."""
    with Server(tmp_path, window_steps=3, abort_after={tuple(QUEUED["prompt"]): 0}) as s:
        serve_concurrently(s, [OTHER, QUEUED])
        serve_concurrently(s, WAVES[1])
    reqs = {r["user"]: r for r in records(tmp_path, "requests.jsonl")}
    q, e = reqs["na-q"], reqs["na-q"]["engine_id"]
    recs = ST.load(tmp_path / ST.SERVED)
    assert [(r["kind"], r.get("reason")) for r in recs if r.get("request") == e] == [("arrive", None), ("finish", "abort")]
    assert not any(row[0] == e for r in recs if r["kind"] == "step" for row in r["rows"])
    assert (q["finish"]["reason"], q["finish"]["k"], q["streamed_ids"], q["windows"]) == ("abort", 0, [], [])
    assert q["check"]["passed"] and s.windows.summary()["passed"], q["check"]
    ran = s.windows.check(sc.Ledger(q["http_id"], e, q["prompt_ids"], q["ingress"], q["user"], streamed=[],
                                    finish={"reason": "length", "stop_reason": None}, finished=True))
    assert ran["problems"] == ["no committed row"]
    rep = ADV.write(tmp_path, compiler=C)
    assert rep["refused"] == {} and rep["open"] == [] and all(r.get("request") != e for w in ADV.windows_of(recs)[0].values() for r in w)
    bodies, last = {b["user"]: b for b in (OTHER, QUEUED, *WAVES[1])}, None
    for w in sorted(map(int, rep["windows"])):
        data = (tmp_path / "advice" / f"w{w}" / "private" / "advice.bin").read_bytes()
        a = WA.decode(data)
        if last is not None:
            WA.follows(last, a)
        last = a
        assert "na-q" not in {x.id for x in a.requests}
        compile_steps(REG, data, {x.id: bodies[x.id] for x in a.requests})
    whole = ADV.from_trace(recs, ADV.io_record(tmp_path / "requests.jsonl"), carry=())
    assert whole.dropped == (e,) and sorted(x.id for x in whole.advice.requests) == ["na-4", "na-5", "na-7"]
    assert len(whole.advice.release) == sum(len(r["streamed_ids"]) for r in reqs.values())
    compile_steps(REG, WA.encode(whole.advice), {x.id: bodies[x.id] for x in whole.advice.requests})
