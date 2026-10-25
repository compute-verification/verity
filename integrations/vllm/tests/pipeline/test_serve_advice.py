"""`verity-vllm serve-openai` records what `verity-vllm advice` reads, and the advice of each of its Commit windows compiles back to
the steps it served: a stand-in server (its HTTP side and engine loop as `main` runs them, the scheduler `tests.engine.fake_scheduler`
under the committer stand-in of `test_serve_openai`) serves two waves of requests, the engine idle between them, so in at least
two Commit windows; requests.jsonl
carries each body's `user`, advice_trace.jsonl the scheduler's steps and each window's open and close, and each window's advice,
committed under w<k>/, compiles (`predict.steps.compile_steps`) to the rows the trace recorded, each request at the shape it ran."""
from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from tests.engine.fake_scheduler import FakeRequest, FakeScheduler, runner_order
from tests.pipeline.test_serve_openai import Committer, Runner, Tokenizer, records, serve_concurrently
from verity.core.protocols.certifier import advice as I1
from verity_vllm.engine import step_trace as ST
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_protocol as proto
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered, compile_steps

C = ADV.registered_compiler("test-compiler", "ab" * 64)
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
FINISH = {0: "stop", 1: "length"}


class SchedEngine:
    """vLLM's ``LLMEngine`` as the loop drives it, its decisions the stand-in scheduler's: each ``step`` schedules (whole prompts at
    a request's first step, else one decode), runs the scheduled rows through the runner in Model Runner V2's order under vLLM's
    internal ids ``<engine id>-<8 hex>``, and returns each emitted token as a DELTA output under the engine id."""

    def __init__(self, runner: Runner) -> None:
        self.runner, self.sched, self.n, self.slot, self.engine_id = runner, FakeScheduler(budget=256, max_seqs=4), 0, {}, {}

    def add_request(self, request_id: str, prompt: dict, params: dict) -> None:
        self.n += 1
        rid, ids = f"{request_id}-{self.n:08x}", list(prompt["prompt_token_ids"])
        self.slot[rid], self.engine_id[rid] = self.n % 64, request_id
        self.runner.sampler.sampling_states.seeds.gpu[self.n % 64] = params["seed"]
        outputs = [3 + (sum(ids) + 7 * k) % 200 for k in range(params["max_tokens"])]
        self.sched.add_request(FakeRequest(rid, ids, outputs, params["max_tokens"], sampling_params=SimpleNamespace(**params)))

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
        return outs


class Server:
    """`serve_openai.main`'s wiring over the stand-ins: the advice trace on the scheduler for the server's lifetime, the Commit's
    windows marking it."""

    def __init__(self, out: Path) -> None:
        self.runner = Runner()
        self.engine = SchedEngine(self.runner)
        self.trace = sc.advice_trace(out, self.engine.sched)
        self.windows = sc.CommitWindows(Committer(self.runner), out, run_id="t", window_steps=64, eos_ids=[Tokenizer.eos_token_id],
                                        clock=lambda: 0.0, wall=lambda: 0.0, marks=self.trace.window)
        self.loop = so.EngineLoop(self.engine, self.windows)
        self.server = so.Server(proto=proto.PlainProtocol(True), served_name="m", max_model_len=512, tokenizer=Tokenizer(), defaults={},
                                submit=self.loop.submit, abort=self.loop.abort, clock=lambda: 7)
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


def _served(tmp_path: Path) -> Path:
    with Server(tmp_path) as s:
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


def test_each_commit_windows_advice_is_committed_and_compiles_to_the_steps_it_served(tmp_path):
    out = _served(tmp_path)
    rep = ADV.write(out, compiler=C)
    commits = {w["window"]: w for w in records(out, "windows.jsonl") if not w.get("empty")}
    assert rep["refused"] == {} and rep["open"] == [] and sorted(map(int, rep["windows"])) == sorted(commits)
    index = json.loads((out / "advice" / "windows.json").read_text())
    reg_ = Registered.of(Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY},
                             target={"compute_capability": [12, 0], "num_sms": 188}), default_seed=7)
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
        c = compile_steps(reg_, data, ingress)
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
