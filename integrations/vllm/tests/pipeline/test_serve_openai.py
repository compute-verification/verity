"""`verity-vllm serve-openai` on CPU: request parsing and SSE framing as vLLM's OpenAI server does them, and, over a stand-in
engine and committer, the HTTP stream, the HTTP id -> engine id -> runner id map and the check of committed ids against
streamed ones."""
from __future__ import annotations

import collections
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from verity.core.protocols.certifier import active as A, compile_slice as CS, io_tie as IT, release as REL, served_rows as SRW
from verity_vllm.commit import served_record as SR
from verity_vllm.engine.served_seed import DEFAULT_SEED
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_protocol as proto
from verity_vllm.pipeline import serve_seal as ss
from verity_vllm.pipeline.serve_admit import Admission

CERTIFIER_BODY = {"model": "m", "prompt": [128000, 791, 4062], "max_tokens": 4, "min_tokens": 4, "stream": True,
                  "ignore_eos": True, "temperature": 0}


# ---------------------------------------------------------------------------------------------------------------- protocol

def test_certifier_body_parses_to_its_sampling():
    p = proto.PlainProtocol()
    req = p.parse(dict(CERTIFIER_BODY, seed=0))
    assert (req.prompt, req.stream, req.max_tokens, req.include_usage) == ([128000, 791, 4062], True, 4, False)
    sp = p.sampling_params(req, p.max_tokens(req, 512, 3, {"temperature": 0.6}), {"temperature": 0.6})
    assert sp == {"max_tokens": 4, "min_tokens": 4, "ignore_eos": True, "temperature": 0, "top_p": 1.0, "top_k": 0, "seed": 0,
                  "output_kind": "delta"}


def test_text_prompt_and_defaults():
    p = proto.PlainProtocol()
    req = p.parse({"prompt": ["hello"], "stream": False})
    assert (req.prompt, req.stream, req.max_tokens, req.add_special_tokens) == ("hello", False, 16, True)
    assert p.sampling_params(req, 16, {"temperature": 0.6})["temperature"] == 0.6
    assert p.sampling_params(req, 16, {})["output_kind"] == "final_only"


def test_max_tokens_as_vllm_bounds_it():
    p = proto.PlainProtocol()
    assert p.max_tokens(p.parse({"prompt": [1] * 500, "max_tokens": 64}), 512, 500, {}) == 12
    assert p.max_tokens(p.parse({"prompt": [1], "max_tokens": None}), 512, 1, {"max_tokens": 7}) == 7
    with pytest.raises(ValueError, match="exceeds model's maximum context length"):
        p.max_tokens(p.parse({"prompt": [1] * 600}), 512, 600, {})


@pytest.mark.parametrize("body", [{"prompt": []}, {"prompt": [1, -2]}, {"prompt": [[1], [2]]}, {"prompt": 3}, {"prompt": [1], "n": 2},
                                  {"prompt": [1], "echo": True}, {"prompt": [1], "logprobs": 1},
                                  {"prompt": [1], "stream_options": {"include_usage": True}}])
def test_refused_bodies(body):
    with pytest.raises(proto.RequestError):
        proto.PlainProtocol().parse(body)


def test_usage_rule_is_vllms():
    assert proto.should_include_usage(None, True) == (True, True)
    assert proto.should_include_usage(None, False) == (False, False)
    assert proto.should_include_usage({"include_usage": True}, False) == (True, False)
    assert proto.should_include_usage({"include_usage": True, "continuous_usage_stats": True}, False) == (True, True)
    assert proto.should_include_usage({"include_usage": False, "continuous_usage_stats": True}, False) == (False, False)


def test_frames_are_vllms_serialization():
    """vLLM's `completion_stream_generator` at d9105ea80: a choice chunk is `model_dump_json(exclude_unset=True)` of a
    CompletionStreamResponse whose choice sets every field it names (so its nulls stay), the usage chunk
    `exclude_unset=False, exclude_none=True` with `choices=[]`; UsageInfo's field order is prompt, total, completion."""
    p = proto.PlainProtocol()
    base = dict(request_id="cmpl-1-0", created=7, model="m")
    assert p.chunk(**base, text=" é", finish_reason=None, stop_reason=None, prompt_token_ids=None, token_ids=None, usage=None) == (
        '{"id":"cmpl-1-0","object":"text_completion","created":7,"model":"m","choices":[{"index":0,"text":" é","logprobs":null,'
        '"finish_reason":null,"stop_reason":null,"prompt_token_ids":null,"token_ids":null}]}')
    assert p.chunk(**base, text="x", finish_reason="length", stop_reason=None, prompt_token_ids=None, token_ids=[5],
                   usage=proto.usage_of(3, 4)).endswith('"token_ids":[5]}],"usage":{"prompt_tokens":3,"total_tokens":7,"completion_tokens":4}}')
    assert p.usage_chunk(**base, usage=proto.usage_of(128, 16)) == (
        '{"id":"cmpl-1-0","object":"text_completion","created":7,"model":"m","choices":[],'
        '"usage":{"prompt_tokens":128,"total_tokens":144,"completion_tokens":16}}')
    assert proto.sse("{}") == b"data: {}\n\n" and proto.SSE_DONE == b"data: [DONE]\n\n"
    assert proto.http_chunk(b"data: x\n\n") == b"9\r\ndata: x\n\n\r\n"


def test_plain_frames_agree_with_vllm():
    """Where vLLM is installed (the GPU nodes' env), the plain frames are its frames."""
    pytest.importorskip("vllm.entrypoints.openai.completion.protocol")
    got = proto.framing_agrees(proto.VllmProtocol())
    assert got["agrees"], got["differences"]


#: a two-token id-form stream at P = 140, every byte: each event 138 bytes and two spaces, then the end
TWO_IDS_AT_140 = (
    b'data: {"object":"chat.completion.chunk","created":0,"model":"verity","choices":[{"index":0,"token_id":      4853,'
    b'"finish_reason":null}]}  \n\n'
    b'data: {"object":"chat.completion.chunk","created":0,"model":"verity","choices":[{"index":0,"token_id":      4884,'
    b'"finish_reason":null}]}  \n\n'
    b"data: [DONE]\n\n")


def test_ids_frames_are_the_scans_id_events():
    """`IdsProtocol`: each id its own `io_tie.id_event` of P bytes, however many ids a delta carries; the stream's end
    is the scan's trailer; no usage, and a stream returns its ids, whatever the request asks; P below an event's 138
    bytes is refused."""
    p = proto.IdsProtocol(force_include_usage=True, payload=140, base=proto.PlainProtocol(True))
    assert (p.name, p.http, proto.PlainProtocol.http) == ("ids", False, True)
    assert p.events(token_ids=[4853, 4884], text="ab", finish_reason=None) + proto.SSE_DONE == TWO_IDS_AT_140
    assert proto.SSE_DONE == IT.TRAILER and p.events(token_ids=[]) == b""
    req = p.parse(dict(CERTIFIER_BODY, stream_options={"include_usage": True, "continuous_usage_stats": True}))
    assert (req.include_usage, req.continuous_usage, req.return_token_ids) == (False, False, True)
    assert not p.parse(dict(CERTIFIER_BODY, stream=False)).return_token_ids
    assert proto.PROTOCOLS["ids"] is proto.IdsProtocol and proto.ID_PAYLOAD == 512 - A.HEADER.size
    with pytest.raises(ValueError, match="at least 138"):
        proto.IdsProtocol(payload=137, base=proto.PlainProtocol())


def test_command_is_registered_and_defaults_to_the_setting_that_fits_the_certifiers_plan():
    opts = cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--out", "/tmp/x", "--no-commit", "--port", "0"], environ={})
    assert (opts.port, opts.commit, opts.precision, opts.gpu_tree, opts.window_steps) == (0, False, "verity_int8", True, 64)
    assert (opts.scheme, opts.of_record) == ("rows-sha512", None)
    assert (opts.protocol, opts.run_id, opts.hf_home) == ("vllm", "local", None)
    assert so.engine_options(opts)["quantization"] == "verity_int8" and "attention_backend" not in so.engine_options(opts)
    exact = cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--precision", "int8_exact", "--no-gpu-tree"], environ={})
    assert so.engine_options(exact)["attention_backend"] == "CUSTOM" and not exact.gpu_tree
    assert "enable_prefix_caching" not in so.engine_options(opts)
    cached = cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--prefix-caching"], environ={})
    assert so.engine_options(cached)["enable_prefix_caching"] is True
    env = cli.parse("serve-openai", ["--model", "m", "--revision", "r"], environ={"RESEARCH_RUN_ID": "r1", "HF_HOME": "/h"})
    assert (env.run_id, env.hf_home) == ("r1", "/h")
    with pytest.raises(SystemExit):
        cli.parse("serve-openai", ["--out", "/tmp/x"], environ={})


@pytest.mark.parametrize("precision, parts, scope", [("verity_int8", ("linears",), "linears"),
                                                     ("int8_exact", ("linears", "attention"), "all"), ("pin", ("linears",), None)])
def test_every_precision_registers_the_verity_int8_method_before_the_engine(precision, parts, scope):
    got: list = []
    assert so.register_int8(precision, got.append) == scope
    assert got == [parts] and "linears" in got[0]
    assert set(so.INT8_PARTS) == set(so.PRECISIONS)


def test_preflight_refuses_a_gpu_tree_without_ninja_before_the_model_loads(tmp_path):
    (man := tmp_path / "manifest.json").write_text("{}")
    opts = cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--required-manifest", str(man)], environ={})
    with pytest.raises(cli.UsageError, match=r"ninja is\) on PATH$"):
        so.preflight(opts, which=lambda _: None)
    so.preflight(opts, which=lambda _: "/v/bin/ninja")
    so.preflight(cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--no-commit"], environ={}), which=lambda _: None)
    with pytest.raises(cli.UsageError, match="window-steps"):
        so.preflight(cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--window-steps", "0"], environ={}),
                     which=lambda _: "/v/bin/ninja")


def test_the_served_commit_is_rows_sha512_unless_a_recorded_sha256_run_is_served_again(tmp_path):
    """rows-sha512 is the default: a committing run that names no scheme is refused before the model loads until it
    names its bench row's manifest, and is never served under SHA-256. vllm-v1 is served only by name, to serve again a recorded
    vllm-v1 run of the same model (``--of-record``: its serve.json says vllm-v1, or predates the field); the committer refuses
    it too, before it builds anything."""
    def parse(*more: str) -> so.Options:
        return cli.parse("serve-openai", ["--model", "m", "--revision", "r", *more], environ={})

    def refusal(opts: so.Options, which: Any = lambda _: "/v/bin/ninja") -> str | None:
        try:
            so.preflight(opts, which=which)
        except cli.UsageError as e:
            return str(e)
        return None

    def record(name: str, **rec: Any) -> str:
        (d := tmp_path / name).mkdir()
        (d / "serve.json").write_text(json.dumps({"model": "m", "revision": "r", "commit": True, **rec}))
        return str(d)

    (man := tmp_path / "manifest.json").write_text("{}")
    assert parse().scheme == ss.ROWS == ss.SCHEMES[0] == "rows-sha512" and ss.V1 in ss.SCHEMES
    assert "(the default)" in refusal(parse()) and "--required-manifest ROW_DIR/manifest.json" in refusal(parse())
    assert "drop --no-gpu-tree" in refusal(parse("--no-gpu-tree", "--required-manifest", str(man)))
    assert refusal(parse("--required-manifest", str(man))) is None and refusal(parse("--no-commit")) is None
    with pytest.raises(cli.UsageError, match="--required-manifest ROW_DIR/manifest.json"):
        ss.committer(parse(), llm=None)

    v1, old = ("--scheme", "vllm-v1"), record("v1", scheme="vllm-v1")
    assert "commits SHA-256" in refusal(parse(*v1)) and "--of-record RUN_DIR" in refusal(parse(*v1))
    with pytest.raises(cli.UsageError, match="commits SHA-256"):
        ss.committer(parse(*v1), llm=None)
    assert refusal(parse(*v1, "--of-record", old)) is None
    assert refusal(parse(*v1, "--of-record", record("before-serve-json-named-it"))) is None
    assert refusal(parse(*v1, "--of-record", old, "--no-gpu-tree"), which=lambda _: None) is None
    assert refusal(parse(*v1, "--of-record", old), which=lambda _: None).endswith("or pass --no-gpu-tree")
    assert "committed under rows-sha512, not under vllm-v1" in refusal(parse(*v1, "--of-record", record("rows", scheme="rows-sha512")))
    assert "committed nothing" in refusal(parse(*v1, "--of-record", record("uncommitted", commit=False, scheme=None)))
    assert "served m2@r, not m@r" in refusal(parse(*v1, "--of-record", record("other", model="m2", scheme="vllm-v1")))
    assert "not a recorded serve-openai run" in refusal(parse(*v1, "--of-record", str(tmp_path / "none")))
    assert "not under --scheme rows-sha512" in refusal(parse("--required-manifest", str(man), "--of-record", old))


def test_retain_exclude_takes_the_attention_interiors_only_under_the_rows_sha512_commit(tmp_path):
    """``--retain-exclude`` (default off, and $VERITY_RETAIN_EXCLUDE's value when set) names the attention interiors only, the
    values no reader of a sealed window opens; anything else, and a run with no rows-sha512 Commit, is refused before the model
    loads, and by the committer before it builds anything."""
    (man := tmp_path / "manifest.json").write_text("{}")
    rows = ("--model", "m", "--revision", "r", "--required-manifest", str(man))

    def refusal(*more: str, environ: dict[str, str] | None = None) -> str | None:
        try:
            so.preflight(cli.parse("serve-openai", [*rows, *more], environ=environ or {}), which=lambda _: "/v/bin/ninja")
        except cli.UsageError as e:
            return str(e)
        return None

    assert cli.parse("serve-openai", list(rows), environ={}).retention.retain_exclude == ""
    assert cli.parse("serve-openai", list(rows), environ={"VERITY_RETAIN_EXCLUDE": "hidden_m1"}).retention.retain_exclude == "hidden_m1"
    for ok in ("hidden_m1", "hidden", "fa3_hidden_m1", "fa2_hidden_m1_stream,fa3_hidden_m1"):
        assert refusal("--retain-exclude", ok) is None, ok
    assert "['gemm_qkv'] aren't attention interiors" in refusal("--retain-exclude", "hidden_m1,gemm_qkv")
    assert "aren't attention interiors" in refusal(environ={"VERITY_RETAIN_EXCLUDE": "weights"})
    assert "needs the Commit under --scheme rows-sha512" in refusal("--retain-exclude", "hidden_m1", "--no-commit")
    v1 = ("--scheme", "vllm-v1", "--of-record", str(tmp_path))
    assert "needs the Commit under --scheme rows-sha512" in refusal("--retain-exclude", "hidden_m1", *v1)
    with pytest.raises(cli.UsageError, match="aren't attention interiors"):
        ss.committer(cli.parse("serve-openai", [*rows, "--retain-exclude", "weights"], environ={}), llm=None)


# ---------------------------------------------------------------------------------------------------------------- stand-ins

class Runner:
    """Model Runner V2 as the committer reads it: ``execute_model`` runs a step's input batch, ``sample`` returns its sampled
    ids, and ``sampler.sampling_states.seeds`` holds each request's seed in its state slot."""

    def __init__(self) -> None:
        self.sampler = SimpleNamespace(sampling_states=SimpleNamespace(seeds=SimpleNamespace(gpu=torch.zeros(64, dtype=torch.int64))))

    def execute_model(self, sched: Any) -> None:
        pass

    def sample(self, ib: Any, sampled: list[int]) -> list[int]:
        return sampled


class Engine:
    """vLLM's ``LLMEngine`` as the loop drives it: each ``step`` is one forward over every live request (its whole prompt
    at its first step, else its last sampled token), whose sample is a function of its inputs; outputs are DELTA, one
    token each, under the engine id, while the runner sees vLLM's internal id ``<id>-<8 hex>``. Each request's seed goes to
    its state slot (``seed_override``: a seed other than the one admitted). With ``block``, a prefix cache: a request's first
    row starts after the longest whole-block prefix (at most its prompt but one) of the ids a finished request computed."""

    def __init__(self, runner: Runner, seed_override: int | None = None, block: int | None = None) -> None:
        self.runner, self.reqs, self.n, self.seed_override, self.block, self.computed = runner, {}, 0, seed_override, block, []

    def cached(self, prompt: list[int]) -> int:
        if not self.block:
            return 0
        common = max((next((k for k, (a, b) in enumerate(zip(c, prompt)) if a != b), min(len(c), len(prompt))) for c in self.computed),
                     default=0)
        return min(common, len(prompt) - 1) // self.block * self.block

    def add_request(self, request_id: str, prompt: dict, params: Any) -> None:
        self.n += 1
        slot = self.n % 64
        self.runner.sampler.sampling_states.seeds.gpu[slot] = params["seed"] if self.seed_override is None else self.seed_override
        ids = list(prompt["prompt_token_ids"])
        self.reqs[request_id] = {"internal": f"{request_id}-{self.n:08x}", "prompt": ids, "out": [],
                                 "max": params["max_tokens"], "slot": slot, "hit": self.cached(ids)}

    def abort_request(self, ids: list[str]) -> None:
        for i in ids:
            self.reqs.pop(i, None)

    def has_unfinished_requests(self) -> bool:
        return bool(self.reqs)

    def step(self) -> list[Any]:
        live = list(self.reqs.items())
        ids, qsl, pos, starts = [], [0], [], []
        for _, r in live:
            P = len(r["prompt"])
            rows, start = (r["prompt"][r["hit"]:], r["hit"]) if not r["out"] else (r["out"][-1:], P + len(r["out"]) - 1)
            ids += rows
            qsl.append(len(ids))
            pos += range(start, start + len(rows))
            starts.append(start)
        ib = SimpleNamespace(num_reqs=len(live), req_ids=[r["internal"] for _, r in live], query_start_loc_np=np.array(qsl),
                             idx_mapping=torch.tensor([r["slot"] for _, r in live], dtype=torch.int32), positions=torch.tensor(pos),
                             logits_indices=torch.tensor(qsl[1:]) - 1, num_computed_tokens_np=np.array(starts),
                             prefill_len_np=np.array([len(r["prompt"]) for _, r in live]))
        self.runner.execute_model(SimpleNamespace(ib=ib, ids=ids))
        sampled = [(sum(ids[qsl[i]:qsl[i + 1]]) + 31 * len(r["out"])) % 32000 for i, (_, r) in enumerate(live)]
        self.runner.sample(ib, sampled)
        outs = []
        for (eid, r), t in zip(live, sampled):
            r["out"].append(t)
            done = len(r["out"]) >= r["max"]
            outs.append(SimpleNamespace(request_id=eid, finished=done, prompt_token_ids=r["prompt"],
                                        outputs=[SimpleNamespace(token_ids=[t], text=f" w{t}", finish_reason="length" if done else None,
                                                                 stop_reason=None)]))
            if done:
                self.computed.append(r["prompt"] + r["out"][:-1])
                del self.reqs[eid]
        return outs


class Committer:
    """`NativeHostCommitter` as `CommitWindows` uses it: its patches of the runner begin a step at ``execute_model`` and
    commit the embedding's input ids there, and the sampled ids at ``sample``, where its sources bind the sample's inputs
    (`served_record.SamplerInputsSource`), all host-retained; ``flip`` changes one committed sampled id, at (step, row) of
    the first window that reaches that step."""

    def __init__(self, runner: Runner, flip: tuple[int, int] | None = None) -> None:
        self.runner, self.flip, self.sources, self.errors = runner, flip, [], []
        self.attach_source(SR.SamplerInputsSource(self))
        self.reset()

    def attach_source(self, src: Any) -> None:
        self.sources.append(src)

    def reset(self) -> None:
        self._layouts, self._step_index = {}, 0

    def _commit(self, step: int, name: str, values: list[int], dtype: str) -> None:
        raw = np.asarray(values, dtype=dtype).tobytes()
        self._layouts.setdefault(step, []).append(
            SimpleNamespace(name=name, dtype=dtype, nbytes=len(raw), host=memoryview(raw), dev_off=-1, stream_off=-1))

    def _acquire(self, name: str, t: Any, job: Any) -> None:
        self._commit(job.step, name, t.tolist(), "int64")

    def install(self) -> None:
        orig_exec, orig_sample = self.runner.execute_model, self.runner.sample

        def execute_model(sched: Any) -> Any:
            self._step_index += 1
            self._commit(self._step_index - 1, "model.embed_tokens/in", sched.ids, "int64")
            return orig_exec(sched)

        def sample(ib: Any, sampled: list[int]) -> Any:
            step, vals = self._step_index - 1, list(sampled)
            if self.flip and self.flip[0] == step:
                vals[self.flip[1]] += 1
                self.flip = None
            self._commit(step, SR.SAMPLED, vals, "int32")
            for s in self.sources:
                s.on_sample(ib, len(vals), SimpleNamespace(step=step))
            return orig_sample(ib, sampled)

        self.runner.execute_model, self.runner.sample = execute_model, sample

    def uninstall(self) -> None:
        del self.runner.execute_model, self.runner.sample

    def finalize(self) -> Any:
        steps = [SimpleNamespace(root=bytes([s + 1]) * 64, n_bytes=sum(m.nbytes for m in self._layouts[s])) for s in range(self._step_index)]
        return SimpleNamespace(run_root=b"\x01" * 64, steps=steps, extra={"errors": [], "errors_n": 0, "identity_check": {"ok": True}})


class Tokenizer:
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return ([1] if add_special_tokens else []) + [100 + len(w) for w in text.split()]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"t{i}" for i in ids]


class Serving:
    """The server over the stand-ins, its engine loop on a thread of its own, as `main` runs it."""

    def __init__(self, out: Path, *, commit: bool = True, flip: tuple[int, int] | None = None, window_steps: int = 64,
                 idle_close_steps: int = 1, force_include_usage: bool = True, seed_override: int | None = None,
                 served: dict | None = None, marks: Any = None, block: int | None = None, protocol: Any = None,
                 arm: Any = None) -> None:
        self.runner = Runner()
        self.windows = sc.CommitWindows(Committer(self.runner, flip), out, run_id="t", window_steps=window_steps,
                                       idle_close_steps=idle_close_steps, eos_ids=[Tokenizer.eos_token_id],
                                       clock=lambda: 0.0, wall=lambda: 0.0, served=served, marks=marks,
                                       arm=arm(self.runner) if arm is not None else None) if commit else None
        self.loop = so.EngineLoop(Engine(self.runner, seed_override), self.windows)
        self.server = so.Server(proto=protocol or proto.PlainProtocol(force_include_usage), served_name="m", max_model_len=512,
                                tokenizer=Tokenizer(),
                                defaults={"temperature": 1.0}, submit=self.loop.submit, abort=self.loop.abort, clock=lambda: 7)
        self.http = so.HttpThread(self.server, "127.0.0.1", 0)
        self.loop.engine.block = block

    def __enter__(self) -> Serving:
        self.engine_thread = threading.Thread(target=self.loop.run, daemon=True)
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

    def post(self, path: str, body: dict) -> tuple[int, bytes]:
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def send(self, request: bytes) -> bytes:
        """``request``'s bytes on a connection of its own, and every byte the server sends until it closes."""
        host, port = self.http.address
        with socket.create_connection((host, port)) as c:
            c.sendall(request)
            got = b""
            while chunk := c.recv(65536):
                got += chunk
        return got


def events(raw: bytes) -> list[str]:
    assert raw.endswith(proto.SSE_DONE)
    return [e[len("data: "):] for e in raw.decode().split("\n\n") if e]


# ---------------------------------------------------------------------------------------------------------------- serving

def test_stream_is_the_certifiers(tmp_path):
    with Serving(tmp_path, commit=False) as s:
        models = json.loads(urllib.request.urlopen(s.url + "/v1/models").read())
        assert [m["id"] for m in models["data"]] == ["m"]
        status, raw = s.post("/tokenize", {"model": "m", "prompt": "a bb ccc"})
        assert status == 200 and json.loads(raw)["tokens"] == [1, 101, 102, 103]
        status, raw = s.post("/v1/completions", CERTIFIER_BODY)
    assert status == 200
    ev = events(raw)
    assert ev[-1] == "[DONE]"
    chunks = [json.loads(e) for e in ev[:-1]]
    text = [c for c in chunks if c["choices"]]
    assert len(text) == 4 and text[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["text"].startswith(" w") for c in text)
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == {"prompt_tokens": 3, "total_tokens": 7, "completion_tokens": 4}
    assert len({c["id"] for c in chunks}) == 1 and chunks[0]["id"].startswith("cmpl-")


def test_without_forced_usage_no_usage_chunk(tmp_path):
    with Serving(tmp_path, commit=False, force_include_usage=False) as s:
        _, raw = s.post("/v1/completions", CERTIFIER_BODY)
        _, raw_u = s.post("/v1/completions", dict(CERTIFIER_BODY, stream_options={"include_usage": True}))
    assert all(json.loads(e)["choices"] for e in events(raw)[:-1])
    assert json.loads(events(raw_u)[-2])["usage"]["completion_tokens"] == 4


def test_errors_and_whole_response(tmp_path):
    with Serving(tmp_path, commit=False) as s:
        status, raw = s.post("/v1/completions", dict(CERTIFIER_BODY, model="other"))
        assert status == 404 and json.loads(raw)["error"]["type"] == "NotFoundError"
        status, raw = s.post("/v1/completions", dict(CERTIFIER_BODY, prompt=[1] * 600))
        assert status == 400 and "maximum context length" in json.loads(raw)["error"]["message"]
        status, raw = s.post("/v1/completions", {"model": "m", "prompt": "a bb", "max_tokens": 3, "return_token_ids": True})
    body = json.loads(raw)
    assert status == 200 and body["usage"]["completion_tokens"] == 3 and body["choices"][0]["prompt_token_ids"] == [1, 101, 102]
    assert body["choices"][0]["text"] == "".join(f" w{t}" for t in body["choices"][0]["token_ids"])


def id_request(payload: int = proto.ID_PAYLOAD) -> bytes:
    """The network certifier's id request in cells of P + 9 bytes (`active.id_request`, offset 0)."""
    return A.id_request(model="m", user="u", prompt=[128000, 791, 4062], max_tokens=2, frame_bytes=payload + A.HEADER.size,
                        offset=0)


def shims_request(port: int) -> bytes:
    """An id request as the network certifier's live shim sends it straight to the endpoint
    (`benchmarks/network_traces/active_replay.LiveServer._serve`): its fields, the user forwarded and
    certifier_serve's `UPSTREAM_FIELDS` added, compact, in a POST of the shim's own head with a release."""
    body = json.loads(id_request().partition(b"\r\n\r\n")[2])
    name = body.pop("user")
    raw = json.dumps(body | {"user": name} | {"return_token_ids": True}, separators=(",", ":")).encode()
    return (b"POST /v1/completions HTTP/1.1\r\nhost: 127.0.0.1:%d\r\ncontent-type: application/json\r\n" % port
            + REL.release_head(0, b"run") + b"content-length: %d\r\nconnection: close\r\n\r\n" % len(raw) + raw)


def dechunked(raw: bytes) -> tuple[bytes, bytes]:
    """An HTTP response's head and its chunked body's bytes."""
    head, _, rest = raw.partition(b"\r\n\r\n")
    body = b""
    while (size := int(rest[:rest.index(b"\r\n")], 16)):
        start = rest.index(b"\r\n") + 2
        body, rest = body + rest[start:start + size], rest[start + size + 2:]
    return head, body


def test_ids_answers_in_id_form_exactly_the_wardens_id_requests():
    """`IdsProtocol.of`: itself for an id request in its P + 9-byte cells, with or without the network certifier's release
    headers (`release.with_release`); its base for anything else, the shim's rebuilt POST, an id request in other cells, or
    bytes that aren't a request."""
    p = proto.IdsProtocol(payload=proto.ID_PAYLOAD, base=proto.PlainProtocol())
    assert p.of(id_request()) is p and p.of(REL.with_release(id_request(), 3, b"run")) is p
    assert p.of(shims_request(18097)) is p.base and p.of(id_request(140)) is p.base
    assert p.of(b"") is p.base and p.of(id_request()[:-1]) is p.base
    assert proto.PlainProtocol().of(id_request()).name == "plain"


def test_under_ids_the_wardens_id_request_streams_raw_id_events_then_done(tmp_path):
    """``--protocol ids`` on the network certifier's own id request (`active.id_request` in its P + 9-byte cells, offset 0, with the
    release headers the certifier adds) with the usage forced: the server sends no HTTP head, no chunking and no usage,
    each sampled id one event of P bytes, then ``data: [DONE]``, every byte as `TWO_IDS_AT_140` at P = 140. At the
    scan's P = 503 every P bytes are one id event, the cells the certifier makes of them, and the whole is what
    warden_serve's id-form parser (`parse_id_response`) takes: `io_tie.id_event`s, then `io_tie.TRAILER`. A whole
    response under ids is still the JSON one."""
    for payload in (140, proto.ID_PAYLOAD):
        ids = proto.IdsProtocol(True, payload=payload, base=proto.PlainProtocol(True))
        with Serving(tmp_path / str(payload), commit=False, protocol=ids) as s:
            raw = s.send(REL.with_release(id_request(payload), 0, b"run"))
            assert s.send(id_request(payload)) == raw
            status, whole = s.post("/v1/completions", dict(CERTIFIER_BODY, stream=False, max_tokens=2))
        assert raw == b"".join(IT.id_event(t, payload) for t in (4853, 4884)) + IT.TRAILER
        if payload == 140:
            assert raw == TWO_IDS_AT_140
        else:
            cells = [raw[k:k + payload] for k in range(0, len(raw), payload)]
            assert [len(c) for c in cells] == [payload, payload, len(IT.TRAILER)]
            assert [int(c[len(IT.ID_EVENT_HEAD):len(IT.ID_EVENT_HEAD) + IT.ID_DIGITS]) for c in cells[:2]] == [4853, 4884]
        assert status == 200 and json.loads(whole)["usage"]["completion_tokens"] == 2


def test_under_ids_the_network_certifiers_clients_get_vllms_http_and_json(tmp_path):
    """The same endpoint under ``--protocol ids`` for the network certifier (certifier_serve, straight to the
    endpoint): ``/v1/models`` and ``/tokenize`` as ever, its shim's streamed POST (the id request's fields with
    ``return_token_ids``) an HTTP response, chunked, of vLLM's JSON chunks carrying the token ids, and its probe's
    (urllib) the same."""
    ids = proto.IdsProtocol(True, payload=proto.ID_PAYLOAD, base=proto.PlainProtocol(True))
    with Serving(tmp_path, commit=False, protocol=ids) as s:
        assert [m["id"] for m in json.loads(urllib.request.urlopen(s.url + "/v1/models").read())["data"]] == ["m"]  # noqa: S310 (the test's own 127.0.0.1 endpoint)
        assert json.loads(s.post("/tokenize", {"model": "m", "prompt": "a bb ccc"})[1])["tokens"] == [1, 101, 102, 103]
        head, body = dechunked(s.send(shims_request(s.http.address[1])))
        status, probed = s.post("/v1/completions", dict(CERTIFIER_BODY, max_tokens=2, return_token_ids=True))
    assert head.startswith(b"HTTP/1.1 200 OK\r\n") and b"transfer-encoding: chunked" in head
    for raw in (body, probed):
        chunks = [json.loads(e) for e in events(raw)[:-1]]
        assert [t for c in chunks for ch in c["choices"] for t in ch["token_ids"]] == [4853, 4884]
        assert chunks[-1]["usage"]["completion_tokens"] == 2
    assert status == 200


def records(out: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (out / name).read_text().splitlines()]


def serve_concurrently(s: Serving, bodies: list[dict]) -> list[bytes]:
    got: list[Any] = [None] * len(bodies)

    def one(i: int) -> None:
        got[i] = s.post("/v1/completions", bodies[i])[1]

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return got


@pytest.mark.parametrize("window_steps", [64, 2])
def test_commit_record_maps_and_checks_every_request(tmp_path, window_steps):
    bodies = [dict(CERTIFIER_BODY, prompt=[128000, 10 + i, 20 + i], max_tokens=3 + i, min_tokens=3 + i) for i in range(4)]
    bodies.append({"model": "m", "prompt": "x yy zzz", "max_tokens": 5, "stream": True, "ignore_eos": True, "temperature": 0})
    with Serving(tmp_path, window_steps=window_steps) as s:
        raws = serve_concurrently(s, bodies)
    reqs = {r["http_id"]: r for r in records(tmp_path, "requests.jsonl")}
    assert len(reqs) == len(bodies)
    for body, raw in zip(bodies, raws):
        chunks = [json.loads(e) for e in events(raw)[:-1]]
        rec = reqs[chunks[0]["id"]]
        assert rec["engine_id"] == rec["http_id"] + "-0"
        assert len(rec["runner_ids"]) == 1 and rec["runner_ids"][0].startswith(rec["engine_id"] + "-")
        prompt = body["prompt"] if isinstance(body["prompt"], list) else Tokenizer().encode(body["prompt"])
        assert rec["prompt_ids"] == prompt
        streamed_text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
        assert streamed_text == "".join(f" w{t}" for t in rec["streamed_ids"]) and len(rec["streamed_ids"]) == body["max_tokens"]
        assert rec["check"]["passed"], rec["check"]
        assert len(rec["rows"]) == body["max_tokens"] and rec["rows"][0][3] - rec["rows"][0][2] == len(prompt)
        assert rec["finish"] == {"reason": "length", "stop_reason": None} and rec["committed_seeds"] == [DEFAULT_SEED]
    wins = records(tmp_path, "windows.jsonl")
    assert all(w["passed"] for w in wins)
    assert {e for w in wins for e in w["requests"]} == {r["engine_id"] for r in reqs.values()}
    assert all(w["ids_name"] == "model.embed_tokens/in" for w in wins if not w.get("empty"))
    trace = records(tmp_path, "step_trace.jsonl")
    assert [(t["window"], t["step_root"]) for t in trace] == [(w["window"], r) for w in wins for r in w.get("step_roots", [])]
    assert all(t["rows"] and t["t_end"] is not None for t in trace)
    if window_steps == 2:
        assert len([w for w in wins if not w.get("empty")]) > 1 and any(len(r["windows"]) > 1 for r in reqs.values())
    assert s.windows.summary()["passed"]


def test_a_later_turn_read_from_the_prefix_cache_is_checked_against_the_turns_that_committed_it(tmp_path):
    def turn(s: Serving, prompt: list[int]) -> list[int]:
        body = dict(CERTIFIER_BODY, prompt=prompt, max_tokens=3, min_tokens=3, stream=False, return_token_ids=True)
        return prompt + json.loads(s.post("/v1/completions", body)[1])["choices"][0]["token_ids"]

    with Serving(tmp_path, block=4) as s:
        turn(s, turn(s, turn(s, [128000] + list(range(10, 19))) + [40, 41]) + [50])
    a, b, c = records(tmp_path, "requests.jsonl")
    assert a["check"]["passed"] and a["check"]["cache_hit"] == 0
    assert b["check"]["passed"] and b["check"]["cache_hit"] == 12 and b["check"]["prefix_spans"] == [[a["engine_id"], 0, 12]], b["check"]
    assert c["check"]["passed"] and c["check"]["cache_hit"] == 16, c["check"]
    assert c["check"]["prefix_spans"] == [[a["engine_id"], 0, 12], [b["engine_id"], 12, 16]]
    got = s.windows.summary()
    assert got["passed"] and (got["cache_hits"], got["cached_positions"]) == (2, 28), got


def test_the_gpu_arms_client_serves_shared_prefix_turns_that_the_server_checks(tmp_path):
    from tests.pipeline import serve_prefix_gpu as client

    sent = tmp_path / "client.jsonl"
    with Serving(tmp_path, block=4) as s:
        assert client.main(["--url", s.url, "--out", str(sent), "--requests", "6", "--prompt-range", "8:32", "--tokens", "4",
                            "--system-prompts", "2", "--system-tokens", "24", "--turns", "2", "--max-model-len", "512"]) == 0
    got = client.compare(records(tmp_path, "client.jsonl"), records(tmp_path, "requests.jsonl"))
    assert got["passed"] and got["later_turns"] > 0 and got["cache_hits"] >= got["later_turns"], got
    assert s.windows.summary()["passed"]
    bad = records(tmp_path, "client.jsonl")
    bad[0]["streamed_ids"] = bad[0]["streamed_ids"][:-1]
    assert not client.compare(bad, records(tmp_path, "requests.jsonl"))["passed"]


def test_each_request_records_when_it_was_received_submitted_admitted_and_first_streamed(tmp_path):
    """The wall-clock stamps that tell late bytes (``received``) from a hold before the engine (``submitted``,
    ``admitted``), on one clock the server, the engine loop and the windows share."""
    ticks, lock = iter(range(1, 1_000_000)), threading.Lock()

    def clock() -> float:
        with lock:
            return float(next(ticks))

    s = Serving(tmp_path)
    s.server.clock = s.loop.clock = s.windows.wall = clock
    with s:
        assert s.post("/v1/completions", dict(CERTIFIER_BODY, prompt=[1, 2, 3]))[0] == 200
    (rec,) = records(tmp_path, "requests.jsonl")
    order = ["head", "received", "submitted", "admitted", "first_token"]
    assert set(rec["times"]) == set(order), rec["times"]
    stamps = [rec["times"][k] for k in order]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps), rec["times"]


@pytest.mark.parametrize("idle_close_steps, windows", [(1, 3), (100, 1)])
def test_a_window_closes_at_idle_once_it_holds_idle_close_steps(tmp_path, idle_close_steps, windows):
    with Serving(tmp_path, idle_close_steps=idle_close_steps) as s:
        for i in range(3):
            assert s.post("/v1/completions", dict(CERTIFIER_BODY, prompt=[1, 2 + i]))[0] == 200
    wins = [w for w in records(tmp_path, "windows.jsonl") if not w.get("empty")]
    assert len(wins) == windows and all(w["passed"] and w["seconds"] == {"open": 0.0, "close": 0.0} for w in wins)
    assert s.windows.summary()["passed"] and s.windows.summary()["requests_checked"] == 3


class LoopSpy:
    """`CommitWindows` as the engine loop drives it, recording the order requests reach the engine and each step's live ones."""

    is_open = False

    def __init__(self) -> None:
        self.admitted_order: list[str] = []
        self.live_at_steps: list[list[str]] = []

    def admitted(self, adm: Any) -> None:
        self.admitted_order.append(adm.engine_id)

    def before_step(self, live: Any = ()) -> None:
        self.live_at_steps.append(sorted(live))

    def streamed(self, *_a: Any) -> None:
        pass

    def aborted(self, *_a: Any) -> None:
        pass

    def idle(self) -> None:
        pass

    def stepped(self) -> None:
        pass


def test_window_requests_holds_the_engine_to_its_cap_and_serves_the_rest_in_admission_order():
    spy, got = LoopSpy(), {}
    loop = so.EngineLoop(Engine(Runner()), spy, cap=2)
    eids, lengths = [f"cmpl-{i}-0" for i in range(5)], [3, 1, 4, 2, 2]
    for i, (eid, n) in enumerate(zip(eids, lengths)):
        loop.submit(Admission(f"cmpl-{i}", eid, [1, 2 + i], {"max_tokens": n, "seed": 0},
                              lambda d, e=eid: got.setdefault(e, []).append(d), {}))
    loop.abort(eids[4])                                  # held, so it never reaches the engine
    loop.drain("done")
    loop.run()
    assert spy.admitted_order == eids[:4], "held requests go in in admission order, the aborted one never"
    assert max(len(live) for live in spy.live_at_steps) == 2 and loop.held == collections.deque()
    assert {e: sum(len(d.token_ids) for d in ds) for e, ds in got.items()} == dict(zip(eids[:4], lengths[:4]))
    assert all(ds[-1].finished and ds[-1].error is None for ds in got.values())


def test_a_window_closes_before_a_step_that_would_bring_in_more_than_window_requests(tmp_path):
    com = SimpleNamespace(_step_index=0, reset=lambda: None, install=lambda: None)
    w = sc.CommitWindows(com, tmp_path, run_id="t", window_steps=64, request_cap=2)
    closed: list[list[str]] = []

    def close() -> None:
        closed.append(sorted(w.in_window))
        w.is_open = False

    w.close = close
    for live in (["a"], ["a", "b"], ["b"], ["b", "c"], ["c"], ["c", "d"], ["d", "e"]):
        w.before_step(live)
        com._step_index += 1
    close()
    assert closed == [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]], "each window at most 2, a running request carried on"
    unbounded = sc.CommitWindows(com, tmp_path, run_id="t", window_steps=64)
    for live in (["a"], ["a", "b"], ["b", "c"]):
        unbounded.before_step(live)
    assert sorted(unbounded.in_window) == ["a", "b", "c"], "without the cap nothing closes on a third request"


def test_check_fails_on_a_committed_id_that_was_not_served(tmp_path):
    with Serving(tmp_path, flip=(1, 0)) as s:
        serve_concurrently(s, [CERTIFIER_BODY, dict(CERTIFIER_BODY, prompt=[5, 6])])
    failed = [r["check"] for r in records(tmp_path, "requests.jsonl") if not r["check"]["passed"]]
    assert len(failed) == 1 and any(p.startswith("samples: 1 position(s) differ") for p in failed[0]["problems"]), failed
    assert not s.windows.summary()["passed"]


def test_a_seedless_request_commits_the_default_seed_and_a_seeded_one_its_own(tmp_path):
    with Serving(tmp_path) as s:
        serve_concurrently(s, [CERTIFIER_BODY, dict(CERTIFIER_BODY, prompt=[5, 6], seed=1234)])
    by_rule = {r["ingress"]["seed_rule"]: r for r in records(tmp_path, "requests.jsonl")}
    assert by_rule["default"]["ingress"]["seed"] == DEFAULT_SEED and by_rule["default"]["committed_seeds"] == [DEFAULT_SEED]
    assert "seed" not in by_rule["default"]["ingress"]["brought"]
    assert by_rule["request"]["ingress"]["seed"] == 1234 and by_rule["request"]["committed_seeds"] == [1234]
    assert all(r["check"]["passed"] for r in by_rule.values()) and s.windows.summary()["passed"]


def test_a_committed_seed_other_than_the_one_admitted_fails(tmp_path):
    with Serving(tmp_path, seed_override=7) as s:
        s.post("/v1/completions", CERTIFIER_BODY)
    (rec,) = records(tmp_path, "requests.jsonl")
    assert not rec["check"]["passed"] and any(p.startswith("seeds: committed [7]") for p in rec["check"]["problems"])
    assert not s.windows.summary()["passed"]


def test_the_ingress_record_holds_the_sampling_the_request_brought(tmp_path):
    body = dict(CERTIFIER_BODY, temperature=0.7, top_k=50, ignore_eos=False, stop=["."])
    with Serving(tmp_path) as s:
        assert s.post("/v1/completions", body)[0] == 200
    (rec,) = records(tmp_path, "requests.jsonl")
    ing = rec["ingress"]
    assert ing["brought"] == {k: body[k] for k in ("temperature", "top_k", "max_tokens", "min_tokens", "ignore_eos", "stop")}
    assert (ing["params"]["temperature"], ing["params"]["top_k"], ing["params"]["top_p"]) == (0.7, 50, 1.0)
    assert rec["check"]["passed"], rec["check"]


@pytest.mark.parametrize("top_p", [0.9, 0.5])
def test_top_p_below_one_is_refused_naming_the_gap(tmp_path, top_p):
    with Serving(tmp_path) as s:
        status, raw = s.post("/v1/completions", dict(CERTIFIER_BODY, top_p=top_p))
        ok, _ = s.post("/v1/completions", dict(CERTIFIER_BODY, top_p=1))
    err = json.loads(raw)["error"]
    assert status == 400 and err["param"] == "top_p" and "split count" in err["message"] and f"top_p={top_p}" in err["message"]
    assert "split count" in err["message"] and ok == 200
    assert len(records(tmp_path, "requests.jsonl")) == 1          # the refused request never reached the engine


# ---------------------------------------------------------------------------------------------------------------- served rows

ENGINE = {"max_num_seqs": 8, "max_num_batched_tokens": 2048, "max_model_len": 512, "num_gpu_blocks": 64, "block_size": 16}


def served_statics(**size_class: int) -> dict:
    return sc.served_statics(ENGINE, {**dict.fromkeys(sc.SERVED_CLASS, 1), "R": 8, "P": 3, "IW": 6, **size_class})


def private(out: Path, w: int) -> dict:
    return json.loads((out / "served_rows" / f"w{w}.json").read_text())


def recommit(statics: dict, by_step: dict, ids: dict, steps: int, run_root: str, priv: dict):
    return sc.commit_served(statics, by_step, ids, steps, bytes.fromhex(run_root), salts=[bytes.fromhex(s) for s in priv["salts"]])


def served_run(out: Path, statics: dict, marks: Any = None) -> tuple[list[dict], list[dict], list[dict]]:
    bodies = [dict(CERTIFIER_BODY, prompt=[128000, 10 + i, 20 + i], max_tokens=3 + i, min_tokens=3 + i) for i in range(4)]
    with Serving(out, window_steps=4, served=statics, marks=marks) as s:
        serve_concurrently(s, bodies)
        assert s.post("/v1/completions", dict(CERTIFIER_BODY, prompt=[1, 2, 3, 4]))[0] == 200
    return records(out, "windows.jsonl"), records(out, "requests.jsonl"), records(out, "step_trace.jsonl")


def test_each_window_commits_the_rows_it_ran_at_its_close_under_its_run_root(tmp_path):
    """At each close, before the close is announced (and so before anything can sample over the window), its windows.jsonl
    line holds the public record of the rows it ran, bound to its run root: exactly step_trace.jsonl's rows under the
    requests' arrival ids, as `served_rows.register` commits them, each slice window's registered row the slice's ``srv``
    row, its b‖c a leaf of the published root. The record holds no row or salt."""
    statics = served_statics()
    closed: list[dict] = []

    def marks(w: int, event: str) -> None:
        if event == "close":
            line = records(tmp_path, "windows.jsonl")[-1]
            assert line["window"] == w
            closed.append(line)

    wins, reqs, trace = served_run(tmp_path, statics, marks)
    full = [w for w in wins if not w.get("empty")]
    assert len(full) > 1 and all(w["passed"] for w in wins) and [w["window"] for w in closed] == [w["window"] for w in wins]
    assert all("served_rows" in w for w in closed if not w.get("empty"))
    assert sorted(r["arrival"] for r in reqs) == list(range(len(reqs)))
    arrival = {rid: r["arrival"] for r in reqs for rid in r["runner_ids"]}
    S = SimpleNamespace(**statics)
    for w in full:
        rec, priv = w["served_rows"], private(tmp_path, w["window"])
        assert rec["binding"] == SRW.binding(bytes.fromhex(w["run_root"])).hex() and rec["statics"] == statics
        assert rec["count"] == -(-w["steps"] // S.P)
        by_step = {t["step"]: [tuple(r) for r in t["rows"]] for t in trace if t["window"] == w["window"]}
        ids = {rid: arrival[rid] for rows in by_step.values() for rid, *_ in rows}
        assert priv["ids"] == ids
        served, session = recommit(statics, by_step, ids, w["steps"], w["run_root"], priv)
        assert {k: v for k, v in rec.items() if k != "statics"} == served.record()
        for k, win in enumerate(session):
            assert served.registration.rows[k] == SRW._row(CS.srv_row(S, win))[0]
            assert SRW.leaf_ok(rec, bytes.fromhex(w["run_root"]), k, served.bc(k), served.registration.open(k).path)
        text = json.dumps(w)
        assert not any(s in text for s in priv["salts"])


def test_rows_it_cant_commit_fail_their_window_by_name(tmp_path):
    """An arrival id wider than the class's IW can't be packed: the window that holds it fails, named, and the run fails."""
    wins, _, _ = served_run(tmp_path, served_statics(IW=1))
    bad = [w for w in wins if "error" in w.get("served_rows", {})]
    assert bad and all(not w["passed"] and "doesn't fit its 1 bits" in w["served_rows"]["error"] for w in bad)


def test_the_served_class_defaults_to_the_engines_and_refuses_too_few_rows():
    def opts(*more: str) -> so.Options:
        return cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--max-num-seqs", "8", "--window-steps", "16", *more],
                         environ={})

    assert so.served_class(opts()) == {**dict.fromkeys(sc.SERVED_CLASS, 1), "R": 8, "P": 16, "IW": 16}
    assert so.served_class(opts("--served-rows-class", '{"P": 7, "IW": 10}'))["P"] == 7
    with pytest.raises(cli.UsageError, match="a JSON object of"):
        so.served_class(opts("--served-rows-class", '{"X": 1}'))
    with pytest.raises(cli.UsageError, match="under --max-num-seqs 8"):
        so.preflight(opts("--served-rows-class", '{"R": 4}'), which=lambda _: "/v/bin/ninja")
