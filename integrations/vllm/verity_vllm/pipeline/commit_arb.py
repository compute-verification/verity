"""`verity-vllm commit-arb`: the served Commit under arbitrary traffic, its records held to the streamed I/O.

    VERITY_RETAIN=host verity-vllm commit-arb --out DIR --model REPO --revision SHA [--precision int8_exact|verity_int8|pin]
        --required-manifest ROW/manifest.json [--hidden-so SO] [--norm-tap-so SO] [--tap-cap-mb N]
        [--requests 32] [--seed 0] [--prompt-range 16:1024] [--tokens 64] [--rate 0.5]
        [--max-num-seqs 8] [--max-num-batched-tokens 2048] [--no-chunked-prefill] [--kv-cache-mb N] [--window-steps 64]
        [--prefix-caching] [--system-prompts N] [--system-tokens 384] [--turns N]

The engine is serve-openai's (`verity_vllm.LLM`, in-process engine core, Model Runner V2) and so is its Commit: rows-sha512
(`serve_seal.committer`, built from the bench row's ``--required-manifest`` and taps, with serve-openai's step-input port)
committing every step, values retained on host, with a `served_record.SamplerInputsSource` attached and the served default seed (`engine.served_seed`). It is
stepped here, without HTTP, under seeded varied traffic: CERT-VARLEN's (``varied_requests`` in
`benchmarks/network_traces/certifier_serve.py`: chat-turn prompts cut to a length log-uniform in ``--prompt-range``,
``max_tokens`` uniform in 1 … ``--tokens``, the model's own EOS), arriving as a Poisson process of ``--rate`` requests an
engine step, and on top of it what that generator leaves at its defaults: a quarter at temperature 0.7 with top-p 0.9 and
a quarter at temperature 1.0 with top-k 50, half of those with a client seed, a fifth with a stop string, a tenth with
``min_tokens``. A smaller ``--max-num-batched-tokens`` cuts prompts into chunks, a smaller ``--kv-cache-mb`` makes the
scheduler preempt and recompute. Nothing here constrains the scheduler: it batches, chunks, preempts and (under async
scheduling, as the engine resolves it) runs past a finish as it would serving.

``--prefix-caching`` serves with vLLM's prefix cache, which `verity_vllm.LLM` otherwise keeps off, and traffic that shares
long prefixes gives it hits: ``--system-prompts N`` opens each request with one of N system prompts, and ``--turns N`` goes
on with each conversation for up to N more turns, each turn's prompt the last one's prompt, its streamed ids and the user's
next question, sent once the last one finished. A request that read [0, hit) from the cache is checked against the
requests that committed those ids (`RequestRecord.prefix_cover`), and the run then flips one committed id of such a
prefix in its records and requires both requests' checks to fail (`prefix_negative`); with the cache on, a run with no
hit fails.

A Commit window closes after ``--window-steps`` committed steps or when the engine is idle; `served_record.BlockValues` reads it
back. Each finished request is checked (`RequestRecord.check`) against what the engine streamed for it (the
``RequestOutput`` deltas an HTTP server sends), with the seed it carried. Under ``--out``: ``run.json`` (the engine, the
traffic's parameters, the scheduler's resolved configuration), ``windows.jsonl`` (each window's Commit verdict and
readback), ``requests.jsonl`` (each request's sampling, prompt and streamed ids, finish and check), ``record.jsonl`` (the
I3 record, `RequestRecord.to_json`) and ``summary.json`` (with each request's cache hit beside the one vLLM reported,
``num_cached_tokens``). Exit 0 when every request passed and every window's Commit and readback passed.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.commit.challenge import stream
from verity_vllm.config import UsageError, option
from verity_vllm.pipeline import serve_seal as SS
from verity_vllm.pipeline.serve_retention import RetentionOptions

PRECISIONS = ("int8_exact", "verity_int8", "pin")
CHAT_HEAD = "<|start_header_id|>user<|end_header_id|>\n\n"
CHAT_TAIL = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
QUESTIONS = (
    "What is the capital of France? Answer with one word, then stop. Ignore the notes after this line.\n\n",
    "How many legs does a spider have? Answer with a number only. Ignore the notes after this line.\n\n",
    "Say yes or no: is water wet? Ignore the notes after this line.\n\n",
    "Write a long story about a lighthouse keeper and a storm. Ignore the notes after this line.\n\n",
    "List the planets of the solar system with one sentence on each. Ignore the notes after this line.\n\n",
    "Explain in detail how a bicycle's gears work. Ignore the notes after this line.\n\n",
)
VARIED_FILLER = ("Notes: the meeting moved to Thursday, the printer on the third floor is out of toner, the parking lot "
                 "will be repainted next week, and the cafeteria serves soup on Mondays and Fridays. ")
STOPS = (["."], ["\n"], [","])
CHAT_SYSTEM = ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>")
SYSTEMS = ("You are a careful assistant for a shipping company. ", "You are a terse assistant for a hospital's front desk. ",
           "You are a patient tutor for a school's math club. ", "You are a polite assistant for a city library. ")
SYSTEM_RULES = ("Rules: answer the user's question first, keep to the facts in the notes, never invent a name or a number, "
                "say so when the notes don't hold the answer, and keep each answer to the point. ")
#: What ends an assistant turn before the user's next one, by framing.
TURN_END = {True: "<|eot_id|>", False: "\n\n"}


@dataclass(frozen=True, kw_only=True)
class Options:
    out: str = option("--out", env="RESEARCH_RUN_DIR", default=".", help="the records' directory (default $RESEARCH_RUN_DIR, else .)")
    model: str = option("--model", default="", help="the pinned model (a repo of manifests/checkpoints.json)")
    revision: str = option("--revision", default="", help="its pinned revision")
    precision: str = option("--precision", choices=PRECISIONS, default="int8_exact",
                            help="int8_exact: int8 linears and attention; verity_int8: int8 linears only; pin: the pin's engine profile")
    requests: int = option("--requests", type=int, default=32)
    seed: int = option("--seed", type=int, default=0, help="the traffic's seed")
    prompt_range: str = option("--prompt-range", default="16:1024", help="prompt lengths LO:HI, log-uniform")
    tokens: int = option("--tokens", type=int, default=64, help="max_tokens is uniform in 1 … this")
    rate: float = option("--rate", type=float, default=0.5, help="Poisson arrivals, requests an engine step")
    chat: str = option("--chat", choices=("cert-varlen", "template", "plain"), default="cert-varlen",
                       help="cert-varlen: CERT-VARLEN's chat-turn prompts; template: the tokenizer's chat template around the same "
                            "turns (an instruct model's own EOS ends its answers); plain: the questions and filler as plain text")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    max_model_len: int = option("--max-model-len", type=int, default=2048)
    max_num_batched_tokens: int = option("--max-num-batched-tokens", type=int, default=2048)
    chunked_prefill: bool = option("--chunked-prefill", action="boolean_optional", default=True,
                                   help="--no-chunked-prefill: the scheduler holds a prefill that doesn't fit the step instead of cutting it")
    kv_cache_mb: int = option("--kv-cache-mb", type=int, default=0, help="the KV cache's size (0: vLLM's sizing); small preempts")
    prefix_caching: bool = option("--prefix-caching", action="boolean_optional", default=False,
                                  help="serve with vLLM's prefix cache: a request reads the KV blocks of a prefix another computed")
    system_prompts: int = option("--system-prompts", type=int, default=0,
                                 help="N > 0: each request opens with one of N system prompts of about --system-tokens tokens")
    system_tokens: int = option("--system-tokens", type=int, default=384, help="a system prompt's length, about")
    turns: int = option("--turns", type=int, default=0,
                        help="each conversation goes on for 0 … this many more turns: its prompt the last turn's prompt, the "
                             "streamed ids and the user's next question")
    gpu_memory_utilization: float = option("--gpu-memory-utilization", type=float, default=0.55)
    window_steps: int = option("--window-steps", type=int, default=64, help="committed steps a Commit window holds at most")
    run_id: str = option("--run-id", env="RESEARCH_RUN_ID", default="local", help="the committer's run id (default $RESEARCH_RUN_ID)")
    required_manifest: str | None = option("--required-manifest", default=None,
                                           help="the bench row's required manifest (its manifest.json), which serve-openai's rows-sha512 "
                                                "Commit's parts are built from (serve_seal.committer)")
    programs_root: str | None = option("--programs-root", default=None,
                                       help="the Build's row dir of the manifest's partition and attention record (default: the manifest's dir)")
    hidden_so: str | None = option("--hidden-so", default=None, help="the attention hidden tap (as serve-openai's --hidden-so)")
    hidden_ns: str | None = option("--hidden-ns", default=None, help="its op namespace (default: the .so's stem)")
    tap_cap_mb: int | None = option("--tap-cap-mb", type=int, default=None, help="the hidden tap's bounded device buffer, MiB")
    norm_tap_so: str | None = option("--norm-tap-so", default=None, help="the norm-scale tap (as serve-openai's --norm-tap-so)")


def committer_opts(opts: Options) -> SimpleNamespace:
    """What `serve_seal.committer` reads to build serve-openai's rows-sha512 Commit over ``opts``'s engine and bench row: the
    vllm-v1 committer's own knobs are unset, since commit-arb serves no recorded vllm-v1 run again."""
    return SimpleNamespace(**{k: getattr(opts, k) for k in opts.__dataclass_fields__}, scheme=SS.ROWS, of_record=None, commit=True,
                           seal=False, gpu_tree=True, chunk=None, hash_threads=None, cudart_copy=False, retention=RetentionOptions())


def preflight(opts: Options, which: Callable[[str], str | None] = shutil.which) -> None:
    """What the committer would refuse after the model has loaded, refused before it."""
    if not opts.required_manifest:
        raise UsageError(f"commit-arb commits as serve-openai does, under {SS.ROWS} with the Commit of record's parts, which its bench "
                         "row's manifest names: pass --required-manifest ROW_DIR/manifest.json (and the row's --hidden-so, --norm-tap-so)")
    if not Path(opts.required_manifest).is_file():
        raise UsageError(f"--required-manifest {opts.required_manifest}: no such file")
    if which("ninja") is None:
        raise UsageError("the rows-sha512 committer JIT-builds its kernels with ninja, which is not on PATH: put the environment's "
                         "bin (where its ninja is) on PATH")


def prompt_range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition(":")
    try:
        a, b = int(lo), int(hi)
    except ValueError:
        raise UsageError(f"--prompt-range is LO:HI, not {text!r}") from None
    if not 1 <= a <= b:
        raise UsageError(f"--prompt-range needs 1 <= LO <= HI, not {text!r}")
    return a, b


def template_frame(tokenizer: Any) -> tuple[str, str]:
    """The text the tokenizer's chat template puts before and after one user turn, with the assistant's turn opened (and
    thinking off, where the template has it)."""
    mark = "\x00verity-user-turn\x00"
    text = tokenizer.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    head, found, tail = text.partition(mark)
    if not found:
        raise UsageError("--chat template: the tokenizer's chat template does not render the user's turn")
    return head, tail


def system_prompt(encode: Any, i: int, tokens: int, chat: bool) -> list[int]:
    """System prompt ``i`` (`SYSTEMS`, then `SYSTEM_RULES` until about ``tokens`` tokens), framed as a chat-turn system message
    or as plain text, without the BOS."""
    bos = len(encode(""))
    text, rules = SYSTEMS[i % len(SYSTEMS)] + f"(Desk {i}.) ", ""
    while len(encode(text + rules)) - bos < tokens:
        rules += SYSTEM_RULES
    sys_head, sys_tail = CHAT_SYSTEM if chat else ("", "\n\n")
    return encode(sys_head + text + rules + sys_tail)[bos:]


def sampling_of(rng: Any, max_tokens: int) -> dict[str, Any]:
    sampling: dict[str, Any] = {"max_tokens": rng.randint(1, max_tokens), "temperature": 0.0}
    kind = rng.random()
    if kind >= 0.5:
        sampling.update({"temperature": 0.7, "top_p": 0.9} if kind < 0.75 else {"temperature": 1.0, "top_k": 50})
        if rng.random() < 0.5:
            sampling["seed"] = rng.randrange(1 << 31)
    if rng.random() < 0.2:
        sampling["stop"] = list(STOPS[rng.randrange(len(STOPS))])
    if rng.random() < 0.1:
        sampling["min_tokens"] = min(2, sampling["max_tokens"])
    return sampling


def traffic(encode: Any, n: int, *, seed: int, prompt_tokens: tuple[int, int], max_tokens: int, rate: float,
            chat: bool = True, frame: tuple[str, str] | None = None, system_prompts: int = 0, system_tokens: int = 384,
            turns: int = 0) -> list[dict[str, Any]]:
    """``n`` requests in arrival order: CERT-VARLEN's prompts and caps (the module docstring) arriving at Poisson engine
    steps, each with the sampling it asks for. ``encode(text)`` is the tokenizer's (special tokens added); ``frame`` the
    text around a user turn in place of CERT-VARLEN's (`template_frame`).

    With ``system_prompts`` N, each request opens with one of N system prompts (`system_prompt`, the user turn's prompt length
    then as drawn). With ``turns``, each conversation goes on for 0 … ``turns`` more turns, listed after the ``n`` first turns:
    a turn ``follows`` the one before it, and arrives ``gap`` steps after it finished, its prompt that turn's prompt, its
    streamed ids and ``suffix`` (the user's next question, framed), which `run` puts together then. With both 0, the draws
    and the requests are the same as without them."""
    lo, hi = prompt_tokens
    head, tail = frame or ((CHAT_HEAD, CHAT_TAIL) if chat else ("", "\n\nAnswer: "))
    bos = encode("")
    tail_ids = encode(tail)[len(bos):]
    filler, k = [], 1
    while len(filler) < hi:
        filler, k = encode(VARIED_FILLER * k)[len(bos):], 2 * k
    heads = [encode(head + q) for q in QUESTIONS]
    systems = [system_prompt(encode, i, system_tokens, chat) for i in range(system_prompts)]
    suffixes = [encode(TURN_END[chat] + head + q.split(" Ignore the notes")[0] + tail)[len(bos):] for q in QUESTIONS]
    rng, at, out, later = stream(seed), 0.0, [], []
    for j in range(n):
        at += rng.expovariate(rate)
        length = round(lo * (hi / lo) ** rng.random())
        h = heads[rng.randrange(len(heads))]
        user = ((h + filler)[:max(0, length - len(tail_ids))] + tail_ids)[:length]
        if systems:
            s = systems[rng.randrange(len(systems))]
            user = user[len(bos):] if user[:len(bos)] == bos else user
            prompt = bos + s + user
        else:
            prompt = user
        rid = f"varied.{j:05d}"
        out.append({"request_id": rid, "arrive_step": int(at), "prompt": prompt, "sampling": sampling_of(rng, max_tokens)})
        for t in range(1, rng.randint(0, turns) + 1 if turns else 1):
            later.append({"request_id": f"{rid}.t{t}", "follows": rid if t == 1 else f"{rid}.t{t - 1}", "gap": rng.randint(0, 4),
                          "suffix": suffixes[rng.randrange(len(suffixes))], "sampling": sampling_of(rng, max_tokens)})
    return out + later


def engine_options(opts: Options) -> dict[str, Any]:
    kw: dict[str, Any] = {"max_num_seqs": opts.max_num_seqs, "max_model_len": opts.max_model_len,
                          "max_num_batched_tokens": opts.max_num_batched_tokens, "gpu_memory_utilization": float(opts.gpu_memory_utilization),
                          "enforce_eager": True}
    if opts.kv_cache_mb:
        kw["kv_cache_memory_bytes"] = int(opts.kv_cache_mb) << 20
    if not opts.chunked_prefill:
        kw["enable_chunked_prefill"] = False
    if opts.prefix_caching:
        kw["enable_prefix_caching"] = True
    if opts.precision == "int8_exact":
        kw.update(quantization="verity_int8", attention_backend="CUSTOM")
    elif opts.precision == "verity_int8":
        kw.update(quantization="verity_int8")
    return kw


def build(opts: Options) -> tuple[Any, Any]:
    """The engine (`verity_vllm.LLM`) and its committer, serve-openai's rows-sha512 Commit (`serve_seal.committer`, with the
    step-input port and the sampler's inputs bound)."""
    from verity_vllm.engine.engine_profile import apply_env

    apply_env()
    if opts.precision != "pin":
        from verity_vllm.engine import int8_ranks
        int8_ranks.register(("linears", "attention") if opts.precision == "int8_exact" else ("linears",))
    from verity_vllm.llm import LLM

    llm = LLM(opts.model, revision=opts.revision, **engine_options(opts))
    if opts.precision == "int8_exact":
        from verity_vllm.program.frontend.kernel_readback import int8_exact_of
        if (got := int8_exact_of(llm.model)) != "all":
            raise RuntimeError(f"an int8_exact engine whose layers run int8 scope {got!r}, not 'all'")
    return llm, SS.committer(committer_opts(opts), llm, label="commit-arb")


class Windows:
    """The committer's windows over the stepped engine, each read back into one `served_record.BlockValues`."""

    def __init__(self, com: Any, record: Any, out: Path, window_steps: int) -> None:
        self.com, self.record, self.out, self.window_steps = com, record, out, int(window_steps)
        self.window, self.is_open, self.records = -1, False, []

    def before_step(self) -> None:
        if self.is_open and int(self.com._step_index) >= self.window_steps:
            self.close()
        if not self.is_open:
            self.window += 1
            self.com.reset()
            self.com.install()
            self.is_open = True

    def close(self) -> None:
        from verity_vllm.commit.served_record import window_commit

        if not self.is_open:
            return
        rc = self.com.finalize() if int(self.com._step_index) else None
        readback = self.record.absorb(self.com, self.window) if rc is not None else None
        self.com.uninstall()
        self.is_open = False
        rec = {"window": self.window, "commit": window_commit(rc) if rc is not None else {"passed": True, "steps": 0, "empty": True},
               "readback": readback}
        rec["passed"] = rec["commit"]["passed"] and (readback is None or readback["passed"])
        self.records.append(rec)
        with open(self.out / "windows.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def next_turn(q: dict[str, Any], parent: dict[str, Any], streamed: list[int], step: int, max_model_len: int) -> bool:
    """Turn ``q``'s prompt once the turn it follows finished at ``step``: that turn's prompt, its streamed ids and ``q``'s
    suffix, arriving ``gap`` steps on; False (``q`` skipped) when it and its ``max_tokens`` would pass ``max_model_len``."""
    prompt = list(parent["prompt"]) + list(streamed) + list(q["suffix"])
    if len(prompt) + int(q["sampling"]["max_tokens"]) > max_model_len:
        q["skipped"] = f"{len(prompt)} prompt ids and {q['sampling']['max_tokens']} tokens pass --max-model-len {max_model_len}"
        return False
    q["prompt"], q["arrive_step"] = prompt, step + 1 + int(q["gap"])
    return True


def prefix_negative(record: Any, checks: list[tuple[dict, dict | None, dict]], streamed: Mapping[str, list[int]],
                    seed_of: Callable[[dict], int]) -> dict[str, Any] | None:
    """The negative control on the run's own records: in the first request that read a prefix from the cache, the middle
    position of its first span flipped in the record of the request that committed it; both requests' checks must then fail.
    None when no request read from the cache."""
    import copy

    hit = next(((q, c) for q, _f, c in checks if c.get("prefix_spans")), None)
    if hit is None:
        return None
    q, c = hit
    src, lo, hi = c["prefix_spans"][0]
    p = (lo + hi) // 2
    flipped = copy.deepcopy(record.requests[src])
    flipped.inputs[p] += 1
    requests = {**record.requests, src: flipped}
    by_id = {x["request_id"]: x for x, _f, _c in checks}
    reader = record.requests[q["request_id"]].check(q["prompt"], streamed[q["request_id"]], seed=seed_of(q), prefix_sources=requests)
    owner = flipped.check(by_id[src]["prompt"], streamed[src], seed=seed_of(by_id[src]), prefix_sources=requests)
    return {"reader": q["request_id"], "source": src, "position": p, "reader_caught": not reader["passed"],
            "source_caught": not owner["passed"], "reader_problems": reader["problems"][:2], "source_problems": owner["problems"][:2],
            "caught": not reader["passed"] and not owner["passed"]}


def run(llm: Any, com: Any, requests: list[dict[str, Any]], out: Path, window_steps: int, *, max_model_len: int,
        prefix_caching: bool = False) -> dict[str, Any]:
    """Serve ``requests`` on the committing engine, step by step, and check every request's record (the module docstring)."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    from verity_vllm.commit.served_record import BlockValues
    from verity_vllm.engine import served_seed

    engine = llm.vllm.llm_engine
    seed_hooks = served_seed.install(engine)
    by_id = {q["request_id"]: q for q in requests}
    record = BlockValues(known=by_id)
    windows = Windows(com, record, out, window_steps)
    streamed: dict[str, list[int]] = {rid: [] for rid in by_id}
    finish: dict[str, dict[str, Any]] = {}
    cached: dict[str, int] = {}
    turns: dict[str, list[dict[str, Any]]] = {}
    for q in requests:
        if "follows" in q:
            turns.setdefault(q["follows"], []).append(q)
    pending, step, idle = [q for q in requests if "follows" not in q], 0, 0
    try:
        while pending or engine.has_unfinished_requests():
            while pending and pending[0]["arrive_step"] <= step:
                q = pending.pop(0)
                q["admitted_step"] = step
                engine.add_request(q["request_id"], {"prompt_token_ids": q["prompt"]},
                                   SamplingParams(**q["sampling"], output_kind=RequestOutputKind.DELTA))
            if not engine.has_unfinished_requests():
                windows.close()                               # a window ends when the engine is idle
                step, idle = step + 1, idle + 1
                continue
            windows.before_step()
            for o in engine.step():
                c = o.outputs[0]
                streamed[o.request_id].extend(int(t) for t in c.token_ids)
                if getattr(o, "num_cached_tokens", None) is not None:
                    cached.setdefault(o.request_id, int(o.num_cached_tokens))
                if o.finished:
                    finish[o.request_id] = {"reason": c.finish_reason, "stop_reason": c.stop_reason, "step": step}
                    for nq in turns.get(o.request_id, ()):
                        if next_turn(nq, by_id[o.request_id], streamed[o.request_id], step, max_model_len):
                            pending.append(nq)
                    pending.sort(key=lambda x: x["arrive_step"])
            step += 1
    finally:
        windows.close()
        seed_hooks.uninstall()
    for q in requests:
        if "follows" in q and "prompt" not in q and "skipped" not in q:
            q["skipped"] = f"the turn it follows ({q['follows']}) was not served to its finish"
    tok = llm.get_tokenizer()
    eos_ids = [int(tok.eos_token_id)] if tok.eos_token_id is not None else []
    decode = lambda ids: tok.decode(list(ids), skip_special_tokens=True)  # noqa: E731
    seed_of = lambda q: q["sampling"].get("seed", served_seed.DEFAULT_SEED)  # noqa: E731
    checks, skipped = [], []
    with open(out / "requests.jsonl", "w") as fh, open(out / "record.jsonl", "w") as fr:
        for q in requests:
            rid = q["request_id"]
            if "skipped" in q:
                skipped.append(rid)
                fh.write(json.dumps({"request_id": rid, "follows": q["follows"], "skipped": q["skipped"]}) + "\n")
                continue
            fin = finish.get(rid)
            ends = None if fin is None else {**fin, "max_tokens": q["sampling"]["max_tokens"],
                                             "min_tokens": q["sampling"].get("min_tokens", 0), "eos_ids": eos_ids}
            chk = record.check(rid, q["prompt"], streamed[rid], seed=seed_of(q), finish=ends, decode=decode)
            if fin is None:
                chk = {**chk, "passed": False, "problems": ["never finished", *chk.get("problems", [])]}
            chk["engine_cached"] = cached.get(rid)
            checks.append((q, fin, chk))
            fh.write(json.dumps({"request_id": rid, "follows": q.get("follows"), "arrive_step": q["arrive_step"],
                                 "admitted_step": q.get("admitted_step"), "sampling": q["sampling"], "prompt_ids": q["prompt"],
                                 "streamed_ids": streamed[rid], "finish": fin, "check": chk}) + "\n")
            if rid in record.requests:
                fr.write(json.dumps(record.requests[rid].to_json()) + "\n")
    negative = prefix_negative(record, checks, streamed, seed_of)
    return summarize(checks, windows.records, engine_steps=step, idle_steps=idle, prefix_caching=prefix_caching,
                     negative=negative, skipped=skipped)


def summarize(checks: list[tuple[dict, dict | None, dict]], windows: list[dict], *, engine_steps: int, idle_steps: int,
              prefix_caching: bool = False, negative: dict[str, Any] | None = None, skipped: Sequence[str] = ()) -> dict[str, Any]:
    """The run's counts and verdict. With ``prefix_caching``, it passes only when some request read a prefix from the cache
    and the negative control on the records (`prefix_negative`) was caught."""
    from collections import Counter

    failed = [q["request_id"] for q, _f, c in checks if not c["passed"]]
    finishes = Counter(("stop-string" if isinstance(f.get("stop_reason"), str) else str(f.get("reason"))) if f else "unfinished"
                       for _q, f, _c in checks)
    total = lambda k: sum(int(c.get(k) or 0) for _q, _f, c in checks)  # noqa: E731
    bad_windows = [w["window"] for w in windows if not w["passed"]]
    hits = sum(1 for _q, _f, c in checks if c.get("cache_hit"))
    prefix_problems = []
    if prefix_caching and not hits:
        prefix_problems.append("prefix caching was on and no request read a prefix from the cache")
    if hits and not (negative or {}).get("caught"):
        prefix_problems.append(f"the negative control (a flipped committed id in a reused prefix) was not caught: {negative}")
    return {"requests": len(checks), "passed_requests": len(checks) - len(failed), "failed_requests": failed[:20],
            "windows": len(windows), "failed_windows": bad_windows, "engine_steps": engine_steps, "idle_steps": idle_steps,
            "committed_steps": sum(int(w["commit"].get("steps") or 0) for w in windows),
            "finishes": dict(finishes),
            "temperatures": dict(Counter(str(q["sampling"].get("temperature")) for q, _f, _c in checks)),
            "client_seeds": sum("seed" in q["sampling"] for q, _f, _c in checks),
            "prompt_lengths": [min(len(q["prompt"]) for q, _f, _c in checks), max(len(q["prompt"]) for q, _f, _c in checks)] if checks else [],
            "streamed_lengths": [min(c["streamed_ids"] for _q, _f, c in checks), max(c["streamed_ids"] for _q, _f, c in checks)] if checks else [],
            "chunked_requests": sum(1 for _q, _f, c in checks if int(c.get("discarded_samples") or 0)),
            "discarded_samples": total("discarded_samples"), "preemptions": total("preemptions"),
            "recomputed_positions": total("recomputed_positions"), "overrun_rows": total("overrun_rows"),
            "cache_hits": hits, "cached_positions": total("cache_hit"),
            "multi_source_hits": sum(1 for _q, _f, c in checks if len(c.get("prefix_spans") or ()) > 1),
            "engine_cache_hits": sum(1 for _q, _f, c in checks if c.get("engine_cached")),
            "cache_hit_disagrees": [q["request_id"] for q, _f, c in checks
                                    if c.get("engine_cached") is not None and int(c.get("cache_hit") or 0) != c["engine_cached"]][:20],
            "turns": sum(1 for q, _f, _c in checks if q.get("follows")), "skipped_turns": len(skipped),
            "prefix_negative": negative, "prefix_problems": prefix_problems,
            "passed": bool(checks) and not failed and not bad_windows and not prefix_problems}


def scheduler_config(engine: Any) -> dict[str, Any]:
    """What the engine resolved of the settings that decide its schedules (async scheduling is vLLM's to resolve)."""
    vc = engine.vllm_config
    sc, cc = vc.scheduler_config, vc.cache_config
    return {**{k: getattr(sc, k, None) for k in ("async_scheduling", "enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs",
                                                 "max_model_len", "policy")},
            **{k: getattr(cc, k, None) for k in ("enable_prefix_caching", "block_size", "num_gpu_blocks")}}


def main(opts: Options) -> int:
    if not (opts.model and opts.revision):
        raise UsageError("commit-arb needs --model REPO --revision SHA, a pin of manifests/checkpoints.json")
    if opts.window_steps < 1 or opts.requests < 1 or opts.tokens < 1 or opts.rate <= 0:
        raise UsageError("--window-steps, --requests and --tokens are at least 1, --rate positive")
    lo, hi = prompt_range(opts.prompt_range)
    if min(opts.system_prompts, opts.turns) < 0:
        raise UsageError("--system-prompts and --turns are at least 0")
    if opts.chat == "template" and (opts.system_prompts or opts.turns):
        raise UsageError("--system-prompts and --turns frame CERT-VARLEN's chat turns or plain text, not --chat template")
    system = opts.system_tokens + 64 if opts.system_prompts else 0
    if hi + system + opts.tokens > opts.max_model_len:
        raise UsageError(f"--max-model-len {opts.max_model_len} holds no prompt of {hi + system} and {opts.tokens} tokens")
    preflight(opts)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    llm, com = build(opts)
    tok = llm.get_tokenizer()
    reqs = traffic(lambda s: list(tok.encode(s)), opts.requests, seed=opts.seed, prompt_tokens=(lo, hi), max_tokens=opts.tokens,
                   rate=opts.rate, chat=opts.chat != "plain", frame=template_frame(tok) if opts.chat == "template" else None,
                   system_prompts=opts.system_prompts, system_tokens=opts.system_tokens, turns=opts.turns)
    cfg = scheduler_config(llm.vllm.llm_engine)
    (out / "run.json").write_text(json.dumps({"options": {k: getattr(opts, k) for k in opts.__dataclass_fields__},
                                              "engine": llm.engine_record, "scheduler": cfg, "arena": com.arena_sizing_record,
                                              "committer": com.name, "scheme": com.commit_scheme,
                                              "required_manifest_path": com.required_manifest_path, "run_id": opts.run_id},
                                             indent=1, default=str))
    try:
        summary = run(llm, com, reqs, out, opts.window_steps, max_model_len=opts.max_model_len, prefix_caching=opts.prefix_caching)
    finally:
        com.close()
    summary["scheduler"] = cfg
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"COMMIT-ARB {'PASS' if summary['passed'] else 'FAIL'}: {json.dumps(summary, default=str)}", flush=True)
    return 0 if summary["passed"] else 1
