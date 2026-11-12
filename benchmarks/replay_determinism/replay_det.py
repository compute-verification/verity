"""Replay determinism of a served window without batch invariance, on one GPU: does replaying a window's recorded
schedule give the served bytes, and does a request alone give the bytes it got inside its batch?

One in-process vLLM engine (`verity_vllm.engine.env.ENGINE`, eager, prefix caching off, synchronous scheduling, a fixed
KV page count) per phase, each phase a fresh process, on Moonlight (`MODEL`) or any local checkpoint (a dense baseline);
``--batch-invariant`` serves a phase with ``VLLM_BATCH_INVARIANT=1`` instead:

- ``workload``: the window's requests, built from text in this tree in the model's chat format: prompt lengths and
  output lengths log-uniform, half greedy and half sampled with a per-request seed, every request ``ignore_eos`` so the
  schedule doesn't depend on the tokens, and Poisson arrivals;
- ``serve``: the requests added at their wall-clock arrival times while the engine steps; each arrival is recorded with the
  number of forward steps before it, and each step's batch (rows in the model runner's order, each row's request, first
  position and length, the KV pages each request gained);
- ``replay --record SERVED``: the same requests added before the same forward steps; the schedule must come out equal;
- ``alone``: each request served by itself, one after another.

With ``--capture`` a phase also fingerprints, at every forward step, every row (one token) of every module output of the
model, the top-k weights and ids each MoE layer's router selects, and every logits row (with its top 8): a row's
fingerprint is the wrapping int64 sum of its 32-bit (16-bit, 8-bit) words times fixed odd keys, so a single changed bit
changes it, and rows of runs are compared by fingerprint (`compare.py`).  The rows go to ``--out`` (``rows.npy``,
``keys.npy``: step, request, position and input token of each row) beside the step records (``steps.jsonl``) and tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"
REVISION = "4e735b07a89f73647dfab71ab91b840f362ede5b"
ALLOW = ["*.json", "*.safetensors", "*.py", "tiktoken.model"]
#: Moonlight's chat-format special tokens (tokenizer_config.json's added_tokens_decoder)
IM_END, IM_USER, IM_ASSISTANT, IM_SYSTEM, IM_MIDDLE = 163586, 163587, 163588, 163594, 163601
#: tokenization_moonshot.py's pre-tokenizer
PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
ASKS = ("Summarize the following text.\n\n", "List the terms this text defines, with one line on each.\n\n",
        "Explain the following to an engineer who is new to the project.\n\n", "Continue this document.\n\n")
#: the engine every phase builds
ENGINE_ARGS = dict(dtype="bfloat16", seed=0, max_model_len=8192, max_num_batched_tokens=2048, max_num_seqs=64,
                   enforce_eager=True, enable_prefix_caching=False, async_scheduling=False, gpu_memory_utilization=0.85,
                   num_gpu_blocks_override=8192, disable_log_stats=True, skip_tokenizer_init=True)
TOPK = 8
KEY_WORDS = 1 << 18
#: a column's value in a step where its label didn't fire
ABSENT = np.int64(0x5EED5EED5EED5EED)


def fingerprint_keys() -> tuple[np.ndarray, np.ndarray]:
    """The fixed odd keys: KEY_WORDS for a row's words, and those that fold a long row's chunks."""
    g = np.random.default_rng(0x5EED)
    return (g.integers(-(1 << 62), 1 << 62, KEY_WORDS, dtype=np.int64) | 1,
            g.integers(-(1 << 62), 1 << 62, 4096, dtype=np.int64) | 1)


def digest_rows(t: Any, rows: int, keys: Any, chunk_keys: Any) -> Any:
    """[rows] int64: each row's words times ``keys``, summed with wraparound, plus its byte length times a fixed odd
    constant; a row longer than the keys is cut into chunks whose sums are folded with ``chunk_keys``."""
    import torch

    b = t.detach().contiguous().view(torch.uint8).reshape(rows, -1)
    n = b.shape[1]
    w = b.view(torch.int32) if n % 4 == 0 else b.view(torch.int16) if n % 2 == 0 else b
    w = w.to(torch.int64)
    m, k = w.shape[1], len(keys)
    if m <= k:
        d = (w * keys[:m]).sum(dim=1)
    else:
        c = -(-m // k)
        w = torch.nn.functional.pad(w, (0, c * k - m)).view(rows, c, k)
        d = ((w * keys).sum(dim=2) * chunk_keys[:c]).sum(dim=1)
    return d + n * 0x9E3779B97F4A7C15 % (1 << 63)


def _identity(model_dir: Path) -> tuple[str, str]:
    """The checkpoint's name and revision: a hub snapshot's (``models--ORG--NAME/snapshots/REV``), else Moonlight's pin."""
    p = model_dir.resolve()
    if p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
        return p.parent.parent.name.removeprefix("models--").replace("--", "/"), p.name
    return MODEL, REVISION


def _chat(model_dir: Path) -> tuple[Any, list[int], list[int]]:
    """The model's text encoder and the chat format's tokens before and after the user's text: Moonlight's tiktoken format,
    else the checkpoint's own chat template, else (a base model) just its BOS token."""
    if (model_dir / "tiktoken.model").exists():
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        enc = tiktoken.Encoding(name="moonlight", pat_str=PAT, mergeable_ranks=load_tiktoken_bpe(str(model_dir / "tiktoken.model")),
                                special_tokens={})

        def encode(s: str) -> list[int]:
            return enc.encode(s, disallowed_special=())
        head = [IM_SYSTEM, *encode("system"), IM_MIDDLE, *encode("You are a helpful assistant"), IM_END,
                IM_USER, *encode("user"), IM_MIDDLE]
        return encode, head, [IM_END, IM_ASSISTANT, *encode("assistant"), IM_MIDDLE]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))

    def encode(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)
    if not tok.chat_template:
        return encode, [tok.bos_token_id] if tok.bos_token_id is not None else [], []
    mark = "\x00USER\x00"
    text = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": mark}],
                                   tokenize=False, add_generation_prompt=True, date_string="06 Nov 2026")
    before, after = text.split(mark)
    return encode, encode(before), encode(after)


def workload(model_dir: Path, corpus_root: Path, n: int, seed: int, out: Path) -> None:
    encode, head, tail = _chat(model_dir)
    paths = sorted(p for p in corpus_root.rglob("*.md")
                   if not any(part.startswith(".") or part == "node_modules" for part in p.relative_to(corpus_root).parts))
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths)[:4_000_000]
    corpus = encode(text)
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        body = int(round(math.exp(rng.uniform(math.log(16), math.log(6000)))))
        max_tokens = int(round(math.exp(rng.uniform(math.log(8), math.log(512)))))
        ask = encode(ASKS[int(rng.integers(len(ASKS)))])
        body = min(body, ENGINE_ARGS["max_model_len"] - max_tokens - len(head) - len(tail) - len(ask))
        off = int(rng.integers(0, len(corpus) - body))
        sampled = i % 2 == 1
        if i >= 4:
            t += float(rng.exponential(0.15))
        reqs.append({"id": f"r{i:03d}", "arrival": round(t, 4), "prompt": [*head, *ask, *corpus[off:off + body], *tail],
                     "max_tokens": max_tokens, "temperature": 0.8 if sampled else 0.0, "top_p": 0.95 if sampled else 1.0,
                     "seed": 1000 + i if sampled else None})
    out.parent.mkdir(parents=True, exist_ok=True)
    model, revision = _identity(model_dir)
    out.write_text(json.dumps({"model": model, "revision": revision, "seed": seed, "corpus_files": len(paths), "requests": reqs}))
    lens = [len(r["prompt"]) for r in reqs]
    print(f"workload: {n} requests, prompts {min(lens)}..{max(lens)} tokens (total {sum(lens)}), "
          f"outputs {sum(r['max_tokens'] for r in reqs)} tokens, last arrival {t:.2f}s")


def _pin_env(batch_invariant: bool) -> dict[str, str]:
    from verity_vllm.engine.env import ENGINE, pin, served_defaults

    cleared = served_defaults()
    if batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    pinned = pin({**ENGINE, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1", "HF_HUB_OFFLINE": "1"})
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0 (run.sh sets it): the scheduler and the runner iterate sets of request ids")
    return {"pinned": pinned, "cleared": cleared, "VLLM_BATCH_INVARIANT": "1" if batch_invariant else None}


def _engine(model_dir: Path) -> Any:
    from verity_vllm.engine import stock

    return stock.engine(str(model_dir), **ENGINE_ARGS)


def _kernels(model: Any) -> dict[str, list[str]]:
    """The attention implementations, quantization methods and MoE kernels the model's layers hold, by class name."""
    found: dict[str, set[str]] = {"impl": set(), "quant_method": set(), "moe": set()}
    for _, mod in model.named_modules():
        for attr in ("impl", "quant_method"):
            if (v := getattr(mod, attr, None)) is not None and not isinstance(v, (str, int)):
                found[attr].add(type(v).__name__)
        if (v := getattr(mod, "routed_experts", None)) is not None:
            qm = getattr(v, "quant_method", None)
            found["moe"].add(f"{type(v).__name__}/{type(qm).__name__}/{type(getattr(qm, 'moe_kernel', None) or getattr(qm, 'fused_experts', None)).__name__}")
    return {k: sorted(v) for k, v in found.items()}


def _tensors(out: Any) -> list[Any]:
    import torch

    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for o in out for t in _tensors(o)]
    return []


class Recorder:
    """Every forward step's batch, from the model runner (`execute_model`, `sample_tokens`), and with ``capture`` every row's
    fingerprint of every module output, router selection and logits row."""

    def __init__(self, engine: Any, capture: bool) -> None:
        from verity_vllm.engine.handle import model_runner
        from verity_vllm.engine.hooks import Hooks

        self.runner = model_runner(engine)
        self.model = self.runner.model
        self.capture = capture
        self.hooks = Hooks(tag="replay-determinism")
        self.handles: list[Any] = []
        self.steps: list[dict[str, Any]] = []
        self.executed = 0
        self.labels: dict[str, int] = {}
        self.whole: list[dict[str, str]] = []
        self.row_blocks: list[np.ndarray] = []
        self.key_blocks: list[np.ndarray] = []
        self.logit_keys: list[np.ndarray] = []
        self.logit_digest: list[np.ndarray] = []
        self.logit_top: list[np.ndarray] = []
        self.logit_topi: list[np.ndarray] = []
        self._cur: dict[str, Any] | None = None
        self._step_rows: dict[str, Any] = {}
        self._step_whole: dict[str, Any] = {}
        self._seen: dict[str, int] = {}
        self._t_pad = 0
        self._keys = None
        self._chunk_keys = None

    def install(self) -> Recorder:
        import torch

        if self.capture:
            dev = next(self.model.parameters()).device
            self._keys, self._chunk_keys = (torch.from_numpy(k).to(dev) for k in fingerprint_keys())
            self.handles.append(self.model.register_forward_pre_hook(self._pre, with_kwargs=True))
            for name, mod in self.model.named_modules():
                if name == "logits_processor":
                    self.handles.append(mod.register_forward_hook(self._logits))
                elif name:
                    self.handles.append(mod.register_forward_hook(self._make_fwd(name)))
                router = getattr(mod, "router", None)
                if router is not None and callable(getattr(router, "select_experts", None)):
                    self.hooks.wrap(router, "select_experts", self._make_router(f"{name}.router.select_experts"))
        self.hooks.wrap(self.runner, "execute_model", self._execute_model)
        self.hooks.wrap(self.runner, "sample_tokens", self._sample_tokens)
        return self

    def uninstall(self) -> None:
        for h in self.handles:
            h.remove()
        self.hooks.uninstall()

    # --- fingerprints ---------------------------------------------------------------------------------------------------

    def _digest_rows(self, t: Any, rows: int) -> Any:
        return digest_rows(t, rows, self._keys, self._chunk_keys)

    def _put(self, label: str, t: Any) -> None:
        k = self._seen.get(label, 0)
        self._seen[label] = k + 1
        label = label if k == 0 else f"{label}#{k}"
        if t.dim() >= 1 and t.shape[0] == self._t_pad and t.numel() > 0:
            self._step_rows[label] = self._digest_rows(t, self._t_pad)
        else:
            self._step_whole[label] = self._digest_rows(t.reshape(1, -1) if t.numel() else t.new_zeros(1, 1), 1)

    def _pre(self, module: Any, args: tuple, kwargs: dict) -> None:
        if self._cur is not None:
            pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
            self._t_pad = int(pos.shape[-1])

    def _make_fwd(self, name: str) -> Any:
        def hook(module: Any, args: tuple, out: Any) -> None:
            if self._cur is None or self._cur["phase"] != "forward":
                return
            ts = _tensors(out)
            for i, t in enumerate(ts):
                self._put(name if len(ts) == 1 else f"{name}:{i}", t)
        return hook

    def _make_router(self, label: str) -> Any:
        def make(orig: Any) -> Any:
            def select_experts(*a: Any, **kw: Any) -> Any:
                w, ids = orig(*a, **kw)
                if self._cur is not None and self._cur["phase"] == "forward":
                    self._put(f"{label}.weights", w)
                    self._put(f"{label}.ids", ids)
                return w, ids
            return select_experts
        return make

    def _logits(self, module: Any, args: tuple, out: Any) -> None:
        import torch

        if self._cur is None or self._cur["phase"] != "sample":
            return
        logits = out.float()
        top = torch.topk(logits, TOPK, dim=-1)
        self._cur["logits"] = (self._digest_rows(out, out.shape[0]), top.values, top.indices)

    # --- the runner's step ---------------------------------------------------------------------------------------------

    def _execute_model(self, orig: Any) -> Any:
        def execute_model(scheduler_output: Any, *a: Any, **kw: Any) -> Any:
            dummy = kw.get("dummy_run", False)
            if dummy or scheduler_output.total_num_scheduled_tokens == 0:
                return orig(scheduler_output, *a, **kw)
            self._cur = {"phase": "forward", "so": scheduler_output}
            self._step_rows, self._step_whole, self._seen = {}, {}, {}
            out = orig(scheduler_output, *a, **kw)
            ib = self.runner.execute_model_state.input_batch
            T, R = int(ib.num_tokens), int(ib.num_reqs)
            self._cur.update(phase="sampled", T=T, T_pad=int(ib.num_tokens_after_padding), req_ids=list(ib.req_ids[:R]),
                             qsl=np.array(ib.query_start_loc_np[:R + 1], dtype=np.int64),
                             start=np.array(ib.num_computed_tokens_np[:R], dtype=np.int64),
                             positions=ib.positions[:T].clone(), input_ids=ib.input_ids[:T].clone())
            return out
        return execute_model

    def _sample_tokens(self, orig: Any) -> Any:
        def sample_tokens(*a: Any, **kw: Any) -> Any:
            if self._cur is None:
                return orig(*a, **kw)
            self._cur["phase"] = "sample"
            out = orig(*a, **kw)
            self._finish_step()
            return out
        return sample_tokens

    def _finish_step(self) -> None:
        import torch

        cur, self._cur = self._cur, None
        so, T, rids, qsl, start = cur["so"], cur["T"], cur["req_ids"], cur["qsl"], cur["start"]
        R = len(rids)
        pages: dict[str, list[int]] = {}
        for r in so.scheduled_new_reqs:
            pages[r.req_id] = [b for g in r.block_ids for b in g]
        cached = so.scheduled_cached_reqs
        for rid, ids in zip(cached.req_ids, cached.new_block_ids):
            if ids is not None and any(len(g) for g in ids):
                pages[rid] = [b for g in ids for b in g]
        step = self.executed
        self.steps.append({"step": step, "T": T, "T_pad": cur["T_pad"],
                           "rows": [[rid, int(start[i]), int(qsl[i + 1] - qsl[i])] for i, rid in enumerate(rids)],
                           "pages": pages, "new": [r.req_id for r in so.scheduled_new_reqs],
                           "resumed": sorted(getattr(cached, "resumed_req_ids", None) or ()),
                           "preempted": sorted(getattr(so, "preempted_req_ids", None) or ()),
                           "finished": sorted(so.finished_req_ids)})
        self.executed += 1
        if not self.capture:
            return
        for label in self._step_rows:
            self.labels.setdefault(label, len(self.labels))
        cols = torch.full((len(self.labels), T), int(ABSENT), dtype=torch.int64, device=self._keys.device)
        for label, d in self._step_rows.items():
            cols[self.labels[label]] = d[:T]
        pos = cur["positions"].to(torch.int64)
        ids = cur["input_ids"].to(torch.int64)
        if "logits" not in cur:
            raise RuntimeError(f"step {step}: logits_processor never ran in sample_tokens; the logits hook sees nothing")
        digest, top, topi = cur["logits"]
        if digest.shape[0] != R:
            raise RuntimeError(f"step {step}: {digest.shape[0]} logits rows for {R} requests")
        whole = [d.flatten() for d in self._step_whole.values()]
        packed = torch.cat([cols.flatten(), pos, ids, digest.to(torch.int64), topi.to(torch.int64).flatten(),
                            top.float().contiguous().view(torch.int32).to(torch.int64).flatten(), *whole]).cpu().numpy()
        o = 0
        rows = packed[o:o + cols.numel()].reshape(len(self.labels), T).T
        o += cols.numel()
        pos_np, ids_np = packed[o:o + T], packed[o + T:o + 2 * T]
        o += 2 * T
        dig_np = packed[o:o + R]
        o += R
        topi_np = packed[o:o + R * TOPK].reshape(R, TOPK)
        o += R * TOPK
        top_np = packed[o:o + R * TOPK].astype(np.int32).view(np.float32).reshape(R, TOPK)
        o += R * TOPK
        self.whole.append({k: hex(int(v)) for k, v in zip(self._step_whole, packed[o:])})
        req = np.repeat(np.array([int(r.rsplit("-", 1)[0].lstrip("r")) for r in rids], dtype=np.int64), np.diff(qsl))
        self.row_blocks.append(np.ascontiguousarray(rows))
        self.key_blocks.append(np.stack([np.full(T, step), req, pos_np, ids_np], axis=1))
        last = start + np.diff(qsl) - 1
        self.logit_keys.append(np.stack([np.full(R, step), req[qsl[1:] - 1], last], axis=1))
        self.logit_digest.append(dig_np)
        self.logit_topi.append(topi_np)
        self.logit_top.append(top_np)

    def save(self, out: Path) -> None:
        with (out / "steps.jsonl").open("w") as f:
            for s in self.steps:
                f.write(json.dumps(s, separators=(",", ":")) + "\n")
        if not self.capture:
            return
        L = len(self.labels)
        rows = np.concatenate([np.pad(b, ((0, 0), (0, L - b.shape[1])), constant_values=ABSENT) for b in self.row_blocks])
        np.save(out / "rows.npy", rows)
        np.save(out / "keys.npy", np.concatenate(self.key_blocks))
        np.save(out / "logit_keys.npy", np.concatenate(self.logit_keys))
        np.save(out / "logit_digest.npy", np.concatenate(self.logit_digest))
        np.save(out / "logit_top.npy", np.concatenate(self.logit_top))
        np.save(out / "logit_topi.npy", np.concatenate(self.logit_topi))
        (out / "labels.json").write_text(json.dumps(sorted(self.labels, key=self.labels.get)))
        with (out / "whole.jsonl").open("w") as f:
            for w in self.whole:
                f.write(json.dumps(w, separators=(",", ":")) + "\n")


def _params(r: dict[str, Any]) -> Any:
    from verity_vllm.engine import stock

    return stock.sampling(n=1, temperature=r["temperature"], top_p=r["top_p"], seed=r["seed"], max_tokens=r["max_tokens"],
                          ignore_eos=True, detokenize=False)


def _collect(outs: list[Any], tokens: dict[str, list[int]]) -> None:
    for o in outs:
        if o.finished:
            tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]


def run(mode: str, model_dir: Path, wl: Path, out: Path, capture: bool, record: Path | None, batch_invariant: bool) -> None:
    t0 = time.monotonic()
    env = _pin_env(batch_invariant)
    out.mkdir(parents=True, exist_ok=True)
    reqs = json.loads(wl.read_text())["requests"]
    engine = _engine(model_dir)
    t_ready = time.monotonic()
    rec = Recorder(engine, capture).install()
    tokens: dict[str, list[int]] = {}
    arrivals: list[dict[str, Any]] = []
    by_id = {r["id"]: r for r in reqs}
    if mode == "serve":
        pending = sorted(reqs, key=lambda r: r["arrival"])
        start = time.monotonic()
        while pending or engine.has_unfinished_requests():
            now = time.monotonic() - start
            while pending and pending[0]["arrival"] <= now:
                r = pending.pop(0)
                engine.add_request(r["id"], {"prompt_token_ids": r["prompt"]}, _params(r))
                arrivals.append({"id": r["id"], "before_step": rec.executed, "t": round(now, 4)})
            if not engine.has_unfinished_requests():
                time.sleep(max(0.0, pending[0]["arrival"] - now))
                continue
            _collect(engine.step(), tokens)
        (out / "arrivals.json").write_text(json.dumps(arrivals))
    elif mode == "replay":
        pending = json.loads((record / "arrivals.json").read_text())
        while pending or engine.has_unfinished_requests():
            while pending and pending[0]["before_step"] == rec.executed:
                r = by_id[pending.pop(0)["id"]]
                engine.add_request(r["id"], {"prompt_token_ids": r["prompt"]}, _params(r))
            if pending and pending[0]["before_step"] < rec.executed:
                raise SystemExit(f"replay passed step {pending[0]['before_step']} before {pending[0]['id']} arrived")
            if not engine.has_unfinished_requests():
                raise SystemExit(f"replay idle at step {rec.executed} with {pending[0]['id']} due at step {pending[0]['before_step']}")
            before = rec.executed
            _collect(engine.step(), tokens)
            if rec.executed > before + 1:
                raise SystemExit(f"engine step at {before} ran {rec.executed - before} forward steps, not one")
    elif mode == "alone":
        for r in reqs:
            engine.add_request(r["id"], {"prompt_token_ids": r["prompt"]}, _params(r))
            while engine.has_unfinished_requests():
                _collect(engine.step(), tokens)
    else:
        raise SystemExit(f"unknown mode {mode}")
    t_done = time.monotonic()
    rec.uninstall()
    rec.save(out)
    missing = sorted(set(by_id) - set(tokens))
    if missing:
        raise SystemExit(f"{len(missing)} requests never finished: {missing[:8]}")
    (out / "tokens.json").write_text(json.dumps({k: tokens[k] for k in sorted(tokens)}))
    import torch

    from verity_vllm.engine import stock

    (out / "timing.json").write_text(json.dumps({
        "mode": mode, "capture": capture, "load_s": t_ready - t0, "serve_s": t_done - t_ready, "total_s": time.monotonic() - t0,
        "steps": rec.executed, "tokens_out": sum(map(len, tokens.values())), "env": env, "vllm": stock.version(), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(), "engine_args": ENGINE_ARGS, "kernels": _kernels(rec.model),
    }, indent=1, default=str))
    print(f"{mode}: {rec.executed} steps, {sum(map(len, tokens.values()))} tokens out, load {t_ready - t0:.1f}s, run {t_done - t_ready:.1f}s")


def fetch(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION, allow_patterns=ALLOW, local_dir=str(dest))
    print(f"{MODEL}@{REVISION} in {dest}: {sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) / 1e9:.1f} GB")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the model's pinned revision")
    f.add_argument("--dir", type=Path, required=True)
    w = sub.add_parser("workload", help="build the window's requests")
    w.add_argument("--model", type=Path, required=True)
    w.add_argument("--corpus", type=Path, required=True, help="a directory whose *.md files the prompts are cut from")
    w.add_argument("--n", type=int, default=48)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--out", type=Path, required=True)
    for mode in ("serve", "replay", "alone"):
        p = sub.add_parser(mode)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--workload", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--capture", action="store_true", help="fingerprint every row of every module output and logits row")
        p.add_argument("--batch-invariant", action="store_true", help="serve with VLLM_BATCH_INVARIANT=1")
        if mode == "replay":
            p.add_argument("--record", type=Path, required=True, help="the serve phase's --out")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.dir)
    elif a.cmd == "workload":
        workload(a.model, a.corpus, a.n, a.seed, a.out)
    else:
        run(a.cmd, a.model, a.workload, a.out, a.capture, getattr(a, "record", None), a.batch_invariant)


if __name__ == "__main__":
    sys.exit(main())
