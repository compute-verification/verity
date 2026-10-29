"""One serving engine of a whole-node run (run.sh's SERVE=node, `node.py` its prover): vLLM in process on one GPU, its
weights packed into one arena the store's shard on that GPU reads in place, served from the encoded store by decoding
each weight as its module runs.

~~~text
arena      at load (`GPUModelRunner.load_model`, after), every parameter of two or more dimensions copied into one uint8
           buffer of --arena-gib, each from a chunk boundary in module order, and the parameter pointed at its copy;
           the rest of the buffer zeros (through host memory, one engine at a time, when the GPU can't hold the buffer
           beside the loaded weights: `Arena`). The buffer is exported over CUDA IPC (`engine.<g>.json` in --dir: the
           GPU's UUID, the handle, the offset and size, the weights' chunks and each entry's name, offset and bytes),
           so the prover's pass reads these bytes as they are served. Norms and other one-dimensional parameters stay
           outside it
plain      a greedy probe alone in the batch (`probe`), then the arena's SHA-256 a GiB at a time (`plain_sha256` in
           the status, which `node.py --model-dir` holds to the pinned checkpoint's arena), then the steady load
           (`Load`) at --step-hz, with --align-steps at the host clock's multiples of 1/hz so the node's engines
           step together: `status.<g>.json` says ready, then baseline once --plain-s of it has run
encode     the prover's ctl.json sets pause: the engine aborts what it serves, waits for its GPU, says paused and
           steps no more while the prover encodes the arena in place (`device.DeviceStore.encode`)
decode     ctl.json brings the registration's key and the arena's first block in the store: the engine sets the key
           in kv.so, decodes the whole arena a GiB at a time against the plain SHA-256s, runs the probe again against
           the plain one, then hooks every module that holds an arena weight (a forward pre-hook decodes it at its
           store index into a scratch buffer and points the parameter there, a post-hook points it back; the LM head
           by the logits processor's hooks, since vLLM reads it there without calling it), says decoding with both
           results, and serves again
stop       ctl.json's stop: each phase's steps, tokens and step times (`summary`) to --out/engine.<g>.json, and exit
~~~

Every step's wall time, tokens and phase (ctl.json's, or plain-baseline for the first --plain-s) go to
--out/engine.<g>.steps.jsonl. The probe and the hash are the gate that decoding at the store's indices gives back the
served bytes, bit for bit, before anything is timed.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=3 python engine.py --gpu 3 --dir B --out OUT --kv kv.so
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

BLOCK = 4096
GIB = 1 << 30
RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")


def pack(sizes: list[int], capacity: int) -> list[int]:
    """Each of `sizes` bytes' offset in an arena of `capacity`, each from a chunk boundary, in order."""
    out, at = [], 0
    for n in sizes:
        out.append(at)
        at += -(-n // BLOCK) * BLOCK
    if at > capacity:
        raise ValueError(f"the weights take {at} bytes in blocks, more than the arena's {capacity}")
    return out


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


class IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def export(ptr: int) -> dict:
    """The CUDA IPC handle of the allocation holding device pointer `ptr`, and ptr's offset in it."""
    cu = ctypes.CDLL("libcuda.so.1")
    base, size = ctypes.c_uint64(), ctypes.c_size_t()
    if rc := cu.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr)):
        raise RuntimeError(f"cuMemGetAddressRange: {rc}")
    h = IpcHandle()
    if rc := cu.cuIpcGetMemHandle(ctypes.byref(h), ctypes.c_uint64(base.value)):
        raise RuntimeError(f"cuIpcGetMemHandle: {rc} (an expandable-segments allocation has no IPC handle)")
    return {"handle": ctypes.string_at(ctypes.addressof(h), 64).hex(), "offset": ptr - base.value,
            "allocation": size.value}


class Decoder:
    """kv.so's decode at the registration's key, on this process's one GPU."""

    def __init__(self, path: str) -> None:
        self.so = ctypes.CDLL(path)
        v, q, i = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        self.so.kv_set_key.argtypes = [v, i, v]
        self.so.kv_set_mask.argtypes = [v]
        self.so.kv_decode.argtypes = [v, v, q, q, i, i, v, v]
        self.masked = bool(self.so.kv_mask())

    def set_key(self, key: dict) -> None:
        """`key` as node.py sends it: its round constants, tau and mask, each little-endian words, as hex."""
        if key["masked"] != self.masked or key["bits"] != self.so.kv_bits():
            raise ValueError(f"kv.so is built for {self.so.kv_bits()} bits, masked={self.masked}, not the key's")
        self.rounds, self.words = key["rounds"], BLOCK // key["word_bytes"]
        c, tau = (ctypes.create_string_buffer(bytes.fromhex(key[k])) for k in ("c", "tau"))
        if rc := self.so.kv_set_key(ctypes.cast(c, ctypes.c_void_p), self.rounds, ctypes.cast(tau, ctypes.c_void_p)):
            raise RuntimeError(f"kv_set_key: {rc}")
        if self.masked:
            m = ctypes.create_string_buffer(bytes.fromhex(key["mask"]))
            if rc := self.so.kv_set_mask(ctypes.cast(m, ctypes.c_void_p)):
                raise RuntimeError(f"kv_set_mask: {rc}")

    def decode(self, src: int, dst: int, blocks: int, first: int, stream: int) -> None:
        """`blocks` chunks at device address src, the store's chunks first … first + blocks − 1, decoded to dst."""
        if rc := self.so.kv_decode(src, dst, blocks * self.words, first * self.words, self.rounds, 0, None, stream):
            raise RuntimeError(f"kv_decode: {rc}")


@contextlib.contextmanager
def one_at_a_time(path: Path):
    """An exclusive lock on `path` (made if absent) while the block runs, released when the process ends."""
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Arena:
    """The engine's weights in one buffer (`pack`): `entries`, each (offset, bytes, parameter, its modules). When the
    GPU can't hold the buffer beside the loaded weights (a 32B model's 61 GiB on a 96 GB GPU), the weights go to host
    memory first and come back into the buffer, under `lock` so that one engine of the node does it at a time."""

    def __init__(self, torch, model, capacity: int, lock: Path | None = None) -> None:
        self.torch, seen, params = torch, {}, []
        for mname, mod in model.named_modules():
            for pname, p in mod._parameters.items():
                if p is None or p.dim() < 2 or not p.is_cuda or not p.is_contiguous():
                    continue
                if id(p) in seen:
                    seen[id(p)][1].append((mname, mod, pname))
                    continue
                seen[id(p)] = (p, [(mname, mod, pname)])
                params.append(id(p))
        sizes = [seen[k][0].numel() * seen[k][0].element_size() for k in params]
        offsets = pack(sizes, capacity)
        device = seen[params[0]][0].device
        t0 = time.monotonic()
        self.staged = torch.cuda.mem_get_info(device)[0] < capacity + GIB
        if self.staged and lock is None:
            raise ValueError("the arena needs the weights staged through host memory, and a lock to take turns")
        with one_at_a_time(lock) if self.staged else contextlib.nullcontext():
            if self.staged:
                for k in params:
                    seen[k][0].data = seen[k][0].data.cpu()
                torch.cuda.empty_cache()
            self.buf = torch.zeros(capacity, dtype=torch.uint8, device=device)
            self.entries = []
            for k, off, n in zip(params, offsets, sizes):
                p, owners = seen[k]
                self.buf[off:off + n].copy_(p.data.reshape(-1).view(torch.uint8))
                p.data = self.buf[off:off + n].view(p.dtype).view(p.shape)
                self.entries.append((off, n, p, owners))
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.pack_s = time.monotonic() - t0
        self.weights_bytes = sum(sizes)
        self.weights_blocks = (offsets[-1] + -(-sizes[-1] // BLOCK) * BLOCK) // BLOCK
        self.outside_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) - self.weights_bytes

    def view(self, off: int, n: int, p):
        return self.buf[off:off + n].view(p.dtype).view(p.shape)

    def hashes(self, scratch, decode=None) -> list[str]:
        """The arena's SHA-256 a GiB at a time, each GiB as it is or, with `decode` (first block, Decoder), decoded."""
        torch, out = self.torch, []
        for at in range(0, self.buf.numel(), GIB):
            n = min(GIB, self.buf.numel() - at)
            if decode is None:
                chunk = self.buf[at:at + n]
            else:
                first, dec = decode
                dec.decode(self.buf.data_ptr() + at, scratch.data_ptr(), n // BLOCK, first + at // BLOCK,
                           torch.cuda.current_stream().cuda_stream)
                chunk = scratch[:n]
            out.append(hashlib.sha256(chunk.cpu().numpy()).hexdigest())
        return out


class Hooks:
    """Decode at use (`install`): each module holding arena weights decodes them into `scratch` before its forward and
    points them back after; the LM head by the logits processor's forward, which reads it without calling it."""

    def __init__(self, torch, arena: Arena, scratch, decoder: Decoder, first: int, model) -> None:
        self.torch, self.arena, self.scratch, self.dec, self.first = torch, arena, scratch, decoder, first
        self.calls, self.handles = 0, []
        lm_head = getattr(model, "lm_head", None)
        by_module: dict[int, tuple] = {}
        for off, n, p, owners in arena.entries:
            for mname, mod, pname in owners:
                target = getattr(model, "logits_processor", None) if mod is lm_head else mod
                if target is None:
                    raise ValueError(f"{mname} holds an arena weight and no module reads it")
                by_module.setdefault(id(target), (target, []))[1].append((off, n, p))
        self.modules = list(by_module.values())

    def install(self) -> None:
        for mod, items in self.modules:
            self.handles.append(mod.register_forward_pre_hook(lambda m, a, items=items: self.before(items)))
            self.handles.append(mod.register_forward_hook(lambda m, a, o, items=items: self.after(items)))

    def before(self, items) -> None:
        stream, at = self.torch.cuda.current_stream().cuda_stream, 0
        for off, n, p in items:
            self.dec.decode(self.arena.buf.data_ptr() + off, self.scratch.data_ptr() + at, -(-n // BLOCK),
                            self.first + off // BLOCK, stream)
            p.data = self.scratch[at:at + n].view(p.dtype).view(p.shape)
            at += -(-n // BLOCK) * BLOCK
        self.calls += 1

    def after(self, items) -> None:
        for off, n, p in items:
            p.data = self.arena.view(off, n, p)


def aligned(perf: float, clock: float, interval: float) -> float:
    """The perf counter's reading at the host clock's next multiple of `interval`, given both read at one moment."""
    return perf + interval - clock % interval


class Load:
    """The steady load: `concurrency` requests in flight, each `prompt` random tokens and `tokens` greedy ones (the
    end of sequence ignored), new ones added as others finish, one engine step every 1/`hz` seconds. With `align`,
    each step starts at the next multiple of 1/`hz` of the host's clock, so the node's engines step together: a pass
    reads every GPU's shard, and steps at independent times slow it whenever any one GPU is in a step."""

    def __init__(self, engine, concurrency: int, prompt: int, tokens: int, hz: float, seed: int,
                 align: bool = False) -> None:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        self.engine, self.concurrency, self.prompt, self.interval = engine, concurrency, prompt, 1.0 / hz
        self.align = align
        self.params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True, detokenize=False,
                                     output_kind=RequestOutputKind.DELTA)
        self.rng, self.live, self.n, self.next_at = random.Random(seed), set(), 0, time.perf_counter()

    def step(self) -> dict:
        while len(self.live) < self.concurrency:
            prompt = [self.rng.randrange(1000, 120000) for _ in range(self.prompt)]
            self.engine.add_request(f"load-{self.n}", {"prompt_token_ids": prompt}, self.params)
            self.live.add(f"load-{self.n}")
            self.n += 1
        t0 = time.perf_counter()
        outs = self.engine.step()
        t1 = time.perf_counter()
        tokens = 0
        for o in outs:
            tokens += sum(len(c.token_ids) for c in o.outputs)
            if o.finished:
                self.live.discard(o.request_id)
        now = time.time()
        if self.align:
            self.next_at = aligned(time.perf_counter(), now, self.interval)
        else:
            self.next_at = max(self.next_at + self.interval, t1)
        return {"t": now, "ms": (t1 - t0) * 1e3, "tokens": tokens, "running": len(self.live)}

    def wait(self) -> float:
        """Seconds until the next step is due."""
        return max(0.0, self.next_at - time.perf_counter())

    def abort(self) -> None:
        if self.live:
            self.engine.abort_request(list(self.live))
        self.live.clear()


def probe(engine, tag: str, prompts: list[list[int]], tokens: int) -> list[list[int]]:
    """Greedy completions of `prompts`, alone in the batch."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True, detokenize=False)
    ids = [f"probe-{tag}-{k}" for k in range(len(prompts))]
    for i, p in zip(ids, prompts):
        engine.add_request(i, {"prompt_token_ids": p}, sp)
    got = {}
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                got[o.request_id] = list(o.outputs[0].token_ids)
    return [got[i] for i in ids]


def summary(steps: list[dict]) -> dict:
    """Each phase's steps: their count, tokens, span (first start to last end, s), tokens a second over it, and step
    time percentiles (ms)."""
    out = {}
    for s in steps:
        out.setdefault(s["phase"], []).append(s)
    res = {}
    for phase, xs in out.items():
        ms = sorted(x["ms"] for x in xs)
        span = xs[-1]["t"] - xs[0]["t"] + xs[-1]["ms"] / 1e3
        at = lambda q: ms[min(len(ms) - 1, int(q * len(ms)))]
        tokens = sum(x["tokens"] for x in xs)
        res[phase] = {"steps": len(xs), "tokens": tokens, "span_s": span, "tokens_per_s": tokens / span if span else None,
                      "steps_per_s": len(xs) / span if span else None, "step_ms": {"p50": at(0.5), "p99": at(0.99),
                                                                                  "max": ms[-1]}}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, required=True, help="this engine's place among the node's (its shard's)")
    ap.add_argument("--dir", type=Path, required=True, help="the handshake directory node.py reads and writes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kv", required=True, help="kv.so at the registration's codec")
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B")
    ap.add_argument("--arena-gib", type=int, default=16, help="the arena, whole GiB (the store's chunks)")
    ap.add_argument("--util", type=float, default=0.45, help="vLLM's gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--step-hz", type=float, default=2.0, help="engine steps a second, at most")
    ap.add_argument("--align-steps", action="store_true",
                    help="step at the host clock's multiples of 1/hz, every engine of the node together")
    ap.add_argument("--plain-s", type=float, default=60.0, help="the plain baseline's seconds")
    ap.add_argument("--probe-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args()
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise SystemExit("run with VLLM_ENABLE_V1_MULTIPROCESSING=0: the arena is made in this process")
    import importlib

    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    g, d = a.gpu, a.dir
    status = lambda state, **kw: write_json(d / f"status.{g}.json", {"gpu": g, "state": state, "pid": os.getpid(), **kw})
    status("loading")
    made: dict = {}

    def wrap(cls) -> None:
        load = cls.load_model

        def load_model(self, *args, **kw):
            load(self, *args, **kw)
            model = self.model
            while hasattr(model, "runnable"):
                model = model.runnable
            made["model"] = model
            made["arena"] = Arena(torch, model, a.arena_gib * GIB, d / "arena.lock")

        cls.load_model = load_model

    for name in RUNNERS:
        try:
            wrap(importlib.import_module(name).GPUModelRunner)
        except (ImportError, AttributeError):
            pass
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=a.model, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=a.util,
        max_model_len=a.max_model_len, max_num_seqs=max(a.concurrency, 4), enable_prefix_caching=False,
        skip_tokenizer_init=True, seed=a.seed))
    arena, model = made["arena"], made["model"]
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    info = {"gpu": g, "uuid": uuid, "bytes": arena.buf.numel(), "weights_bytes": arena.weights_bytes,
            "weights_blocks": arena.weights_blocks, "outside_bytes": arena.outside_bytes,
            "params": len(arena.entries), "model": a.model, "staged": arena.staged, "pack_s": arena.pack_s,
            "entries": [[f"{owners[0][0]}.{owners[0][2]}", off, n] for off, n, _, owners in arena.entries],
            **export(arena.buf.data_ptr())}
    write_json(d / f"engine.{g}.json", info)
    rng = random.Random(a.seed)
    prompts = [[rng.randrange(1000, 120000) for _ in range(32 + 16 * k)] for k in range(4)]
    plain = probe(engine, "plain", prompts, a.probe_tokens)
    largest = max(-(-n // BLOCK) * BLOCK for _, n, _, _ in arena.entries)
    scratch = torch.empty(max(GIB, largest), dtype=torch.uint8, device="cuda:0")
    plain_hash = info["plain_sha256"] = arena.hashes(scratch)
    status("ready", **info)
    load = Load(engine, a.concurrency, a.prompt_tokens, a.max_tokens, a.step_hz, a.seed + 1 + g, align=a.align_steps)
    log = (a.out / f"engine.{g}.steps.jsonl").open("w")
    steps, ctl, seq, started, baseline = [], {}, -1, time.time(), False
    paused, hooks, checks = False, None, None
    ctl_path = d / "ctl.json"
    while True:
        try:
            got = json.loads(ctl_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            got = None
        if got is not None and got.get("seq", -1) != seq:
            ctl, seq = got, got["seq"]
        if ctl.get("stop"):
            break
        if ctl.get("pause") and not paused:
            load.abort()
            torch.cuda.synchronize()
            paused = True
            status("paused")
        if paused and ctl.get("key") and hooks is None:
            dec = Decoder(a.kv)
            dec.set_key(ctl["key"])
            first = ctl["bases"][uuid]
            decoded = arena.hashes(scratch, (first, dec))
            hooks = Hooks(torch, arena, scratch, dec, first, model)
            hooks.install()
            again = probe(engine, "decoded", prompts, a.probe_tokens)
            checks = {"arena_decodes": decoded == plain_hash, "probe_tokens_equal": again == plain,
                      "first_block": first, "hooked_modules": len(hooks.modules), "decode_calls": hooks.calls,
                      "probe": {"plain": plain, "decoded": again}}
            status("decoding", **checks)
        if paused and not ctl.get("pause") and hooks is not None:
            paused = False
        if paused:
            time.sleep(0.05)
            continue
        wait = load.wait()
        if wait > 0:
            time.sleep(min(wait, 0.05))
            continue
        try:
            s = load.step()
        except Exception as e:  # a dropping session serves zeroed chunks' decodes; record, don't stop
            s = {"t": time.time(), "ms": 0.0, "tokens": 0, "running": len(load.live), "error": repr(e)[:200]}
            load.abort()
        s["phase"] = ctl.get("phase") or ("plain-baseline" if not baseline else "plain")
        if not baseline and time.time() - started >= a.plain_s:
            baseline = True
            status("baseline", **info)
        steps.append(s)
        log.write(json.dumps(s) + "\n")
        if len(steps) % 64 == 0:
            log.flush()
    log.close()
    out = {"schema": "verity/remote-session/engine/v0", **info, "checks": checks, "args": vars(a) | {"dir": str(d),
           "out": str(a.out)}, "decode_calls": hooks.calls if hooks else 0, "phases": summary(steps),
           "errors": sum(1 for s in steps if "error" in s)}
    write_json(a.out / f"engine.{g}.json", out)
    status("stopped", phases=out["phases"])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
