"""vLLM serving under the remote bandwidth audit: an in-process vLLM engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
with `verity_vllm.protocols.remote_bandwidth.binder.Binder` on it registers its weights and KV pool in a `Store`, and
`verity_vllm.protocols.remote_bandwidth.remote.StoreBackend` proves that store to a verifier while the engine serves a trace of requests:
before each pass the engine runs its next step (its allocs, appends and frees, one memory-map/v1 line), and the pass
reads every live block at it (`prover.py`'s serve modes: ``serve``, or ``servedrop:K``, which withholds the first K
blocks appended).

    python serve.py --connect HOST:PORT --out OUT [--plan serve]    against verifier.py on a machine of its own
                                                                     (`run.sh` ROLE=prover SERVE=tiny)
    python serve.py --local --out OUT [--plan servedrop:1]           the verifier in this process, its clock the
                                                                     prover's passes, a tick each (`PassClock`)

The engine serves `TINY`, the vLLM tests' tiny Llama with 64-wide heads (so a KV page's plane is a 4 KiB block, the
memory map's), under random weights (``load_format="dummy"``), eager and without prefix caching (the binder refuses a
prefix hit), one session a process: a `Store` serves one session.  The trace: a request every `EVERY` steps, and one
whenever none is in flight, so every step schedules; their prompt and output lengths cycle through `SHAPES`, and a
prompt longer than a step's `BUDGET` tokens is prefilled over several steps, so pages fill across steps, requests
finish and free theirs, and freed pages are reused.

OUT/serve.json holds the store (`Binder.stats`), the requests, the prover's session result and, with --local, the
verifier's (its record and ledgers, the memory line among them: on the pass clock a window's span is ticks, which
the rebuild charge reads as nanoseconds, so it credits nothing there and only the held and read bytes mean anything).
It needs `verity_vllm.protocols.remote_bandwidth.remote.StoreBackend` and vLLM on a GPU (`tests/test_vllm_serve.py`).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "integrations" / "vllm")]

import wire  # noqa: E402

TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "num_hidden_layers": 2, "vocab_size": 256,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 256, "hidden_act": "silu",
        "torch_dtype": "bfloat16"}
PAGES, SLOTS, BUDGET, MAX_LEN = 24, 16, 64, 128
"""The KV pool's pages, a page's slots (the binder tells a cache's dimensions apart by size, so PAGES is neither SLOTS
nor TINY's KV heads), the tokens a step schedules at most, and a request's at most."""
EVERY = 4
SHAPES = ((40, 9), (17, 3), (100, 5), (16, 12), (50, 4), (23, 7))
"""(prompt tokens, output tokens) of the trace's requests, in turn."""


class PassClock:
    """Time is the prover's passes: one tick as each ends.  A wait returns the inbox's next message, or None once the
    clock reaches its end; once the verifier has said the passes are done, time jumps to the wait's end."""

    def __init__(self) -> None:
        self.t, self.done, self.inbox = 0, False, None

    def now(self) -> int:
        return self.t

    def tick(self) -> None:
        if self.inbox is None:
            self.t += 1
            return
        with self.inbox.mutex:
            self.t += 1
            self.inbox.not_empty.notify_all()

    def wait(self, inbox: queue.Queue, until_ns):
        self.inbox = inbox
        with inbox.not_empty:
            while True:
                if until_ns is not None and self.done:
                    self.t = max(self.t, until_ns)
                if until_ns is not None and self.t >= until_ns:
                    return None
                if inbox._qsize():
                    break
                inbox.not_empty.wait()
            return inbox._get()

    def idle(self) -> None:
        with self.inbox.mutex:
            self.done = True
            self.inbox.not_empty.notify_all()

    def busy(self) -> None:
        self.done = False


class Trace:
    """The engine's steps as the session runs them: step s adds the requests due at s, then runs `LLMEngine.step`."""

    def __init__(self, engine) -> None:
        from vllm import SamplingParams
        self.engine, self.params = engine, SamplingParams
        self.requests: list[dict] = []
        self.steps = 0

    def add(self, s: int) -> None:
        n = len(self.requests)
        prompt, out = SHAPES[n % len(SHAPES)]
        ids = [(7 * n + 3 * k + 1) % TINY["vocab_size"] for k in range(prompt)]
        params = self.params(max_tokens=out, min_tokens=out, ignore_eos=True, temperature=0.0, detokenize=False)
        self.engine.add_request(f"r{n}", {"prompt_token_ids": ids}, params)
        self.requests.append({"id": f"r{n}", "arrived": s, "prompt": prompt, "output": out, "finished": None})

    def __call__(self, s: int) -> None:
        if s % EVERY == 0 or not self.engine.has_unfinished_requests():
            self.add(s)
        for o in self.engine.step():
            if o.finished:
                self.requests[int(o.request_id[1:])]["finished"] = s
        self.steps = s + 1


def engine(work: Path):
    """The in-process engine over `TINY`, its config in ``work``; greedy, so without FlashInfer's sampler, whose first
    call builds it with ninja."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    model = work / "tiny"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps(TINY))
    args = EngineArgs(model=str(model), load_format="dummy", skip_tokenizer_init=True, enforce_eager=True,
                      enable_prefix_caching=False, block_size=SLOTS, num_gpu_blocks_override=PAGES,
                      max_model_len=MAX_LEN, max_num_seqs=4, max_num_batched_tokens=BUDGET, dtype="bfloat16", seed=0,
                      gpu_memory_utilization=0.1)
    return LLMEngine.from_engine_args(args)


def backend_class(remote):
    class Backend(remote.StoreBackend):
        """The store's backend, its passes ticking ``clock`` if it has one; `layout` is the hello's public layout,
        each registered region's kind, tier and blocks."""

        def __init__(self, *args, clock: PassClock | None = None, **kw) -> None:
            super().__init__(*args, **kw)
            self.clock = clock

        def layout(self) -> list[dict]:
            return [{"kind": r.kind, "tier": r.tier, "first": r.first, "blocks": r.blocks}
                    for r in self.store.layout().regions]

        def run_pass(self, seed: bytes, plan=None):
            out = super().run_pass(seed, plan)
            if self.clock is not None:
                self.clock.tick()
            return out

    return Backend


def verify_locally(lsock, clock: PassClock, a: argparse.Namespace, out: dict) -> None:
    import verifier

    sock, _ = lsock.accept()
    lsock.close()
    v = None
    try:
        v = verifier.Verifier(wire.Conn(sock, clock=clock.now), source=os.urandom(32), windows=a.windows,
                              window_ns=a.window_ns, samples=a.samples, delta=2.0 ** -40, jitter_ns=10 ** 9,
                              clock=clock, log=lambda m: None, lifetime=a.lifetime)
        v.register()
        out["verifier"] = v.session(0, v.allowance(v.ping(4)), None)
    except BaseException as e:  # noqa: BLE001
        out["error"] = e
        raise
    finally:
        if v is not None:
            v.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--connect", metavar="HOST:PORT", help="the verifier's address")
    where.add_argument("--local", action="store_true", help="run the verifier in this process")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", default="serve", help="serve, or servedrop:K")
    ap.add_argument("--codec", default="p512/k2", help="the registration's codec (`wire.codec_named`)")
    ap.add_argument("--node", default="verity-vllm/serve-tiny", help="the node's name in the hello")
    ap.add_argument("--windows", type=int, default=3, help="with --local")
    ap.add_argument("--window-ns", type=int, default=6, help="with --local: passes a window, a tick each")
    ap.add_argument("--samples", type=int, default=4096, help="with --local")
    ap.add_argument("--lifetime", type=int, default=1, help="with --local: L, windows")
    a = ap.parse_args()
    if "," in a.plan:
        ap.error("one session a process: a Store serves one session")
    a.out.mkdir(parents=True, exist_ok=True)

    from verity_vllm.protocols.remote_bandwidth import remote
    from verity_vllm.protocols.remote_bandwidth.binder import Binder
    from verity_vllm.protocols.remote_bandwidth.registry import Store

    store = Store()
    binder = Binder(store, passes=False).install()
    with tempfile.TemporaryDirectory() as work:
        trace = Trace(engine(Path(work)))
        clock = PassClock() if a.local else None
        backend = backend_class(remote)(store, trace, clock=clock)
        result, th = {}, None
        if a.local:
            lsock = wire.listen("127.0.0.1", 0)
            address = f"127.0.0.1:{lsock.getsockname()[1]}"
            th = threading.Thread(target=verify_locally, args=(lsock, clock, a, result), daemon=True)
            th.start()
        else:
            address = a.connect
        (prover_result,) = remote.prove(backend, address, a.plan, node=a.node.encode(),
                                        codec=wire.codec_named(a.codec), log=lambda m: print(m, flush=True))
        if th is not None:
            th.join()
            if "error" in result:
                raise result["error"]
    binder.uninstall()
    out = {"schema": "verity/remote-session/serve/v0", "plan": a.plan, "model": TINY, "pages": PAGES,
           "slots": SLOTS, "budget": BUDGET, "store": binder.stats(), "layout": backend.layout(),
           "trace": {"steps": trace.steps, "requests": trace.requests}, "prover": prover_result,
           **({"verifier": result["verifier"]} if a.local else {})}
    (a.out / "serve.json").write_text(json.dumps(out, indent=1, default=str))
    v = out.get("verifier", {})
    mem = v.get("ledger", {}).get("memory", {})
    print(f"serve: {a.plan}: prover accepted={prover_result.get('accepted')} verifier accepted={v.get('accepted')} "
          f"failure={v.get('failure')} steps={trace.steps} requests={len(trace.requests)} "
          f"blocks={out['store']['blocks']} memory held={mem.get('windows', [{}])[0].get('held')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
