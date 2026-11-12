"""vLLM serving under the remote bandwidth audit: an in-process vLLM engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
with `verity_vllm.protocols.remote_bandwidth.binder.Binder` on it registers its weights and KV pool in a `NodeMemory`, and
`verity_vllm.protocols.remote_bandwidth.remote.StoreBackend` proves that store to a verifier while the engine serves a trace of requests:
before each pass the engine runs its next step (its allocs, appends and frees, one memory-map/v1 line), and the pass
reads every live block at it (`prover.py`'s serve modes: ``serve``, or ``servedrop:K``, which withholds the first K
blocks appended).

    python serve.py --connect HOST:PORT --out OUT [--plan serve]    against verifier.py on a machine of its own
                                                                     (`run.sh` ROLE=prover SERVE=tiny)
    python serve.py --local --out OUT [--plan servedrop:1]           the verifier in this process, its clock the
                                                                     prover's passes, a tick each (`PassClock`)

Either exits 0 only when the session was accepted, and `REJECTED` when it was not.

Two more forms need no engine and no GPU: what `run.sh` asks before it takes a lease, and after its prover ends.

    python serve.py --fit --pass-ms MS [--row-file ROW.json | --steps S] [--windows W] [--window-ms MS]
        the session's windows for a run of S engine steps (a row's `expected_engine_steps`; neither: a run with no
        end, this file's trace), printed as "WINDOWS WINDOW_MS": derived when --window-ms is not given, checked when
        it is (`fit`), and refused with the reason, exit `REFUSED`, where no session fits
    python serve.py --verdict --out OUT --prover-rc RC [--verifier-rc RC] [--record JSON] [--window-ms MS]
        one line saying how a served session ended, and the exit code that says it (`verdict`): from the prover's
        exit code, OUT/verifier.json and the verifier's exit code (with --verifier-rc: `run.sh` ran the verifier
        too) and the prover's record of the session (serve.json, or a row's commit/remote_session.json)

The engine serves `TINY`, the vLLM tests' tiny Llama with 64-wide heads (so a KV page's plane is a 4 KiB block, the
memory map's), under random weights (``load_format="dummy"``), eager and without prefix caching (the binder refuses a
prefix hit), one session a process: a `NodeMemory` serves one session.  The trace: a request every `EVERY` steps, and one
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
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "adapters" / "vllm")]

from verity.protocols.poub.verifier.session import wire  # noqa: E402

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

REFUSED, VERIFIER_RED, REJECTED = 3, 5, 6
"""`run.sh`'s exit codes that this file decides: a plan refused before anything ran, a prover that ended 0 with no
accepted session on record, and a session that was not accepted."""
WINDOWS, PASSES, MIN_PASSES = 3, 4, 2
"""What `fit` derives: the session's windows, the passes a window is sized for, and the fewest it is sized for: at
two, a pass half as fast as stated still lands one root in every window."""


class Refused(ValueError):
    """No session of these windows fits the run; the message is the reason `run.sh` prints."""


def session_steps(windows: int, window_ms: int, pass_ms: int) -> int:
    """The most engine steps a session of ``windows`` windows of ``window_ms`` takes when a pass, with the engine step
    before it, comes every ``pass_ms``: a window holds at most ⌊window_ms / pass_ms⌋ + 1 passes, and the run's first
    step is over before the first window opens."""
    return windows * (window_ms // pass_ms + 1) + 1


def fit(steps: int | None, pass_ms: int, windows: int | None = None, window_ms: int | None = None) -> tuple[int, int]:
    """(windows, window_ms) of a session served by a run of ``steps`` engine steps (None: a run with no end), a pass
    every ``pass_ms``.  A served session's windows are the verifier's wall clock and its passes the run's steps, so
    windows that outlast the run end the session rejected after the whole run, and windows shorter than a pass
    reject it at the first with no root.  Without ``window_ms`` it is derived: `PASSES` passes a window, or half what
    the run has room for where that is fewer, never under `MIN_PASSES`.  With it, the same sum checks it.  `Refused`
    where nothing fits."""
    w = WINDOWS if windows is None else windows
    if w < 1 or pass_ms < 1 or (steps is not None and steps < 1) or (window_ms is not None and window_ms < 1):
        raise Refused(f"windows, their milliseconds, a pass's and a run's steps are positive: WINDOWS={w} "
                      f"WINDOW_MS={window_ms} PASS_MS={pass_ms}, a run of {steps} steps")
    room = None if steps is None else (steps - 1) // w - 1
    if window_ms is None:
        if room is not None and room < MIN_PASSES:
            raise Refused(f"a run of {steps} engine steps holds no session of {w} windows of {MIN_PASSES} passes each "
                          f"({w * (MIN_PASSES + 1) + 1} steps): state fewer WINDOWS")
        return w, (PASSES if room is None else max(MIN_PASSES, min(PASSES, room // 2))) * pass_ms
    passes = window_ms // pass_ms
    if passes < 1:
        raise Refused(f"a window of {window_ms} ms holds no pass at {pass_ms} ms a pass (PASS_MS: an engine step and "
                      f"the pass after it), and the verifier rejects a window with no root: state WINDOW_MS of "
                      f"{MIN_PASSES * pass_ms} or more")
    if room is not None and passes > room:
        raise Refused(f"{w} windows of {window_ms} ms take up to {session_steps(w, window_ms, pass_ms)} engine steps at "
                      f"a pass every {pass_ms} ms (PASS_MS), and the run is {steps} steps: the session would end "
                      f"rejected after the whole run, \"the run ended after {steps} steps, before the session's last "
                      "pass\"; " + (f"WINDOW_MS under {(room + 1) * pass_ms} fits" if room >= 1 else
                                    "state fewer WINDOWS"))
    return w, window_ms


def row_steps(path: Path) -> int:
    """The engine steps of the row file ``path``'s run, its ``expected_engine_steps``; `Refused` for a file that
    states none, or whose target does not declare synchronous scheduling, which a Commit's session refuses once its
    engine is about to be built (`verity_vllm.protocols.remote_bandwidth.remote.CommitSession.target`)."""
    try:
        row = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"the row file {path} can't be read: {e}") from None
    steps = row.get("expected_engine_steps")
    if type(steps) is not int or steps < 1:
        raise Refused(f"the row file {path} states no expected_engine_steps, the run's length the session must fit")
    got = ((row.get("engine") or {}).get("target") or {}).get("async_scheduling")
    if got is not False:
        raise Refused(f"the row file {path}'s target declares async_scheduling {got!r}, and a Commit's session serves "
                      'an engine that schedules synchronously: a row whose target declares "async_scheduling": false')
    return steps


def pass_ms_seen(session: dict, window_ms: int) -> int | None:
    """The milliseconds a pass took in a verifier's ``session`` record, from its windows that ended with another
    after them (the last one seen may have been cut short); None when it saw none."""
    full = [w.get("passes") for w in (session.get("windows_seen") or [])[:-1]]
    full = [p for p in full if type(p) is int and p > 0]
    return round(window_ms * len(full) / sum(full)) if full else None


def verdict(prover_rc: int, verifier: dict | None, record: dict | None, window_ms: int | None = None,
            verifier_rc: int = 0) -> tuple[int, str]:
    """(`run.sh`'s exit code, the line that says why) of a served session: ``prover_rc`` the prover's exit code,
    ``verifier`` the verifier's record where `run.sh` ran it too (verifier.json; {} when it left none) and
    ``verifier_rc`` its exit code, ``record`` the prover's record of its session (``accepted``, and ``error`` or
    ``failure``), None when it left none.  0 only when every side that has a record accepted; `REJECTED` when either
    says not accepted (the verifier rejected the session, or it failed where the prover's record says), whatever the
    prover's code, so a session that was not accepted and a Commit that failed beside it end apart; else the
    prover's code when it is not 0; else `VERIFIER_RED`: a prover that ended 0 with no accepted session on record, or
    beside a verifier that ended red."""
    sessions = (verifier or {}).get("sessions") or []
    said = [s.get("accepted") for s in sessions]
    mine = None if record is None else record.get("accepted")
    if False in said or mine is False:
        s = next((s for s in sessions if s.get("accepted") is False), None)
        why = [] if s is None else ["the verifier rejected it" + (f", {json.dumps(s['failure'])}" if s.get("failure") else "")]
        if mine is False:
            said_why = [str(record[k]) for k in ("error", "failure") if record.get(k)]
            why.append("the prover's record: " + ("; ".join(said_why) if said_why else "not accepted"))
        seen = pass_ms_seen(s or (sessions[0] if sessions else {}), window_ms) if window_ms else None
        if seen:
            why.append(f"its passes came every {seen} ms: state PASS_MS={seen}")
        return REJECTED, "session not accepted: " + "; ".join(why)
    if prover_rc != 0:
        return prover_rc, (f"the prover ended {prover_rc} with "
                           + ("its session accepted" if mine is True else "no session on record"))
    if verifier is not None and not (said and all(a is True for a in said)):
        return VERIFIER_RED, ("the prover ended 0 and the verifier left no accepted session on record"
                              + (f": it refused the registration, {verifier['refused']}" if verifier.get("refused") else ""))
    if mine is not True:
        return VERIFIER_RED, "the prover ended 0 and left no record of an accepted session"
    if verifier_rc != 0:
        return VERIFIER_RED, f"the session is accepted on record and the verifier ended {verifier_rc}"
    return 0, "session accepted" + (f" ({len(said)} on the verifier's record)" if said else "")


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
    from verity.protocols.poub.verifier.session import verifier

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


def fit_main(a: argparse.Namespace) -> int:
    try:
        steps = row_steps(a.row_file) if a.row_file else a.steps
        windows, window_ms = fit(steps, a.pass_ms, a.windows, a.window_ms)
    except Refused as e:
        print(f"refused: {e}", flush=True)
        return REFUSED
    print(windows, window_ms, flush=True)
    return 0


def read_json(path: Path) -> dict | None:
    """The JSON object in ``path``; None where no file is, and, said on stdout, where what is there can't be read as
    one (a record cut short), which a verdict takes as no record."""
    if not Path(path).is_file():
        return None
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        print(f"serve: {path} holds no record ({type(e).__name__}: {e})", flush=True)
        return None
    return doc if isinstance(doc, dict) else None


def verdict_main(a: argparse.Namespace) -> int:
    record = read_json(a.record) if a.record else None
    if record is not None and isinstance(record.get("prover"), dict) and "accepted" not in record:
        record = record["prover"]
    both = a.verifier_rc is not None
    rc, line = verdict(a.prover_rc, (read_json(a.out / "verifier.json") or {}) if both else None, record,
                       a.window_ms, a.verifier_rc or 0)
    print(f"remote session verdict: {line} (exit {rc})", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--connect", metavar="HOST:PORT", help="the verifier's address")
    where.add_argument("--local", action="store_true", help="run the verifier in this process")
    where.add_argument("--fit", action="store_true", help="print the windows of a session that fits the run, or refuse")
    where.add_argument("--verdict", action="store_true", help="say how a served session ended, and exit with its code")
    ap.add_argument("--out", type=Path, help="the run's out/ (not with --fit)")
    ap.add_argument("--plan", default="serve", help="serve, or servedrop:K")
    ap.add_argument("--codec", default="p512/k2", help="the registration's codec (`wire.codec_named`)")
    ap.add_argument("--node", default="verity-vllm/serve-tiny", help="the node's name in the hello")
    ap.add_argument("--windows", type=int, help="with --local (default 3); with --fit: the session's windows")
    ap.add_argument("--window-ns", type=int, default=6, help="with --local: passes a window, a tick each")
    ap.add_argument("--samples", type=int, default=4096, help="with --local")
    ap.add_argument("--lifetime", type=int, default=1, help="with --local: L, windows")
    ap.add_argument("--pass-ms", type=int, help="with --fit: the milliseconds from one pass to the next")
    ap.add_argument("--row-file", type=Path, help="with --fit: the row whose run serves the session")
    ap.add_argument("--steps", type=int, help="with --fit: the run's engine steps, where no row file states them")
    ap.add_argument("--window-ms", type=int,
                    help="with --fit: the windows to check (default: derived); with --verdict: the windows the "
                         "verifier ran, to say what a pass took in a rejected session")
    ap.add_argument("--prover-rc", type=int, help="with --verdict: the prover's exit code")
    ap.add_argument("--verifier-rc", type=int,
                    help="with --verdict: the exit code of the verifier this run started, whose record is "
                         "OUT/verifier.json")
    ap.add_argument("--record", type=Path, help="with --verdict: the prover's record of its session")
    a = ap.parse_args(argv)
    if a.fit:
        if a.pass_ms is None or (a.row_file and a.steps is not None):
            ap.error("--fit takes --pass-ms, and --row-file or --steps, not both")
        return fit_main(a)
    if a.out is None:
        ap.error("--out is required")
    if a.verdict:
        if a.prover_rc is None:
            ap.error("--verdict takes --prover-rc")
        return verdict_main(a)
    if a.windows is None:
        a.windows = 3
    if "," in a.plan:
        ap.error("one session a process: a NodeMemory serves one session")
    a.out.mkdir(parents=True, exist_ok=True)

    from verity_vllm.protocols.remote_bandwidth import remote
    from verity_vllm.protocols.remote_bandwidth.binder import Binder
    from verity_vllm.protocols.remote_bandwidth.registry import NodeMemory

    store = NodeMemory()
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
    accepted = prover_result.get("accepted") is True and (not a.local or v.get("accepted") is True)
    return 0 if accepted else REJECTED


if __name__ == "__main__":
    sys.exit(main())
