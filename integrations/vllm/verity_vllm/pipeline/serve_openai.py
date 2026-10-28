"""`verity-vllm serve-openai`: an OpenAI-compatible HTTP server whose engine is the in-process committing vLLM engine.

    verity-vllm serve-openai --model REPO --revision SHA --out DIR [--port 18082] [--precision verity_int8|int8_exact|pin]
        [--protocol vllm|plain|ids [--id-payload 503]] [--no-commit] [--window-steps 64]
        [--force-include-usage | --no-force-include-usage] [--retain-exclude hidden_m1]
        [--required-manifest ROW/manifest.json --hidden-so SO --norm-tap-so SO | --scheme vllm-v1 --of-record RUN] [--seal]
        [--no-seal-off-loop] [--seal-hold-gb 64] [--seal-warm-gb 0] [--window-gb 32] [--disk-floor-gb 300] [--copy-records-to $OUT]
        [--admit arrival|release-order [--admit-hold-ms 5000]] [--switch-interval-ms 0.1]
        [--protocols pouw:eps1-sm120-h1fx1 [--protocols-config JSON] [--challenger-gate CMD --challenger-peer NAME]]
        [--remote-verifier HOST:PORT [--remote-kv-lib SO [--remote-session-lib SO]] [--remote-filler-gib 0]]

It serves what the network certifier sends (`benchmarks/network_traces/certifier_serve.py`): ``/v1/models``,
``/tokenize`` and ``/v1/completions``, the prompt as token ids or text, streamed as vLLM's own server streams it (one
``data: {...}`` event per engine output, the usage chunk, ``data: [DONE]``), or as one JSON response. vLLM's OpenAI server
is the reference: under ``--protocol vllm`` (the default) a request is parsed by its ``CompletionRequest`` (defaults,
validation, ``to_sampling_params``) and every chunk is its ``CompletionStreamResponse``, serialized the way its
``completion_stream_generator`` does; ``max_tokens`` is bounded by its ``get_max_tokens`` rule. `serve_protocol.PlainProtocol`
(``--protocol plain``) is the same framing without vLLM, for the CPU tests; `serve_protocol.framing_agrees` holds the two to
each other and the server records the result at start. ``--protocol ids`` (`serve_protocol.IdsProtocol`) answers a network
certifier's id request (exactly ``active.id_request`` in its ``--id-payload`` + 9-byte cells, its release headers aside) in the
scan's id form: no HTTP head, one ``io_tie.id_event`` of ``--id-payload`` bytes a token, no usage chunk, then ``data: [DONE]``. Every
other request (the network certifier's shim and probes, the load) is answered as ``--protocol vllm`` answers it.

Sampling parameters are values the request brings: the server applies vLLM's neutral
defaults, never the model's ``generation_config``, and records each request's ingress (the sampling fields it sent, the
engine's resulting parameters, the seed and its rule). A request without a ``seed`` is sampled under the public default,
`engine.served_seed.DEFAULT_SEED`. A ``top_p`` below 1 is refused (`TOP_P_REFUSAL`): the top-p sampler's split count is not
pinned yet, so its draw is not a function of the committed values.

The default precision is ``verity_int8`` (int8 linears, vLLM's attention) with the committer's GPU tree: the one setting
whose 16-token probe (0.81 s at 4 in flight) fits the certifier's plan (about 2.6 s); ``int8_exact`` takes 12.9 s even
without the Commit. The ``verity_int8`` quantization method is registered under every precision, so a pin whose checkpoint
names it loads under ``--precision pin`` too.

The engine is `verity_vllm.LLM` (in-process engine core, Model Runner V2), stepped by one thread, the process's main
thread (`EngineLoop`): each HTTP request is admitted with ``add_request`` under the engine id ``cmpl-<hex>-0`` (vLLM's
``f"{request_id}-{i}"``), and vLLM's scheduler batches whatever is admitted; each engine output goes back to its
request's connection as it comes. The HTTP side is an asyncio server on its own thread, one request per connection.
Requests reach the engine in the order the frontend parsed them (``--admit arrival``, the default), or in the network
certifier's release order (``--admit release-order``: `serve_admit.ReleaseOrder`, which says what the certifier sends), where each step's
rows also keep the scheduler's order, so a step's new rows are in admission order (`serve_admit.keep_scheduler_rows`;
``serve.json``'s ``step_rows``). Under ``--required-manifest`` the engine schedules as the bench row declares (`serve_row`).

With the Commit on (the default), the server's committer (`serve_seal.committer`) commits every engine step exactly as it does
a declared workload, values retained on host, and a `commit.served_record.SamplerInputsSource` binds each sampled row's seed and position. A
Commit window (`verity_vllm.pipeline.serve_commit.CommitWindows`) opens at the first engine step after the engine was idle
and closes when the engine is idle again (once it holds ``--idle-close-steps``), or after ``--window-steps`` steps (it
reopens before the next step): each window is one Commit, with its own run root. At each close the server reads the
window back from the committed bytes into the served record (`commit.served_record.Record`, COMMIT-ARB's): each request's
committed input ids (the step-input port's ``step.input/ids``, `served_record.IDS_NAMES`), sampled ids and
seeds, by position. Once a request has finished, `Record.check` holds them to what was served: its prompt ids and
streamed ids, the seed it carried, and where its finish says the stream ends. ``--prefix-caching``: vLLM's prefix cache
(off otherwise); a hit's check names the requests that committed its prefix (`RequestRecord.prefix_cover`).

Under ``--out`` the server writes ``serve.json`` (the engine, the protocol in use, the address), ``windows.jsonl`` (one
line a Commit: its run root, step roots, the committer's errors and identity check, the requests it holds, and the public
record of the step rows it ran, committed at its close under its run root for the compile slice, `serve_commit.commit_served`,
whose rows and salts go to ``served_rows/w<k>.json``), ``requests.jsonl`` (one line a request: the HTTP id, the engine id, the
body's ``user``, its release and its place among the engine's adds, the runner's ids, the prompt ids, the streamed ids, its
ingress, its finish, its committed rows and seeds, and its check), ``request_users.jsonl`` (`serve_commit.REQUEST_USERS`) and
`serve_commit.advice_trace`'s file. The Commit is rows-sha512 (``--scheme``'s default: hm96-sha512 rows in frame-v3-sha512 trees,
64-byte roots), as the bench row's declared Commit commits, with its committer parts built from the row's ``--required-manifest``
and taps (the acquisition plan under ``acquisition/``). ``--scheme vllm-v1`` (SHA-256, 32-byte roots) only serves again the
recorded vllm-v1 run ``--of-record`` names (`serve_seal.scheme_refusal`). ``--seal`` seals each closed
window as a deferred Commit's row dir (`serve_seal`), each seal a line of ``seals.jsonl``. ``--retain-exclude hidden_m1``
commits the attention interiors without retaining them, and each seal's manifest names them (`serve_seal.RetentionOptions`). Each
window's finalize, its readback and records, then its seal run on a thread of their own (`serve_seal.Sealer`), in window
order, while the engine serves the next window: the engine loop's part of a close only detaches the window from the
committer (`NativeHostCommitter.detach`), and it waits only while the windows awaiting their finalize and seal hold
``--seal-hold-gb`` of committed values, in the committer's pinned pool, which
``--seal-warm-gb`` grows before serving rather than on the engine loop. A window closes once its steps have committed
``--window-gb``, which bounds each bundle on disk; under ``--seal`` the server refuses to start unless ``--out``'s filesystem
has ``--disk-floor-gb`` free beyond one such window, and once a window's seal would take it under that floor, the window is
not sealed and the server stops: it refuses new requests, finishes those in flight, and exits. ``--copy-records-to`` (the
run's ``$OUT``) gets a copy of the records (`serve_commit.MIRRORED`) after each close and each seal, and at the stop before
the seals still pending. SIGTERM or SIGINT stops it: the open window is closed and its requests checked, the pending seals
finish, and ``summary.json`` counts the requests and checks, the seals, and the admissions' order (`ReleaseOrder.summary`);
it is written before the pending seals too, and its ``final`` is true only once every window's records are written.

``--protocols pouw:eps1-sm120-h1fx1`` serves PoUW H1 on this server's own Commit, one H1 window a Commit window
(`serve_h1`), each window's H1 outcome its ``windows.jsonl`` line's ``pouw_h1``. With ``--challenger-gate`` and
``--challenger-peer`` each window's coins come from a session of its own on the auditor host's challenger, reserved
through the challenger's gate (`serve_h1.GateSessions`); without them they are in-process, and no window credits anything.
``--remote-verifier HOST:PORT`` proves the engine's store (its weights, and ``--remote-filler-gib``) to the remote
bandwidth audit's verifier in lockstep with the server's engine steps from its first, a pass a step (`serve_remote`): its
record is ``remote_session.json``, and a rejected session fails the exit.

The engine runs eager, so each step makes hundreds of torch calls, each of which releases the GIL and takes it back. While
the seal's thread runs Python, each take waits out the interpreter's switch interval, so the server serves with
``--switch-interval-ms`` (0.1 ms; Python's is 5 ms). ``pauses.jsonl``
(`serve_seal.Watchdog`) records each stall of the process of 0.2 s or more.
"""
from __future__ import annotations

import asyncio
import collections
import json
import queue
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError, group, option
from verity_vllm.engine.served_seed import DEFAULT_SEED
from verity_vllm.pipeline import serve_h1 as SH, serve_remote as SR, serve_row as RW, serve_seal as SS
from verity_vllm.pipeline.serve_admit import ADMIT, Admission, Delta, ReleaseOrder, engine_id_of, keep_scheduler_rows, new_http_id
from verity_vllm.pipeline.serve_commit import (SERVED_CLASS, CommitWindows, advice_trace, disk_record, ingress_of, param_of,
                                               records_mirror, served_statics, user_of)
from verity_vllm.pipeline.serve_protocol import (ID_PAYLOAD, PROTOCOLS, SSE_DONE, Completion, PlainProtocol, RequestError,
                                                 VllmProtocol, error_body, framing_agrees, http_chunk, sse, usage_of)

PRECISIONS = ("verity_int8", "int8_exact", "pin")
#: precision -> (the int8 parts it registers, the int8 scope its layers must then run; None: not checked). Every precision
#: registers the linears, which is the ``verity_int8`` quantization method a pin's checkpoint may name.
INT8_PARTS = {"int8_exact": (("linears", "attention"), "all"), "verity_int8": (("linears",), "linears"), "pin": (("linears",), None)}
TOP_P_REFUSAL = ("serve-openai refuses top_p={top_p} < 1: vLLM's top-p sampler runs with a split count the engine does not pin, "
                 "so a top-p draw is not a function of the committed values. Send top_p 1 (top_k and temperature are "
                 "served); the refusal lifts once the split count is pinned.")


@dataclass(frozen=True, kw_only=True)
class Options:
    out: str = option("--out", env="RESEARCH_RUN_DIR", default=".", help="the records' directory (default $RESEARCH_RUN_DIR, else .)")
    host: str = option("--host", default="127.0.0.1")
    port: int = option("--port", type=int, default=18082)
    model: str = option("--model", help="the pinned model (the int8 pin: manifests/checkpoints-int8.json)")
    revision: str = option("--revision", help="its pinned revision")
    served_model_name: str | None = option("--served-model-name", default=None, help="the name /v1/models lists (default --model)")
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None, help="the checkpoint store (default $HF_HOME)")
    run_id: str = option("--run-id", env="RESEARCH_RUN_ID", default="local", help="the Commit's run id (default $RESEARCH_RUN_ID)")
    protocol: str = option("--protocol", choices=tuple(PROTOCOLS), default="vllm",
                           help="vllm: vLLM's own request and response classes; plain: the same framing without them; ids: a "
                                "network certifier's id requests streamed in the scan's id form, raw, and every other request as vllm "
                                "(serve_protocol.IdsProtocol)")
    id_payload: int = option("--id-payload", type=int, default=ID_PAYLOAD,
                             help="--protocol ids: each id event's bytes, the network certifier's cell payload B − 9 (default 503: the scan's "
                                  "B = 512)")
    precision: str = option("--precision", choices=PRECISIONS, default="verity_int8",
                            help="verity_int8 (default): int8 linears only, as vLLM's own server serves the int8 pin, the one that fits "
                                 "the certifier's plan; int8_exact: int8 linears and attention (quantization verity_int8, attention "
                                 "CUSTOM); pin: the pin's engine profile")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    max_model_len: int = option("--max-model-len", type=int, default=512)
    max_num_batched_tokens: int = option("--max-num-batched-tokens", type=int, default=2048, help="tokens a step schedules")
    prefix_caching: bool = option("--prefix-caching", action="boolean_optional", default=False, help="serve with vLLM's prefix cache")
    gpu_memory_utilization: float = option("--gpu-memory-utilization", type=float, default=0.55)
    seed: int = option("--seed", type=int, default=0)
    commit: bool = option("--commit", action="boolean_optional", default=True, help="commit every engine step (default on)")
    window_steps: int = option("--window-steps", type=int, default=64, help="engine steps a Commit window holds at most")
    idle_close_steps: int = option("--idle-close-steps", type=int, default=1,
                                   help="close a Commit window when the engine idles once it holds this many steps (default 1: at every idle)")
    window_requests: int = option("--window-requests", type=int, default=0,
                                  help="at most this many distinct requests in a Commit window, the class's slots (0: no bound): the "
                                       "engine holds at most this many unfinished requests, the rest wait in admission order, and a "
                                       "window closes before a step that would bring in one more (the running ones carry on)")
    chunk: int = option("--chunk", type=int, default=65536,
                        help="vllm-v1: the committer's leaf, in bytes of the packed step stream (as `commit`'s); a rows-sha512 row "
                             "is step_rows.ROW_BYTES")
    hash_threads: int = option("--hash-threads", type=int, default=2, help="vllm-v1: the committer's hashlib threads")
    gpu_tree: bool = option("--gpu-tree", action="boolean_optional", default=True,
                            help="the committer's GPU-resident chunk tree, no D2H and no hashlib (default on, and rows-sha512 needs it; "
                                 "its kernel is JIT-built with ninja, which must be on PATH)")
    cudart_copy: bool = option("--cudart-copy", action="store_true", default=False,
                               help="vllm-v1: the committer's snapshots by cudaMemcpyAsync")
    scheme: str = option("--scheme", choices=SS.SCHEMES, default=SS.ROWS,
                         help="the Commit's scheme: rows-sha512 (default: the declared Commit's committer, hm96-sha512 rows, 64-byte "
                              "roots; needs --required-manifest) or vllm-v1 (NativeHostCommitter, SHA-256, 32-byte roots), only with "
                              "--of-record")
    of_record: str | None = option("--of-record", default=None,
                                   help="--scheme vllm-v1 only: the recorded serve-openai run (its dir, with serve.json) this one serves "
                                        "again, which committed under vllm-v1 and served this --model and --revision")
    seal: bool = option("--seal", action="boolean_optional", default=False,
                        help="seal each closed window as a deferred Commit's row dir, OUT/windows/w<k>/ (serve_seal.seal_window)")
    seal_off_loop: bool = option("--seal-off-loop", action="boolean_optional", default=True,
                                 help="finalize and seal each window on a thread of its own while the engine serves the next one "
                                      "(serve_seal.Sealer; default on); --no-seal-off-loop does both on the engine loop, which "
                                      "stops serving for them")
    seal_hold_gb: float = option("--seal-hold-gb", type=float, default=64.0,
                                 help="the committed GB the windows awaiting their seal may hold (pinned host memory) before a "
                                      "close waits for room")
    seal_warm_gb: float = option("--seal-warm-gb", type=float, default=0.0,
                                 help="grow the committer's pinned pool by this many GB before serving (serve_seal.warm_pool), so "
                                      "the held windows and the open one find blocks ready (about --seal-hold-gb plus --window-gb); "
                                      "0: the pool grows on the engine loop as windows need it")
    window_gb: float = option("--window-gb", type=float, default=32.0,
                              help="close a Commit window once its steps have committed this many GB, which bounds its sealed "
                                   "bundle (0: no bound)")
    disk_floor_gb: float = option("--disk-floor-gb", type=float, default=300.0,
                                  help="--seal: refuse to start unless OUT's filesystem has this many GB free beyond one window "
                                       "(--window-gb), and stop cleanly (no new requests) rather than seal a window that would "
                                       "leave less (0: no floor)")
    copy_records_to: str | None = option("--copy-records-to", default=None,
                                         help="copy serve.json, requests.jsonl, request_users.jsonl, windows.jsonl and the step "
                                              "traces into this directory (the run's $OUT) after each window and at the stop")
    switch_interval_ms: float = option("--switch-interval-ms", type=float, default=0.1,
                                       help="the interpreter's switch interval while serving (Python's is 5 ms): how long the engine's "
                                            "thread may wait for the GIL after each torch call while the seal's thread runs Python "
                                            "(0: leave Python's)")
    required_manifest: str | None = option("--required-manifest", default=None,
                                           help="rows-sha512: the bench row's required manifest (its manifest.json), which the Commit of "
                                                "record's committer parts and the sealed binding map are built from")
    programs_root: str | None = option("--programs-root", default=None,
                                       help="the Build's row dir of the manifest's partition and attention record (default: the manifest's dir)")
    hidden_so: str | None = option("--hidden-so", default=None, help="rows-sha512: the attention hidden tap (as the Commit's --hidden-so)")
    hidden_ns: str | None = option("--hidden-ns", default=None, help="its op namespace (default: the .so's stem)")
    tap_cap_mb: int | None = option("--tap-cap-mb", type=int, default=None, help="the hidden tap's bounded device buffer, MiB (as the Commit's)")
    norm_tap_so: str | None = option("--norm-tap-so", default=None, help="rows-sha512: the norm-scale tap (as the Commit's --norm-tap-so)")
    served_rows_class: str | None = option("--served-rows-class", default=None,
                                           help="the compile slice's size class the served rows are committed under, as JSON "
                                                "{R, W, A, IW, Q, D, P, K} (serve_commit.SERVED_CLASS; default R --max-num-seqs, "
                                                "P --window-steps, IW 16, the rest 1: only R, P and IW shape the rows)")
    force_include_usage: bool = option("--force-include-usage", action="boolean_optional", default=True,
                                       help="vLLM's --enable-force-include-usage: the usage chunk on every stream, and usage on each "
                                            "chunk (default on)")
    admit: str = option("--admit", choices=ADMIT, default="arrival",
                        help="arrival (default): each request to the engine once parsed; release-order: in the network certifier's release "
                             "order (x-verity-release), each once every earlier-released one is in")
    admit_hold_ms: int = option("--admit-hold-ms", type=int, default=5000,
                                help="release-order: skip a missing earlier release once a later request has waited this long")
    protocols: str | None = option("--protocols", default=None,
                                   help=f"{SH.ENTRY}: PoUW H1 on this server's Commit, one H1 window a Commit window (serve_h1)")
    protocols_config: str | None = option("--protocols-config", default=None,
                                          help="--protocols' settings, {\"pouw\": {...}}: a JSON object, or a JSON file (serve_h1.config_of)")
    gate: SH.GateOptions = group(SH.GateOptions)
    remote: SR.RemoteOptions = group(SR.RemoteOptions)
    retention: SS.RetentionOptions = group(SS.RetentionOptions)


# ---------------------------------------------------------------------------------------------------------------- engine loop

class EngineLoop:
    """The one thread that steps the engine: admissions come through `submit`, outputs go back through each admission's
    ``deliver``. ``engine`` is vLLM's ``LLMEngine`` (``add_request``, ``abort_request``, ``step``,
    ``has_unfinished_requests``); ``commit`` a `CommitWindows` or None. With ``cap`` (``--window-requests``), at most ``cap``
    requests are in the engine at once: later ones wait in ``held``, in admission order, and go in as running ones finish."""

    _STOP = object()

    def __init__(self, engine: Any, commit: CommitWindows | None = None, cap: int = 0,
                 clock: Callable[[], float] = time.time) -> None:
        self.engine, self.commit, self.cap, self.clock = engine, commit, int(cap), clock
        self.q: queue.Queue = queue.Queue()
        self.live: dict[str, Admission] = {}
        self.held: collections.deque[Admission] = collections.deque()
        self.error: str | None = None
        self.draining: str | None = None
        self.steps = 0

    def submit(self, adm: Admission) -> None:
        adm.times.setdefault("submitted", self.clock())
        self.q.put(adm)

    def abort(self, engine_id: str) -> None:
        self.q.put(("abort", engine_id))

    def stop(self) -> None:
        self.q.put(self._STOP)

    def drain(self, why: str) -> None:
        """Stop once the requests in flight have finished; every request admitted after this is refused with ``why``."""
        self.q.put(("drain", why))

    def _take(self, item: Any) -> bool:
        """Apply one queued item; False for the stop."""
        if item is self._STOP:
            return False
        if isinstance(item, tuple) and item[0] == "drain":
            self.draining = self.draining or str(item[1])
            return True
        if isinstance(item, tuple):
            eid = item[1]
            if self.live.pop(eid, None) is not None:
                self.engine.abort_request([eid])
                if self.commit is not None:
                    self.commit.aborted(eid, "client")
                self._release()
            else:
                self.held = collections.deque(a for a in self.held if a.engine_id != eid)
            return True
        if self.draining is not None:
            item.deliver(Delta([], "", error=f"the endpoint is stopping: {self.draining}"))
            return True
        if self.cap and (self.held or len(self.live) >= self.cap):
            self.held.append(item)
        else:
            self._add(item)
        return True

    def _add(self, item: Admission) -> None:
        try:
            self.engine.add_request(request_id=item.engine_id, prompt={"prompt_token_ids": list(item.prompt_ids)}, params=item.params)
        except Exception as e:  # noqa: BLE001  (the engine's refusal of one request goes back to it)
            item.deliver(Delta([], "", error=f"{type(e).__name__}: {e}"))
            return
        item.times["admitted"] = self.clock()
        self.live[item.engine_id] = item
        if self.commit is not None:
            self.commit.admitted(item)

    def _release(self) -> None:
        """Held requests into the engine, in admission order, while it has room under ``cap``; those held before a drain
        are still served."""
        while self.held and len(self.live) < self.cap:
            self._add(self.held.popleft())

    def _deliver(self, out: Any) -> None:
        adm = self.live.get(out.request_id)
        if adm is None:
            return
        for c in out.outputs[:1]:
            d = Delta(list(c.token_ids), c.text or "", c.finish_reason, c.stop_reason,
                      list(out.prompt_token_ids) if out.prompt_token_ids is not None else None)
            if self.commit is not None:
                self.commit.streamed(adm.engine_id, d.token_ids, d.finish_reason, d.stop_reason)
            adm.deliver(d)
        if out.finished:
            self.live.pop(out.request_id, None)

    def run(self) -> None:
        running = True
        try:
            while running:
                idle = not self.engine.has_unfinished_requests()
                if idle and self.commit is not None:
                    self.commit.idle()
                if idle and self.draining is not None:
                    break
                item = self.q.get() if idle else self._poll()
                while item is not None:
                    if not self._take(item):
                        running = False
                        break
                    item = self._poll()
                if not running or not self.engine.has_unfinished_requests():
                    continue
                if self.commit is not None:
                    self.commit.before_step(self.live)
                for out in self.engine.step():
                    self._deliver(out)
                if self.commit is not None:
                    self.commit.stepped()
                self.steps += 1
                self._release()
        except BaseException as e:
            self.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            for adm in list(self.live.values()):
                adm.deliver(Delta([], "", error=self.error or "the server stopped"))
                if self.commit is not None:
                    self.commit.aborted(adm.engine_id, "server")
            self.live.clear()
            for adm in self.held:
                adm.deliver(Delta([], "", error=self.error or "the server stopped"))
            self.held.clear()
            if self.commit is not None and self.commit.is_open:
                self.commit.close()

    def _poll(self) -> Any:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------------------------------------------- HTTP

class Server:
    """The OpenAI-compatible endpoint: one request per connection (``connection: close``), the engine behind
    ``loop_submit`` (`EngineLoop.submit`), the protocol `PlainProtocol` or `VllmProtocol`."""

    def __init__(self, *, proto: PlainProtocol, served_name: str, max_model_len: int, tokenizer: Any, defaults: Mapping[str, Any],
                 submit: Callable[[Admission], None], abort: Callable[[str], None], model_root: str = "",
                 clock: Callable[[], float] = time.time, order: ReleaseOrder | None = None) -> None:
        self.proto, self.served_name, self.max_model_len = proto, served_name, int(max_model_len)
        self.tokenizer, self.defaults, self.order = tokenizer, dict(defaults), order
        self.submit, self.abort = (submit, abort) if order is None else (order.submit, order.abort)
        self.model_root, self.clock = model_root or served_name, clock
        self.served = 0
        self.streaming: set[asyncio.StreamWriter] = set()      # responses whose head is written

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            t_head = self.clock()
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
            body = await reader.readexactly(int(headers.get("content-length") or 0))
            await self.route(method, path.split("?", 1)[0], body, writer, headers, head + body,
                             {"head": t_head, "received": self.clock()})
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    @staticmethod
    def _head(status: int, ctype: str, length: int | None) -> bytes:
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 500: "Internal Server Error"}.get(status, "")
        size = b"content-length: %d\r\n" % length if length is not None else b"transfer-encoding: chunked\r\n"
        return (b"HTTP/1.1 %d %s\r\ncontent-type: %s\r\n" % (status, reason.encode(), ctype.encode())) + size + b"connection: close\r\n\r\n"

    async def _json(self, writer: asyncio.StreamWriter, status: int, payload: bytes) -> None:
        writer.write(self._head(status, "application/json", len(payload)) + payload)
        await writer.drain()

    async def route(self, method: str, path: str, body: bytes, writer: asyncio.StreamWriter,
                    headers: Mapping[str, str] | None = None, request: bytes = b"", times: Mapping[str, float] | None = None) -> None:
        release = None
        try:
            if path == "/health":
                return await self._json(writer, 200, b"")
            if path == "/v1/models" and method == "GET":
                return await self._json(writer, 200, json.dumps(self.models()).encode())
            if path in ("/v1/completions", "/tokenize") and method != "POST":
                raise RequestError(f"{path} takes POST", 405, "MethodNotAllowed")
            if path == "/tokenize":
                return await self._json(writer, 200, json.dumps(self.tokenize(self._body(body))).encode())
            if path == "/v1/completions":
                release = self._release(headers)
                return await self.completions(self._body(body), writer, release, self.proto.of(request), times)
            raise RequestError(f"no route {method} {path}", 404, "NotFoundError")
        except RequestError as e:
            await self._json(writer, e.status, error_body(str(e), e.status, e.kind, e.param))
        except (ConnectionError, asyncio.IncompleteReadError):
            raise
        except Exception as e:  # noqa: BLE001  (a fault of ours answers 500, before any byte of a stream is written)
            if writer not in self.streaming:
                await self._json(writer, 500, error_body(f"{type(e).__name__}: {e}", 500, "InternalServerError"))
        finally:
            self.streaming.discard(writer)
            if release is not None:
                self.order.passed(release)

    def _release(self, headers: Mapping[str, str]) -> tuple[str, int] | None:
        try:
            return None if self.order is None else self.order.key(headers or {})
        except ValueError as e:
            raise RequestError(str(e)) from e

    @staticmethod
    def _body(body: bytes) -> Any:
        try:
            return json.loads(body or b"{}")
        except ValueError as e:
            raise RequestError(f"the body is not JSON: {e}") from e

    def models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": self.served_name, "object": "model", "created": int(self.clock()), "owned_by": "vllm",
                                            "root": self.model_root, "parent": None, "max_model_len": self.max_model_len}]}

    def _check_model(self, model: str | None) -> None:
        if model is not None and model != self.served_name:
            raise RequestError(f"The model `{model}` does not exist.", 404, "NotFoundError", "model")

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=add_special_tokens)]

    def tokenize(self, b: Any) -> dict[str, Any]:
        if not isinstance(b, dict) or not isinstance(b.get("prompt"), str):
            raise RequestError("/tokenize takes {\"prompt\": string}", param="prompt")
        self._check_model(b.get("model"))
        ids = self.encode(b["prompt"], bool(b.get("add_special_tokens", True)))
        strs = self.tokenizer.convert_ids_to_tokens(ids) if b.get("return_token_strs") else None
        return {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids, "token_strs": strs}

    def admit(self, b: Any, deliver: Callable[[Delta], None], release: tuple[str, int] | None = None,
              proto: Any = None, times: Mapping[str, float] | None = None) -> tuple[Completion, Admission]:
        """Parse a completion request into the engine's admission (ids, sampling parameters), under ``proto`` (by default
        the server's)."""
        proto = proto or self.proto
        req = proto.parse(b)
        self._check_model(req.model)
        prompt_ids = self.encode(req.prompt, req.add_special_tokens) if isinstance(req.prompt, str) else [int(t) for t in req.prompt]
        try:
            max_tokens = proto.max_tokens(req, self.max_model_len, len(prompt_ids), self.defaults)
        except ValueError as e:
            raise RequestError(str(e)) from e
        try:
            params = proto.seeded(proto.sampling_params(req, max_tokens, self.defaults))
        except ValueError as e:                              # SamplingParams' own validation
            raise RequestError(str(e)) from e
        top_p = param_of(params, "top_p")
        if top_p is not None and float(top_p) < 1:
            raise RequestError(TOP_P_REFUSAL.format(top_p=top_p), param="top_p")
        http_id = new_http_id()
        return req, Admission(http_id, engine_id_of(http_id), prompt_ids, params, deliver, ingress_of(proto.brought(req), params),
                              user_of(req.raw), release, dict(times or {}))

    async def completions(self, b: Any, writer: asyncio.StreamWriter, release: tuple[str, int] | None = None,
                          proto: Any = None, times: Mapping[str, float] | None = None) -> None:
        proto = proto or self.proto
        loop = asyncio.get_running_loop()
        deltas: asyncio.Queue = asyncio.Queue()
        req, adm = self.admit(b, lambda d: loop.call_soon_threadsafe(deltas.put_nowait, d), release, proto, times)
        created, model = int(self.clock()), self.served_name
        self.submit(adm)
        self.served += 1
        n_prompt, n_out, finished = len(adm.prompt_ids), 0, False
        try:
            if not req.stream:
                text, ids, last = [], [], Delta([], "")
                while not last.finished:
                    last = await deltas.get()
                    text.append(last.text)
                    ids.extend(last.token_ids)
                finished = True
                if last.error:
                    raise RequestError(last.error, 500, "InternalServerError")
                payload = proto.response(request_id=adm.http_id, created=created, model=model, text="".join(text),
                                         finish_reason=last.finish_reason, stop_reason=last.stop_reason,
                                         prompt_token_ids=adm.prompt_ids if req.return_token_ids else None,
                                         token_ids=ids if req.return_token_ids else None, usage=usage_of(n_prompt, len(ids)))
                return await self._json(writer, 200, payload.encode())
            http = proto.http
            frame = http_chunk if http else bytes
            if http:
                writer.write(self._head(200, "text/event-stream; charset=utf-8", None))
            self.streaming.add(writer)
            first = True
            while not finished:
                d = await deltas.get()
                finished = d.finished
                if d.error:
                    writer.write(frame(sse(error_body(d.error, 500, "InternalServerError").decode())))
                    break
                if not d.text and not d.token_ids and not n_out:
                    continue                                     # vLLM: no empty chunk before the first token
                n_out += len(d.token_ids)
                usage = usage_of(n_prompt, n_out) if req.continuous_usage else None
                data = proto.events(request_id=adm.http_id, created=created, model=model, text=d.text, finish_reason=d.finish_reason,
                                    stop_reason=d.stop_reason, prompt_token_ids=adm.prompt_ids if (first and req.return_token_ids) else None,
                                    token_ids=d.token_ids if req.return_token_ids else None, usage=usage)
                first = False
                writer.write(frame(data))
                await writer.drain()
            if req.include_usage and n_out:
                writer.write(frame(sse(proto.usage_chunk(request_id=adm.http_id, created=created, model=model,
                                                         usage=usage_of(n_prompt, n_out)))))
            writer.write(frame(SSE_DONE) + (b"0\r\n\r\n" if http else b""))
            await writer.drain()
        finally:
            if not finished:
                self.abort(adm.engine_id)


class HttpThread(threading.Thread):
    """``server`` on ``host:port`` in this thread's own event loop: ``ready`` is set once it listens (``address``), and
    `shutdown` stops it from any thread."""

    def __init__(self, server: Server, host: str, port: int) -> None:
        super().__init__(name="http", daemon=True)
        self.server, self.host, self.port = server, host, port
        self.ready = threading.Event()
        self.address: tuple[str, int] | None = None
        self.error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._done: asyncio.Event | None = None

    def run(self) -> None:
        async def main() -> None:
            self._loop, self._done = asyncio.get_running_loop(), asyncio.Event()
            srv = await asyncio.start_server(self.server.handle, self.host, self.port, limit=1 << 22, backlog=256)
            self.address = tuple(srv.sockets[0].getsockname()[:2])
            self.ready.set()
            async with srv:
                await self._done.wait()

        try:
            asyncio.run(main())
        except BaseException as e:
            self.error = e
            self.ready.set()

    def shutdown(self) -> None:
        if self._loop is not None and self._done is not None:
            self._loop.call_soon_threadsafe(self._done.set)
        self.join()


# ---------------------------------------------------------------------------------------------------------------- main

def engine_options(opts: Options) -> dict[str, Any]:
    kw: dict[str, Any] = {"max_num_seqs": opts.max_num_seqs, "max_model_len": opts.max_model_len,
                          "max_num_batched_tokens": opts.max_num_batched_tokens, "gpu_memory_utilization": float(opts.gpu_memory_utilization),
                          "enforce_eager": True, "seed": opts.seed, **({"enable_prefix_caching": True} if opts.prefix_caching else {})}
    if opts.precision == "int8_exact":
        kw.update(quantization="verity_int8", attention_backend="CUSTOM")
    elif opts.precision == "verity_int8":
        kw.update(quantization="verity_int8")
    return kw


def register_int8(precision: str, register: Callable[[Sequence[str]], Any] | None = None) -> str | None:
    """Register ``precision``'s int8 parts (`engine.int8_ranks.register`, or ``register``) before the engine is built, the
    ``verity_int8`` quantization method under every precision; returns the int8 scope its layers must then run (None: not
    checked)."""
    parts, scope = INT8_PARTS[precision]
    if register is None:
        from verity_vllm.engine.int8_ranks import register
    register(parts)
    return scope


def served_class(opts: Options) -> dict[str, int]:
    """The served rows' size class: ``--served-rows-class`` over its defaults."""
    given = json.loads(opts.served_rows_class) if opts.served_rows_class else {}
    if not isinstance(given, dict) or set(given) - set(SERVED_CLASS):
        raise UsageError(f"--served-rows-class {opts.served_rows_class}: a JSON object of {', '.join(SERVED_CLASS)}")
    return {**dict.fromkeys(SERVED_CLASS, 1), "R": opts.max_num_seqs, "P": opts.window_steps, "IW": 16,
            **{k: int(v) for k, v in given.items()}}


def disk_floor_of(opts: Options, usage: Callable[[Any], Any] = shutil.disk_usage) -> SS.DiskFloor | None:
    """``--seal``'s `serve_seal.DiskFloor` over the filesystem ``--out`` is on (its nearest existing directory), None without
    a floor."""
    if not (opts.seal and opts.disk_floor_gb > 0):
        return None
    p = Path(opts.out).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return SS.DiskFloor(p, int(opts.disk_floor_gb * SS.GB), usage)


def preflight(opts: Options, which: Callable[[str], str | None] = shutil.which,
              usage: Callable[[Any], Any] = shutil.disk_usage) -> None:
    """What would fail after the model has loaded, refused before it."""
    if opts.window_steps < 1 or opts.admit_hold_ms < 1:
        raise UsageError("--window-steps and --admit-hold-ms are at least 1")
    if opts.window_requests < 0:
        raise UsageError("--window-requests is at least 0 (0: no bound)")
    if (c := served_class(opts))["R"] < opts.max_num_seqs:
        raise UsageError(f"--served-rows-class R {c['R']} is under --max-num-seqs {opts.max_num_seqs}: a full step's rows wouldn't fit")
    if min(opts.window_gb, opts.seal_hold_gb, opts.seal_warm_gb, opts.disk_floor_gb, opts.switch_interval_ms) < 0:
        raise UsageError("--window-gb, --seal-hold-gb, --seal-warm-gb, --disk-floor-gb and --switch-interval-ms are at least 0")
    if opts.seal and not opts.commit:
        raise UsageError("--seal seals each Commit window: it needs the Commit (drop --no-commit)")
    if (floor := disk_floor_of(opts, usage)) is not None and (why := floor.refusal(int(opts.window_gb * SS.GB))):
        raise UsageError(f"--seal: {why}")
    if why := SS.scheme_refusal(opts):
        raise UsageError(why)
    if why := SH.refusal(opts) or SR.refusal(opts):
        raise UsageError(why)
    if opts.required_manifest and not Path(opts.required_manifest).is_file():
        raise UsageError(f"--required-manifest {opts.required_manifest}: no such file")
    if opts.commit and opts.gpu_tree and which("ninja") is None:
        raise UsageError("--gpu-tree JIT-builds its kernel with ninja, which is not on PATH: put the environment's bin (where its "
                         f"ninja is) on PATH{', or pass --no-gpu-tree' if opts.scheme == SS.V1 else ''}")


def build(opts: Options, session: SR.ServeSession | None = None, row: Mapping[str, Any] | None = None) -> tuple[Any, Any]:
    """The engine (`verity_vllm.LLM`), scheduling as the bench ``row`` declares, and with the Commit on, its committer, sized
    from the engine's envelope, with a `served_record.SamplerInputsSource` attached; built under a remote ``session``."""
    from verity_vllm.engine import env as engine_env

    engine_env.apply_engine(opts.hf_home)
    if opts.commit:
        engine_env.apply_target({"VERITY_RETAIN": "host"})
    scope = register_int8(opts.precision)
    from verity_vllm.llm import LLM

    kw = engine_options(opts) | RW.engine_args(row)
    llm = RW.checked(row, LLM(opts.model, revision=opts.revision, **(session.engine_args(kw) if session is not None else kw)))
    if scope is not None:
        from verity_vllm.program.frontend.kernel_readback import int8_exact_of
        if (got := int8_exact_of(llm.model)) != scope:
            raise RuntimeError(f"a {opts.precision} engine whose layers run int8 scope {got!r}, not {scope!r}")
    if session is not None:
        session.built(llm)
    if not opts.commit:
        return llm, None
    return llm, SS.committer(opts, llm)


def main(opts: Options) -> int:
    preflight(opts)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    session = SR.session_of(opts, out)
    llm, com = build(opts, session, row := RW.of(opts))
    engine = llm.vllm.llm_engine
    proto = PROTOCOLS[opts.protocol](opts.force_include_usage, **({"payload": opts.id_payload} if opts.protocol == "ids" else {}))
    reference = getattr(proto, "base", proto)
    framing = framing_agrees(reference) if isinstance(reference, VllmProtocol) else None
    defaults: dict[str, Any] = {}                            # vLLM's neutral defaults: the request brings its sampling
    tok = llm.get_tokenizer()
    trace = advice_trace(out, llm)
    mirror = records_mirror(out, opts.copy_records_to)
    windows = sealer = warmed = arm = None
    if com is not None:
        from verity_vllm.engine.step_trace import scheduler_config

        arm = SH.arm_of(opts, llm, com, out)
        warmed = SS.warm_pool(com, opts.seal_warm_gb)
        sealer = SS.Sealer(int(opts.seal_hold_gb * SS.GB)) if opts.seal and opts.seal_off_loop else None
        windows = CommitWindows(com, out, run_id=opts.run_id, window_steps=opts.window_steps, idle_close_steps=opts.idle_close_steps,
                                eos_ids=[int(tok.eos_token_id)] if tok.eos_token_id is not None else [],
                                decode=lambda ids: tok.decode(list(ids), skip_special_tokens=True), marks=trace.window,
                                seal=SS.seal_of(opts) if opts.seal else None,
                                served=served_statics(scheduler_config(trace.sched), served_class(opts)), sealer=sealer,
                                window_bytes=int(opts.window_gb * SS.GB) or None, disk=disk_floor_of(opts), mirror=mirror,
                                request_cap=opts.window_requests, arm=arm)
    if session is not None:
        session.begin(windows)
    loop = EngineLoop(session.paced(engine) if session is not None else engine, windows, cap=opts.window_requests)
    if windows is not None:
        windows.on_floor = loop.drain
    served = opts.served_model_name or opts.model
    order = rows = None
    if opts.admit == "release-order":
        from verity_vllm.engine.handle import model_runner
        from verity_vllm.engine.step_trace import scheduler_of
        order = ReleaseOrder(loop.submit, loop.abort, hold_s=opts.admit_hold_ms / 1000)
        rows = keep_scheduler_rows(model_runner(llm.vllm), scheduler_of(engine))
    server = Server(proto=proto, served_name=served, max_model_len=opts.max_model_len, tokenizer=tok,
                    defaults=defaults, submit=loop.submit, abort=loop.abort, model_root=opts.model, order=order)
    http = HttpThread(server, opts.host, opts.port)
    http.start()
    http.ready.wait()
    if http.address is None:
        raise RuntimeError(f"the HTTP server did not bind {opts.host}:{opts.port}: {http.error!r}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        # the queue's lock is not reentrant: the stop is queued from a thread of its own, never from inside the handler
        signal.signal(sig, lambda *_: threading.Thread(target=loop.stop, daemon=True).start())
    bound = [http.address]
    if opts.switch_interval_ms:
        sys.setswitchinterval(opts.switch_interval_ms / 1000)
    watchdog = SS.Watchdog(out / "pauses.jsonl")
    watchdog.thread.start()
    record = {"address": list(bound[0]), "served_model_name": served, "model": opts.model, "revision": opts.revision,
              "precision": opts.precision, "protocol": proto.name, "id_payload": getattr(proto, "payload", None),
              "framing": framing, "defaults": dict(defaults),
              "generation_config_not_applied": engine.vllm_config.model_config.get_diff_sampling_param(),
              "default_seed": DEFAULT_SEED, "top_p": "refused below 1", "committer": getattr(com, "name", None),
              "admit": opts.admit, "admit_hold_ms": opts.admit_hold_ms if order is not None else None,
              "step_rows": "scheduler" if rows is not None else "runner",
              "commit": opts.commit, "scheme": getattr(com, "commit_scheme", SS.V1) if com is not None else None,
              "of_record": opts.of_record,
              "sealed": "windows/w<k>/" if windows is not None and windows.seal is not None else None,
              "served_rows_statics": windows.served if windows is not None else None,
              "seal_off_loop": sealer is not None, "seal_hold_gb": opts.seal_hold_gb if sealer is not None else None, "pool_warmed": warmed,
              "finalize_off_loop": sealer is not None and windows.finalize_on_loop is None,
              "finalize_on_loop": windows.finalize_on_loop if windows is not None else None,
              "window_gb": opts.window_gb or None, "disk_floor": disk_record(windows), "copy_records_to": opts.copy_records_to,
              "weights": getattr(com, "weights", None), "window_steps": opts.window_steps, "idle_close_steps": opts.idle_close_steps,
              "window_requests": opts.window_requests or None, "protocols": arm.describe() if arm is not None else None,
              "remote_session": session.describe() if session is not None else None, "row_scheduling": row,
              "committer_options": {k: getattr(opts, k) for k in ("chunk", "hash_threads", "gpu_tree", "cudart_copy", "required_manifest",
                                                                  "hidden_so", "hidden_ns", "tap_cap_mb", "norm_tap_so")},
              "required_manifest_path": getattr(com, "required_manifest_path", None), "retain_exclude": opts.retention.retain_exclude or None,
              "arena": getattr(com, "arena_sizing_record", None), "engine": llm.engine_record, "run_id": opts.run_id,
              "switch_interval_s": sys.getswitchinterval(), "watchdog": {"period_s": watchdog.period_s, "min_s": watchdog.min_s},
              "clocks": {"perf_counter": time.perf_counter(), "time": time.time()}, **SS.target_of(llm.model)}
    (out / "serve.json").write_text(json.dumps(record, indent=1, default=str))
    if mirror is not None:
        mirror()
    print(f"SERVE-OPENAI listening on {bound[0][0]}:{bound[0][1]} model={served} commit={opts.commit} protocol={proto.name}", flush=True)
    try:
        loop.run()
    finally:
        http.shutdown()
        if mirror is not None:
            mirror()                                     # the requests closed at the stop, before the seals still pending
        def stopped(final: bool) -> dict[str, Any]:
            got = {"final": final, "served": server.served, "engine_steps": loop.steps, "engine_error": loop.error, "stopped_by": loop.draining,
                   "commit": windows.summary() if windows is not None else None, "admit": order and order.summary(),
                   "sealer": sealer.summary() if sealer is not None else None, "pauses": watchdog.summary(),
                   "mirror_errors": mirror.errors[:10] if mirror else None,
                   "remote_session": session.end() if session is not None else None}
            (out / "summary.json").write_text(json.dumps(got, indent=1))
            if mirror is not None:
                mirror()
            return got

        if sealer is not None:                           # a cancel's SIGKILL can land before the pending seals end
            stopped(False)
            print(f"SERVE-OPENAI stopping: {sealer.summary()['held'] / SS.GB:.1f} GB of windows still to seal", flush=True)
            sealer.close()
        if arm is not None:                              # each H1 window's release (off the loop) ends before ``final``
            arm.drain()
        watchdog.stop()
        summary = stopped(True)
        print(f"SERVE-OPENAI stopped: {json.dumps(summary)}", flush=True)
        trace.uninstall()
        if rows is not None:
            rows.uninstall()
        if com is not None:
            com.close()
    passed = windows is None or windows.summary()["passed"] or not server.served
    return 0 if loop.error is None and passed and SR.accepted(session) else 1
