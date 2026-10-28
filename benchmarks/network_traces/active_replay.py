"""The active certifier on recorded traces, in real time: `verity.core.protocols.certifier.active` serving a replay of honest traces'
configurations over localhost, then the audit of the records it published.

Each configuration is one run of three processes on the host's monotonic clock, from an origin they share:
- **the peer**, outside the boundary, sends each request of the trace as B-byte link frames (`active.request_frames`,
  the canonical framing: its length, then its bytes, on one header) in the middle of the ingress bucket its arrival
  falls in, so the ingress certifier enqueues it at ⌈t/τ_in⌉, `replay.py`'s
  model; it reads the wire and keeps one SHA-256 per window of it;
- **the proxy**, `active.Proxy` at the calibration's r, r_in, B and Σ_sync, with the prover's syncs given before the run;
  it publishes every record raw and committed (`commitment.commit`) from a worker thread, and the server gets each
  request with its place in the release order (`ReleaseProxy`, `release`), which no frame or record holds;
- **the server**, an OpenAI-compatible endpoint: a request reaching it in bucket R gets the trace's token emissions,
  frame k as one server-sent event written in the middle of bucket R + d_k − 1, where d_k is the frame's
  `ready − anchor` in the trace. An event's `id:` is its position in the canonical order (ready, anchor, session, k),
  and what follows it is `active.event` of the frame's token (`frame_data`: ` tk`, or the plan's `tokens`), so every
  egress frame is the canonical framing's template on the token's bytes.

**The honest prover** declares (`--syncs model`) `calibration.choose_syncs` over the run the model predicts: each request released by the
ingress certifier's rule, its frame k ready d_k buckets after its release. The replay keeps every bucket the certifier sees and
leaves half a bucket of margin around each write; a write later than that moves its bucket.

**The audit** reads each request's release from the raw ingress records (`active.Requests` on their frames in order:
the k-th request on the link is stream k), rebuilds the Program's frames (the server's events, in canonical order,
anchored at the release) and runs `Audit.check_link` and `check_ingress_link` on the raw
records and on the committed ones. It also checks that the proxy's records are `run_shaper`'s on the ready buckets the
proxy saw, that its hand times are the rule's, and that the peer's digest of each window of the wire is the records'.

**The salt** (`run_virtual(salt=True)`, and `mockdc.py run --salt`): the proxy's per-window salt is on, so each window's
ingress record holds the salt frame, which carries the placeholder and never the salt, and the server receives the salt
on a connection of its own, logging each window's SHA-512 of it (`fingerprint`), never the salt. Once the run's
records are published the proxy answers the verifier's "reveal" call with every window's salt (`salts.json`), each
revealed on a stand-in for the window's registration (`stand_in`: the replay registers nothing with a proof system),
and the audit's `salt` section reads the salt frames against them (`active.check_salts`, on the raw ingress records),
counts the salt frames in each ingress record and in the egress records, and checks that no egress frame holds a salt
and that the server received each salt.

**Maintenance** (`outage_plan`, and `mockdc.py`'s `maintenance` strategy): the block's certifier is down for one
window. The block is drained for it, so no request is enqueued in it, and the plan is refused if anything of the
windows before would still be in flight there. The proxy serves the windows before (`Proxy(windows=w)`) and closes the
link; nothing listens on the link for the outage window, so no record is published for it; then a proxy back from
maintenance (`Proxy(start=w + 1, position=p)`) serves the rest, and the peer connects again. The proxy log joins the
two. The certifier's audit has no verdict for the outage window, and its comparisons with `run_shaper` leave it out.

Latencies are in ms: shaping is hand minus ready, queueing release minus hand, first frame the first frame's release
minus its anchor (`replay.py`'s) and minus the trace's send time. `model` beside each is `replay.replay_configuration`
on the same configuration of the trace, lane F's replay model.

Each process runs with Python's cyclic garbage collector off: a full collection over a run's heap is a pause longer
than the slack, and the proxy would fail the window closed (`monitor-unavailable`).

**A live server** (`--server-url`, or `--vllm MODEL` to start one as `record.py` does): the server is a real vLLM
OpenAI endpoint behind a shim inside the boundary (`LiveServer`). The peer sends each request of the trace as
`record.py` sent it (its session names the workload's request: prompt token ids, `max_tokens`, `ignore_eos`), padded to
whole link frames. The shim forwards it to vLLM, cuts vLLM's stream at `\\n\\n` and writes back each completion
chunk's text as canonical events (`active.events`: one a token, or one per `TOKEN_BYTES` of a longer one), each with
`id:` its position in write order, and logs every event it wrote; it writes nothing from 1.5 buckets before the horizon on. The
audit's Program is that log. A live prover can't choose syncs on frames it hasn't seen, so by default it declares one
value for every window (`--syncs envelope`): the largest δ and the smallest ρ the calibration's honest prover declared,
met by every window any declared value meets. Under an online rule (`--syncs cover-K-M`, `online`) the proxy's own
prover declares each window's sync at its first tick: the rule on the frames it had read by `--prover-lag` buckets
before, chosen in a worker process off the proxy's tick, or the envelope if that choice isn't ready by then (the
proxy log's `prover.unready`). The audit checks the syncs the proxy logged, and `declared_are_planned` says whether
they are what the rule declares, with every choice ready, on the model's run. The sender's
write times then move the declarations, so an adversarial strategy's wire differs from honest's from the second window
on, within the log2 #Σ_sync bits per window the bound charges. Live, `--strategies` makes the shim the adversary: it
holds each event until its planned bucket, never before vLLM sent it, and `windows_whose_declared_differs` counts the
windows whose declared sync is not the honest run's; under `covert` the shim delays each stream's first event as
`covert.LiveSender` says: the covert sender on a live server, which can only delay what the model writes. The audit also
counts the windows the trace's model syncs and a hindsight prover (`choose_syncs` on the frames as read) would
have met. A frame read after its hand time under the declared sync, which an online rule's honest miss and a sender's
held frames both give, faults the window it is handed in `coverage-never-established`; `mismatches_from_late_frames`
counts the mismatched egress windows whose record is the shaper's rule on the frames as read: windows that closed
before such a frame was handed.

    python active_replay.py --calibration REPORT.json --out DIR [--windows 60] [--config-windows 4:10] TRACE.json ...
    python active_replay.py --vllm MODEL --calibration REPORT.json --out DIR --config NAME ... TRACE.json ...
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import gc
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import random
import re
import resource
import signal
import ssl
import struct
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import covert
import replay
from verity.core.protocols.certifier import online
from verity.core.protocols.certifier.active import (
    CANONICAL_STREAM,
    DATA,
    HEADER,
    ORDINAL,
    SALT,
    TOKEN_BYTES,
    Events,
    InlineExecutor,
    Proxy,
    Requests,
    VirtualClock,
    check_salts,
    COMPACT_RECORD,
    compact_cell,
    compact_record,
    compact_record_data,
    compact_unit,
    decode_frame,
    encode_frame,
    event,
    event_body,
    event_data,
    events,
    fresh_salt,
    id_event,
    id_event_data,
    id_request,
    request_frame_count,
    request_frames,
    wait_until,
    worker,
)
from verity.core.protocols.certifier.lean_audit import Audit
from verity.core.protocols.certifier.calibration import choose_syncs
from verity.core.protocols.certifier.release import release_head, release_of, with_release
from verity.core.protocols.certifier.commitment import CommittedRecord, commit
from verity.core.protocols.certifier.grid import (
    COMPLETE,
    GridParams,
    Reason,
    Record,
    Status,
    Certifier,
    row_count,
)
from verity.core.protocols.certifier.schedule import (
    ClockSync,
    LogicalFrame,
    SyncSet,
    handing_buckets,
    run_shaper,
)
from verity.core.protocols.certifier.served import Admission, ServedProfile, admits
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.service import Registration

MODEL = "llama-3.1-8b-instruct"
HTTP_HEAD = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\n\r\n"
STARTUP_S = 3.0


def frame_data(plan: dict, session: str, k: int) -> bytes:
    """The token bytes frame k of a session's response carries: the plan's (`tokens`, hex by session), else ` tk`."""
    tokens = plan.get("tokens", {}).get(session)
    return bytes.fromhex(tokens[k]) if tokens is not None else b" t%d" % k


def token_frames(token: bytes) -> list[str]:
    """A token's frames as a plan's `tokens` holds them: its bytes cut every `TOKEN_BYTES`, as `active.events` cuts."""
    return [token[i : i + TOKEN_BYTES].hex() for i in range(0, max(1, len(token)), TOKEN_BYTES)]


def event_bytes(data: bytes, position: int, first: bool) -> bytes:
    """A response frame as the server writes it: `id:` its position, then the canonical event of its token's bytes,
    the HTTP head before the first."""
    return (HTTP_HEAD if first else b"") + b"id: %d\n" % position + event(data)


def _fit(build, frames: int, frame_bytes: int) -> bytes:
    """`build(pad)` with the least pad that takes exactly `frames` link frames (`active.request_frames`)."""
    room, out, pad = frame_bytes - HEADER.size, build(0), 0
    target = (frames - 1) * room + 1 - 4
    while len(out) < target:
        pad += target - len(out)
        out = build(pad)
    if request_frame_count(len(out), frame_bytes) != frames:
        raise ValueError(f"a request of {frames} frames can't be built")
    return out


def request_bytes(session: str, frames: int, frame_bytes: int) -> bytes:
    """A streaming chat request naming its session (`user`), its prompt padded to take exactly `frames` link frames."""

    def build(pad: int) -> bytes:
        body = json.dumps({"model": MODEL, "stream": True, "user": session,
                           "messages": [{"role": "user", "content": "x" * pad}]}).encode()
        return (b"POST /v1/chat/completions HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\n"
                b"content-length: %d\r\n\r\n" % len(body)) + body

    return _fit(build, frames, frame_bytes)


LIVE_PATH = b"/v1/completions"
_WORKLOADS: dict[str, dict[int, dict]] = {}


def live_body(session: str | None, model: str) -> dict:
    """The request `record.py` sent for a session named `workload.instance.index`, its first prompt token drawn from the
    session's name; one token of a one-token prompt for a request no session names. Without `return_token_ids`, with
    which vLLM puts the prompt's token ids in the first event, past a frame."""
    import record

    if session is None:
        q = {"prompt": [1], "max_tokens": 1}
    else:
        name, _, index = session.split(".")
        if name not in _WORKLOADS:
            _WORKLOADS[name] = {r["index"]: r for r in record.load_workload(name)}
        r = _WORKLOADS[name][int(index)]
        q = {"prompt": [random.Random(session).randrange(1000, 100_000), *r["prompt"][1:]], "max_tokens": r["max_tokens"]}
    body = json.loads(record.request_body(model, q))
    del body["return_token_ids"]
    return body


def live_request_bytes(session: str, body: dict, frames: int | None, frame_bytes: int) -> bytes:
    """The live request: `body` with `user` the session, padded with spaces to take exactly `frames` link frames (the
    fewest it fits in, with None)."""
    raw = json.dumps(body | {"user": session}, separators=(",", ":")).encode()

    def build(pad: int) -> bytes:
        return (b"POST " + LIVE_PATH + b" HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\n"
                b"content-length: %d\r\n\r\n" % (len(raw) + pad)) + raw + b" " * pad

    return _fit(build, frames or request_frame_count(len(build(0)), frame_bytes), frame_bytes)


# Records on disk: the reference's `Record`, each slot that is a link frame stored as (kind, stream, payload) and any
# other slot as its bytes, then zlib. `unpack_record` gives back the same Record.

def pack_record(record: Record) -> bytes:
    allocation = len(record.grid[0]) if record.grid else 0
    head = json.dumps({"link": record.link, "window": record.window, "allocation": allocation,
                       "reason": record.status.reason.value if record.status.reason else None}).encode()
    parts = [struct.pack(">II", len(head), len(record.grid)), head]
    for row in record.grid:
        n = row_count(row)
        if any(slot is not None for slot in row[n:]):
            raise ValueError("a row that isn't left-packed")
        parts.append(struct.pack(">I", n))
        for slot in row[:n]:
            same = False
            if len(slot) >= HEADER.size:
                kind, stream, payload = decode_frame(slot)
                try:
                    same = encode_frame(kind, stream, payload, len(slot)) == slot
                except ValueError:
                    pass
            if same:
                parts += [struct.pack(">BBII", 0, kind, stream, len(payload)), payload]
            else:
                parts += [struct.pack(">BI", 1, len(slot)), slot]
    return zlib.compress(b"".join(parts), 1)


def unpack_record(blob: bytes, frame_bytes: int, intern: dict[bytes, bytes] | None = None) -> Record:
    data = zlib.decompress(blob)
    head_len, rows = struct.unpack_from(">II", data)
    i = 8 + head_len
    head = json.loads(data[8:i])
    grid = []
    for _ in range(rows):
        (n,), i = struct.unpack_from(">I", data, i), i + 4
        slots = []
        for _ in range(n):
            if data[i] == 0:
                _, kind, stream, length = struct.unpack_from(">BBII", data, i)
                i += 10
                slot = encode_frame(kind, stream, data[i : i + length], frame_bytes)
            else:
                (length,) = struct.unpack_from(">I", data, i + 1)
                i += 5
                slot = data[i : i + length]
            i += length
            slots.append(intern.get(slot, slot) if intern is not None else slot)
        grid.append(tuple(slots) + (None,) * (head["allocation"] - n))
    status = Status(Reason(head["reason"])) if head["reason"] else COMPLETE
    return Record(head["link"], head["window"], tuple(grid), status)


def write_blobs(path: Path, blob: bytes) -> None:
    with open(path, "ab") as fh:
        fh.write(struct.pack(">I", len(blob)) + blob)


def read_blobs(path: Path) -> list[bytes]:
    data, out, i = path.read_bytes() if path.exists() else b"", [], 0
    while i < len(data):
        (n,) = struct.unpack_from(">I", data, i)
        out.append(data[i + 4 : i + 4 + n])
        i += 4 + n
    return out


def committed_line(record: CommittedRecord) -> str:
    return json.dumps({"link": record.link, "window": record.window, "digest": str(record.digest),
                       "status": record.status.reason.value if record.status.reason else "complete"})


def read_committed(path: Path) -> list[CommittedRecord]:
    from verity.ml.commitments.identity import Digest

    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        d = json.loads(line)
        status = COMPLETE if d["status"] == "complete" else Status(Reason(d["status"]))
        out.append(CommittedRecord(d["link"], d["window"], Digest(d["digest"]), status))
    return out


class Setup(replay.Setup):
    """The calibration report's parameters at its own B, the frame size the active certifier runs at; the egress grid
    at the report's `egress_frame_bytes` when it has one (compact records, `certifier_serve.varied_plan`)."""

    def __init__(self, report: dict) -> None:
        super().__init__(report)
        g = report["grid"]
        self.egress_link, self.ingress_link = report["egress"][0]["link"], report["ingress"][0]["link"]
        e, i = self.egress[self.egress_link], self.ingress[self.ingress_link]
        self.egress_params = GridParams(e.buckets, e.allocation, g.get("egress_frame_bytes", g["frame_bytes"]),
                                        e.bucket_ms, e.queue_capacity)
        self.ingress_params = GridParams(i.buckets, i.allocation, g["frame_bytes"], i.bucket_ms, i.queue_capacity)


SYNC_POLICIES = ("model", "envelope", "an online rule (last, cover-K-M[-fallback])")


def parse_sync_policy(name: str) -> str:
    """`model`, `envelope`, or an online rule (`online.rule`), which the proxy declares window by window."""
    if name not in ("model", "envelope"):
        online.rule(name)
    return name


def plan_configuration(cfg: dict, setup: Setup, live_model: str | None = None, syncs: str = "model",
                       lag: int = 1) -> dict:
    """The run of one configuration: its sessions (rank = list order, stream id, d_k per frame), its requests (stream,
    session, frames, the trace's send time and the peer's), the model's release of each request and the prover's
    syncs: `model`, chosen over the model's run; `envelope`, the calibration's envelope in every window; or an online
    rule (`sync_rule`), which the proxy declares at each window's first tick from the frames it read by `lag` buckets
    before (`prover_lag`, 1 … T − 1), and `syncs` holds what it declares on the model's run with every choice ready.
    With `live_model`, a request takes the frames of its live body when they are more than the trace's."""
    parse_sync_policy(syncs)
    e, i = setup.egress_params, setup.ingress_params
    T, windows = e.buckets, replay.config_windows(cfg, e.buckets)
    if syncs not in ("model", "envelope") and not 0 < lag < T:
        raise ValueError(f"the proxy's online prover chooses off its tick: a lag of 1 … {T - 1} buckets, not {lag}")
    if len(cfg["egress"]) != 1 or len(cfg["ingress"]) != 1:
        raise ValueError("a configuration with one egress and one ingress link")
    first, ds = {}, defaultdict(list)
    for pos, f in enumerate(cfg["egress"][0]["frames"]):
        first.setdefault(f["session"], (f["anchor"], pos))
        ds[f["session"]].append((f["index"], f["ready"] - f["anchor"]))
    names = sorted(first, key=first.__getitem__)
    sessions_by_bucket = defaultdict(list)
    for s in names:
        sessions_by_bucket[first[s][0]].append(s)
    groups_by_bucket = defaultdict(list)
    for t, n in sorted(Counter(cfg["ingress"][0]["arrival_ms"]).items()):
        groups_by_bucket[math.floor(t / e.bucket_ms)].append((t, n))
    send: dict[str | None, tuple[float, int]] = {}
    extra, unmatched = [], 0
    for b in sorted(set(sessions_by_bucket) | set(groups_by_bucket)):
        ss, gs = sessions_by_bucket.get(b, []), groups_by_bucket.get(b, [])
        for s, g in zip(ss, gs):
            send[s] = g
        extra += gs[len(ss):]
        for s in ss[len(gs):]:
            send[s] = (b * e.bucket_ms + e.bucket_ms / 2, 1)
            unmatched += 1
    sessions = []
    for rank, s in enumerate(names):
        frames = sorted(ds[s])
        if [k for k, _ in frames] != list(range(len(frames))):
            raise ValueError(f"{cfg.get('name')}: session {s}'s indices aren't 0 … n−1")
        if any(b < a for (_, a), (_, b) in itertools.pairwise(frames)):
            raise ValueError(f"{cfg.get('name')}: session {s}'s frames aren't ready in order")
        sessions.append([s, 0, [d for _, d in frames]])
    requests = [[t, n, s] for s, (t, n) in send.items()] + [[t, n, None] for t, n in extra]
    requests.sort(key=lambda q: (q[0], q[2] is None, q[2] or ""))
    stream_of = {}
    for sid, q in enumerate(requests, start=1):
        if q[2] is not None:
            stream_of[q[2]] = sid
    for row in sessions:
        row[1] = stream_of[row[0]]
    tau_in = i.bucket_ms
    peer = []
    for sid, (t, n, s) in enumerate(requests, start=1):
        enq = math.ceil(t / tau_in)
        if live_model is not None:
            fewest = live_request_bytes(s or f"none-{sid}", live_body(s, live_model), None, i.frame_bytes)
            n = max(n, request_frame_count(len(fewest), i.frame_bytes))
        peer.append({"stream": sid, "session": s, "frames": n, "trace_ms": t, "enqueue": enq,
                     "send_ms": (enq - 1) * tau_in + tau_in / 2})
    release = model_releases(peer, i, windows, e.bucket_ms)
    frames, ready = model_frames(sessions, release, windows * T)
    sync_set = setup.sync_set
    chosen, _, misses = choose_syncs([(frames, ready)], sync_set, T)
    eager = ClockSync(min(sync_set.deltas), max(sync_set.rhos))
    chosen = (list(chosen) + [eager] * windows)[:windows]
    if syncs == "model":
        declared = chosen
    elif syncs == "envelope":
        declared = [setup.envelope] * windows
    else:
        declared, _, _ = online.declare([(frames, ready)], sync_set, T, online.rule(syncs), windows, setup.envelope,
                                        lag=lag)
    out = {"name": cfg.get("name"), "windows": windows, "sessions": sessions, "requests": peer,
           "model_release": {str(k): v for k, v in release.items()}, "model_misses": misses,
           "unmatched_sessions": unmatched, "ingress_only_requests": len(extra),
           "syncs": [[s.delta, str(s.rho)] for s in declared], "sync_policy": syncs,
           "model_syncs": [[s.delta, str(s.rho)] for s in chosen]}
    if syncs not in ("model", "envelope"):
        out["sync_rule"], out["prover_lag"] = syncs, lag
    if live_model is not None:
        out["live_model"] = live_model
    return out


def model_releases(requests: Sequence[dict], params: GridParams, windows: int, egress_bucket_ms: int,
                   admission: Admission | None = None, salt: bool = False) -> dict[int, int]:
    """Each request's release as the ingress certifier's rule gives it: its frames enqueued at their bucket, in send
    order, and released r_in per bucket; the egress bucket of its last frame's release. With `salt`, each window's
    first bucket puts the certifier's salt frame at the head of the queue, ahead of its arrivals and of any queue
    carried in. With `admission`, a served link's admission rule (`active.Bucketer(served=…)`): a request is enqueued
    only while its window, with it, still ends idle (`served.admits`), behind the certifier's own frame where the
    profile counts one."""
    pending = sorted((q["enqueue"], q["send_ms"], q["stream"], q["frames"]) for q in requests)
    out, queue, k, counts, T = {}, [], 0, [], params.buckets
    for t in range(windows * T):
        if t % T == 0:
            counts = []
            if salt or (admission is not None and admission.profile.own):
                queue.insert(0, (None, False))
        while k < len(pending) and pending[k][0] <= t:
            _, _, s, n = pending[k]
            if admission is not None and not admits(counts, len(queue) + n, admission.sync(t // T), admission.profile,
                                                    admission.allocation, params.allocation, T):
                break
            queue += [(s, j == n - 1) for j in range(n)]
            k += 1
        counts.append(min(len(queue), params.allocation))
        for s, last in queue[: params.allocation]:
            if last:
                out[s] = t * params.bucket_ms // egress_bucket_ms
        del queue[: params.allocation]
    return out


def program_keys(sessions: Sequence[list], release: dict[int, int], horizon: int) -> list[tuple[int, int, int, int]]:
    """The Program's frames ready before the horizon, in canonical order: (ready, anchor, session rank, k)."""
    keys = []
    for rank, (_, stream, ds) in enumerate(sessions):
        r = release.get(stream)
        if r is not None:
            keys += [(r + d, r, rank, k) for k, d in enumerate(ds) if r + d < horizon]
    keys.sort()
    return keys


def model_frames(sessions, release, horizon):
    keys = program_keys(sessions, release, horizon)
    return ([LogicalFrame(p.to_bytes(8, "big"), a, k) for p, (_, a, _, k) in enumerate(keys)],
            [rd for rd, _, _, _ in keys])


def program(plan: dict, release: dict[int, int], params: GridParams) -> tuple[list[LogicalFrame], list[int]]:
    """The Program's frames at B bytes, the server's events in canonical order, and the ready bucket each is meant for."""
    sessions = plan["sessions"]
    keys = program_keys(sessions, release, plan["windows"] * params.buckets)
    frames = [LogicalFrame(encode_frame(DATA, CANONICAL_STREAM, event(frame_data(plan, sessions[rank][0], k)),
                                        params.frame_bytes), a, k) for _, a, rank, k in keys]
    return frames, [rd for rd, _, _, _ in keys]


def program_streams(plan: dict, release: dict[int, int], params: GridParams) -> list[int]:
    """The stream of each of `program`'s frames, in its order."""
    keys = program_keys(plan["sessions"], release, plan["windows"] * params.buckets)
    return [plan["sessions"][rank][1] for _, _, rank, _ in keys]


STRATEGIES = ("honest", "burst", "spread", "bits", "late", "covert")
LIVE_TAMPERS = ("extra", "drop", "token", "byte")


def other_token(data: bytes) -> bytes:
    """Bytes of a canonical event's length other than `data`, which isn't empty: its first byte's low bit flipped."""
    return bytes([data[0] ^ 1]) + data[1:]


def other_hex(data: bytes) -> bytes:
    """`data` with its first hex digit's low bit flipped (``0``-``1``, ``a``-``b``, ``e``-``f``), so a response whose text
    is the hex of its outputs carries one output byte changed and stays hex; `data` without a hex digit is refused."""
    for i, c in enumerate(data):
        if chr(c) in "0123456789abcdef":
            return data[:i] + b"%x" % (int(chr(c), 16) ^ 1) + data[i + 1:]
    raise ValueError("an event without a hex digit has no output byte to change")


def other_id(token_id: int, eos: frozenset) -> int:
    """A token id other than `token_id` and none of `eos`: the first of `token_id` XOR 1, 2, … that isn't one."""
    return next(t for k in itertools.count(1) if (t := token_id ^ k) not in eos)


def adversary_writes(plan: dict, sync_set: SyncSet, buckets: int, strategy: str, seed: int = 0) -> dict[str, list]:
    """A server inside the boundary that signals by its write times: each frame's write bucket and position, over the
    model's releases. `burst` writes a session's frames in the bucket its request reaches it, `spread` frame k in the
    bucket before its logical hand time, `bits` one or the other per session by a random bit, and `late` is honest
    but holds one session in window 0, from its first frame on, until the hand time of the most lenient sync. On a live
    plan that isn't request-major the canonical order is the order the shim writes, so a session written behind frames
    due later than it is ready is on time: `late` holds it on, to the first bucket by which every other frame written
    is due. A stream's writes keep their order, and nothing is written in the last bucket or after."""
    windows = plan["windows"]
    horizon = windows * buckets
    release = {int(k): v for k, v in plan["model_release"].items()}
    keys = program_keys(plan["sessions"], release, horizon)
    frames = [LogicalFrame(p.to_bytes(8, "big"), a, k) for p, (_, a, _, k) in enumerate(keys)]
    hands = handing_buckets(frames, [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]], buckets)
    names = [s[0] for s in plan["sessions"]]
    rng = random.Random(seed)
    bits = {name: rng.randrange(2) for name in names}
    held = None
    if strategy == "late":
        lenient = handing_buckets(frames, [ClockSync(max(sync_set.deltas), min(sync_set.rhos))] * windows, buckets)
        p = next(p for p, (_, a, _, k) in enumerate(keys) if k == 0 and a < buckets and lenient[p] < horizon - 2)
        b = lenient[p]
        if "live_model" in plan and not plan.get("request_major"):
            others = [(ready - 1, h) for (ready, _, rank, _), h in zip(keys, hands) if rank != keys[p][2]]
            while any(w <= b < h for w, h in others):
                b += 1
        held = (keys[p][2], b)
    out: dict[str, list] = defaultdict(list)
    for p, (ready, anchor, rank, _) in enumerate(keys):
        if strategy == "burst" or (strategy == "bits" and bits[names[rank]]):
            b = anchor
        elif strategy in ("spread", "bits"):
            b = hands[p] - 1
        elif strategy == "late":
            b = max(ready - 1, held[1]) if rank == held[0] else ready - 1
        else:
            raise ValueError(f"strategy is one of {STRATEGIES[1:]}, not {strategy!r}")
        out[names[rank]].append([b, p])
    for name, ws in out.items():
        last = -1
        for w in ws:
            w[0] = last = max(w[0], last)
        out[name] = [w for w in ws if w[0] < horizon - 1]
    return dict(out)


def adversary_plan(plan: dict, setup: Setup, strategy: str, seed: int = 0) -> dict:
    """The plan an adversarial server runs: `adversary_writes`' writes, or, for `covert`, `covert.encode_writes`' (a
    sender that steers the online prover's declared sync with a random message), with the symbols it sent kept for
    the audit's `covert` measurement. Under syncs fixed before the run (`envelope`, `model`) the covert sender steers
    nothing: the control. On a live server the plan carries the sender's parameters, and the shim's
    `covert.LiveSender` draws the symbols as it serves. On a live shim only, on time and logged: `extra` writes one
    event more at the end of the first session's response, after what the endpoint sent and the shim's padding
    (`pad_to`); `drop` writes that response's first event not at all, counted as written so the padding doesn't make
    it up; `byte` writes that response's first event that carries bytes with one hex digit changed (`other_hex`: one
    output byte, when the text is the outputs' hex), so only the output tie refuses it; and `token` writes every event
    that carries bytes, of every response to a request enqueued in window 0,
    with other bytes (`other_token`), so the frame counts are honest's and only a check of the frames' tokens against
    the committed sampled ids (the I/O tie) refuses it. The tie draws positions in each window, so a window whose every
    token is wrong is refused at any number of draws, where one wrong frame among N would be refused only at the share
    1/N a draw."""
    if strategy == "token":
        T = setup.egress_params.buckets
        window0 = [q["session"] for q in plan.get("requests", ()) if "enqueue" in q and q["enqueue"] // T == 0]
        return plan | {"strategy": strategy, "seed": seed, "token": window0 or [plan["sessions"][0][0]]}
    if strategy in LIVE_TAMPERS:
        return plan | {"strategy": strategy, "seed": seed, strategy: plan["sessions"][0][0]}
    if strategy != "covert":
        return plan | {"strategy": strategy, "seed": seed,
                       "writes": adversary_writes(plan, setup.sync_set, setup.egress_params.buckets, strategy, seed)}
    T = setup.egress_params.buckets
    horizon = plan["windows"] * T
    release = {int(k): v for k, v in plan["model_release"].items()}
    rule = online.rule(plan["sync_rule"]) if "sync_rule" in plan else None
    if "live_model" in plan:
        s, env = setup.sync_set, setup.envelope
        return plan | {"strategy": strategy, "seed": seed,
                       "covert": {"seed": covert.stream_seed(seed, plan["name"]), "deltas": list(s.deltas),
                                  "rhos": [str(r) for r in s.rhos], "envelope": [env.delta, str(env.rho)],
                                  "margin": covert.margin_of(plan["sync_rule"]) if rule else 0}}
    fixed = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    enc = covert.encode_writes(program_keys(plan["sessions"], release, horizon), plan["sessions"], setup.sync_set, T,
                               plan["windows"], rule, setup.envelope, covert.stream_seed(seed, plan["name"]), horizon,
                               fixed)
    return plan | {"strategy": strategy, "seed": seed, "writes": enc["writes"],
                   "covert": {"symbols": enc["symbols"], "caps": enc["caps"], "predicted": enc["declared"],
                              "margin": covert.margin_of(plan["sync_rule"]) if rule else 0}}


def covert_symbols(plan: dict, buckets: int) -> tuple[list[int], int]:
    """What a run's sender sent per window, for `covert.measure`, and the rule's δ margin: the covert sender's symbols;
    for `bits`, how many of the sessions released in each window burst; for any other run, a random stream
    independent of it, so its measured rate is the estimator's null (the honest control)."""
    margin = covert.margin_of(plan["sync_rule"]) if "sync_rule" in plan else 0
    if "covert" in plan:
        return plan["covert"]["symbols"], margin
    if plan.get("strategy") == "bits":
        rng = random.Random(plan["seed"])
        bits = {name: rng.randrange(2) for name, _, _ in plan["sessions"]}
        release = {int(k): v for k, v in plan["model_release"].items()}
        out = [0] * plan["windows"]  # how many of each window's sessions burst rather than spread
        for name, stream, _ in plan["sessions"]:
            if (r := release.get(stream)) is not None and r // buckets < plan["windows"]:
                out[r // buckets] += bits[name]
        return out, margin
    return covert.control_symbols(plan["windows"], covert.stream_seed(plan.get("seed", 0), plan["name"])), margin


def outage_plan(plan: dict, setup: Setup, window: int = 1) -> dict:
    """The plan of a block whose certifier is down for maintenance in window `window` and back after it: the block is
    drained for that window (no request is enqueued in it, so the server writes nothing in it, and the requests left
    are numbered 1, 2, ... again, in their order on the link), and the proxy back
    from maintenance (`Proxy(start=window + 1, position=...)`) takes the Program's frames from `outage["position"]` on
    and the peer's requests from ingress bucket `outage["bucket"]` on. It is refused if anything of the windows before
    would still be in flight in the outage window: a request not yet released, or a frame not yet ready, handed or
    released."""
    e, i = setup.egress_params, setup.ingress_params
    T, windows = e.buckets, plan["windows"]
    if "sync_rule" in plan:
        raise ValueError("an online prover declares from window 0 on: an outage plan takes syncs fixed before the run")
    if not 0 < window < windows - 1:
        raise ValueError(f"the outage is in one of windows 1 … {windows - 2}, after a window and before one")
    kept = [q for q in plan["requests"] if q["enqueue"] // i.buckets != window]
    new = {q["stream"]: k for k, q in enumerate(kept, start=1)}  # the k-th request on the link is stream k
    requests = [q | {"stream": new[q["stream"]]} for q in kept]
    sessions = [[name, new[s], ds] for name, s, ds in plan["sessions"] if s in new]
    release = model_releases(requests, i, windows, e.bucket_ms)
    lo, hi = window * T, (window + 1) * T
    before = {str(new[int(s)]): r for s, r in plan["model_release"].items() if int(s) in new}
    if {str(s): r for s, r in release.items()} != before:
        raise ValueError(f"draining window {window} moves another request's release: one was still queued in it")
    if any(release.get(q["stream"], windows * T) >= lo for q in requests if q["enqueue"] < window * i.buckets):
        raise ValueError(f"a request of the windows before {window} is released in it or later")
    frames, ready = model_frames(sessions, release, windows * T)
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    hands = handing_buckets(frames, syncs, T)
    if any(f.anchor < lo and max(r, h) >= lo for f, r, h in zip(frames, ready, hands)):
        raise ValueError(f"a frame of the windows before {window} is ready or handed in it or later")
    quiet = next(r for r in run_shaper(frames, ready, syncs, Certifier(setup.egress_link, e)) if r.window == window)
    if any(row_count(row) for row in quiet.grid):
        raise ValueError(f"a frame of the windows before {window} is released in it")
    return plan | {"requests": requests, "sessions": sessions,
                   "model_release": {str(k): v for k, v in release.items()},
                   "outage": {"window": window, "position": sum(f.anchor < lo for f in frames),
                              "bucket": (window + 1) * i.buckets}}


def served(plan: dict) -> list[tuple[int, int]]:
    """The windows the block's certifier serves, as runs [start, end): every window, or, for an outage plan, the
    windows before the outage and the windows after it."""
    if (o := plan.get("outage")) is None:
        return [(0, plan["windows"])]
    return [(0, o["window"]), (o["window"] + 1, plan["windows"])]


# The three processes.

def peak_rss_mb() -> float:
    """This process's peak resident set (`VmHWM`; `ru_maxrss` survives exec, so a spawned child reports its parent's)."""
    status = Path("/proc/self/status").read_text()
    return int(re.search(r"VmHWM:\s+(\d+) kB", status).group(1)) / 1024


def _quiet_runtime() -> None:
    gc.disable()
    sys.setswitchinterval(0.001)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # a live run holds two sockets per stream in flight


def _usage() -> dict:
    return {"max_rss_mb": peak_rss_mb(), "cpu_s": sum(os.times()[:2]), "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "cpus": sorted(os.sched_getaffinity(0))}


class Server:
    """The OpenAI-compatible endpoint inside the boundary. `handle` serves one stream's connection; `serve` writes, in
    the middle of each bucket b, the events due in it: the honest server's frame k of a request it received in bucket R
    is due in R + d_k − 1, with the next position in canonical order; an adversary's (`plan["writes"]`) at its planned
    bucket and position. `notify` is called on every step, for a driver on a virtual clock."""

    def __init__(self, plan: dict, params: dict, clock=time.monotonic, sleep=asyncio.sleep, notify=lambda: None):
        self.clock, self.sleep, self.notify = clock, sleep, notify
        self.tau, self.horizon = params["bucket_ms"] / 1000, plan["windows"] * params["buckets"]
        self.sessions = {name: (rank, ds) for rank, (name, _, ds) in enumerate(plan["sessions"])}
        self.plan = plan
        self.writes = plan.get("writes")
        self.batches: dict[int, list] = defaultdict(list)
        self.processed, self.position, self.origin = -1, 0, None
        self.writers: set = set()
        self.log = {"receipt_late_ms": [], "batch_late_ms": [], "late_registrations": 0, "requests": 0, "events": 0}

    def _write(self, item) -> None:
        _, k, scheduled, name, writer, position = item
        if position is None:
            position, self.position = self.position, self.position + 1
        writer.write(event_bytes(frame_data(self.plan, name, k), position, k == 0))
        self.log["events"] += 1
        if k == scheduled - 1:
            writer.close()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.add(writer)
        try:
            first = await reader.readexactly(1)
            if first[0] == SALT:
                await read_salts(first, reader, self.log, self.notify)
                return
            head = first + await reader.readuntil(b"\r\n\r\n")
            length = int(re.search(rb"content-length: (\d+)", head, re.IGNORECASE).group(1))
            body = json.loads(await reader.readexactly(length))
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return
        now = self.clock() - self.origin
        r = math.floor(now / self.tau)
        self.log["requests"] += 1
        self.log["receipt_late_ms"].append(round((now - r * self.tau) * 1000, 3))
        name = body.get("user")
        rank, ds = self.sessions.get(name, (None, []))
        if self.writes is not None:
            planned = self.writes.get(name, [])
            items = [(b, (p,), k, len(planned), name, writer, p) for k, (b, p) in enumerate(planned)]
        else:
            n = sum(1 for d in ds if r + d < self.horizon)
            items = [(r + d - 1, (r, rank, k), k, n, name, writer, None) for k, d in enumerate(ds[:n])]
        if not items:
            writer.close()
        for b, *item in items:
            if b <= self.processed:
                self.log["late_registrations"] += 1
                self._write(item)
            else:
                self.batches[b].append(item)
        self.notify()

    async def serve(self, origin: float) -> None:
        self.origin = origin
        for b in range(self.horizon - 1):
            deadline = origin + b * self.tau + self.tau / 2
            if deadline > self.clock():
                await wait_until(deadline, self.clock, self.sleep)
            self.log["batch_late_ms"].append(round((self.clock() - deadline) * 1000, 3))
            for item in sorted(self.batches.pop(b, []), key=lambda x: x[0]):
                self._write(item)
            self.processed = b
            self.notify()

    def close(self) -> None:
        for w in self.writers:
            w.close()


def fingerprint(salt: bytes) -> str:
    return hashlib.sha512(salt).hexdigest()


async def read_salts(first: bytes, reader: asyncio.StreamReader, log: dict, notify) -> None:
    """The salt stream's connection, from its first byte: each salt frame (its header, then the length the header
    gives) puts its window and the salt's `fingerprint` in `log["salts"]`, never the salt itself."""
    head = first + await reader.readexactly(HEADER.size - 1)
    while True:
        _, _, payload = decode_frame(head + await reader.readexactly(HEADER.unpack(head)[2]))
        log.setdefault("salts", []).append([int.from_bytes(payload[:8], "big"), fingerprint(payload[8:])])
        notify()
        try:
            head = await reader.readexactly(HEADER.size)
        except asyncio.IncompleteReadError as e:
            if e.partial:
                raise
            return


def canonical_events(sse: bytes) -> list[bytes]:
    """A vLLM server-sent event as canonical events (`active.events`): the text of its completion chunk's choices, one
    event a token, or one per `TOKEN_BYTES` of a longer one; none for `[DONE]` or a line that isn't a chunk."""
    for line in sse.split(b"\n"):
        if line.startswith(b"data:") and (data := line[5:].strip()) != b"[DONE]":
            try:
                chunk = json.loads(data)
            except ValueError:
                return []
            choices = chunk.get("choices") or []
            if not choices:
                return []
            return events("".join(c.get("text") or c.get("delta", {}).get("content") or "" for c in choices).encode())
    return []


def chunk_tokens(sse: bytes) -> list[int] | None:
    """A vLLM server-sent event's sampled token ids: each choice's `token_ids` (vLLM's `return_token_ids`, or the
    committing server's), else its logprobs' `token_id:N` tokens (`return_tokens_as_token_ids`); None for `[DONE]` or
    a line that isn't a chunk."""
    for line in sse.split(b"\n"):
        if line.startswith(b"data:") and (data := line[5:].strip()) != b"[DONE]":
            try:
                chunk = json.loads(data)
            except ValueError:
                return None
            ids = []
            for c in chunk.get("choices") or []:
                got = c.get("token_ids")
                if got is None:
                    got = [int(t.removeprefix("token_id:")) for t in (c.get("logprobs") or {}).get("tokens") or ()]
                ids += got
            return ids
    return None


def id_events(sse: bytes, request: int) -> list[bytes]:
    """A vLLM server-sent event as id events (`active.id_event`) of request `request`: one a sampled token
    (`chunk_tokens`), and none for a chunk of none, so neither the endpoint's chunking nor its text reaches the wire."""
    return [id_event(request, t) for t in chunk_tokens(sse) or ()]


def compact_records(sse: bytes, request: int, eos: frozenset = frozenset(), start: int = 0,
                    max_tokens: int | None = None) -> list[bytes]:
    """A vLLM server-sent event as compact records (`active.compact_record`) of request `request`, one a sampled token
    as `id_events` has them, `start` the response's outputs before it: an EOS id (`eos`), or the last of a chunk whose
    `finish_reason` is "stop", is the stop record, which carries that id, and the response's `max_tokens`-th output,
    or the last of a chunk whose `finish_reason` is "length", carries `COMPACT_LAST`. vLLM v1 sends a merged chunk's
    finish in a chunk of no tokens after it, so the count marks a response that ends at `max_tokens`; one that ends
    short of it by length (at the model's context) is left unmarked, which the certified I/O refuses."""
    ids = chunk_tokens(sse) or []
    finish = None
    for line in sse.split(b"\n"):
        if line.startswith(b"data:") and (data := line[5:].strip()) != b"[DONE]":
            finish = next((c.get("finish_reason") for c in json.loads(data).get("choices") or ()
                           if c.get("finish_reason")), None)
    out = []
    for k, t in enumerate(ids):
        end = finish if k == len(ids) - 1 else None
        if t in eos or end == "stop":
            out.append(compact_record(request, t, "stop"))
        else:
            last = end == "length" or max_tokens is not None and start + k + 1 == max_tokens
            out.append(compact_record(request, t, "length" if last else None))
    return out


async def _http_body(reader: asyncio.StreamReader, head: bytes):
    """A response's body as it arrives: de-chunked under `transfer-encoding: chunked`, else read to its end."""
    if re.search(rb"transfer-encoding:\s*chunked", head, re.IGNORECASE):
        while size := int((await reader.readuntil(b"\r\n")).split(b";")[0], 16):
            yield await reader.readexactly(size)
            await reader.readexactly(2)
    else:
        while data := await reader.read(1 << 16):
            yield data


class LiveServer:
    """A live OpenAI endpoint inside the boundary, behind a shim: `handle` serves one stream's connection by sending its
    request (less its padding, and less `user`, which names the session, but on a varied plan, so that an endpoint
    that records it, as serve-openai's ``requests.jsonl`` does, names each request as the certified I/O does and
    `verity-vllm advice` takes it as the advice's request id; with the certifier's release headers, `release`, copied
    on and logged by session in `released`) to the endpoint at `upstream`, and writing back the
    text of each completion chunk of the response as canonical events (`canonical_events`), each with `id:` its
    position in write order, the HTTP head with the first. Every event
    written is logged as (position, session, k, the bucket it was written in, its bytes). From 1.5 buckets before the
    horizon on it writes nothing more, and closes the stream, so that every event logged is read before the last bucket.
    An adversary's shim (`plan["writes"]`) holds event k of a session until the middle of its planned bucket, never
    writing an event before the endpoint has sent it; an event past the plan waits for the last planned bucket. Under
    the `covert` strategy, `covert.LiveSender` holds each stream's first event until the bucket its symbol names.
    `open` counts the streams whose response hasn't ended and aren't held, and `held` the deadlines of the held ones,
    for a driver on a virtual clock. The endpoint is plain HTTP at `params["upstream"]`, or HTTPS with `tls`, under the
    path `prefix`, sent the bearer key in the environment variable `api_key_env` names, if any. With
    `plan["request_major"]` (a served plan's) the endpoint serves streams side by side but the shim writes them in the
    order it received their requests, each stream's first event only after the stream before it has ended, so the
    canonical order is request-major however the endpoint interleaves its responses. With `plan["pad_to"]` = F (a
    served plan's, F its body's tokens) a response the endpoint ends after fewer than F canonical events, having sent
    some tokens together in one chunk, gets empty canonical events after its last, up to F, written and logged as the
    endpoint's are, so every response is F events long; one of more than F isn't cut. With `plan["token_ids"]` (a
    varied plan's, `certifier_serve.varied_plan`, whose proxy has `ordinals`) the shim reads each request's ingress
    ordinal from the header that opens its connection, asks the endpoint for each chunk's token ids (the plan's
    `upstream_fields`, merged into every body it forwards) and writes one id event a sampled token (`id_events`), or,
    with `plan["compact"]`, one compact record (`compact_records`) as its unit (`active.compact_unit`, the position
    and the record, after the HTTP head with the first) for a proxy with `compact`."""

    def __init__(self, plan: dict, params: dict, clock=time.monotonic, sleep=asyncio.sleep, notify=lambda: None):
        self.clock, self.sleep, self.notify = clock, sleep, notify
        self.upstream = tuple(params["upstream"])
        self.tls = ssl.create_default_context() if params.get("tls") else None
        self.path = params.get("prefix", "").encode() + LIVE_PATH
        key = os.environ[params["api_key_env"]] if params.get("api_key_env") else None
        self.authorization = b"" if key is None else b"authorization: Bearer %s\r\n" % key.encode()
        self.tau, self.horizon = params["bucket_ms"] / 1000, plan["windows"] * params["buckets"]
        self.writes, self.extra, self.pad_to = plan.get("writes"), plan.get("extra"), plan.get("pad_to")
        self.token_ids, self.upstream_fields = bool(plan.get("token_ids")), plan.get("upstream_fields") or {}
        self.compact = bool(plan.get("compact"))
        if self.compact and not self.token_ids:
            raise ValueError("compact records are a varied plan's, of token ids")
        self.drop, self.token, self.byte = plan.get("drop"), frozenset(plan.get("token") or ()), plan.get("byte")
        self.eos = frozenset((plan.get("varied") or {}).get("eos_token_ids") or ())
        self.forward_user = bool(plan.get("varied"))
        self.held: Counter = Counter()
        self.windows = plan["windows"]
        self.position, self.origin, self.open = 0, None, 0
        self.serial, self.tickets, self.turn, self.ended = bool(plan.get("request_major")), 0, 0, set()
        self.turns: defaultdict = defaultdict(asyncio.Event)
        self.sender = None
        if (c := plan.get("covert")) is not None:
            rule = online.rule(plan["sync_rule"]) if "sync_rule" in plan else None
            self.sender = covert.LiveSender(SyncSet(tuple(c["deltas"]), tuple(Fraction(r) for r in c["rhos"])),
                                            params["buckets"], rule, ClockSync(c["envelope"][0], Fraction(c["envelope"][1])),
                                            c["seed"], [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]])
        self.writers: set = set()
        self.events: list[tuple[int, str, int, int, bytes]] = []
        self.log = {"receipt_late_ms": [], "batch_late_ms": [], "late_registrations": 0, "requests": 0, "events": 0,
                    "failed": [], "cut_at_horizon": 0, "first_event_ms": [], "server": "live", "padded": [],
                    "released": []}

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.add(writer)
        ordinal = None
        try:
            first = await reader.readexactly(1)
            if first[0] == SALT:
                await read_salts(first, reader, self.log, self.notify)
                return
            if first[0] == ORDINAL:
                ordinal = HEADER.unpack(first + await reader.readexactly(HEADER.size - 1))[1]
                first = await reader.readexactly(1)
            if self.token_ids and ordinal is None:
                raise ValueError("a request without its ingress ordinal, on a plan of id events")
            head = first + await reader.readuntil(b"\r\n\r\n")
            length = int(re.search(rb"content-length:\s*(\d+)", head, re.IGNORECASE).group(1))
            body = json.loads(await reader.readexactly(length))
        except (asyncio.IncompleteReadError, ConnectionError, ValueError) as e:
            self.log["failed"].append(repr(e)[:200])
            writer.close()
            return
        self.open += 1
        ticket, self.tickets = self.tickets, self.tickets + 1
        try:
            await self._serve(body, writer, ticket, ordinal, release_of(head))
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError) as e:
            self.log["failed"].append(repr(e)[:200])
        finally:
            self.ended.add(ticket)
            while self.turn in self.ended:
                self.turn += 1
            self.turns[self.turn].set()
            self.open -= 1
            writer.close()
            self.notify()

    async def _await_turn(self, ticket: int) -> None:
        if not self.serial or ticket <= self.turn:
            return
        self.open -= 1
        self.notify()
        try:
            await self.turns[ticket].wait()
        finally:
            self.open += 1

    async def _response_events(self, chunks, name: str, ordinal: int | None = None, max_tokens: int | None = None):
        """A response's canonical events, one list per chunk of its body as it arrives; under `pad_to`, empty events
        (`active.event(b"")`, what a chunk of no text gives) after its last, up to `pad_to`, counted in `padded`; for
        the `extra` session, one event more after those; for the `drop` session, its first left out; for a `token`
        session, each that carries bytes carrying `other_token` of them; for the `byte` session, its first that carries
        bytes carrying `other_hex` of them (one output byte of a hex response). `tampered` logs each event changed or left
        out, as (session, its index in the response). Under `token_ids`, id events of request `ordinal` (`id_events`);
        a `token` session's carry other ids (`other_id`), its EOS ids left as they are, so each response keeps its
        length and finish."""
        cutter, n, made, byte = Events(), 0, [0], name == self.byte

        def framed(e: bytes) -> list[bytes]:
            if self.compact:
                records = compact_records(e, ordinal, self.eos, made[0], max_tokens)
                made[0] += len(records)
                return records
            return id_events(e, ordinal) if self.token_ids else canonical_events(e)

        async for data in chunks:
            bodies = [b for e in cutter.feed(data) for b in framed(e)]
            if bodies and n == 0 and name == self.drop:
                self.log.setdefault("tampered", []).append([name, 0])
                n, bodies = len(bodies), bodies[1:]
                yield bodies
                continue
            if byte:
                for k, b in enumerate(bodies):
                    if text := event_data(b):
                        self.log.setdefault("tampered", []).append([name, n + k])
                        bodies[k], byte = event(other_hex(text)), False
                        break
            if name in self.token:
                for k, b in enumerate(bodies):
                    if self.compact and (d := compact_record_data(b)) is not None:
                        if d[2] != "stop":
                            self.log.setdefault("tampered", []).append([name, n + k])
                            bodies[k] = compact_record(d[0], other_id(d[1], self.eos), d[2])
                    elif self.token_ids and (d := id_event_data(b)) is not None:
                        if d[1] not in self.eos:
                            self.log.setdefault("tampered", []).append([name, n + k])
                            bodies[k] = id_event(d[0], other_id(d[1], self.eos))
                    elif text := event_data(b):
                        self.log.setdefault("tampered", []).append([name, n + k])
                        bodies[k] = event(other_token(text))
            n += len(bodies)
            yield bodies
        if self.pad_to is not None and n < self.pad_to:
            self.log["padded"].append([name, self.pad_to - n])
            yield [event(b"")] * (self.pad_to - n)
        if name == self.extra:
            yield ([compact_record(ordinal, 0)] if self.compact else [id_event(ordinal, 0)] if self.token_ids
                   else canonical_events(b'data: {"choices":[{"text":" extra"}]}\n\n'))

    async def _serve(self, body: dict, writer: asyncio.StreamWriter, ticket: int = 0,
                     ordinal: int | None = None, release: tuple[int, bytes | None] | None = None) -> None:
        received = self.clock() - self.origin
        self.log["requests"] += 1
        self.log["receipt_late_ms"].append(round((received % self.tau) * 1000, 3))
        name = body.pop("user")
        if release is not None:
            self.log["released"].append([name, release[0]])
        raw = json.dumps(body | ({"user": name} if self.forward_user else {}) | self.upstream_fields,
                         separators=(",", ":")).encode()
        up_reader, up_writer = await asyncio.open_connection(*self.upstream[:2], limit=1 << 22, ssl=self.tls)
        try:
            up_writer.write(b"POST " + self.path + b" HTTP/1.1\r\nhost: %s:%d\r\ncontent-type: application/json\r\n"
                            b"%s%scontent-length: %d\r\nconnection: close\r\n\r\n" % (
                                self.upstream[0].encode(), self.upstream[1], self.authorization,
                                b"" if release is None else release_head(*release), len(raw)) + raw)
            up_head = await up_reader.readuntil(b"\r\n\r\n")
            if not up_head.startswith((b"HTTP/1.1 200", b"HTTP/1.0 200")):
                self.log["failed"].append(up_head.split(b"\r\n")[0].decode(errors="replace"))
                return
            k, cutoff = 0, self.origin + (self.horizon - 1.5) * self.tau
            planned = None if self.writes is None else [b for b, _ in self.writes.get(name, [])]
            hold = -1
            anchor = math.floor(received / self.tau)
            async for bodies in self._response_events(_http_body(up_reader, up_head), name, ordinal,
                                                      body.get("max_tokens")):
                for body in bodies:
                    if k == 0:
                        await self._await_turn(ticket)
                    if planned is not None:
                        hold = max(hold, planned[k] if k < len(planned) else hold)
                        await self._hold(self.origin + hold * self.tau + self.tau / 2)
                    if k == 0 and self.sender is not None:
                        await self._hold(min(self.origin + self.sender.hold(anchor) * self.tau + self.tau / 2, cutoff))
                    now = self.clock()
                    if now >= cutoff:
                        self.log["cut_at_horizon"] += 1
                        return
                    p, self.position = self.position, self.position + 1
                    out = (HTTP_HEAD if k == 0 else b"") + (compact_unit(p, body) if self.compact
                                                            else b"id: %d\n" % p + body)
                    writer.write(out)
                    if k == 0:
                        self.log["first_event_ms"].append(round((now - self.origin - received) * 1000, 3))
                    self.events.append((p, name, k, math.floor((now - self.origin) / self.tau), out))
                    if self.sender is not None:
                        self.sender.wrote(p, anchor, k, self.events[-1][3])
                    self.log["events"] += 1
                    k += 1
                self.notify()
        finally:
            up_writer.close()

    async def _hold(self, deadline: float) -> None:
        if deadline <= self.clock():
            return
        self.held[deadline] += 1
        self.open -= 1
        self.notify()
        try:
            await wait_until(deadline, self.clock, self.sleep)
        finally:
            self.open += 1
            self.held[deadline] -= 1
            if not self.held[deadline]:
                del self.held[deadline]

    async def serve(self, origin: float) -> None:
        self.origin = origin

    def close(self) -> None:
        for w in self.writers:
            w.close()
        if self.sender is not None:
            self.log["covert"] = self.sender.log(self.windows)

    def dump_events(self, path: Path) -> None:
        with gzip.open(path, "wt", compresslevel=1) as fh:
            for p, name, k, bucket, event in self.events:
                fh.write(json.dumps([p, name, k, bucket, event.decode("latin-1")]) + "\n")


def read_events(path: Path) -> list[tuple[int, str, int, int, bytes]]:
    with gzip.open(path, "rt") as fh:
        return [(p, name, k, b, e.encode("latin-1")) for p, name, k, b, e in map(json.loads, fh)]


class Peer:
    """The client outside the boundary: `send` writes each request's link frames at its send time, `read` reads the wire
    to its end, keeping a SHA-256 per window and each bucket's arrival after its tick. For an outage plan it connects
    once for each run of windows the certifier serves (`phases`). A live plan's request is its own `body` (a varied
    plan's, `certifier_serve.varied_plan`), the plan's `body` for every session (a served plan's,
    `certifier_serve.served_plan`), or `record.py`'s for its session (`live_body`)."""

    def __init__(self, plan: dict, params: dict, clock=time.monotonic, sleep=asyncio.sleep, notify=lambda: None):
        self.clock, self.sleep, self.notify = clock, sleep, notify
        self.B, self.tau = params["frame_bytes"], params["bucket_ms"] / 1000
        self.B_in = params.get("ingress_frame_bytes", self.B)
        self.bucket_bytes = params["allocation"] * self.B
        self.window_bytes = params["buckets"] * self.bucket_bytes
        self.window_s = params["buckets"] * self.tau
        self.by_time = defaultdict(list)
        for q in plan["requests"]:
            self.by_time[q["send_ms"]].append(q)
        self.live_model, self.body = plan.get("live_model"), plan.get("body")
        self.outage = plan.get("outage")
        self.instants_sent = self.frames_sent = self.bytes = 0
        self.log = {"send_late_ms": [], "window_sha256": [], "bucket_lag_ms": []}

    def phases(self) -> list[tuple[int, range | None]]:
        """The peer's connections: the window each one's wire starts at, and the ingress buckets of the requests it
        sends (all, with None)."""
        if self.outage is None:
            return [(0, None)]
        b = self.outage["bucket"]
        return [(0, range(b)), (self.outage["window"] + 1, range(b, sys.maxsize))]

    def origin_of(self, origin: float, start: int) -> float:
        """The origin a connection's wire counts its buckets from, for one whose wire starts at window `start` after
        the windows already read: the run's, shifted by the windows the link was dead for."""
        return origin + (start - self.bytes // self.window_bytes) * self.window_s

    async def send(self, writer: asyncio.StreamWriter, origin: float, enqueued: range | None = None) -> None:
        for t in sorted(self.by_time):
            qs = [q for q in self.by_time[t] if enqueued is None or q["enqueue"] in enqueued]
            if not qs:
                continue
            deadline = origin + t / 1000
            if deadline > self.clock():
                await wait_until(deadline, self.clock, self.sleep)
            self.log["send_late_ms"].append(round((self.clock() - deadline) * 1000, 3))
            data = b"".join(request_frames(self.request(q), self.B_in) for q in qs)
            writer.write(data)
            self.instants_sent += 1
            self.frames_sent += len(data) // self.B_in
            self.notify()

    def request(self, q: dict) -> bytes:
        name = q["session"] or f"none-{q['stream']}"
        if (fields := q.get("id_request")) is not None:
            return id_request(**fields, user=name, frame_bytes=self.B_in)
        if self.live_model is None:
            return request_bytes(name, q["frames"], self.B_in)
        body = q.get("body") or (self.body if self.body is not None else live_body(q["session"], self.live_model))
        return live_request_bytes(name, body, q["frames"], self.B_in)

    async def read(self, reader: asyncio.StreamReader, origin: float) -> None:
        h = hashlib.sha256()
        while data := await reader.read(1 << 22):
            now, mv, before = self.clock(), memoryview(data), self.bytes // self.bucket_bytes
            while mv:
                take = min(len(mv), self.window_bytes - self.bytes % self.window_bytes)
                h.update(mv[:take])
                self.bytes, mv = self.bytes + take, mv[take:]
                if self.bytes % self.window_bytes == 0:
                    self.log["window_sha256"].append(h.hexdigest())
                    h = hashlib.sha256()
            for t in range(before, self.bytes // self.bucket_bytes):
                self.log["bucket_lag_ms"].append(round((now - origin - t * self.tau) * 1000, 3))
            self.notify()

    def dump(self) -> dict:
        return self.log | {"bytes": self.bytes}


class Publisher:
    """The proxy's `publish`: each record goes to its link's file, packed, and its commitment to `committed.jsonl`, on a
    worker thread, or inline with `threaded=False`."""

    def __init__(self, out: Path, setup: Setup, threaded: bool = True) -> None:
        self.out = out
        self.files = {setup.egress_params: out / "responses.records", setup.ingress_params: out / "requests.records"}
        self.executor = ThreadPoolExecutor(1) if threaded else None
        self.futures: list = []

    def keep(self, record: Record, params: GridParams) -> None:
        write_blobs(self.files[params], pack_record(record))
        with open(self.out / "committed.jsonl", "a") as fh:
            fh.write(committed_line(commit(record, params)) + "\n")

    def __call__(self, record: Record, params: GridParams) -> None:
        if self.executor is None:
            self.keep(record, params)
        else:
            self.futures.append(self.executor.submit(self.keep, record, params))

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            for f in self.futures:
                f.result()


def proxy_log(proxies: Sequence[Proxy]) -> dict:
    """The run's proxy log: one proxy's, or, for an outage plan, the proxies before and after the outage joined, each
    list in order and each count summed, with the windows each served (`served`). A proxy numbers the requests it cut
    from 1 (`active.Requests`), so a later proxy's streams are shifted by the requests the ones before cut. A paced
    proxy's log has its offset and each bucket's pacer report (`paced`, in ms after the bucket's time)."""
    last = proxies[-1]
    shift = list(itertools.accumulate((p.requests.stream for p in proxies), initial=0))
    frames = [[pos, s + shift[j], *rest] for j, p in enumerate(proxies) for pos, s, *rest in p.frames]
    salts = {} if last.salt is None else {"salts": {"released": sum(p.salts_released for p in proxies),
                                                    "delivered": sum(p.salts_delivered for p in proxies)}}
    served_by = {} if len(proxies) == 1 else {"served": [[p.start, p.windows] for p in proxies]}
    paced = {} if last.paced is None else {"paced": {
        "ms": last.paced * 1000, "columns": ["bucket", "arrived_ms", "start_ms", "end_ms", "written"],
        "releases": [[r.t, round(r.arrived * 1000, 3), round(r.start * 1000, 3), round(r.end * 1000, 3), r.written]
                     for p in proxies for r in p.releases]}}
    return {"lateness_ms": {k: [round(x * 1000, 3) for p in proxies for x in p.lateness[k]] for k in last.lateness},
            "syncs": [[s.delta, str(s.rho)] for s in last.shaper.syncs],
            "frames": frames, "hands": [h for p in proxies for h in p.shaper.hands],
            "link_frames_read": sum(p.link_frames_read for p in proxies),
            "streams": sum(len(p.streams) for p in proxies), "slack_ms": last.slack * 1000,
            "forwards": [[st.window, None if st.delivered is None else round((st.delivered - st.deadline) * 1000, 3)]
                         for p in proxies for st in p.streams.values() if st.deadline is not None],
            "prover": None if last.choices is None else last.choices.log} | salts | served_by | paced | _usage()


def stand_in(window: int) -> Registration:
    """A stand-in for the proof system's registration of `window`, which the replay has none of: a record whose
    provenance names the window (`active.served_window`), with its receipt."""
    record = {"window": {"kind": "served", "window": window}}
    return Registration(record, R.receipt(record, "active-replay", received=str(window)), R.digest(record), window)


def reveal_deadline(plan: dict, window_s: float) -> float:
    """The salted proxy's reveal deadline here: the replay plays the verifier only once its whole run is over, so the
    deadline covers every window of the plan and one more, where a deployment's covers one PoUW window."""
    return (plan["windows"] + 1) * window_s


def revealed(proxies: Sequence[Proxy]) -> dict[str, str]:
    """Every served window's salt, as the verifier's `reveal` calls give them once the run's records are published."""
    return {str(w): p.reveal(w, stand_in(w)).hex() for p in proxies for w in range(p.start, p.windows)}


def write_logs(out: Path, proxies: Sequence[Proxy] = (), server: Server | None = None,
               peer: Peer | None = None) -> None:
    if proxies:
        with gzip.open(out / "proxy.json.gz", "wt") as fh:
            json.dump(proxy_log(proxies), fh)
    if server is not None:
        (out / "server.json").write_text(json.dumps(server.log | _usage()))
        if isinstance(server, LiveServer):
            server.dump_events(out / "server.events.gz")
    if peer is not None:
        (out / "peer.json").write_text(json.dumps(peer.dump() | _usage()))


def server_process(plan: dict, params: dict, conn, out: str, host: str = "127.0.0.1", port: int = 0) -> None:
    _quiet_runtime()
    asyncio.run(_server(plan, params, conn, Path(out), host, port))


# The server, the proxy and the peer each talk to their parent over `conn` (anything with `send` and a blocking `recv`):
# each sends one value once it is listening or connected, then receives the origin; the server receives one more value
# before it stops. Each listens on, or connects to, `host`; a `port` of 0 is any free one, which the listener sends.
# A proxy with its salt on receives "reveal" once its run's records are published, the verifier's call, and sends back
# every window's salt (`revealed`). Under an outage plan the proxy stops listening once the windows before the outage
# are served, and listens again half a window before the windows after it, on the same port; the peer connects again a
# quarter of a window before them.

async def _server(plan: dict, params: dict, conn, out: Path, host: str = "127.0.0.1", port: int = 0) -> None:
    loop = asyncio.get_running_loop()
    server = LiveServer(plan, params) if "upstream" in params else Server(plan, params)
    listening = await asyncio.start_server(server.handle, host, port)
    conn.send(listening.sockets[0].getsockname()[1])
    await server.serve(await loop.run_in_executor(None, conn.recv))
    await loop.run_in_executor(None, conn.recv)
    listening.close()
    server.close()
    write_logs(out, server=server)


def prover_args(plan: dict, setup: Setup, background: bool = True) -> dict:
    """The proxy's advice: the plan's syncs, or, under an online rule, its own prover's over the plan's windows at
    `plan["prover_lag"]`, choosing in a worker process (`active.worker`, started here, before the run), or at once
    with `background=False`, for a driver on a virtual clock."""
    if "sync_rule" not in plan:
        return {"syncs": [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]}
    prover = online.Prover(setup.sync_set, setup.egress_params.buckets, online.rule(plan["sync_rule"]), setup.envelope,
                           lag=plan["prover_lag"])
    return {"syncs": [], "prover": prover, "windows": plan["windows"],
            "executor": worker() if background else InlineExecutor()}


class ReleaseQueue(asyncio.Queue):
    """A request's queue to the server (a `Proxy` stream's `out`): its bytes are held until its head is whole, or its
    last byte, then put as `release.with_release` gives them for release `n` of run `run`; the rest passes as it
    comes."""

    def __init__(self, n: int, run: bytes) -> None:
        super().__init__()
        self.n, self.run, self.head = n, run, bytearray()

    def put_nowait(self, item: tuple[bytes, bool]) -> None:
        if self.head is None:
            return super().put_nowait(item)
        data, last = item
        self.head += data
        if last or b"\r\n\r\n" in self.head:
            data, self.head = with_release(bytes(self.head), self.n, self.run), None
            super().put_nowait((data, last))


class ReleaseProxy(Proxy):
    """`active.Proxy` whose server gets each request with the release order (`release.with_release`) under the proxy's
    `run_id`: request s on the link (`active.Requests`' stream s) is release s − 1, since a request ends on the link
    before the next starts and the ingress tick counts each at its end (`requests_released`). The link, its frames and
    the records are `Proxy`'s: only the server's copy of a request changes."""

    def _stream(self, s: int):
        new = s not in self.streams
        st = super()._stream(s)
        if new:
            st.out = ReleaseQueue(s - 1, self.run_id)
        return st


class Stalls(asyncio.Event):
    """Test-only (`--stall`): the proxy's change event, which pauses the proxy right after an ingress tick has released
    a request and before the proxy writes it upstream, once for each (bucket, seconds): at the first such tick at or
    after the bucket, once that boundary's egress tick has run too (`active.TICK_ORDER` runs it after the ingress
    tick), so the pause falls between two boundaries' ticks. `pause(seconds)` stops the proxy; `done` lists each stall
    with how long the proxy's clock saw it last."""

    def __init__(self, proxy: Proxy, stalls: Sequence[tuple[int, float]], pause) -> None:
        super().__init__()
        self.proxy, self.todo, self.pause = proxy, sorted(stalls), pause
        self.released, self.done = 0, []

    def set(self) -> None:
        p = self.proxy
        t = len(p.lateness["ingress"]) - 1
        if len(p.lateness["egress"]) > t:
            if self.todo and p.requests_released > self.released and t >= self.todo[0][0]:
                bucket, seconds = self.todo.pop(0)
                start = p.clock()
                self.pause(seconds)
                self.done.append({"bucket": bucket, "tick": t, "ms": seconds * 1000,
                                  "at_ms": round((start - p.origin) * 1000, 3),
                                  "paused_ms": round((p.clock() - start) * 1000, 3)})
            self.released = p.requests_released
        super().set()


def _stopper():
    """A pause of the whole process: it stops itself (SIGSTOP), and a child forked now, before any thread starts, sends
    SIGCONT the pause after it has stopped, from the core the process was on."""
    r, w = os.pipe()
    parent = os.getpid()
    if os.fork() == 0:
        os.close(w)
        while len(b := os.read(r, 8)) == 8:
            os.sched_setaffinity(0, os.sched_getaffinity(parent))
            while Path(f"/proc/{parent}/stat").read_text().rsplit(") ", 1)[1][0] != "T":
                pass
            time.sleep(struct.unpack("d", b)[0])
            os.kill(parent, signal.SIGCONT)
        os._exit(0)
    os.close(r)

    def pause(seconds: float) -> None:
        os.write(w, struct.pack("d", seconds))
        os.kill(parent, signal.SIGSTOP)

    return pause


def proxy_process(plan: dict, setup_report: dict, upstream: tuple[str, int], conn, out: str,
                  stalls: Sequence[tuple[int, float]] = (), host: str = "127.0.0.1", port: int = 0) -> None:
    _quiet_runtime()
    pause = _stopper() if stalls else None
    asyncio.run(_proxy(plan, Setup(setup_report), upstream, conn, Path(out), stalls, pause, host, port))


def served_args(plan: dict, salt: bool = False) -> dict:
    """A served link's profile for its proxy (`active.Proxy(served=…)`, which admits requests by the admission rule),
    from the plan's `served` (none otherwise), counting the salt frame when the salt is on."""
    if "served" not in plan:
        return {}
    s = plan["served"]
    return {"served": ServedProfile(s["response_frames"], s["request_frames"], plan["windows"], int(salt))}


def phase_args(plan: dict, start: int, end: int) -> dict:
    """The arguments of the proxy that serves windows `start` … `end` − 1 of an outage plan (none otherwise)."""
    if "outage" not in plan:
        return {}
    return {"windows": end, "start": start, "position": plan["outage"]["position"] if start else 0}


async def _proxy(plan: dict, setup: Setup, upstream: tuple[str, int], conn, out: Path,
                 stalls: Sequence[tuple[int, float]] = (), pause=None, host: str = "127.0.0.1", port: int = 0,
                 salt: bool = False) -> None:
    publish = Publisher(out, setup)
    advice = prover_args(plan, setup)
    syncs = advice.pop("syncs")
    window_s = setup.egress_params.buckets * setup.egress_params.bucket_ms / 1000
    proxies: list[Proxy] = []
    for k, (start, end) in enumerate(served(plan)):
        done = asyncio.get_running_loop().create_future()
        holder: dict = {}

        async def on_link(reader, writer, holder=holder, done=done) -> None:
            if "link" in holder:  # one link: a second connection is never the peer's
                writer.close()
                return
            holder["link"] = writer
            await holder["proxy"].run(reader, writer)
            done.set_result(None)

        def make() -> Proxy:
            return ReleaseProxy(upstream, setup.egress_params, setup.ingress_params, setup.sync_set, syncs,
                                origin=origin, publish=publish, egress_link=setup.egress_link,
                                ingress_link=setup.ingress_link, salt=fresh_salt if salt else None,
                                reveal_deadline=reveal_deadline(plan, window_s), ordinals=bool(plan.get("token_ids")),
                                compact=bool(plan.get("compact")), **advice, **phase_args(plan, start, end),
                                **served_args(plan, salt))

        if k == 0:
            listening = await asyncio.start_server(on_link, host, port)
            port = listening.sockets[0].getsockname()[1]
            conn.send(port)
            origin = conn.recv()
            proxy = holder["proxy"] = make()
        else:
            await asyncio.sleep(origin + (start - 0.5) * window_s - time.monotonic())
            proxy = holder["proxy"] = make()
            listening = await asyncio.start_server(on_link, host, port)
        if stalls and k == 0:
            proxy._changed = Stalls(proxy, stalls, pause)
        proxies.append(proxy)
        await done
        listening.close()
    publish.close()
    write_logs(out, proxies)
    if stalls:
        (out / "stalls.json").write_text(json.dumps(proxies[0]._changed.done))
    if salt:
        if (call := conn.recv()) != "reveal":
            raise ValueError(f"a salted proxy's last call is 'reveal', not {call!r}")
        conn.send(revealed(proxies))


def peer_process(plan: dict, params: dict, link_port: int, conn, out: str, host: str = "127.0.0.1") -> None:
    _quiet_runtime()
    asyncio.run(_peer(plan, params, link_port, conn, Path(out), host))


async def _peer(plan: dict, params: dict, link_port: int, conn, out: Path, host: str = "127.0.0.1") -> None:
    peer = Peer(plan, params)
    reader, writer = await asyncio.open_connection(host, link_port, limit=1 << 22)
    conn.send("connected")
    origin = conn.recv()
    for k, (start, enqueued) in enumerate(peer.phases()):
        if k:
            await asyncio.sleep(origin + (start - 0.25) * peer.window_s - time.monotonic())
            reader, writer = await _reconnect(host, link_port, origin + start * peer.window_s)
        sending = asyncio.create_task(peer.send(writer, origin, enqueued))
        await peer.read(reader, peer.origin_of(origin, start))
        sending.cancel()
        writer.close()
    write_logs(out, peer=peer)


async def _reconnect(host: str, port: int, by: float):
    """The link again, once the proxy back from maintenance listens: refused connections are retried until `by`."""
    while True:
        try:
            return await asyncio.open_connection(host, port, limit=1 << 22)
        except ConnectionRefusedError:
            if time.monotonic() >= by:
                raise
            await asyncio.sleep(0.01)


def wire_params(setup: Setup) -> dict:
    e = setup.egress_params
    return {"buckets": e.buckets, "bucket_ms": e.bucket_ms, "allocation": e.allocation, "frame_bytes": e.frame_bytes,
            "ingress_frame_bytes": setup.ingress_params.frame_bytes}


class Evicted(Exception):
    """SIGTERM: the lease is reclaimed."""


def run_configuration(plan: dict, report: dict, out: Path, nice: int | None = None,
                      upstream: tuple[str, int] | None = None, cpus: Sequence[int] | None = None,
                      stalls: Sequence[tuple[int, float]] = ()) -> dict:
    """One configuration in real time: the server, then the proxy, then the peer, released together at one origin.
    With `nice`, every thread of the three runs at that niceness (`renice`, through sudo); with `cpus`, the server's,
    the proxy's and the peer's threads each run on one core of their own. Threads they start later inherit both. With
    `upstream`, the server is `LiveServer` in front of the endpoint there. With `stalls` (test-only), the proxy stops
    itself as `Stalls` says. On `Evicted` the three are terminated and it is raised again."""
    params = wire_params(Setup(report)) | ({"upstream": list(upstream)} if upstream else {})
    started: list = []
    try:
        return _run_configuration(plan, report, out, nice, cpus, params, started, stalls)
    except Evicted:
        for proc in started:
            proc.terminate()
        raise


def _run_configuration(plan: dict, report: dict, out: Path, nice: int | None, cpus: Sequence[int] | None,
                       params: dict, started: list, stalls: Sequence[tuple[int, float]] = ()) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    (s, s_child), (p, p_child), (q, q_child) = ctx.Pipe(), ctx.Pipe(), ctx.Pipe()
    server = ctx.Process(target=server_process, args=(plan, params, s_child, str(out)))
    started.append(server)
    server.start()
    port = s.recv()
    proxy = ctx.Process(target=proxy_process, args=(plan, report, ("127.0.0.1", port), p_child, str(out), stalls))
    started.append(proxy)
    proxy.start()
    link = p.recv()
    peer = ctx.Process(target=peer_process, args=(plan, params, link, q_child, str(out)))
    started.append(peer)
    peer.start()
    q.recv()
    for proc, cpu in zip((server, proxy, peer), cpus or (None,) * 3):
        for tid in os.listdir(f"/proc/{proc.pid}/task"):
            if nice is not None:
                subprocess.run(["sudo", "-n", "renice", "-n", str(nice), "-p", tid], check=True, capture_output=True)
            if cpu is not None:
                os.sched_setaffinity(int(tid), {cpu})
    origin = time.monotonic() + STARTUP_S
    for c in (s, p, q):
        c.send(origin)
    proxy.join()
    peer.join()
    s.send("stop")
    server.join()
    codes = {"server": server.exitcode, "proxy": proxy.exitcode, "peer": peer.exitcode}
    (out / "processes.json").write_text(json.dumps({"exit": codes, "origin": origin, "clock": "real",
                                                    "nice": nice, "cpus": list(cpus) if cpus else None,
                                                    "server": "live" if "upstream" in params else "fake"}))
    return codes


def run_virtual(plan: dict, report: dict, out: Path, upstream: tuple[str, int] | None = None,
                stalls: Sequence[tuple[int, float]] = (), salt: bool = False, proxy_args: dict | None = None,
                publish=None, after=None) -> dict:
    """One configuration on an injected clock: the same sockets, proxy, server and peer in one event loop, the clock
    moved to each instant something is due (a tick, a server batch, a send) once everything due before it is done.
    With `upstream`, the server is `LiveServer` in front of an endpoint that answers without the clock moving. A stall
    (`stalls`, test-only) moves the clock past the pause at once, before anything else runs. With `salt`, the proxy's
    salt is on, and `salts.json` holds what `reveal` gives once the run is over. Under an outage plan the first
    proxy's link is closed with its last window, and at the outage window's first instant the proxy back from
    maintenance takes the peer's second connection. `proxy_args` override the proxy's arguments, `publish` replaces
    `Publisher`, and `after(proxies, clock)` runs on the loop once the logs are written (a caller's reveal endpoint, say);
    a proxy with a challenger key takes no in-process `reveal`, so it writes no `salts.json`."""
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(_virtual(plan, Setup(report), out, upstream, stalls, salt, proxy_args or {}, publish, after))
    (out / "processes.json").write_text(json.dumps({"exit": {"server": 0, "proxy": 0, "peer": 0}, "origin": 0.0,
                                                    "clock": "virtual"}))
    return {"server": 0, "proxy": 0, "peer": 0}


def virtual_process(plan: dict, report: dict, out: Path, stalls: Sequence[tuple[int, float]] = ()) -> None:
    _quiet_runtime()
    run_virtual(plan, report, out, stalls=stalls)


async def _virtual(plan: dict, setup: Setup, out: Path, upstream: tuple[str, int] | None = None,
                   stalls: Sequence[tuple[int, float]] = (), salt: bool = False, proxy_args: dict | None = None,
                   publish=None, after=None) -> None:
    e, i = setup.egress_params, setup.ingress_params
    params = wire_params(setup) | ({"upstream": list(upstream)} if upstream else {})
    clock = VirtualClock(-1.0)
    proxies: list[Proxy] = []

    def notify() -> None:
        proxies[-1]._changed.set()

    server = (LiveServer if upstream else Server)(plan, params, clock, clock.sleep, notify)
    peer = Peer(plan, params, clock, clock.sleep, notify)
    listening = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    publish = publish or Publisher(out, setup, threaded=False)
    advice = prover_args(plan, setup, background=False)
    syncs = advice.pop("syncs")
    runs = served(plan)
    phases = peer.phases()

    async def connect(k: int) -> tuple:
        """The proxy of the k-th run of served windows, the peer's link to it, and the run and the peer's tasks."""
        (start, end), (_, enqueued) = runs[k], phases[k]
        args = {"salt": fresh_salt if salt else None, "ordinals": bool(plan.get("token_ids")),
                "compact": bool(plan.get("compact")),
                "reveal_deadline": reveal_deadline(plan, e.buckets * e.bucket_ms / 1000)} | proxy_args
        proxy = ReleaseProxy(listening.sockets[0].getsockname()[:2], e, i, setup.sync_set, syncs, origin=0.0,
                             clock=clock, sleep=clock.sleep, publish=publish, egress_link=setup.egress_link,
                             ingress_link=setup.ingress_link, **args, **advice, **phase_args(plan, start, end))
        if stalls and k == 0:
            proxy._changed = Stalls(proxy, stalls, lambda seconds: clock.advance(clock.now + seconds))
        proxies.append(proxy)
        accepted: asyncio.Future = asyncio.get_running_loop().create_future()
        link = await asyncio.start_server(lambda r, w: accepted.set_result(asyncio.create_task(proxy.run(r, w))),
                                          "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection(*link.sockets[0].getsockname()[:2], limit=1 << 22)
        running = await accepted
        return link, writer, running, [asyncio.create_task(peer.send(writer, 0.0, enqueued)),
                                       asyncio.create_task(peer.read(reader, peer.origin_of(0.0, start)))]

    async def disconnect(link, writer, running, tasks) -> None:
        await running
        await tasks[1]
        tasks[0].cancel()
        writer.close()
        link.close()

    phase = 0
    current = await connect(phase)
    serving = asyncio.create_task(server.serve(0.0))
    horizon = plan["windows"] * e.buckets
    tau_e, tau_i = e.bucket_ms / 1000, i.bucket_ms / 1000
    instants = sorted({t * tau_e for t in range(horizon)} | {t * tau_i for t in range(plan["windows"] * i.buckets)}
                      | {b * tau_e + tau_e / 2 for b in range(horizon - 1)} | {t / 1000 for t in peer.by_time})
    sends = sorted(t / 1000 for t in peer.by_time)
    end = (horizon - 1) * tau_e

    def ticks(x: float, tau: float, buckets: int) -> int:
        n = math.floor(x / tau + 1e-9) + 1 if x >= 0 else 0
        return sum(max(0, min(n, b * buckets) - a * buckets) for a, b in runs)

    def total(f) -> int:
        return sum(f(p) for p in proxies)

    def settled(x: float) -> bool:
        ticks_e = total(lambda p: len(p.lateness["egress"]))
        batches = min(horizon - 1, math.floor((x - tau_e / 2) / tau_e + 1e-9) + 1) if x >= tau_e / 2 else 0
        return (ticks_e >= ticks(x, tau_e, e.buckets)
                and total(lambda p: len(p.lateness["ingress"])) >= ticks(x, tau_i, i.buckets)
                and getattr(server, "processed", horizon) + 1 >= batches and getattr(server, "open", 0) == 0
                and not any(d <= x + 1e-9 for d in getattr(server, "held", ()))
                and peer.instants_sent >= bisect.bisect_right(sends, x + 1e-9)
                and total(lambda p: p.link_frames_read) >= peer.frames_sent
                and server.log["requests"] >= total(lambda p: p.requests_released)
                and len(server.log.get("salts", ())) >= total(lambda p: p.salts_released)
                and total(lambda p: len(p.frames)) >= server.log["events"]
                and peer.bytes >= ticks_e * e.allocation * e.frame_bytes)

    for x in instants:
        if x > end + 1e-9:
            break
        if phase + 1 < len(runs) and x >= runs[phase][1] * e.buckets * tau_e - 1e-9:
            await disconnect(*current)
            phase += 1
            current = await connect(phase)
        clock.advance(x + 1e-9)
        await proxies[-1].until(lambda _, x=x: settled(x))
    await disconnect(*current)
    await serving
    listening.close()
    server.close()
    publish.close()
    write_logs(out, proxies, server, peer)
    if stalls:
        (out / "stalls.json").write_text(json.dumps(proxies[0]._changed.done))
    if salt and proxies[0].challenger_key is None:
        (out / "salts.json").write_text(json.dumps(revealed(proxies)))
    if after is not None:
        await after(proxies, clock)


# The audit.

def ingress_releases(records: Sequence[Record], ingress: GridParams, egress: GridParams) -> tuple[dict, dict]:
    """Each stream's release, as the egress bucket of its last frame's ingress bucket, and every ingress frame's
    release bucket by stream, read from the ingress records in window order: their `DATA` frames cut into requests as
    the proxy cuts them (`active.Requests`), the k-th request stream k."""
    last, frames, split = {}, defaultdict(list), Requests()
    for rec in sorted(records, key=lambda r: r.window):
        for t, row in enumerate(rec.grid):
            b = rec.window * ingress.buckets + t
            for slot in row[: row_count(row)]:
                kind, _, payload = decode_frame(slot)
                if kind != DATA:
                    continue
                stream, _, end = split.feed(payload)
                frames[stream].append(b)
                if end:
                    last.setdefault(stream, b * ingress.bucket_ms // egress.bucket_ms)
    return last, frames


def percentiles_ms(values: Sequence[float]) -> dict:
    return replay.hist_percentiles(Counter(values)) if values else {}


def released_sessions(records: Sequence[Record]) -> dict[str, int]:
    """Each live request's session (its `user`, `none-k` without one) and the stream the proxy released it as, read
    from the ingress records as `ingress_releases` cuts them. A fault that drops a window's queue drops its requests,
    so the streams after it aren't the plan's."""
    data, split = defaultdict(bytes), Requests()
    for rec in sorted(records, key=lambda r: r.window):
        for row in rec.grid:
            for slot in row[: row_count(row)]:
                kind, _, payload = decode_frame(slot)
                if kind == DATA:
                    stream, chunk, _ = split.feed(payload)
                    data[stream] += chunk
    out = {}
    for stream, raw in data.items():
        try:
            user = json.loads(raw.partition(b"\r\n\r\n")[2]).get("user")
        except ValueError:
            user = None
        out[user or f"none-{stream}"] = stream
    return out


def live_program(plan: dict, release: dict[int, int], params: GridParams,
                 path: Path, stream_of: dict[str, int] | None = None) -> tuple[list[LogicalFrame], list[int]]:
    """The Program a live server served: the shim's events in position order, each anchored at its stream's release,
    and the ready bucket each was meant for, the one after the bucket it was written in; a frame is what follows the
    event's `id:` line, as the proxy frames it, or under `plan["compact"]` the unit's record as its cell
    (`active.compact_cell`). An event longer than a frame is kept unpadded, as a frame of the wrong size, which the
    Certifier faults as malformed just as the proxy did. `stream_of` maps a session to its stream (`released_sessions`),
    the plan's by default."""
    stream_of = stream_of or {q["session"] or f"none-{q['stream']}": q["stream"] for q in plan["requests"]}
    events = sorted(read_events(path))
    if [p for p, *_ in events] != list(range(len(events))):
        raise ValueError(f"{path}: the positions aren't 0 … n−1")

    def frame(event: bytes) -> bytes:
        if plan.get("compact"):
            return compact_cell(event[-COMPACT_RECORD.size :], params.frame_bytes)
        body = event_body(event)
        if HEADER.size + len(body) > params.frame_bytes:
            return HEADER.pack(DATA, CANONICAL_STREAM, len(body)) + body
        return encode_frame(DATA, CANONICAL_STREAM, body, params.frame_bytes)

    frames = [LogicalFrame(frame(event), release[stream_of[name]], k) for _, name, k, _, event in events]
    return frames, [b + 1 for *_, b, _ in events]


def live_streams(plan: dict, path: Path, stream_of: dict[str, int] | None = None) -> list[int]:
    """The stream of each of `live_program`'s frames, in its order."""
    stream_of = stream_of or {q["session"] or f"none-{q['stream']}": q["stream"] for q in plan["requests"]}
    return [stream_of[name] for _, name, *_ in sorted(read_events(path))]


def live_comparisons(plan: dict, setup: Setup, frames: Sequence[LogicalFrame], ready: Sequence[int]) -> dict:
    """What other provers would have met on a live run's frames as read: the windows whose every frame is ready by
    its hand time under the trace model's syncs, and the windows a hindsight prover (`choose_syncs`) misses."""
    T, windows = setup.egress_params.buckets, plan["windows"]
    model_syncs = [ClockSync(d, Fraction(r)) for d, r in plan["model_syncs"]]
    late = {f.anchor // T for f, h, r in zip(frames, handing_buckets(frames, model_syncs, T), ready) if r > h}
    try:
        chosen, _, misses = choose_syncs([(frames, ready)], setup.sync_set, T)
        hindsight = {"misses": misses, "delta": sorted({s.delta for s in chosen}),
                     "rho": [str(r) for r in sorted({s.rho for s in chosen})]}
    except ValueError as err:
        hindsight = {"error": str(err)}
    return {"model_syncs_met_windows": sum(w not in late for w in range(windows)), "hindsight": hindsight}


def audit_configuration(plan: dict, report: dict, run: Path, audit=Audit, client: Path | None = None) -> dict:
    """The audit of one run's published records, and the comparisons with the reference and the model. The verifier
    is `audit`, a class with `audit.Audit`'s interface; the peer's `peer.json` is in `client`, by default `run`."""
    setup = Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    T, windows = e.buckets, plan["windows"]
    horizon = windows * T
    with gzip.open(run / "proxy.json.gz", "rt") as fh:
        log = json.load(fh)
    declared = log["syncs"] if "sync_rule" in plan else plan["syncs"]
    syncs = [ClockSync(d, Fraction(r)) for d, r in declared]
    in_records = [unpack_record(b, i.frame_bytes) for b in read_blobs(run / "requests.records")]
    release, in_frames = ingress_releases(in_records, i, e)
    live = "live_model" in plan
    if live and "covert" in plan:
        plan = plan | {"covert": plan["covert"] | json.loads((run / "server.json").read_text())["covert"]}
    if live:
        stream_of = released_sessions(in_records)
        frames, meant = live_program(plan, release, e, run / "server.events.gz", stream_of)
        streams = live_streams(plan, run / "server.events.gz", stream_of)
    else:
        frames, meant = program(plan, release, e)
        streams = program_streams(plan, release, e)
    intern = {f.payload: f.payload for f in frames}
    out_records = [unpack_record(b, e.frame_bytes, intern) for b in read_blobs(run / "responses.records")]
    committed = read_committed(run / "committed.jsonl")
    verdicts = []
    for kind, records in (("raw", in_records + out_records), ("committed", committed)):
        a = audit()
        v = a.check_link(setup.egress_link, records, frames, syncs, setup.sync_set, e)
        v += a.check_ingress_link(setup.ingress_link, records, windows)
        verdicts.append(v)
    rows = [{"config": plan["name"], "link": a.link, "window": a.window,
             "direction": "egress" if a.link == setup.egress_link else "ingress",
             "raw": a.violation, "committed": b.violation} for a, b in zip(*verdicts, strict=True)]

    seen = {p: (stream, k, anchor, ready) for p, stream, k, anchor, ready in log["frames"]}
    ready = [seen[p][3] if p in seen else horizon for p in range(len(frames))]
    ref = run_shaper(frames, ready, syncs, Certifier(setup.egress_link, e))
    logical = handing_buckets(frames, syncs, T)
    rule, last = [], 0
    for h, r in zip(logical, ready, strict=True):
        last = max(h, r, last)
        rule.append(last)
    hands = log["hands"]
    model_release = {int(k): v for k, v in plan["model_release"].items()}
    unserved = {plan["outage"]["window"]} if "outage" in plan else set()
    comparisons = {
        "records_are_run_shapers": [r for r in ref if r.window not in unserved] ==
            sorted((r for r in out_records), key=lambda r: r.window),
        "hands_are_the_rules": hands == [h for h in rule if h < horizon],
        "hands_max_diff_from_rule": max((abs(a - b) for a, b in zip(hands, rule)), default=0),
        "frames_read": len(log["frames"]), "frames_meant": len(frames),
        "frames_log_is_the_programs": sorted((p, s, k, a) for p, (s, k, a, _) in seen.items()) ==
            [(p, streams[p], f.index, f.anchor) for p, f in enumerate(frames) if p in seen],
        "ready_off_meant": sum(1 for p, (_, _, _, r) in seen.items() if p < len(meant) and r != meant[p]),
        "releases_off_model": sum(1 for s, r in release.items() if model_release.get(s) != r)
            + sum(1 for s in model_release if s not in release),
        "model_misses": plan["model_misses"],
        "declared_are_planned": declared == plan["syncs"],
    }
    shaped = {r.window: r.grid for r in ref}
    by_window = {r.window: r.grid for r in out_records}
    mismatched = [r["window"] for r in rows if r["direction"] == "egress" and r["raw"] == "mismatch"]
    comparisons["mismatches"] = len(mismatched)
    comparisons["mismatches_from_late_frames"] = sum(w in by_window and by_window[w] == shaped.get(w) for w in mismatched)
    symbols, margin = covert_symbols(plan, T)
    if "covert" in plan:
        comparisons["covert_declared_are_predicted"] = declared == plan["covert"]["predicted"]
    comparisons["covert"] = {"symbols": symbols, "declared": declared, "margin": margin} | \
        covert.measure(symbols, declared, setup.sync_set, margin=margin)
    if live:
        comparisons |= live_comparisons(plan, setup, frames, ready)
    else:
        model_hands = handing_buckets(model_frames(plan["sessions"], model_release, horizon)[0], syncs, T)
        comparisons["hands_max_diff_from_model"] = max((abs(a - b) for a, b in zip(hands, model_hands)), default=0)

    released, real, slots, p = {}, 0, 0, 0
    comparisons["released_not_the_programs"] = 0
    # The shaper releases in canonical order, so a released slot is the next Program frame with its bytes: frames may
    # repeat (one token's event), and a faulted window drops some.
    for rec in sorted(out_records, key=lambda r: r.window):
        for t, row in enumerate(rec.grid):
            n = row_count(row)
            real, slots = real + n, slots + len(row)
            for slot in row[:n]:
                q = next((j for j in range(p, len(frames)) if frames[j].payload == slot), None)
                if q is None:
                    comparisons["released_not_the_programs"] += 1
                else:
                    released[q], p = rec.window * T + t, q + 1
    send_ms = {q["stream"]: q["trace_ms"] for q in plan["requests"]}
    hist = {k: Counter() for k in ("shaping", "queueing", "first_frame", "first_frame_from_send")}
    for p, f in enumerate(frames):
        if p < len(hands):
            hist["shaping"][(hands[p] - ready[p]) * e.bucket_ms] += 1
        if p in released:
            hist["queueing"][(released[p] - hands[p]) * e.bucket_ms] += 1
            if f.index == 0:
                hist["first_frame"][(released[p] - f.anchor) * e.bucket_ms] += 1
                s = streams[p]
                hist["first_frame_from_send"][round(released[p] * e.bucket_ms - send_ms[s])] += 1
    delivery = Counter()
    for q in plan["requests"]:
        for b in in_frames.get(q["stream"], []):
            delivery[round(b * i.bucket_ms - q["trace_ms"], 3)] += 1

    peer = json.loads(((client or run) / "peer.json").read_text())
    zero = bytes(e.allocation * e.frame_bytes)
    wire = []
    for rec in sorted(out_records, key=lambda r: r.window):
        h = hashlib.sha256()
        for row in rec.grid:
            n = row_count(row)
            h.update(b"".join(row[:n]))
            h.update(zero[: (e.allocation - n) * e.frame_bytes])
        wire.append(h.hexdigest())
    comparisons["wire_is_the_records"] = wire == peer["window_sha256"]
    server = json.loads((run / "server.json").read_text())
    lateness = log["lateness_ms"]
    timing = {
        "slack_ms": log["slack_ms"],
        "tick_late_ms": {k: percentiles_ms(v) | {"over_slack": sum(x > log["slack_ms"] for x in v), "ticks": len(v)}
                         for k, v in lateness.items()},
        "server_batch_late_ms": percentiles_ms(server["batch_late_ms"]),
        "server_receipt_late_ms": percentiles_ms(server["receipt_late_ms"]),
        "server_late_registrations": server["late_registrations"],
        "peer_send_late_ms": percentiles_ms(peer["send_late_ms"]),
        "wire_bucket_lag_ms": percentiles_ms(peer["bucket_lag_ms"]),
        "cpu_s": {"proxy": log["cpu_s"], "server": server["cpu_s"], "peer": peer["cpu_s"]},
        "max_rss_mb": {"proxy": log["max_rss_mb"], "server": server["max_rss_mb"], "peer": peer["max_rss_mb"]},
    }
    if live:
        timing |= {"server_first_event_ms": percentiles_ms(server["first_event_ms"]),
                   "server_failed": len(server["failed"]), "server_failures": server["failed"][:20],
                   "server_cut_at_horizon": server["cut_at_horizon"], "server_requests": server["requests"],
                   "server_padded": {"responses": len(padded := server.get("padded", [])),
                                     "events": sum(n for _, n in padded)}}
    out = {"config": plan["name"], "windows": windows, "rows": rows, "comparisons": comparisons, "timing": timing,
           "egress": {"real": real, "slots": slots, "frames": len(frames)},
           "hist": {k: {str(x): n for x, n in v.items()} for k, v in hist.items()},
           "ingress_delivery": {str(x): n for x, n in delivery.items()},
           "lateness_ms": lateness}
    if (path := run / "salts.json").exists():
        out["salt"] = audit_salt(setup, {int(w): bytes.fromhex(s) for w, s in json.loads(path.read_text()).items()},
                                 in_records, out_records, server, windows, audit())
    return out


def audit_salt(setup: Setup, salts: dict[int, bytes], in_records: Sequence[Record], out_records: Sequence[Record],
               server: dict, windows: int, audit) -> dict:
    """The audit's read of a salted run's salt frames (`check_salts`, on the raw ingress records and the salts `reveal`
    gave), and where the salt frames went: their count in each window's ingress record and in the egress records,
    whether any egress frame holds a salt, and whether the server received each window's salt."""
    verdicts = check_salts(audit, setup.ingress_link, in_records, windows, salts)

    def salt_frames(r: Record) -> int:
        return sum(len(f) >= HEADER.size and decode_frame(f)[0] == SALT
                   for row in r.grid for f in row[: row_count(row)])

    egress = [f for r in out_records for row in r.grid for f in row[: row_count(row)]]
    return {"accepted": all(v.accepted for v in verdicts),
            "violations": [[v.window, v.violation] for v in verdicts if not v.accepted],
            "ingress_salt_frames": [salt_frames(r) for r in sorted(in_records, key=lambda r: r.window)],
            "egress_salt_frames": sum(salt_frames(r) for r in out_records),
            "egress_holds_salt": any(s in f for f in egress for s in salts.values()),
            "block_received": sorted(map(tuple, server.get("salts", []))) ==
                              [(w, fingerprint(salts[w])) for w in sorted(salts)]}


def select(traces: Sequence[Path], setup: Setup, windows: int, lo: int, hi: int) -> list[tuple[Path, str]]:
    """Configurations of lo … hi windows, one per trace file in order and then a second round, until `windows`."""
    pool = []
    for path in traces:
        doc = replay.read_trace(path)
        sizes = [(c["name"], replay.config_windows(c, setup.egress_params.buckets)) for c in doc["configurations"]]
        pool.append([(path, name, n) for name, n in sizes if lo <= n <= hi])
        del doc
    picked, total, round_ = [], 0, 0
    while total < windows and any(len(p) > round_ for p in pool):
        for p in pool:
            if len(p) > round_ and total < windows:
                path, name, n = p[round_]
                picked.append((path, name))
                total += n
        round_ += 1
    return picked


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _audit_job(args) -> dict:
    plan_path, report, run, strategy = args
    os.nice(19)
    with gzip.open(plan_path, "rt") as fh:
        plan = json.load(fh)
    result = audit_configuration(plan, report, Path(run))
    with gzip.open(Path(run) / "audit.json.gz", "wt") as fh:
        json.dump(result, fh)
    return {k: v for k, v in result.items() if k in ("config", "windows", "rows", "comparisons", "timing")} | \
        {"run": run, "strategy": strategy}


def summarize(results: Sequence[dict], models: Sequence[dict]) -> dict:
    rows = [r for res in results for r in res["rows"]]
    hist = defaultdict(Counter)
    real = slots = 0
    late = defaultdict(list)
    for res in results:
        with gzip.open(Path(res["run"]) / "audit.json.gz", "rt") as fh:
            full = json.load(fh)
        for k, v in full["hist"].items():
            hist[k].update({float(x): n for x, n in v.items()})
        hist["ingress_delivery"].update({float(x): n for x, n in full["ingress_delivery"].items()})
        real, slots = real + full["egress"]["real"], slots + full["egress"]["slots"]
        for k, v in full["lateness_ms"].items():
            late[k] += v
    out = {"configurations": len(results), "windows": sum(r["windows"] for r in results)}
    for side in ("egress", "ingress"):
        mine = [r for r in rows if r["direction"] == side]
        out[side] = {"link_windows": len(mine), "accepted": {k: sum(r[k] is None for r in mine)
                                                             for k in ("raw", "committed")},
                     "violations": [r for r in mine if r["raw"] or r["committed"]]}
    out["egress"]["padding_share"] = 1 - real / slots if slots else None
    out["latency_ms"] = {k: replay.hist_percentiles(v) for k, v in hist.items()}
    slack = results[0]["timing"]["slack_ms"] if results else None
    out["tick_late_ms"] = {k: percentiles_ms(v) | {"ticks": len(v), "over_slack": sum(x > slack for x in v)}
                           for k, v in late.items()}
    out["comparisons"] = {k: all(r["comparisons"][k] for r in results)
                          for k in ("records_are_run_shapers", "hands_are_the_rules", "frames_log_is_the_programs",
                                    "wire_is_the_records")}
    out["comparisons"] |= {k: max((r["comparisons"][k] for r in results if k in r["comparisons"]), default=None)
                           for k in ("hands_max_diff_from_rule", "hands_max_diff_from_model")}
    out["comparisons"] |= {k: sum(r["comparisons"][k] for r in results)
                           for k in ("ready_off_meant", "releases_off_model", "model_misses",
                                     "released_not_the_programs", "mismatches", "mismatches_from_late_frames")}
    out["comparisons"]["declared_are_planned"] = all(r["comparisons"].get("declared_are_planned", True)
                                                     for r in results)
    if any("hindsight" in r["comparisons"] for r in results):
        out["live"] = {
            "model_syncs_met_windows": sum(r["comparisons"]["model_syncs_met_windows"] for r in results),
            "hindsight_misses": sum(r["comparisons"]["hindsight"].get("misses", 0) for r in results),
            "hindsight_errors": [r["comparisons"]["hindsight"]["error"] for r in results
                                 if "error" in r["comparisons"]["hindsight"]],
            "server_requests": sum(r["timing"]["server_requests"] for r in results),
            "server_failed": sum(r["timing"]["server_failed"] for r in results),
            "server_cut_at_horizon": sum(r["timing"]["server_cut_at_horizon"] for r in results)}
    if models:
        mt, mrows = replay.new_tally(), []
        for m in models:
            mrows += m["rows"]
            replay.add_tally(mt, m["tally"])
        ms = replay.summarize(mrows, mt, Setup(models[0]["report"]))
        out["model"] = {"egress": {k: ms["egress"][k] for k in ("link_windows", "accepted", "padding_share",
                                                               "latency_ms")},
                        "ingress": {k: ms["ingress"][k] for k in ("link_windows", "accepted", "delivery_delay_ms")}}
    out["per_configuration"] = [{"config": r["config"], "windows": r["windows"], "comparisons": r["comparisons"],
                                 "tick_late_ms": r["timing"]["tick_late_ms"], "timing": r["timing"],
                                 "accepted": {k: sum(x[k] is None for x in r["rows"]) for k in ("raw", "committed")},
                                 "link_windows": len(r["rows"])} for r in results]
    return out


def against_honest(run: Path, honest: Path) -> dict:
    """Whether a run's wire, as the peer read it, and its committed records are the honest run's, and the windows whose
    sync the proxy declared differently from honest's (under an online rule; else none)."""
    def wire(p: Path) -> list[str]:
        return json.loads((p / "peer.json").read_text())["window_sha256"]

    def committed(p: Path) -> list[str]:
        return sorted((p / "committed.jsonl").read_text().splitlines())

    def declared(p: Path) -> list:
        with gzip.open(p / "proxy.json.gz", "rt") as fh:
            return json.load(fh).get("syncs") or []

    return {"wire_is_honests": wire(run) == wire(honest), "committed_is_honests": committed(run) == committed(honest),
            "windows_whose_wire_differs": [w for w, (a, b) in enumerate(zip(wire(run), wire(honest))) if a != b],
            "windows_whose_declared_differs": [w for w, (a, b) in enumerate(zip(declared(run), declared(honest)))
                                               if a != b]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="active_replay.py", description=__doc__.splitlines()[0])
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--windows", type=int, default=60, help="windows to replay, in whole configurations")
    ap.add_argument("--config-windows", default="4:10", help="the configurations' sizes, lo:hi windows")
    ap.add_argument("--config", dest="configs", action="append", help="a configuration's name, instead of --windows")
    ap.add_argument("--truncate", type=int, help="replay only each configuration's first N windows")
    ap.add_argument("--strategies", default="honest",
                    help=f"the server's write timing, comma-separated: {', '.join(STRATEGIES)}; each one is compared "
                         "with the honest run")
    ap.add_argument("--seed", type=int, default=0, help="the bits strategy's seed")
    ap.add_argument("--clock", choices=("real", "virtual"), default="real",
                    help="real: three processes on the host's clock; virtual: one event loop on an injected clock")
    ap.add_argument("--nice", type=int, help="run the three processes at this niceness, e.g. -20 (through sudo renice)")
    ap.add_argument("--cpus", type=lambda s: [int(c) for c in s.split(",")],
                    help="pin the server, the proxy and the peer each to one core, e.g. 1,2,3")
    ap.add_argument("--reaudit", action="store_true",
                    help="audit and summarize the runs already under --out, from their records, instead of running them")
    ap.add_argument("--server-url", help="a live OpenAI endpoint, e.g. http://127.0.0.1:8000, instead of the fake server")
    ap.add_argument("--vllm", metavar="MODEL", help="start a vLLM server for MODEL as record.py does, and use it live")
    ap.add_argument("--vllm-cpus", help="with --vllm: its CPUs (taskset -c); the harness takes the rest of its own")
    ap.add_argument("--served-model", default="served", help="the live endpoint's model name")
    ap.add_argument("--syncs", type=parse_sync_policy,
                    help=f"the prover's syncs: {', '.join(SYNC_POLICIES)} (default: envelope live, else model)")
    ap.add_argument("--prover-lag", type=int, default=1, metavar="BUCKETS",
                    help="under an online rule: window w's sync is the rule's on the frames ready by w·T − BUCKETS, "
                         "chosen off the proxy's tick, 1 … T − 1 (the envelope if not ready at w·T)")
    ap.add_argument("--stall", dest="stalls", action="append", default=[], metavar="BUCKET:MS",
                    type=lambda s: (int(s.split(":")[0]), float(s.split(":")[1]) / 1000),
                    help="test-only: stop the proxy for MS right after the first ingress tick at or after BUCKET that "
                         "releases a request, in every configuration (`Stalls`)")
    ap.add_argument("--max-minutes", type=float,
                    help="start no configuration that would end later than this from the start")
    args = ap.parse_args(argv)
    started = time.monotonic()
    strategies = args.strategies.split(",")
    if strategies[0] != "honest" or any(s not in STRATEGIES for s in strategies):
        raise SystemExit(f"--strategies starts with honest and names some of {STRATEGIES}")
    live = bool(args.server_url or args.vllm)
    if live and args.clock != "real":
        raise SystemExit("a live server runs on the real clock")
    sync_policy = args.syncs or ("envelope" if live else "model")
    report = json.loads(args.calibration.read_text())
    setup = Setup(report)
    if sync_policy not in ("model", "envelope") and not 0 < args.prover_lag < setup.egress_params.buckets:
        raise SystemExit(f"--prover-lag is 1 … {setup.egress_params.buckets - 1} buckets under an online rule")
    lo, hi = (int(x) for x in args.config_windows.split(":"))
    traces = sorted(args.traces)
    if args.configs:
        wanted = set(args.configs)
        picked = [(p, n) for p in traces for n in [c["name"] for c in replay.read_trace(p)["configurations"]]
                  if n in wanted]
    else:
        picked = select(traces, setup, args.windows, lo, hi)
    runs = args.out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if not args.reaudit:
        (args.out / "selection.json").write_text(json.dumps([[str(p), n] for p, n in picked], indent=1))
    def evicted(*_) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise Evicted

    signal.signal(signal.SIGTERM, evicted)
    vllm: list = []
    try:
        upstream = None
        if live and not args.reaudit:
            upstream = start_vllm(args, args.out, vllm) if args.vllm else _host_port(args.server_url)
        return _main(args, report, setup, picked, runs, strategies, live, sync_policy, upstream, started)
    except Evicted:
        print("evicted before the first configuration", flush=True)
        return 143
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if vllm:
            import record

            record.stop_server(vllm[0])


def _host_port(url: str) -> tuple[str, int]:
    m = re.fullmatch(r"https?://([^:/]+):(\d+)/?", url)
    if not m:
        raise SystemExit(f"--server-url is http://HOST:PORT, not {url!r}")
    return m.group(1), int(m.group(2))


def start_vllm(args, out: Path, started: list) -> tuple[str, int]:
    """vLLM's OpenAI server on this host's GPU, with `record.py`'s arguments, once it answers /health; the process is
    appended to `started` as soon as it exists."""
    import record

    port = record.free_port()
    model = record.local_model(args.vllm, Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
            "--served-model-name", args.served_model, "--host", "127.0.0.1", "--port", str(port), "--seed", "0",
            "--max-model-len", "2048", "--disable-uvicorn-access-log"]
    if args.vllm_cpus:
        argv = ["taskset", "-c", args.vllm_cpus, *argv]
        mine = os.sched_getaffinity(0) - set(os.sched_getaffinity(0) & _cpu_list(args.vllm_cpus))
        if mine:
            os.sched_setaffinity(0, mine)
    (out / "vllm.json").write_text(json.dumps({"argv": argv, "harness_cpus": sorted(os.sched_getaffinity(0))}))
    with open(out / "vllm.log", "wb") as log:
        started.append(server := subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT))
    t0 = time.monotonic()
    record.wait_ready(server, port, t0 + 900)
    print(f"vLLM ready in {time.monotonic() - t0:.0f} s on port {port}", flush=True)
    return "127.0.0.1", port


def _cpu_list(text: str) -> set[int]:
    out = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def _main(args, report, setup, picked, runs, strategies, live, sync_policy, upstream, started) -> int:
    ctx = multiprocessing.get_context("spawn")
    audits, models, processes = [], [], []
    stopped = None
    with ctx.Pool(1, initializer=signal.signal, initargs=(signal.SIGTERM, signal.SIG_IGN),
                  maxtasksperchild=1) as pool:
        for path, name in picked:
            if stopped:
                break
            doc = replay.read_trace(path)
            cfg = next(c for c in doc["configurations"] if c["name"] == name)
            del doc
            if args.truncate:
                cfg = truncate(cfg, args.truncate, setup.egress_params)
            plan = plan_configuration(cfg, setup, args.served_model if live else None, sync_policy, args.prover_lag)
            span_s = plan["windows"] * setup.egress_params.buckets * setup.egress_params.bucket_ms / 1000 + STARTUP_S
            if (not args.reaudit and args.max_minutes
                    and time.monotonic() - started + span_s + 10 > 60 * args.max_minutes):
                stopped = f"{name}: would end after --max-minutes"
                print(stopped, flush=True)
                break
            mrows, mtally, _ = replay.replay_configuration(cfg, replay.Setup(report), plan.get("sync_rule", "hindsight"),
                                                           lag=plan.get("prover_lag", 0))
            del cfg
            for strategy in strategies:
                run = runs / slug(name) / strategy
                if args.reaudit and not (run / "processes.json").exists():
                    print(f"{name} {strategy}: not run", flush=True)
                    break
                if not args.reaudit:
                    run.mkdir(parents=True, exist_ok=True)
                    mine = plan if strategy == "honest" else adversary_plan(plan, setup, strategy, args.seed)
                    with gzip.open(run / "plan.json.gz", "wt") as fh:
                        json.dump(mine, fh)
                    t0 = time.monotonic()
                    if args.clock == "virtual":
                        job = ctx.Process(target=virtual_process, args=(mine, report, run, args.stalls))
                        job.start()
                        job.join()
                        if job.exitcode:
                            return 1
                    else:
                        try:
                            run_configuration(mine, report, run, args.nice, upstream, args.cpus, args.stalls)
                        except Evicted:
                            stopped = f"evicted in {name}"
                            print(stopped, flush=True)
                            break
                    print(f"{name} {strategy}: {plan['windows']} windows in {time.monotonic() - t0:.0f} s", flush=True)
                processes.append(json.loads((run / "processes.json").read_text()))
                if any(processes[-1]["exit"].values()):
                    print(f"{name} {strategy}: exit {processes[-1]['exit']}", flush=True)
                    return 1
                audits.append(pool.apply_async(_audit_job, ((run / "plan.json.gz", report, str(run), strategy),)))
            else:
                models.append({"rows": mrows, "tally": mtally, "report": report})
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        results = []
        for a in audits:
            res = a.get()
            acc = {k: sum(r[k] is None for r in res["rows"]) for k in ("raw", "committed")}
            if res["strategy"] != "honest":
                res["against_honest"] = against_honest(Path(res["run"]), Path(res["run"]).parent / "honest")
            print(f"{res['config']} {res['strategy']}: accepted {acc} of {len(res['rows'])}; {res['comparisons']}; "
                  f"{res.get('against_honest', '')}", flush=True)
            results.append(res)
        pool.close()  # the workers ignore SIGTERM, so the `with`'s terminate would wait on a fresh one forever
        pool.join()
    honest = [r for r in results if r["strategy"] == "honest"]
    summary = summarize(honest, models) | {"calibration": str(args.calibration),
                                           "clock": sorted({p["clock"] for p in processes}),
                                           "nice": sorted({p.get("nice") for p in processes}, key=str),
                                           "cpus": sorted({str(p.get("cpus")) for p in processes}),
                                           "selection": [[str(p), n] for p, n in picked],
                                           "server": "live" if live else "fake", "sync_policy": sync_policy,
                                           "prover_lag": args.prover_lag,
                                           "stopped": stopped}
    summary["adversarial"] = {}
    for s in strategies[1:]:
        mine = [r for r in results if r["strategy"] == s]
        summary["adversarial"][s] = {
            "link_windows": sum(len(r["rows"]) for r in mine),
            "accepted": {k: sum(x[k] is None for r in mine for x in r["rows"]) for k in ("raw", "committed")},
            "violations": [x for r in mine for x in r["rows"] if x["raw"] or x["committed"]],
            "wire_is_honests": [r["against_honest"]["wire_is_honests"] for r in mine],
            "committed_is_honests": [r["against_honest"]["committed_is_honests"] for r in mine],
            "windows_whose_wire_differs": [r["against_honest"]["windows_whose_wire_differs"] for r in mine],
            "windows_whose_declared_differs": [r["against_honest"]["windows_whose_declared_differs"] for r in mine],
            "records_are_run_shapers": all(r["comparisons"]["records_are_run_shapers"] for r in mine),
            "ready_off_honest": sum(r["comparisons"]["ready_off_meant"] for r in mine),
            "tick_late_ms": [r["timing"]["tick_late_ms"] for r in mine]}
    summary["covert"] = {}
    for s in strategies:
        mine = [r["comparisons"] for r in results if r["strategy"] == s]
        if mine:
            pairs = [(c["covert"]["symbols"], c["covert"]["declared"]) for c in mine]
            summary["covert"][s] = covert.measure_runs(pairs, setup.sync_set, margin=mine[0]["covert"]["margin"]) | \
                {"margin": mine[0]["covert"]["margin"], "runs": pairs}
            if "covert_declared_are_predicted" in mine[0]:
                summary["covert"][s]["declared_are_predicted"] = all(c["covert_declared_are_predicted"] for c in mine)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(args.out / "verdicts.jsonl", "w") as fh:
        for res in results:
            fh.writelines(json.dumps(r | {"strategy": res["strategy"]}) + "\n" for r in res["rows"])
    e, i = summary["egress"], summary["ingress"]
    print(f"{summary['configurations']} configurations, {summary['windows']} windows; egress {e['accepted']} of "
          f"{e['link_windows']}; ingress {i['accepted']} of {i['link_windows']}; {summary['comparisons']}")
    for s, a in summary["adversarial"].items():
        print(f"{s}: accepted {a['accepted']} of {a['link_windows']}; wire is honest's {a['wire_is_honests']}; "
              f"committed is honest's {a['committed_is_honests']}")
    for s, m in summary["covert"].items():
        print(f"{s} through the declared sync: MI {m['mutual_information_full_bits']:.3f} bits/window (null "
              f"{m['mutual_information_null_bits']:.3f}), decoded {m['decoded_bits_per_window']:.3f} bits/window, "
              f"symbol error rate {m['symbol_error_rate']:.3f}, over {m['window_decisions']} window-decisions")
    return 143 if stopped and stopped.startswith("evicted") else 0


def truncate(cfg: dict, windows: int, params: GridParams) -> dict:
    """The configuration's first `windows` windows: its frames anchored and its arrivals in them."""
    lo, _ = cfg["labels"]["windows"]
    end = windows * params.buckets
    out = dict(cfg)
    out["egress"] = [{"link": l["link"], "frames": [f for f in l["frames"] if f["anchor"] < end]} for l in cfg["egress"]]
    out["ingress"] = [{"link": l["link"], "arrival_ms": [t for t in l["arrival_ms"] if t < end * params.bucket_ms]}
                      for l in cfg["ingress"]]
    out["labels"] = cfg["labels"] | {"windows": [lo, lo + windows]}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
