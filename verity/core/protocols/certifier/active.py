"""An active certifier on a live byte stream: an asyncio TCP proxy that runs the reference's certifiers on one boundary link.

The proxy is the I/O shell around the reference. It owns a monotonic clock and a ticker per grid, socket reads and
writes, and the framing of byte streams into B-byte frames. Everything that decides a grid is the reference's:
`schedule.eligible` and `schedule.run_shaper`'s handing rule, and `grid.Certifier` on egress (at r) and on ingress (at r_in).
Its records are `grid.Record`s, published raw and through `commitment.commit`, which `audit.Audit.check_link` and
`check_ingress_link` read.

**Topology.** One TCP connection is the boundary link, the wire, between the proxy and the outside peer. Inside, the
proxy opens one TCP connection per stream to the server, and writes it the stream's request as the ingress certifier
releases it. Every frame on the link is B bytes: a header (kind u8, stream u32, length u32), the payload, and zero
fill. Padding is B zero bytes (kind `PAD`); `DATA` is a payload.

**The canonical framing** is every sender's, so that each direction's frames are one public template's: the same
header on every frame, then the sender's bytes at a fixed place, then the same bytes after them, so nothing else of a
frame is free. On egress a frame is one event (`event`): the
`DATA` header on stream `CANONICAL_STREAM` with length `EVENT_BYTES`, `EVENT_HEAD`, the base64 of at most
`TOKEN_BYTES` bytes padded with NUL to `EVENT_TEXT` characters, and `EVENT_TAIL`, so a token is one frame, or one per
`TOKEN_BYTES` of it (`events`). An id event (`id_event`, `ID_EVENT_BYTES`) is that object with no text: its request's
ingress ordinal and one sampled token's id, each in `ID_DIGITS` decimal characters, so a response is one event a token
and its frames name their request whatever order the responses interleave in. A compact record (`compact_record`,
`COMPACT_RECORD`, 8 bytes) carries the same two fields in binary, with the response's finish in the token field's
flags (`COMPACT_LAST`, `COMPACT_STOP`), and its frame is the record zero-filled to B with no header (`compact_cell`,
`compact_placement`), so B can be 8 (`Proxy(compact=True)`). On ingress a request is its length
(u32be) and its bytes, cut into `DATA` frames of B − 9 bytes on `CANONICAL_STREAM`, the last zero-filled
(`request_frames`): it ends where its length says, so every frame's header is the same. An id request (`id_request`) is
a token-id prompt's `/v1/completions` request whose every field is at a place its prompt's length fixes, and its frames
(`id_request_frames`) a function of that length alone. `event_placement` and `request_placement` are those templates'
places.

**Ticks.** Each grid's ticker sleeps once a tick, to the tick's boundary (`origin` + t·τ). The tickers that wake
together then run their ticks in one step of the event loop, in `TICK_ORDER`, ingress first (`_run_due`): each reads the
clock as its turn comes, which is when it reads its link, and that reading is the tick's lateness. One loop serves both
grids, the link and every server connection, so whatever runs between a boundary and a tick's turn makes the tick late.
An egress tick hands the frames read and writes a whole bucket to the link, and ingress ticks run after it at shared
boundaries read their link past the slack on a loaded host and failed honest windows closed; an ingress tick releases a
bucket's few request frames, so the egress tick after it is late by little, more only at a window's last bucket, where
the ingress tick publishes its record. Two tickers wake together when their
boundaries meet (both timers expire in one turn of the loop, or one `VirtualClock.advance` passes both); the device
difftest's harness wakes one ticker a tick event, so there each step runs that tick alone, as `CertifierDevice.step`
takes one event, and the trace orders a shared boundary's two ticks either way.

**The pacer** (`Proxy(paced=…)`, off by default): a bucket written on the loop leaves when its tick runs, so when the
server writes within a bucket, which the prover chooses, moves when the bucket reaches the client. With `paced`, a
number of seconds past `slack`, the egress tick hands its bucket to a process of its own (`pacer.Pacer`) instead, which
writes it to the link at `origin` + t·τ + `paced` on the host's monotonic clock, a time fixed before the bucket is made,
so the loop's lateness within the slack never moves it. A bucket that reached the pacer after its time, or whose release
started more than `slack` after it, faults its window `monitor-unavailable`, as a late tick does, and one the link
refused, `delivery-refused`. A client slow to read holds bytes in the pacer, never a later bucket's release: a tick
that finds more than one bucket past its time and not yet written faults its window `delivery-refused`, as the loop's
own backlog of more than a bucket does unpaced. A window's record is published once its last bucket's writes are
reported, so a fault on that bucket is its window's.

**Egress.** The server's bytes on a stream are cut after every `\\n\\n`, the end of a server-sent event, so a frame's
content is fixed by the bytes and never by how they were read. The event's `id:` field is its position in the
Program's canonical order on the link, and the frame is what follows that line (`event_body`): neither the position
nor an HTTP response head before it is recorded. Under `compact` the server's bytes are an HTTP response head, then
units of `COMPACT_UNIT_BYTES` (`CompactUnits`): the position (u32be) and a compact record of the connection's own
request, and the frame is the record (`compact_cell`). A frame's anchor is the egress bucket in which the ingress certifier
released its stream's request's last frame (the request is recorded there), and its logical index
is its position in its stream. A frame read at time s is ready at bucket ⌊s/τ_b⌋ + 1, the first that starts after it.
At bucket t the shaper hands, in canonical order, each frame whose `max(h(k), ready(k), last)` is at most t, where
`h(k) = max(eligible(k), h(k−1))` under the declared sync of its anchor's window (`run_shaper`'s rule). `Certifier.tick`
then releases min(queue, r) of them, left-packed, and the proxy writes those and r − n padding frames. A frame handed at
a bucket after h(k) is late: the declared sync didn't cover it, and the window it is handed in fails closed.

**Ingress.** A frame read from the link at time s is enqueued at bucket ⌈s/τ_in⌉ (`calibration.size_ingress_ms`'s
model) and released by the ingress `Certifier` at r_in. The released `DATA` frames are cut into requests by their
lengths (`Requests`): the k-th request on the link is stream k, and its bytes, without the length or the fill, go to
its stream's server connection.

**The salt** (`Proxy(salt=…)`, off by default) is `salt_bits` bits (512 by default) per PoUW window from the
certifier's own randomness (`fresh_salt`), released into the block only. PoUW window p is the `pouw_window` = k
certifier windows p·k … p·k + k − 1 (k = 1 by default). At p's first ingress tick, t_s, the certifier draws p's salt
and enqueues its own salt frame at the head of the queue, ahead of that bucket's arrivals and of any queue carried
from the window before (`Bucketer(first=…)`), so it leaves in that bucket's slot 0: kind `SALT` on `SALT_STREAM`,
its payload u64be p followed by `PLACEHOLDER`, 64 zero bytes. The frame and its slot are fixed by the window and the
traffic, so every record is a function of the traffic alone and none depends on the salt. When the ingress certifier
releases that frame (the `Bucketer` knows it by its place in the queue, so a link frame of the same bytes is never taken
for it), the proxy writes the salt frame with the salt in place of the placeholder (the header, p and the salt, no fill)
on a server connection of its own, which it never reads: nothing the server writes there reaches egress. The egress
side never sees the salt.

The certifier's record cells are its grids and two sealed slots per PoUW window: `_salts[p]`, which holds the salt and
is opened only by `reveal`, and `_weights[p]`, the 64-byte root of the weights PoUW's noise seed reads. The prover
hands that root once (`commit_weights(p, root, signature)`), signed under the prover key registered with the
certifier (`prover_key`, over `weights_message`), while p's first ingress tick, the salt's draw, is still ahead;
`committed(p, registration)` opens it on `reveal`'s conditions. So the certifier's word orders the prover's weights
before the salt, and no one else can fix them. The record's open cells (`Accounting.Attribution.openCells`) are the
grids, a function of the traffic, and what the certifier sends into the block is a function of its record.
`reveal(p, registration)` takes the PoUW window's registration as the proof service returned it
(`sampled_proofs.service.Registration`) and fails closed without it: it refuses unless the receipt is that
registration's (`one_stage.registration.receipt`) and `registered` reads p off the registration, while p's last
ingress record is unpublished (t_d), once the certifier's clock is more than `reveal_deadline` past p's end (one PoUW
window by default), for a window whose salt was never drawn, and for one it has revealed already. So the salt reaches
one registration of p, made within the deadline after p's work. `check_salts` is the audit's read of it: p's first raw
ingress record holds exactly one salt frame of p, the placeholder's, and the revealed salt has `salt_bits` bits.

**The reveal over the network** (`Proxy(challenger_key=…, signer=…)`, both or neither). Deployed, the certifier runs on
the link, off the prover's node, and the auditor's challenger on an auditor host. With the keys, `reveal` and
`committed` take the challenger's `request`: its LMS signature under `challenger_key`, the key the auditor's
deployment record (never the prover's configuration) registered for it, over `reveal_request_message(signer.key_id,
run_id, ingress link, p, registration.digest, SALT_KIND or WEIGHTS_KIND)`, where `run_id` is the 32 bytes the proxy drew
at construction (`Proxy.run_id`). They refuse one that doesn't verify before reading anything else, so no one else
spends p's one reveal, nor does a request for another certifier or another run, and a registration counts only if the
challenger signed its digest, which binds its record. They answer a `Revealed`: the value, the certifier's key id and
run id, and its own signature (`signer`, the network certifier's key) over `reveal_response_message`, which binds the
value too, so a consumer can key p on the certifier run, (key id, run id, link) (`Proxy.certifier_run`, the same on a
`RemoteCertifier`), where two proxies on one link name don't collide. The
salt needs authenticity, not confidentiality: after t_d the block's server already holds it, and the verifier fills
`pouw.salt` as a public input, so nothing is encrypted. `serve_reveals(proxy, host, port)` serves both calls beside
`run`, one JSON line per request and per answer, and `RemoteCertifier` is the challenger's client, which takes the
certifier's key and run id from the deployment record and returns the value it verified, as a keyless proxy does in
process. A refusal is unsigned: forged on the path, it costs the window as dropping the answer would. Given
`certificates`, the endpoint also serves each certifier window's certificates' public part to any caller, unsigned: each
signs a salted digest of its record's head, a hiding commitment, and verifies on its own signature. What the keys
leave on `trusted-host` is the certifier's own (its signing key and leaf state on its host, and its word on what
it signs) and the deployment record's custody (the keys and the run id it gives each side). Without the keys,
`reveal(p, registration)` and `committed(p, registration)` are the in-process calls the tests and replays make, and
hold only on the auditor's side.

**Sealed pair links** (`Proxy(pair=…)`, off by default; PROTOCOL.md, "Sealed pair links"): when the link's peer is
another of the auditor's certifiers, the frames between them are sealed under `verity.ml.crypto.aead`, keys of the
pair's session secret per direction (`Pair.of`). The egress tick seals each of its bucket's r slots, frames and padding
alike, after the egress certifier records it (`Sealer`, slot j of bucket t at nonce (t // T, t mod T, j)), and the
link carries r units of B + 64 bytes a bucket. The ingress side opens the k-th unit it reads as slot k of the peer's
egress grid (`Opener`) before the ingress certifier takes it. So every record, and everything the audit reads, is the
unsealed run's; only the link's bytes change. The first unit that doesn't open is refused, and so is every one after
it: its ingress window and every later one fault `malformed`.

**Back from maintenance** (`Proxy(start=k, position=p)`, off by default): a certifier that was down serves from window k
of the trusted clock on. Its first tick of each grid is the window's first bucket, its first records are window k's, and
the first egress frame it takes is the Program's at position p, the frames before it being those of the windows before.
It carries nothing from before: a window it was down for has no record. The windows before k are another proxy's, the
same certifier's before it went down (`windows`, which ends a proxy's run early), or none.

**Advice.** The proxy runs the windows of its declared syncs, one per window, which the prover gives before the run,
or which an online prover (`online.Prover`, with a lag of 1 … T − 1 buckets) declares at each window's first tick. A
frame anchored past them is never handed, nor is any frame behind it (`schedule.handing_buckets`). The online
choice for window w costs hundreds of ms on a live configuration, against a slack of tens, so it is never made on a
tick: right after tick w·T − lag the proxy hands what it reads (`Prover.snapshot`, the frames ready by then and the
earlier declarations) to a worker process, off its event loop and its GIL, and the tick at w·T declares the result
if it is done, else the envelope (`LaggedChoices`). Window 0's is started before the first tick. The envelope is in
Σ_sync, so an unready choice changes nothing the audit checks or charges: a declaration is advice, checked only for
membership in Σ_sync, and the bound charges log2 #Σ_sync bits per window whatever the value.

**Faults fail closed** through `Certifier.fault`: the rest of the window is padding, the queue is dropped, and the status
names the first reason (W6). Since the Program's grid carries a queue into the next window, an egress window that
closes faulted while its queue would still hold frames faults the next one too, with the same reason. They are:
- a late tick (`monitor-unavailable`): its turn ("Ticks") came more than `slack` after its boundary. A bucket it missed
  is still emitted, as padding, so every window keeps its T rows and no bucket slips into the next;
- a declaration the online prover refuses (`monitor-unavailable`, `online.Refused`): a frame read for the window's
  choice, at tick w·T − lag, is ready no later than its anchor's bucket, which an ingress tick read before its boundary
  makes. The egress window that declares is faulted, and its sync is the choice made once those frames are read;
- an overrun: the link still holds a whole bucket unwritten at a tick (`delivery-refused`), or the queue or the
  shaper's buffer passes Q (`overflow`); a server that refuses a stream's connection is `delivery-refused` on ingress;
- a late forward (`delivery-refused`): a released request not written to its server connection by `slack` after its
  release tick, nor by the end of its anchor's egress bucket. The server answers it late, so its events, and those
  the server numbers after them, aren't the Program's. Its ingress window is faulted if still open, and every egress
  window from the check, at an egress tick before handing, until the stream has ended upstream and every frame that
  arrived before its end is handed;
- an advice violation: a window's declared sync outside Σ_sync (`coverage-never-established`), checked at the window's
  start; the grid still follows the declared sync, as the audit's expected grid does;
- a late frame (`coverage-never-established`): a frame handed at a bucket t after its h(k), because its bytes were
  ready after h(k), a frame before it was handed late (`last`), or it was read after bucket h(k)'s tick. The sync
  declared for its anchor's window didn't cover the response, so the Program's grid holds the frame earlier than the
  certifier could: the window of bucket t is faulted, and the frame is dropped with the queue. A window closed before
  the frame was handed can't be faulted any more; its record is the Program's on the frames handed by its close;
- a malformed frame (`malformed`): an event whose body is longer than a frame, without a position or with one already
  seen, on a stream with no released request, or bytes left unframed at the stream's end; under `compact`, a unit
  whose record isn't one or names another request than its connection's; on a sealed pair link, a
  unit that doesn't open at its slot's nonce, and every window after it.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import heapq
import itertools
import json
import math
import multiprocessing
import os
import resource
import secrets
import socket
import struct
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field, replace

from verity.ml.commitments.identity import tagged_sha512
from verity.ml.commitments.scheme import DIGEST_BYTES
from verity.ml.crypto import aead
from verity.core.protocols.certifier.audit import Audit, Verdict
from verity.core.protocols.certifier.certificate import PublicKey, Signer, SignerRefused
from verity.core.protocols.certifier.grid import (
    GridParams,
    Reason,
    Record,
    Row,
    Certifier,
    Status,
    row_count,
)
from verity.core.protocols.certifier.online import Prover, Refused
from verity.core.protocols.certifier.pacer import Pacer, Report
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, eligible
from verity.core.protocols.certifier.served import Admission, ServedProfile, admits
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.service import Registration

HEADER = struct.Struct(">BII")
PAD, DATA, SALT = 0, 1, 3
#: the kind of the header that opens a stream's server connection under `Proxy(ordinals=True)`; never on a link
ORDINAL = 4
EVENT_END = b"\n\n"
CANONICAL_STREAM = 0
TOKEN_BYTES = 64
EVENT_TEXT = 4 * -(-TOKEN_BYTES // 3)
EVENT_HEAD = (b'data: {"object":"chat.completion.chunk","created":0,"model":"verity","choices":[{"index":0,'
              b'"delta":{"base64":"')
EVENT_TAIL = b'"},"finish_reason":null}]}' + EVENT_END
EVENT_BYTES = len(EVENT_HEAD) + EVENT_TEXT + len(EVENT_TAIL)
LENGTH = struct.Struct(">I")
SALT_STREAM = 0xFFFF_FFFF
SALT_BITS = 512
SALT_BYTES = SALT_BITS // 8
PLACEHOLDER = bytes(64)
SALT_FRAME_BYTES = HEADER.size + 8 + len(PLACEHOLDER)
WEIGHTS_ROOT_BYTES = DIGEST_BYTES["sha512"]  # a frame-v3-sha512 root
WEIGHTS_TAG = "verity-network-certifier/weights/v1"


def weights_message(link: str, window: int, root: bytes, context: bytes = b"") -> bytes:
    """What the prover signs to hand the certifier of ingress link `link` PoUW window `window`'s weights' root. `context`
    is the deployment's binding of the run (an epoch's stream, say), so a message signed for one run is never taken
    in another."""
    name = link.encode()
    return tagged_sha512(WEIGHTS_TAG, struct.pack(">H", len(name)) + name + struct.pack(">Q", window) + root +
                         struct.pack(">I", len(context)) + context)


REVEAL_REQUEST_TAG = "verity-network-certifier/reveal-request/v1"
REVEAL_RESPONSE_TAG = "verity-network-certifier/reveal-response/v1"
SALT_KIND, WEIGHTS_KIND = b"salt", b"weights"
#: a proxy's run id, drawn at its construction (`Proxy.run_id`)
RUN_ID_BYTES = 32
#: the longest request or answer line `serve_reveals` and `RemoteCertifier` read: a deployed LMS signature is 34,756
#: bytes, twice that in hex, beside a registration
REVEAL_LINE_BYTES = 1 << 20
#: the longest a ticker waits at once. Linux lets an epoll wait end late by 0.1 % of its timeout, 0.5 % in a niced
#: process, up to 100 ms (`select_estimate_accuracy`), so one wait for an origin 10 s off woke the first tick 48 ms
#: late at nice 19, past the slack, and faulted window 0. A wait of at most this ends at most 0.5 ms late
MAX_WAIT_S = 0.1
#: the order in which the ticks due at one instant run (the module docstring's "Ticks"): an ingress tick releases a
#: bucket's few request frames, an egress tick hands what was read and writes a whole bucket to the link
TICK_ORDER = ("ingress", "egress")


def _reveal_fields(key_id: bytes, run_id: bytes, link: str, window: int, digest: str, what: bytes) -> bytes:
    if what not in (SALT_KIND, WEIGHTS_KIND):
        raise ValueError(f"a reveal is of {SALT_KIND!r} or {WEIGHTS_KIND!r}, not {what!r}")
    if not (isinstance(key_id, bytes) and len(key_id) < 1 << 16):
        raise ValueError("a certifier's key id is bytes, under 64 KiB")
    if not (isinstance(run_id, bytes) and len(run_id) == RUN_ID_BYTES):
        raise ValueError(f"a certifier's run id is {RUN_ID_BYTES} bytes")
    if not (type(window) is int and 0 <= window < 1 << 64):
        raise ValueError(f"a PoUW window is a u64, not {window!r}")
    if not isinstance(digest, str):
        raise ValueError(f"a registration's digest is a string, not {type(digest).__name__}")
    name, d = link.encode(), digest.encode()
    if len(name) >= 1 << 16:
        raise ValueError("a link's name is under 64 KiB")
    return struct.pack(">H", len(key_id)) + key_id + run_id + struct.pack(">H", len(name)) + name + \
        struct.pack(">Q", window) + struct.pack(">I", len(d)) + d + struct.pack(">B", len(what)) + what


def reveal_request_message(key_id: bytes, run_id: bytes, link: str, window: int, digest: str, what: bytes) -> bytes:
    """What the auditor's challenger signs to ask the certifier whose key id is `key_id`, in its run `run_id`
    (`Proxy.run_id`), on ingress link `link`, for PoUW window `window`'s salt (`what` = `SALT_KIND`) or weights' root
    (`WEIGHTS_KIND`) on the registration whose digest is `digest`: the tagged SHA-512 of u16be len(key_id) ‖ key_id ‖
    run_id (`RUN_ID_BYTES`) ‖ u16be len(link) ‖ link ‖ u64be window ‖ u32be len(digest) ‖ digest ‖ u8 len(what) ‖
    what."""
    return tagged_sha512(REVEAL_REQUEST_TAG, _reveal_fields(key_id, run_id, link, window, digest, what))


def reveal_response_message(key_id: bytes, run_id: bytes, link: str, window: int, digest: str, what: bytes,
                            value: bytes) -> bytes:
    """What the certifier signs to answer that request with `value`: the request's fields, then u32be len(value) ‖ value,
    under the response's own tag."""
    if not isinstance(value, bytes):
        raise ValueError("a revealed value is bytes")
    return tagged_sha512(REVEAL_RESPONSE_TAG, _reveal_fields(key_id, run_id, link, window, digest, what) +
                         struct.pack(">I", len(value)) + value)


@dataclass(frozen=True, slots=True)
class Revealed:
    """The certifier's signed answer to a reveal request: PoUW window `window`'s salt or weights' root, `value`, its
    `signature` over `reveal_response_message`, and the certifier's `key_id` and `run_id` that it binds. With the link
    it verifies for, they are the certifier run (`Proxy.certifier_run`, `RemoteCertifier.certifier_run`). Keyed on the
    certifier run and the window, two proxies on one link name, or two runs of one, don't collide."""

    window: int
    value: bytes
    signature: bytes
    key_id: bytes
    run_id: bytes

    def verify(self, key: PublicKey, run_id: bytes, link: str, digest: str, what: bytes) -> bool:
        """Whether `key` signed this in run `run_id` as ingress link `link`'s certifier's answer to a request for `what`
        on the registration whose digest is `digest`: it names that key's id and that run, and the signature
        verifies."""
        if self.key_id != key.key_id or self.run_id != run_id:
            return False
        try:
            message = reveal_response_message(self.key_id, self.run_id, link, self.window, digest, what, self.value)
        except ValueError:
            return False
        return isinstance(self.signature, bytes) and key.verify(message, self.signature)


def encode_frame(kind: int, stream: int, payload: bytes, frame_bytes: int) -> bytes:
    """One B-byte link frame: the header, the payload, then zero fill."""
    if HEADER.size + len(payload) > frame_bytes:
        raise ValueError(f"a {len(payload)}-byte payload doesn't fit a {frame_bytes}-byte frame")
    return HEADER.pack(kind, stream, len(payload)) + payload + bytes(frame_bytes - HEADER.size - len(payload))


def decode_frame(frame: bytes) -> tuple[int, int, bytes]:
    """(kind, stream, payload) of a link frame; padding is (`PAD`, 0, b"")."""
    kind, stream, n = HEADER.unpack_from(frame)
    return kind, stream, bytes(frame[HEADER.size : HEADER.size + n])


def event(data: bytes) -> bytes:
    """The canonical event of at most `TOKEN_BYTES` bytes: `EVENT_HEAD`, their base64 padded with NUL to `EVENT_TEXT`
    characters, `EVENT_TAIL`; `EVENT_BYTES` long whatever the data."""
    if len(data) > TOKEN_BYTES:
        raise ValueError(f"an event carries at most {TOKEN_BYTES} bytes, not {len(data)}")
    return EVENT_HEAD + base64.b64encode(data).ljust(EVENT_TEXT, b"\0") + EVENT_TAIL


def events(token: bytes) -> list[bytes]:
    """A token's events: its bytes cut every `TOKEN_BYTES`, one event each, and one event for an empty token."""
    return [event(token[i : i + TOKEN_BYTES]) for i in range(0, max(1, len(token)), TOKEN_BYTES)]


def event_data(body: bytes) -> bytes | None:
    """The bytes a canonical event carries, or None for bytes that aren't one."""
    text = body[len(EVENT_HEAD) : len(EVENT_HEAD) + EVENT_TEXT]
    if len(body) != EVENT_BYTES or not body.startswith(EVENT_HEAD) or not body.endswith(EVENT_TAIL):
        return None
    try:
        data = base64.b64decode(text.rstrip(b"\0"), validate=True)
    except binascii.Error:
        return None
    return data if len(data) <= TOKEN_BYTES and event(data) == body else None


ID_DIGITS = 10
ID_EVENT_HEAD = EVENT_HEAD[: EVENT_HEAD.index(b'"delta"')] + b'"request":'
ID_EVENT_MID = b',"token_id":'
ID_EVENT_TAIL = b',"finish_reason":null}]}' + EVENT_END
ID_EVENT_BYTES = len(ID_EVENT_HEAD) + ID_DIGITS + len(ID_EVENT_MID) + ID_DIGITS + len(ID_EVENT_TAIL)


def _u32_field(x: int, what: str) -> bytes:
    if type(x) is not int or not 0 <= x < 1 << 32:
        raise ValueError(f"{what} is a u32, not {x!r}")
    return b"%*d" % (ID_DIGITS, x)


def id_event(request: int, token_id: int) -> bytes:
    """The canonical event of one sampled token: `ID_EVENT_HEAD`, its request's ingress ordinal (the stream the
    request is on the link), `ID_EVENT_MID`, the token's id, each right-aligned in `ID_DIGITS` decimal characters,
    then `ID_EVENT_TAIL`; `ID_EVENT_BYTES` long whatever the two, and nothing else in it is free."""
    return (ID_EVENT_HEAD + _u32_field(request, "a request's ordinal") + ID_EVENT_MID +
            _u32_field(token_id, "a token id") + ID_EVENT_TAIL)


def id_event_data(body: bytes) -> tuple[int, int] | None:
    """(request ordinal, token id) of an id event (`id_event`), or None for bytes that aren't one."""
    a = len(ID_EVENT_HEAD)
    b = a + ID_DIGITS + len(ID_EVENT_MID)
    fields = body[a : a + ID_DIGITS], body[b : b + ID_DIGITS]
    if len(body) != ID_EVENT_BYTES or not all(f.lstrip(b" ").isdigit() for f in fields):
        return None
    request, token_id = (int(f) for f in fields)
    try:
        return (request, token_id) if id_event(request, token_id) == body else None
    except ValueError:
        return None


#: a compact record (`compact_record`): its request's ingress ordinal, then its token field, each u32be
COMPACT_RECORD = struct.Struct(">II")
#: the token field's flag on a response's last output (finish `length`, or `stop` with `COMPACT_STOP`)
COMPACT_LAST = 1 << 31
#: the token field's flag, beside `COMPACT_LAST`, on a sampled stop token (finish `stop`): the field still carries the
#: stop token's id, whichever of the model's EOS ids was sampled
COMPACT_STOP = 1 << 30
#: what the server writes the proxy for each record under `Proxy(compact=True)`: its canonical position, u32be
COMPACT_POSITION = struct.Struct(">I")
COMPACT_UNIT_BYTES = COMPACT_POSITION.size + COMPACT_RECORD.size
COMPACT_FINISHES = (None, "length", "stop")
COMPACT_FLAGS = {None: 0, "length": COMPACT_LAST, "stop": COMPACT_LAST | COMPACT_STOP}


def compact_record(request: int, token_id: int, finish: str | None = None) -> bytes:
    """One sampled token as a compact record, `COMPACT_RECORD.size` bytes: its request's ingress ordinal (≥ 1, so no
    record is all zero) and its token field, the token's id (below `COMPACT_STOP`) with its finish's flags
    (`COMPACT_FLAGS`): none while more outputs follow, `COMPACT_LAST` on the response's last output at `max_tokens`
    (`finish` "length"), and `COMPACT_LAST` and `COMPACT_STOP` on a sampled stop token (`finish` "stop"), whose id is
    the EOS id the model sampled. A field with `COMPACT_STOP` and not `COMPACT_LAST` is no record's."""
    if type(request) is not int or not 1 <= request < 1 << 32:
        raise ValueError(f"a request's ordinal is a u32 ≥ 1, not {request!r}")
    if finish not in COMPACT_FINISHES:
        raise ValueError(f"a compact record's finish is one of {COMPACT_FINISHES}, not {finish!r}")
    if type(token_id) is not int or not 0 <= token_id < COMPACT_STOP:
        raise ValueError(f"a compact token id is below {COMPACT_STOP}, not {token_id!r}")
    return COMPACT_RECORD.pack(request, token_id | COMPACT_FLAGS[finish])


def compact_cell(record: bytes, frame_bytes: int) -> bytes:
    """A compact record's egress frame: the record, zero-filled to B; padding, B zero bytes, is never one."""
    if len(record) != COMPACT_RECORD.size or frame_bytes < COMPACT_RECORD.size:
        raise ValueError(f"a compact cell is a {COMPACT_RECORD.size}-byte record in a frame of at least its size")
    return record + bytes(frame_bytes - COMPACT_RECORD.size)


def compact_record_data(cell: bytes) -> tuple[int, int, str | None] | None:
    """(request ordinal, token id, finish) of a compact cell or record (`compact_record`); None for bytes that aren't
    one."""
    if len(cell) < COMPACT_RECORD.size or any(cell[COMPACT_RECORD.size :]):
        return None
    request, field = COMPACT_RECORD.unpack_from(cell)
    finish = next(f for f in ("stop", "length", None) if field & COMPACT_FLAGS[f] == COMPACT_FLAGS[f])
    got = (request, field & (COMPACT_STOP - 1), finish)
    try:
        return got if compact_record(*got) == bytes(cell[: COMPACT_RECORD.size]) else None
    except ValueError:
        return None


def compact_records_of(request: int, sampled: Sequence[int], eos: Iterable[int], aborted: bool = False) -> list[bytes]:
    """The compact records of a response whose sampled ids are `sampled`: each id unmarked but the last, a stop where
    it is one of the model's EOS ids (`eos`) and the length's mark where it isn't; every one unmarked when its client
    aborted it (`aborted`), since no record says an abort."""
    stops = frozenset(eos)
    return [compact_record(request, t, None if aborted or k < len(sampled) - 1 else "stop" if t in stops else "length")
            for k, t in enumerate(sampled)]


def compact_finish_violations(sampled: Mapping[int, Sequence[int]], eos: Iterable[int], cells: Iterable[bytes], *,
                              max_tokens: Mapping[int, int], aborted: Collection[int] = ()) -> list[tuple[int, str]]:
    """A link's compact cells held to the Commit's sampled ids (`sampled`, each request's by its ingress ordinal):
    ordinal o's cells, in canonical order, are `compact_records_of(o, sampled[o], eos, o in aborted)`, one for one, so
    each carries the id the model sampled there, the response is as long as the Commit's, and its finish is the
    Commit's, a stop only where the Commit sampled EOS. `aborted` are the ordinals whose finish in the Commit is a
    client's abort, each one's response held to 1 ≤ k < `max_tokens[o]` ids and none of them EOS; any other's length
    finish is at `max_tokens[o]` exactly. `cells` are the link's frames in canonical order, padding left out. A
    violation is (ordinal, why), at most one an ordinal, 0 for a cell that isn't a record. The certifier commits
    whatever ids and finish the server writes; this is the rule the I/O tie holds each response's length and finish
    to. An abort is the Commit's word, not something on the link: the advice charges its `output_len` in full."""
    stops = frozenset(eos)
    got: dict[int, list[tuple[int, str | None]]] = defaultdict(list)
    out: list[tuple[int, str]] = []
    for cell in cells:
        if (c := compact_record_data(cell)) is None:
            out.append((0, "a cell that isn't a compact record"))
        else:
            got[c[0]].append(c[1:])
    for o in sorted(set(got) | set(sampled)):
        if o not in sampled:
            out.append((o, "no request of the Commit is this ordinal"))
            continue
        ids, m, gone = sampled[o], max_tokens[o], o in aborted
        want = [compact_record_data(r)[1:] for r in compact_records_of(o, ids, stops, gone)]
        mine = got.get(o, [])
        k = next((k for k, (g, w) in enumerate(zip(mine, want)) if g != w), None)
        if k is not None and mine[k][0] != want[k][0]:
            out.append((o, f"output {k} is id {mine[k][0]} where the Commit sampled {want[k][0]}"))
        elif k is not None:
            out.append((o, f"finish {mine[k][1] or 'none'} on output {k} where the Commit's is {want[k][1] or 'none'}"))
        elif len(mine) != len(want):
            out.append((o, f"{len(mine)} records where the Commit sampled {len(want)} ids"))
        elif gone and not ids:
            out.append((o, "an abort before any output"))
        elif gone and len(ids) >= m:
            out.append((o, f"an abort at {len(ids)} outputs where max_tokens is {m}"))
        elif gone and stops & set(ids):
            out.append((o, "an abort on an EOS id"))
        elif want and want[-1][1] == "length" and len(ids) != m:
            out.append((o, f"a length finish at {len(ids)} outputs where max_tokens is {m}"))
    return out


def compact_unit(position: int, record: bytes) -> bytes:
    """What the server writes the proxy for one record: its canonical position (`COMPACT_POSITION`), then the record."""
    if len(record) != COMPACT_RECORD.size:
        raise ValueError(f"a compact record is {COMPACT_RECORD.size} bytes")
    return COMPACT_POSITION.pack(position) + record


def compact_placement(frame_bytes: int) -> tuple[int, int, bytes, bytes]:
    """The compact egress template's (p, s, prefix, suffix): the record at 0 … `COMPACT_RECORD.size`, zero fill after
    it, and nothing else."""
    if frame_bytes < COMPACT_RECORD.size:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for a compact record")
    return 0, COMPACT_RECORD.size, b"", b""


#: an id request's HTTP head, up to its length (`id_request`)
ID_REQUEST_HEAD = (b"POST /v1/completions HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\n"
                   b"content-length:")
#: the room of an id request's `model` and `user`, each its JSON string padded with spaces
ID_NAME_BYTES = 64
#: a prompt id's place in an id request: `[` or `,`, then the id in `ID_DIGITS` characters
ID_FIELD_BYTES = 1 + ID_DIGITS


def _ratio_field(x: float, what: str) -> bytes:
    s = b"%8.6f" % x if isinstance(x, (int, float)) and not isinstance(x, bool) else b""
    if not (s and 0 <= x < 10 and float(s) == x):
        raise ValueError(f"{what} is a number in [0, 10) of at most six decimals, not {x!r}")
    return s


def _name_field(s: str, what: str) -> bytes:
    raw = json.dumps(s).encode() if isinstance(s, str) else b""
    if not raw or len(raw) > ID_NAME_BYTES:
        raise ValueError(f"{what} is a string of at most {ID_NAME_BYTES} bytes as JSON, not {s!r}")
    return raw.ljust(ID_NAME_BYTES)


def id_request(*, model: str, user: str, prompt: Sequence[int], max_tokens: int, temperature: float = 0.0,
               top_p: float = 1.0, seed: int = 0, frame_bytes: int, offset: int = LENGTH.size) -> bytes:
    """A canonical `/v1/completions` request of a token-id prompt, whose every field is at a place fixed by the
    prompt's length and the frame size alone, in the link frames `request_frames` puts it in: `ID_REQUEST_HEAD`, its
    body's length in `ID_DIGITS` characters (HTTP's optional whitespace before it), in the first frame, then the body,
    streamed, with no stop strings: `max_tokens` and `seed` in `ID_DIGITS` characters, `temperature` and `top_p` in
    eight (six decimals), `model` and `user` each in `ID_NAME_BYTES`, and each prompt id in its own `ID_FIELD_BYTES`.
    No field straddles a frame: spaces (JSON's whitespace) before one that would move it, with its key, to the next.
    At 4096-byte frames every field but the prompt's is in the first frame. `offset` is the request's first byte's
    place in its first frame's payload: LENGTH's 4 bytes on the active certifier's link, 0 in the warden's cells
    (`warden.cells`, which carry the bytes alone)."""
    if offset not in (0, LENGTH.size):
        raise ValueError(f"an id request starts its first frame's payload at 0 or {LENGTH.size}, not {offset!r}")
    if not prompt or isinstance(prompt, (str, bytes)):
        raise ValueError("an id request's prompt is at least one token id")
    pieces = [b'{"max_tokens":' + _u32_field(max_tokens, "max_tokens"),
              b',"temperature":' + _ratio_field(temperature, "temperature"),
              b',"top_p":' + _ratio_field(top_p, "top_p"), b',"seed":' + _u32_field(seed, "a seed"),
              b',"stop":[],"stream":true', b',"model":' + _name_field(model, "a model"),
              b',"user":' + _name_field(user, "a user"), b',"prompt":',
              *((b"," if k else b"[") + _u32_field(t, "a prompt id") for k, t in enumerate(prompt)), b"]}"]
    room = frame_bytes - HEADER.size
    at = offset + len(ID_REQUEST_HEAD) + ID_DIGITS + 4
    if at > room or max(map(len, pieces)) > room:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for an id request's head or a field")
    parts = []
    for p in pieces:
        gap = -at % room if at // room != (at + len(p) - 1) // room else 0
        parts.append(b" " * gap + p)
        at += gap + len(p)
    raw = b"".join(parts)
    return ID_REQUEST_HEAD + b"%*d" % (ID_DIGITS, len(raw)) + b"\r\n\r\n" + raw


def id_request_fields(request: bytes, frame_bytes: int) -> dict | None:
    """An id request's fields (`id_request`'s keywords but `frame_bytes`), or None for bytes that aren't one."""
    try:
        body = json.loads(request.partition(b"\r\n\r\n")[2])
        fields = {k: body[k] for k in ("model", "user", "prompt", "max_tokens", "temperature", "top_p", "seed")}
        return fields if id_request(**fields, frame_bytes=frame_bytes) == request else None
    except (ValueError, KeyError, TypeError):
        return None


def id_request_frames(prompt_len: int, frame_bytes: int) -> int:
    """The link frames an id request of a `prompt_len`-token prompt takes (`request_frames`)."""
    return request_frame_count(len(id_request(model="", user="", prompt=[0] * prompt_len, max_tokens=1,
                                              frame_bytes=frame_bytes)), frame_bytes)


def event_placement(frame_bytes: int) -> tuple[int, int, bytes, bytes]:
    """The egress template's (p, s, prefix, suffix): an event's frame is its header and `EVENT_HEAD`, the base64 text
    at p, `EVENT_TAIL` at s = p + `EVENT_TEXT`, and zero fill."""
    head = HEADER.pack(DATA, CANONICAL_STREAM, EVENT_BYTES) + EVENT_HEAD
    if len(head) + EVENT_TEXT + len(EVENT_TAIL) > frame_bytes:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for an event")
    return len(head), len(head) + EVENT_TEXT, head, EVENT_TAIL


def data_frames(data: bytes, frame_bytes: int) -> bytes:
    """Bytes on a link: `DATA` frames of B − 9 bytes on `CANONICAL_STREAM`, the last zero-filled, one for no bytes."""
    room = frame_bytes - HEADER.size
    if room < 1:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for data")
    return b"".join(encode_frame(DATA, CANONICAL_STREAM, data[i : i + room].ljust(room, b"\0"), frame_bytes)
                    for i in range(0, max(1, len(data)), room))


def request_frames(request: bytes, frame_bytes: int) -> bytes:
    """A request on the link: its length (u32be) and its bytes as `data_frames`."""
    if frame_bytes - HEADER.size < LENGTH.size:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for a request's length")
    return data_frames(LENGTH.pack(len(request)) + request, frame_bytes)


def request_frame_count(length: int, frame_bytes: int) -> int:
    """The frames `request_frames` puts a request of `length` bytes in."""
    return -(-(LENGTH.size + length) // (frame_bytes - HEADER.size))


def request_placement(frame_bytes: int) -> tuple[int, int, bytes, bytes]:
    """The ingress template's (p, s, prefix, suffix), every `data_frames` frame's: its header, then B − 9 bytes of
    data at p = 9 (a request's length and bytes), zero past them."""
    return HEADER.size, frame_bytes, HEADER.pack(DATA, CANONICAL_STREAM, frame_bytes - HEADER.size), b""


class Requests:
    """Cuts the link's released `DATA` payloads into requests, one frame at a time (no I/O). A request starts at a
    frame, with its length (u32be, zero past the payload's end), and ends at the frame that holds its last byte; the
    rest of that frame is fill. The k-th request is stream k. `feed` gives a frame's stream, the request's bytes in it
    and whether it ends the request."""

    def __init__(self) -> None:
        self.stream = self.left = 0

    def feed(self, payload: bytes) -> tuple[int, bytes, bool]:
        if not self.left:
            self.stream += 1
            self.left = int.from_bytes(payload[: LENGTH.size].ljust(LENGTH.size, b"\0"), "big")
            payload = payload[LENGTH.size :]
        data = payload[: self.left]
        self.left -= len(data)
        return self.stream, data, not self.left


def fresh_salt(window: int, nbytes: int = SALT_BYTES) -> bytes:
    """PoUW window `window`'s salt, `nbytes` bytes of the certifier's own randomness. A source of coins handed to the
    certifier has the same signature."""
    return secrets.token_bytes(nbytes)


def salt_frame(window: int, value: bytes, frame_bytes: int) -> bytes:
    """PoUW window `window`'s salt frame, carrying `value`: `PLACEHOLDER` in the grid, the salt on the server's
    connection."""
    return encode_frame(SALT, SALT_STREAM, window.to_bytes(8, "big") + value, frame_bytes)


def fresh_pair_secret() -> bytes:
    """A sealed pair link's session secret, `aead.SECRET_BYTES` bytes of the certifier host's own randomness."""
    return secrets.token_bytes(aead.SECRET_BYTES)


class Sealer:
    """The sending certifier's half of a sealed pair link (no I/O): slot `slot` of bucket t of its egress grid `params`
    (t counted from window 0) is sealed at `aead.nonce(t // T, t % T, slot)` under `keys`. It refuses (`RuntimeError`)
    a slot at or before the last it sealed, so no nonce is used twice under its keys: a proxy back from maintenance in
    the same session takes the same sealer."""

    def __init__(self, keys: aead.Keys, params: GridParams) -> None:
        self.keys, self.params = keys, params
        self.last: tuple[int, int] | None = None

    def seal(self, t: int, slot: int, frame: bytes) -> bytes:
        p = self.params
        if len(frame) != p.frame_bytes or not 0 <= slot < p.allocation or t < 0:
            raise ValueError(f"a sealed slot is one of {p.allocation} frames of {p.frame_bytes} bytes in a bucket ≥ 0")
        if self.last is not None and (t, slot) <= self.last:
            raise RuntimeError(f"slot {slot} of bucket {t} is at or before the last sealed, {self.last}: a nonce "
                               "is used once under a pair's keys")
        self.last = (t, slot)
        return aead.seal(self.keys, aead.nonce(t // p.buckets, t % p.buckets, slot), frame)

    def seal_bucket(self, t: int, frames: Sequence[bytes]) -> list[bytes]:
        """Bucket t's r slots, the frames then the padding, sealed in slot order."""
        if len(frames) != self.params.allocation:
            raise ValueError(f"a bucket is {self.params.allocation} slots")
        return [self.seal(t, j, f) for j, f in enumerate(frames)]


class Opener:
    """The receiving certifier's half of a sealed pair link (no I/O): the k-th unit read off the link is slot k of the
    sending certifier's egress grid `params` from window `start` on, opened at that slot's nonce under `keys`
    (`open`). The first unit that doesn't open is refused, and so is every unit after it (`refused`): a frame missing
    from the stream changes everything after it, so the link stays failed closed."""

    def __init__(self, keys: aead.Keys, params: GridParams, start: int = 0) -> None:
        self.keys, self.params = keys, params
        self.position = start * params.buckets * params.allocation
        self.refused = False

    @property
    def unit_bytes(self) -> int:
        return self.params.frame_bytes + aead.TAG_BYTES

    def open(self, unit: bytes) -> bytes | None:
        """The next unit's frame, or None if it, or any unit before it, doesn't open."""
        p, k = self.params, self.position
        self.position += 1
        if self.refused:
            return None
        t, slot = divmod(k, p.allocation)
        frame = aead.unseal(self.keys, aead.nonce(t // p.buckets, t % p.buckets, slot), unit) \
            if len(unit) == self.unit_bytes else None
        if frame is None:
            self.refused = True
        return frame


@dataclass(frozen=True, slots=True)
class Pair:
    """One certifier's half of a sealed pair link (`Proxy(pair=…)`): its egress sealed by `sealer`, its ingress opened
    by `opener`, both under keys of the pair's session secret, which only the pair's two certifier hosts hold."""

    sealer: Sealer
    opener: Opener

    @classmethod
    def of(cls, secret: bytes, session: bytes, out: str, into: str, egress: GridParams, peer: GridParams, *,
           peer_start: int = 0) -> Pair:
        """The half that seals its egress grid `egress` under direction `out`'s keys and opens what the pair's other
        certifier seals of its egress grid `peer` (from window `peer_start` on) under direction `into`'s, in session
        `session` under `secret` (`aead.keys`)."""
        if out == into:
            raise ValueError("a pair link's two directions have names of their own, so keys of their own")
        return cls(Sealer(aead.keys(secret, session, out), egress),
                   Opener(aead.keys(secret, session, into), peer, peer_start))


def served_window(registration: Registration) -> int | None:
    """The window a registration's provenance names when it is exactly `{"kind": "served", "window": w}` (the network
    certifier's own, `service.LinkWindows.provenance`), else None."""
    window = registration.record.get("window") if isinstance(registration.record, Mapping) else None
    if isinstance(window, Mapping) and set(window) == {"kind", "window"} and window["kind"] == "served" and \
            type(window["window"]) is int:
        return window["window"]
    return None


def event_position(event: bytes) -> int | None:
    """The event's `id:` field, its canonical position on the link, or None."""
    for line in event.split(b"\n"):
        if line.startswith(b"id:"):
            value = line[3:].strip()
            return int(value) if value.isdigit() else None
    return None


def event_body(event: bytes) -> bytes:
    """What the proxy frames of an event: its lines after the first `id:` line (`event_position`'s), so neither the
    position nor anything before it, such as an HTTP response head, is recorded."""
    lines = event.split(b"\n")
    for i, line in enumerate(lines):
        if line.startswith(b"id:"):
            return b"\n".join(lines[i + 1 :])
    return event


class Events:
    """Cuts a byte stream after every `\\n\\n`. The cuts depend only on the bytes, not on how they were read."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer += data
        out = []
        while (i := self.buffer.find(EVENT_END)) >= 0:
            out.append(bytes(self.buffer[: i + len(EVENT_END)]))
            del self.buffer[: i + len(EVENT_END)]
        return out


class CompactUnits:
    """Cuts a server connection under `Proxy(compact=True)`: an HTTP response head, through its first `\\r\\n\\r\\n`,
    then `COMPACT_UNIT_BYTES`-byte units, each (position, record). The cuts depend only on the bytes, not on how they
    were read; `buffer` holds what is left uncut."""

    def __init__(self) -> None:
        self.buffer, self.head = bytearray(), False

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buffer += data
        if not self.head:
            if (i := self.buffer.find(b"\r\n\r\n")) < 0:
                return []
            del self.buffer[: i + 4]
            self.head = True
        n = len(self.buffer) - len(self.buffer) % COMPACT_UNIT_BYTES
        out = [(COMPACT_POSITION.unpack_from(self.buffer, k)[0],
                bytes(self.buffer[k + COMPACT_POSITION.size : k + COMPACT_UNIT_BYTES]))
               for k in range(0, n, COMPACT_UNIT_BYTES)]
        del self.buffer[:n]
        return out


class Shaper:
    """The honest shaper and the egress certifier on one link, fed frames as they arrive (no I/O).

    `arrive` takes a frame at its canonical position with its ready bucket; `tick` hands what is due at the current
    bucket by `run_shaper`'s rule, ticks the certifier and returns its row, with the window's record at its last bucket.
    Fed every frame at its ready bucket, `hands` is `run_shaper`'s hand times, and the records are `run_shaper`'s while
    no frame is late. A frame handed after its h(k) faults the window it is handed in `coverage-never-established`.

    `queued` is what the certifier's queue would hold had no fault dropped it. A window that closes faulted with frames
    still queued faults the next one too, with the same reason: the Program's grid carries them over.

    With a `prover`, the syncs after the given ones are its: it sees every frame the shaper takes, and declares each of
    the first `windows` windows' syncs at the window's first tick, before anything is handed. The proxy's is
    `LaggedChoices`, whose `declare` only takes a choice already made; an `online.Prover` chooses at the tick. A
    declaration it refuses (`online.Refused`) faults the window `monitor-unavailable`, and the next one, on the frames
    it has read, is the window's.

    With `start`, it begins at window `start`'s first bucket, its certifier at that window, and the first frame it
    takes is the one at `position`: a shaper back from maintenance, which carries nothing from before."""

    def __init__(self, certifier: Certifier, syncs: Sequence[ClockSync], sync_set: SyncSet,
                 prover: Prover | LaggedChoices | None = None, windows: int | None = None, start: int = 0,
                 position: int = 0) -> None:
        self.certifier, self.syncs, self.sync_set = certifier, list(syncs), sync_set
        self.prover, self.windows = prover, len(self.syncs) if windows is None else windows
        if start:
            certifier.window = start
        self.pending: dict[int, tuple[LogicalFrame, int]] = {}
        self.next = position
        self.h = self.last = self.queued = 0
        self.t = start * certifier.params.buckets
        self.hands: list[int] = []
        self._check_advice()

    def _check_advice(self) -> None:
        w = self.certifier.window
        if w < len(self.syncs) and self.syncs[w] not in self.sync_set:
            self.certifier.fault(Reason.COVERAGE_NEVER_ESTABLISHED)

    def arrive(self, position: int, frame: LogicalFrame, ready: int) -> None:
        if position < self.next or position in self.pending:
            self.certifier.fault(Reason.MALFORMED)
        elif len(self.pending) >= self.certifier.params.queue_capacity:
            self.certifier.fault(Reason.OVERFLOW)
        else:
            self.pending[position] = (frame, ready)
            if self.prover is not None:
                self.prover.arrive(0, position, frame, ready)

    def tick(self) -> tuple[Row, Record | None]:
        T = self.certifier.params.buckets
        if self.prover is not None and self.t % T == 0 and self.t // T == len(self.syncs) < self.windows:
            try:
                sync = self.prover.declare()
            except Refused:
                self.certifier.fault(Reason.MONITOR_UNAVAILABLE)
                sync = self.prover.declare()
            self.syncs.append(sync)
            self._check_advice()
        while self.next in self.pending:
            frame, ready = self.pending[self.next]
            w = frame.anchor // T
            if w >= len(self.syncs):
                break
            h = max(eligible(frame, self.syncs[w]), self.h)
            hand = max(h, ready, self.last)
            if hand > self.t:
                break
            del self.pending[self.next]
            if self.t > h:
                self.certifier.fault(Reason.COVERAGE_NEVER_ESTABLISHED)
            self.h, self.last, self.next = h, hand, self.next + 1
            self.hands.append(hand)
            self.certifier.enqueue(frame.payload)
            self.queued += 1
        row = self.certifier.tick()
        self.queued -= min(self.queued, self.certifier.params.allocation)
        self.t += 1
        record = None
        if self.t % T == 0:
            record = self.certifier.close_window()
            if self.queued and not record.status.complete:
                self.certifier.fault(record.status.reason)
            self._check_advice()
        return row, record


class Bucketer:
    """The ingress certifier on one link, fed frames as they arrive (no I/O): a frame arriving at bucket time s is
    `arrive`d with ⌈s⌉ and enqueued at that tick; `tick` returns the released row, with the record at a window's end.
    With `first`, each window's first tick enqueues the certifier's own frame `first(window)`, if it gives one, at the
    head of the queue: ahead of that bucket's arrivals and of any queue carried from the window before, so it leaves in
    slot 0 of that bucket (r_in ≥ 1) and the window's request frames are bucket 0's count less one, exactly. The
    bucketer follows it by its place, never by its bytes: `own` maps the slot of the row last released that holds one
    to its window, and one a fault drops is never released. With `start`, it begins at window `start`'s first bucket,
    its certifier at that window.

    With `served` (off by default), a served link's admission rule (`served.Admission`): arrived frames are enqueued
    only as whole requests of the profile's F_in frames, and only while the window, with the request, still ends idle
    (`served.admits`: every frame handed and released by its last bucket); the rest wait, in order, for the next window.
    So each window's egress counts derive from its own ingress counts (`served.derive`), as N2's served counts check.
    It takes an own frame exactly when the profile counts one (`own`), first in each window's first bucket, which the
    admission projects with the requests' frames."""

    def __init__(self, certifier: Certifier, first: Callable[[int], bytes | None] | None = None,
                 start: int = 0, *, served: Admission | None = None) -> None:
        T, r_in = certifier.params.buckets, certifier.params.allocation
        if served is not None and ((first is not None) != bool(served.profile.own) or not admits(
                [], served.profile.own + served.profile.request_frames, served.sync(start), served.profile,
                served.allocation, r_in, T)):
            raise ValueError("a served link takes an own frame exactly when its profile counts one, and its window "
                             "holds that frame, one request and its response "
                             f"(F = {served.profile.response_frames}) under the admission's sync")
        self.certifier = certifier
        self.first = first
        self.served = served
        self.counts: list[int] = []  # the window's ingress counts so far, under `served`
        if start:
            certifier.window = start
        self.waiting: deque[tuple[int, bytes]] = deque()
        self.t = start * certifier.params.buckets
        self.own: dict[int, int] = {}

    def arrive(self, frame: bytes, bucket: int) -> None:
        self.waiting.append((bucket, frame))

    def tick(self) -> tuple[Row, Record | None]:
        c, T = self.certifier, self.certifier.params.buckets
        own = None
        if self.first is not None and self.t % T == 0 and (frame := self.first(self.t // T)) is not None:
            own = self.t // T
            c.enqueue(frame, first=True)
        a = self.served
        if a is None:
            while self.waiting and self.waiting[0][0] <= self.t:
                c.enqueue(self.waiting.popleft()[1])
        else:
            n, w = a.profile.request_frames, self.t // T
            if self.t % T == 0:
                self.counts = []
            while len(self.waiting) >= n and self.waiting[n - 1][0] <= self.t and \
                    admits(self.counts, c.queued + n, a.sync(w), a.profile, a.allocation, c.params.allocation, T):
                for _ in range(n):
                    c.enqueue(self.waiting.popleft()[1])
        row = c.tick()
        if a is not None:
            self.counts.append(row_count(row))
        # a fault before the release empties the queue, the own frame with it, and pads the row
        self.own = {} if own is None or row[0] is None else {0: own}
        self.t += 1
        return row, c.close_window() if self.t % T == 0 else None


def _choose(snapshot: Prover) -> tuple[ClockSync, float]:
    t0 = time.perf_counter()
    sync = snapshot.choose()
    return sync, round((time.perf_counter() - t0) * 1000, 1)


def reserve_fds(limit: int = 1 << 16) -> None:
    """Grow this process's fd table now, to its open-files limit or `limit` if less. Linux grows it on demand, and once
    the process has a second thread (an executor's, a publisher's) each growth waits for an RCU grace period, tens of
    ms, inside the `socket()` that needed it: live, the first burst of upstream connections then missed its slack."""
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    top = (limit if soft == resource.RLIM_INFINITY else min(soft, limit)) - 1
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        if top > fd:
            os.dup2(fd, top)
            os.close(top)
    finally:
        os.close(fd)


def worker() -> ProcessPoolExecutor:
    """The proxy's worker for its online choices: one spawned process, started before it returns, so that its
    start-up imports don't share window 0 with the proxy's forwards. A thread would hold the proxy's GIL."""
    pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    pool.submit(int).result()
    return pool


class InlineExecutor(Executor):
    """An executor that runs each call at `submit`, for a driver on a `VirtualClock`, where no tick waits for it: every
    choice is ready."""

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as err:
            future.set_exception(err)
        return future


class LaggedChoices:
    """The online prover's declarations off the proxy's tick (no I/O). `start()`, right after tick w·T − lag (window
    0's before the first tick), submits window w's choice on `prover.snapshot()` to `executor`; `declare()` at tick w·T
    declares it if it is done, else the envelope, without waiting. `log` has the lag, the windows declared the envelope
    for want of a choice (`unready`) and each choice's time in ms (`choose_ms`)."""

    def __init__(self, prover: Prover, executor: Executor) -> None:
        if not 0 < prover.lag < prover.T:
            raise ValueError("the proxy's online prover chooses off its tick: a lag of 1 … T − 1 buckets")
        self.prover, self.executor = prover, executor
        self._next: tuple[int, Future] | None = None
        self._refused: int | None = None
        self.log: dict = {"lag": prover.lag, "unready": [], "choose_ms": []}

    def arrive(self, link: int, position: int, frame: LogicalFrame, ready: int) -> None:
        self.prover.arrive(link, position, frame, ready)

    def start(self) -> None:
        w = len(self.prover.syncs)
        try:
            snapshot = self.prover.snapshot()
        except Refused:
            self._refused = w
            snapshot = self.prover.snapshot()
        self._next = (w, self.executor.submit(_choose, snapshot))

    def declare(self) -> ClockSync:
        """Window w's declaration; `Refused` once if its snapshot was, and then the choice made on what it read."""
        w, sync = len(self.prover.syncs), None
        if self._refused == w:
            self._refused = None
            raise Refused("a frame is ready only after its anchor's bucket")
        if self._next is not None and self._next[0] == w and self._next[1].done():
            sync, ms = self._next[1].result()
            self.log["choose_ms"].append(ms)
        else:
            self.log["unready"].append(w)
        self._next = None
        return self.prover.declare(self.prover.envelope if sync is None else sync)


async def wait_until(deadline: float, clock: Callable[[], float],
                     sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
    """Sleep until `clock()` reaches `deadline`, in waits of at most `MAX_WAIT_S`, so that a far deadline is met as
    closely as a near one."""
    while deadline - clock() > MAX_WAIT_S:
        await sleep(MAX_WAIT_S)
    await sleep(deadline - clock())


async def precise_sleep(delay: float) -> None:
    """`asyncio.sleep(delay)` in waits of at most `MAX_WAIT_S` against the loop's clock: the proxy's sleep unless its
    caller injects one. An injected sleep (a `VirtualClock`, the device difftest's harness) gets one wait a tick."""
    loop = asyncio.get_running_loop()
    await wait_until(loop.time() + delay, loop.time)


class VirtualClock:
    """A clock its caller moves, for runs driven event by event (tests, and replays without the host's timer):
    `clock()` reads it, `sleep` waits until `advance` passes the deadline."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self._waiters: list[tuple[float, int, asyncio.Future]] = []
        self._seq = itertools.count()

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (self.now + delay, next(self._seq), future))
        await future

    def advance(self, to: float) -> None:
        self.now = max(self.now, to)
        while self._waiters and self._waiters[0][0] <= self.now:
            future = heapq.heappop(self._waiters)[2]
            if not future.done():
                future.set_result(None)


@dataclass
class _Stream:
    anchor: int | None = None
    count: int = 0
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    events: Events = field(default_factory=Events)
    units: CompactUnits = field(default_factory=CompactUnits)
    task: asyncio.Task | None = None
    window: int | None = None  # the ingress window of its release
    deadline: float | None = None  # when its request must be written upstream
    delivered: float | None = None  # when it was
    ended: int | None = None  # the ready bucket of its end upstream


def _faulted(record: Record, reason: Reason) -> Record:
    """`record` with `reason` as its status, unless it has one already: a window records its first fault."""
    return record if not record.status.complete else replace(record, status=Status(reason))


class Proxy:
    """The active certifier on one link: `run(reader, writer)` serves the link connection from `origin`, the clock's time
    of bucket 0, for the windows of `syncs`, or `windows` of them declared by an online `prover`, then closes it.

    - `ingress_link`, the ingress link's name, has no default: with `origin` it names this certifier's run in a
      certified PoUW window's stream (`served_commit.certifier_run`), so two proxies never share a run by a default
      name;
    - the prover's choices (`LaggedChoices`) run on `executor`, by default a `worker()` process, which the proxy shuts
      down when `run` ends; on a `VirtualClock`, pass `InlineExecutor()`;
    - `publish(record, params)` receives every window's record, egress and ingress, as it closes (by default they are
      kept in `records`);
    - `frames` logs every egress frame read, as (position, stream, index, anchor, ready), and `shaper.hands` its hand
      bucket by position; `lateness` holds each tick's lateness in seconds, per grid; `link_frames_read` and
      `requests_released` count the ingress frames read and the requests released to the server;
    - with `salt`, a source of one salt of `salt_bits` bits per PoUW window of `pouw_window` certifier windows, such as
      `fresh_salt`, the ingress grid carries each PoUW window's salt frame at its first bucket, the server receives
      the salt on its own connection (`salts_released` and `salts_delivered` count them), and `reveal(p,
      registration)` gives the verifier PoUW window p's salt, once, when p is registered (`registered` reads the window
      off a registration, by default `served_window`), its last ingress record is published, and the certifier's clock
      is no more than `reveal_deadline` past p's end (one PoUW window by default); with `prover_key`,
      `commit_weights(p, root, signature)` records the weights' root the prover signed (`weights_message`, under
      `prover_context`) before p's salt is drawn, and `committed(p, registration)` gives it on `reveal`'s conditions;
    - with `challenger_key` and `signer`, both or neither, `reveal(p, registration, request)` and `committed(p,
      registration, request)` take only a request the auditor's challenger signed under `challenger_key`
      (`reveal_request_message`) and answer a `Revealed` that `signer`, the network certifier's own key, signed
      (`reveal_response_message`), which `serve_reveals` serves to a `RemoteCertifier`; without them they are the
      in-process calls, which take no request and return the value. `challenger_key` comes from the auditor's
      deployment record, never from the prover's configuration, and `trusted-host` covers that record's custody.
      Both messages bind the signer's `key_id` and `run_id`, 32 bytes (`RUN_ID_BYTES`) the proxy draws from
      `run_nonce` at construction (the certifier's own randomness by default), so a request or an answer of one run,
      or of another proxy on the same link name, is never taken in another;
    - with `start` and `position`, it is back from maintenance: it serves windows `start` … `windows` − 1, its first
      egress frame the Program's at `position`;
    - with `served`, a served link's profile (`served.ServedProfile`): its ingress side admits requests by the
      admission rule (`Bucketer(served=…)`) under each window's declared sync (the slowest of Σ_sync under an online
      prover, or past `syncs`), so that each window ends idle, which N2's served counts check. With `salt` the profile
      counts its frame (`own` 1), at a PoUW window of one certifier window, so that frame heads every window;
    - with `ordinals`, each stream's server connection opens with a header (`HEADER`, 9 bytes) of kind `ORDINAL`
      whose stream is the stream's, the request's ingress ordinal, before the request's bytes, so that the server
      can name the request in its response's id events (`id_event`);
    - with `compact` (which takes `ordinals`), each server connection is compact records (`CompactUnits`: an HTTP
      response head, then (position, record) units, `compact_unit`), and each egress frame is a record zero-filled to
      B (`compact_cell`), B ≥ 8: a record that isn't one (`compact_record_data`) or names another request than its
      connection's is malformed. The Lean certifier device models it (`CertifierDevice`'s `Config.compact`), and the
      gate's rehearsal runs it (`certifier-rehearse --traffic varied --compact B`);
    - with `pair` (off by default), the link is a sealed pair link to another of the auditor's certifiers: each egress
      slot, frame or padding, is written sealed (`Pair.sealer`) after the egress certifier records it, and each unit
      read is opened (`Pair.opener`) before the ingress certifier takes it, so every record is the unsealed run's. A
      unit that doesn't open faults its ingress window `malformed`, and every later one too (`link_frames_read` counts
      units then);
    - with `paced`, seconds past `slack` (off by default), the egress tick hands each bucket to the pacer
      (`pacer.Pacer`), which writes it to the link `paced` after its boundary on the host's monotonic clock, `origin`'s
      clock then; `releases` holds each bucket's `pacer.Report`, and each egress record is published once its last
      bucket's is in;
    - `until(predicate)` waits until the proxy's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, upstream: tuple[str, int], egress: GridParams, ingress: GridParams, sync_set: SyncSet,
                 syncs: Sequence[ClockSync], *, origin: float, slack: float = 0.025,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = precise_sleep,
                 publish: Callable[[Record, GridParams], None] | None = None,
                 egress_link: str = "responses", ingress_link: str,
                 prover: Prover | None = None, windows: int | None = None, executor: Executor | None = None,
                 salt: Callable[[int, int], bytes] | None = None, salt_bits: int = SALT_BITS, pouw_window: int = 1,
                 registered: Callable[[Registration], int | None] = served_window, start: int = 0,
                 position: int = 0, reveal_deadline: float | None = None, prover_key: PublicKey | None = None,
                 prover_context: bytes = b"", challenger_key: PublicKey | None = None,
                 signer: Signer | None = None, run_nonce: Callable[[int], bytes] = secrets.token_bytes,
                 served: ServedProfile | None = None, ordinals: bool = False, compact: bool = False,
                 pair: Pair | None = None, paced: float | None = None) -> None:
        if paced is not None and not (isinstance(paced, (int, float)) and paced > slack):
            raise ValueError(f"a paced bucket leaves past the slack, so that a tick in time hands it before its time, "
                             f"not {paced!r} s after its boundary")
        if (egress.buckets * egress.bucket_ms) != (ingress.buckets * ingress.bucket_ms):
            raise ValueError("the egress and ingress grids have windows of the same length")
        if compact and not (ordinals and egress.frame_bytes >= COMPACT_RECORD.size):
            raise ValueError(f"compact records name their request by its ingress ordinal (ordinals) and fill egress "
                             f"frames of at least {COMPACT_RECORD.size} bytes")
        if pair is not None and (pair.sealer.params != egress or pair.opener.params.frame_bytes != ingress.frame_bytes):
            raise ValueError("a pair link seals this certifier's egress grid and opens frames of its ingress size")
        if prover is not None and syncs:
            raise ValueError("an online prover declares every window's sync")
        if served is not None and (egress.buckets, egress.bucket_ms) != (ingress.buckets, ingress.bucket_ms):
            raise ValueError("a served link's grids share T and τ_b: its anchors are ingress buckets")
        if served is not None and (served.own != (salt is not None) or salt is not None and pouw_window != 1):
            raise ValueError("a served link's profile counts the salt frame (own 1) exactly when the salt is on, and "
                             "then every window has one (a PoUW window of one certifier window)")
        if salt is not None and ingress.frame_bytes < SALT_FRAME_BYTES:
            raise ValueError(f"a salt frame needs ingress frames of at least {SALT_FRAME_BYTES} bytes")
        if not (type(salt_bits) is int and salt_bits > 0 and salt_bits % 8 == 0):
            raise ValueError(f"a salt is a positive whole number of bytes, not {salt_bits!r} bits")
        if not (type(pouw_window) is int and pouw_window >= 1):
            raise ValueError(f"a PoUW window is a positive number of certifier windows, not {pouw_window!r}")
        self.upstream, self.egress, self.ingress, self.ordinals = upstream, egress, ingress, ordinals
        self.compact = compact
        self.origin, self.slack, self.clock, self.sleep = origin, slack, clock, sleep
        self.windows = len(syncs) if windows is None else windows
        if not 0 <= start <= self.windows or position < 0:
            raise ValueError(f"a proxy starts at one of windows 0 … {self.windows}, at a position ≥ 0")
        if start and prover is not None:
            raise ValueError("an online prover declares from window 0 on: a proxy back from maintenance takes syncs")
        self.start = start
        reserve_fds()
        self.choices = None if prover is None else LaggedChoices(prover, executor or worker())
        self.shaper = Shaper(Certifier(egress_link, egress), syncs, sync_set, self.choices, self.windows, start,
                             position)
        self.salt, self.salt_bits, self.pouw_window, self.registered = salt, salt_bits, pouw_window, registered
        pouw_s = pouw_window * ingress.buckets * ingress.bucket_ms / 1000
        self.reveal_deadline = pouw_s if reveal_deadline is None else reveal_deadline
        if not (isinstance(self.reveal_deadline, (int, float)) and self.reveal_deadline > 0):
            raise ValueError(f"a reveal deadline is a positive number of seconds, not {reveal_deadline!r}")
        if not isinstance(prover_context, bytes):
            raise ValueError("the prover's context is bytes")
        self.prover_key, self.prover_context = prover_key, prover_context
        if (challenger_key is None) != (signer is None):
            raise ValueError("a signed reveal takes the challenger's key and the certifier's signer, an in-process one "
                             "neither")
        self.challenger_key, self.signer = challenger_key, signer
        self.run_id = run_nonce(RUN_ID_BYTES)
        if not (isinstance(self.run_id, bytes) and len(self.run_id) == RUN_ID_BYTES):
            raise ValueError(f"a run id is {RUN_ID_BYTES} bytes")
        self._salts: dict[int, bytes] = {}
        self._revealed: set[int] = set()
        self._weights: dict[int, bytes] = {}
        self._seal = threading.Lock()  # `commit_weights` may run off the loop: the draw and the commit never interleave
        self._salt_out: asyncio.Queue | None = None
        self._salt_task: asyncio.Task | None = None
        self.salts_released = self.salts_delivered = 0
        slowest = ClockSync(max(sync_set.deltas), min(sync_set.rhos))
        admission = None if served is None else Admission(
            served, egress.allocation, lambda w: syncs[w] if prover is None and w < len(syncs) else slowest)
        self.bucketer = Bucketer(Certifier(ingress_link, ingress), None if salt is None else self._draw_salt, start,
                                 served=admission)
        self.pair, self.paced = pair, paced
        self.releases: list[Report] = []
        self._pacer: Pacer | None = None
        self._reports_task: asyncio.Task | None = None
        self._unwritten: Counter[int] = Counter()
        self._held: dict[int, Record] = {}
        self.records: list[Record] = []
        self.publish = publish or (lambda record, params: self.records.append(record))
        self.streams: dict[int, _Stream] = {}
        self.requests = Requests()
        self.frames: list[tuple[int, int, int, int, int]] = []
        self.lateness: dict[str, list[float]] = {"egress": [], "ingress": []}
        self.link_frames_read = self.requests_released = 0
        self._released: list[_Stream] = []
        self._late: list[_Stream] = []
        self._due: dict[str, tuple[int, float, Certifier, Callable[[int], None]]] = {}
        self._changed = asyncio.Event()

    async def until(self, predicate: Callable[[Proxy], bool]) -> None:
        while not predicate(self):
            self._changed.clear()
            await self._changed.wait()

    async def run(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        link = asyncio.create_task(self._read_link(reader))
        try:
            await asyncio.gather(self._egress(writer), self._ingress())
        finally:
            link.cancel()
            tasks = [st.task for st in self.streams.values() if st.task] + \
                [t for t in (self._salt_task, self._reports_task) if t]
            for task in tasks:
                task.cancel()
            await asyncio.gather(link, *tasks, return_exceptions=True)
            if self._pacer is not None:
                await self._pacer.kill()
            writer.close()
            if self.choices is not None:
                await asyncio.to_thread(self.choices.executor.shutdown, True, cancel_futures=True)
            self._changed.set()

    def _bucket_ready(self, now: float) -> int:
        return math.floor((now - self.origin) * 1000 / self.egress.bucket_ms) + 1

    async def _ticks(self, grid: str, params: GridParams, certifier: Certifier, tick: Callable[[int], None]) -> None:
        """`grid`'s ticker: one `sleep` a tick, to the tick's boundary, then a turn of `_run_due` once every ticker
        that woke with it has said so."""
        tau = params.bucket_ms / 1000
        for t in range(self.start * params.buckets, self.windows * params.buckets):
            deadline = self.origin + t * tau
            await self.sleep(deadline - self.clock())
            self._due[grid] = (t, deadline, certifier, tick)
            await asyncio.sleep(0)
            self._run_due()

    def _run_due(self) -> None:
        """Every tick whose ticker has woken, in `TICK_ORDER`, in one step: each reads the clock as its turn comes, when
        it reads its link, and one read more than `slack` past its boundary faults its grid `monitor-unavailable`."""
        for grid in TICK_ORDER:
            if (due := self._due.pop(grid, None)) is not None:
                t, deadline, certifier, tick = due
                late = self.clock() - deadline
                self.lateness[grid].append(late)
                if late > self.slack:
                    certifier.fault(Reason.MONITOR_UNAVAILABLE)
                tick(t)

    async def _egress(self, writer: asyncio.StreamWriter) -> None:
        p, pair = self.egress, self.pair
        bucket_bytes = p.allocation * (p.frame_bytes if pair is None else p.frame_bytes + aead.TAG_BYTES)
        padding = memoryview(bytes(p.allocation * p.frame_bytes))
        pad = bytes(p.frame_bytes)
        tau = p.bucket_ms / 1000
        pacer = None
        if self.paced is not None:
            pacer = self._pacer = await Pacer.start({0: writer.get_extra_info("socket").fileno()})
            self._reports_task = asyncio.create_task(self._reports(pacer))
        choices = self.choices
        if choices is not None and self.windows:
            choices.start()

        def tick(t: int) -> None:
            if pacer is None:
                backlog = writer.transport.get_write_buffer_size()
                refused, dropped = backlog > bucket_bytes, backlog > 4 * bucket_bytes
            else:
                behind = pacer.behind(self.clock())
                refused, dropped = behind > 1, behind > 4
            if refused:
                self.shaper.certifier.fault(Reason.DELIVERY_REFUSED)
            self._check_forwarding()
            row, record = self.shaper.tick()
            n = row_count(row)
            if not dropped:
                if pair is None:
                    cells = [*row[:n], padding[: (p.allocation - n) * p.frame_bytes]]
                else:
                    cells = pair.sealer.seal_bucket(t, [*row[:n], *[pad] * (p.allocation - n)])
                if pacer is None:
                    writer.writelines(cells)
                else:
                    pacer.hand(t, self.origin + t * tau + self.paced, [(0, b"".join(cells), False)])
                    self._unwritten[t // p.buckets] += 1
            if choices is not None and (t + choices.prover.lag) % p.buckets == 0 and \
                    (t + choices.prover.lag) // p.buckets < self.windows:
                choices.start()
            if record is not None:
                if self._unwritten[record.window]:
                    self._held[record.window] = record
                else:
                    self.publish(record, p)
            self._changed.set()

        await self._ticks("egress", p, self.shaper.certifier, tick)
        if pacer is None:
            await writer.drain()
            return
        await pacer.close()
        await self._reports_task
        for w in sorted(self._held):
            self.publish(_faulted(self._held.pop(w), Reason.MONITOR_UNAVAILABLE), p)
        self._changed.set()

    async def _reports(self, pacer: Pacer) -> None:
        async for report in pacer.reports():
            self._report(report)

    def _report(self, report: Report) -> None:
        """A bucket's release or its writes as the pacer reports them. A bucket that reached the pacer after its time,
        or whose release started more than `slack` after it, faults its window `monitor-unavailable`, and one the link
        refused, `delivery-refused`: the open window's certifier, or the record held for a closed one. A window's record
        is published once the writes of every bucket of it that was handed are reported."""
        w = report.t // self.egress.buckets
        if report.end is None:
            reason = Reason.MONITOR_UNAVAILABLE if report.arrived > 0 or report.start > self.slack else None
        else:
            self.releases.append(report)
            reason = None if report.written else Reason.DELIVERY_REFUSED
        if reason is not None:
            if w in self._held:
                self._held[w] = _faulted(self._held[w], reason)
            elif w == self.shaper.certifier.window:
                self.shaper.certifier.fault(reason)
        if report.end is not None:
            self._unwritten[w] -= 1
            if not self._unwritten[w] and w in self._held:
                self.publish(self._held.pop(w), self.egress)
        self._changed.set()

    def _check_forwarding(self) -> None:
        """Fault every window a late forward can reach: a released request not written upstream by its deadline makes
        its ingress window delivery-refused, if still open, and the egress windows too until the stream has ended and
        every frame that arrived before its end is handed, since the server's events can't be the Program's before."""
        now, waiting = self.clock(), []
        for st in self._released:
            if st.delivered is None and now < st.deadline:
                waiting.append(st)
            elif st.delivered is None or st.delivered > st.deadline:
                if self.bucketer.certifier.window == st.window:
                    self.bucketer.certifier.fault(Reason.DELIVERY_REFUSED)
                self._late.append(st)
        self._released = waiting
        self._late = [st for st in self._late if not self._settled(st)]
        if self._late:
            self.shaper.certifier.fault(Reason.DELIVERY_REFUSED)

    def _settled(self, st: _Stream) -> bool:
        return st.ended is not None and self.shaper.t >= st.ended and \
            all(ready > st.ended for _, ready in self.shaper.pending.values())

    async def _ingress(self) -> None:
        p = self.ingress
        tau, tau_e = p.bucket_ms / 1000, self.egress.bucket_ms / 1000

        def tick(t: int) -> None:
            if self.pair is not None and self.pair.opener.refused:
                self.bucketer.certifier.fault(Reason.MALFORMED)
            row, record = self.bucketer.tick()
            anchor = t * p.bucket_ms // self.egress.bucket_ms
            own = self.bucketer.own
            for i, frame in enumerate(row[: row_count(row)]):
                if i in own:
                    self._release_salt(own[i] // self.pouw_window)
                    continue
                kind, _, payload = decode_frame(frame)
                if kind != DATA:
                    continue
                s, data, last = self.requests.feed(payload)
                st = self._stream(s)
                st.out.put_nowait((data, last))
                if last and st.anchor is None:
                    st.anchor, st.window = anchor, t // p.buckets
                    st.deadline = self.origin + min(t * tau + self.slack, (anchor + 1) * tau_e)
                    self._released.append(st)
                    self.requests_released += 1
            if record is not None:
                self.publish(record, p)
            self._changed.set()

        await self._ticks("ingress", p, self.bucketer.certifier, tick)

    def _draw_salt(self, window: int) -> bytes | None:
        """At a PoUW window's first certifier window, draw its salt and give the certifier's own frame, the
        placeholder's."""
        if window % self.pouw_window:
            return None
        p, n = window // self.pouw_window, self.salt_bits // 8
        salt = self.salt(p, n)
        if len(salt) != n:
            raise ValueError(f"the salt source gave {len(salt)} bytes, not {n}")
        with self._seal:
            self._salts[p] = salt
        return salt_frame(p, PLACEHOLDER, self.ingress.frame_bytes)

    def _release_salt(self, p: int) -> None:
        """Queue PoUW window p's salt for the server, its frame being released. A frame of kind `SALT` read from the
        link is never the certifier's, and is left alone like any frame of another kind."""
        self.salts_released += 1
        if self._salt_out is None:
            self._salt_out = asyncio.Queue()
            self._salt_task = asyncio.create_task(self._salt_upstream(self._salt_out))
        salt = self._salts[p]
        self._salt_out.put_nowait(salt_frame(p, salt, HEADER.size + 8 + len(salt)))

    async def _salt_upstream(self, out: asyncio.Queue) -> None:
        """The salt stream's server connection: written, never read, so that nothing on it reaches egress."""
        try:
            _, writer = await asyncio.open_connection(*self.upstream)
        except OSError:
            self.bucketer.certifier.fault(Reason.DELIVERY_REFUSED)
            return
        try:
            while True:
                writer.write(await out.get())
                await writer.drain()
                self.salts_delivered += 1
                self._changed.set()
        finally:
            writer.close()

    @property
    def certifier_run(self) -> tuple[bytes, bytes, str]:
        """This certifier run's identity, (key id, run id, link): the signer's key id, `run_id` and the ingress link
        name, the one the reveal messages bind. Every signed answer binds it, and consumers key PoUW windows on it: a
        `Revealed`'s `key_id` and `run_id`, with the link it verifies for, are it, and a `RemoteCertifier` of this proxy
        gives the same. A proxy without a signer has no key id, so it raises `ValueError`."""
        if self.signer is None:
            raise ValueError("a proxy without a signer has no key id, so no certifier run")
        return self.signer.key_id, self.run_id, self.bucketer.certifier.link

    def reveal(self, window: int, registration: Registration | None,
               request: bytes | None = None) -> bytes | Revealed:
        """PoUW window `window`'s salt, for the verifier, given the window's registration as the proof service returned
        it, once. It fails closed (`RuntimeError`): without a registration whose receipt is its own and off which
        `registered` reads `window`, while the window's last ingress record is unpublished (t_d), once the certifier's
        clock is more than `reveal_deadline` past the window's end, for a window whose salt wasn't drawn (one this proxy
        was down for at its start), and for a window whose salt it has revealed already. So a registration made after
        the window's work, or a second one naming the same window, gets no salt.

        With a challenger key it first refuses a `request` that isn't the registered challenger's signature over
        `reveal_request_message(signer.key_id, run_id, ingress link, window, registration.digest, SALT_KIND)`, so no
        one else spends the window's one reveal, nor does a request for another certifier or run, and a registration
        the challenger didn't sign (its digest binds its record) is refused; it answers a `Revealed` its signer signed.
        A signer that refuses leaves the reveal spent and gives nothing. In process (no keys) it takes no request and
        returns the salt."""
        self._authenticate(window, registration, request, SALT_KIND)
        self._opens(window, registration, "salt")
        if window not in self._salts:
            raise RuntimeError(f"window {window}'s salt wasn't drawn")
        with self._seal:
            if window in self._revealed:
                raise RuntimeError(f"window {window}'s salt is revealed already")
            self._revealed.add(window)
        return self._respond(window, registration, SALT_KIND, self._salts[window])

    def _authenticate(self, window: int, registration: Registration | None, request: bytes | None,
                      what: bytes) -> None:
        """Refuse a request the registered challenger didn't sign for this certifier's key id and run, `window`,
        `registration` and `what`; in process (no challenger key), any request."""
        if self.challenger_key is None:
            if request is not None:
                raise RuntimeError("no challenger key is registered with this certifier, so it takes no signed request")
            return
        try:
            message = reveal_request_message(self.signer.key_id, self.run_id, self.bucketer.certifier.link, window,
                                             getattr(registration, "digest", None), what)
        except ValueError:
            message = None
        if message is None or not isinstance(request, bytes) or not self.challenger_key.verify(message, request):
            raise RuntimeError(f"the request for window {window}'s {what.decode()} isn't signed by the registered "
                               "challenger for this certifier's run")

    def _respond(self, window: int, registration: Registration, what: bytes, value: bytes) -> bytes | Revealed:
        if self.signer is None:
            return value
        key_id = self.signer.key_id
        message = reveal_response_message(key_id, self.run_id, self.bucketer.certifier.link, window,
                                          registration.digest, what, value)
        try:
            return Revealed(window, value, self.signer.sign(message), key_id, self.run_id)
        except SignerRefused as e:
            raise RuntimeError(f"the certifier's signer refused window {window}'s {what.decode()}: {e}") from None

    def _opens(self, window: int, registration: Registration | None, what: str) -> None:
        """Refuse unless PoUW window `window` may be opened now on `registration` (`reveal`'s conditions but the once)."""
        if self.salt is None:
            raise ValueError("the salt is off")
        if not self._registers(window, registration):
            raise RuntimeError(f"no registration of window {window} with its receipt, so its {what} isn't revealed")
        if (window + 1) * self.pouw_window > self.bucketer.certifier.window:
            raise RuntimeError(f"window {window}'s ingress records aren't all published, so its {what} isn't revealed")
        end = self.origin + (window + 1) * self.pouw_window * self.ingress.buckets * self.ingress.bucket_ms / 1000
        if self.clock() > end + self.reveal_deadline:
            raise RuntimeError(f"window {window} ended more than {self.reveal_deadline:g} s ago on the certifier's "
                               f"clock, so its {what} isn't revealed")

    def commit_weights(self, window: int, root: bytes, signature: bytes) -> None:
        """Record PoUW window `window`'s weights' root, `WEIGHTS_ROOT_BYTES` bytes, in its sealed slot, so that the
        certifier's word orders it before the window's salt. Only the registered prover's root is taken: `signature`
        verifies under `prover_key` over `weights_message(ingress link, window, root, prover_context)`. It refuses
        (`RuntimeError`) without a prover key, on a signature that doesn't verify, once the window's first ingress tick,
        the salt's draw, has run or was missed (a proxy back from maintenance after it), and a second root for the
        window."""
        if self.salt is None:
            raise ValueError("the salt is off")
        if not isinstance(root, bytes) or len(root) != WEIGHTS_ROOT_BYTES:
            raise ValueError(f"a weights' root is {WEIGHTS_ROOT_BYTES} bytes")
        if not (type(window) is int and 0 <= window and window * self.pouw_window < self.windows):
            raise ValueError(f"PoUW window {window!r} isn't one this proxy serves")
        if self.prover_key is None:
            raise RuntimeError("no prover key is registered with this certifier, so it records no weights")
        message = weights_message(self.bucketer.certifier.link, window, root, self.prover_context)
        if not isinstance(signature, bytes) or not self.prover_key.verify(message, signature):
            raise RuntimeError(f"PoUW window {window}'s weights' root isn't signed by the registered prover")
        with self._seal:
            if window in self._salts or self.bucketer.t > window * self.pouw_window * self.ingress.buckets:
                raise RuntimeError(f"PoUW window {window}'s salt is drawn or its draw has passed, so its weights "
                                   "can't be recorded before it")
            if window in self._weights:
                raise RuntimeError(f"PoUW window {window}'s weights are already recorded")
            self._weights[window] = root

    def committed(self, window: int, registration: Registration | None,
                  request: bytes | None = None) -> bytes | Revealed:
        """PoUW window `window`'s weights' root, recorded before its salt, on `reveal`'s conditions but the once (it is a
        hiding commitment, and taking it doesn't reveal the salt), its request signed over `WEIGHTS_KIND` and its answer
        signed as `reveal`'s are; it fails closed (`RuntimeError`) for a window whose weights weren't recorded."""
        self._authenticate(window, registration, request, WEIGHTS_KIND)
        self._opens(window, registration, "weights' root")
        if window not in self._weights:
            raise RuntimeError(f"no weights were recorded before PoUW window {window}'s salt")
        return self._respond(window, registration, WEIGHTS_KIND, self._weights[window])

    def _registers(self, window: int, registration: Registration | None) -> bool:
        if not isinstance(registration, Registration) or not isinstance(registration.receipt, Mapping):
            return False
        try:
            digest = R.digest(registration.record)
        except (AttributeError, TypeError, ValueError):
            return False
        receipt = registration.receipt
        return receipt.get("format") == R.RECEIPT and receipt.get("registration") == digest == registration.digest \
            and self.registered(registration) == window

    async def _read_link(self, reader: asyncio.StreamReader) -> None:
        opener = None if self.pair is None else self.pair.opener
        size, buffer = self.ingress.frame_bytes if opener is None else opener.unit_bytes, bytearray()
        while data := await reader.read(1 << 16):
            bucket = math.ceil((self.clock() - self.origin) * 1000 / self.ingress.bucket_ms)
            buffer += data
            whole = len(buffer) - len(buffer) % size
            for i in range(0, whole, size):
                frame = bytes(buffer[i : i + size])
                if opener is not None and (frame := opener.open(frame)) is None:
                    self.bucketer.certifier.fault(Reason.MALFORMED)
                else:
                    self.bucketer.arrive(frame, bucket)
                self.link_frames_read += 1
            del buffer[:whole]
            self._changed.set()

    def _stream(self, s: int) -> _Stream:
        if s not in self.streams:
            st = self.streams[s] = _Stream()
            st.task = asyncio.create_task(self._upstream(s, st))
        return self.streams[s]

    async def _upstream(self, s: int, st: _Stream) -> None:
        try:
            reader, writer = await asyncio.open_connection(*self.upstream)
        except OSError:
            self.bucketer.certifier.fault(Reason.DELIVERY_REFUSED)
            st.ended = self._bucket_ready(self.clock())
            return
        writer.transport.set_write_buffer_limits(0)
        if self.ordinals:
            writer.write(HEADER.pack(ORDINAL, s, 0))

        async def pump() -> None:
            while True:
                payload, last = await st.out.get()
                writer.write(payload)
                if last and st.delivered is None:
                    await writer.drain()
                    st.delivered = self.clock()

        pumping = asyncio.create_task(pump())
        try:
            while data := await reader.read(1 << 16):
                ready = self._bucket_ready(self.clock())
                if self.compact:
                    for position, record in st.units.feed(data):
                        self._egress_record(s, st, position, record, ready)
                else:
                    for event in st.events.feed(data):
                        self._egress_event(s, st, event, ready)
                self._changed.set()
            if st.events.buffer or st.units.buffer:
                self.shaper.certifier.fault(Reason.MALFORMED)
        finally:
            st.ended = self._bucket_ready(self.clock())
            pumping.cancel()
            writer.close()

    def _egress_event(self, s: int, st: _Stream, event: bytes, ready: int) -> None:
        position, body = event_position(event), event_body(event)
        if position is None or HEADER.size + len(body) > self.egress.frame_bytes:
            self.shaper.certifier.fault(Reason.MALFORMED)
            return
        self._egress_frame(s, st, position, encode_frame(DATA, CANONICAL_STREAM, body, self.egress.frame_bytes), ready)

    def _egress_record(self, s: int, st: _Stream, position: int, record: bytes, ready: int) -> None:
        got = compact_record_data(record)
        if got is None or got[0] != s:
            self.shaper.certifier.fault(Reason.MALFORMED)
            return
        self._egress_frame(s, st, position, compact_cell(record, self.egress.frame_bytes), ready)

    def _egress_frame(self, s: int, st: _Stream, position: int, payload: bytes, ready: int) -> None:
        if st.anchor is None:
            self.shaper.certifier.fault(Reason.MALFORMED)
            return
        frame = LogicalFrame(payload, st.anchor, st.count)
        self.frames.append((position, s, st.count, st.anchor, ready))
        st.count += 1
        self.shaper.arrive(position, frame, ready)


_KINDS = {"salt": SALT_KIND, "weights": WEIGHTS_KIND}
_MALFORMED = ("a request is a JSON object of window, kind (salt or weights), registration (record, receipt, digest) "
              "and request (the challenger's signature, hex), or of window and kind certificates")


def _line(answer: Mapping) -> bytes:
    return json.dumps(answer).encode() + b"\n"


def _reveal_answer(proxy: Proxy, line: bytes, certificates: Callable[[int], Mapping | None] | None = None) -> dict:
    """`serve_reveals`' answer to one request line."""
    try:
        req = json.loads(line)
        window, what = req["window"], req["kind"]
        if what != "certificates":
            what, reg = _KINDS[what], req["registration"]
            registration = Registration(reg["record"], reg["receipt"], reg["digest"], 0)
            request = bytes.fromhex(req["request"])
    except (ValueError, KeyError, TypeError):
        return {"refused": _MALFORMED}
    if what == "certificates":
        if certificates is None:
            return {"refused": "this certifier serves no certificates"}
        if not (type(window) is int and window >= 0):
            return {"refused": _MALFORMED}
        if (public := certificates(window)) is None:
            return {"refused": f"window {window}'s records aren't all certified, so its certificates aren't served"}
        return dict(public) | {"window": window}
    try:
        out = (proxy.reveal if what == SALT_KIND else proxy.committed)(window, registration, request)
    except (RuntimeError, ValueError) as e:
        return {"refused": str(e)}
    return {"window": out.window, "value": out.value.hex(), "signature": out.signature.hex(),
            "key_id": out.key_id.hex(), "run_id": out.run_id.hex()}


async def serve_reveals(proxy: Proxy, host: str, port: int, *,
                        certificates: Callable[[int], Mapping | None] | None = None) -> asyncio.Server:
    """The certifier's reveal endpoint, on `run`'s loop: it answers each JSON line a connection sends, `{"window",
    "kind": "salt" | "weights", "registration": {"record", "receipt", "digest"}, "request"}` (the challenger's signature
    in hex), with one JSON line, `{"window", "value", "signature", "key_id", "run_id"}` (hex), the `Revealed` of
    `proxy.reveal` or `proxy.committed`, or `{"refused": reason}` for whatever they refuse and for a line that isn't a
    request. The registration's `order` isn't sent: the certifier reads its record, receipt and digest. It serves only
    a proxy with a challenger key and a signer, and each call runs off the loop, since verifying and signing hash for
    milliseconds and the ticks have `slack`. With `certificates`, `{"window", "kind": "certificates"}`, which needs no
    signature, is answered `certificates(window)` with its `window`: the public part of the window's certificates, the
    caller's to keep (the module docstring's "The reveal over the network"), refused while it gives None, before every
    record of the window is certified. Returns the listening server: `close()` stops it."""
    if proxy.challenger_key is None:
        raise ValueError("a proxy serves reveals only with a challenger key and a signer")

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except ValueError:
                    while await reader.read(1 << 16):  # to the sender's end, so the refusal isn't lost to a reset
                        pass
                    writer.write(_line({"refused": f"a request is one line of at most {REVEAL_LINE_BYTES} bytes"}))
                    break
                if not line:
                    break
                writer.write(_line(await asyncio.to_thread(_reveal_answer, proxy, line, certificates)))
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port, limit=REVEAL_LINE_BYTES)


class RemoteCertifier:
    """The auditor challenger's handle on a network certifier's reveal endpoint (`serve_reveals`) at `host`:`port`, in
    place of the `Proxy` an in-process caller holds. `run_id` is that proxy's `Proxy.run_id`, which the auditor's
    deployment record gives beside `certifier_key`, as it gives the certifier `challenger_key` (not a signed hello,
    which would sign for any caller or need a signed request of its own). `reveal(p, registration)` and `committed(p,
    registration)` sign the request for `certifier_key`'s key id and `run_id` with `challenger_signer` (whose key the
    certifier registered as `challenger_key`), send it, and return the value once the answer names that key id and run
    and its signature verifies under `certifier_key` over `reveal_response_message` for ingress link `link` and p, so a
    consumer can key p on `certifier_run`, (`key_id`, `run_id`, `link`). A refusal, an answer of another run, one
    that doesn't verify or names another window, a signer that refuses and a connection that fails all raise
    `RuntimeError`, as a `Proxy`'s refusal does, so a caller that fails closed on one fails closed on them. Each call
    is one connection, one request line (then its end) and one answer line; `timeout` bounds its socket's waits
    (None: no bound)."""

    def __init__(self, host: str, port: int, certifier_key: PublicKey, challenger_signer: Signer, link: str,
                 run_id: bytes, *, timeout: float | None = None) -> None:
        if not (isinstance(run_id, bytes) and len(run_id) == RUN_ID_BYTES):
            raise ValueError(f"a certifier's run id is {RUN_ID_BYTES} bytes")
        self.host, self.port, self.certifier_key, self.challenger_signer = host, port, certifier_key, challenger_signer
        self.link, self.run_id, self.timeout = link, run_id, timeout

    @property
    def key_id(self) -> bytes:
        """The certifier's key id, `certifier_key`'s."""
        return self.certifier_key.key_id

    @property
    def certifier_run(self) -> tuple[bytes, bytes, str]:
        """The certifier run this client asks, (key id, run id, link): its proxy's `Proxy.certifier_run`, the identity
        every answer it takes binds and consumers key PoUW windows on."""
        return self.key_id, self.run_id, self.link

    def reveal(self, window: int, registration: Registration | None) -> bytes:
        return self._ask(window, registration, SALT_KIND)

    def committed(self, window: int, registration: Registration | None) -> bytes:
        return self._ask(window, registration, WEIGHTS_KIND)

    def _ask(self, window: int, registration: Registration | None, what: bytes) -> bytes:
        kind = what.decode()
        if not isinstance(registration, Registration):
            raise RuntimeError(f"no registration of window {window}, so its {kind} isn't asked for")
        body = {"window": window, "kind": kind, "registration": {
            "record": registration.record, "receipt": registration.receipt, "digest": registration.digest}}
        try:
            message = reveal_request_message(self.certifier_key.key_id, self.run_id, self.link, window,
                                             registration.digest, what)
            json.dumps(body)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"window {window}'s request can't be made: {e}") from None
        try:
            request = self.challenger_signer.sign(message)
        except SignerRefused as e:
            raise RuntimeError(f"the challenger's signer refused window {window}'s request: {e}") from None
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock, \
                    sock.makefile("rwb") as stream:
                stream.write(_line(body | {"request": request.hex()}))
                stream.flush()
                sock.shutdown(socket.SHUT_WR)
                answer = stream.readline(REVEAL_LINE_BYTES)
        except OSError as e:
            raise RuntimeError(f"the certifier at {self.host}:{self.port} gave no answer for window {window}'s "
                               f"{kind}: {e}") from None
        try:
            out = json.loads(answer)
            if isinstance(out, dict) and set(out) == {"refused"}:
                raise RuntimeError(f"the certifier refused window {window}'s {kind}: {out['refused']}")
            revealed = Revealed(out["window"], *(bytes.fromhex(out[f]) for f in ("value", "signature", "key_id",
                                                                                  "run_id")))
        except (ValueError, KeyError, TypeError):
            raise RuntimeError(f"the certifier's answer for window {window}'s {kind} isn't one") from None
        if revealed.run_id != self.run_id:
            raise RuntimeError(f"the answer for window {window}'s {kind} is another run's, not run "
                               f"{self.run_id.hex()[:16]}…'s")
        if revealed.window != window or not revealed.verify(self.certifier_key, self.run_id, self.link,
                                                            registration.digest, what):
            raise RuntimeError(f"the answer for window {window}'s {kind} isn't signed by the certifier's key")
        return revealed.value


def check_salts(audit: Audit, link: str, records: Sequence[Record], windows: int, salts: Mapping[int, bytes], *,
                pouw_window: int = 1, salt_bits: int = SALT_BITS) -> list[Verdict]:
    """The audit's read of the salt on ingress link `link` for certifier windows 0 … windows − 1, given the salts
    `reveal` gave by PoUW window of `pouw_window` certifier windows: PoUW window p, at its first certifier window
    w = p·`pouw_window`, is accepted if w has exactly one raw record, which holds exactly one salt frame of p, carrying
    `PLACEHOLDER`, as the first frame of its bucket 0 (slot 0, where `Bucketer` releases it), and p's revealed salt has
    `salt_bits` bits. Anything else, a missing or moved salt included, is a `salt` violation at w. The place is pinned
    because the served counts (`served.derive`) take bucket 0's first `own` frames for the salt: the same counts with
    the salt in a later bucket would derive other request ends. The verdicts join `audit`'s. A committed or certified
    record doesn't show its frames, so a proof opens the salt frame there, at bucket 0 slot 0."""
    by_window: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        if not isinstance(r, Record):
            raise ValueError("the salt check reads raw records")
        if r.link == link:
            by_window[r.window].append(r)
    out = []
    for w in range(0, windows, pouw_window):
        p = w // pouw_window
        found, salt, prefix = by_window.get(w, []), salts.get(p), p.to_bytes(8, "big")
        accepted = False
        if len(found) == 1 and salt is not None and 8 * len(salt) == salt_bits:
            grid = found[0].grid
            held = [x for row in grid for k, s, x in map(decode_frame, row[: row_count(row)])
                    if k == SALT and s == SALT_STREAM and x[:8] == prefix]
            first = decode_frame(grid[0][0]) if grid and row_count(grid[0]) else None
            accepted = held == [prefix + PLACEHOLDER] and first == (SALT, SALT_STREAM, prefix + PLACEHOLDER)
        out.append(Verdict(link, w, None if accepted else "salt"))
    audit.verdicts.extend(out)
    return out
