"""The active network certifier: the network certifier as a relay that retransmits everything on a fixed schedule, so
it rebuilds the framing of every connection it carries, and that releases a server's bytes only in the egress record's
one form.

The network certifier runs on an auditor host (`trusted-host`; co-located with the server for the demo, a named
assumption). It terminates each client's TCP connection, opens one to the server for it, and forwards each direction's
bytes unchanged in content or not at all: no tokenizer, API, request or stop string is the network certifier's. The one
addition is on the server's side (`release`): each HTTP request's place in the release order and its release bucket on
the server's copy, and, with `watermark`, a stream that tells the server each bucket's close. What the network certifier
decides is the framing and the timing, both public functions of the schedule (`WardenParams`), and it certifies every
cell it sends with the certifier's grid (`grid.Certifier`) and certificates (`certificate.Issuer`, in the runner).

**Cells.** Every cell is B = `cell_bytes` bytes, `active.encode_frame(kind, ordinal, payload)`: the header (kind u8,
the connection's ordinal u32, the payload's length u32), at most P = B − 9 payload bytes, zero fill. `DATA` carries
exactly P bytes, `END` the last 0 … P of a direction, `PAD` none (marked padding), `FAIL` none (the connection failed
closed, for whatever reason), and `REFUSED` none (a connection the network certifier doesn't take). Both directions are
cut the same way: byte k of a direction is in its cell ⌊k/P⌋, so a cell's content is fixed by the bytes and never by how
they were read. The client speaks cells both ways through a thin, workload-blind shim (`cells`, `Shim`); the server
gets the ingress payloads as bytes (co-located in the demo; deployed, its own shim strips the cells as the client's
does).

**Admission.** The k-th connection to arrive, from 0, has ordinal k + 1, as the active certifier's `Requests` streams
it: ordinal 0 is no connection's. At most `lanes` connections are served at once; one that arrives with every lane
taken waits in a FIFO queue of at most `queue`, else it gets one `REFUSED` cell, on ordinal 0, and is closed.
The queue's head is admitted at the first bucket with a free lane, and holds that lane, r cells a bucket each way,
until its last bucket. Nothing admitted is dropped or reordered. A refusal is traffic past what the window's Programs
can hold (`every`: a slot stride of `every` buckets has `program_slots` = `lanes` Program slots, the scan's
`ingress_scan.Class`), so the network certifier certifies no record of a window in which it refused a connection:
`withhold` gets that window's three records in place of `publish`, and the period check refuses an execution phase
missing any record's certificate. Under the lengths rows' form 2 the queue's head is admitted only at a bucket from
which it ends inside the run, and an arrival past the last such bucket is turned away, which is no refusal ("The
lengths rows").

**The schedule.** Connection q, admitted at bucket s on the network certifier's own schedule, carries r = `slots` cells
a bucket each way from s on:
- ingress: the client's bytes as `DATA` cells, as soon as a whole cell's bytes have arrived, then `END` once the
  client's `END` cell (or its end of stream) has arrived, `PAD` in a slot with neither; the bucket of the `END` is a,
  and ingress has no cells after it. Ingress must end within `ingress_buckets` buckets and `ingress_cells` cells. Its
  release never waits on the server: the server gets a connection's ingress whole once its `END` is released, as
  `release.with_release` puts it, N the count of `END`s the network certifier released before (so the order of the `END`
  cells in the ingress records), the run the network certifier's `run_id` and the bucket a; what it releases before
  then, or before the server's connection opens, is held for the server (at most the `ingress_cells` cells' bytes the
  ingress bound allows; past them the connection fails closed, and the server gets none of it);
- egress: `PAD` in every slot of the buckets before s + D, D = `delay`; then egress cell i is in bucket
  s + D + ⌊i/r⌋, slot i mod r: `DATA` with the server's bytes iP … (i+1)P − 1 if it has sent them, else `END` with its
  remaining 0 … P bytes if it has ended, else `FAIL` (late): the cell's deadline passed. The rest of the bucket of
  an `END` or a `FAIL` is `PAD`, and the connection ends with that bucket. Egress holds at most `egress_cells` cells.
Egress is based on the admission bucket s, never on the ingress's end, the server's connect or its response: nothing
the server does moves s + D. So the egress cells on a connection, their sizes and their buckets are a function of (s,
the number of egress cells, the failure's cell) and the public parameters alone, s of the arrivals and the admission
rule; the ingress cells are a function of s and the client's bytes.

**The egress form** (`WardenParams.egress`). `id`, the default, is the egress record (`PROTOCOL.md`'s "The egress
record"): the network certifier releases a `DATA` or `END` cell it cut only if `io_tie.parse_cell` reads it, so only an
`io_tie.data_cell` or `io_tie.end_cell` on the connection's ordinal leaves, and it fails the connection (format) at any
other cell, refusing and never rewriting. An honest id-form server writes one `io_tie.id_event` of exactly P bytes an
output and then `io_tie.TRAILER`, so the cut gives one event a `DATA` cell and an `END` holding the trailer; any other
bytes (an HTTP head, a response id, an event of another size, an error body) fail at their first cell. `bytes` relays
any server's bytes in cells: a rehearsal's form, for a server that answers in another (vLLM's stock framing, say). Its
parameters' digest is no id form's, and the period check reads only the id form's schedules.

**The response cap** (`WardenParams.max_tokens_cap`, none by default): a schedule may register a cap on every response,
in the id egress form, where a response's tokens are its `DATA` cells, one id event each. The network certifier then
reads each request's `max_tokens` from its first request cell, where `active.id_request` puts it
(`active.id_request_max_tokens`), before cutting that cell. A request whose field isn't there as a u32 or exceeds the cap
is refused (cap), never clamped: none of its bytes is cut into a cell or reaches the server, its ingress is padding, and
it fails at once, as the client's other problems do. Every other request's response is held to its `max_tokens`: an
egress slot past that many `DATA` cells, with a whole cell waiting, fails (limit), so each response the client receives
holds at most its request's `max_tokens` tokens, hence at most the cap (`NetTiming.Guarantees.ResponseCapped`). The
cap and its reason enter the parameters' digest, so every record head binds them; a schedule without a cap hashes
neither and keeps its digest. The period check reads the cap with the schedule and publishes it (`max_tokens_cap`).

**The lengths rows** (`WardenParams.lengths`, none by default; 1 and 2 are their forms, `LENGTHS_FORMS`): a schedule
may have its network certifier keep, of each request it releases, its place in the release order, its request's
bytes, its response's outputs and whether it failed, a row of K = `WardenParams.stride_releases` slots a slot stride.
Such a schedule has the id egress form, `ingress_buckets` ≤ D, a cell that holds an id request's head and first prompt
field (`io_request.layout`), at most 2^16 egress cells, D + ⌊(`egress_cells` − 1)/r⌋ ≤ `window`, so that a connection
ends within a window of its admission, and a K one proof block returns a row of (`connection_rows.check_slots`);
`WardenParams` takes no other. The form enters the parameters' digest; a schedule without lengths rows hashes nothing
of it and keeps its digest, and its network certifier does nothing of this section. With lengths rows the network
certifier:
- releases no connection whose client has failed: once a client's problem is set (ingress, malformed, cap) it cuts no
  further request cell of it, so the connection has no `END`, no place in the release order and nothing for the
  server. Every released connection therefore ends at an egress slot, in its `END` or a server's `FAIL`, and holds its
  lane from its release through its admission's bucket + D;
- fails such a connection at its first egress slot, in bucket s + D, and never before: until then both its directions
  are padding. So every connection it admits, released or not, holds its lane through its admission's bucket + D at
  least, the bucket through which the ingress proof block's scan holds the connection's slot (`ingress_scan.held`),
  and a lane it frees is a slot the scan has freed: a client's failure leaves the next admission a slot;
- releases only canonical id requests: it cuts a request cell only where the ingress proof block's reference reads it
  as the next cell of an id request on the connection (`io_request.read`: its head and first prompt field in the first
  cell, the length its head says, its `END` at its last byte). A request that is none is cut up to the first cell that
  doesn't read so and then fails as its client's (malformed), unreleased: no server sees it, so no server's failure
  follows from it;
- hands each window's releases (`Released`, to `Warden(lengths=…)`) by the slot stride of each one's `END` and in
  release order, at most K a stride. Window w's are handed once window w + 1's records are published, where each of
  their connections has ended, and only where both windows' three records were published complete: a window whose
  cells have no certificate, or whose responses may reach into one that has none, has no lengths rows, and neither has
  the run's last window. `connection_rows` lays them out as the lengths and fails tables, commits them and signs
  their one head. That is form 1, which says nothing of an execution phase it doesn't hand.

Form 2 ends every execution phase of a run (a window here), once and in their order: in its hand, or in the same
call without releases, on which the network certifier signs that the execution phase has no head
(`connection_rows`' "The word that no head comes"). Under it the network certifier:
- hands execution phase w where the one before it (if there is one), w itself and the one after it (if there is one)
  were each published complete, their three records, none withheld. Every cell of a release of w is in those three:
  its request's cells begin at most `ingress_buckets` − 1 buckets before its `END`, and its connection ends at most
  `WardenParams.span` buckets after its admission, each at most a window. So no hand holds a byte that left in an
  execution phase without a certificate, which form 1's can: a request whose first cells left in the one before;
- ends execution phase w once execution phase w + 1's records are published, and the run's last with its own
  records, so the run's last close ends two;
- admits a connection only at a bucket s with s + `span` inside the run, so that every release of the run's last
  execution phase has ended by its close: that execution phase admits in its first T − `span` buckets and no later.
  A connection still queued after them waits and is cut at the run's end with no cell, as one still queued then
  always is;
- turns away an arrival it handles once its ticks have reached the first bucket that admits nothing. The arrival
  takes no ordinal, is never queued and is no refusal, so no record is withheld for it; `turned_away` counts it. It
  gets one `REFUSED` cell on ordinal 0, the same bytes at every turn-away, and its stream's end with it. They leave
  with the next tick's bucket as that bucket's cells do, never at the arrival: written on the loop by that tick, or
  with `paced` handed to the pacer as a part of that bucket for the arrival's socket, the link's last, which ends
  the stream. The cell is in no record. The link stays open after: what the client still sends is read and dropped,
  none of it reaching a server, and the link is closed at the client's end of stream, or at the run's end. A link
  closed with its client's bytes unread, as a request it never read is, is reset, and a reset client may lose the
  cell. Which arrivals are turned away is a function of the tick count at each one's arrival, the run's length
  and the schedule: nothing the queue or the lanes hold, and so nothing a server does, moves it. An arrival after
  the run's last tick has no tick left and is closed with no cell. Before those buckets a full queue refuses as it
  does (Admission), and that execution phase's records are withheld;
- ends each execution phase it has not ended without releases when the run ends early, whatever ended it.

**The content rows** (`WardenParams.content`, none by default: the classes of a schedule with lengths rows, each
(P, O, rows) under form 1 and (P, O) under form 2, `content_rows`' "The classes"): the network certifier also keeps,
of each request it releases, the request's bytes as it cut them into cells and the id of each `DATA` cell it released,
four bytes an id, and hands them with the release (`Released.request`, `Released.ids`), so that `connection_rows` lays
three more tables a class under the same head. Under form 2 those tables hold a row a release filed in the class and
no other, at most `WardenParams.phase_releases`, a bound on an execution phase's releases. The hand copies an
execution phase's ids and makes nothing else: the tables are laid, committed and signed by whoever takes the hand,
never on the tick.
Both are read off its own cells and nothing the server says. It keeps them from the release to the hand that carries
it, or to where that hand would have been for a release it never hands (the lengths rows' hand, under either form),
and no longer; of a connection it never released it keeps nothing. The classes enter the parameters' digest; a
schedule without content rows hashes nothing of them and keeps its digest, and its network certifier keeps and hands
what it did.

**Fail closed.** A connection fails at the first of: an egress cell whose bytes aren't there at its deadline (late),
a server that hasn't ended by the last of `egress_cells` cells or sends past its request's `max_tokens` under a response
cap (limit), the server's connection failing (upstream), an egress cell out of the egress form (format), ingress past
its bounds (ingress), a client cell of another kind than `DATA` or `END` (malformed), a request refused under the
response cap (cap). Its `FAIL` cell is in the slot of the next egress cell, and it is in the egress record.
Every `FAIL` is the same bytes: the reason is a frame of the network certifier's `fails` record (`reason_frame`),
witness the client never sees. The server's four reasons show only at an egress slot, so the server, which controls
whether and where a connection fails, chooses one of at most (cells + 1) outcomes; the client's three fail at once
(with lengths rows at the connection's first egress slot), and are the client's own doing. A tick that runs more than `slack` past its boundary faults every grid's window
(`monitor-unavailable`), as the active certifier's do.

**The pacer** (`Warden(paced=…)`, off by default): a bucket written on the loop leaves when its tick runs, and the
loop reads every server, so when a server writes within a bucket, which the prover chooses, moves when the bucket
reaches the clients. With `paced`, seconds past `slack`, each tick hands its bucket, every live connection's cells as a
part for that connection's socket, to a process of its own (`pacer.Pacer`), which releases it at `origin` + t·τ +
`paced` on the host's monotonic clock, a time fixed before the bucket is made; each connection's socket reaches the
pacer at its admission, and its last bucket's part closes the pacer's copy. A bucket that reached the pacer after its
time, or whose release started more than `slack` after it, faults every grid's window `monitor-unavailable`, as a late
tick does. A window's three records are published once every bucket of it that was handed is released, or by the
next window's first tick, faulted `monitor-unavailable` if one isn't by then. A client that reads slowly holds only its
own bytes in the pacer, and one more than `WRITE_BACKLOG` behind is closed, as on the loop.

**The watermark** (`Warden(watermark=…)`, off by default): one connection to the server, opened by `watermark` when the
run starts, on which the network certifier sends `release.watermark_head` and then a line for each tick t,
`release.watermark_chunk(t, n, lost)`, n the `END`s released through bucket t. Tick t makes the line, and it is sent,
in order, once each release of bucket t has been written whole to its server connection with that transport's buffer
emptied, or is lost, its connection closing, failed or ended first; the line names the lost ones. A release still on
its way (its connection opening, its bytes buffered) holds its line and the later ones until it arrives or its
connection ends. The run's end sends the lines still held, then the terminating chunk. The network certifier never
reads that connection and no tick waits on it: the lines sent before it opens are kept for it, and once more than
`WRITE_BACKLOG` of them are unwritten it is closed. It is the one connection to the server the network certifier opens
with no admission: every other is opened by `connect` when the client's connection is admitted. Each line's t and n
are functions of the ingress record, its lost releases the server connections' state, and no cell or record changes.

**The records.** Each direction's grid has r·`lanes` slots a bucket of B-byte frames, and its frames are exactly the
cells the network certifier sent that way: every connection's r cells of the bucket, in ordinal order, padding included.
So the records are the wire, and each window's frames root (`certificate.frames_root`) attests every byte on it, but
for the one cell that is on no connection, a refusal's or a turn-away's `REFUSED`: a refusal's window has no
certificate, and a turn-away's is one constant in the bucket its arrival sets. The
third, `fails`, has `lanes` slots a bucket and one frame a failed connection, its reason, in the bucket of its `FAIL`
(`WardenParams.fails_grid`). `audit` is the read of the records in the clear: every connection's cells are the
schedule's and the egress form's, every `FAIL` has its one reason, and no connection failed (with lengths rows, none
that was released).

`PROTOCOL.md`'s "The active network certifier" has the leak bound. The module imports only the standard library and
`verity`.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import secrets
import time
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property
from typing import Any, NamedTuple

from verity.commitments.identity import canonical_json_bytes, tagged_sha512
from verity.protocols.certifier.verifier.active import (DATA, HEADER, ID_DIGITS, MAX_TOKENS_AT, PAD, RUN_ID_BYTES,
                                                   decode_frame, encode_frame, id_request_max_tokens, precise_sleep)
from verity.protocols.certifier.verifier.grid import Certifier, GridParams, Reason, Record, Row, Status, row_count
from verity.protocols.certifier.verifier.io_tie import parse_cell
from verity.protocols.certifier.verifier.pacer import Pacer, Report
from verity.protocols.certifier.verifier.release import WATERMARK_END, watermark_chunk, watermark_head, with_release

PARAMETERS_TAG = "verity-network-warden/parameters/v3"
#: the network certifier's cell kinds, beside `active`'s PAD and DATA
END, FAIL, REFUSED = 0x20, 0x21, 0x22
KIND_NAMES = {PAD: "pad", DATA: "data", END: "end", FAIL: "fail", REFUSED: "refused"}
#: why a connection failed: the server's three, the client's two, the server's out-of-form cell, then the client's
#: request past the response cap (a schedule without one has the first six, `WardenParams.reasons`); only the `fails`
#: record holds them
REASONS = ("late", "limit", "upstream", "ingress", "malformed", "format", "cap")
#: the reasons a client's own bytes give; the rest are the server's
CLIENT_REASONS = ("ingress", "malformed", "cap")
#: what egress releases (the module docstring's egress form): the egress record's id-form cells, or any bytes
EGRESS_FORMS = ("id", "bytes")
#: the bytes the network certifier reads from a server at once
READ_BYTES = 1 << 16
#: the most a client connection may hold unwritten before the network certifier stops writing to it
WRITE_BACKLOG = 1 << 22
#: the pacer's key of a run's n-th turned-away arrival, from 1, is `AWAY_KEYS` − n: its keys are u32, and a
#: connection's is its ordinal, which counts up from 1
AWAY_KEYS = 1 << 32


@dataclass(frozen=True)
class WardenParams:
    """The schedule's public parameters, fixed before the run (`WardenParams.digest` binds them)."""

    cell_bytes: int = 256  # B
    slots: int = 4  # r: a connection's cells a bucket, each way
    bucket_ms: int = 100  # τ
    delay: int = 10  # D: buckets from admission to the first egress cell's
    lanes: int = 16  # connections served at once
    queue: int = 64  # connections waiting for a lane, past which REFUSED
    window: int = 600  # T: buckets a record covers
    ingress_cells: int = 64
    ingress_buckets: int = 50
    egress_cells: int = 4096
    every: int = 1  # a slot stride's buckets: the Programs' boundaries are every `every` buckets of a window
    egress: str = "id"  # the egress form, one of `EGRESS_FORMS`
    max_tokens_cap: int | None = None  # the response cap (the module docstring's), or none
    lengths: int | None = None  # the lengths rows' form (the module docstring's "The lengths rows"): 1 or 2, or none
    content: tuple[tuple[int, ...], ...] | None = None  # the content rows' classes (`content_rows.classes`), or none


    def __post_init__(self) -> None:
        for name in INTEGER_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} is a positive integer, not {value!r}")
        if self.egress not in EGRESS_FORMS:
            raise ValueError(f"egress is one of {', '.join(EGRESS_FORMS)}, not {self.egress!r}")
        if self.cell_bytes <= HEADER.size + 1:
            raise ValueError(f"a cell holds its {HEADER.size}-byte header and a byte")
        if self.window % self.every:
            raise ValueError(f"every ({self.every}) divides the window ({self.window}): a window is whole timesteps")
        cap = self.max_tokens_cap
        if cap is not None:
            if type(cap) is not int or not 0 < cap < min(1 << 32, self.egress_cells):
                raise ValueError(f"max_tokens_cap is a u32 from 1 to egress_cells − 1 ({self.egress_cells - 1}), "
                                 f"not {cap!r}: a capped response's END is its egress cell max_tokens")
            if self.egress != "id":
                raise ValueError("a response cap counts id events, so its egress form is id")
            if self.payload < MAX_TOKENS_AT + ID_DIGITS:
                raise ValueError(f"a {self.cell_bytes}-byte cell doesn't hold an id request's max_tokens: "
                                 f"under a response cap a cell is at least {HEADER.size + MAX_TOKENS_AT + ID_DIGITS} bytes")
        if self.lengths is not None:
            if type(self.lengths) is not int or self.lengths not in LENGTHS_FORMS:
                raise ValueError(f"lengths is 1 or 2, the lengths rows' forms, or unset, not {self.lengths!r}")
            if self.egress != "id":
                raise ValueError("the lengths rows count id events, so their egress form is id")
            if self.ingress_buckets > self.delay:
                raise ValueError(f"with lengths rows ingress_buckets ({self.ingress_buckets}) is at most the delay "
                                 f"({self.delay}): a released connection holds its lane from its release")
            if self.span > self.window or self.egress_cells > 1 << 16:
                raise ValueError(f"with lengths rows a connection ends within a window of its admission and its "
                                 f"egress cells count in a u16: delay + (egress_cells − 1) // slots ({self.span}) is "
                                 f"at most the window ({self.window}) and egress_cells at most {1 << 16}")
            _request_form().layout(self.cell_bytes)
            _connection_rows().check_slots(self.stride_releases, self._stride_that_fits)
        if self.content is not None:
            object.__setattr__(self, "content", _content_rows().classes(self))

    def _stride_that_fits(self, most: int) -> str:
        """What a schedule whose slot stride releases more than `most` requests changes: its longest stride that
        releases no more, the other parameters as they are."""
        fits = [E for E in range(1, self.every) if self.window % E == 0
                and _stride_releases(self.lanes, E, self.delay, self.ingress_buckets) <= most]
        return (f"this schedule's slot stride of {self.every} buckets releases up to {self.stride_releases} requests, "
                f"a slot each: " + (f"`every` is at most {fits[-1]} with its lanes, delay and ingress_buckets" if fits
                                    else f"no stride of {self.lanes} lanes releases as few, so at most {most} lanes"))

    @property
    def program_slots(self) -> int:
        """A slot stride's Program slots, one a lane (`ingress_scan.Class.SLOTS`, for a class that serves the
        network certifier)."""
        return self.lanes

    @property
    def window_programs(self) -> int:
        """A window's Program slots: `window / every` slot strides of `program_slots`."""
        return self.window // self.every * self.program_slots

    @property
    def stride_releases(self) -> int:
        """K: the most requests the network certifier releases in one slot stride with lengths rows, a lengths row's
        slots. A released connection holds its lane from its release through its admission's bucket + D, at least
        L = D + 2 − `ingress_buckets` buckets, so any L buckets hold at most `lanes` releases; and it holds it D + 1
        buckets around its release, so a stride's releases are live at one of 1 + ⌈(`every` − 1) / (D + 1)⌉ buckets.
        With lengths rows it is at most a row's slots (`connection_rows.check_slots`)."""
        return _stride_releases(self.lanes, self.every, self.delay, self.ingress_buckets)

    @property
    def phase_releases(self) -> int:
        """R: a bound on the requests the network certifier releases in one execution phase with lengths rows,
        `lanes` · ⌈(T + `ingress_buckets` − 1) / (D + 1)⌉. A release is at most `ingress_buckets` − 1 buckets after
        its connection's admission, so an execution phase's releases are of connections admitted in T +
        `ingress_buckets` − 1 buckets in a row; and a connection holds its lane through its admission's bucket + D,
        so the connections admitted in any D + 1 buckets in a row are live together, at most `lanes` of them. It is
        a bound: at some schedules a load releases R in an execution phase, and at others none does."""
        return self.lanes * -(-(self.window + self.ingress_buckets - 1) // (self.delay + 1))

    @property
    def span(self) -> int:
        """The most buckets from a connection's admission to its last egress cell's: D + ⌊(`egress_cells` − 1) / r⌋
        (the module docstring's schedule). With lengths rows a connection ends by then, and it is at most `window`."""
        return self.delay + (self.egress_cells - 1) // self.slots

    @property
    def closes(self) -> bool:
        """Whether the lengths rows end every execution phase of a run, its last included: their form 2."""
        return self.lengths == 2

    @property
    def reasons(self) -> tuple[str, ...]:
        """The `fails` record's reasons under this schedule: `REASONS`, without `cap` when it has no response cap."""
        return REASONS if self.max_tokens_cap is not None else REASONS[:-1]

    @property
    def payload(self) -> int:
        """P: the bytes a `DATA` cell carries."""
        return self.cell_bytes - HEADER.size

    @property
    def grid(self) -> GridParams:
        """Each direction's record: r·lanes cells a bucket of B bytes."""
        return GridParams(self.window, self.lanes * self.slots, self.cell_bytes, self.bucket_ms)

    @property
    def fails_grid(self) -> GridParams:
        """The `fails` record: at most one reason a connection, so `lanes` a bucket."""
        return GridParams(self.window, self.lanes, self.cell_bytes, self.bucket_ms)

    @cached_property
    def digest(self) -> bytes:
        """The 64 bytes each record head's `parameters` carries. A schedule without a response cap has neither the
        cap nor its reason in what it hashes, one without lengths rows not their form and one without content rows
        not their classes, so its digest is the one it had before any of them was a parameter."""
        values = {k: v for k, v in asdict(self).items() if k not in OPTIONAL_FIELDS or v is not None}
        return tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(values | {"kinds": sorted(KIND_NAMES),
                                                                            "reasons": list(self.reasons)}))


#: `WardenParams`' fields that are unset by default, each left out of the digest of a schedule that leaves it unset
OPTIONAL_FIELDS = ("max_tokens_cap", "lengths", "content")
#: the lengths rows' forms (the module docstring's "The lengths rows"): which execution phases are handed, and whether
#: one that isn't is ended without releases
LENGTHS_FORMS = (1, 2)
#: `WardenParams`' integer fields, every one but `egress` and the optional ones: the schedule the period check reads
#: (`scheduleOf`) beside its `max_tokens_cap` when it has one, whose egress form is the id form and which has no lengths
#: rows
INTEGER_FIELDS = tuple(f.name for f in fields(WardenParams) if f.name not in ("egress", *OPTIONAL_FIELDS))


def _stride_releases(lanes: int, every: int, delay: int, ingress_buckets: int) -> int:
    """`WardenParams.stride_releases` of a schedule with these parameters."""
    return lanes * min(-(-every // (delay + 2 - ingress_buckets)), 1 + -(-(every - 1) // (delay + 1)))


def _request_form():
    """`io_request`, whose reading of a request cell (`io_request.read`) a schedule with lengths rows holds every
    request to. Imported at its first use: `io_request` imports this module."""
    from verity.protocols.certifier.verifier import io_request
    return io_request


def _connection_rows():
    """`connection_rows`, whose bound on a lengths row's slots (`connection_rows.check_slots`) a schedule with lengths
    rows holds its slot stride's releases to. Imported at its first use: `connection_rows` imports this module."""
    from verity.protocols.certifier.verifier import connection_rows
    return connection_rows


def _content_rows():
    """`content_rows`, whose reading of a schedule's classes (`content_rows.classes`) a schedule with content rows
    holds them to. Imported at its first use: `content_rows` imports this module."""
    from verity.protocols.certifier.verifier import content_rows
    return content_rows


def _id_request_cell(params: WardenParams, c: _Connection, made: bytes) -> bool:
    """With lengths rows: whether `made`, the next cell cut from c's request, is that cell of a canonical id request
    as the ingress proof block reads it (`io_request.read`, against the bytes its request has left), which it then
    counts as cut. It reads the schedule and the connection, nothing of the device: `Warden._ingress` cuts the same
    cells for a caller that stands in for the device with its parameters alone."""
    form = _request_form()
    got = form.read(form.layout(params.cell_bytes), made, c.ordinal, c.left)
    if not isinstance(got, form.Frame):
        return False
    c.left = (c.left or got.length) - got.used
    return True


def cell(kind: int, ordinal: int, payload: bytes, cell_bytes: int) -> bytes:
    return encode_frame(kind, ordinal, payload, cell_bytes)


def read_cell(data: bytes, cell_bytes: int) -> tuple[int, int, bytes]:
    """(kind, ordinal, payload) of a cell the network certifier makes, or ValueError: its kind, length and zero fill
    exact."""
    if len(data) != cell_bytes:
        raise ValueError(f"a cell is {cell_bytes} bytes, not {len(data)}")
    kind, ordinal, payload = decode_frame(data)
    size = len(payload)
    if kind not in KIND_NAMES or HEADER.unpack_from(data)[2] != size or \
            cell(kind, ordinal, payload, cell_bytes) != data:
        raise ValueError("not a warden cell")
    if (kind == DATA and size != cell_bytes - HEADER.size) or (kind in (PAD, FAIL, REFUSED) and size):
        raise ValueError(f"a {KIND_NAMES[kind]} cell of {size} bytes")
    return kind, ordinal, payload


def reason_frame(ordinal: int, reason: str, cell_bytes: int) -> bytes:
    """A frame of the `fails` record: why connection `ordinal` failed, as one byte, its index in `REASONS`."""
    return encode_frame(FAIL, ordinal, bytes([REASONS.index(reason)]), cell_bytes)


def read_reason(data: bytes, cell_bytes: int) -> tuple[int, str]:
    """(ordinal, reason) of a `reason_frame`, or ValueError."""
    if len(data) != cell_bytes:
        raise ValueError(f"a reason is {cell_bytes} bytes, not {len(data)}")
    kind, ordinal, payload = decode_frame(data)
    if kind != FAIL or len(payload) != 1 or payload[0] >= len(REASONS) or \
            reason_frame(ordinal, REASONS[payload[0]], cell_bytes) != data:
        raise ValueError("not a reason frame")
    return ordinal, REASONS[payload[0]]


def cells(data: bytes, cell_bytes: int, ordinal: int = 0) -> bytes:
    """A direction's bytes as cells: `DATA` cells of P bytes, then one `END` with the rest (the client shim's sending
    half, and the network certifier's own cut)."""
    p = cell_bytes - HEADER.size
    whole = (len(data) - 1) // p if data else 0
    out = [cell(DATA, ordinal, data[i * p:(i + 1) * p], cell_bytes) for i in range(whole)]
    return b"".join(out) + cell(END, ordinal, data[whole * p:], cell_bytes)


@dataclass
class Shim:
    """The client's thin shim, reading half: every cell the network certifier sends on one connection, until it closes,
    each kept with the time it arrived; `data` is the `DATA` and `END` payloads, `outcome` "end", "fail" or "refused",
    or "cut" when the connection closed before any of those, or inside a cell."""

    cell_bytes: int
    clock: Callable[[], float] = time.monotonic
    data: bytearray = field(default_factory=bytearray)
    arrivals: list[tuple[float, int]] = field(default_factory=list)
    outcome: str | None = None
    ordinal: int | None = None

    async def read(self, reader: asyncio.StreamReader) -> Shim:
        while True:
            try:
                raw = await reader.readexactly(self.cell_bytes)
            except asyncio.IncompleteReadError as e:
                if e.partial or self.outcome is None:
                    self.outcome = "cut"
                return self
            kind, ordinal, payload = read_cell(raw, self.cell_bytes)
            self.arrivals.append((self.clock(), kind))
            self.ordinal = ordinal
            if self.outcome is not None:
                if kind != PAD:
                    self.outcome = "cut"
                continue
            if kind in (DATA, END):
                self.data += payload
            if kind in (END, FAIL, REFUSED):
                self.outcome = KIND_NAMES[kind]


@dataclass(eq=False)
class _Connection:
    ordinal: int
    reader: asyncio.StreamReader
    writer: Any
    done: asyncio.Future
    arrived: int
    admitted: int | None = None
    pad: bytes = b""  # its PAD cell, made once (`_pad`): a window's grids hold one a connection, not one a slot
    server: Any = None
    ingress: bytearray = field(default_factory=bytearray)  # received, not yet released
    held: bytearray = field(default_factory=bytearray)  # released, held for the server until the END's release
    unsent: bytearray = field(default_factory=bytearray)  # released whole before the server's connection opened
    release: int | None = None  # N: its END's place in the network certifier's release order
    wrote: bool = False  # its release was written whole to its server connection, open
    written: bool | None = None  # with the watermark: its release reached the server's transport (True) or was lost
    ingress_ended: bool = False  # the client's END has been received
    ingress_end: int | None = None  # a: the bucket the network certifier released ingress's END
    ingress_cells: int = 0
    egress: bytearray = field(default_factory=bytearray)  # the server's bytes, not yet released
    egress_ended: bool = False
    egress_cells: int = 0  # egress slots past s + D, released
    problem: str | None = None  # the client's: failed at the next slot
    max_tokens: int | None = None  # under a response cap, its request's, read at its first request cell
    refused: bool = False  # under a response cap, its request was refused (cap): no request cell is cut
    left: int = 0  # with lengths rows, its request's bytes still to be cut (`io_request.read`'s carry)
    upstream: str | None = None  # the server connection's: failed at the first egress slot without a whole cell
    failed: str | None = None  # one of REASONS
    failed_at: int | None = None  # the egress cell of the FAIL
    ended: int | None = None  # the bucket of the connection's last cells
    bytes_in: int = 0
    bytes_out: int = 0
    request: bytes = b""  # with content rows, its request's bytes as cut, from its release to its hand
    ids: bytearray = field(default_factory=bytearray)  # with content rows, each released DATA cell's id, a u32
    tasks: list[asyncio.Task] = field(default_factory=list)


class Released(NamedTuple):
    """One released request as the lengths rows hold it (the module docstring's "The lengths rows"): `order`, N, its
    place in the release order; `request_bytes`, its request's; `outputs`, n, its response's `DATA` cells before its
    `END` or `FAIL`; and `failed`, whether it ended in a `FAIL`. With content rows (the module docstring's "The
    content rows") also `request`, those bytes as the network certifier cut them, and `ids`, each of those cells' id
    in order, a little-endian u32 each (`content_rows.ids_of` reads them); without them both are empty."""

    order: int
    request_bytes: int
    outputs: int
    failed: bool
    request: bytes = b""
    ids: bytes = b""


def _pad(c: _Connection, cell_bytes: int) -> bytes:
    """c's PAD cell, made at its first use and kept, however c was admitted."""
    if not c.pad:
        c.pad = cell(PAD, c.ordinal, b"", cell_bytes)
    return c.pad


class Warden:
    """The active network certifier (the module docstring): `handle(reader, writer)` serves one client connection,
    `run()` ticks the schedule from `origin`, the clock's time of bucket 0, for `windows` record windows, then cuts
    whatever is still served or queued. `connect()` opens a server connection, a (reader, writer) pair.

    - `publish(record, params)` receives each window's records, ingress, egress, then fails, as they close (by
      default they are kept in `records`), and `withhold(record, params)` in its place those of a window in which a
      connection was refused (`refused_windows`; by default kept in `withheld`): nothing of that window is to be
      certified; `log` has one entry a connection once it ends, `refused` the ordinal-less refusals' count,
      `turned_away` that of the arrivals form 2 turned away in the run's last buckets, which are no refusals,
      `lateness` each tick's in seconds;
    - `release(link, row)`, when given, receives each grid's row as it ticks, ingress, egress, then fails, before
      that tick's `publish`: a publisher in another process builds each record from them as it fills, so that the
      tick never hands it a whole window at once;
    - `requests_released` counts the `END`s released, each connection's N its count before (`log`'s `release`, its
      bucket `ingress_end`), and `run_id`, `RUN_ID_BYTES` sampled from `run_nonce` at construction, names the run on
      the server's side;
    - with lengths rows (`WardenParams.lengths`; the module docstring's "The lengths rows"), `lengths(window,
      strides)` receives each window's releases by slot stride, a window later (by default they are kept in
      `lengths`); under form 2 it is called once for every execution phase of the run, in their order, the last one
      at the run's last close, with `strides` None for an execution phase that is not handed;
    - with `watermark`, a connect like `connect` (off by default), the watermark stream (the module docstring's "The
      watermark"): `watermarked` is the last line's (t, n), `watermark_lost` the releases its lines named lost, and
      `watermark_error` why the stream stopped, if it did;
    - with `paced`, seconds past `slack` (off by default), each tick hands its bucket to the pacer (`pacer.Pacer`;
      the module docstring's "The pacer"), which releases it `paced` after its boundary on the host's monotonic clock,
      `origin`'s clock then, to each connection's socket (`writer.get_extra_info("socket")`); `releases` holds each
      bucket's `pacer.Report` once its writes end;
    - `until(predicate)` waits until the warden's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, params: WardenParams, connect: Callable[[], Awaitable[tuple[Any, Any]]], *, origin: float,
                 windows: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = precise_sleep,
                 publish: Callable[[Record, GridParams], None] | None = None,
                 withhold: Callable[[Record, GridParams], None] | None = None,
                 release: Callable[[str, Row], None] | None = None, slack: float = 0.025,
                 egress_link: str = "responses", ingress_link: str = "requests", fails_link: str = "fails",
                 run_nonce: Callable[[int], bytes] = secrets.token_bytes, paced: float | None = None,
                 watermark: Callable[[], Awaitable[tuple[Any, Any]]] | None = None,
                 lengths: Callable[[int, tuple[tuple[Released, ...], ...]], None] | None = None) -> None:
        if paced is not None and not (isinstance(paced, (int, float)) and slack < paced < params.bucket_ms / 1000):
            raise ValueError(f"a paced bucket leaves past the slack and before the next boundary, so that a tick in "
                             f"time hands it before its time, not {paced!r} s after its boundary")
        self.paced = paced
        self.releases: list[Report] = []
        self._pacer: Pacer | None = None
        self._unreleased: Counter[int] = Counter()
        self._held: dict[int, tuple[Callable[[Record, GridParams], None], list[tuple[Record, GridParams]]]] = {}
        self.run_id = run_nonce(RUN_ID_BYTES)
        if not (isinstance(self.run_id, bytes) and len(self.run_id) == RUN_ID_BYTES):
            raise ValueError(f"a run id is {RUN_ID_BYTES} bytes")
        self.requests_released = 0
        self.watermark = watermark
        self.watermarked: tuple[int, int] | None = None
        self.watermark_lost: list[int] = []
        self.watermark_error: str | None = None
        self._watermark_writer: Any = None
        self._watermark_unsent = bytearray()
        self._watermark_lines: deque[tuple[int, int, list[_Connection]]] = deque()
        self.params, self.connect = params, connect
        self.origin, self.windows, self.clock, self.sleep, self.slack = origin, windows, clock, sleep, slack
        self.egress = Certifier(egress_link, params.grid)
        self.ingress = Certifier(ingress_link, params.grid)
        self.fails = Certifier(fails_link, params.fails_grid)
        self.records: list[Record] = []
        self.publish = publish or (lambda record, grid: self.records.append(record))
        self.withheld: list[Record] = []
        self.withhold = withhold or (lambda record, grid: self.withheld.append(record))
        self.release = release
        self.lengths: list[tuple[int, tuple[tuple[Released, ...], ...] | None]] = []
        # `lengths` as it is called: under form 2 also with no strides, for a window that is not handed
        self.hand_lengths: Callable[[int, tuple[tuple[Released, ...], ...] | None], None] = (
            lengths or (lambda window, strides: self.lengths.append((window, strides))))
        self._released: dict[int, list[list[_Connection]]] = {}  # each open window's releases, by slot stride
        self._whole: dict[int, bool] = {}  # whether a closed window's three records were published complete
        self._ended = 0  # under form 2, the windows whose lengths rows are ended: 0 … _ended − 1
        #: the first bucket that admits nothing: under form 2, the first from which a connection may end past the run
        self._admits = windows * params.window - (params.span if params.closes else 0)
        self.queue: deque[_Connection] = deque()
        self.active: list[_Connection] = []
        self.ordinals = 0
        self.refused = 0
        self.refused_windows: set[int] = set()
        self.turned_away = 0
        self._away = cell(REFUSED, 0, b"", params.cell_bytes)  # every turn-away's cell, made once
        #: the arrivals turned away since the last tick: each one's count, its reader and its writer
        self._turned: list[tuple[int, asyncio.StreamReader, Any]] = []
        self._leaving: dict[Any, asyncio.Task] = {}  # a turned-away link from its cell's tick until it is closed
        self._over = False  # the run has ended: no tick is left
        self.log: list[dict] = []
        self.lateness: list[float] = []
        self.ticks = 0
        self._changed = asyncio.Event()

    async def until(self, predicate: Callable[[Warden], bool]) -> None:
        while not predicate(self):
            self._changed.clear()
            await self._changed.wait()

    def _bucket(self) -> int:
        return math.ceil((self.clock() - self.origin) * 1000 / self.params.bucket_ms)

    async def handle(self, reader: asyncio.StreamReader, writer: Any) -> None:
        if self.params.closes and self.ticks >= self._admits:
            # turned away (the module docstring's "The lengths rows"): by the tick count and the schedule alone,
            # read before the queue; its cell is the next tick's to release
            self.turned_away += 1
            if self._over:
                writer.close()
            else:
                self._turned.append((self.turned_away, reader, writer))
            self._changed.set()
            return
        if len(self.queue) >= self.params.queue:
            self.refused += 1
            self.refused_windows.add(self.ticks // self.params.window)
            writer.write(cell(REFUSED, 0, b"", self.params.cell_bytes))
            writer.close()
            self._changed.set()
            return
        connection = _Connection(self.ordinals + 1, reader, writer, asyncio.get_running_loop().create_future(),
                                 self._bucket())
        self.ordinals += 1
        self.queue.append(connection)
        self._changed.set()
        try:
            await connection.done
        finally:
            writer.close()

    async def run(self) -> None:
        tau = self.params.bucket_ms / 1000
        reports = None
        if self.paced is not None:
            self._pacer = await Pacer.start(backlog=WRITE_BACKLOG)
            reports = asyncio.create_task(self._reports(self._pacer))
        opening = asyncio.create_task(self._open_watermark()) if self.watermark is not None else None
        try:
            for t in range(self.windows * self.params.window):
                deadline = self.origin + t * tau
                await self.sleep(deadline - self.clock())
                late = self.clock() - deadline
                self.lateness.append(late)
                if late > self.slack:
                    for grid in (self.ingress, self.egress, self.fails):
                        grid.fault(Reason.MONITOR_UNAVAILABLE)
                self._tick(t)
                self.ticks += 1
                self._changed.set()
            if self._pacer is not None:
                await self._pacer.close()
                await reports
        finally:
            self._over = True
            for connection in [*self.active, *self.queue]:
                self._end(connection, cut=True)
            self.active, self.queue = [], deque()
            for _, _, writer in self._turned:
                writer.close()
            self._turned.clear()
            for writer, leaving in self._leaving.items():
                leaving.cancel()
                writer.close()
            self._leaving.clear()
            if opening is not None:
                opening.cancel()
                self._end_watermark()
            if self._pacer is not None:
                reports.cancel()
                await self._pacer.kill()
                for w in sorted(self._held):
                    self._emit(w, Reason.MONITOR_UNAVAILABLE)
            if self.params.closes:
                for w in range(self._ended, self.windows):
                    self._end_lengths(w, False)
            self._changed.set()

    def _tick(self, t: int) -> None:
        T = self.params.window
        if t % T == 0 and t // T - 1 in self._held:
            self._emit(t // T - 1, Reason.MONITOR_UNAVAILABLE)
        if t % T == 0 and self.params.lengths is not None:
            self._released[t // T] = [[] for _ in range(T // self.params.every)]
        while self.queue and len(self.active) < self.params.lanes and t < self._admits:
            self._admit(self.queue.popleft(), t)
        parts, released = [], []
        for c in self.active:
            if c.ingress_end is None and c.failed is None:
                for frame in self._ingress(c, t):
                    self.ingress.enqueue(frame)
                if c.ingress_end == t:
                    released.append(c)
                    if self.params.lengths is not None:
                        stride = self._released[t // T][t % T // self.params.every]
                        if len(stride) >= self.params.stride_releases:
                            raise RuntimeError(f"bucket {t}'s slot stride releases more than "
                                               f"{self.params.stride_releases} requests, the lengths rows' slots")
                        stride.append(c)
            out = self._egress(c, t)
            for frame in out:
                self.egress.enqueue(frame)
            if self._pacer is None:
                self._write(c, b"".join(out))
            else:
                parts.append((c.ordinal, b"".join(out), c.ended is not None))
        for n, reader, writer in self._turned:
            if writer.is_closing():
                continue
            if self._pacer is None:
                writer.write(self._away)
                if writer.can_write_eof():
                    writer.write_eof()
            else:
                self._pacer.add(AWAY_KEYS - n, writer.get_extra_info("socket").fileno())
                parts.append((AWAY_KEYS - n, self._away, True))
            self._leaving[writer] = asyncio.create_task(self._leave(reader, writer))
        self._turned.clear()
        if parts:
            self._pacer.hand(t, self.origin + t * self.params.bucket_ms / 1000 + self.paced, parts)
            self._unreleased[t // T] += 1
        for c in self.active:
            if c.ended is not None:
                self._end(c)
        self.active = [c for c in self.active if c.ended is None]
        if self.watermark is not None:
            self._watermark_lines.append((t, self.requests_released, released))
            self._flush_watermark()
        for grid in (self.ingress, self.egress, self.fails):
            row = grid.tick()
            if self.release is not None:
                self.release(grid.link, row)
        if (t + 1) % T == 0:
            w = t // T
            out = self.withhold if w in self.refused_windows else self.publish
            records = [(self.ingress.close_window(), self.params.grid), (self.egress.close_window(), self.params.grid),
                       (self.fails.close_window(), self.params.fails_grid)]
            self._held[w] = (out, records)
            if not self._unreleased[w]:
                self._emit(w)

    async def _leave(self, reader: asyncio.StreamReader, writer: Any) -> None:
        """A turned-away link from the tick that releases its cell and its stream's end: what its client still sends is
        read and dropped, to the client's end of stream, and the link is closed then. A link closed with its client's
        bytes unread is reset, and a reset client may lose the cell it has not read."""
        try:
            while await reader.read(READ_BYTES):
                pass
        except OSError:
            pass
        finally:
            writer.close()
            self._leaving.pop(writer, None)
            self._changed.set()

    def _emit(self, w: int, reason: Reason | None = None) -> None:
        """Window w's held records, each with `reason` as its status unless it has one: a window records its first
        fault."""
        out, records = self._held.pop(w)
        whole = out is self.publish
        for record, grid in records:
            if reason is not None and record.status.complete:
                record = replace(record, status=Status(reason))
            whole = whole and record.status.complete
            out(record, grid)
        if self.params.lengths is not None:
            self._emit_lengths(w, whole)

    def _emit_lengths(self, w: int, whole: bool) -> None:
        """The lengths rows that window w's records, just published (`whole`: complete, and none withheld), were the
        last to wait on (the module docstring's "The lengths rows"). Form 1: window w − 1's, handed where it and
        window w were published complete: a window whose cells have no certificate, or whose releases' responses
        reach into one that has none, gets no lengths rows. Form 2: window w − 1's, handed where window w − 2 (if
        any), it and window w were, and at the run's last window that window's own, handed where the window before
        it (if any) and it were; each one ended without releases where it isn't handed."""
        if not self.params.closes:
            before = self._whole.pop(w - 1, False)
            self._whole[w] = whole
            self._end_lengths(w - 1, whole and before)
            return
        earlier = self._whole.pop(w - 2) if w >= 2 else True
        self._whole[w] = whole
        if w:
            self._end_lengths(w - 1, whole and self._whole[w - 1] and earlier)
        if w == self.windows - 1:
            self._end_lengths(w, whole and (not w or self._whole[w - 1]))

    def _end_lengths(self, w: int, handed: bool) -> None:
        """Window w's releases by slot stride to `hand_lengths` when `handed`, each of them ended (`WardenParams`:
        with lengths rows a connection ends within a window of its admission, and under form 2 inside the run).
        Otherwise form 1 drops them, and form 2 ends the window with the same call and no releases."""
        strides = self._released.pop(w, None)
        if strides is None or not handed:
            if self.params.closes:
                self._ended = w + 1
                self.hand_lengths(w, None)
            return
        if any(c.ended is None or c.failed in CLIENT_REASONS for stride in strides for c in stride):
            raise RuntimeError(f"window {w} released a request whose connection is live, or failed as its client's, "
                               f"when its lengths rows are handed")
        self._ended = w + 1
        self.hand_lengths(w, tuple(tuple(
            Released(c.release, c.bytes_in, c.egress_cells - (c.failed is None), c.failed is not None, c.request,
                     bytes(c.ids)) for c in stride) for stride in strides))

    async def _reports(self, pacer: Pacer) -> None:
        async for report in pacer.reports():
            self._report(report)

    def _report(self, report: Report) -> None:
        """A bucket's release or its writes as the pacer reports them. A bucket that reached the pacer after its time,
        or whose release started more than `slack` after it, faults every grid's window `monitor-unavailable`: the open
        window's certifiers, or the records held for a closed one, published once every bucket of it that was handed is
        released. A connection's writes fault nothing, as on the loop: a client that doesn't read holds its own."""
        if report.end is not None:
            self.releases.append(report)
            return
        w = report.t // self.params.window
        if report.arrived > 0 or report.start > self.slack:
            if w in self._held:
                out, records = self._held[w]
                self._held[w] = (out, [(r if not r.status.complete else
                                        replace(r, status=Status(Reason.MONITOR_UNAVAILABLE)), g) for r, g in records])
            elif w == self.ingress.window:
                for grid in (self.ingress, self.egress, self.fails):
                    grid.fault(Reason.MONITOR_UNAVAILABLE)
        self._unreleased[w] -= 1
        if not self._unreleased[w] and w in self._held:
            self._emit(w)
        self._changed.set()

    def _admit(self, c: _Connection, t: int) -> None:
        c.admitted = t
        _pad(c, self.params.cell_bytes)
        self.active.append(c)
        if self._pacer is not None:
            self._pacer.add(c.ordinal, c.writer.get_extra_info("socket").fileno())
        c.tasks = [asyncio.create_task(self._client(c)), asyncio.create_task(self._server(c))]

    async def _client(self, c: _Connection) -> None:
        """Read the client's cells into `c.ingress` until its `END` (or its end of stream), at most `ingress_cells`
        cells' bytes. Only the payload of each passes: the network certifier makes every cell it sends."""
        B, p = self.params.cell_bytes, self.params.payload
        try:
            while not c.ingress_ended:
                try:
                    raw = await c.reader.readexactly(B)
                except asyncio.IncompleteReadError as e:
                    if e.partial:
                        c.problem = c.problem or "malformed"
                    c.ingress_ended = True
                    break
                kind, _, n = HEADER.unpack_from(raw)
                if kind not in (DATA, END) or n > p:
                    c.problem = c.problem or "malformed"
                    return
                c.ingress += raw[HEADER.size:HEADER.size + n]
                c.bytes_in += n
                c.ingress_ended = kind == END
                if c.bytes_in > self.params.ingress_cells * p:
                    c.problem = c.problem or "ingress"
                    return
                self._changed.set()
        except OSError:
            c.problem = c.problem or "malformed"
        finally:
            self._changed.set()

    async def _server(self, c: _Connection) -> None:
        """Open the server connection and read its bytes into `c.egress` until it ends. Past `egress_cells` cells'
        bytes it stops reading: those can't be sent, and the slot of the cell past the last fails (limit)."""
        cap = self.params.egress_cells * self.params.payload
        try:
            c.server = await self.connect()
            if c.unsent:
                c.wrote = not c.server[1].is_closing()
                c.server[1].write(bytes(c.unsent))
                c.unsent.clear()
            if self.watermark is not None:
                self._flush_watermark()
            self._changed.set()
            reader = c.server[0]
            while c.bytes_out <= cap:
                data = await reader.read(READ_BYTES)
                if not data:
                    c.egress_ended = True
                    return
                c.egress += data
                c.bytes_out += len(data)
                self._changed.set()
        except OSError as e:
            c.upstream = f"{type(e).__name__}: {e}"
            if self.watermark is not None:
                self._flush_watermark()
        finally:
            self._changed.set()

    def _ingress(self, c: _Connection, t: int) -> list[bytes]:
        """Connection c's r ingress cells at bucket t, their payloads held for the server until the `END`'s release,
        then written whole with the release order, or kept until its connection opens (at most the ingress bound's
        bytes): the release never waits on the server."""
        B, p, r = self.params.cell_bytes, self.params.payload, self.params.slots
        out, sent, pad = [], [], _pad(c, B)
        for _ in range(r):
            if c.ingress_end is None and not c.refused and not c.ingress_cells and not sent and \
                    self.params.max_tokens_cap is not None and (len(c.ingress) >= p or c.ingress_ended):
                c.max_tokens = id_request_max_tokens(bytes(c.ingress[:p]))
                if c.max_tokens is None or c.max_tokens > self.params.max_tokens_cap:
                    c.refused, c.problem = True, c.problem or "cap"
            if c.ingress_end is not None or c.refused or (c.problem is not None and self.params.lengths is not None):
                out.append(pad)
            elif len(c.ingress) > p or (len(c.ingress) == p and not c.ingress_ended) or c.ingress_ended:
                kind = DATA if len(c.ingress) > p or not c.ingress_ended else END
                made = cell(kind, c.ordinal, bytes(c.ingress[:p]), B)
                if self.params.lengths is not None and not _id_request_cell(self.params, c, made):
                    c.problem = c.problem or "malformed"
                    out.append(pad)
                    continue
                sent.append(bytes(c.ingress[:p]))
                del c.ingress[:p]
                out.append(made)
                if kind == END:
                    c.ingress_end = t
            else:
                out.append(pad)
        c.ingress_cells += len(sent)
        if c.ingress_end is None and t - c.admitted + 1 >= self.params.ingress_buckets:
            c.problem = c.problem or "ingress"
        c.held += b"".join(sent)
        if c.ingress_end == t:
            c.release, self.requests_released = self.requests_released, self.requests_released + 1
            if self.params.content is not None:
                c.request = bytes(c.held)
            whole = with_release(bytes(c.held), c.release, self.run_id, t)
            c.held.clear()
            if c.server is None:
                c.unsent += whole
            elif not c.server[1].is_closing():
                c.server[1].write(whole)
                c.wrote = True
        return out

    def _egress(self, c: _Connection, t: int) -> list[bytes]:
        """Connection c's r egress cells at bucket t (the module docstring's schedule), from s + D on. A client's
        problem fails the connection at once (with lengths rows at its first egress slot, so it holds its lane through
        s + D as every other connection does); the server's only at an egress slot, so where it fails is one of the
        egress cells. A `DATA` or `END` cell out of the egress form fails it in that cell's slot, and none of its bytes
        leaves. The reason goes to the `fails` record, never into the cell."""
        B, p, r, D = self.params.cell_bytes, self.params.payload, self.params.slots, self.params.delay
        out, pad = [], _pad(c, B)
        for _ in range(r):
            if c.ended is not None:
                out.append(pad)
                continue
            failed = None
            if c.problem is not None and (self.params.lengths is None or t >= c.admitted + D):
                failed = c.problem
            elif t < c.admitted + D:
                out.append(pad)
                continue
            else:
                whole = len(c.egress) > p or (len(c.egress) == p and not c.egress_ended)
                kind = None
                if c.egress_ended and not whole:
                    kind = END
                elif c.egress_cells + 1 >= self.params.egress_cells or \
                        (whole and c.max_tokens is not None and c.egress_cells >= c.max_tokens):
                    failed = "limit"
                elif whole:
                    kind = DATA
                else:
                    failed = "upstream" if c.upstream else "late"
                if kind is not None:
                    made = cell(kind, c.ordinal, bytes(c.egress[:p]), B)
                    if self.params.egress == "bytes" or (read := parse_cell(made)) is not None:
                        out.append(made)
                        del c.egress[:p]
                        c.egress_cells += 1
                        if kind == DATA and self.params.content is not None:
                            c.ids += read.id.to_bytes(4, "little")
                        if kind == END:
                            c.ended = t
                        continue
                    failed = "format"
            c.failed, c.failed_at, c.ended = failed, c.egress_cells, t
            out.append(cell(FAIL, c.ordinal, b"", B))
            self.fails.enqueue(reason_frame(c.ordinal, failed, B))
        return out

    def _write(self, c: _Connection, data: bytes) -> None:
        writer = c.writer
        if writer.is_closing():
            return
        if writer.transport.get_write_buffer_size() > WRITE_BACKLOG:
            writer.close()
            return
        writer.write(data)

    async def _open_watermark(self) -> None:
        """Open the watermark stream and send its head, then the lines made before it opened."""
        try:
            _, writer = await self.watermark()
        except OSError as e:
            self.watermark_error = f"{type(e).__name__}: {e}"
            return
        if self.watermark_error is not None:
            writer.close()
            return
        self._watermark_writer = writer
        self._send_watermark(watermark_head(self.run_id) + bytes(self._watermark_unsent))
        self._watermark_unsent.clear()

    def _written(self, c: _Connection) -> bool | None:
        """Whether c's release has reached its server connection: written whole to it open and its transport's buffer
        emptied (True), or lost, the connection closing, failed or ended first (False); None while neither holds."""
        if c.written is None:
            w = None if c.server is None else c.server[1]
            if c.wrote and not w.is_closing() and w.transport.get_write_buffer_size() == 0:
                c.written = True
            elif (w is not None and w.is_closing()) or c.upstream is not None or c.done.done():
                c.written = False
        return c.written

    def _flush_watermark(self) -> None:
        """Send the lines made, in order, each once every release of its bucket has reached the server or been lost,
        naming the lost: a release still on its way holds its line and the later ones, never a tick, at most until its
        connection ends."""
        while self._watermark_lines:
            t, n, released = self._watermark_lines[0]
            done = [self._written(c) for c in released]
            if None in done:
                return
            self._watermark_lines.popleft()
            lost = [c.release for c, ok in zip(released, done) if not ok]
            self.watermarked = (t, n)
            self.watermark_lost += lost
            self._send_watermark(watermark_chunk(t, n, lost))

    def _send_watermark(self, data: bytes) -> None:
        """Write `data` on the watermark stream, or keep it until the stream opens, never waiting: a stream with more
        than `WRITE_BACKLOG` unwritten is closed, and a stopped one takes nothing more."""
        if self.watermark_error is not None:
            return
        w = self._watermark_writer
        if w is None:
            self._watermark_unsent += data
            if len(self._watermark_unsent) > WRITE_BACKLOG:
                self.watermark_error = "backlog"
                self._watermark_unsent.clear()
        elif w.is_closing():
            self.watermark_error = "closed"
        elif w.transport.get_write_buffer_size() > WRITE_BACKLOG:
            self.watermark_error = "backlog"
            w.close()
        else:
            w.write(data)

    def _end_watermark(self) -> None:
        """At the run's end, every connection ended: the lines still held, the terminating chunk, and the stream
        closed."""
        self._flush_watermark()
        if self._watermark_writer is None:
            self.watermark_error = self.watermark_error or "unopened"
            return
        self._send_watermark(WATERMARK_END)
        self._watermark_writer.close()

    def _end(self, c: _Connection, cut: bool = False) -> None:
        for task in c.tasks:
            task.cancel()
        if self.watermark is not None and c.release is not None:
            self._written(c)
        if c.server is not None:
            c.server[1].close()
        c.writer.close()
        if not c.done.done():
            c.done.set_result(None)
        self.log.append({"ordinal": c.ordinal, "arrived": c.arrived, "admitted": c.admitted,
                         "ingress_end": c.ingress_end, "release": c.release, "ended": c.ended,
                         "ingress_cells": c.ingress_cells, "max_tokens": c.max_tokens,
                         "egress_cells": c.egress_cells, "bytes_in": c.bytes_in, "bytes_out": c.bytes_out,
                         "failed": c.failed, "failed_at": c.failed_at, "upstream": c.upstream, "cut": cut})


def audit(params: WardenParams, ingress: Sequence[Record], egress: Sequence[Record], fails: Sequence[Record]) -> dict:
    """The read of a run's records in the clear (a rehearsal's; a deployment's is N2's, in ZK). Every cell is a warden
    cell; in every bucket of each direction each live connection has r cells, in ordinal order, and admissions are in
    ordinal order; each connection's ingress is data and padding, then one `END` and padding to its bucket a; and its
    egress is exactly the schedule's for (its admission s, its egress cells): padding through bucket s + D − 1 (up to
    its `FAIL`, for a client's failure before then), then `DATA` in every slot, then one `END` or `FAIL`, then padding
    to the bucket's end. Every `FAIL` has one reason in the `fails` record, in its bucket, and every reason its `FAIL`.
    In the id egress form every egress `DATA` and `END` cell is one `io_tie.parse_cell` reads. Every reason is the
    schedule's (`WardenParams.reasons`). Under a response cap each connection's `max_tokens` is read from its first
    request cell as the network certifier read it: a request cut into cells has one at most the cap, its response at
    most that many `DATA` cells, and only a request with no cells is refused (cap). A window is accepted when its three
    records are complete, its cells are the schedule's and the egress form's and no connection failed in it. With
    lengths rows every `FAIL` is at an egress slot, a client's reason (`CLIENT_REASONS`) is a connection's whose request
    has no `END` and a server's one whose request has, and a connection that was never released fails no window: its
    failure is its client's, no server saw its request, and its entry still names its reason. Each
    connection's entry has its buckets, cells, outcome, its failure's reason, its `max_tokens` under a response cap
    (else None) and the SHA-256 of each direction's bytes; the verdict names the schedule's `max_tokens_cap`."""
    T, r, D, cap = params.window, params.slots, params.delay, params.max_tokens_cap
    verdicts: dict[int, list[str]] = defaultdict(list)
    for record in [*ingress, *egress, *fails]:
        if not record.status.complete:
            verdicts[record.window].append(f"{record.link}: {record.status.reason.value}")
    reasons: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for record in fails:
        for b, row in enumerate(record.grid):
            for f in row[:row_count(row)]:
                try:
                    o, why = read_reason(f, params.cell_bytes)
                except ValueError as e:
                    verdicts[record.window].append(f"fails bucket {record.window * T + b}: {e}")
                    continue
                if why not in params.reasons:
                    verdicts[record.window].append(f"fails bucket {record.window * T + b}: {why}, not this schedule's")
                reasons[o].append((record.window * T + b, why))
    per: dict[int, dict[str, dict[int, list[tuple[int, bytes]]]]] = defaultdict(lambda: {"ingress": {}, "egress": {}})
    total = {"ingress": 0, "egress": 0}
    for name, records in (("ingress", ingress), ("egress", egress)):
        for record in records:
            for b, row in enumerate(record.grid):
                t = record.window * T + b
                frames = row[:row_count(row)]
                try:
                    got = [read_cell(f, params.cell_bytes) for f in frames]
                except ValueError as e:
                    verdicts[record.window].append(f"{name} bucket {t}: {e}")
                    continue
                if name == "egress" and params.egress == "id" and \
                        any(k in (DATA, END) and parse_cell(f) is None for f, (k, _, _) in zip(frames, got)):
                    verdicts[record.window].append(f"egress bucket {t}: a cell out of the id egress form")
                total[name] += len(got)
                order = [o for _, o, _ in got[::r]]
                if len(got) % r or got != sorted(got, key=lambda c: c[1]) or len(set(order)) != len(order) or \
                        any(o != got[i * r + j][1] for i, o in enumerate(order) for j in range(r)):
                    verdicts[record.window].append(f"{name} bucket {t}: not r cells a connection in ordinal order")
                for kind, o, payload in got:
                    per[o][name].setdefault(t, []).append((kind, payload))
    connections, last_admitted = [], -1
    for n, o in enumerate(sorted(per)):
        ing, eg = per[o]["ingress"], per[o]["egress"]
        s = min([*ing, *eg])
        problems = [f"connection {o} where {n + 1} is next"] if o != n + 1 else []
        if s < last_admitted:
            problems.append(f"admitted at {s}, before connection {o - 1}")
        last_admitted = s
        if sorted(ing) != list(range(s, s + len(ing))) or sorted(eg) != list(range(s, s + len(eg))):
            problems.append("its buckets aren't consecutive from its admission")
        ing_cells = [c for t in sorted(ing) for c in ing[t]]
        eg_cells = [c for t in sorted(eg) for c in eg[t]]
        kinds = [k for k, _ in ing_cells]
        a = None
        if END in kinds:
            e = kinds.index(END)
            a = s + e // r
            if any(k not in (DATA, PAD) for k in kinds[:e]) or any(k != PAD for k in kinds[e + 1:]) or \
                    len(kinds) != (a - s + 1) * r:
                problems.append("its ingress isn't data and padding, then one END and padding to its bucket")
        elif any(k not in (DATA, PAD) for k in kinds):
            problems.append("its ingress isn't data and padding")
        kinds = [k for k, _ in eg_cells]
        live = [k for k in kinds if k != PAD]
        failed = failed_at = None
        if live:
            last = live[-1]
            early = live == [FAIL] and kinds.index(FAIL) < D * r and params.lengths is None
            lead = kinds.index(FAIL) if early else D * r
            expected = [PAD] * lead + [DATA] * (len(live) - 1) + [last]
            expected += [PAD] * (-len(expected) % r)
            if last not in (END, FAIL) or kinds != expected:
                problems.append("its egress isn't padding to s + D, data in every slot, one END or FAIL, padding")
        got = reasons.pop(o, [])
        if FAIL in kinds:
            failed_at = live.index(FAIL)
            if [t for t, _ in got] == [s + kinds.index(FAIL) // r]:
                failed = got[0][1]
            else:
                failed = "fail"
                problems.append("its FAIL hasn't one reason, in its bucket")
        elif got:
            problems.append("it has a reason and no FAIL")
        requested = [p for k, p in ing_cells if k in (DATA, END)]
        max_tokens = None
        if cap is not None and requested:
            max_tokens = id_request_max_tokens(requested[0])
            if max_tokens is None or max_tokens > cap:
                problems.append(f"its request's max_tokens ({max_tokens}) is past the cap, and was released")
            elif live.count(DATA) > max_tokens:
                problems.append(f"its response has {live.count(DATA)} DATA cells, past its max_tokens ({max_tokens})")
        if failed == "cap" and requested:
            problems.append("it was refused (cap) after its request was released")
        clients = failed in CLIENT_REASONS
        if params.lengths is not None and failed not in (None, "fail") and clients != (a is None):
            problems.append(f"it failed ({failed}) and its request was {'never ' if a is None else ''}released")
        data_in = b"".join(p for k, p in ing_cells if k in (DATA, END))
        data_out = b"".join(p for k, p in eg_cells if k in (DATA, END))
        ended = max(eg) if live and live[-1] in (END, FAIL) else None
        for problem in problems:
            verdicts[s // T].append(f"connection {o}: {problem}")
        if failed and not (params.lengths is not None and clients and a is None):
            verdicts[max(eg) // T].append(f"connection {o} failed closed ({failed}) at egress cell {failed_at}")
        connections.append({"ordinal": o, "admitted": s, "ingress_end": a, "ended": ended,
                            "egress_cells": len(live), "ingress_bytes": len(data_in), "egress_bytes": len(data_out),
                            "failed": failed, "failed_at": failed_at, "max_tokens": max_tokens,
                            "ingress_sha256": hashlib.sha256(data_in).hexdigest(),
                            "egress_sha256": hashlib.sha256(data_out).hexdigest()})
    for o, got in reasons.items():
        for t, _ in got:
            verdicts[t // T].append(f"a reason for connection {o}, which has no cells")
    windows = sorted({rec.window for rec in [*ingress, *egress, *fails]})
    out = [{"window": w, "accepted": not verdicts.get(w), "problems": verdicts.get(w, [])} for w in windows]
    return {"accepted": bool(out) and all(w["accepted"] for w in out), "windows": out, "connections": connections,
            "cells": total, "max_tokens_cap": cap}
