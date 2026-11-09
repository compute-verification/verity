"""attacks: what the served path must refuse or charge on the active network certifier's path (`warden`), each attack
with the check that refuses it or the bound's term that charges it, the prover's relay that plays the server's and the
connection's attacks between the network certifier's server connections and the endpoint, and the judge that reads an
attack's served rehearsal as its verdict.

    python benchmarks/network_traces/attacks.py table [--markdown]
    python benchmarks/network_traces/attacks.py relay --attack NAME --listen PORT --to PORT --cell-bytes B
        [--bucket-ms 100] [--delay 27] [--nth 1] [--log FILE]
    python benchmarks/network_traces/attacks.py judge NAME --front OUT --serve SERVE_DIR
    python benchmarks/network_traces/attacks.py rehearse PORT SERVE_DIR OUT [--attacks NAME,NAME...] -- FRONT_FLAGS...

**The attacks** (`ATTACKS`). Each names its case (`CASES`), the party that plays it, what it does, its outcome
(``refused``; ``charged``, inside a term the bound already counts; or ``gap``, what no check on the served path sees),
the check or the term (``by``), the reason its verdict names (``reason``), and the check's kind:
- ``device``: the network certifier's own check (`warden_serve.check`: its certificates, `warden.audit` and every
  connection's cells against the schedule, on the auditor's host, which `trusted-host` covers), or a read of the cells
  it certified (`warden_chain.io`);
- ``proof``: Lean's registered reads of the network certifier's signed rows (`warden_rows.refusal`);
- ``clear``: a check in the clear on what the prover records or opens: the piece's `certifier_piece.bind`, which
  reads the endpoint's records against the certified I/O (it refuses what a relay between the network certifier and
  an honest endpoint does; the verdict of record waits on the egress tie, which binds the committed values to the
  certified cells), the records audit's compile of an execution phase's advice opening (`prove_path.compile_advice`),
  and ADVICE-LINE (`served_verdicts.advice_line`); a ``waits`` names the piece whose refusal of record hasn't landed.
An attack the served rehearsal plays (``served``) also names the network certifier's check's outcome (``gate``: True
when it accepts the run) and the piece's (``piece``): ``bound`` (steps 1 and 2 pass), ``reject``, or ``missing``
(`certifier_piece.NotRun`: no served answer certified, the certifier missing). An attack with ``waits`` lands with the
piece it waits on, whose test holds its refusal of record. The bound's terms are Lean's
`NetTiming.Guarantees.PeriodLeakAccepted`'s, k_exfil ≤ k_sched + k_fail + Σ_j (ℓ_a,j + k_skip,j + T_j log2 d_jit)
over the period's m execution phases: a failure, the server's choice of which connections fail and where, is inside
k_fail, log2(cap_c + 1) over every connection c the schedule can carry (`Schedule.caps`), served or not; a cell count
the server moves is inside k_sched, over the same connections; a hold or a chunking that keeps every cell in its
schedule's bucket is inside T_j log2 d_jit, T_j phase j's buckets, at d_jit = 2 (`JitterBounded` holds at no less;
`leak_bounds.JITTER_BOUND`); none adds a term.

**The relay** (`relay`, `Relay`): the prover's process between the network certifier's server connections (on the
served path, the gate's tunnel to the prover's loopback) and the endpoint, the server's side, which the prover owns.
Connection k, in accept order from 1, is relayed to ``--to``, its request bytes unchanged and its response through the
attack's egress (`egress`), which reads the response a P-byte event at a time (P = B − the cell header's 9 bytes, the
cells' payload and the id-form response's event size) and writes it:
- ``honest``: as read;
- ``jitter``: event i after holding it half a bucket when the i-th of the prover's secret bits is 1, so at most
  half a bucket an event, each before its cell's deadline for a response of at most 2·D events;
- ``chunk``: a byte at a time;
- ``late``: the whole response after holding its first byte D + 2 buckets, past its first cell's deadline;
- ``pad``: a cell's worth of padding bytes (P spaces) before the response's tail, its trailer;
- ``token``: its first event's id one other (`other_event`); ``token-pad``: one more id event before the tail.
``jitter`` and ``chunk`` play on every connection, the others on completion ``--nth``: the connection carrying the
network certifier's ``--nth`` completion request (`COMPLETION`), its port probes and model listing aside. Two act on
a connection: ``extra`` answers completion ``--nth`` itself, an id-form response of the request's
`warden_serve.fake_ids`, and never opens the endpoint, so the network certifier carries a response the endpoint didn't
serve; ``bypass`` first sends a copy of completion ``--nth``'s request straight to the endpoint (its release headers
gone, its ``user`` another of the same width), a request served around the network certifier, whose answer it drops,
then relays the connection; ``bypass-user`` sends the copy under the request's own ``user``. ``--log`` gets, once
each connection ends, its line: the connection, its completion's number, the attack, its bytes each way, the holds
and the bits (the prover's own record, kept on its node).

**The judge** (`judge`): an attack's served rehearsal (`certifier_piece.py front --attack`, its OUT's
``session.json`` and `SPAN`, and the serve dir it kept the run in) read as the attack's verdict (`verdict`): the network
certifier's check's outcome, and the piece's steps 1 and 2 (`bound`) on the requests the endpoint recorded for the run,
the front's warm-ups aside (every list in its ``warmup.json``: ``--warmup``'s and ``--stop-prompt``'s): each carrying
the run's release, and each carrying no run's that the endpoint received
(``times.received``) while the front ran, by the node's clock (`SPAN`), so a request around the network certifier is
read wherever its Commit period falls and another front's never. ``ok`` when both are the attack's and the piece names
the attack's reason.

**The rehearsal** (`rehearse`): served_verdicts.sh's ``CERTIFIER_FRONT`` (``LOAD=front``, the fronts' groups the run's
load), one front after another against the one serve-openai, each playing one attack (``--attacks``, `SERVED` by
default, in their order) at the front's flags after ``--`` and the attack's ``--tokens`` where it names one (``pad`` and
``token-pad`` under the egress cells' cap, ``pad-cap`` at it: `UNDER_CAP`, `AT_CAP`), each judged. OUT/NAME is attack
NAME's front's OUT, with its `SPAN` and verdict.json; OUT/attacks.json holds the verdicts, which name the run's requests
and stay on the node, and OUT/attacks.public.json and stdout their public forms (`public`). Exit 0 when every verdict is
``ok``. The piece reads exactly one kept run, so a serve dir that a rehearsal of more than one attack served answers no
piece: the rehearsal's run is the serve step alone.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import certifier_piece as CP  # noqa: E402
import leak_bounds as LB  # noqa: E402
import warden_chain as WC  # noqa: E402
import warden_serve as WS  # noqa: E402

from verity.core.protocols.certifier import io_tie as IT  # noqa: E402
from verity.core.protocols.certifier.active import id_request  # noqa: E402
from verity.core.protocols.certifier.release import RELEASE_HEADER, RUN_HEADER  # noqa: E402

#: the attacks' cases, the served path's list of what must be refused or charged
CASES = ("an altered output token", "timing or chunking games", "size padding", "advice after its coin",
         "advice over the cap", "an extra connection", "traffic around the network certifier")
OUTCOMES = ("refused", "charged", "gap")
KINDS = ("device", "proof", "clear")
PIECE = ("bound", "reject", "missing")
#: the relay's attacks on a response (`EVERY`'s on every connection, the rest on the ``--nth``) and on a connection
EGRESS = ("honest", "jitter", "chunk", "late", "pad", "token", "token-pad")
EVERY = ("jitter", "chunk")
CONNECTION = ("extra", "bypass", "bypass-user")
#: the request line ``--nth`` counts: the network certifier's id-form completion (`active.ID_REQUEST_HEAD`), never its
#: port probes or its ``GET /v1/models``
COMPLETION = b"POST /v1/completions "
RELAYED = EGRESS + CONNECTION
#: the reasons the verdicts name, as their checks word them: `warden.audit`'s, `warden_chain.io`'s, the network
#: certifier's check's, and `certifier_piece.bind`'s
LATE = "failed closed (late)"
LIMIT = "failed closed (limit)"
NOT_WHOLE = "complete False"
NOT_ID = "isn't an id-form cell"
WITHHELD = "isn't certified"
OTHER_OUTPUT = "streamed_ids aren't the certified output"
NOT_SEEN, UNSERVED = CP.NOT_SEEN, CP.UNSERVED


@dataclass(frozen=True)
class Attack:
    name: str
    case: str
    party: str
    what: str
    outcome: str
    by: str
    reason: str
    kind: str
    gate: bool | None = None
    piece: str | None = None
    relay: str | None = None
    waits: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    #: the front's ``--tokens`` its served rehearsal plays it at (the front's own when None)
    tokens: int | None = None

    def __post_init__(self) -> None:
        if self.case not in CASES or self.outcome not in OUTCOMES or self.kind not in KINDS:
            raise ValueError(f"{self.name}: case, outcome and kind are CASES', OUTCOMES' and KINDS'")
        if self.relay is not None and self.relay not in RELAYED:
            raise ValueError(f"{self.name}: the relay plays {RELAYED}, not {self.relay!r}")
        if (self.piece is not None and self.piece not in PIECE) or (self.relay is not None) != (self.piece is not None):
            raise ValueError(f"{self.name}: a relayed attack names the piece's outcome, one of {PIECE}; no other does")

    @property
    def served(self) -> bool:
        return self.relay is not None


AUDIT = "the network certifier's check (warden_serve.check)"
BIND = "the piece's bind (certifier_piece.bind)"
REGISTERED = "Lean's registered reads (warden_rows.refusal), a5_warden.py's refused run"
A5 = "benchmarks/one_stage/tests/test_a5_warden.py"
PP = "benchmarks/prove_path/tests/test_prove_path.py"
TESTS = "benchmarks/network_traces/tests/test_attacks.py"
FAILED = ("inside k_fail: where a connection fails is the server's choice, which log2(cap_c + 1) counts for every "
          "connection c the schedule can carry; the rehearsal certifies no served answer")
JITTER = f"inside T_j log2 d_jit (`JitterBounded` at d_jit = {LB.JITTER_BOUND}, leak_bounds.JITTER_BOUND)"
#: the front's ids out for a response at V3c's scan class's egress cells' cap (its ids and the trailer), and one under
AT_CAP = CP.SS.A.SCAN_CLASSES[CP.V3C_SCAN_CLASS][0] - 1
UNDER_CAP = AT_CAP - 1

ATTACKS = {a.name: a for a in [
    Attack("jitter", "timing or chunking games", "server", "each id event held 0 or half a bucket by a secret bit",
           "charged", f"{JITTER}: every cell leaves in its schedule's bucket, so a hold moves a cell at most to the "
           "bucket the jitter term pays for; a hold past a cell's deadline fails its connection, inside k_fail", "",
           "device", gate=True, piece="bound", relay="jitter", tests=(TESTS,)),
    Attack("chunk", "timing or chunking games", "server", "the response written a byte at a time", "charged",
           f"{JITTER}: every cell's content is cut at its fixed offsets and leaves in its schedule's bucket; a cell "
           "whose bytes miss its bucket fails its connection, inside k_fail", "",
           "device", gate=True, piece="bound", relay="chunk", tests=(TESTS,)),
    Attack("late", "timing or chunking games", "server", "the response held past its first cell's deadline",
           "charged", f"{AUDIT}: the connection failed closed at its first egress cell; {FAILED}", LATE, "device",
           gate=False, piece="missing", relay="late", tests=(TESTS,)),
    Attack("swapped-read", "timing or chunking games", "prover", "a statement reading two units' cells swapped",
           "refused", REGISTERED, "made for another units' wiring", "proof", tests=(A5,)),
    Attack("pad", "size padding", "server", "a cell of padding bytes before the response's trailer", "refused",
           f"the certified egress (warden_chain.io): a DATA cell that is no output's, which the piece rejects; "
           f"{AUDIT} refuses the client's response (complete False)", NOT_ID, "device", gate=False, piece="reject",
           relay="pad", tests=(TESTS,), tokens=UNDER_CAP),
    Attack("pad-cap", "size padding", "server", "the padding cell on a response at the egress cells' cap",
           "charged", f"{AUDIT}: the connection failed closed at its last egress cell; {FAILED}", LIMIT, "device",
           gate=False, piece="missing", relay="pad", tests=(TESTS,), tokens=AT_CAP),
    Attack("pad-read", "size padding", "prover", "a statement reading a pad in a data cell's place", "refused",
           REGISTERED, "made for another units' wiring", "proof", tests=(A5,)),
    Attack("pad-named-read", "size padding", "prover", "a pad in a data cell's place, its rows under the Program's "
           "value", "refused", REGISTERED, "made for another units' wiring", "proof", tests=(A5,)),
    Attack("token-pad", "size padding", "server", "one more id event before the trailer", "refused",
           f"{BIND}: the certified output is one id longer than the endpoint's (a cell count the server moves is "
           "inside k_sched; the ids are the egress tie's)", OTHER_OUTPUT, "clear", gate=True, piece="reject",
           relay="token-pad", waits=("the egress tie",), tests=(TESTS,), tokens=UNDER_CAP),
    Attack("token", "an altered output token", "server", "the first event's id one other", "refused",
           f"{BIND}: the certified output isn't the endpoint's", OTHER_OUTPUT, "clear", gate=True, piece="reject",
           relay="token", waits=("the egress tie",), tests=(TESTS,)),
    Attack("altered-read", "an altered output token", "prover", "a statement reading a cell one bit off the "
           "network certifier's", "refused", REGISTERED, "doesn't open", "proof", tests=(A5,)),
    Attack("advice-late", "advice after its coin", "prover", "an execution phase's advice committed after that "
           "phase's coin", "refused", "ADVICE-LINE (served_verdicts.advice_line): every closed execution phase's "
           "advice line out before sampled proofs' and the network certifier's first coin", "after sampled proofs' or "
           "the network certifier's first coin", "clear", waits=("advice v1's commitment",)),
    Attack("advice-cap", "advice over the cap", "prover", "an execution phase's advice longer than the registered "
           "cap", "refused", "the advice's registered width ℓ_a: prove_path.compile_advice refuses an opening over "
           "ℓ_a before it reads it, and ℓ_a is in the registered compiler's digest (registering that compiler, capped "
           "at ℓ_a, before the served path's first execution phase is advice v1's)", "over the registered cap",
           "clear", waits=("advice v1's commitment",), tests=(PP,)),
    Attack("extra", "an extra connection", "server", "one connection answered by the relay, never by the endpoint",
           "refused", f"{BIND}: every certified request is a served one", UNSERVED, "clear", gate=True,
           piece="reject", relay="extra", tests=(TESTS,)),
    Attack("over-capacity", "an extra connection", "client", "a connection arriving with the queue full", "refused",
           f"{AUDIT}: REFUSED to the connection, and no record of its execution phase certified", WITHHELD, "device",
           tests=(TESTS,)),
    Attack("bypass", "traffic around the network certifier", "server", "a request served straight to the endpoint "
           "under a user no certified request names", "refused", f"{BIND}: every served request is a certified one",
           NOT_SEEN, "clear", gate=True, piece="reject", relay="bypass", tests=(TESTS,)),
    Attack("bypass-user", "traffic around the network certifier", "server", "a request served straight to the "
           "endpoint under a certified request's user", "refused", f"{BIND}: every served request carries the "
           "network certifier's release", NOT_SEEN, "clear", gate=True, piece="reject", relay="bypass-user",
           tests=(TESTS,)),
    Attack("other-route", "traffic around the network certifier", "server", "bytes leaving the server's host by "
           "another route than the network certifier's link", "gap", "nothing on the served path: the server's "
           "host can reach anyone by any other route (warden_serve.py's \"What it can't enforce here\")", "", "device"),
]}


def table(markdown: bool = True) -> str:
    q = "`" if markdown else ""
    rows = [("attack", "case", "party", "outcome", "by", "kind", "where")]
    for a in ATTACKS.values():
        where = ("gap" if a.outcome == "gap" else ", ".join(
            [*(["tests"] if a.tests else []), *(["served"] if a.served else []),
             *([f"of record waits: {', '.join(a.waits)}"] if a.waits else [])]))
        rows.append((f"{q}{a.name}{q}", a.case, a.party, a.outcome, a.by, a.kind, where))
    if not markdown:
        return "\n".join("\t".join(r) for r in rows)
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


# --- the relay ---------------------------------------------------------------------------------------------------------

def other_event(event: bytes, payload: int) -> bytes:
    """An id event's id one other (its id XOR 1), the same P bytes otherwise; ValueError for bytes that aren't one."""
    token = int(event[len(IT.ID_EVENT_HEAD):len(IT.ID_EVENT_HEAD) + IT.ID_DIGITS])
    if IT.id_event(token, payload) != event:
        raise ValueError("not an id event")
    return IT.id_event(token ^ 1, payload)


def bypass_request(request: bytes, frame_bytes: int, same_user: bool) -> bytes:
    """The server's copy of a network-certifier-form id request as a request of its own: its release headers gone and,
    unless `same_user`, its ``user`` another of the same width (its last character changed)."""
    head, _, body = request.partition(b"\r\n\r\n")
    kept = b"\r\n".join(x for x in head.split(b"\r\n")
                        if x.partition(b":")[0].strip().lower() not in (RELEASE_HEADER, RUN_HEADER))
    not_id = "the request isn't a network-certifier-form id request"
    try:
        doc = json.loads(body)
        fields = {k: doc[k] for k in WC.FIELDS}
        ok = id_request(**fields, frame_bytes=frame_bytes, offset=0) == kept + b"\r\n\r\n" + body
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{not_id}: {type(e).__name__}: {e}") from e
    if not ok:
        raise ValueError(not_id)
    if not same_user:
        user = fields["user"]
        fields["user"] = user[:-1] + ("~" if user[-1:] != "~" else "^") if user else "~"
    return id_request(**fields, frame_bytes=frame_bytes, offset=0)


async def read_request(reader: asyncio.StreamReader) -> bytes:
    """An HTTP request whole: its head through the blank line, then its Content-Length's bytes."""
    head = await reader.readuntil(b"\r\n\r\n")
    m = re.search(rb"(?im)^content-length:\s*(\d+)", head)
    return head + (await reader.readexactly(int(m.group(1))) if m else b"")


async def egress(attack: str, src: asyncio.StreamReader, dst: Any, *, payload: int, hold_s: float, late_s: float,
                 bits: Callable[[], int], sleep: Callable[[float], Awaitable[None]]) -> dict:
    """The response from `src` to `dst` as the attack writes it (the module docstring's relay); what it did."""
    done: dict = {"events": 0, "bytes": 0, "held_s": 0.0, "bits": []}

    async def put(data: bytes) -> None:
        if attack == "chunk":
            for i in range(len(data)):
                dst.write(data[i:i + 1])
                await dst.drain()
        elif data:
            dst.write(data)
            await dst.drain()
        done["bytes"] += len(data)

    first = True
    while True:
        try:
            event, tail = await src.readexactly(payload), False
        except asyncio.IncompleteReadError as e:
            event, tail = e.partial, True
        if attack == "late" and first:
            await sleep(late_s)
            done["held_s"] += late_s
        first = False
        if tail:
            if attack == "pad":
                await put(b" " * payload)
            elif attack == "token-pad":
                await put(IT.id_event(0, payload))
            await put(event)
            return done
        if attack == "token" and done["events"] == 0:
            event = other_event(event, payload)
        if attack == "jitter":
            bit = bits() & 1
            done["bits"].append(bit)
            if bit:
                await sleep(hold_s)
                done["held_s"] += hold_s
        await put(event)
        done["events"] += 1


class Relay:
    """The prover's relay (the module docstring): `handle(reader, writer)` relays one connection of the network
    certifier's, `open()` opens one to the endpoint, a (reader, writer) pair."""

    def __init__(self, attack: str, open: Callable[[], Awaitable[tuple[Any, Any]]], *, cell_bytes: int,
                 bucket_s: float, delay: int, nth: int = 1, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 bits: Callable[[], int] = lambda: secrets.randbits(1),
                 log: Callable[[dict], None] = lambda line: None) -> None:
        if attack not in RELAYED:
            raise ValueError(f"the relay plays {RELAYED}, not {attack!r}")
        self.attack, self.open, self.frame_bytes, self.payload = attack, open, cell_bytes, cell_bytes - IT.HEADER.size
        self.hold_s, self.late_s, self.nth = bucket_s / 2, (delay + 2) * bucket_s, nth
        self.sleep, self.bits, self.log = sleep, bits, log
        self.connections = self.completions = 0
        self.lines: list[dict] = []
        self._copies: set[asyncio.Task] = set()

    async def handle(self, reader: asyncio.StreamReader, writer: Any) -> None:
        self.connections += 1
        line = {"connection": self.connections, "attack": self.attack if self.attack in EVERY else "honest"}
        try:
            request = await read_request(reader)
            line["request_bytes"] = len(request)
            if request.startswith(COMPLETION):
                self.completions += 1
                line["completion"] = self.completions
            mine = line.get("completion") == self.nth
            if mine:
                line["attack"] = self.attack
            if mine and self.attack == "extra":
                answer = WS.id_response(WS.fake_ids(json.loads(request.partition(b"\r\n\r\n")[2])), self.payload)
                writer.write(answer)
                await writer.drain()
                line["response_bytes"] = len(answer)
                return
            if mine and self.attack in ("bypass", "bypass-user"):
                copy = bypass_request(request, self.frame_bytes, same_user=self.attack == "bypass-user")
                c_reader, c_writer = await self.open()
                c_writer.write(copy)
                await c_writer.drain()
                task = asyncio.ensure_future(self._drop(line["connection"], c_reader, c_writer))
                self._copies.add(task)
                task.add_done_callback(self._copies.discard)
                line["copy_bytes"] = len(copy)
            up_reader, up_writer = await self.open()
            up_writer.write(request)
            await up_writer.drain()
            attack = line["attack"] if line["attack"] in EGRESS else "honest"
            line |= await egress(attack, up_reader, writer, payload=self.payload, hold_s=self.hold_s,
                                 late_s=self.late_s, bits=self.bits, sleep=self.sleep)
            up_writer.close()
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, OSError, ValueError) as e:
            line["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.lines.append(line)
            self.log(line)
            writer.close()

    async def _drop(self, k: int, reader: asyncio.StreamReader, writer: Any) -> None:
        """Connection `k`'s copy's answer read whole and dropped; its bytes, or why it ended short, go to the log."""
        got: dict = {"connection": k, "copy": "dropped", "bytes": 0}
        try:
            while data := await reader.read(1 << 16):
                got["bytes"] += len(data)
        except (ConnectionError, OSError) as e:
            got["error"] = f"{type(e).__name__}: {e}"
        finally:
            writer.close()
            self.log(got)


async def serve_relay(attack: str, listen: int, to: int, *, cell_bytes: int, bucket_s: float, delay: int, nth: int,
                      log: Path | None) -> None:
    def write(line: dict) -> None:
        if log is not None:
            with open(log, "a") as f:
                f.write(json.dumps(line, sort_keys=True) + "\n")
        print(f"relay {attack}: {json.dumps({k: v for k, v in line.items() if k != 'bits'}, sort_keys=True)}",
              flush=True)

    relay = Relay(attack, lambda: asyncio.open_connection("127.0.0.1", to), cell_bytes=cell_bytes, bucket_s=bucket_s,
                  delay=delay, nth=nth, log=write)
    server = await asyncio.start_server(relay.handle, "127.0.0.1", listen)
    print(f"relay listening on {listen} to {to}: {attack}", flush=True)
    async with server:
        await server.serve_forever()


# --- the judge ---------------------------------------------------------------------------------------------------------

BOUND = "every request served is a certified one, each once, and every certified request is served"
#: the node's clock around a rehearsal's front, which `rehearse` writes beside its session
SPAN = "span.json"


def received(r: dict) -> float | None:
    return (r.get("times") or {}).get("received")


def piece_of(warden: Path, rows: list[dict] | Callable[[str], list[dict]]) -> dict:
    """The piece's steps 1 and 2 on the network certifier run `warden` (`certifier_piece.rehearsed`, then
    `certifier_piece.bind` of the endpoint's requests `rows`, or of those `rows` gives for the run's id), less step 1's
    rows audit, which holds the network certifier's own records to their schedule and so judges no relay's play:
    ``{"verdict": "bound" | "reject" | "missing", "why"}``."""
    try:
        CP.rehearsed(warden)
        dep, _ = WC.deployment(warden)
        rows = rows(dep["run_id"]) if callable(rows) else rows
        CP.bind(rows, WC.io(warden), dep["run_id"], rows)
    except CP.Rejected as e:
        return {"verdict": "reject", "why": str(e)}
    except CP.NotRun as e:
        return {"verdict": "missing", "why": str(e)}
    except SystemExit as e:
        return {"verdict": "missing", "why": f"the kept warden run: {e}"}
    return {"verdict": "bound", "why": BOUND}


def bound(serve_dir: Path, session: dict, span: tuple[float, float], skip: set[str] = frozenset()) -> dict:
    """`piece_of` the front's kept run and the requests the endpoint recorded for it, but the users `skip`: each
    carrying the run's release, and each carrying no run's that it received (its ``times.received``, the node's clock)
    within `span`, the node's clock around the front, or that has no such stamp."""
    run = session.get("run")
    if not run:
        return {"verdict": "missing", "why": f"the front kept no run: {session.get('refused')}"}
    lo, hi = span

    def rows(run_id: str) -> list[dict]:
        return [r for r in CP._lines(serve_dir / "requests.jsonl") if r.get("user") not in skip
                and (CP.release_run(r) == run_id
                     or (CP.release_run(r) is None and (received(r) is None or lo <= received(r) <= hi)))]

    unpacked = None
    try:
        unpacked, warden = CP.unpack(serve_dir / CP.PRIVATE / run / "fetch.tar.gz")
        return piece_of(warden, rows)
    except CP.NotRun as e:
        return {"verdict": "missing", "why": str(e)}
    finally:
        if unpacked is not None:
            CP.rm(unpacked)


def verdict(name: str, session: dict, piece: dict) -> dict:
    """The attack's verdict from its front's ``session.json`` (`served_session.outcome`: the gate's ``rc`` and its
    check's ``verdict``) and the piece's outcome (`bound`): ``ok`` when the network certifier's check and the piece
    came out as the attack's, the piece naming its reason."""
    a = ATTACKS[name]
    accepted = session.get("rc") == 0 and (session.get("verdict") or {}).get("accepted") is True
    got, why = piece.get("verdict"), piece.get("why") or ""
    ok = a.served and accepted == a.gate and got == a.piece and a.reason in why
    return {"attack": name, "case": a.case, "outcome": a.outcome, "kind": a.kind, "by": a.by,
            "gate_accepted": accepted, "piece": got, "why": why[:800],
            "expected": {"gate_accepted": a.gate, "piece": a.piece, "reason": a.reason}, "ok": bool(ok)}


def judge(name: str, front: Path, serve_dir: Path) -> dict:
    session = json.loads((front / "session.json").read_text())
    span = json.loads((front / SPAN).read_text())
    warm = front / "warmup.json"
    skip = {r["user"] for rs in json.loads(warm.read_text()).values() for r in rs} if warm.is_file() else set()
    return verdict(name, session, bound(serve_dir.resolve(), session, (span["start_unix"], span["end_unix"]), skip))


# --- the rehearsal -----------------------------------------------------------------------------------------------------

#: the served attacks in the order a rehearsal plays them: those around the network certifier last, so no copy of theirs
#: falls in another front's span
SERVED = tuple(a.name for a in ATTACKS.values() if a.served)
PUBLIC = ("attack", "case", "outcome", "kind", "gate_accepted", "piece", "expected", "ok")


def public(v: dict) -> dict:
    """A verdict as it leaves the node: the outcomes, the attack's and whether the piece named its reason, never the
    piece's why, which names the run's requests."""
    return {k: v[k] for k in PUBLIC} | {"named": ATTACKS[v["attack"]].reason in v["why"]}


def rehearse(port: int, serve_dir: Path, out: Path, names: list[str], flags: list[str], *,
             run: Callable[[list[str]], int] = lambda argv: subprocess.run(argv).returncode,
             clock: Callable[[], float] = time.time) -> list[dict]:
    """Each attack of `names` in turn: `certifier_piece.py front PORT SERVE_DIR OUT/NAME FLAGS --attack RELAY` (and
    ``--tokens`` the attack's, when it names one), the node's clock around it in OUT/NAME/`SPAN`, then `judge`'s
    verdict in OUT/NAME/verdict.json and its public form (`public`) on stdout. OUT/attacks.json gets every verdict,
    OUT/attacks.public.json their public forms."""
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    got = []
    for name in names:
        a, front = ATTACKS[name], out / name
        argv = [sys.executable, str(HERE / "certifier_piece.py"), "front", str(port), str(serve_dir), str(front),
                *flags, "--attack", a.relay, *(["--tokens", str(a.tokens)] if a.tokens is not None else [])]
        start = clock()
        rc = run(argv)
        front.mkdir(parents=True, exist_ok=True)
        (front / SPAN).write_text(json.dumps({"start_unix": start, "end_unix": clock(), "rc": rc}) + "\n")
        try:
            v = judge(name, front, serve_dir)
        except (OSError, ValueError, KeyError) as e:
            v = verdict(name, {}, {"verdict": "missing", "why": f"the front (exit {rc}) left no session: "
                                                                f"{type(e).__name__}: {e}"})
        (front / "verdict.json").write_text(json.dumps(v, indent=1, sort_keys=True) + "\n")
        got.append(v)
        print(f"ATTACK {json.dumps(public(v), sort_keys=True)}", flush=True)
    (out / "attacks.json").write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    (out / "attacks.public.json").write_text(json.dumps([public(v) for v in got], indent=1, sort_keys=True) + "\n")
    (out / "session.json").write_text(json.dumps({"rc": 0 if got and all(v["ok"] for v in got) else 1,
                                                  "ended": f"the rehearsal of {len(got)} attacks ended"}) + "\n")
    return got


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="attacks.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--markdown", action="store_true")
    r = sub.add_parser("relay")
    r.add_argument("--attack", required=True, choices=RELAYED)
    r.add_argument("--listen", type=int, required=True)
    r.add_argument("--to", type=int, required=True)
    r.add_argument("--cell-bytes", type=int, required=True, help="B: the cells' bytes, the schedule's")
    r.add_argument("--bucket-ms", type=int, default=100)
    r.add_argument("--delay", type=int, default=27, help="D: buckets from admission to the first egress cell")
    r.add_argument("--nth", type=int, default=1,
                   help="the completion request, from 1, the attack plays on (jitter and chunk: every connection)")
    r.add_argument("--log", type=Path)
    j = sub.add_parser("judge")
    j.add_argument("name", choices=sorted(a.name for a in ATTACKS.values() if a.served))
    j.add_argument("--front", type=Path, required=True, help="the front's OUT")
    j.add_argument("--serve", type=Path, required=True, help="the serve dir the front kept its run in")
    h = sub.add_parser("rehearse")
    h.add_argument("port", type=int)
    h.add_argument("serve_dir", type=Path)
    h.add_argument("out", type=Path)
    h.add_argument("--attacks", default=",".join(SERVED), help=f"NAME,NAME...: of {','.join(SERVED)} (all of them); "
                   "after them, -- and the fronts' flags")
    return p


def main(argv: list[str] | None = None) -> int:
    argv, flags = sys.argv[1:] if argv is None else list(argv), []
    if argv[:1] == ["rehearse"] and "--" in argv:
        argv, flags = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    p = parser()
    a = p.parse_args(argv)
    if a.cmd == "rehearse":
        names = [x for x in a.attacks.split(",") if x]
        if not names or (bad := sorted(set(names) - set(SERVED))):
            p.error(f"--attacks {a.attacks!r}: names of {','.join(SERVED)}, not {bad or 'none'}")
        got = rehearse(a.port, a.serve_dir.resolve(), a.out.resolve(), names, flags)
        return 0 if all(v["ok"] for v in got) else 1
    if a.cmd == "table":
        print(table(a.markdown))
        return 0
    if a.cmd == "relay":
        asyncio.run(serve_relay(a.attack, a.listen, a.to, cell_bytes=a.cell_bytes, bucket_s=a.bucket_ms / 1000,
                                delay=a.delay, nth=a.nth, log=a.log))
        return 0
    got = judge(a.name, a.front, a.serve)
    print(json.dumps(got, sort_keys=True))
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
