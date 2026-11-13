"""A response open at a record close continues after it: the released-count stamps the network certifier writes at
the close, and the continuation the reference builds from each.

**The edge** is a record close at a public bucket b inside one network certifier session: the session's schedule,
ordinals and release counter run on through it. Only what the records before b certify crosses it. A response still
open at b is cut there, and its request continues after b as an ordinary arrival: the same request with the ids
already released appended to its prompt, and that many fewer tokens left.

**The stamps.** A request is *released* once its last request cell is in an ingress bucket, and N numbers the released
requests in the order of those cells (by bucket, then as the ingress record holds them). For a request released
before b, its response

- has e id cells in egress buckets before b. A cell counts by the slot the schedule releases it in, which is the
  bucket the egress record holds it in: never by when its bytes reached the certifier, by when or whether the client
  read them, nor by what the server had computed by then (an id the server has sent whose slot is at or after b
  doesn't count);
- is *open* at b when no end cell of its (the network certifier's `END` or `FAIL`) is in an egress bucket before b.

At b the certifier stamps each open response (q, N, e), q its ordinal, in N order: `stamps`. A stamp is structural:
the certifier reads its cells' kinds, ordinals and buckets, and no request field and no id. Every part is read from
the records before b, so the stamps are a function of them and of b alone: no party chooses a stamp, no cell at or
after b moves one, and a request whose cells straddle b has no N yet, so no stamp, and arrives after b as any request
does. A session's first close, b = 0, has no stamp.

**The continuations.** The reference reads what the records certify of each stamped response, its request and its e
released ids, and splits them (`continuations`). One whose ids hold one of the deployment's end-of-sequence ids, or
whose e is its request's `max_tokens`, is *finished*: it is stamped while its end cell isn't before b, and never
continued. One whose e exceeds its `max_tokens` is refused (`over`). Every other one continues: its continuation is
its certified request with the e released ids after its prompt and `max_tokens` − e in place of `max_tokens`, the pair
(q, P + e) for a prompt of P ids, the token row prompt ‖ released ids, and its model, user, seed, temperature and
top_p unchanged. e counts from the request's release, across every close before b, so the continuation at a later
close is the same whether it is built from the request or from an earlier close's continuation (`resumed`), and the
outputs a response adds between two closes are the difference of its two stamps. 0 < `max_tokens` − e ≤ `max_tokens`
always, so no continuation asks for nothing, or for more than its certified request did, whatever bound that was held
to.

**Fail closed.** `check` holds the stamps a record carries to `stamps`' and refuses, by name: a stamp of a response
that has an end cell before b or was never released (`extra`), an open response with none, finished or not
(`missing`), a response stamped twice (`duplicate`), another N than the records' (`release`), another e (`count`), and
the right stamps out of N order (`order`). Records the rule can't read are refused too (`records`, below), and a
stamped response past its `max_tokens` (`over`).

**The records.** `Response` is one released request and its response as certified records give them, whatever their
form. `read` reads the network certifier's own records (`warden.py`'s schedule in the id egress form: `Record`s of
`WardenParams.window` buckets) up to b, through `warden.read_cell`, `io_tie.parse_cell` and `active.id_request`: a
request is its connection's `DATA` cells and its `END`, released at the `END`'s bucket, and its response's id cells
are its `DATA` cells, one id event each. It reads no bucket at or after b, and refuses a close that isn't a whole
number of complete execution phases the records cover, a cell that isn't the certifier's, a request that isn't a
canonical id request, an egress cell out of the id form, an id cell before its request's release or after its end
cell, and a request cell after its `END`. That each cell is in its schedule's slot is `warden.audit`'s check, which
this one doesn't repeat. `edge` is the rule on those records: the stamps and the continuations at b.
"""
from __future__ import annotations

import json
from typing import Collection, NamedTuple, Sequence

from verity.protocols.certifier.verifier import warden as W
from verity.protocols.certifier.verifier.active import DATA, PAD, id_request
from verity.protocols.certifier.verifier.grid import BlockGrid, row_count
from verity.protocols.certifier.verifier.io_tie import parse_cell

#: why the rule refuses: what `check` names in a record's stamps, then what the records themselves show
REFUSALS = ("extra", "missing", "duplicate", "release", "count", "order", "over", "records")


class Refused(ValueError):
    """The rule's refusal: `name` is one of `REFUSALS`."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name = name


# --- the reference ---------------------------------------------------------------------------------------------------

class Request(NamedTuple):
    """A certified id request's fields, `active.id_request`'s keywords: `id_request(**r._asdict(), frame_bytes=B,
    offset=0)` is the request in the network certifier's cells."""

    model: str
    user: str
    prompt: tuple[int, ...]
    max_tokens: int
    temperature: float
    top_p: float
    seed: int


class Response(NamedTuple):
    """One released request and its response in certified records: its ordinal q, its release number N, the bucket its
    request was released in, the request, each id cell of its response as (egress bucket, id) in the order released,
    and the bucket of its end cell where the records hold one."""

    q: int
    n: int
    released: int
    request: Request
    cells: tuple[tuple[int, int], ...]
    ended: int | None = None


class Stamp(NamedTuple):
    """What the network certifier stamps for a response open at a close: its ordinal, its release number and its id
    cells before the close."""

    q: int
    n: int
    e: int


class Continuation(NamedTuple):
    """A stamped response's request after the close: its stamp's (q, N, e) and the request that continues it, whose
    prompt is the token row (the certified prompt, then the e released ids) and whose `max_tokens` is the certified
    one less e. (q, P + e) is (q, len(request.prompt))."""

    q: int
    n: int
    e: int
    request: Request


class Edge(NamedTuple):
    """A close's stamps and the continuations of the stamped responses that aren't finished, both in N order."""

    stamps: tuple[Stamp, ...]
    continuations: tuple[Continuation, ...]


def released(r: Response, b: int) -> tuple[int, ...]:
    """The ids of r's cells in buckets before b: e is their number."""
    return tuple(x for t, x in r.cells if t < b)


def is_open(r: Response, b: int) -> bool:
    """Whether r is open at b: released before b, with no end cell in a bucket before b."""
    return r.released < b and not (r.ended is not None and r.ended < b)


def finished(r: Response, b: int, eos: Collection[int]) -> bool:
    """Whether the ids of r released before b finish it: one of them is an end-of-sequence id, or they are its
    request's `max_tokens`. Raises `Refused` (`over`) where they outnumber its `max_tokens`."""
    ids = released(r, b)
    if len(ids) > r.request.max_tokens:
        raise Refused("over", f"response {r.q} has {len(ids)} id cells before bucket {b}, past its max_tokens "
                              f"({r.request.max_tokens})")
    return len(ids) == r.request.max_tokens or any(x in eos for x in ids)


def continues(r: Response, b: int, eos: Collection[int]) -> bool:
    """Whether r continues after b: open at b and not finished. Raises `Refused` (`over`) as `finished` does, for a
    response open at b."""
    return is_open(r, b) and not finished(r, b, eos)


def _checked(responses: Sequence[Response]) -> list[Response]:
    """The responses in N order, refused (`records`) where two share an ordinal or a release number, a release order
    isn't the releases' bucket order, or a response's cells aren't in bucket order from its release."""
    out = sorted(responses, key=lambda r: r.n)
    if len({r.q for r in out}) != len(out) or len({r.n for r in out}) != len(out):
        raise Refused("records", "two responses share an ordinal or a release number")
    if any(a.released > c.released for a, c in zip(out, out[1:])):
        raise Refused("records", "the release numbers aren't in the order of the releases' buckets")
    for r in out:
        buckets = [t for t, _ in r.cells]
        if buckets != sorted(buckets) or (buckets and buckets[0] < r.released):
            raise Refused("records", f"response {r.q}'s id cells aren't in bucket order from its release")
    return out


def stamps(responses: Sequence[Response], b: int) -> tuple[Stamp, ...]:
    """The stamps at b, in N order: (q, N, e) of each response open at b. It reads no request and no id."""
    return tuple(Stamp(r.q, r.n, len(released(r, b))) for r in _checked(responses) if is_open(r, b))


def continuations(responses: Sequence[Response], b: int, eos: Collection[int]) -> tuple[Continuation, ...]:
    """The continuations at b, in N order, `eos` the deployment's end-of-sequence ids: each stamped response that
    isn't finished, as its request with its released ids after its prompt and `max_tokens` less their number. Raises
    `Refused` (`over`) for a stamped response past its `max_tokens`."""
    out = []
    for r in _checked(responses):
        if continues(r, b, eos):
            ids = released(r, b)
            out.append(Continuation(r.q, r.n, len(ids), r.request._replace(
                prompt=r.request.prompt + ids, max_tokens=r.request.max_tokens - len(ids))))
    return tuple(out)


def resumed(r: Response, b: int, eos: Collection[int]) -> Response:
    """A response that continues after b as the records after b give it against its continuation: the continuation's
    request, released at b, and the id cells from b on. Its stamp at a later close counts only those, and its
    continuation there is the one the response's own stamp gives."""
    if not continues(r, b, eos):
        raise ValueError(f"response {r.q} doesn't continue after bucket {b}")
    ids = released(r, b)
    return r._replace(released=b, cells=r.cells[len(ids):], request=r.request._replace(
        prompt=r.request.prompt + ids, max_tokens=r.request.max_tokens - len(ids)))


def check(written: Sequence[Stamp], responses: Sequence[Response], b: int) -> None:
    """Holds the stamps a record carries at b to the records': raises `Refused`, by the module docstring's names,
    unless `written` is `stamps(responses, b)`."""
    want = stamps(responses, b)
    written = [Stamp(*s) for s in written]
    got, by_q = [s.q for s in written], {s.q: s for s in want}
    if len(set(got)) != len(got):
        raise Refused("duplicate", f"response {next(q for q in got if got.count(q) > 1)} is stamped twice")
    for s in written:
        if s.q not in by_q:
            raise Refused("extra", f"response {s.q} is stamped and isn't open at bucket {b}: it has an end cell "
                                   "before it, or no release")
    for s in want:
        if s.q not in got:
            raise Refused("missing", f"response {s.q} is open at bucket {b} and has no stamp")
    for s in written:
        if s.n != by_q[s.q].n:
            raise Refused("release", f"response {s.q} is stamped release {s.n}; the records' is {by_q[s.q].n}")
        if s.e != by_q[s.q].e:
            raise Refused("count", f"response {s.q} is stamped {s.e} id cells before bucket {b}; the records hold "
                                   f"{by_q[s.q].e}")
    if tuple(written) != want:
        raise Refused("order", "the stamps aren't in release order")


# --- the network certifier's records ---------------------------------------------------------------------------------

def _request(q: int, data: bytes, cell_bytes: int) -> Request:
    """The fields of connection q's request, a canonical id request in the network certifier's cells (`id_request` at
    offset 0). Raises `Refused` (`records`) for other bytes."""
    try:
        body = json.loads(data.partition(b"\r\n\r\n")[2])
        r = Request(**{k: body[k] for k in Request._fields})._replace(prompt=tuple(body["prompt"]))
        canonical = id_request(**r._asdict(), frame_bytes=cell_bytes, offset=0)
    except (ValueError, KeyError, TypeError) as e:
        raise Refused("records", f"connection {q}'s request isn't a canonical id request ({e})") from None
    if canonical != data:
        raise Refused("records", f"connection {q}'s request isn't a canonical id request: its bytes aren't the ones "
                                 "its fields give")
    return r


def _buckets(params: W.WardenParams, records: Sequence[BlockGrid], b: int, what: str):
    """Each bucket before b of a link's records with its cells, in bucket order."""
    T, by_window = params.window, {rec.window: rec for rec in records}
    if len(by_window) != len(records):
        raise Refused("records", f"two {what} records of one execution phase")
    for w in range(b // T):
        rec = by_window.get(w)
        if rec is None or not rec.status.complete or len(rec.grid) != T:
            raise Refused("records", f"no complete {what} record of execution phase {w}, before bucket {b}")
        for i, row in enumerate(rec.grid):
            t, out = w * T + i, []
            for slot in row[:row_count(row)]:
                try:
                    out.append((slot, *W.read_cell(slot, params.cell_bytes)))
                except ValueError as e:
                    raise Refused("records", f"{what} bucket {t}: {e}") from None
            yield t, out


def read(params: W.WardenParams, ingress: Sequence[BlockGrid], egress: Sequence[BlockGrid], b: int) -> list[Response]:
    """The requests released before b and their responses' cells before b, in release order, from the network
    certifier's ingress and egress records (the module docstring's "The records"). Raises `Refused` (`records`)."""
    if type(b) is not int or b < 0 or b % params.window:
        raise Refused("records", f"a record closes at a whole number of execution phases of {params.window} buckets, "
                                 f"not at bucket {b!r}")
    if params.egress != "id":
        raise Refused("records", f"a response's ids are read in the id egress form, not {params.egress!r}")
    data: dict[int, list[bytes]] = {}
    at: dict[int, tuple[int, int]] = {}
    for t, cells in _buckets(params, ingress, b, "ingress"):
        for _, kind, q, payload in cells:
            if kind == PAD:
                continue
            if kind not in (DATA, W.END) or q in at:
                raise Refused("records", f"ingress bucket {t}: connection {q}'s cell isn't its request's next")
            data.setdefault(q, []).append(payload)
            if kind == W.END:
                at[q] = (len(at), t)
    requests = {q: _request(q, b"".join(data[q]), params.cell_bytes) for q in at}
    ids: dict[int, list[tuple[int, int]]] = {q: [] for q in at}
    ended: dict[int, int] = {}
    for t, cells in _buckets(params, egress, b, "egress"):
        for slot, kind, q, _ in cells:
            if kind == PAD:
                continue
            if kind in (W.END, W.FAIL):
                if q in ended or (kind == W.END and parse_cell(slot) is None):
                    raise Refused("records", f"egress bucket {t}: connection {q}'s end cell isn't its one end")
                ended[q] = t
                continue
            got = parse_cell(slot) if kind == DATA else None
            if got is None or q in ended or q not in at or t < at[q][1]:
                raise Refused("records", f"egress bucket {t}: connection {q}'s cell isn't an id cell of a released "
                                         "request's response before its end")
            ids[q].append((t, got.id))
    return [Response(q, n, t, requests[q], tuple(ids[q]), ended.get(q)) for q, (n, t) in at.items()]


def edge(params: W.WardenParams, ingress: Sequence[BlockGrid], egress: Sequence[BlockGrid], b: int,
         eos: Collection[int]) -> Edge:
    """The close at b on the network certifier's records: the stamps it writes and the continuations they give,
    `eos` the deployment's end-of-sequence ids."""
    responses = read(params, ingress, egress, b)
    return Edge(stamps(responses, b), continuations(responses, b, eos))
