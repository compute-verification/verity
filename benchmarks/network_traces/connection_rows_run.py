"""The connection rows of a `warden_serve` run: a schedule's lengths rows and content rows as flags, the publisher's
keep of each execution phase's hand, the offline check of what it kept, and the cost of a close.

**The flags** (`flags`, `schedule`, `flags_of`). `--lengths 1` or `--lengths 2` gives the schedule lengths rows
of that form (`warden`'s "The lengths rows") and `--content P,O[,ROWS] ...` its content rows' classes, in order
(`content_rows`' "The classes"): a class's size of prompt ids, its size of ids released and, under form 1 and only
there, the rows of each of its three tables. Under form 2 a class is `P,O`: its tables hold a row a release filed in
it and no other. `--request-words`, with `--lengths 2`, `--content` and the schedule's `--max-tokens-cap`, gives it
request words (`content_rows`' "A request's words"): a fourth table a class, each released request's `max_tokens`
and count of prompt ids as the network certifier reads them. `warden_serve`'s `serve` and `rehearse` take the three
flags, and `rehearse` passes them to its `serve`. Without them the schedule, its digest and everything the
run writes are what they were.

**The hand** (`hand`, `Keeper`). With lengths rows the network certifier hands an execution phase's releases once
the next one's records are published (`Warden.hand_lengths`): on its ticker, after that execution phase's last bucket
has left, or after the pacer has reported it released. The ticker only queues the hand, a slot stride an item, so the
queue's thread pickles a stride's releases at a time and never an execution phase's. The hand carries every release,
filed in a class or not, so its time on the ticker follows the releases and not the classes' rows. The publisher's
process, which alone holds the key, lays the tables (`connection_rows.tables`), commits them and signs their one
head: a line of `rows.jsonl` (the registration, or the execution phase under `refused`, and the cost: the tables,
their rows and bytes, and the build's, the commit's and the signature's ms) and the rows' salts in `rows.salts`
(witness, table by table). It keeps no row: the prover, who holds the records, lays the tables again from them
(`connection_rows.of_records`). The head takes one more leaf of the key an execution phase. A hand the tables refuse
has no line and no salts: under form 2 one that files more than R = `WardenParams.phase_releases` releases in a
class is no execution phase's of the schedule, the keep raises naming the release, the class and R, and `serve`'s
publisher process ends on it, which fails the run at its next record's close.

Under form 2 the network certifier ends every execution phase of the run, the last at the run's last close, and one
it doesn't hand without releases. For that one the publisher's process signs the word that it has no head
(`connection_rows`' "The word that no head comes"): its line of `rows.jsonl` holds the word's record under
`no_head` in place of a registration, and its blob of `rows.salts` is empty. So every execution phase has one line
and one blob, in the run's order, and takes one leaf of the key. Under form 2 a class's tables are its held rows,
so a line's rows, its blob of salts and the publisher's build and commit follow the execution phase's releases, and
`read` takes each table's salts by the rows the line's registration counts.

**The check** (`check`). Every execution phase that is handed, and no other, has a registration that
`connection_rows.check_registration` accepts under the network certifier's key and session, and its head is the head
of the tables laid from the records' own reading of the same cells, under the kept salts: the rows say what the
records say. Under form 1 the handed ones are those whose three records and the next one's were published complete.
Under form 2 they are those whose records, the ones before theirs and the ones after were, and every other
execution phase of the run has the word that it has no head, which `connection_rows.check_no_head` accepts: an
execution phase with neither, with both, or with the one where the other is due is a problem. The reading takes each
release's place in the release order from the run's `warden.json` where there is one, since the records stop
counting it at an execution phase that isn't complete.

**The cost** of a close, for a schedule and its classes:

    python connection_rows_run.py cost --out FILE [--lengths 1|2] [--classes P,O[,ROWS] ...] [--request-words]
        [--load grid|rows] [--phases 5] [--cell-bytes 512] [--slots 8] [--lanes 16] [--window 600] [--every 1]
        [--delay 30] [--ingress-cells 96] [--egress-cells 4097] [--height 10]

It loads an execution phase (`loaded`): with `--load grid`, releases of each class in turn, each of its class's
longest request and most ids, until the classes' rows, the lengths table's slots or the cells the schedule's grids
carry in an execution phase are used up; with `--load rows`, every row of every class, whether or not a grid carries
that much. `--lengths` is the lengths rows' form, 1 by default. Under 2 the classes are `P,O`, the default ones the
same sizes, and an execution phase holds at most R = `WardenParams.phase_releases` releases over all the classes:
`--load grid` stops there too, and `--load rows` is R releases of the one class whose rows are the most bytes, each
its longest request and most ids: a ceiling, whether or not an execution phase releases that much. With
`--request-words` (under `--lengths 2`) the schedule has request words under the greatest response cap it takes, and
each release's request is the canonical id request of its class's most prompt ids, which the publisher's build
reads the words of. It then times, over `--phases` execution phases: the device's own hand of it to
`warden_serve`'s publisher process (`Warden._emit_lengths`, what the ticker does at the close: `hand_ms`), the
pickling of the queue's items, which the queue's thread does in the ticker's process (`pickle_ms`, and the longest
item's), and, in the publisher's process, the build, the commit and the signature (`rows.jsonl`'s cost). The result
is `--out`'s JSON.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import certifier_key as ck
from verity.protocols.accounting.certifier.verifier import connection_rows as CR
from verity.protocols.accounting.certifier.verifier import content_rows as CT
from verity.protocols.accounting.certifier.verifier.id_form import id_request
from verity.protocols.accounting.certifier.verifier.run_clock import VirtualClock
from verity.protocols.accounting.certifier.verifier.warden import LENGTHS_FORMS, Released, Warden, WardenParams, _Connection
from runner_parts import _class, hand
from verity.protocols.accounting.certifier.verifier.connection_rows_check import ROWS, _spread

#: the classes the cost is measured at by default: each (P, O) a power of four apart, 16 rows each under form 1
CLASSES = tuple((size, size, 16) for size in (2, 16, 64, 256, 1024, 4096))


# The schedule's flags.


# The hand, and the publisher's keep of it.


# The offline check.


# The cost of a close.

def loaded(params: WardenParams, load: str) -> list[tuple[int, int]]:
    """An execution phase's releases for the cost, in release order: each one's (request bytes, ids), the longest
    request and the most ids its class holds in the schedule (the module docstring's "The cost"). Where a class's
    rows are a count (`content_rows.counted`) the releases are at most the execution phase's, `phase_releases`, and
    `rows` is that many of the one class whose rows are the most bytes. In a schedule with request words a
    release's request is a canonical id request, so a class whose longest one the schedule's ingress doesn't take
    is refused."""
    if params.content is None or load not in ("grid", "rows"):
        raise ValueError("the cost is of a schedule with content rows, loaded to its grid or to its rows")
    B, slots = params.cell_bytes, CR.strides_of(params) * params.stride_releases
    if CT.counted(params):
        slots = min(slots, params.phase_releases)
    cells = params.slots * params.lanes * params.window
    listed = CT.sized(params)
    most = params.ingress_cells * params.payload
    sizes = [(min(CT.request_room(B, p), most), min(o, params.egress_cells - 1)) for p, o, _ in listed]
    if params.request_words:
        for i, (p, _, _) in enumerate(listed):
            if CT.request_room(B, p) > most:
                raise ValueError(f"with request words the cost's requests are canonical id requests, and class {i}'s "
                                 f"of {p} prompt ids is {CT.request_room(B, p)} bytes, past the {most} the "
                                 f"schedule's ingress takes: more ingress_cells, or a class of fewer prompt ids")
    if CT.counted(params) and load == "rows":
        own = [i for i, (size, n) in enumerate(sizes)
               if CT.file(params, Released(0, size, n, False, bytes(size), bytes(4 * n))) == i]
        widest = max(own, key=lambda i: CT.request_bytes(B, listed[i][0]) + CT.response_bytes(listed[i][1]))
        return [sizes[widest]] * slots
    left = [rows for _, _, rows in listed]
    out: list[tuple[int, int]] = []
    request_cells = ids = 0
    while len(out) < slots and any(left):
        for i, (size, n) in enumerate(sizes):
            if not left[i] or len(out) >= slots or CT.file(params, Released(0, size, n, False, bytes(size),
                                                                             bytes(4 * n))) != i:
                left[i] = 0
                continue
            needs = -(-size // params.payload)
            if load == "grid" and (request_cells + needs > cells or ids + n + 1 > cells):
                left[i] = 0
                continue
            request_cells, ids, left[i] = request_cells + needs, ids + n + 1, left[i] - 1
            out.append((size, n))
    return out


def _request(params: WardenParams, n: int, size: int, ids: int) -> bytes:
    """Release n's request for the cost, of `size` bytes: any bytes, or in a schedule with request words the
    canonical id request of that length, of its class's prompt ids, whose `max_tokens` is the release's ids."""
    if not params.request_words:
        return bytes([n % 251 + 1]) * size
    B = params.cell_bytes
    prompt = next(p for p, _, _ in CT.sized(params) if CT.request_room(B, p) == size)
    return id_request(model="", user="", prompt=[n] * prompt, max_tokens=ids, frame_bytes=B, offset=0)


def _device(params: WardenParams, releases: Sequence[tuple[int, int]], lengths) -> Warden:
    """A network certifier at the close of execution phase 1 with `releases` as execution phase 0's, each ended in
    its `END`, laid in its slot strides in order. Under form 2 its run goes on past execution phase 1, so that
    close hands execution phase 0's rows and ends no other. It is never run, so its clock is one nothing moves."""
    clock = VirtualClock(0.0)
    device = Warden(params, None, origin=0.0, windows=3 if params.closes else 2, clock=clock, sleep=clock.sleep,
                    lengths=lengths)
    K = params.stride_releases
    made = [_Connection(n + 1, None, None, None, 0, release=n, bytes_in=size, egress_cells=ids + 1, ended=0,
                        request=_request(params, n, size, ids),
                        ids=bytearray((n % 251 + 1).to_bytes(4, "little") * ids))
            for n, (size, ids) in enumerate(releases)]
    strides = [made[j * K:(j + 1) * K] for j in range(CR.strides_of(params))]
    device._released[0], device._whole[0] = strides, True
    return device


def cost(params: WardenParams, load: str, phases: int, height: int) -> dict:
    """The module docstring's cost of a close of `params` under `load`, over `phases` execution phases."""
    import warden_serve as ws  # at its use: it imports this module

    releases = loaded(params, load)
    hands, pickles, longest = [], [], []
    with tempfile.TemporaryDirectory(prefix="connection-rows-cost-") as scratch:
        out, key = Path(scratch) / "run", Path(scratch) / "key"
        out.mkdir()
        ck.load_key(key, height)
        publish = ws.PublisherProcess(out, key, bytes(32), params)
        try:
            for phase in range(phases):
                handed: list = []

                def lengths(window: int, strides, phase=phase, handed=handed) -> None:
                    handed.append(strides)
                    publish.lengths(phase, strides)

                device = _device(params, releases, lengths)
                t0 = time.perf_counter()
                device._emit_lengths(1, True)
                hands.append((time.perf_counter() - t0) * 1000)
                each = []
                for item in hand(phase, handed[0]):
                    t0 = time.perf_counter()
                    pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
                    each.append((time.perf_counter() - t0) * 1000)
                pickles.append(sum(each))
                longest.append(max(each))
        finally:
            publish.close()
        kept = [json.loads(x)["cost"] for x in (out / ROWS).read_text().splitlines()]
    return {"parameters": params.digest.hex(), "classes": [list(c) for c in params.content], "load": load,
            "releases": len(releases), "handed_bytes": sum(size + 4 * ids for size, ids in releases),
            "tables": kept[0]["tables"], "rows": kept[0]["rows"], "row_bytes": kept[0]["row_bytes"],
            "phases": phases, "hand_ms": _spread(hands), "pickle_ms": _spread(pickles),
            "pickle_longest_item_ms": _spread(longest),
            **{k: _spread([c[k] for c in kept]) for k in ("build_ms", "commit_ms", "sign_ms")},
            "publisher_usage": publish.usage}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="connection_rows_run.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cost", help="the cost of a close: the hand, the build, the commit and the signature")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--lengths", type=int, choices=LENGTHS_FORMS, default=1, help="the lengths rows' form")
    c.add_argument("--classes", type=_class, nargs="+", default=None, metavar="P,O[,ROWS]",
                   help="the content rows' classes: P,O,ROWS under --lengths 1 and P,O under --lengths 2 (default: "
                        "six classes a power of four apart, of 16 rows each under --lengths 1)")
    c.add_argument("--request-words", action="store_true",
                   help="under --lengths 2: the schedule has request words, under a response cap of its most ids "
                        "(egress_cells − 1), and each release's request is a canonical id request")
    c.add_argument("--load", choices=("grid", "rows"), default="grid")
    c.add_argument("--phases", type=int, default=5)
    c.add_argument("--height", type=int, default=ck.HEIGHT, help="the LMS key's tree height")
    for name, default in (("cell_bytes", 512), ("slots", 8), ("lanes", 16), ("window", 600), ("every", 1),
                          ("delay", 30), ("ingress_cells", 96), ("egress_cells", 4097)):
        c.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)
    return ap


def costed(args: argparse.Namespace) -> WardenParams:
    """The schedule the `cost` command's flags name."""
    classes = args.classes or (CLASSES if args.lengths == 1 else [c[:2] for c in CLASSES])
    words = {"max_tokens_cap": args.egress_cells - 1, "request_words": True} if args.request_words else {}
    return WardenParams(cell_bytes=args.cell_bytes, slots=args.slots, lanes=args.lanes, window=args.window,
                        every=args.every, delay=args.delay, ingress_buckets=args.delay,
                        ingress_cells=args.ingress_cells, egress_cells=args.egress_cells, lengths=args.lengths,
                        content=tuple(classes), **words)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    params = costed(args)
    result = cost(params, args.load, args.phases, args.height)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
