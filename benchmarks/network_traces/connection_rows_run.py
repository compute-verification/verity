"""The connection rows of a `warden_serve` run: a schedule's lengths rows and content rows as flags, the publisher's
keep of each execution phase's hand, the offline check of what it kept, and the cost of a close.

**The flags** (`flags`, `schedule`, `flags_of`). `--lengths 1` gives the schedule lengths rows (`warden`'s "The
lengths rows") and `--content P,O,ROWS ...` its content rows' classes, in order (`content_rows`' "The classes"): a
class's size of prompt ids, its size of ids released, and the rows of each of its three tables. `warden_serve`'s
`serve` and `rehearse` take both, and `rehearse` passes them to its `serve`. Without them the schedule, its digest
and everything the run writes are what they were.

**The hand** (`hand`, `Keeper`). With lengths rows the network certifier hands an execution phase's releases once
the next one's records are published (`Warden.hand_lengths`): on its ticker, after that execution phase's last bucket
has left, or after the pacer has reported it released. The ticker only queues the hand, a slot stride an item, so the
queue's thread pickles a stride's releases at a time and never an execution phase's. The hand carries every release,
filed in a class or not, so its time on the ticker follows the releases and not the classes' rows. The publisher's
process, which alone holds the key, lays the tables (`connection_rows.tables`), commits them and signs their one
head: a line of `rows.jsonl` (the registration, or the execution phase under `refused`, and the cost: the tables,
their rows and bytes, and the build's, the commit's and the signature's ms) and the rows' salts in `rows.salts`
(witness, table by table). It keeps no row: the prover, who holds the records, lays the tables again from them
(`connection_rows.of_records`). The head takes one more leaf of the key an execution phase.

**The check** (`check`). Every execution phase whose three records and the next one's were published complete, and
no other, has a registration that `connection_rows.check_registration` accepts under the network certifier's key and
session, and its head is the head of the tables laid from the records' own reading of the same cells, under the kept
salts: the rows say what the records say.

**The cost** of a close, for a schedule and its classes:

    python connection_rows_run.py cost --out FILE [--classes P,O,ROWS ...] [--load grid|rows] [--phases 5]
        [--cell-bytes 512] [--slots 8] [--lanes 16] [--window 600] [--every 1] [--delay 30] [--ingress-cells 96]
        [--egress-cells 4097] [--height 10]

It loads an execution phase (`loaded`): with `--load grid`, releases of each class in turn, each of its class's
longest request and most ids, until the classes' rows, the lengths table's slots or the cells the schedule's grids
carry in an execution phase are used up; with `--load rows`, every row of every class, whether or not a grid carries
that much. It then times, over `--phases` execution phases: the device's own hand of it to `warden_serve`'s publisher
process (`Warden._emit_lengths`, what the ticker does at the close: `hand_ms`), the pickling of the queue's items,
which the queue's thread does in the ticker's process (`pickle_ms`, and the longest item's), and, in the publisher's
process, the build, the commit and the signature (`rows.jsonl`'s cost). The result is `--out`'s JSON.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import active_replay as ar
import certifier_serve as cs
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.active import VirtualClock
from verity.protocols.certifier.verifier.certificate import Direction, Issuer
from verity.protocols.certifier.verifier.grid import BlockGrid
from verity.protocols.certifier.verifier.warden import Released, Warden, WardenParams, _Connection

#: the publisher queue's name for a hand's items, where a record's rows have their link's
HAND = "connection-rows"
ROWS, SALTS = "rows.jsonl", "rows.salts"
#: the classes the cost is measured at by default: each (P, O) a power of four apart, 16 rows each
CLASSES = tuple((size, size, 16) for size in (2, 16, 64, 256, 1024, 4096))


# The schedule's flags.

def _class(text: str) -> tuple[int, int, int]:
    try:
        p, o, rows = (int(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"a class is P,O,ROWS, three integers, not {text!r}") from None
    return p, o, rows


def flags(q: argparse.ArgumentParser) -> None:
    """The connection rows as `q`'s flags, beside its schedule's."""
    q.add_argument("--lengths", type=int, choices=(1,), default=None,
                   help="the lengths rows' form (warden's \"The lengths rows\"): the network certifier hands each "
                        "execution phase's releases, and the publisher commits and signs their tables (default: none)")
    q.add_argument("--content", type=_class, nargs="+", default=None, metavar="P,O,ROWS",
                   help="with --lengths: the content rows' classes, in order (content_rows' \"The classes\"): a "
                        "class's size of prompt ids, its size of ids released, and its tables' rows (default: none)")


def schedule(params: WardenParams, args: argparse.Namespace) -> WardenParams:
    """`params` with the connection rows `args` names (`flags`); the same schedule where it names none."""
    content = getattr(args, "content", None)
    return replace(params, lengths=getattr(args, "lengths", None), content=content and tuple(content))


def flags_of(params: WardenParams) -> list[str]:
    """The flags that give a schedule `params`' connection rows: none for a schedule without them."""
    out = [] if params.lengths is None else ["--lengths", str(params.lengths)]
    return out + ([] if params.content is None else ["--content", *(",".join(map(str, c)) for c in params.content)])


# The hand, and the publisher's keep of it.

def hand(window: int, strides: Sequence[Sequence[Released]]) -> Iterator[tuple]:
    """An execution phase's hand as the publisher queue's items: one a slot stride that holds a release, then its
    close with the strides' count."""
    for j, stride in enumerate(strides):
        if stride:
            yield HAND, (window, j), stride
    yield HAND, (window, None), len(strides)


class Keeper:
    """The publisher's keep of the hands (the module docstring's "The hand"): called with each item of `hand`, less
    its name, it lays, commits and signs an execution phase's tables at its close. `keep` takes a whole hand, as
    `Warden(lengths=…)` gives it to a publisher in the network certifier's own process."""

    def __init__(self, out: Path, issuer: Issuer, params: WardenParams, link: str) -> None:
        self.out, self.issuer, self.params, self.link = out, issuer, params, link
        self.open: dict[int, dict[int, Sequence[Released]]] = {}

    def __call__(self, at: tuple[int, int | None], data) -> None:
        window, j = at
        if j is not None:
            self.open.setdefault(window, {})[j] = data
            return
        got = self.open.pop(window, {})
        self.keep(window, tuple(tuple(got.get(j, ())) for j in range(data)))

    def keep(self, window: int, strides: Sequence[Sequence[Released]]) -> None:
        t0 = time.perf_counter()
        tables = CR.tables(self.params, strides)
        t1 = time.perf_counter()
        rows = CR.Rows.commit(self.link, window, tables, self.params, self.issuer.session)
        head = rows.head
        t2 = time.perf_counter()
        cert = self.issuer.sign(self.link, Direction.EGRESS, window, head.digest())
        t3 = time.perf_counter()
        cost = {"tables": len(tables), "rows": sum(len(t) for t in tables.values()),
                "row_bytes": sum(len(r) for t in tables.values() for r in t),
                "build_ms": round((t1 - t0) * 1000, 3), "commit_ms": round((t2 - t1) * 1000, 3),
                "sign_ms": round((t3 - t2) * 1000, 3)}
        line = ({"refused": [self.link, Direction.EGRESS.value, window]} if cert is None
                else {"registration": CR.registration(head, cert, self.params)})
        ar.write_blobs(self.out / SALTS, rows.witness)
        with open(self.out / ROWS, "a") as fh:
            fh.write(json.dumps(line | {"cost": cost}) + "\n")


# The offline check.

def read(run: Path, params: WardenParams) -> list[tuple[dict, dict[str, list[bytes]]]]:
    """`rows.jsonl` and `rows.salts`, line by line: each execution phase's registration (or refusal) and cost, and
    its rows' salts by table."""
    path = run / ROWS
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    blobs = ar.read_blobs(run / SALTS)
    listed = CR.tables_of(params)
    size = WR.SALT_BYTES * sum(rows for _, rows, _ in listed)
    if len(blobs) != len(lines) or any(len(b) != size for b in blobs):
        raise ValueError(f"{ROWS} has {len(lines)} lines and {SALTS} {len(blobs)} blobs, each of {size} bytes")
    out = []
    for doc, blob in zip(lines, blobs):
        salts, at = {}, 0
        for name, rows, _ in listed:
            salts[name] = [blob[at + k * WR.SALT_BYTES:at + (k + 1) * WR.SALT_BYTES] for k in range(rows)]
            at += rows * WR.SALT_BYTES
        out.append((doc, salts))
    return out


def _spread(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": round(statistics.median(xs), 3) if xs else None,
            "max": round(max(xs), 3) if xs else None}


def check(run: Path, dep: Mapping, params: WardenParams, ingress: Sequence[BlockGrid],
          egress: Sequence[BlockGrid], fails: Sequence[BlockGrid], link: str) -> tuple[dict | None, list[str]]:
    """The module docstring's check of a run's connection rows under its deployment record `dep`: the summary (the
    registered execution phases, of those that have connection rows, and the cost) and the problems. None, and no
    problem, for a schedule without lengths rows."""
    if params.lengths is None:
        return None, []
    key, session, windows = cs.read_public(dep["key"]), bytes.fromhex(dep["session"]), dep["windows"]
    kept_records = run / "certified.jsonl"
    withheld = {doc["refused"][2] for doc in map(json.loads, kept_records.read_text().splitlines())
                if doc.get("withheld")} if kept_records.exists() else set()
    whole = {w for w in range(windows) if w not in withheld
             and all(any(r.window == w and r.status.reason is None for r in records)
                     for records in (ingress, egress, fails))}
    want = [w for w in range(windows - 1) if w in whole and w + 1 in whole]
    problems: list[str] = []
    try:
        kept = {doc["registration"]["window"] if "registration" in doc else doc["refused"][2]: (doc, salts)
                for doc, salts in read(run, params)}
    except (ValueError, KeyError, TypeError) as e:
        kept = {}
        problems.append(f"connection rows: {ROWS} and {SALTS} don't read: {e}")
    try:
        held = CR.of_records(params, ingress, egress)
    except ValueError as e:
        held = {}
        problems.append(f"connection rows: the records don't read as releases: {e}")
    costs, registered = [], 0
    for w in sorted(set(want) | set(kept)):
        doc, salts = kept.get(w, (None, None))
        if w not in want:
            problems.append(f"execution phase {w} has connection rows, and its records or the next one's weren't "
                            f"published complete")
            continue
        ok = False
        if doc is not None and "registration" in doc and w in held:
            costs.append(doc["cost"])
            try:
                head, _ = CR.check_registration(doc["registration"], key, session, params, link, w)
                ok = CR.Rows.commit(link, w, CR.tables(params, held[w]), params, session, salts).head == head
                if not ok:
                    problems.append(f"execution phase {w}'s connection rows aren't its records' releases")
            except ValueError as e:
                problems.append(f"execution phase {w}'s connection rows: {e}")
        registered += ok
        if not ok:
            problems.append(f"execution phase {w}'s connection rows aren't registered")
    summary = {"registered": registered, "of": len(want), "tables": sorted({c["tables"] for c in costs}),
               "rows": sorted({c["rows"] for c in costs}), "row_bytes": sorted({c["row_bytes"] for c in costs}),
               **{k: _spread([c[k] for c in costs]) for k in ("build_ms", "commit_ms", "sign_ms")}}
    return summary, problems


# The cost of a close.

def loaded(params: WardenParams, load: str) -> list[tuple[int, int]]:
    """An execution phase's releases for the cost, in release order: each one's (request bytes, ids), the longest
    request and the most ids its class holds in the schedule (the module docstring's "The cost")."""
    if params.content is None or load not in ("grid", "rows"):
        raise ValueError("the cost is of a schedule with content rows, loaded to its grid or to its rows")
    B, slots = params.cell_bytes, CR.strides_of(params) * params.stride_releases
    cells = params.slots * params.lanes * params.window
    sizes = [(min(CT.request_room(B, p), params.ingress_cells * params.payload), min(o, params.egress_cells - 1))
             for p, o, _ in params.content]
    left = [rows for _, _, rows in params.content]
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


def _device(params: WardenParams, releases: Sequence[tuple[int, int]], lengths) -> Warden:
    """A network certifier at the close of execution phase 1 with `releases` as execution phase 0's, each ended in
    its `END`, laid in its slot strides in order. It is never run, so its clock is one nothing moves."""
    clock = VirtualClock(0.0)
    device = Warden(params, None, origin=0.0, windows=2, clock=clock, sleep=clock.sleep, lengths=lengths)
    K = params.stride_releases
    made = [_Connection(n + 1, None, None, None, 0, release=n, bytes_in=size, egress_cells=ids + 1, ended=0,
                        request=bytes([n % 251 + 1]) * size, ids=bytearray((n % 251 + 1).to_bytes(4, "little") * ids))
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
        cs.load_key(key, height)
        publish = ws.PublisherProcess(out, key, bytes(32), params, base_ms=0)  # a scratch key's first session
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
    c.add_argument("--classes", type=_class, nargs="+", default=list(CLASSES), metavar="P,O,ROWS")
    c.add_argument("--load", choices=("grid", "rows"), default="grid")
    c.add_argument("--phases", type=int, default=5)
    c.add_argument("--height", type=int, default=cs.HEIGHT, help="the LMS key's tree height")
    for name, default in (("cell_bytes", 512), ("slots", 8), ("lanes", 16), ("window", 600), ("every", 1),
                          ("delay", 30), ("ingress_cells", 96), ("egress_cells", 4097)):
        c.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    params = WardenParams(cell_bytes=args.cell_bytes, slots=args.slots, lanes=args.lanes, window=args.window,
                          every=args.every, delay=args.delay, ingress_buckets=args.delay,
                          ingress_cells=args.ingress_cells, egress_cells=args.egress_cells, lengths=1,
                          content=tuple(args.classes))
    result = cost(params, args.load, args.phases, args.height)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
