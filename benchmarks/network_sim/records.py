"""The real certifier's published records as the accounting layer's `datacenter.observe.Record`s, and their audit.

Each certifier record (`verity.core.protocols.certifier.grid.Record`: one link-window's grid and status) becomes a
`Record`: the link by its ends in the direction of travel (egress from the block's end of its uplink to the certifier,
ingress back), the window, the real frames in each bucket (the grid's occupancy), the certifier, and a SHA-512 digest of
the status and the real frames in order. `published` reads a block's from a lab run's directory. `declared` gives the
ones the Program says an honest run publishes: egress, the reference shaper on the Program's frames at the declared
syncs; ingress, the certifier's bucketer on the plan's requests, each at its arrival bucket.

`datacenter.claims.audit` reads a `Trace`, whose expected digests come from a placed circuit's crossings, and a lab run's
traffic is a replay Program instead. So `audit` is that function's record loop against the declared Records, with its
claims and its reasons: a link-window without a record, or whose counts or digest aren't the declared ones, is
`Refused`; any other is `Carried`, its only freedom the window's sync advice, log2 of the sync set's size bits.
`profile` gives the declared occupancy in the form `claims.audit` takes.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "network_traces")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import active_replay as ar  # noqa: E402
import graph  # noqa: E402, F401  (puts `datacenter` on the path)
from datacenter import Datacenter  # noqa: E402
from datacenter.claims import Carried, Refused  # noqa: E402
from datacenter.observe import Record  # noqa: E402
from verity.core.protocols.certifier import grid  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync, run_shaper  # noqa: E402

DOMAIN = b"network_sim/certifier-record/v1\0"
NO_RECORD = "no record: the certifier was down or its record was withheld"
OCCUPANCY = "occupancy differs from the declared profile"
CONTENT = "a frame is not a committed crossing"


def digest(r: grid.Record) -> bytes:
    """The record's status and its real frames in grid order. Frames are all B bytes and the counts are the record's
    own, so the bytes hashed fix the grid."""
    h = hashlib.sha512(DOMAIN)
    h.update((r.status.reason.value if r.status.reason else "").encode() + b"\0")
    for row in r.grid:
        for slot in row[: grid.row_count(row)]:
            h.update(slot)
    return h.digest()


def to_record(r: grid.Record, link: tuple[str, str], certifier: str) -> Record:
    return Record(link, r.window, grid.occupancy(r.grid), digest(r), certifier)


def uplink(dc: Datacenter, block: str) -> tuple[str, str]:
    """The block's one boundary link, as its end in the block and the certifier at its other end."""
    ours = [e for e in dc.boundary if any((b := dc.block_of(x)) is not None and b.name == block for x in (e.a, e.b))]
    if len(ours) != 1:
        raise ValueError(f"block {block} has {len(ours)} boundary links: the lab runs one certifier on one uplink")
    (e,) = ours
    inside = e.a if dc.block_of(e.a) is not None else e.b
    return inside, dc.certifier_of(e).name


def directions(dc: Datacenter, block: str, setup: ar.Setup) -> tuple[dict[str, tuple[str, str]], str]:
    """Each certifier link's ends in its direction of travel, by the certifier's name for it, and the certifier."""
    inside, cert = uplink(dc, block)
    return {setup.egress_link: (inside, cert), setup.ingress_link: (cert, inside)}, cert


def published(run: Path, setup: ar.Setup, dc: Datacenter, block: str) -> list[Record]:
    """The records a block's certifier published in a lab run, from its directory."""
    ends, cert = directions(dc, block, setup)
    out = []
    for path, params in ((run / "responses.records", setup.egress_params), (run / "requests.records", setup.ingress_params)):
        for blob in ar.read_blobs(path):
            r = ar.unpack_record(blob, params.frame_bytes)
            out.append(to_record(r, ends[r.link], cert))
    return out


def declared(plan: dict, setup: ar.Setup, dc: Datacenter, block: str) -> list[Record]:
    """The records an honest run of `plan` publishes, from the plan alone."""
    ends, cert = directions(dc, block, setup)
    e, i = setup.egress_params, setup.ingress_params
    release = {int(k): v for k, v in plan["model_release"].items()}
    frames, ready = ar.program(plan, release, e)
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    out = [to_record(r, ends[r.link], cert) for r in run_shaper(frames, ready, syncs, grid.Certifier(setup.egress_link, e))]
    peer = ar.Peer(plan, ar.wire_params(setup))
    arrivals: dict[int, list[bytes]] = defaultdict(list)
    for q in sorted(plan["requests"], key=lambda q: (q["enqueue"], q["send_ms"], q["stream"])):
        data = ar.request_frames(q["stream"], peer.request(q), peer.B)
        arrivals[q["enqueue"]] += [data[k : k + peer.B] for k in range(0, len(data), peer.B)]
    bucketer = grid.Certifier(setup.ingress_link, i)
    for t in range(plan["windows"] * i.buckets):
        for f in arrivals.get(t, []):
            bucketer.enqueue(f)
        bucketer.tick()
        if (t + 1) % i.buckets == 0:
            out.append(to_record(bucketer.close_window(), ends[setup.ingress_link], cert))
    return out


def audit(records: list[Record], expected: list[Record], syncs: int) -> list[Carried | Refused]:
    """`claims.audit`'s record loop: each expected link-window needs a record, and each record the expected counts and
    digest; a sync set of `syncs` values leaves each carried window log2 syncs bits of advice."""
    want = {(d.link, d.window): d for d in expected}
    have = {(r.link, r.window) for r in records}
    out: list[Carried | Refused] = [Refused(f"link {link} window {w}", NO_RECORD) for link, w in want if (link, w) not in have]
    for r in records:
        d = want.get((r.link, r.window))
        where = f"link {r.link} window {r.window}"
        if d is None or r.counts != d.counts:
            out.append(Refused(where, OCCUPANCY))
        elif r.digest != d.digest:
            out.append(Refused(where, CONTENT))
        else:
            out.append(Carried(r.link, r.window, math.log2(syncs)))
    return out


def profile(records: list[Record]) -> dict[tuple[str, str], dict[int, int]]:
    """The occupancy the records state, each link's buckets numbered from its first window's first."""
    out: dict[tuple[str, str], dict[int, int]] = {}
    for r in records:
        row = out.setdefault(r.link, {})
        row.update({r.window * len(r.counts) + t: n for t, n in enumerate(r.counts) if n})
    return out


def to_json(r: Record) -> dict:
    return {"link": list(r.link), "window": r.window, "counts": list(r.counts), "digest": r.digest.hex(),
            "certifier": r.certifier}
