"""The real certifier's published records as the accounting layer's `datacenter.observe.Record`s.

Each certifier record (`verity.core.protocols.certifier.grid.Record`: one link-window's grid and status) becomes a
`Record`: the port (the block's certifier, `egress` for what leaves the block and `ingress` for what enters it), the
window, the real frames in each bucket (the grid's occupancy) and a SHA-512 digest of the status and the real frames in
order. `published` reads a block's from a lab run's directory. `declared` gives the ones the Program says an honest run
publishes, by port and window: egress, the reference shaper on the Program's frames at the declared syncs; ingress, the
certifier's bucketer on the plan's requests, each at its arrival bucket, behind each window's salt frame when the salt
is on, and none for a window the verifier's plan has the block out for maintenance in. `cadence` is the certifiers'
parameters as a `datacenter.Cadence`. `datacenter.check_records(published, declared, cadence)` is the audit.
"""

from __future__ import annotations

import hashlib
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
from datacenter import Cadence, Datacenter, Interval, Plan, Port, Record  # noqa: E402
from verity.core.protocols.certifier import active  # noqa: E402
from verity.core.protocols.certifier import grid as cgrid  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync, run_shaper  # noqa: E402

DOMAIN = b"network_sim/certifier-record/v1\0"
TICK_MS = 1


def digest(r: cgrid.Record) -> bytes:
    """The record's status and its real frames in grid order. Frames are all B bytes, so the bytes hashed fix the
    frames; the counts fix where each falls."""
    h = hashlib.sha512(DOMAIN)
    h.update((r.status.reason.value if r.status.reason else "").encode() + b"\0")
    for row in r.grid:
        for slot in row[: cgrid.row_count(row)]:
            h.update(slot)
    return h.digest()


def to_record(r: cgrid.Record, port: Port) -> Record:
    return Record(port, r.window, cgrid.occupancy(r.grid), digest(r))


def cadence(setup: ar.Setup) -> Cadence:
    """The certifiers' parameters as the declaration's cadence, on a trusted clock of 1 ms ticks. Egress and ingress
    must agree on the frame size, the bucket and the window; their allocations may differ, and `Cadence` doesn't hold
    one."""
    e, i = setup.egress_params, setup.ingress_params
    if (e.frame_bytes, e.bucket_ms, e.buckets) != (i.frame_bytes, i.bucket_ms, i.buckets):
        raise ValueError(f"egress {e} and ingress {i} differ in frame size, bucket or window: one Cadence can't hold both")
    return Cadence(frame_bits=8 * e.frame_bytes, bucket_ticks=e.bucket_ms // TICK_MS, buckets=e.buckets,
                   jitter_classes=len(setup.sync_set.deltas) * len(setup.sync_set.rhos))


def uplink(dc: Datacenter, block: str) -> tuple[str, str]:
    """The block's one boundary link, as its end in the block and the certifier at its other end."""
    ours = [e for e in dc.boundary if any((b := dc.block_of(x)) is not None and b.name == block for x in (e.a, e.b))]
    if len(ours) != 1:
        raise ValueError(f"block {block} has {len(ours)} boundary links: the lab runs one certifier on one uplink")
    (e,) = ours
    inside, other = (e.a, e.b) if dc.block_of(e.a) is not None else (e.b, e.a)
    if dc.port(inside, other) is None:
        raise ValueError(f"block {block}'s uplink {e.a}-{e.b} doesn't end at a certifier")
    return inside, other


def ports(dc: Datacenter, block: str, setup: ar.Setup) -> dict[str, Port]:
    """The block's certifier ports, by the certifier's name for each link."""
    inside, cert = uplink(dc, block)
    return {setup.egress_link: dc.port(inside, cert), setup.ingress_link: dc.port(cert, inside)}


def published(run: Path, setup: ar.Setup, dc: Datacenter, block: str) -> list[Record]:
    """The records a block's certifier published in a lab run, from its directory."""
    by = ports(dc, block, setup)
    out = []
    for path, params in ((run / "responses.records", setup.egress_params), (run / "requests.records", setup.ingress_params)):
        for blob in ar.read_blobs(path):
            r = ar.unpack_record(blob, params.frame_bytes)
            out.append(to_record(r, by[r.link]))
    return out


def declared(plan: dict, setup: ar.Setup, dc: Datacenter, block: str,
             salts: dict[int, bytes] | None = None, verifier: Plan | None = None) -> dict[tuple[Port, int], Record]:
    """The records an honest run of `plan` publishes, from the plan alone, by port and window. With `salts`, the salts
    `Proxy.reveal` gave, by window: the ingress certifier enqueues its salt frame, which carries the placeholder and
    never the salt, at the first bucket of each window it has a salt for, ahead of that bucket's arrivals: every window
    it served, so not a maintenance outage, which has no salt. With `verifier`, the verifier's plan
    for the epoch, a window the block is out for maintenance in (`Plan.out`) is left out, as `datacenter.audit`
    leaves it: its ports owe no record then."""
    records = _declared(plan, setup, dc, block, salts)
    per = cadence(setup).window_ticks
    return {(port, w): r for (port, w), r in records.items()
            if verifier is None or not verifier.out(block, Interval(w * per, (w + 1) * per))}


def _declared(plan: dict, setup: ar.Setup, dc: Datacenter, block: str,
              salts: dict[int, bytes] | None) -> dict[tuple[Port, int], Record]:
    by = ports(dc, block, setup)
    e, i = setup.egress_params, setup.ingress_params
    release = {int(k): v for k, v in plan["model_release"].items()}
    frames, ready = ar.program(plan, release, e)
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    out = [to_record(r, by[r.link]) for r in run_shaper(frames, ready, syncs, cgrid.Certifier(setup.egress_link, e))]
    peer = ar.Peer(plan, ar.wire_params(setup))
    arrivals: dict[int, list[bytes]] = defaultdict(list)
    for q in sorted(plan["requests"], key=lambda q: (q["enqueue"], q["send_ms"], q["stream"])):
        data = ar.request_frames(peer.request(q), peer.B)
        arrivals[q["enqueue"]] += [data[k : k + peer.B] for k in range(0, len(data), peer.B)]
    bucketer = cgrid.Certifier(setup.ingress_link, i)
    for t in range(plan["windows"] * i.buckets):
        if salts is not None and t % i.buckets == 0 and (w := t // i.buckets) in salts:
            bucketer.enqueue(active.salt_frame(w, active.PLACEHOLDER, i.frame_bytes))
        for f in arrivals.get(t, []):
            bucketer.enqueue(f)
        bucketer.tick()
        if (t + 1) % i.buckets == 0:
            out.append(to_record(bucketer.close_window(), by[setup.ingress_link]))
    return {(r.port, r.window): r for r in out}


def to_json(r: Record) -> dict:
    return {"port": [r.port.certifier, r.port.direction], "window": r.window, "counts": list(r.counts),
            "digest": r.digest.hex()}
