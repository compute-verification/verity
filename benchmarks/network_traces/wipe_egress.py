"""wipe-egress: the network certifier's count of what can leave a node on the link its warden relays over a memory
wipe's session [t0, t1), read off the gate's warden run (`vy-challenger-gate warden-rehearse`, `wipe_gate.sh`), as the
``wipe-egress/v1`` record the wipe's reader takes: when it is ``covered``, its `wipe_period.certify(p, egress_mib)`
replaces the bound of the wipe signed `UNCOUNTED`; otherwise the wipe stays uncounted.

    python wipe_egress.py RUN --t0-ns T0 --t1-ns T1 [--pad-ms 1000] [--out FILE]

RUN is the gate's `fetch RUN` bundle (a gzipped tar) or the directory it unpacks to (``end.json`` and ``out/``). T0 and
T1 are the wipe's session on its challenger host's real-time clock, in Unix ns. It prints the record (and writes it to
FILE), and exits 0 when the record is covered, 1 when it isn't.

**The count.** The warden's schedule (`WardenParams`) fixes what each 3.2 s link-window can carry out of the node: its
egress record's ``lanes · slots`` cells a bucket and its fails record's ``lanes``, each of B bytes, for ``window``
buckets (`capacity_bits`, 2,621,440 bits at the scan's schedule). The record's windows are every link-window that meets
[t0 − pad, t1 + pad), counted from the deployment's ``origin_unix`` on the warden host's clock (`covering`); the pad
(`PAD_NS`, 1 s) is how far the two hosts' clocks may disagree. ``egress_bits`` is the windows' capacity, ``egress_mib``
it in the wipe's units (`streamed_wipe.EGRESS_UNIT_BITS`), rounded up. With no compute audit over the wipe nothing in
the cells is bound to a computation, so all of it is free: ``K`` is ``egress_bits``, and ``delta`` is 0, since nothing
was sampled.

**What it reads.** Only what the gate publishes: ``end.json``'s verdict and its SHA-512 of each ``out/`` file, the
deployment (``out/deployment.json``: the warden's key, session, schedule and origin) and the certificates alone
(``out/certificates.jsonl``, each signing a salted digest). The cells' counts are witness and never enter the record.
It is ``covered`` when the gate's check accepted the run, ``out/``'s two files are the ones ``end.json`` hashed, the
schedule is the one its digest names, the windows lie within the session, and each window has its egress and fails
certificates verifying under the deployment's key in its session; ``why`` names each failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from fractions import Fraction
from pathlib import Path

import certifier_serve as cs
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.pous import streamed_wipe as sw

FORMAT = "wipe-egress/v1"
#: how far the wipe challenger host's clock and the warden host's may disagree
PAD_NS = 1_000_000_000
#: the bundle's files the count reads, by their path under the run's directory
END, DEPLOYMENT, CERTIFICATES = "end.json", "out/deployment.json", "out/certificates.jsonl"
FILES = (END, DEPLOYMENT, CERTIFICATES)


def capacity_bits(params: WardenParams) -> int:
    """The most bits one link-window carries out: every cell of its egress and fails records, B bytes each."""
    return sum(g.buckets * g.allocation * g.frame_bytes * 8 for g in (params.grid, params.fails_grid))


def window_ns(params: WardenParams) -> int:
    return params.window * params.bucket_ms * 1_000_000


def origin_ns(origin_unix: float) -> int:
    return round(Fraction(origin_unix) * 1_000_000_000)


def covering(origin: int, width: int, t0: int, t1: int, pad: int = PAD_NS) -> range:
    """The link-windows w, each [origin + w·width, origin + (w+1)·width), that meet [t0 − pad, t1 + pad)."""
    return range((t0 - pad - origin) // width, -((origin - t1 - pad) // width))


def read_run(path: Path) -> dict[str, bytes]:
    """`FILES` of a gate run: from its `fetch` bundle (one top directory, the run's id) or the directory it unpacks to."""
    if path.is_dir():
        return {name: (path / name).read_bytes() for name in FILES if (path / name).is_file()}
    got = {}
    try:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                name = member.name.partition("/")[2]
                if name in FILES and member.isfile():
                    got[name] = tar.extractfile(member).read()
    except (OSError, tarfile.TarError) as e:
        raise ValueError(f"{path} is neither a gate run's `fetch` bundle nor the directory it unpacks to: {e}") from None
    return got


def record(files: dict[str, bytes], t0: int, t1: int, pad: int = PAD_NS) -> dict:
    """The ``wipe-egress/v1`` record of the gate run `files` (`read_run`) over [t0, t1) in Unix ns (module docstring)."""
    if not t0 < t1:
        raise ValueError(f"the wipe's window [{t0}, {t1}) is empty: t0 must be before t1")
    if pad < 0:
        raise ValueError(f"the pad is {pad} ns: it is at least 0")
    missing = [name for name in FILES if name not in files]
    if missing:
        raise ValueError(f"the gate run has no {', '.join(missing)}: give its `fetch` bundle or the directory it unpacks to")
    end, dep = json.loads(files[END]), json.loads(files[DEPLOYMENT])
    params = WardenParams(**dep["params"])
    key, session = cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    origin, width, n = origin_ns(dep["origin_unix"]), window_ns(params), dep["windows"]
    windows = covering(origin, width, t0, t1, pad)
    why = []
    if end.get("device") != "warden" or end.get("rc") != 0 or (end.get("verdict") or {}).get("accepted") is not True:
        why.append(f"the gate's check didn't accept the run (device {end.get('device')}, rc {end.get('rc')}, "
                   f"accepted {(end.get('verdict') or {}).get('accepted')})")
    for name in (DEPLOYMENT, CERTIFICATES):
        if hashlib.sha512(files[name]).hexdigest() != (end.get("files") or {}).get(Path(name).name):
            why.append(f"{name} isn't the file the gate's end.json hashed")
    if params.digest.hex() != dep["digest"]:
        why.append("the deployment's schedule isn't the one its digest names")
    if windows.start < 0:
        why.append(f"the window starts at link-window {windows.start}, before the session's origin")
    if windows.stop > n:
        why.append(f"the window ends at link-window {windows.stop}, past the session's {n}")
    certificates = {}
    for line in files[CERTIFICATES].decode().splitlines():
        doc = json.loads(line)
        if "certificate" in doc:
            c = cs.certificate_from_json(doc["certificate"])
            certificates[(c.link, c.direction, c.window)] = c
    links, verified = (dep["link"], f"{dep['link']}-fails"), 0
    for w in range(max(windows.start, 0), min(windows.stop, n)):
        for link in links:
            c = certificates.get((link, Direction.EGRESS, w))
            if c is None:
                why.append(f"{link} egress window {w} has no certificate")
            elif not c.verify(key, session):
                why.append(f"{link} egress window {w}'s certificate doesn't verify under the deployment's key")
            else:
                verified += 1
    covered = not why
    bits = len(windows) * capacity_bits(params)
    return {"format": FORMAT, "gate_run": end.get("run"), "pinned": end.get("pinned"), "link": dep["link"],
            "key": dep["key"], "session": dep["session"], "parameters": dep["digest"],
            "origin_unix_ns": origin, "window_ns": width, "session_windows": n,
            "t0_unix_ns": t0, "t1_unix_ns": t1, "pad_ns": pad, "windows": [windows.start, windows.stop],
            "certificates": verified, "covered": covered, "why": why, "capacity_bits": capacity_bits(params),
            "egress_bits": bits if covered else None,
            "egress_mib": -(-bits // sw.EGRESS_UNIT_BITS) if covered else None,
            "K": bits if covered else None, "delta": "0"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wipe_egress.py", description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, help="the gate's `fetch RUN` bundle, or the directory it unpacks to")
    p.add_argument("--t0-ns", type=int, required=True, help="the wipe's session start, Unix ns")
    p.add_argument("--t1-ns", type=int, required=True, help="the wipe's session end (its signature), Unix ns")
    p.add_argument("--pad-ms", type=int, default=PAD_NS // 1_000_000, help="the two hosts' clocks' disagreement")
    p.add_argument("--out", type=Path, help="also write the record here")
    a = p.parse_args(argv)
    try:
        doc = record(read_run(a.run), a.t0_ns, a.t1_ns, a.pad_ms * 1_000_000)
    except ValueError as e:
        p.error(str(e))
    text = json.dumps(doc, indent=1)
    if a.out is not None:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if doc["covered"] else 1


if __name__ == "__main__":
    sys.exit(main())
