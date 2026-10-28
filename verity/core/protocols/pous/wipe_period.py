"""One wipe of one node at a small fill, the wipe's leg of a period: the memory challenger fills the node's memory with
its streamed labels and times its spot reads (`streamed_wipe`), signs the session, and the verifier answers as the
wipe's component of onsite Verity (`verity/wipe/v0`, `MemoryChallenger.WipeProtocol` in
`verity/core/devices/memory_challenger/lean/`).

`run(node, params, ...)` runs one wipe: in one process (the challenger under `signer` and `log`, on `clock`), or with
the memory challenger a party of its own (`party`, `wipe_party`), which draws its secret, session id and coins on its
own host and signs under its own key while the prover answers over a socket. The prover is `streamed_wipe.Prover` (the
honest server) unless a test passes another, such as `keeping`. Then the verifier answers (`answer`), as the Lean does:

* a commitment of another server or session rejects;
* a session the memory challenger hasn't signed (the prover left, or its reads aren't verified yet) is undecided;
* else `streamed_wipe.verdict`, whose reasons, joined by `"; "`, are a rejection's: a late or wrong read, an untrusted
  key, a bad signature, an uncovered batch size.

What an accepted wipe certifies is `certify`'s: `{"certified", "egress_mib", "eps", "foreign_bits", "k", "memory_bits",
"r"}`, `memory_bits = M` the declared tiers' bits (the fill's `n · 512` and `R` past it, the memory it doesn't reach),
`k` the session's timed reads, which the memory challenger chose (`wipe_party --reads`), and `r = ratio(k)`, the
largest `r` with `(r − 1)^k · 2^129 ≤ r^k`. When the bound covers the session (`k` at most `MAX_READS`, an egress count
that isn't saturated) `foreign_bits` is `Pous.Guarantees.StreamedWipeMeetsReads`' threshold
`512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` at `eps = 2^-128` (`foreign_bits`), with `E` the egress the network
certifier counted over the session. Off it, `foreign_bits` is `M` and `eps` 0, the vacuous statement, and the profile
names each gap. `EXPERIMENTAL` says what the bound rests on.

`leg(period_dir, node, out)` is the wipe's leg of `benchmarks/all_protocols`: one wipe as `period.json`'s `wipe` says
(`CONFIG`: `SEGMENTS` 32 MiB segments in host memory by default, or a GPU's memory through `wipe_store`, the whole
device declared, the driver's reserve included (`wipe_store.Cuda.info`'s `total_bytes`), so that its memory outside the
fill counts in `R`, the server's disks ahead of it in the same session, `wipe.disk`, and the host's DRAM after it,
`wipe.dram`), the memory challenger on `period.json`'s `challenger_host` (`wipe_party.side_of`; this node by default, a
process whose key and session log are beside the periods). No network certifier counts the session's egress yet, so the
challenger signs `UNCOUNTED` and an accepted leg's bound is vacuous: its verdict is real and its quantity isn't. The
leg's record names what the bound doesn't cover (`uncertified`): the key off its own host, an uncounted egress, the
node's memory outside the declared tiers (`undeclared`), a GPU's driver reserve when NVML gave no total, what its disks
leave outside the boundary (`disk_gaps`) and each link leaving the node that no network certifier counts
(`wipe_party.link`).
`main` runs the leg on this node and prints its `summary` (`benchmarks/pous/streamed_wipe/demo.sh` runs it on one GPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import socket
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.core.deploy.stream import Side
from verity.core.protocols import answer as A
from verity.ml.commitments.identity import canonical_json_bytes

from . import memory_challenger as mc
from . import streamed_wipe as sw
from . import wipe_party, wipe_store
from .period import serve
from .wipe_party import READS, UNCOUNTED

PROTOCOL = "verity/wipe/v0"
"""`MemoryChallenger.WipeProtocol.ID`; a test signer's wipes answer under `memory_challenger.TEST_PREFIX` + it."""
EPS = Fraction(1, 2 ** 128)
MAX_READS = 1 << 17
"""The most reads the certificate is computed at: `ratio` compares numbers of about `k · log₂ r` bits."""
SAMPLING_BITS = 129
"""`StreamedWipeMeetsReads`' sampling share of the error, `(1 − 1/r)^k ≤ 2^-129`."""
COUNT_BITS = 130
"""The threshold's constant: the compression share's `log₂(1/δ)`, 129, and one more for `k < 2^(⌊log₂ k⌋ + 1)`."""
SEGMENT_BYTES = 1 << 25
SEGMENT_LABELS = SEGMENT_BYTES // sw.LABEL_BYTES
SEGMENTS = 1
"""The leg's fill, in 32 MiB segments."""
LIMIT_NS = 20_000_000
"""The leg's deadline a read: 20 ms, a socket round trip on one machine or a nearby one with room to spare."""
TIER = "host-dram"
EXPERIMENTAL = (
    "foreign_bits at eps = 2^-128 is Pous.Guarantees.StreamedWipeMeetsReads' threshold at the session's k and r = "
    "ratio(k), proved for the streamed-label fill in the wipe game; that game's acceptance is not this verdict (the "
    "guarantee is frozen, no-function), and eps leaves out its PRF advantage and its premises: every tier declared, the "
    "egress certified by the network certifier, no side channel")
OTHER = "the wipe's commitment is not of the period's server and session"
UNSIGNED = "the memory challenger has not signed the wipe: its reads are incomplete or not yet verified"


def params_for(segments: int = SEGMENTS, *, k: int = READS, limit_ns: int = LIMIT_NS, tier: str = TIER,
               tier_bytes: int | None = None, tiers: Sequence[tuple[str, int]] | None = None) -> sw.WipeParams:
    """A wipe of `segments` 32 MiB segments in one tier of `tier_bytes` (by default the fill's), or over `tiers` when
    given (a joint store's, `wipe_store.Joint.declared`), `k` reads one at a time under `limit_ns` each."""
    fill = segments * SEGMENT_BYTES
    if tiers is None:
        tiers = ((tier, fill if tier_bytes is None else tier_bytes),)
    return sw.WipeParams(segments * SEGMENT_LABELS, k, 1, limit_ns, tuple((t, b) for t, b in tiers))


def memory_bits(p: sw.WipeParams) -> int:
    """`M`: every declared tier in bits, the fill and `R` past it."""
    return p.labels * sw.LABEL_BITS + p.outside_bits


def ratio(k: int) -> int:
    """`MemoryChallenger.WipeProtocol.ratio`: the largest `r` with `(r − 1)^k · 2^129 ≤ r^k`, `(1 − 1/r)^k ≤ 2^-129`, by
    bisection between `1`, which meets it for every `k ≥ 1`, and `k // 89 + 2`, which doesn't (`(1 − 1/r)^k ≥
    e^(−k/(r − 1))` and `129 · ln 2 > 89`)."""
    lo, hi = 1, k // 89 + 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (mid - 1) ** k << SAMPLING_BITS <= mid ** k:
            lo = mid
        else:
            hi = mid
    return lo


def foreign_bits(p: sw.WipeParams, r: int) -> int:
    """`StreamedWipeMeetsReads`' threshold before the egress: `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R`."""
    n = p.labels
    return sw.LABEL_BITS * ((n + r - 1) // r - 1) + n + p.k.bit_length() - 1 + COUNT_BITS + p.outside_bits


def gaps(p: sw.WipeParams, egress_mib: int) -> list[str]:
    """`MemoryChallenger.WipeProtocol.gaps`: why the bound doesn't cover an accepted wipe (none when it does)."""
    out = []
    if p.k > MAX_READS:
        out.append(f"k = {p.k}: the certificate is computed for at most {MAX_READS} reads")
    if not 0 <= egress_mib < sw.EGRESS_MAX_UNITS:
        out.append(f"the egress count is {egress_mib} MiB, outside 0 to {sw.EGRESS_MAX_UNITS - 1}: it bounds nothing")
    return out


def ratio_of(p: sw.WipeParams) -> int:
    """`ratio(k)`, or 0 past `MAX_READS`, where it isn't computed."""
    return ratio(p.k) if p.k <= MAX_READS else 0


def certify(p: sw.WipeParams, egress_mib: int) -> dict[str, Any]:
    """`MemoryChallenger.WipeProtocol.certify`'s public outputs, eps an exact ratio."""
    m, r = memory_bits(p), ratio_of(p)
    if gaps(p, egress_mib):
        return {"certified": False, "egress_mib": egress_mib, "eps": "0", "foreign_bits": m, "k": p.k, "memory_bits": m,
                "r": r}
    return {"certified": True, "egress_mib": egress_mib, "eps": str(EPS),
            "foreign_bits": foreign_bits(p, r) + egress_mib * sw.EGRESS_UNIT_BITS, "k": p.k, "memory_bits": m, "r": r}


def profile(p: sw.WipeParams, egress_mib: int) -> dict[str, Any]:
    """`MemoryChallenger.WipeProtocol.profileJson`: the wipe the session ran, the bound and every gap."""
    return {"batch": p.batch,
            "certificate": {"eps": str(EPS), "max_reads": MAX_READS, "outside_bits": p.outside_bits, "r": ratio_of(p),
                            "reads": p.k},
            "experimental": EXPERIMENTAL, "gaps": gaps(p, egress_mib), "k": p.k, "labels": p.labels,
            "limit_ns": p.limit_ns, "tiers": [[t, b] for t, b in p.tiers]}


def outcome(server: bytes, session: bytes, trusted: Collection[bytes], commitment: sw.WipeCommitment,
            signed: sw.SignedWipe | None, schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> tuple:
    """`MemoryChallenger.WipeProtocol.check` on decoded inputs: `("accept", public, profile)`, `("reject", reason)` or
    `("undecided", reason)`."""
    if commitment.server != server or commitment.session != session:
        return ("reject", OTHER)
    if signed is None:
        return ("undecided", UNSIGNED)
    v = sw.verdict(signed, commitment, trusted=set(trusted), schemes=schemes)
    if not v.accepted:
        return ("reject", "; ".join(v.reasons))
    return ("accept", certify(commitment.params, v.egress_mib), profile(commitment.params, v.egress_mib))


def digest(commitment: sw.WipeCommitment) -> bytes:
    """SHA-512 of the commitment's canonical JSON."""
    return hashlib.sha512(canonical_json_bytes(commitment.to_json())).digest()


def params_json(server: bytes, session: bytes, trusted: Sequence[bytes]) -> dict[str, Any]:
    return {"server": server.hex(), "session": session.hex(), "trusted": [t.hex() for t in trusted]}


def case(server: bytes, session: bytes, trusted: Sequence[bytes], commitment: sw.WipeCommitment,
         signed: sw.SignedWipe | None) -> dict[str, Any]:
    """The wipe protocol's case, `{"commitment": hex, "evidence": hex, "params": JSON}`."""
    evidence = {"signed": None if signed is None else signed.to_json()}
    return {"commitment": canonical_json_bytes(commitment.to_json()).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": params_json(server, session, trusted)}


def answer(server: bytes, session: bytes, trusted: Collection[bytes], commitment: sw.WipeCommitment,
           signed: sw.SignedWipe | None, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
           protocol: str = PROTOCOL) -> dict[str, Any]:
    """The wipe protocol's answer (`Core.Answer.canonical`) on decoded inputs."""
    verdict, *rest = outcome(server, session, trusted, commitment, signed, schemes)
    if verdict == "accept":
        return A.Answer.accept(protocol, digest(commitment), *rest).to_json()
    return A.Answer(protocol, A.Verdict(verdict), rest[0], digest(commitment)).to_json()


class Keeping(sw.Prover):
    """A prover that keeps `foreign` labels' worth of its own prior content in place of the first `foreign` labels it
    is sent, and answers from what it holds."""

    def __init__(self, params: sw.WipeParams, foreign: int) -> None:
        super().__init__(params)
        self.foreign = foreign

    def receive(self, start: int, data: bytes) -> None:
        super().receive(start, data)
        end = min(self.foreign, start + len(data) // sw.LABEL_BYTES)
        for i in range(start, end):
            self.store[sw.LABEL_BYTES * i:sw.LABEL_BYTES * (i + 1)] = hashlib.sha512(b"prior content %d" % i).digest()


def keeping(foreign: int) -> Callable[[sw.WipeParams], sw.Prover]:
    """`Keeping`'s factory: a prover that keeps `foreign` labels' worth of prior content."""
    return lambda params: Keeping(params, foreign)


@dataclass(frozen=True)
class Wipe:
    """A period's wipe: the protocol's answer and case, how the session ran (`ended`, `reads`, the challenger's `host`;
    with the challenger a party of its own the session's `window` on its clock and the prover's `connection`; with the
    native streamer its gate and figures, `streamer` and `fill`) and the signed session, if signed."""

    answer: dict[str, Any]
    case: dict[str, Any]
    report: dict[str, Any]
    signed: sw.SignedWipe | None = None

    @property
    def verdict(self) -> str:
        return self.answer["verdict"]


def timing(signed: sw.SignedWipe | None) -> dict[str, Any] | None:
    """The signed session's timings on the memory challenger's clock: the fill, its first label sent to its last, and
    each read's round trip, a batch's send to its answer's receipt."""
    if signed is None:
        return None
    size, ns = signed.commitment.params.labels * sw.LABEL_BYTES, signed.fill_end_ns - signed.fill_start_ns
    return {"fill": {"bytes": size, "ns": ns, "GBps": wipe_store.gbps(size, ns)},
            "reads": wipe_store.percentiles([r.elapsed_ns for r in signed.batches])}


def run(node: str, params: sw.WipeParams | None = None, *, signer: mc.Signer | None = None,
        log: mc.SessionLog | None = None, party: Side | None = None,
        prover: Callable[[sw.WipeParams], sw.Prover] = sw.Prover, egress_bits: int = UNCOUNTED,
        clock: Callable[[], int] = time.perf_counter_ns, secret: bytes | None = None, session_id: bytes | None = None,
        coin_secret: bytes | None = None, stop_after: int | None = None, trusted: Collection[bytes] | None = None,
        schemes: Mapping[str, mc.Verify] = mc.SCHEMES, protocol: str = PROTOCOL,
        chunk: int = wipe_party.CHUNK) -> Wipe:
    """One wipe of `node` (module docstring), the server id its name's UTF-8. In one process the challenger signs under
    `signer` with `egress_bits`, claims its session in `log` and reads `clock`; `secret`, `session_id` and `coin_secret`
    are fresh from the OS unless a test gives them. With `party` its side runs where `party` starts it, with its own
    secret, coins, key, log, clock and egress count, and those arguments are unused. `prover` makes the prover's side,
    which with `stop_after` leaves after that many reads. `trusted` is the verifier's list of memory-challenger keys (by
    default the challenger's), `schemes` the signature schemes it accepts and `protocol` the id it answers under."""
    params = params_for() if params is None else params
    server = node.encode()
    if party is None:
        assert signer is not None and log is not None, "an in-process wipe needs the challenger's signer and log"
        c = sw.Challenger(secrets.token_bytes(32) if secret is None else secret, server,
                          secrets.token_bytes(32) if session_id is None else session_id, params, signer, log=log,
                          clock=clock, coin_secret=coin_secret)
        p = prover(params)
        for first, data in c.fill(chunk):
            p.receive(first, data)
        if (sync := getattr(p, "sync", None)) is not None:
            sync()
        while c.ended is None and (stop_after is None or len(c.batches) < stop_after):
            c.receive(p.answer(c.challenge()))
        signed = None if c.ended is None else c.close(egress_bits)
        commitment, key = c.commitment, signer.public_key
        report = {"ended": c.ended, "host": node, "reads": len(c.batches)}
    else:
        got = wipe_party.session(party, wipe_party.request_of(node, params), prover(params), stop_after=stop_after)
        commitment, key = wipe_party.commitment_of(got["commitment"]), bytes.fromhex(got["public_key"])
        signed = None if got["signed"] is None else wipe_party.signed_of(got["signed"])
        report = {"ended": got["ended"], "host": got["host"], "reads": got["reads"]}
        report |= {k: got[k] for k in ("connection", "fill", "streamer", "window") if k in got}
    trusted = (key,) if trusted is None else tuple(trusted)
    return Wipe(answer(server, commitment.session, trusted, commitment, signed, schemes, protocol),
                case(server, commitment.session, trusted, commitment, signed), report, signed)


CONFIG = {"store": "python", "device": 0, "segments": None, "reserve_bytes": 1 << 30, "tier": None,
          "limit_ns": LIMIT_NS, "streamer": "python", "threads": None, "foreign_bytes": 0, "placement": None,
          "egress_bits": None, "reads": None, "links": [], "disk": None, "dram": None}
"""`period.json`'s `wipe` and its defaults: the prover's `store` (`python`, `streamed_wipe.Prover`; `host`, host memory
through `wipe_store`; `cuda`, GPU `device`'s memory, declared whole, the driver's reserve included
(`wipe_store.Cuda.info`'s `total_bytes`), so that what the fill doesn't reach counts in `R`; `disk`, the server's disks
alone), the fill in 32 MiB `segments` (`SEGMENTS`; for `cuda`, `None` fills the device's free memory less
`reserve_bytes`), the disks (`disk`, `DISK`: in the same session as a `host` or `cuda` store, ahead of it,
`wipe_store.Joint`), the host's DRAM (`dram`, `DRAM`: in the same session as a `cuda` or `disk` store, after it), the
declared `tier` (the store's own by default; a joint store's are its parts'), the deadline a read, and the fill's
streamer (`native`, `wipe_stream` on
`threads` CPUs, or `python`) when the challenger runs on this node. A challenger on its own host streams as its
`challenger.wipe.command` says. `foreign_bytes` makes the prover the cheat that keeps that much prior content in place
of the fill's first labels (`foreign_labels`; `"fM"`, one label past the bound at the expected reads), and `placement`
`"simulated"` names a challenger on this node the stand-in for one on its own host (`wipe_party.placement`).
`egress_bits` is the network certifier's count over the session, which a challenger on this node signs (`wipe_party
--egress-bits`); `None` signs `UNCOUNTED`. `reads` is the session's timed reads: a challenger on this node is started
with it (`wipe_party --reads`); one on its own host chooses its own (its `challenger.wipe.command`), and the leg refuses
a session at another count. `None`: whatever the challenger chooses (`READS` here). `links` are the links leaving the
node besides the one to a challenger on its own host, which the leg adds from the connection it made: each `{"name",
"to", "via"}`, with `mbps` and `counted_by` (the network certifier that counts it) when known (`wipe_party.link`)."""
LINK_KEYS = {"name", "to", "via", "mbps", "counted_by"}
DISK = {"paths": None, "bytes": None, "reserve_bytes": wipe_store.DISK_RESERVE, "spare_bytes": 0, "direct": True}
"""`wipe.disk` and its defaults: the targets (`paths`, each a directory the fill takes a file in or a block device
written raw, `wipe_store.Disk`), the fill's bytes on them (`None`: all their room, each directory's free space less
`reserve_bytes`; whole segments either way), the drives' `spare_bytes` (spare area and cache, from their datasheets,
declared so they count in `R`) and `direct` (`O_DIRECT`; `None` falls back to buffered I/O where it is refused)."""
DRAM = {"bytes": None, "reserve_bytes": 16 << 30}
"""`wipe.dram` and its defaults: the fill's `bytes` on the host's DRAM (`None`: what `dram_available` says less
`reserve_bytes`; whole segments either way), the DRAM declared whole (`wipe_store.HostMemory(whole=True)`), so that
what the fill doesn't reach counts in `R`."""


def config(doc: Mapping[str, Any]) -> dict[str, Any]:
    """`period.json`'s `wipe` over `CONFIG`'s defaults; an unknown key or value raises."""
    got = doc.get("wipe") or {}
    if unknown := set(got) - set(CONFIG):
        raise ValueError(f"period.json's wipe has unknown keys {sorted(unknown)}: it takes {sorted(CONFIG)}")
    c = CONFIG | got
    if c["store"] not in ("python", "host", "cuda", "disk") or c["streamer"] not in ("python", "native"):
        raise ValueError(f"period.json's wipe: store is python, host, cuda or disk and streamer python or native, not {got}")
    if (d := c["disk"]) is not None:
        if not isinstance(d, dict) or (unknown := set(d) - set(DISK)):
            raise ValueError(f"period.json's wipe.disk takes {sorted(DISK)}, not {d!r}")
        c["disk"] = d = DISK | d
        if not (isinstance(d["paths"], list) and d["paths"] and all(type(p) is str for p in d["paths"])
                and (d["bytes"] is None or type(d["bytes"]) is int and d["bytes"] > 0)
                and all(type(d[k]) is int and d[k] >= 0 for k in ("reserve_bytes", "spare_bytes"))
                and any(d["direct"] is v for v in (True, False, None))):
            raise ValueError(f"period.json's wipe.disk: paths a list of directories or block devices, bytes null or a "
                             f"positive byte count, reserve_bytes and spare_bytes byte counts, direct true, false or "
                             f"null, not {d!r}")
    if (c["store"] == "disk" and c["disk"] is None or c["store"] == "python" and c["disk"] is not None
            or c["disk"] is not None and c["tier"] is not None):
        raise ValueError(f"period.json's wipe: store disk takes wipe.disk, which joins a host or cuda store too (never "
                         f"python), and a store with disks declares its parts' tiers (no wipe.tier), not {got}")
    if (h := c["dram"]) is not None:
        if not isinstance(h, dict) or (unknown := set(h) - set(DRAM)):
            raise ValueError(f"period.json's wipe.dram takes {sorted(DRAM)}, not {h!r}")
        c["dram"] = h = DRAM | h
        if not ((h["bytes"] is None or type(h["bytes"]) is int and h["bytes"] > 0)
                and type(h["reserve_bytes"]) is int and h["reserve_bytes"] >= 0):
            raise ValueError(f"period.json's wipe.dram: bytes null or a positive byte count and reserve_bytes a byte "
                             f"count, not {h!r}")
        if c["store"] not in ("cuda", "disk") or c["tier"] is not None:
            raise ValueError(f"period.json's wipe: wipe.dram joins a cuda or disk store (a host store is the DRAM "
                             f"already), which declares its parts' tiers (no wipe.tier), not {got}")
    f, e, k = c["foreign_bytes"], c["egress_bits"], c["reads"]
    if not (f == "fM" or type(f) is int and f >= 0) or c["placement"] not in (None, "simulated") or not (
            e is None or type(e) is int and e >= 0) or not (k is None or type(k) is int and 1 <= k < 1 << 32):
        raise ValueError(f"period.json's wipe: foreign_bytes is a byte count or \"fM\", placement null or "
                         f"\"simulated\", egress_bits null or a bit count and reads null or 1 to 2^32 - 1, not {got}")
    for e in c["links"]:
        if not (isinstance(e, dict) and set(e) <= LINK_KEYS and all(type(e.get(k)) is str for k in ("name", "to", "via"))
                and (e.get("mbps") is None or type(e["mbps"]) is int)
                and (e.get("counted_by") is None or type(e["counted_by"]) is str)):
            raise ValueError(f"period.json's wipe.links: each is {{\"name\", \"to\", \"via\"}} (strings) with an optional "
                             f"int \"mbps\" and string \"counted_by\", not {e!r}")
    return c


def foreign_labels(foreign_bytes: int | str, p: sw.WipeParams) -> int:
    """The labels a cheat of `foreign_bytes` keeps prior content in place of: whole labels covering the bytes, or for
    `"fM"` the fewest labels past `p`'s published `foreign_bits` with no egress, the least a wipe must reject."""
    if foreign_bytes == "fM":
        return min(p.labels, certify(p, 0)["foreign_bits"] // sw.LABEL_BITS + 1)
    return min(p.labels, -(-int(foreign_bytes) // sw.LABEL_BYTES))


def dram_available(meminfo: Path = Path("/proc/meminfo")) -> int:
    """The host's DRAM a fill can take without swapping or reclaiming anyone's working set: `MemAvailable` (free
    memory and the page cache the kernel would drop), else the free pages."""
    for line in meminfo.read_text().splitlines() if meminfo.is_file() else ():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def store_of(c: Mapping[str, Any]) -> wipe_store.Memory | None:
    """The prover's store as `c` says (`CONFIG`): none for `python`; with `disk` or `dram`, a `wipe_store.Joint` whose
    parts are in order the disks (whole segments of their room, or `disk.bytes`), so that the fill's first labels, the
    ones a cheat keeps its own content in place of, are on the disks, the `host` or `cuda` store at `segments_of`, and
    the host's DRAM declared whole (whole segments of `dram_available` less `dram.reserve_bytes`, or `dram.bytes`)."""
    memory = {"python": lambda: None, "disk": lambda: None, "host": wipe_store.HostMemory,
              "cuda": lambda: wipe_store.Cuda(c["device"])}[c["store"]]()
    if c["disk"] is None and c["dram"] is None:
        return memory
    disk = dram = None
    try:
        parts: list[tuple[wipe_store.Memory, int]] = []
        if (d := c["disk"]) is not None:
            disk = wipe_store.Disk(d["paths"], reserve_bytes=d["reserve_bytes"], spare_bytes=d["spare_bytes"],
                                   direct=d["direct"])
            room = disk.mem_info()[0]
            if d["bytes"] is not None and d["bytes"] > room:
                raise RuntimeError(f"wipe.disk.bytes is {d['bytes']}, past the {room} bytes the disk targets have room "
                                   "for")
            n = (room if d["bytes"] is None else d["bytes"]) // SEGMENT_BYTES * SEGMENT_BYTES
            if n < SEGMENT_BYTES:
                raise RuntimeError(f"the disk targets have room for {room} bytes, less than one segment")
            parts.append((disk, n))
        if memory is not None:
            parts.append((memory, segments_of(c, memory) * SEGMENT_BYTES))
        if (h := c["dram"]) is not None:
            dram = wipe_store.HostMemory(whole=True)
            room = dram_available() - h["reserve_bytes"]
            if h["bytes"] is not None and h["bytes"] > room:
                raise RuntimeError(f"wipe.dram.bytes is {h['bytes']}, past the {room} bytes of DRAM available less "
                                   "the reserve")
            n = (room if h["bytes"] is None else h["bytes"]) // SEGMENT_BYTES * SEGMENT_BYTES
            if n < SEGMENT_BYTES:
                raise RuntimeError(f"the host has {room} bytes of DRAM available past the reserve, less than one "
                                   "segment")
            parts.append((dram, n))
        return wipe_store.Joint(parts)
    except BaseException:
        for m in (disk, memory, dram):
            if m is not None:
                m.close()
        raise


def segments_of(c: Mapping[str, Any], memory: wipe_store.Memory | None) -> int:
    """The fill's 32 MiB segments: `c`'s, else the device's free memory less the reserve, else `SEGMENTS`."""
    if c["segments"] is not None:
        return c["segments"]
    if not isinstance(memory, wipe_store.Cuda):
        return SEGMENTS
    free, _ = memory.mem_info()
    if (n := (free - c["reserve_bytes"]) // SEGMENT_BYTES) < 1:
        raise RuntimeError(f"the device has {free} bytes free, less than one segment past the reserve")
    return n


NOT_DISKS = re.compile(r"(loop|z?ram)\d+")
"""`/sys/block` names that are no disk of their own: a loop device (a file on another disk) and a RAM disk (DRAM)."""


def disks(sys_block: Path = Path("/sys/block"), skip: Collection[str] = ()) -> dict[str, int]:
    """Every whole block device of this host, mounted or not (`/sys/block`), by name with its size in bytes: all but
    loop and RAM devices (`NOT_DISKS`), a device-mapper or md device over others (its `slaves`, each counted itself),
    an empty one, and those in `skip` (`wipe_store.whole_disks`: the disks a store declares)."""
    out = {}
    for d in sorted(sys_block.iterdir()):
        if d.name in skip or NOT_DISKS.fullmatch(d.name) or (d / "slaves").is_dir() and any((d / "slaves").iterdir()):
            continue
        if size := int((d / "size").read_text()) * 512:
            out[d.name] = size
    return out


def _parts(store: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    return [] if store is None else list(store.get("parts") or [store])


def undeclared(tier: str, declared_bytes: int, store: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The memory this host holds outside the declared tier `tier` (`declared_bytes` of it), which `M` doesn't count:
    the device's memory outside the tier (from `store`, `wipe_store`'s `info`, when it gives `total_bytes` past the
    tier), the host's DRAM (outside the     tier when it is the tier; none when it is declared whole) and each of its block devices, mounted or not, whole
    (`disks`). A joint store's (`wipe_store.Joint`) tiers are its parts', each at its declared bytes, and the disks a
    `disk` part declares are no one's undeclared memory."""
    out, parts = [], _parts(store)
    if store is not None and "parts" in store:
        declared = {p["tier"]: p.get("total_bytes") or p["bytes"] for p in parts}
    else:
        declared = {tier: declared_bytes}
        if store is not None and store.get("total_bytes", 0) > declared_bytes:
            out.append({"tier": store["tier"], "bytes": store["total_bytes"] - declared_bytes,
                        "what": f"{store.get('device', 'the device')}'s memory outside the declared tier"})
    dram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if (rest := dram - declared.get("host-dram", 0)) > 0:
        out.append({"tier": "host-dram", "bytes": rest, "what": "the host's DRAM outside the declared tier"
                    if "host-dram" in declared else "the host's DRAM"})
    skip = {d for p in parts for d in p.get("disks") or {}}
    out += [{"tier": "disk", "bytes": n, "what": f"the host's block device {name}" + (
        ", outside the declared disks" if skip else "")} for name, n in disks(skip=skip).items()]
    return out


def disk_gaps(store: Mapping[str, Any] | None) -> list[str]:
    """What a `disk` part of the prover's store (`wipe_store.Disk.info`) leaves outside the boundary: a virtual disk's
    backing store, the hypervisor's (`wipe_store.VIRTUAL_DRIVERS`), whose snapshots and replicas the fill doesn't
    reach; a drive's spare area and cache when `spare_bytes` doesn't declare them; and buffered I/O, which the page
    cache can answer a read from."""
    out = []
    for p in _parts(store):
        if p.get("tier") != "disk":
            continue
        for name, d in sorted(p["disks"].items()):
            if d["driver"] in wipe_store.VIRTUAL_DRIVERS:
                out.append(f"disk {name}: a virtual disk ({d['driver']}), its backing store the hypervisor's: a "
                           "snapshot or replica the cloud keeps of it is outside the boundary, which Covered assumes away")
            elif not p["spare_bytes"]:
                out.append(f"disk {name}: its spare area and cache are undeclared (spare_bytes 0), so M doesn't count "
                           "what they keep")
        if not p["direct"]:
            out.append("disk: buffered I/O, not O_DIRECT: the page cache can answer a read the disk doesn't keep")
    return out


def uncertified(where: Mapping[str, Any], signed: sw.SignedWipe | None) -> list[str]:
    """What an accepted wipe's bound doesn't cover, from its placement (`wipe_party.placement`) and signed session: the
    memory challenger off its own host (its key on the prover's node), an egress signed `UNCOUNTED`, each undeclared
    tier, a GPU's driver reserve when NVML gave no total (`wipe_store.Cuda.info`; a joint store's GPU part too), what
    the prover's disks leave outside the boundary (`disk_gaps`) and each link leaving the node that no network
    certifier counts."""
    out = [] if where["kind"] == "own-host" else [f"key custody: {where[wipe_party.PREMISE]}"]
    if signed is not None and signed.egress_bits >= UNCOUNTED:
        out.append("egress: signed UNCOUNTED, no network certifier's count over the session, so the bound is vacuous")
    out += [f"tier {u['tier']}: {u['what']} ({u['bytes']} bytes) is undeclared: the fill doesn't cover it and M doesn't "
            "count it" for u in where["undeclared_tiers"]]
    out += [f"tier {s['tier']}: the memory {s['device']}'s driver reserves is undeclared: the tier is CUDA's total "
            f"({s['cuda_total_bytes']} bytes), which leaves it out, since NVML gave none ({s.get('nvml_error')}); the "
            "fill doesn't cover it and M doesn't count it" for s in _parts(where.get("prover_store"))
            if s.get("backend") == "cuda" and s.get("nvml_total_bytes") is None]
    out += disk_gaps(where.get("prover_store"))
    out += [f"link {e['name']}: {e['ends'][0]} to {e['ends'][1]} ({e['via']}) is counted by no network certifier, so what "
            "leaves on it is outside E" for e in where["links"] if e["counted_by"] is None]
    return out


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The wipe's leg of `benchmarks/all_protocols` (`leg(period_dir, node, out) -> answer`): one wipe as `period.json`'s
    `wipe` says (`CONFIG`; by default `SEGMENTS` 32 MiB segments kept by `streamed_wipe.Prover`), the memory challenger a
    party of its own on `period.json`'s `challenger_host` (`wipe_party.side_of`; on this node by default, a process
    whose key and session log are in `wipe-challenger` beside the periods) and the prover here. A `wipe_store` prover
    gates its store before the session. With `wipe-protocol` among the period's tools, the Lean answers (the verifier of
    record) and must agree with the Python; it writes `out/wipe.case.json` (onsite's wipe case) and `out/wipe.json`.
    Which host ran each party, whether `trusted-host`'s premise holds, the memory the prover kept the fill in,
    the links leaving the node (`wipe.links`, and the challenger's on its own host as the prover connected to it) and
    the node's memory outside the declared tier (`undeclared`) go in `wipe.json` and in an accepted answer's profile
    (`placement`, `wipe_party.placement`); `wipe.json` adds the prover's own timings (`prover`: its store, gate, fill
    and device read latency), the signed session's on the challenger's clock (`session`: the fill's GB/s, each read's
    round trip), the streamer's (`report.fill`), the session's window on each side's real-time clock (`window`) and
    what the bound doesn't cover (`uncertified`)."""
    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools, c = doc.get("tools") or {}, config(doc)
    extra = (("--streamer", "build", *(("--threads", str(c["threads"])) if c["threads"] else ()))
             if c["streamer"] == "native" else ()) + (() if c["egress_bits"] is None else
                                                       ("--egress-bits", str(c["egress_bits"]))) + (
        () if c["reads"] is None else ("--reads", str(c["reads"])))
    party = wipe_party.side_of(doc, node, Path(period_dir).parent / "wipe-challenger", extra)
    if party.host != node and c["egress_bits"] is not None:
        raise ValueError(f"period.json's wipe.egress_bits reaches only a challenger on this node: the challenger on "
                         f"{party.host} signs the count its own side is started with (challenger.wipe.command)")
    memory, store = store_of(c), None
    try:
        tier = c["tier"] or (TIER if memory is None else memory.tier)
        reads = c["reads"] or READS
        if isinstance(memory, wipe_store.Joint):
            params = params_for(sum(b for _, b in memory.parts) // SEGMENT_BYTES, k=reads, limit_ns=c["limit_ns"],
                                tiers=memory.declared())
        else:
            params = params_for(segments_of(c, memory), k=reads, limit_ns=c["limit_ns"], tier=tier,
                                tier_bytes=None if memory is None else memory.info().get("total_bytes"))
        foreign = foreign_labels(c["foreign_bytes"], params)
        store = None if memory is None else wipe_store.StoreProver(params, memory, foreign=foreign)
        if store is not None:
            store.gate()
        kept = None if memory is None else memory.info() | {"bytes": store.size}
        t, p0 = time.monotonic(), time.time_ns()
        in_python = keeping(foreign) if foreign else sw.Prover
        wipe = run(node, params, party=party, prover=in_python if store is None else lambda _: store)
        seconds, p1 = round(time.monotonic() - t, 2), time.time_ns()
        prover = None if store is None else store.record()
    finally:
        if store is not None:
            store.close()
        if memory is not None:
            memory.close()
    k = json.loads(bytes.fromhex(wipe.case["commitment"]))["k"]
    if c["reads"] is not None and k != c["reads"]:
        raise RuntimeError(f"the memory challenger on {party.host} ran {k} reads, not period.json's wipe.reads "
                           f"{c['reads']}: its command chooses them (wipe_party --reads)")
    links = [wipe_party.link(e["name"], node, e["to"], e["via"], mbps=e.get("mbps"), counted_by=e.get("counted_by"))
             for e in c["links"]]
    if party.host != node and not any(e["to"] == party.host for e in c["links"]):
        conn = wipe.report["connection"]
        links.append(wipe_party.link("challenger", node, party.host, f"{conn['local']} to {conn['peer']}"))
    where = wipe_party.placement(node, party.host, store=kept, simulated=c["placement"] == "simulated", links=links,
                                 undeclared=undeclared(tier, memory_bits(params) // 8, kept))
    window = {"challenger": wipe.report.get("window"),
              "prover": {"t0": wipe_party.utc(p0), "t1": wipe_party.utc(p1),
                         "clock": f"{socket.gethostname()}'s real-time clock, around the whole session (the "
                                  "challenger's start included)"}}
    answer, lean = wipe.answer, None
    if "wipe-protocol" in tools:
        lean = serve(tools["wipe-protocol"], [wipe.case])[0]
        if {k: v for k, v in lean.items() if k != "reason"} != {k: v for k, v in answer.items() if k != "reason"}:
            raise RuntimeError(f"wipe-protocol answered {lean}, the Python {answer}")
        answer = lean
    if answer["profile"] is not None:
        answer = answer | {"profile": answer["profile"] | {"placement": where}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "wipe.case.json").write_text(json.dumps(wipe.case, sort_keys=True))
    (out / "wipe.json").write_text(json.dumps({"answer": answer, "lean": lean, "placement": where, "prover": prover,
                                               "report": wipe.report, "seconds": seconds, "session": timing(wipe.signed),
                                               "uncertified": uncertified(where, wipe.signed), "window": window,
                                               "wipe": c | {"foreign_labels": foreign, "k": k,
                                                            "memory_bits": memory_bits(params),
                                                            "outside_bits": params.outside_bits,
                                                            "segments": params.labels // SEGMENT_LABELS, "tier": tier}},
                                              sort_keys=True, indent=1))
    return answer


def summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """`wipe.json`'s headline: the verdict and who gave it, the reads the challenger ran (`k`), where each party ran, the
    session's window on the challenger's clock (`t0`, `t1`), what the bound doesn't cover, and the fill's and the reads'
    figures on each side."""
    a, prover, session, fill = record["answer"], record["prover"] or {}, record["session"] or {}, record["report"].get("fill")
    w, window = record["wipe"], (record.get("window") or {}).get("challenger") or {}
    return {"verdict": a["verdict"], "reason": a["reason"], "public": a["public"], "judge": "lean" if record["lean"] else
            "python", "k": w.get("k"), "segments": w["segments"], "tier": w["tier"],
            "foreign_labels": w.get("foreign_labels", 0),
            "placement": record["placement"], "seconds": record["seconds"],
            "t0": window.get("t0"), "t1": window.get("t1"), "uncertified": record.get("uncertified", []),
            "challenger": {"fill_GBps": (session.get("fill") or {}).get("GBps"), "reads": session.get("reads"),
                           "streamer": fill},
            "prover": {"fill_GBps": (prover.get("fill") or {}).get("GBps"),
                       "copy_GBps": (prover.get("fill") or {}).get("copy_GBps"), "device_reads": prover.get("reads"),
                       "sync_ns": (prover.get("fill") or {}).get("sync_ns"), "gate": prover.get("gate")}}


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m verity.core.protocols.pous.wipe_period --period DIR --out DIR [--node NAME] [--expect VERDICT]`: the
    leg on this node, its summary printed; exit 0 when the verdict is the expected one (`accept` by default), 1
    otherwise."""
    ap = argparse.ArgumentParser(prog="python -m verity.core.protocols.pous.wipe_period", description=main.__doc__)
    ap.add_argument("--period", type=Path, required=True, help="the period's directory, holding period.json")
    ap.add_argument("--out", type=Path, required=True, help="where wipe.json and wipe.case.json go")
    ap.add_argument("--node", default=socket.gethostname(), help="the prover's node: the wiped server's id")
    ap.add_argument("--expect", choices=("accept", "reject"), default="accept",
                    help="the verdict that exits 0 (reject: a cheat's run)")
    a = ap.parse_args(argv)
    answer = leg(a.period, a.node, a.out)
    print(json.dumps(summary(json.loads((a.out / "wipe.json").read_text())), indent=1, sort_keys=True), flush=True)
    return 0 if answer["verdict"] == a.expect else 1


__all__ = ["CONFIG", "COUNT_BITS", "DISK", "DRAM", "EPS", "EXPERIMENTAL", "LIMIT_NS", "LINK_KEYS", "MAX_READS",
           "NOT_DISKS", "PROTOCOL", "READS", "SAMPLING_BITS", "SEGMENTS", "SEGMENT_BYTES", "UNCOUNTED", "Keeping", "Wipe", "answer",
           "case", "certify", "config", "digest", "disk_gaps", "disks", "dram_available", "foreign_bits",
           "foreign_labels", "gaps", "keeping", "leg",
           "main", "memory_bits",
           "outcome", "params_for", "profile", "ratio", "ratio_of", "run", "segments_of", "serve", "store_of", "summary",
           "timing", "uncertified", "undeclared"]

if __name__ == "__main__":
    sys.exit(main())
