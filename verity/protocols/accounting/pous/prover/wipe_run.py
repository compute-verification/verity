"""One wipe of one node at a small fill, the prover's end of it: the memory challenger's side fills the node's memory with
its streamed labels and times its spot reads, the prover keeps the fill and answers them, and the verifier answers the
session as the wipe's component of onsite Verity (`wipe_period`).

`run(node, params, ...)` runs one wipe: in one process (the challenger under `signer` and `log`, on `clock`), or with
the memory challenger a party of its own (`party`, `wipe_party`), which samples its secret, session id and coins on its
own host and signs under its own key while the prover answers over a socket. The prover is `wipe_store.Prover` (the
honest server) unless a test passes another, such as `wipe_store.keeping`. Then the verifier answers
(`wipe_period.answer`). `Side` is the challenger's side as what starts a run hands it over, wherever `period.json` puts
it, and `session(side, request, prover)` is the prover's end of it: one session with the side, answered from `prover`
(`answer`, over `wipe_party`'s frames), with the connection's two addresses as it saw them.

`leg(period_dir, node, out, side_of=...)` is the wipe's leg of `benchmarks/all_protocols`: one wipe as `period.json`'s
`wipe` says (`CONFIG`: `SEGMENTS` 32 MiB segments in host memory by default, or a GPU's memory through `wipe_store`, the
whole device declared, the driver's reserve included (`wipe_store.Cuda.info`'s `total_bytes`), so that its memory
outside the fill counts in `R`, the server's disks ahead of it in the same session, `wipe.disk`, and the host's DRAM
after it, `wipe.dram`), the memory challenger on `period.json`'s `challenger_host` (this node by default, a process
whose key and session log are beside the periods), its side handed in by the caller that starts the run (`side_of`).
The verifier trusts one memory-challenger key, the one whose SHA-512 `wipe.trusted_sha512` pins: a session signed under another is rejected, and a challenger on its own host with no
key pinned is refused before its session. Only a challenger on this node, whose key the run made, is trusted under the
key its session came with. No network certifier counts the session's egress yet, so the check is given no
count unless `wipe.egress_bits` names one, and an accepted leg certifies nothing either way (`wipe_period.uncounted`): its verdict
is real and its quantity isn't. The leg's record says `trusted-host` holds only for a session signed under the challenger host's key
that `wipe_party.AUDITOR_KEYS` pins, never from the hosts' names: a session accepted under a key the run pinned itself
is a rehearsal with no bound. It names what the bound doesn't cover (`uncertified`): the key custody of any other
session, an uncounted egress, the node's memory outside the declared tiers (`undeclared`), a GPU's driver reserve when NVML gave no total, what its disks
leave outside the boundary (`wipe_period.disk_gaps`) and each link leaving the node that no network certifier counts
(`wipe_party.link`).
`main` runs the leg on this node and prints its `summary` (`benchmarks/pous/streamed_wipe/demo.sh` runs it on one GPU,
through `wipe_on_node.py`, which hands it the runtime's `Side.of`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import socket
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol, TypeVar

from verity.protocols.accounting.pous.device import memory_challenger as mc
from verity.protocols.accounting.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.pous.verifier import wipe_party
from verity.protocols.accounting.pous.verifier import wipe_period as W
from verity.protocols.accounting.pous.verifier.period import decided_by, serve
from verity.protocols.accounting.pous.verifier.wipe_party import ACK, CHALLENGE, DONE, END, FILL, READS, _read
from verity.protocols.accounting.pous.verifier.wipe_period import (LIMIT_NS, PROTOCOL, SEGMENT_BYTES, SEGMENT_LABELS,
                                                                   SEGMENTS, TIER, case, egress_of, foreign_labels,
                                                                   memory_bits, params_for, uncertified)

from . import wipe_store


T = TypeVar("T")


class Side(Protocol):
    """The wipe challenger's side as what starts a run hands it over (the runtime's `Side` is one): the machine it runs
    on, and `talk(request, drive)`, one session with it: `request` sent, `drive` handed the connected socket, the side's
    report back with what `drive` returned."""

    host: str

    def talk(self, request: Mapping[str, Any], drive: Callable[[socket.socket], T]) -> tuple[dict[str, Any], T]: ...


def answer(f: BinaryIO, prover: wipe_store.Prover, stop_after: int | None = None) -> int:
    """The prover's side over the connection `f`: each fill frame kept by `prover` (read straight into its store when
    it has `receive_into`, as `wipe_store.StoreProver`), the fill acknowledged once it is in the store (after the
    prover's `sync`, when it has one: a disk's writes on the media), each batch answered from it until the challenger
    ends the session (or, with `stop_after`, the prover leaves after that many batches); the batches answered."""
    n = 0
    into, sync = getattr(prover, "receive_into", None), getattr(prover, "sync", None)
    while (tag := _read(f, 1)) != END:
        if tag == FILL:
            first, count = int.from_bytes(_read(f, 8), "big"), int.from_bytes(_read(f, 4), "big")
            if into is not None:
                into(first, count, f)
            else:
                prover.receive(first, _read(f, sw.LABEL_BYTES * count))
        elif tag == DONE:
            if sync is not None:
                sync()
            f.write(ACK)
            f.flush()
        elif tag == CHALLENGE:
            if stop_after is not None and n >= stop_after:
                return n
            count = int.from_bytes(_read(f, 4), "big")
            f.write(prover.answer([int.from_bytes(_read(f, 8), "big") for _ in range(count)]))
            f.flush()
            n += 1
        else:
            raise ValueError(f"the challenger sent the frame tag {tag!r}")
    return n


def session(side: Side, request: Mapping[str, Any], prover: wipe_store.Prover, *,
            stop_after: int | None = None) -> dict[str, Any]:
    """One wipe with the challenger's side `side` starts: `request` sent, the fill kept and each batch answered by
    `prover`, the side's report back, with `answered`, the batches the prover answered, and `connection`, the
    socket's `local` and `peer` addresses (`HOST:PORT`) on the prover's side."""

    def drive(s: socket.socket) -> tuple[int, dict[str, str]]:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection = {k: f"{a[0]}:{a[1]}" for k, a in (("local", s.getsockname()), ("peer", s.getpeername()))}
        with s.makefile("rwb") as f:
            return answer(f, prover, stop_after), connection

    out, (answered, connection) = side.talk(request, drive)
    return out | {"answered": answered, "connection": connection}


@dataclass(frozen=True)
class Wipe:
    """A period's wipe: the protocol's answer and case, how the session ran (`ended`, `reads`, the challenger's `host`,
    the SHA-512 of the key it signs under, `key_sha512`, and whether that is the key the verifier pinned, `key_pinned`;
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
        prover: Callable[[sw.WipeParams], wipe_store.Prover] = wipe_store.Prover, egress_mib: int | None = None,
        clock: Callable[[], int] = sw.raw_clock_ns, secret: bytes | None = None, session_id: bytes | None = None,
        coin_secret: bytes | None = None, stop_after: int | None = None, trusted_sha512: str | None = None,
        schemes: Mapping[str, mc.Verify] = mc.SCHEMES, protocol: str = PROTOCOL,
        chunk: int = wipe_party.CHUNK) -> Wipe:
    """One wipe of `node` (module docstring), the server id its name's UTF-8. In one process the challenger signs under
    `signer`, claims its session in `log` and reads `clock` (the raw monotonic clock, which the session's header
    names); `secret`, `session_id` and `coin_secret`
    are fresh from the OS unless a test gives them. With `party` its side runs where `party` starts it, with its own
    secret, coins, key, log and clock, and those arguments are unused. `prover` makes the prover's side,
    which with `stop_after` leaves after that many reads. `egress_mib` is the evidence's egress count, which the
    verifier's check takes beside the signed session (`egress_of`); the challenger never holds it. `trusted_sha512` is the verifier's pin, the SHA-512 in hex of
    the one memory-challenger key it trusts: a session signed under another key is rejected. With none it trusts the key
    the session came with, which only a challenger whose key this run made may be given (never one on its own host).
    `schemes` are the signature schemes it accepts and `protocol` the id it answers under."""
    params = params_for() if params is None else params
    server = node.encode()
    if party is None:
        assert signer is not None and log is not None, "an in-process wipe needs the challenger's signer and log"
        c = sw.WipeMemoryChallenger(secrets.token_bytes(32) if secret is None else secret, server,
                                    secrets.token_bytes(32) if session_id is None else session_id, params, signer,
                                    log=log, clock=clock, coin_secret=coin_secret)
        p = prover(params)
        for first, data in c.fill(chunk):
            p.receive(first, data)
        if (sync := getattr(p, "sync", None)) is not None:
            sync()
        while c.ended is None and (stop_after is None or len(c.batches) < stop_after):
            c.receive(p.answer(c.challenge()))
        signed = None if c.ended is None else c.close()
        commitment, key = c.commitment, signer.public_key
        report = {"ended": c.ended, "host": node, "reads": len(c.batches)}
    else:
        got = session(party, wipe_party.request_of(node, params), prover(params), stop_after=stop_after)
        commitment, key = wipe_party.commitment_of(got["commitment"]), bytes.fromhex(got["public_key"])
        signed = None if got["signed"] is None else wipe_party.signed_of(got["signed"])
        report = {"ended": got["ended"], "host": got["host"], "reads": got["reads"]}
        report |= {k: got[k] for k in ("connection", "fill", "streamer", "window") if k in got}
    key_sha = hashlib.sha512(key).hexdigest()
    report |= {"key_sha512": key_sha, "key_pinned": key_sha == trusted_sha512}
    trusted = (key,) if trusted_sha512 in (None, key_sha) else ()
    return Wipe(W.answer(server, commitment.session, trusted, commitment, signed, schemes, protocol,
                       egress_mib=egress_mib),
                case(server, commitment.session, trusted, commitment, signed, egress_mib), report, signed)


CONFIG = {"store": "python", "device": 0, "segments": None, "reserve_bytes": 1 << 30, "tier": None,
          "limit_ns": LIMIT_NS, "streamer": "python", "threads": None, "foreign_bytes": 0, "placement": None,
          "egress_bits": None, "reads": None, "links": [], "disk": None, "dram": None, "trusted_sha512": None}
"""`period.json`'s `wipe` and its defaults: the prover's `store` (`python`, `wipe_store.Prover`; `host`, host memory
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
`egress_bits` is the network certifier's count over the session, in bits, which the verifier's check is given as
evidence in whole MiB (`streamed_wipe.egress_units`), wherever the challenger runs; `None` gives none, and the check
takes `EGRESS_MAX_UNITS`, which bounds nothing. `reads` is the session's timed reads: a challenger on this node is started
with it (`wipe_party --reads`); one on its own host chooses its own (its `challenger.wipe.command`), and the leg refuses
a session at another count. `None`: whatever the challenger chooses (`READS` here). `links` are the links leaving the
node besides the one to a challenger on its own host, which the leg adds from the connection it made: each `{"name",
"to", "via"}`, with `mbps` and `counted_by` (the network certifier that counts it) when known (`wipe_party.link`).
`trusted_sha512` is the verifier's pin, the SHA-512 of the memory challenger's public key in 128 lowercase hex digits:
the only key a session is accepted under. A challenger on its own host needs it; `None` trusts the key a challenger on
this node made for the run."""
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
    if (pin := c["trusted_sha512"]) is not None and not (type(pin) is str and re.fullmatch("[0-9a-f]{128}", pin)):
        raise ValueError(f"period.json's wipe.trusted_sha512 is null or the SHA-512 of the memory challenger's public "
                         f"key in 128 lowercase hex digits, not {pin!r}")
    return c


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


def undeclared(tier: str, declared_bytes: int, store: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The memory this host holds outside the declared tier `tier` (`declared_bytes` of it), which `M` doesn't count:
    the device's memory outside the tier (from `store`, `wipe_store`'s `info`, when it gives `total_bytes` past the
    tier), the host's DRAM (outside the     tier when it is the tier; none when it is declared whole) and each of its block devices, mounted or not, whole
    (`disks`). A joint store's (`wipe_store.Joint`) tiers are its parts', each at its declared bytes, and the disks a
    `disk` part declares are no one's undeclared memory."""
    out, parts = [], W._parts(store)
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


def leg(period_dir: Path, node: str, out: Path, *, side_of: Callable[..., Side]) -> dict[str, Any]:
    """The wipe's leg of `benchmarks/all_protocols` (`leg(period_dir, node, out, side_of=...) -> answer`): one wipe as
    `period.json`'s `wipe` says (`CONFIG`; by default `SEGMENTS` 32 MiB segments kept by `wipe_store.Prover`), the
    memory challenger a party of its own on `period.json`'s `challenger_host` (`side_of(period, node, key, local,
    what)`, which the caller that starts the run hands in, the runtime's `Side.of`; on this node by default, a process
    whose key and session log are in `wipe-challenger` beside the periods) and the prover here. The verifier trusts the
    key `wipe.trusted_sha512` pins and no other; a challenger on its own host with none pinned is refused (ValueError)
    before its session. A `wipe_store` prover gates its store before the session.
    With `wipe-protocol` among the period's tools, the Lean answers (the
    verifier) and must agree with the Python; it writes `out/wipe.case.json` (onsite's wipe case) and `out/wipe.json`,
    which names the program that answered and the host it ran on (`decided_by`, `period.decided_by`).
    Which host ran each party, whether `trusted-host`'s premise holds (the session's key against the pin
    `wipe_party.AUDITOR_KEYS` holds for the challenger's host), the memory the prover kept the fill in,
    the links leaving the node (`wipe.links`, and the challenger's on its own host as the prover connected to it) and
    the node's memory outside the declared tier (`undeclared`) go in `wipe.json` and in an accepted answer's profile
    (`placement`, `wipe_party.placement`); `wipe.json` adds the prover's own timings (`prover`: its store, gate, fill
    and device read latency), the signed session's on the challenger's clock (`session`: the fill's GB/s, each read's
    round trip), the streamer's (`report.fill`), the session's window on each side's real-time clock (`window`) and
    what the bound doesn't cover (`uncertified`)."""
    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools, c = doc.get("tools") or {}, config(doc)
    extra = (("--streamer", "build", *(("--threads", str(c["threads"])) if c["threads"] else ()))
             if c["streamer"] == "native" else ()) + (() if c["reads"] is None else ("--reads", str(c["reads"])))
    party = side_of(doc, node, "wipe", wipe_party.command(Path(period_dir).parent / "wipe-challenger", extra),
                    "the wipe")
    egress = None if c["egress_bits"] is None else sw.egress_units(c["egress_bits"])
    if party.host != node and c["trusted_sha512"] is None:
        raise ValueError(f"period.json's wipe.trusted_sha512 is unset with the memory challenger on {party.host}: the "
                         "verifier trusts a challenger on its own host only under its pinned key (the SHA-512 of its "
                         "public key, as its host gives it)")
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
        in_python = wipe_store.keeping(foreign) if foreign else wipe_store.Prover
        wipe = run(node, params, party=party, prover=in_python if store is None else lambda _: store,
                   egress_mib=egress, trusted_sha512=c["trusted_sha512"])
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
    where = wipe_party.placement(node, party.host, wipe.report["key_sha512"], store=kept,
                                 simulated=c["placement"] == "simulated", links=links,
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
    (out / "wipe.json").write_text(json.dumps({"answer": answer, "decided_by": decided_by(tools.get("wipe-protocol")),
                                               "lean": lean, "placement": where, "prover": prover,
                                               "report": wipe.report, "seconds": seconds, "session": timing(wipe.signed),
                                               "uncertified": uncertified(
                                                   where, None if wipe.signed is None else egress_of(egress)),
                                               "window": window,
                                               "wipe": c | {"foreign_labels": foreign, "k": k,
                                                            "memory_bits": memory_bits(params),
                                                            "outside_bits": params.outside_bits,
                                                            "segments": params.labels // SEGMENT_LABELS, "tier": tier}},
                                              sort_keys=True, indent=1))
    return answer


def summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """`wipe.json`'s headline: the verdict and who gave it (`decided_by`: the program, `wipe-protocol` or `python`, and
    the host it ran on), the SHA-512 of the challenger key the session was signed under (`key_sha512`) and whether it is
    the key the verifier pinned (`key_pinned`; an accept without it trusted the key of a challenger on the node itself),
    the reads the challenger ran (`k`), where each party ran, the session's window on the challenger's clock (`t0`,
    `t1`), what the bound doesn't cover, and the fill's and the reads' figures on each side."""
    a, prover, session, fill = record["answer"], record["prover"] or {}, record["session"] or {}, record["report"].get("fill")
    w, window = record["wipe"], (record.get("window") or {}).get("challenger") or {}
    return {"verdict": a["verdict"], "reason": a["reason"], "public": a["public"], "decided_by": record["decided_by"],
            "key_sha512": record["report"]["key_sha512"], "key_pinned": record["report"]["key_pinned"],
            "k": w.get("k"), "segments": w["segments"], "tier": w["tier"],
            "foreign_labels": w.get("foreign_labels", 0),
            "placement": record["placement"], "seconds": record["seconds"],
            "t0": window.get("t0"), "t1": window.get("t1"), "uncertified": record.get("uncertified", []),
            "challenger": {"fill_GBps": (session.get("fill") or {}).get("GBps"), "reads": session.get("reads"),
                           "streamer": fill},
            "prover": {"fill_GBps": (prover.get("fill") or {}).get("GBps"),
                       "copy_GBps": (prover.get("fill") or {}).get("copy_GBps"), "device_reads": prover.get("reads"),
                       "sync_ns": (prover.get("fill") or {}).get("sync_ns"), "gate": prover.get("gate")}}


def main(argv: Sequence[str] | None = None, *, side_of: Callable[..., Side]) -> int:
    """`--period DIR --out DIR [--node NAME] [--expect VERDICT]`: the wipe on this node, its challenger's side where
    `side_of` puts it (the runtime's `Side.of`, handed in by the program that starts the run), its summary printed; exit
    0 when the verdict is the expected one (`accept` by default), 1 otherwise."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--period", type=Path, required=True, help="the period's directory, holding period.json")
    ap.add_argument("--out", type=Path, required=True, help="where wipe.json and wipe.case.json go")
    ap.add_argument("--node", default=socket.gethostname(), help="the developer's node: the wiped server's id")
    ap.add_argument("--expect", choices=("accept", "reject"), default="accept",
                    help="the verdict that exits 0 (reject: a cheat's run)")
    a = ap.parse_args(argv)
    answer = leg(a.period, a.node, a.out, side_of=side_of)
    print(json.dumps(summary(json.loads((a.out / "wipe.json").read_text())), indent=1, sort_keys=True), flush=True)
    return 0 if answer["verdict"] == a.expect else 1


__all__ = ["CONFIG", "DISK", "DRAM", "LINK_KEYS", "NOT_DISKS", "Side", "Wipe", "answer", "config", "disks",
           "dram_available", "leg", "main", "run", "segments_of", "session", "store_of", "summary", "timing",
           "undeclared"]
