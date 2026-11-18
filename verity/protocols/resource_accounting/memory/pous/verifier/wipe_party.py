"""The streamed wipe's memory attester as a party of its own, so it runs on a host the developer doesn't control (the
challenger host, `trusted-device`) while the prover stays on its node: `challenger_party`'s shape, for the wipe. This
module is what both sides and the verifier share: the wire's frames, the request (`request_of`, `params_of`), the
report's readers (`commitment_of`, `signed_of`) and where the challenger ran (`placement`). The challenger's side is
`verity/protocols/resource_accounting/memory/pous/attester/wipe_challenger.py` (`serve`, `report`, `main`).

The challenger listens and the prover connects. The challenger streams the fill as frames `F ‖ u64 first ‖ u32 count ‖
labels` and ends it with `D`, which the prover acknowledges with `A` once it holds the fill, so that no read's time
includes the fill still in flight. Then it sends each batch as `C ‖ u32 count ‖ u64 index …` (reading the send time),
and the prover answers with the batch's labels, 64 bytes an index (the challenger reads the receipt time); `E` ends the
session, after its last batch or a late or wrong one. The challenger samples the labels' secret, the session id and its coins on its own host, claims the
id in its own session log, and signs under its own key. It signs no egress count and is given none: the wipe's check
takes the count as evidence beside the signed session (`wipe_period`). It times the number of reads its own command
names (`--reads`, `READS` by default), which the commitment
publishes and the certificate is computed at: a request that names `k` is refused, so the prover never chooses it. A
prover that leaves before the session ends leaves it unsigned. The side reports `{"commitment", "ended",
"figures_sha512", "host", "public_key", "reads", "signed", "window"}`, `signed` null when unsigned, `figures_sha512` the
digest of the node kind's figures whose floors it held the request to (`--figures`; null when it held it to none), and `window` the session's `t0` (the
prover connected) and `t1` (the session signed, or ended unsigned) on the challenger host's real-time clock: the
interval a network attester counts the session's egress over.

`command(device, extra)` is the argv that starts the challenger's side (`python -m
verity.protocols.resource_accounting.memory.pous.attester.wipe_challenger`, `MODULE`) on this machine; the prover's end of the session,
which answers it, is `pous/prover/wipe_run.py`'s (`session`, `answer`). The wipe's two parties as the deployment layer
runs them speak `wipe_parties`'s messages: `pous/attester/wipe_challenger.py` and `pous/prover/wipe_prover.py`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from verity.protocols.resource_accounting.memory.poub.verifier import timing
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc

from . import streamed_wipe as sw

MODULE = "verity.protocols.resource_accounting.memory.pous.attester.wipe_challenger"
"""The module `command` starts as the challenger's side: `wipe_challenger.main`."""
PREMISE = mc.PREMISE
PARTY = "memory-challenger"
"""The configuration's party whose `[[pin]]` entries for the wipe name the keys `placement` reads."""
FILL, DONE, ACK, CHALLENGE, END = b"F", b"D", b"A", b"C", b"E"
CHUNK = 1 << 14
"""Labels a fill frame: 1 MiB."""
READS = 2152
"""The timed spot reads a challenger runs when its command doesn't name them (`--reads`)."""
REQUEST_KEYS = {"labels", "limit_ns", "node", "tiers"}


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ConnectionError(f"the peer closed after {len(b)} of {n} bytes")
    return b


def figures_pinned(path: Path, sha512: str) -> timing.Figures:
    """The node kind's figures from `path` (`timing.Figures.of_json`, what `benchmarks/poub_session/sheets.py NODE`
    writes), held to the digest the challenger's own command pins, as PoUB's verifier holds its `--figures`."""
    figures = timing.Figures.of_json(json.loads(Path(path).read_text()))
    if figures.sha512() != sha512.lower():
        raise ValueError(f"--figures {path}: its SHA-512 is {figures.sha512()}, not {sha512}, the one --figures-sha512 "
                         "pins: the figures are the verifier's configuration's, never the prover's")
    return figures


def floors_of(figures: timing.Figures) -> dict[str, int]:
    """The node kind's pinned capacity by wipe tier, from its figures (`timing.Figures`, read from the node's data
    sheets): every GPU's HBM and the host's DRAM. The figures pin no disk's size and no drive's spare area, so a disk
    tier has no floor (`floor_refusal` refuses it)."""
    return {"gpu-hbm": figures.hbm_bytes, "host-dram": figures.host_dram_bytes}


def floor_refusal(tiers: Sequence[tuple[str, int]], floors: Mapping[str, int]) -> str | None:
    """Why the declared `tiers` don't cover the node kind's pinned `floors` (none when they do): each pinned tier
    declared at no less than its pin, and no tier the floors don't pin, since its size would be the prover's alone.
    Declaring more than a pin is allowed: the floor is one-sided."""
    declared = dict(tiers)
    for t, b in sorted(floors.items()):
        if t not in declared:
            return (f"the request declares no {t}: the node kind pins {b} bytes of it, and memory it leaves out would "
                    "stay outside R")
        if declared[t] < b:
            return f"the request declares {declared[t]} bytes of {t}, under the {b} the node kind pins"
    unpinned = sorted(set(declared) - set(floors))
    if unpinned:
        return (f"the request declares {', '.join(unpinned)}, which the node kind's figures pin no size of: its "
                "size would be the prover's alone")
    return None


def params_of(request: Mapping[str, Any], reads: int = READS,
              floors: Mapping[str, int] | None = None) -> sw.WipeParams:
    """The session's parameters: the prover's request (`REQUEST_KEYS`: its node, fill, deadline and declared tiers) and
    the challenger's own `reads`, refused when `floors` (`floors_of` the node kind's pinned figures, the challenger's
    own) are given and the tiers don't cover them (`floor_refusal`)."""
    if set(request) != REQUEST_KEYS:
        raise ValueError(f"a wipe request is {sorted(REQUEST_KEYS)}, not {sorted(request)}: the challenger chooses the "
                         "reads (--reads), never the prover")
    tiers = tuple((t, b) for t, b in request["tiers"])
    if floors is not None and (why := floor_refusal(tiers, floors)):
        raise ValueError(why)
    return sw.WipeParams(request["labels"], reads, 1, request["limit_ns"], tiers)



def utc(unix_ns: int) -> str:
    """A real-time clock reading as ISO 8601 UTC, to the microsecond."""
    s, ns = divmod(unix_ns, 10 ** 9)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(s)) + f".{ns // 1000:06d}Z"


def commitment_of(j: Mapping[str, Any]) -> sw.WipeCommitment:
    """`WipeCommitment.to_json()`'s inverse, on the challenger's own report."""
    p = sw.WipeParams(j["labels"], j["k"], j["batch"], j["limit_ns"], tuple((t, b) for t, b in j["tiers"]))
    return sw.WipeCommitment(bytes.fromhex(j["server"]), bytes.fromhex(j["session"]), p)


def signed_of(j: Mapping[str, Any]) -> sw.SignedWipe:
    """`SignedWipe.to_json()`'s inverse, on the challenger's own report. A session of another format (`sw.PROTOCOL`),
    one whose header names another clock than the raw one (`sw.CLOCK`), or one with a start mark that is not one
    (`sw.mark_of`) is refused."""
    if j["commitment"].get("protocol") != sw.PROTOCOL:
        raise ValueError(f"a signed wipe is {sw.PROTOCOL}, not {j['commitment'].get('protocol')!r}")
    if j["commitment"].get("clock") != sw.CLOCK:
        raise ValueError(f"a signed wipe's times are on {sw.CLOCK}, not {j['commitment'].get('clock')!r}")
    batches = tuple(sw.BatchRecord(r["b"], r["first"], r["count"], r["send_ns"], r["receipt_ns"], r["wrong"],
                                   bytes.fromhex(r["digest"])) for r in j["batches"])
    return sw.SignedWipe(commitment_of(j["commitment"]), j["ended"], sw.mark_of(j["start_mark"]), j["fill_start_ns"],
                         j["fill_end_ns"], batches, j["scheme"], bytes.fromhex(j["public_key"]),
                         bytes.fromhex(j["signature"]))



def command(device: str | Path, extra: Sequence[str] = ()) -> tuple[str, ...]:
    """The argv that starts the challenger's side as a process on this machine, its key and session log in `device`,
    `extra` more of `main`'s arguments (`--streamer build`)."""
    return (sys.executable, "-m", MODULE, "--device", str(device), "--listen", "127.0.0.1:0", *extra)


def link(name: str, node: str, to: str, via: str, *, mbps: int | None = None,
         counted_by: str | None = None) -> dict[str, Any]:
    """A link leaving the developer's node, as `ResourceAccounting.Placed.Links` declares one: its `ends` (the node, and `to` past
    its boundary), what carries it (`via`), its capacity in Mb/s when known, and the network attester that counts what
    leaves on it (`counted_by`), `None` when none does, so its egress is outside the session's count."""
    return {"name": name, "ends": [node, to], "via": via, "mbps": mbps, "counted_by": counted_by}


def placement(node: str, host: str, key_sha512: str | None, *, pins: Sequence[Mapping[str, str]] = (),
              configuration: str | None = None, committed: Collection[str] = frozenset(), verifier: str | None = None,
              store_gaps: Sequence[str] = (), simulated: bool = False, links: Sequence[Mapping[str, Any]] = (),
              undeclared: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Which host ran each party, and whether `trusted-device`'s premise holds: that the memory attester ran on a
    machine the developer doesn't control (`memory_attester.trusted_device`). It never holds here: under a
    configuration whose hash `configuration` is one of `committed`, when `key_sha512`, the SHA-512 of the key the
    session was signed under, is a key that configuration's `pins` (its `[[pin]]` entries) give `PARTY` for the wipe,
    it reads `memory_attester.MATCHED`; under any other configuration or key, a run's own pin included, it reads
    rehearsal, whatever the names say. `kind` is what the two names say, a label and never the reason: `own-host`,
    `simulated` (the challenger a process of its own on the developer's node, standing in for its own host) or
    `prover-node`. `store_gaps` is what the memory the prover kept the fill in leaves outside the bound
    (`wipe_period.store_gaps`), by tier and never by device: the placement names no device. `links` are the links
    leaving the developer's node (`link`), and `undeclared` the memory the node has outside the fill's declared tiers
    (`{"tier", "bytes", "what"}`), which the bound doesn't count."""
    if simulated and host != node:
        raise ValueError(f"a simulated placement runs the memory attester on the developer's node {node}, not on {host}")
    kind = "own-host" if host != node else "simulated" if simulated else "prover-node"
    names = {"own-host": f"the memory attester ran on {host}, not the developer's {node}",
             "simulated": f"the memory attester ran as a process of its own on the developer's node {node}, standing "
                          "in for a host the developer doesn't control",
             "prover-node": f"the memory attester ran on the developer's node {node}"}[kind]
    premise = mc.trusted_device(pins, PARTY, "wipe", key_sha512, names, configuration=configuration,
                                committed=committed)
    return {"kind": kind, "parties": {"memory_attester": host, "prover": node, "verifier": verifier or node},
            PREMISE: premise, "links": [dict(e) for e in links], "undeclared_tiers": [dict(u) for u in undeclared],
            "store_gaps": list(store_gaps)}


def holds(where: Mapping[str, Any]) -> bool:
    """Whether a placement's `trusted-device` holds, which no placement made on the workload's machine does."""
    return mc.holds(where)


def request_of(node: str, params: sw.WipeParams) -> dict[str, Any]:
    if not params.covered:
        raise ValueError("the wipe's party reveals one index a batch, the reveal the guarantee covers")
    return {"labels": params.labels, "limit_ns": params.limit_ns, "node": node,
            "tiers": [[t, b] for t, b in params.tiers]}



__all__ = ["CHUNK", "PREMISE", "READS", "REQUEST_KEYS", "command", "commitment_of", "link", "params_of",
           "placement", "request_of", "signed_of", "utc"]
