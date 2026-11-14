"""The streamed wipe's verifier, the wipe's component of onsite Verity (`verity/wipe/v0`, `MemoryChallenger.WipeProtocol`
in `verity/protocols/accounting/pous/device/lean/`): the memory challenger fills the node's memory with its streamed
labels and times its spot reads (`streamed_wipe`) and signs the session, and the verifier answers (`answer`), as the
Lean does, on the commitment and the evidence: the signed session and one egress count beside it, `egress_mib`, in whole
MiB (`streamed_wipe.egress_units`), `EGRESS_MAX_UNITS` when none is given (`egress_of`):

* a commitment of another server or session rejects;
* an egress outside 0 to `EGRESS_MAX_UNITS` rejects;
* a session the memory challenger hasn't signed (the prover left, or its reads aren't verified yet) is undecided;
* else `streamed_wipe.verdict`, whose reasons, joined by `"; "`, are a rejection's: a late or wrong read, an untrusted
  key, a bad signature, an uncovered batch size.

An accepted wipe publishes `uncounted`, `certify` at `EGRESS_MAX_UNITS`: `certified` false, whatever count the evidence
names. No network certifier signs that count, so until the check that composes the wipe with the network certifier
holds it to the certifier's, the bound is taken on no count, and the profile names the evidence's count as a gap
(`writer_gap`). `certify` is what that check is to publish: `{"certified", "egress_mib", "eps", "foreign_bits", "k",
"memory_bits", "r"}`, `memory_bits = M` the declared tiers' bits (the fill's `n · 512` and `R` past it, the memory it doesn't reach),
`k` the session's timed reads, which the memory challenger chose (`wipe_party --reads`), and `r = ratio(k)`, the
largest `r` with `(r − 1)^k · 2^129 ≤ r^k`. When the bound covers the session (`k` from 1 to `MAX_READS`, an egress count
that isn't saturated) `foreign_bits` is `Pous.Guarantees.StreamedWipeMeetsReads`' threshold
`512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` at `eps = 2^-128` (`foreign_bits`), with `E` the evidence's egress.
Off it, `foreign_bits` is `M` and `eps` 0, the vacuous statement, and the profile names each gap. `EXPERIMENTAL` says
what the bound rests on. The session signs no egress, and this check holds the evidence's count to nothing: it is the
count whoever writes the evidence gives. That it is the network certifier's count over the span between the session's
two marks is for the check that composes the wipe with the network certifier to establish.


What runs one wipe, the prover's end of its session and the answer here, is `pous/prover/wipe_run.py`. What an accepted wipe's bound doesn't cover is `uncertified`, from its placement
(`wipe_party.placement`): the key custody of a session not signed under the key `wipe_party.AUDITOR_KEYS` pins for the
challenger's host, an uncounted egress, the node's memory outside the declared tiers, a GPU's driver reserve when NVML
gave no total, what its disks leave outside the boundary (`disk_gaps`) and each link leaving the node that no network
certifier counts (`wipe_party.link`). `foreign_labels` is the labels a cheat of a given size keeps prior content in
place of.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from fractions import Fraction
from typing import Any

from verity.commitments.identity import canonical_json_bytes

from verity.protocols.accounting.pous.device import memory_challenger as mc
from . import streamed_wipe as sw
from . import wipe_party
from .wipe_party import READS

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
EGRESS_RANGE = "the evidence's egress is {} MiB, outside 0 to {}"
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
    """`MemoryChallenger.WipeProtocol.gaps`: why the bound doesn't cover an accepted wipe (none when it does). `k = 0` is
    one: `WipeParams` refuses it, but `ratio(0) = 1` doesn't sample."""
    out = []
    if p.k == 0:
        out.append("k = 0: a session of no timed reads bounds nothing")
    elif p.k > MAX_READS:
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


def uncounted(p: sw.WipeParams) -> dict[str, Any]:
    """`MemoryChallenger.WipeProtocol.uncounted`, what an accepted wipe publishes: `certify` at `EGRESS_MAX_UNITS`,
    which certifies nothing, whatever count the evidence names, since no network certifier signs that count."""
    return certify(p, sw.EGRESS_MAX_UNITS)


def writer_gap(egress_mib: int) -> list[str]:
    """`MemoryChallenger.WipeProtocol.writerGap`: the evidence's count, when it names one `gaps` doesn't: the evidence
    writer's, which no network certifier signs."""
    if not 0 <= egress_mib < sw.EGRESS_MAX_UNITS:
        return []
    return [f"the egress count is {egress_mib} MiB, the evidence writer's, which no network certifier signed: no wipe "
            "is certified on it until the check that composes the wipe with the network certifier"]


def profile(p: sw.WipeParams, egress_mib: int) -> dict[str, Any]:
    """`MemoryChallenger.WipeProtocol.profileJson`: the wipe the session ran, the bound and every gap, the evidence's
    count among them."""
    return {"batch": p.batch,
            "certificate": {"eps": str(EPS), "max_reads": MAX_READS, "outside_bits": p.outside_bits, "r": ratio_of(p),
                            "reads": p.k},
            "experimental": EXPERIMENTAL, "gaps": gaps(p, egress_mib) + writer_gap(egress_mib), "k": p.k,
            "labels": p.labels,
            "limit_ns": p.limit_ns, "tiers": [[t, b] for t, b in p.tiers]}


def egress_of(egress_mib: int | None) -> int:
    """`MemoryChallenger.WipeProtocol.Evidence.egress`: the check's one egress count, in whole MiB: the evidence's, or
    `EGRESS_MAX_UNITS`, which bounds nothing, when it gives none."""
    return sw.EGRESS_MAX_UNITS if egress_mib is None else egress_mib


def outcome(server: bytes, session: bytes, trusted: Collection[bytes], commitment: sw.WipeCommitment,
            signed: sw.SignedWipe | None, schemes: Mapping[str, mc.Verify] = mc.SCHEMES, *,
            egress_mib: int | None = None) -> tuple:
    """`MemoryChallenger.WipeProtocol.check` on decoded inputs, the evidence the signed session and `egress_mib` beside
    it (`egress_of`): `("accept", public, profile)`, `("reject", reason)` or `("undecided", reason)`."""
    if commitment.server != server or commitment.session != session:
        return ("reject", OTHER)
    egress = egress_of(egress_mib)
    if not 0 <= egress <= sw.EGRESS_MAX_UNITS:
        return ("reject", EGRESS_RANGE.format(egress, sw.EGRESS_MAX_UNITS))
    if signed is None:
        return ("undecided", UNSIGNED)
    v = sw.verdict(signed, commitment, trusted=set(trusted), schemes=schemes)
    if not v.accepted:
        return ("reject", "; ".join(v.reasons))
    return ("accept", uncounted(commitment.params), profile(commitment.params, egress))


def digest(commitment: sw.WipeCommitment) -> bytes:
    """SHA-512 of the commitment's canonical JSON."""
    return hashlib.sha512(canonical_json_bytes(commitment.to_json())).digest()


def params_json(server: bytes, session: bytes, trusted: Sequence[bytes]) -> dict[str, Any]:
    return {"server": server.hex(), "session": session.hex(), "trusted": [t.hex() for t in trusted]}


def case(server: bytes, session: bytes, trusted: Sequence[bytes], commitment: sw.WipeCommitment,
         signed: sw.SignedWipe | None, egress_mib: int | None = None) -> dict[str, Any]:
    """The wipe protocol's case, `{"commitment": hex, "evidence": hex, "params": JSON}`, the evidence
    `{"egress_mib": N | null, "signed": {...} | null}`."""
    evidence = {"egress_mib": egress_mib, "signed": None if signed is None else signed.to_json()}
    return {"commitment": canonical_json_bytes(commitment.to_json()).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": params_json(server, session, trusted)}


def answer(server: bytes, session: bytes, trusted: Collection[bytes], commitment: sw.WipeCommitment,
           signed: sw.SignedWipe | None, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
           protocol: str = PROTOCOL, *, egress_mib: int | None = None) -> dict[str, Any]:
    """The wipe protocol's answer (`Core.Answer.canonical`) on decoded inputs."""
    verdict, *rest = outcome(server, session, trusted, commitment, signed, schemes, egress_mib=egress_mib)
    public, prof, reason = (*rest, None) if verdict == "accept" else (None, None, rest[0])
    return {"commitment": digest(commitment).hex(), "profile": prof, "protocol": protocol, "public": public,
            "reason": reason, "verdict": verdict}


def foreign_labels(foreign_bytes: int | str, p: sw.WipeParams) -> int:
    """The labels a cheat of `foreign_bytes` keeps prior content in place of: whole labels covering the bytes, or for
    `"fM"` the fewest labels past `p`'s published `foreign_bits` with no egress, the least a wipe must reject."""
    if foreign_bytes == "fM":
        return min(p.labels, certify(p, 0)["foreign_bits"] // sw.LABEL_BITS + 1)
    return min(p.labels, -(-int(foreign_bytes) // sw.LABEL_BYTES))


def _parts(store: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    return [] if store is None else list(store.get("parts") or [store])


VIRTUAL_DRIVERS = ("virtio_blk", "xen_blkfront", "vmw_pvscsi")
"""Drivers of a virtual disk, whose backing store is the hypervisor's."""


def disk_gaps(store: Mapping[str, Any] | None) -> list[str]:
    """What a `disk` part of the prover's store (`wipe_store.Disk.info`) leaves outside the boundary: a virtual disk's
    backing store, the hypervisor's (`VIRTUAL_DRIVERS`), whose snapshots and replicas the fill doesn't
    reach; a drive's spare area and cache when `spare_bytes` doesn't declare them; and buffered I/O, which the page
    cache can answer a read from."""
    out = []
    for p in _parts(store):
        if p.get("tier") != "disk":
            continue
        for name, d in sorted(p["disks"].items()):
            if d["driver"] in VIRTUAL_DRIVERS:
                out.append(f"disk {name}: a virtual disk ({d['driver']}), its backing store the hypervisor's: a "
                           "snapshot or replica the cloud keeps of it is outside the boundary, which Covered assumes away")
            elif not p["spare_bytes"]:
                out.append(f"disk {name}: its spare area and cache are undeclared (spare_bytes 0), so M doesn't count "
                           "what they keep")
        if not p["direct"]:
            out.append("disk: buffered I/O, not O_DIRECT: the page cache can answer a read the disk doesn't keep")
    return out


def uncertified(where: Mapping[str, Any], egress_mib: int | None) -> list[str]:
    """What an accepted wipe's bound doesn't cover, from its placement (`wipe_party.placement`) and the egress its
    check took (`egress_of` of the evidence's; `None` leaves the egress out): the key custody, unless the session was
    signed under the challenger host's pinned key (`wipe_party.holds`), the egress (uncounted at `EGRESS_MAX_UNITS`;
    any other count is the evidence writer's, which no network certifier signed, and certifies no wipe), each
    undeclared tier, a GPU's driver reserve when NVML gave no total (`wipe_store.Cuda.info`; a joint store's GPU part too), what
    the prover's disks leave outside the boundary (`disk_gaps`) and each link leaving the node that no network
    certifier counts."""
    out = [] if wipe_party.holds(where) else [f"key custody: {where[wipe_party.PREMISE]}"]
    if egress_mib is not None and egress_mib >= sw.EGRESS_MAX_UNITS:
        out.append("egress: uncounted, the check was given no network certifier's count over the session, so the bound "
                   "is vacuous")
    elif egress_mib is not None:
        out.append(f"egress: {egress_mib} MiB, the evidence writer's count, which no network certifier signed, so the "
                   "check certifies no wipe on it until the check that composes the wipe with the network certifier")
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


__all__ = ["COUNT_BITS", "EPS", "EXPERIMENTAL", "LIMIT_NS", "MAX_READS", "PROTOCOL", "READS", "SAMPLING_BITS", "SEGMENTS",
           "SEGMENT_BYTES", "VIRTUAL_DRIVERS", "answer", "case", "certify", "digest", "disk_gaps", "egress_of",
           "foreign_bits", "foreign_labels", "gaps", "memory_bits", "outcome", "params_for", "profile", "ratio",
           "ratio_of", "uncertified", "uncounted", "writer_gap"]
