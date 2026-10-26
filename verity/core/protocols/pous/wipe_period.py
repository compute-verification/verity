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

What an accepted wipe certifies is `certify`'s: `{"certified", "egress_mib", "eps", "foreign_bits", "memory_bits"}`,
`memory_bits = M`, the fill's `labels · 512` bits. When the bound covers the session (`READS` timed reads, a fill of at
least `LEAST_FILL_BYTES`, an egress count that isn't saturated) `foreign_bits` is `⌈f·M⌉ + E` at `eps = 2^-128`, with
`f = 1/19` and `E` the egress the network certifier counted over the session. Off it, `foreign_bits` is `M` and `eps`
0, the vacuous statement, and the profile names each gap. `EXPERIMENTAL` says what the bound rests on.

`leg(period_dir, node, out)` is the wipe's leg of `benchmarks/all_protocols`: one wipe of `SEGMENTS` 32 MiB segments on
CPU, the memory challenger on `period.json`'s `challenger_host` (`wipe_party.Party.of`; this node by default, a process
whose key and session log are beside the periods). No network certifier counts the session's egress yet, so the
challenger signs `UNCOUNTED` and an accepted leg's bound is vacuous: its verdict is real and its quantity isn't.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.ml.commitments.identity import canonical_json_bytes

from . import memory_challenger as mc
from . import streamed_wipe as sw
from . import wipe_party
from .period import serve
from .wipe_party import UNCOUNTED

PROTOCOL = "verity/wipe/v0"
"""`MemoryChallenger.WipeProtocol.ID`; a test signer's wipes answer under `memory_challenger.TEST_PREFIX` + it."""
READS = 2152
"""The timed spot reads the bound is at."""
F = Fraction(1, 19)
EPS = Fraction(1, 2 ** 128)
LEAST_FILL_BYTES = 1 << 25
"""The least fill the bound is stated for: one 32 MiB segment."""
SEGMENT_LABELS = LEAST_FILL_BYTES // sw.LABEL_BYTES
SEGMENTS = 1
"""The leg's fill, in 32 MiB segments."""
LIMIT_NS = 20_000_000
"""The leg's deadline a read: 20 ms, a socket round trip on one machine or a nearby one with room to spare."""
TIER = "host-dram"
EXPERIMENTAL = (
    "f = 1/19 at eps = 2^-128 with 2152 timed reads is Pous.Guarantees.StreamedWipeMeetsFamily, proved for the "
    "streamed-label fill this verdict judges, at every fill of at least one 32 MiB segment; eps leaves out its PRF "
    "advantage and its premises: K generated and held only by the challenger, off the prover's node; every tier "
    "declared; the egress certified by the network certifier; no side channel")
OTHER = "the wipe's commitment is not of the period's server and session"
UNSIGNED = "the memory challenger has not signed the wipe: its reads are incomplete or not yet verified"


def params_for(segments: int = SEGMENTS, *, k: int = READS, limit_ns: int = LIMIT_NS,
               tier: str = TIER) -> sw.WipeParams:
    """A wipe of `segments` 32 MiB segments in one tier, `k` reads one at a time under `limit_ns` each."""
    return sw.WipeParams(segments * SEGMENT_LABELS, k, 1, limit_ns, ((tier, segments * LEAST_FILL_BYTES),))


def memory_bits(p: sw.WipeParams) -> int:
    """`M`: the fill in bits, every declared tier."""
    return p.labels * sw.LABEL_BITS


def gaps(p: sw.WipeParams, egress_mib: int) -> list[str]:
    """`MemoryChallenger.WipeProtocol.gaps`: why the bound doesn't cover an accepted wipe (none when it does)."""
    out = []
    if p.k != READS:
        out.append(f"k = {p.k}: the bound is at {READS} timed reads")
    size = p.labels * sw.LABEL_BYTES
    if size < LEAST_FILL_BYTES:
        out.append(f"the fill is {size} bytes, below one 32 MiB segment ({LEAST_FILL_BYTES} bytes), the least the bound "
                   "is stated for")
    if not 0 <= egress_mib < sw.EGRESS_MAX_UNITS:
        out.append(f"the egress count is {egress_mib} MiB, outside 0 to {sw.EGRESS_MAX_UNITS - 1}: it bounds nothing")
    return out


def certify(p: sw.WipeParams, egress_mib: int) -> dict[str, Any]:
    """`MemoryChallenger.WipeProtocol.certify`'s public outputs, eps an exact ratio."""
    m = memory_bits(p)
    if gaps(p, egress_mib):
        return {"certified": False, "egress_mib": egress_mib, "eps": "0", "foreign_bits": m, "memory_bits": m}
    return {"certified": True, "egress_mib": egress_mib, "eps": str(EPS),
            "foreign_bits": -(-m * F.numerator // F.denominator) + egress_mib * sw.EGRESS_UNIT_BITS, "memory_bits": m}


def profile(p: sw.WipeParams, egress_mib: int) -> dict[str, Any]:
    """`MemoryChallenger.WipeProtocol.profileJson`: the wipe the session ran, the bound and every gap."""
    return {"batch": p.batch,
            "certificate": {"eps": str(EPS), "f": str(F), "least_fill_bytes": LEAST_FILL_BYTES, "reads": READS},
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
    public, prof, reason = (*rest, None) if verdict == "accept" else (None, None, rest[0])
    return {"commitment": digest(commitment).hex(), "profile": prof, "protocol": protocol, "public": public,
            "reason": reason, "verdict": verdict}


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
    """A period's wipe: the protocol's answer and case, and how the session ran (`ended`, `reads`, the challenger's
    `host`)."""

    answer: dict[str, Any]
    case: dict[str, Any]
    report: dict[str, Any]

    @property
    def verdict(self) -> str:
        return self.answer["verdict"]


def run(node: str, params: sw.WipeParams | None = None, *, signer: mc.Signer | None = None,
        log: mc.SessionLog | None = None, party: wipe_party.Party | None = None,
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
    trusted = (key,) if trusted is None else tuple(trusted)
    return Wipe(answer(server, commitment.session, trusted, commitment, signed, schemes, protocol),
                case(server, commitment.session, trusted, commitment, signed), report)


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The wipe's leg of `benchmarks/all_protocols` (`leg(period_dir, node, out) -> answer`): one wipe of `SEGMENTS`
    32 MiB segments, the memory challenger a party of its own on `period.json`'s `challenger_host`
    (`wipe_party.Party.of`; on this node by default, a process whose key and session log are in `wipe-challenger`
    beside the periods) and the prover here. With `wipe-protocol` among the period's tools, the Lean answers (the
    verifier of record) and must agree with the Python; it writes `out/wipe.case.json` (onsite's wipe case) and
    `out/wipe.json`. Which host ran each party, and whether `trusted-nebius-host`'s premise holds
    (`wipe_party.placement`), go in `wipe.json` and in an accepted answer's profile (`placement`)."""
    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools = doc.get("tools") or {}
    party = wipe_party.Party.of(doc, node, Path(period_dir).parent / "wipe-challenger")
    where = wipe_party.placement(node, party.host)
    t = time.monotonic()
    wipe = run(node, params_for(), party=party)
    seconds = round(time.monotonic() - t, 2)
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
    (out / "wipe.json").write_text(json.dumps({"answer": answer, "lean": lean, "placement": where,
                                               "report": wipe.report, "seconds": seconds}, sort_keys=True, indent=1))
    return answer


__all__ = ["EPS", "EXPERIMENTAL", "F", "LEAST_FILL_BYTES", "LIMIT_NS", "PROTOCOL", "READS", "SEGMENTS", "UNCOUNTED",
           "Keeping", "Wipe", "answer", "case", "certify", "digest", "gaps", "keeping", "leg", "memory_bits",
           "outcome", "params_for", "profile", "run", "serve"]
