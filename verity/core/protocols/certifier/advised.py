"""N2 on an advised link: a served link under arbitrary traffic, whose responses' lengths, batching and timing serving
chose, so that neither an occupancy profile nor the served profile's FIXLEN derivation (`served`) fixes its egress
counts. The prover's advice states them instead (`LinkAdvice`): I1's advice (`advice.Advice`, v0, one per certifier
session, charged by `advice.bits`) says in which egress bucket each kept output's frame is released, and how long each
request's prompt is, which with the link's public framing (`AdvisedProfile.request_frames`) says where it ends on the
ingress link. Each window's egress counts must be the advice's releases in its buckets, and the advice is held to the
certified ingress head under the declared syncs. A Boolean Definition, a gated part of N2's advised proof block
(`service`'s `NetworkAdvisedLinkWindow` v2), which runs on every link-window; its one output is the ok bit. That unit's
Program holds it as one gate that names its circuit digest (`gate`), pinned in this module's sidecar and held to the
Definition by circuit-check, and N2's in-the-clear check runs its `reference`, which the tests hold to the circuit.

    NetworkAdvisedCounts_v1{T, RIN, OWN}
        (ingress: Array<T, Count>, egress: Array<T, Count>, released: Array<T, Count>, floor: Array<T, Count>,
         received: Count) -> Value<1>
        Count = Array<4, Byte>, a u32 big-endian; Byte = Array<8, Value<1>>

**Statics: the size class, all public.** T is both ends' buckets a window, RIN the ingress allocation and OWN the
ingress certifier's own frames a window (1 with its salt on, else 0), as `served`'s.

**What N2 reads** (`view`):
- of I1's advice, fields `advice.bits` charges: each request's `prompt_len` and `output_len` and each step's rows'
  `request`, `start` and `n`, through `advice.outputs`, which numbers the kept outputs (request j's λ-th, in step then
  row order); and `release[k]`, the session-global egress bucket in which output k's frame is released. It reads no
  `id`, `finish`, `kind`, `cache_hit`, `preempted` or `sources` beyond `Advice`'s own validity, which refuses a
  schedule vLLM couldn't have run;
- `received[w]`, the request frames the ingress certifier releases in window w, which the unit holds to the certified
  ingress head;
- the tie (`IoTieVerdict`): the `IoTieRequest` verdict on each of the link's ingress windows for this advice (its
  `digest`), that the link's j-th certified request is advice request j and its prompt v0's `prompt_len` ids, which
  the verifier of record takes only from its own tie source (`TieSource`, `service.LinkWindows.ties`: IoTieRequest
  proofs it verified itself), never beside the delivered advice, and holds to the link and its ingress windows'
  registrations, which the verdict names;
- the link's public framing, `request_frames[n − 1]`, the ingress frames a request of an n-token prompt takes (CERT-
  VARLEN's id requests, whose frames are a function of the prompt's length alone), fixed before the run;
- each window's declared sync, one of Σ_sync, which the certified egress head's parameters digest binds, and the sync
  the network certifier ran that window under (`clocks`), which every floor reads and the declared one must be: the
  verifier takes it only from its own clock source (`ClockSource`, `service.LinkWindows.clock_source`: `SignedClocks`,
  the certifier's salted clock certificates, each read through its opening), never from the prover's declaration.
  What the certifier signed is witness: the sync and its salt are read only against the salted digest the public
  certificate signs, never from a public answer.

**The ingress ends** (`ends`) are derived, not advised: advice request j is the link's j-th request, and its prompt
length is the certified prompt's, both of which the tie holds (`ends` refuses a link without IoTieRequest's accepted
verdict on every ingress window for the same advice digest, on the link and the ingress registrations the verifier
holds; FCFS admission meets the order, v0's first-row order being the arrival order). So request j ends at request
frame
Σ_{i ≤ j} request_frames[prompt_len_i − 1], and the last request's end must be Σ received, the link's certified
request frames. No field of the advice but v0's says where a request ends, so the ends cost nothing beside
`advice.bits`.

**The window's view** (`view`), the unit's public inputs, computed by the verifier from the advice, the framing and the
syncs the certifier ran:
- `released[t]`, the frames the advice releases in the window's bucket t, at most R, the egress allocation;
- `floor[t]`, the request frames the window's ingress counts must have released by its bucket t: output λ of request j,
  released at b in window v, is eligible no earlier than j's anchor (the bucket that releases its last ingress frame)
  + δ + ⌊λ·q/p⌋ under the sync (δ, p/q) the certifier ran window v under, which is `schedule.eligible` with v's sync,
  and released no earlier; so the link's ingress must have released `ends[j]` request frames by bucket
  b − δ − ⌊λ·q/p⌋, and `floor` is the most of those that fall in the window's buckets, less B_w = Σ_{v < w}
  received[v]. That bucket may fall in a window before v, so a floor that read v's declaration would carry v's word
  into an earlier window's view; it reads the certified sync instead. A window declaring a sync outside Σ_sync, or
  other than the one the certifier ran it under, is refused, and only that window, since no floor reads a
  declaration. δ ≥ 0, so no frame is let out before the bucket that releases its request's last frame. A release in
  that bucket is a served clock's (an engine's first output in its request's bucket); on a certified path the proxy
  hands a frame read in bucket t on at t + 1, so its honest syncs have δ ≥ 1, and a window declaring δ = 0 there is
  refused;
- `received`, `received[w]`.

**What 1 means.** Every ingress count is at most RIN and bucket 0's at least OWN; with bucket 0's OWN own frames taken
off, the ingress counts' prefix sums are at least `floor` in every bucket and sum to `received`; and the egress counts
are `released`.

**Carry-over is stated by the advice**, not forbidden. An output sampled in one window may be released in a later one,
and a response may span any windows: `release` says where each frame went out, so no queue is carried. The ingress
carry is `received`: each window holds its own total to it exactly, so by induction from window 0 (B_0 = 0) on a link
whose every window is accepted (`service.LinkWindows.network_verdict`'s AND), B_w is the request frames the ingress
certifier released before window w.

**Soundness.** With the advice fixed, the counts are determined. On an accepted window w, the egress counts are
`view(link, w).released`, a function of `release` alone: the ingress counts, `received` and the syncs only accept or
refuse, and change no count. Each window reports the advice it read (`digest`, of the advice's canonical encoding and
`received`: its `advice` output), and the network verdict refuses a window that names another than the first accepted
window's. So on a link whose every window is accepted, the egress counts of all its windows are a function of one
advice, and the egress timing channel of accepted runs carries at most `advice.bits` (the accepted-runs bound's A_c),
beside the log2 #Σ_sync bits a window the sync charge already counts. Nothing else is left free: not the frames, not
the ingress timing, not when the server actually wrote, and not where a request ends, which `ends` derives from v0's
prompt lengths. And the advice is held to the ingress head: no frame is released before the ingress certifier had
released its request's `ends[j]` request frames, δ + ⌊λ/ρ⌋ buckets earlier, under a sync Σ_sync holds. Those
are the link's first j + 1 requests whole because v0's prompt lengths are theirs, in arrival order: the tie,
IoTieRequest's verdict for the same advice digest. Without it, a prover who splits the same request frames among the
requests otherwise (a short prompt and a long one swapped) lowers an early request's end, and so the floor, on the
same certified record: the swapped split. The certified prompts are witness, so the verifier can't derive the tie
from them on the host, and a tie the prover states is no tie (one stating the swapped split reopens its acausal
release), nor is a verdict it delivers (a verdict built from its own advice's digest, every window accepted, would
reopen it through the verifier of record), so `LinkAdvice` carries none. The
tie is only IoTieRequest's verdict from the verifier's own tie source (`tie_service.Ties`), on the link and the
ingress registrations it holds, so a verdict on another link or ingress record isn't reused, built from the
IoTieRequest proofs the verifier of record verified on its own lanes (`tie_service.IngressTie.tie`). No proof of
an unpinned size class is accepted, and the served size isn't pinned, so there every advised link is refused
(PROTOCOL.md, fail closed). The floor's syncs are held the same way: a δ below the one the window's traffic was
certified under loosens the floor (δ = 0 declared for a window the certifier ran at δ = 1 accepts an output in its
request's own bucket), so every floor is under the sync the certifier ran, which the verifier takes
only from its own clock source (`ClockSource`, `service.LinkWindows.clock_source`), never from the prover:
`SignedClocks`, the sync the certifier signed for each egress window beside its record, salted as a record digest is
(`certificate.CertifiedClock`), read through its opening against the signed digest under the verifier's own certifier
key and session; without one every advised link is refused. `view` refuses a window whose declared sync isn't that
one. The opening is witness, read by the verifier of record on the host until the unit opens it (PROTOCOL.md's named
gap, "The clock's opening"): on an accepted window the sync it shows is the declared one, a public output already,
and on a refused one it shows the sync the certifier ran, which the unit's opening would keep hidden. A window's
declaration moves no floor, so its refusal
is that window's alone, and the link's verdict, the AND over its windows, refuses the link with it. Neither head signs
the prompt lengths, so the unit itself doesn't hold them to the record, only the verifier's tie source does:
PROTOCOL.md's named gap, with its design.

**Bucket 0** is capped at RIN, which holds the own frame: the ingress certifier releases min(queue, r_in) frames a
bucket (`grid.Certifier`), its own frame queued at the head of the queue, ahead of bucket 0's arrivals and of any
request frames carried from the window before (`active.Bucketer`, `Certifier.enqueue(first=True)`), and a head whose
count passes its allocation doesn't exist (`certificate.RecordHead`). With RIN request frames arriving in bucket 0,
the certified count is RIN, one of them the own frame, and the last request frame is released in bucket 1; with RIN
or more carried in, the own frame still leaves in bucket 0.

**The ends are where the signed counts put them**: request j ends in the first bucket where the
ingress head's signed counts, less one own frame a window, reach `ends[j]`, the prefix sums the floor reads. That is
the bucket that releases its last frame, since each window's own frame leaves in its first bucket.

**Premises** of "an honest advised link is accepted", which N2's claim doesn't rest on: `release` is the certified
egress record's (each kept output one egress frame, in the link's canonical order, which the tie fixes); the releases
fall in the link's windows, each with a declared sync in Σ_sync; `received` is the ingress record's; the link's j-th
request is advice request j (FCFS admission, which IoTieRequest checks, its ordinals being the verifier's: v0
numbered by a runner's batch order, which the derived encoding's `walk.order` maps back, is refused), an id request of
its prompt length (`request_frames`); each window's declared sync is the one the certifier ran it under, with a δ and
ρ that every frame released in it meets (a small δ, a ρ at least a stream's rate); and both ends share T and τ_b.

**Cost** (`size`): about 105 ANDs a bucket, the ingress floor and the egress equality: 63,031 at T = 600, RIN = 1,
OWN = 0 and 64,894 at RIN = 4, OWN = 1, a fiftieth of one head check (3,229,371).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, add, and_reduce, sub
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier.advice import Advice, MalformedAdvice, encode, outputs
from verity.core.protocols.certifier.certificate import Certificate, CertifiedClock, PublicKey
from verity.core.protocols.certifier.head_check import (BIT, BYTE, _coll, _Consts, _part, _refs, ands, gate_of,
                                                        pinned_digest)
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.served import _le_const, _lt, _same, _u32

U32_MAX = (1 << 32) - 1
COUNT = Array(4, BYTE)
DIGEST = "verity-network-certifier/advised-advice/v2"


@dataclass(frozen=True)
class AdvisedProfile:
    """An advised link's public sizes, fixed before the run: the link's windows, which
    `service.LinkWindows.network_verdict` takes by default; `request_frames`, the link's framing, whose entry n − 1 is
    the ingress frames a request of an n-token prompt takes, for every prompt length the link admits (the module
    docstring, "The ingress ends"); and `own`, the ingress certifier's own frames at the head of every window's first
    bucket: 1 with its salt on (`active.Proxy(salt=…)`, a PoUW window of one certifier window), else 0."""

    windows: int
    request_frames: tuple[int, ...]
    own: int = 0

    def __post_init__(self) -> None:
        if not all(type(x) is int for x in (self.windows, self.own)) or self.windows < 1 or self.own not in (0, 1):
            raise ValueError(f"an advised profile has one or more windows and 0 or 1 own frames a window, not {self!r}")
        frames = self.request_frames
        if not isinstance(frames, (tuple, list)) or not frames or \
                not all(type(n) is int and 1 <= n <= U32_MAX for n in frames):
            raise ValueError(f"an advised profile's framing is one positive u32 frame count a prompt length, from 1 "
                             f"token on, not {frames!r:.200}")
        object.__setattr__(self, "request_frames", tuple(frames))


def _u32s(value, what: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not all(type(x) is int and 0 <= x <= U32_MAX for x in value):
        raise MalformedAdvice(f"{what} is a sequence of u32s, not {value!r:.200}")
    return tuple(value)


def _sha512_hex(x: Any) -> bool:
    return type(x) is str and len(x) == 128 and not set(x) - set("0123456789abcdef")


@dataclass(frozen=True)
class IoTieVerdict:
    """The tie of an advised link's requests to the certified ingress (the module docstring, "The ingress ends"):
    the `IoTieRequest` verdict on each of the link's ingress windows, `accepted[w]`, on the link named `link`
    (`service.LinkWindows.link`), whose ingress windows the challenger registered as `registrations` (each one's
    `registration.digest`, in window order), on the request view the verifier computed from the advice whose `digest`
    is `advice`. An accepted verdict on every window says the link's j-th certified request is advice request j and
    its prompt is v0's `prompt_len` ids long. Only the verifier's own tie source (`TieSource`, `tie_service.Ties`)
    gives one, from IoTieRequest proofs it verified itself (`tie_service.IngressTie.tie`), and `view` holds it to the
    link and the registrations the verifier expects, so a verdict on another link or another ingress record isn't
    this one's. At a size class with no pinned IoTieRequest digest every window is refused (PROTOCOL.md, fail
    closed). Construction refuses (`advice.MalformedAdvice`) a `link` that isn't a
    non-empty string, `registrations` and an `advice` that aren't SHA-512 hex digests, and verdicts that aren't one
    bool a registration."""

    link: str
    registrations: tuple[str, ...]
    advice: str
    accepted: tuple[bool, ...]

    def __post_init__(self) -> None:
        if type(self.link) is not str or not self.link:
            raise MalformedAdvice("IoTieRequest's verdict names its link, a non-empty string")
        regs, acc = self.registrations, self.accepted
        if isinstance(regs, (str, bytes)) or not isinstance(regs, (tuple, list)) or \
                not all(_sha512_hex(r) for r in regs):
            raise MalformedAdvice("IoTieRequest's verdict names each ingress window's registration by its "
                                  "`registration.digest`, 128 hex digits")
        if not _sha512_hex(self.advice):
            raise MalformedAdvice("IoTieRequest's verdict names its advice by `advised.digest`, 128 hex digits")
        if isinstance(acc, (str, bytes)) or not isinstance(acc, (tuple, list)) or \
                not all(type(x) is bool for x in acc) or len(acc) != len(regs):
            raise MalformedAdvice("IoTieRequest's verdict is one bool an ingress window's registration")
        object.__setattr__(self, "registrations", tuple(regs))
        object.__setattr__(self, "accepted", tuple(acc))


class TieSource(Protocol):
    """The verifier's own source of the tie (`service.LinkWindows.ties`), never what the prover delivers:
    `registrations(link)` is the verifier's record of the link's ingress windows' registrations with the challenger
    (each one's `registration.digest`, in window order), and `verdict(link, advice, profile)` IoTieRequest's verdict
    (`IoTieVerdict`) for `advice` under `profile`, built only from IoTieRequest proofs the verifier verified itself:
    `tie_service.Ties`. Without one every advised link is refused (PROTOCOL.md, fail closed)."""

    def registrations(self, link: str) -> Sequence[str]: ...

    def verdict(self, link: str, advice: LinkAdvice, profile: AdvisedProfile) -> IoTieVerdict: ...


class ClockSource(Protocol):
    """The verifier's own record of the sync each of a link's windows was certified under
    (`service.LinkWindows.clock_source`), never the prover's declaration: `syncs(link)`, window by window, the sync the
    network certifier, the auditor's device, ran it under (its shaper's, `active.Shaper.syncs`). `view` refuses a
    window that declares another. The verifier of record's is `SignedClocks`, the certifier's signed syncs."""

    def syncs(self, link: str) -> Sequence[ClockSync]: ...


def _named(c: CertifiedClock) -> str:
    x = c.certificate
    if type(x) is not Certificate:
        return f"a clock opening of a {type(x).__name__}"
    return f"{x.link} window {x.window}'s {x.direction.value} clock (key {x.key_id.hex()[:16]}…)"


@dataclass(frozen=True)
class SignedClocks:
    """The verifier of record's clock source (`ClockSource`): the syncs the network certifier signed, one salted clock
    certificate an egress window (`certificate.CertifiedClock`, `certificate.Issuer.salted_clock`), each sync read
    through its opening (the sync and `salt_c`, witness) against the salted digest its certificate signs, under the
    verifier's own certifier `key` and `session`. The openings are the verifier's own read of the certifier's output
    (`service.LinkWindows.certified_clocks`), never a public answer, which carries the certificates alone.
    `syncs(link)` is window 0's on, each the sync of the openings that name `link` and the window and hold
    (`CertifiedClock.refusal`), up to the first window without exactly one such sync. `refused` says why each
    opening it took no sync from was refused: not a salted clock's opening (a record's `CertifiedRecord` among them),
    not an egress certificate, another key's, a sync and salt that don't open the digest signed (an altered
    digest or sync), a signature that doesn't verify (forged, or another session's, or relabelled to another link or
    window), a salt another link-window's opening has too, or a window with two signed syncs. Whoever delivers the
    openings, the prover included, can't forge one."""

    key: PublicKey
    session: bytes
    openings: Sequence[CertifiedClock]
    refused: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        held: list[CertifiedClock] = []
        refused: list[str] = []
        for c in self.openings:
            if type(c) is not CertifiedClock:
                refused.append(f"a {type(c).__name__}: not a salted clock certificate's opening "
                               "(certificate.CertifiedClock), so it gives no sync")
            elif (why := c.refusal(self.key, self.session)) is not None:
                refused.append(f"{_named(c)}: {why}")
            else:
                held.append(c)
        salts: dict[bytes, list[CertifiedClock]] = {}
        for c in held:
            salts.setdefault(c.salt, []).append(c)
        signed: dict[str, dict[int, set[ClockSync]]] = {}
        for c in held:
            if (other := next((o for o in salts[c.salt] if o != c), None)) is not None:
                refused.append(f"{_named(c)}: its salt is also {other.link} window {other.window}'s, under which two "
                               "windows of one sync sign one digest")
            else:
                signed.setdefault(c.link, {}).setdefault(c.window, set()).add(c.sync)
        refused += [f"{link} window {w}: {len(got)} signed syncs, so it has none"
                    for link, windows in signed.items() for w, got in sorted(windows.items()) if len(got) > 1]
        object.__setattr__(self, "_signed", signed)
        object.__setattr__(self, "refused", tuple(refused))

    def syncs(self, link: str) -> tuple[ClockSync, ...]:
        windows, out = self._signed.get(link, {}), []
        while len(got := windows.get(len(out), ())) == 1:
            out.extend(got)
        return tuple(out)


@dataclass(frozen=True)
class LinkAdvice:
    """What the prover delivers on an advised link beside its public profile and syncs: I1's advice and `received[w]`,
    window w's request frames, which the unit holds to the certified ingress head. It carries no tie: one delivered
    beside the advice is the prover's (a verdict built from its own advice's digest reopens the swapped split), so
    `ends` and `view` take IoTieRequest's verdict apart, from the verifier's own tie source (`TieSource`).
    Construction refuses (`advice.MalformedAdvice`) an advice that isn't an `advice.Advice` and a `received` that
    isn't u32s."""

    advice: Advice
    received: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.advice) is not Advice:
            raise MalformedAdvice(f"an advised link's advice is an advice.Advice, not a {type(self.advice).__name__}")
        object.__setattr__(self, "received", _u32s(self.received, "received"))


@dataclass(frozen=True)
class WindowView:
    """The unit's public inputs for one window (`view`): the frames the advice releases in each bucket, the request
    frames the ingress counts must have released by each bucket, and the request frames the window's ingress
    releases."""

    released: tuple[int, ...]
    floor: tuple[int, ...]
    received: int


# --- the reference ----------------------------------------------------------------------------------------------------

def _held(link: LinkAdvice, tie: Any, link_name: str, registrations: Sequence[str]) -> None:
    """Raises `ValueError` unless `tie` is IoTieRequest's verdict (`IoTieVerdict`) on the link `link_name`, on the
    ingress registrations `registrations` the verifier holds for it, for `link`'s advice, accepting every ingress
    window."""
    if type(tie) is not IoTieVerdict:
        raise ValueError("an advised link without IoTieRequest's verdict (the tie, advised.IoTieVerdict) from the "
                         "verifier's own tie source is refused")
    if tie.link != link_name:
        raise ValueError(f"IoTieRequest's verdict is on another link ({tie.link!r:.80}) than {link_name!r:.80}")
    if tie.registrations != tuple(registrations):
        raise ValueError("IoTieRequest's verdict is on other ingress registrations than the verifier's record of the "
                         "link's")
    if tie.advice != digest(link):
        raise ValueError("IoTieRequest's verdict is on another advice than the link's (advised.digest)")
    if len(tie.accepted) != len(link.received) or not all(tie.accepted):
        raise ValueError(f"IoTieRequest's verdict doesn't accept each of the link's {len(link.received)} ingress "
                         f"windows: {list(tie.accepted)!r:.200}")


def ends(link: LinkAdvice, request_frames: Sequence[int], *, tie: IoTieVerdict | None, link_name: str,
         registrations: Sequence[str]) -> tuple[int, ...]:
    """Each request's end on the ingress link, the request frames up to and including its last (the module docstring,
    "The ingress ends"): request j, the link's j-th, ends at Σ_{i ≤ j} request_frames[prompt_len_i − 1], under `tie`,
    IoTieRequest's verdict from the verifier's own tie source (`TieSource`), held to the link `link_name` and the
    ingress registrations `registrations` the verifier holds for it. Raises `ValueError` without a verdict, on one on
    another link, other registrations or another advice, or that doesn't accept every ingress window, on a prompt
    longer than the framing covers, and on ends whose last isn't the windows' total."""
    _held(link, tie, link_name, registrations)
    out, end = [], 0
    for j, q in enumerate(link.advice.requests):
        if q.prompt_len > len(request_frames):
            raise ValueError(f"request {j}'s prompt of {q.prompt_len} tokens is past the link's framing, of at most "
                             f"{len(request_frames)}")
        end += request_frames[q.prompt_len - 1]
        out.append(end)
    if end != sum(link.received):
        raise ValueError(f"the requests take {end} request frames and the windows receive {sum(link.received)}")
    return tuple(out)


def _floor_buckets(link: LinkAdvice, T: int, clocks: Sequence[ClockSync]) -> list[tuple[int, int]]:
    """(request j, bucket) for each kept output: the bucket by which the link's ingress must have released `ends[j]`
    request frames, b − δ − ⌊λ·q/p⌋ under the sync the certifier ran the window that releases it at b under
    (`clocks`), whatever that window declares."""
    out = []
    for o, b in zip(outputs(link.advice), link.advice.release):
        v = b // T
        if v >= len(link.received):
            raise ValueError(f"request {o.request}'s output {o.index} is released in window {v}, past the link's "
                             f"{len(link.received)} windows")
        if v >= len(clocks) or type(clocks[v]) is not ClockSync:
            raise ValueError(f"request {o.request}'s output {o.index} is released in window {v}, and the certifier ran "
                             "it under no sync")
        sync = clocks[v]
        a = b - sync.delta - o.index * sync.rho.denominator // sync.rho.numerator
        if a < 0:
            raise ValueError(f"request {o.request}'s output {o.index} is released in bucket {b}, before its request")
        out.append((o.request, a))
    return out


def _declared(window: int, syncs: Sequence[ClockSync], sync_set: SyncSet, clocks: Sequence[ClockSync]) -> None:
    """Raises `ValueError` unless `window` declares a sync, one of Σ_sync and the one the certifier ran it under
    (`clocks`)."""
    if window >= len(syncs):
        raise ValueError(f"window {window} declares no sync")
    sync = syncs[window]
    if sync not in sync_set:
        raise ValueError(f"window {window} declares {sync!r}, which isn't one of Σ_sync")
    if window >= len(clocks) or clocks[window] != sync:
        ran = "no sync" if window >= len(clocks) else repr(clocks[window])
        raise ValueError(f"window {window} declares {sync!r}, and the certifier ran it under {ran}")


def view(link: LinkAdvice, window: int, T: int, R: int, syncs: Sequence[ClockSync], sync_set: SyncSet,
         request_frames: Sequence[int], *, tie: IoTieVerdict | None, link_name: str,
         registrations: Sequence[str], clocks: Sequence[ClockSync] | None) -> WindowView:
    """The unit's public inputs for `window` (the module docstring) at T buckets a window and egress allocation R, from
    the advice, every window's declared sync, Σ_sync and the link's framing (`AdvisedProfile.request_frames`), under
    `tie`, IoTieRequest's verdict from the verifier's own tie source, which `ends` holds to the link `link_name` and
    its ingress `registrations`, and `clocks`, the syncs the certifier ran the link's windows under, from the
    verifier's own clock source (`ClockSource`). Every floor is under the syncs the certifier ran (`_floor_buckets`),
    so one window's declaration moves no other window's view. Raises `ValueError` on an advice that isn't a
    `LinkAdvice`, a window its `received` doesn't cover, ends `ends` refuses (without the tie, or under one on another
    link, registrations or advice, or not accepting every window), more than R frames released in a bucket, without
    `clocks`, a frame released past the link's windows, in one the clock source has no sync for, or before its request
    could be under the sync the certifier ran; and, in `window` alone, a declared sync that's missing, outside Σ_sync
    or other than the certifier ran it under."""
    if type(link) is not LinkAdvice:
        raise ValueError(f"an advised link's advice is an advised.LinkAdvice, not a {type(link).__name__}")
    if type(sync_set) is not SyncSet:
        raise ValueError(f"the declared syncs are held to Σ_sync, a schedule.SyncSet, not a {type(sync_set).__name__}")
    if clocks is None or isinstance(clocks, (str, bytes)) or not isinstance(clocks, Sequence):
        raise ValueError("an advised link without the syncs the certifier ran its windows under, from the verifier's "
                         "own clock source (advised.ClockSource), is refused")
    if not all(type(x) is int and x >= 1 for x in (T, R)) or type(window) is not int or window < 0:
        raise ValueError("a view has T ≥ 1, R ≥ 1 and a window ≥ 0")
    if window >= len(link.received):
        raise ValueError(f"the advice receives in {len(link.received)} windows, not window {window}")
    end = ends(link, request_frames, tie=tie, link_name=link_name, registrations=registrations)
    per = Counter(link.advice.release)
    over = sorted(b for b, n in per.items() if n > R)
    if over:
        raise ValueError(f"the advice releases {per[over[0]]} frames in bucket {over[0]}, past the allocation {R}")
    lo, hi = window * T, (window + 1) * T
    floor_at: dict[int, int] = {}
    for j, a in _floor_buckets(link, T, clocks):
        if a < hi:
            floor_at[a] = max(floor_at.get(a, 0), end[j])
    _declared(window, syncs, sync_set, clocks)
    base = sum(link.received[:window])
    g, floor = max((n for a, n in floor_at.items() if a < lo), default=0), []
    for t in range(lo, hi):
        g = max(g, floor_at.get(t, 0))
        floor.append(max(0, g - base))
    return WindowView(tuple(per.get(t, 0) for t in range(lo, hi)), tuple(floor), link.received[window])


def digest(link: LinkAdvice) -> str:
    """The SHA-512 of the advice's canonical encoding (`advice.encode`) and `received`: the `advice` output each window
    of an advised link reports, by which `service.LinkWindows.network_verdict` holds every window to one advice."""
    body = {"format": DIGEST, "advice": encode(link.advice).hex(), "received": list(link.received)}
    return hashlib.sha512(canonical_json(body)).hexdigest()


def reference(T: int, RIN: int, OWN: int, ingress: Sequence[int], egress: Sequence[int], released: Sequence[int],
              floor: Sequence[int], received: int) -> bool:
    """`NetworkAdvisedCounts_v1`'s output: every ingress count at most RIN and bucket 0's at least OWN, the ingress
    counts less the OWN own frames at least `floor` in every prefix and summing to `received`, and the egress counts
    `released`."""
    if not all(len(x) == T for x in (ingress, egress, released, floor)) or any(c > RIN for c in ingress) or \
            ingress[0] < OWN:
        return False
    cum = -OWN
    for t in range(T):
        cum += ingress[t]
        if cum < floor[t] or egress[t] != released[t]:
            return False
    return cum == received


# --- the circuit ------------------------------------------------------------------------------------------------------

STATICS = ("T", "RIN", "OWN")
PART_STATICS = STATICS[:-1]


def _check_statics(S) -> None:
    if not all(type(x) is int and x >= 1 for x in (S.T, S.RIN)) or S.OWN not in (0, 1) or S.T * S.RIN > U32_MAX:
        raise ValueError(f"T and RIN are ≥ 1 with T·RIN a u32, and OWN 0 or 1, not {vars(S)}")


def _widths(S) -> SimpleNamespace:
    """The ingress count's bits (RIN) and the ingress prefix's (T·RIN)."""
    return SimpleNamespace(IN=S.RIN.bit_length(), C=(S.T * S.RIN).bit_length())


def _sig_own(S):
    return (("count", COUNT),), Tuple(COUNT, BIT)


def _build_own(C, S, *count):
    """Bucket 0's ingress count less the certifier's own frame, and whether the count is 1 … RIN."""
    W = _widths(S)
    c = _u32(count)
    rest, borrow = sub(C, c, [ONE], 32)
    ok = and_reduce(C, [borrow.inv()] + [b.inv() for b in c[W.IN:]] + [_le_const(C, c[:W.IN], S.RIN)])
    return [[rest[8 * (3 - k) + b] for b in range(8)] for k in range(4)] + [[ok]]


def _sig_bucket(S):
    cum = Array(_widths(S).C, BIT)
    return (("cum", cum), ("ingress", COUNT), ("egress", COUNT), ("released", COUNT), ("floor", COUNT)), Tuple(cum, BIT)


def _build_bucket(C, S, cum, *e):
    """One bucket: the ingress prefix sum, which must be at least the floor, its count at most RIN, and the egress
    count, which must be the advice's release."""
    W = _widths(S)
    c, g, r, f = (_u32(e[4 * k:4 * k + 4]) for k in range(4))
    cum = add(C, cum, c[:W.IN], width=W.C)
    ok = and_reduce(C, [b.inv() for b in c[W.IN:]] + [_le_const(C, c[:W.IN], S.RIN)] + [b.inv() for b in f[W.C:]]
                    + [_lt(C, cum, f[:W.C], W.C).inv()] + _same(C, g, r))
    return [cum, [ok]]


def _sig_end(S):
    return (("cum", Array(_widths(S).C, BIT)), ("received", COUNT)), BIT


def _build_end(C, S, cum, *received):
    """The window's request frames are `received`."""
    return [and_reduce(C, _same(C, cum, _u32(received)))]


NetworkAdvisedOwn = _part("NetworkAdvisedOwn", PART_STATICS, _sig_own, _build_own,
                          "bucket 0's ingress count less the certifier's own frame, and whether it is 1 … RIN")
NetworkAdvisedBucket = _part("NetworkAdvisedBucket", PART_STATICS, _sig_bucket, _build_bucket,
                             "one bucket: the ingress prefix sum is at least the floor, and the egress count is the "
                             "advice's release")
NetworkAdvisedEnd = _part("NetworkAdvisedEnd", PART_STATICS, _sig_end, _build_end,
                          "the window's ingress releases the advice's request frames")


def _sig(S):
    _check_statics(S)
    counts = Array(S.T, COUNT)
    return (("ingress", counts), ("egress", counts), ("released", counts), ("floor", counts), ("received", COUNT)), BIT


def _body(B, S, ingress, egress, released, floor, received):
    _check_statics(S)
    W, K = _widths(S), _Consts(B)
    part = {d.name: bind(d, **{k: getattr(S, k) for k in PART_STATICS})
            for d in (NetworkAdvisedOwn, NetworkAdvisedBucket, NetworkAdvisedEnd)}
    ingress, egress, released, floor = (_refs(x) for x in (ingress, egress, released, floor))
    own = []
    if S.OWN:
        first = _refs(B.call(part["NetworkAdvisedOwn"], _coll(COUNT, ingress[:32])))
        ingress, own = first[:32] + ingress[32:], first[32:]
    out = _refs(B.scan(part["NetworkAdvisedBucket"], _coll(Array(W.C, BIT), K.bits(0, W.C)),
                       [_coll(Array(S.T, COUNT), x) for x in (ingress, egress, released, floor)]))
    end = B.call(part["NetworkAdvisedEnd"], _coll(Array(W.C, BIT), out[:W.C]), received)
    bits = own + out[W.C:] + _refs(end)
    return B.call(BL.and_gate(len(bits)), _coll(Array(len(bits), BIT), bits))


NetworkAdvisedCounts = CompositeDefinition(
    "NetworkAdvisedCounts", 1, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="1 iff the ingress counts are at most RIN, bucket 0's at least OWN, and with its OWN own frames taken off "
        "their prefix sums are at least `floor` and sum to `received`, and the egress counts are `released` "
        "(`advised.reference`)")
NetworkAdvisedCounts.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins (`gate`)
#: the circuit digests circuit-check pins (`head_check.pinned_digest`)
PINS = Path(__file__).with_name("advised.circuit-check.json")


def unit(T: int, RIN: int, OWN: int):
    return bind(NetworkAdvisedCounts, T=T, RIN=RIN, OWN=OWN)


@lru_cache(maxsize=None)
def gate(T: int, RIN: int, OWN: int) -> PrimitiveDefinition:
    """`unit(…)` as one gate of N2's advised proof block (`service.LinkWindows.unit`) that names the Definition and its
    pinned circuit digest, `Gate[NetworkAdvisedCounts_v1{…}@<digest>]_v2`, as `head_check.gate` holds the head check."""
    d = unit(T, RIN, OWN)
    return gate_of(d, pinned_digest(d, PINS), "N2's advised counts")


# --- the arguments ----------------------------------------------------------------------------------------------------

def _count_bits(counts: Sequence[int]) -> list[int]:
    data = b"".join(n.to_bytes(4, "big") for n in counts)
    return [(data[k] >> b) & 1 for k in range(len(data)) for b in range(8)]


def arguments(ingress: Sequence[int], egress: Sequence[int], v: WindowView) -> list[list[int]]:
    """The unit's arguments in its signature's order, each its leaves' bits, 0 or 1."""
    return [_count_bits(ingress), _count_bits(egress), _count_bits(v.released), _count_bits(v.floor),
            _count_bits([v.received])]


def evaluate(key: tuple[int, int, int], cases: Sequence[tuple[Sequence[int], Sequence[int], WindowView]]) -> list[int]:
    """The unit's output on each case, (ingress, egress, view) of one size class `key` = (T, RIN, OWN), in one
    evaluation."""
    args = [list(rows) for rows in zip(*(arguments(*c) for c in cases))]
    return [int(b) for b in evaluate_bits(unit(*key), *args).reshape(-1).tolist()]


def size(T: int, RIN: int, OWN: int) -> int:
    """The unit's ANDs."""
    return ands(unit(T, RIN, OWN))


__all__ = ["AdvisedProfile", "ClockSource", "IoTieVerdict", "LinkAdvice", "NetworkAdvisedCounts", "PINS",
           "SignedClocks", "TieSource", "WindowView", "arguments", "digest", "ends", "evaluate", "gate", "reference",
           "size", "unit", "view"]
