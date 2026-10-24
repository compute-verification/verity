"""N2 on an advised link: a served link under arbitrary traffic, whose responses' lengths, batching and timing serving
chose, so that neither an occupancy profile nor the served profile's FIXLEN derivation (`served`) fixes its egress
counts. The prover's advice states them instead (`LinkAdvice`): I1's advice (`advice.Advice`, v0, one per certifier
session, charged by `advice.bits`) says in which egress bucket each kept output's frame is released, and N2's ingress
part says where each request ends on the ingress link. Each window's egress counts must be the advice's releases in its
buckets, and the advice is held to the certified ingress head under the declared syncs. A Boolean Definition, a gated
part of N2's advised proof unit (`service`'s `NetworkAdvisedLinkWindow` v1), which runs on every link-window; its one
output is the ok bit. That unit's Program holds it as one gate that names its circuit digest (`gate`), pinned in this
module's sidecar and held to the Definition by circuit-check, and N2's verifier of record runs its `reference`, which
the tests hold to the circuit.

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
- of N2's ingress part, which v0 leaves to N2's advice mode: `ingress_end[j]`, the request frames on the ingress link
  up to and including request j's last (v0 lists requests in order of their first row, not their arrival), and
  `received[w]`, the request frames the ingress certifier releases in window w.

**The window's view** (`view`), the unit's public inputs, computed by the verifier from the advice and the declared
syncs:
- `released[t]`, the frames the advice releases in the window's bucket t, at most R, the egress allocation;
- `floor[t]`, the request frames the window's ingress counts must have released by its bucket t: output λ of request j,
  released at b in window v, is eligible no earlier than j's anchor (the bucket that releases its last ingress frame)
  + δ + ⌊λ·q/p⌋ under window v's declared sync (δ, p/q), which is `schedule.eligible` with v's sync, and is released no
  earlier than it is eligible; so the link's ingress must have released `ingress_end[j]` request frames by bucket
  b − δ − ⌊λ·q/p⌋, and `floor` is the most of those that fall in the window's buckets, less
  B_w = Σ_{v < w} received[v];
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
`view(link, w).released`, a function of `release` alone: the ingress counts, the ingress part and the syncs only accept
or refuse, and change no count. Each window reports the advice it read (`digest`, of the advice's canonical encoding
and the ingress part: its `advice` output), and the network verdict refuses a window that names another than the first
accepted window's. So on a link whose every window is accepted, the egress counts of all its windows are a function of
one advice, and the egress timing channel of accepted runs carries at most the advised link's advice bits (`bits`, the
accepted-runs bound's A_c: `advice.bits` and the ingress part's), beside the log2 #Σ_sync bits a window the sync charge
already counts. Nothing else is left free: not the frames, not the ingress timing, not when the server actually wrote.
Of the ingress part, the unit holds `received` to the certified ingress head, so it is charged nothing, and
`ingress_end` is the prover's until IO-TIE ties each request's body to its ingress frames, so `bits` charges it:
bitlen(Σ received) bits a request (`ingress_bits`), to the earliest window whose floor reads it. And the advice is held
to the ingress head: no frame is released before its request's frames were released by the ingress certifier,
δ + ⌊λ/ρ⌋ earlier.

**Premises** of "an honest advised link is accepted", which N2's claim doesn't rest on: `release` is the certified
egress record's (each kept output one egress frame, in the link's canonical order, which IO-TIE fixes); the releases
fall in the link's windows, each with a declared sync; `ingress_end` and `received` are the ingress record's; each
window's declared sync has δ and ρ that every frame released in it meets (a small δ, a ρ at least a stream's rate);
with the salt on, the own frame counts as bucket 0's, so a window that starts with RIN or more request frames carried
in under-counts by one until the own frame is released, which only a release at its exact eligibility notices; and
both ends share T and τ_b.

**Cost** (`size`): about 105 ANDs a bucket, the ingress floor and the egress equality: 63,031 at T = 600, RIN = 1,
OWN = 0 and 64,894 at RIN = 4, OWN = 1, a fiftieth of one head check (3,229,371).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, add, and_reduce, sub
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier import advice as ADV
from verity.core.protocols.certifier.advice import Advice, MalformedAdvice, encode, outputs
from verity.core.protocols.certifier.head_check import (BIT, BYTE, _coll, _Consts, _part, _refs, ands, gate_of,
                                                        pinned_digest)
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.certifier.served import _le_const, _lt, _same, _u32

U32_MAX = (1 << 32) - 1
COUNT = Array(4, BYTE)
DIGEST = "verity-network-certifier/advised-advice/v1"


@dataclass(frozen=True)
class AdvisedProfile:
    """An advised link's public sizes, fixed before the run: the link's windows, which
    `service.LinkWindows.network_verdict` takes by default, and `own`, the ingress certifier's own frames at the head of
    every window's first bucket: 1 with its salt on (`active.Proxy(salt=…)`, a PoUW window of one certifier window),
    else 0."""

    windows: int
    own: int = 0

    def __post_init__(self) -> None:
        if not all(type(x) is int for x in (self.windows, self.own)) or self.windows < 1 or self.own not in (0, 1):
            raise ValueError(f"an advised profile has one or more windows and 0 or 1 own frames a window, not {self!r}")


@dataclass(frozen=True)
class LinkAdvice:
    """What N2 reads on an advised link: I1's advice and N2's ingress part (the module docstring), `ingress_end[j]`
    request j's end on the ingress link and `received[w]` window w's request frames. Construction refuses
    (`advice.MalformedAdvice`) an ingress part that isn't one: not one distinct, positive u32 end a request, or a last
    end that isn't the windows' total."""

    advice: Advice
    ingress_end: tuple[int, ...]
    received: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.advice) is not Advice:
            raise MalformedAdvice(f"an advised link's advice is an advice.Advice, not a {type(self.advice).__name__}")
        for name in ("ingress_end", "received"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or not all(type(x) is int and 0 <= x <= U32_MAX for x in value):
                raise MalformedAdvice(f"{name} is a sequence of u32s, not {value!r:.200}")
            object.__setattr__(self, name, tuple(value))
        ends = self.ingress_end
        if len(ends) != len(self.advice.requests) or 0 in ends or len(set(ends)) != len(ends):
            raise MalformedAdvice(f"the ingress part has one distinct positive end a request, not {ends!r:.200} for "
                                  f"{len(self.advice.requests)} requests")
        if max(ends, default=0) != sum(self.received):
            raise MalformedAdvice(f"the requests end at request frame {max(ends, default=0)} and the windows receive "
                                  f"{sum(self.received)}")


@dataclass(frozen=True)
class WindowView:
    """The unit's public inputs for one window (`view`): the frames the advice releases in each bucket, the request
    frames the ingress counts must have released by each bucket, and the request frames the window's ingress
    releases."""

    released: tuple[int, ...]
    floor: tuple[int, ...]
    received: int


# --- the reference ----------------------------------------------------------------------------------------------------

def _floor_buckets(link: LinkAdvice, T: int, syncs: Sequence[ClockSync]) -> list[tuple[int, int]]:
    """(request j, bucket) for each kept output: the bucket by which the link's ingress must have released
    `ingress_end[j]` request frames, b − δ − ⌊λ·q/p⌋ under the sync of the window that releases it at b."""
    out = []
    for o, b in zip(outputs(link.advice), link.advice.release):
        v = b // T
        if v >= len(syncs):
            raise ValueError(f"request {o.request}'s output {o.index} is released in window {v}, which declares no "
                             "sync")
        sync = syncs[v]
        a = b - sync.delta - o.index * sync.rho.denominator // sync.rho.numerator
        if a < 0:
            raise ValueError(f"request {o.request}'s output {o.index} is released in bucket {b}, before its request")
        out.append((o.request, a))
    return out


def view(link: LinkAdvice, window: int, T: int, R: int, syncs: Sequence[ClockSync]) -> WindowView:
    """The unit's public inputs for `window` (the module docstring) at T buckets a window and egress allocation R, from
    the advice and every window's declared sync. Raises `ValueError` on an advice that isn't a `LinkAdvice`, a window
    its `received` doesn't cover, more than R frames released in a bucket, and a frame released in a window with no
    declared sync or before its request could be."""
    if type(link) is not LinkAdvice:
        raise ValueError(f"an advised link's advice is an advised.LinkAdvice, not a {type(link).__name__}")
    if not all(type(x) is int and x >= 1 for x in (T, R)) or type(window) is not int or window < 0:
        raise ValueError("a view has T ≥ 1, R ≥ 1 and a window ≥ 0")
    if window >= len(link.received):
        raise ValueError(f"the advice receives in {len(link.received)} windows, not window {window}")
    per = Counter(link.advice.release)
    over = sorted(b for b, n in per.items() if n > R)
    if over:
        raise ValueError(f"the advice releases {per[over[0]]} frames in bucket {over[0]}, past the allocation {R}")
    lo, hi = window * T, (window + 1) * T
    floor_at: dict[int, int] = {}
    for j, a in _floor_buckets(link, T, syncs):
        if a < hi:
            floor_at[a] = max(floor_at.get(a, 0), link.ingress_end[j])
    base = sum(link.received[:window])
    g, floor = max((n for a, n in floor_at.items() if a < lo), default=0), []
    for t in range(lo, hi):
        g = max(g, floor_at.get(t, 0))
        floor.append(max(0, g - base))
    return WindowView(tuple(per.get(t, 0) for t in range(lo, hi)), tuple(floor), link.received[window])


def digest(link: LinkAdvice) -> str:
    """The SHA-512 of the advice's canonical encoding (`advice.encode`) and N2's ingress part: the `advice` output each
    window of an advised link reports, by which `service.LinkWindows.network_verdict` holds every window to one
    advice."""
    body = {"format": DIGEST, "advice": encode(link.advice).hex(), "ingress_end": list(link.ingress_end),
            "received": list(link.received)}
    return hashlib.sha512(canonical_json(body)).hexdigest()


def ingress_bits(link: LinkAdvice) -> int:
    """The bits of N2's ingress part the prover chooses: bitlen(Σ received) for each request's `ingress_end`, until
    IO-TIE's tie puts the ends in the certified ingress record; none for `received`, which the unit holds to the
    certified ingress head."""
    return len(link.ingress_end) * sum(link.received).bit_length()


def total_bits(link: LinkAdvice) -> int:
    """An advised link's advice in bits, its `A_c`: I1's (`advice.total_bits`) and the ingress part's
    (`ingress_bits`)."""
    return ADV.total_bits(link.advice) + ingress_bits(link)


def bits_by_window(link: LinkAdvice, T: int, syncs: Sequence[ClockSync]) -> dict[int, int]:
    """The bits charged to each window of T buckets that is charged any: `advice.bits_by_window`'s, and each request's
    `ingress_end`, bitlen(Σ received) bits, to the earliest window whose floor reads it, the one holding the earliest of
    its outputs' floor buckets (`view`), which may precede its first release's. The values sum to `total_bits`. Raises
    `ValueError` where `view` does, on a release in a window with no declared sync or before its request."""
    charge = dict(ADV.bits_by_window(link.advice, T))
    width, first = sum(link.received).bit_length(), {}
    for j, a in _floor_buckets(link, T, syncs):
        first[j] = min(first.get(j, a), a)
    for a in first.values():
        charge[a // T] = charge.get(a // T, 0) + width
    return dict(sorted(charge.items()))


def bits(link: LinkAdvice, window: int, T: int, syncs: Sequence[ClockSync]) -> int:
    """The advised link's advice bits charged to window `window` of T buckets (`bits_by_window`)."""
    return bits_by_window(link, T, syncs).get(window, 0)


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
    """`unit(…)` as one gate of N2's advised proof unit (`service.LinkWindows.unit`) that names the Definition and its
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


__all__ = ["AdvisedProfile", "LinkAdvice", "NetworkAdvisedCounts", "PINS", "WindowView", "arguments", "bits",
           "bits_by_window", "digest", "evaluate", "gate", "ingress_bits", "reference", "size", "total_bits", "unit",
           "view"]
