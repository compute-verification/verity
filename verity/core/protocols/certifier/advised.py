"""N2 on an advised link: a served link under arbitrary traffic, whose responses' lengths, batching and timing serving
chose, so that neither an occupancy profile nor the served profile's FIXLEN derivation (`served`) fixes its egress
counts. The prover's advice (the certifier's I1, one per certifier session, public and charged by `advice.bits`) states
them instead: each window's egress counts are the egress certifier's release of the advice's frames, handed in the
advice's release buckets, and the advice is held to the certified ingress head under the declared syncs. A Boolean
Definition, a gated part of N2's advised proof unit (`service`'s `NetworkAdvisedLinkWindow` v1), which runs on every
link-window; its one output is the ok bit. That unit's Program holds it as one gate that names its circuit digest
(`gate`), pinned in this module's sidecar and held to the Definition by circuit-check, and N2's verifier of record runs
its `reference`, which the tests hold to the circuit.

    NetworkAdvisedCounts_v1{T, R, RIN, OWN}
        (ingress: Array<T, Count>, egress: Array<T, Count>, handed: Array<T, Count>, floor: Array<T, Count>,
         queued: Count, received: Count) -> Value<1>
        Count = Array<4, Byte>, a u32 big-endian; Byte = Array<8, Value<1>>

**Statics: the size class, all public.** T is both ends' buckets a window, R the egress allocation, RIN the ingress
allocation and OWN the ingress certifier's own frames a window (1 with its salt on, else 0), as `served`'s.

**What N2 reads of the advice** (`frames`, `view`), the fields `advice.bits` charges for it:
- `requests[j].prompt_len` and `requests[j].output_len` (the tokens released), and `requests[j].ingress_frames`, the
  request's frames on the ingress link (`active.request_frame_count` of its body), the j-th request listed being the
  link's j-th (requests are indexed by their ingress ordinal);
- each step's rows' `request`, `start` and `n` (`steps[s].rows`);
- `release[s]`, the egress bucket, on the certifier session's clock from window 0's first bucket, in which step s's
  output frames are handed to the egress certifier;
- `received[w]`, the request frames the ingress certifier released in window w: the ingress carry between windows.

It reads no `id`, `finish`, `kind`, `cache_hit`, preemption set, or trailing or recompute mark. `ingress_frames` and
`received` are N2's additions to I1's v0 fields, which carry no request's size on the link and no ingress carry.

**The frames** (`frames`). A row samples one token, the request's next, when it reaches the request's last known
token: `start + n` is the prompt's length plus the tokens sampled before it (a decode row, a prefill's last chunk, and
a preempted request's recomputation's last chunk). A request's first `output_len` samples are released, each one
egress frame (one canonical event, none for `[DONE]`), its logical index λ its place in its response; its samples past
`output_len` are trailing rows' (async scheduling's, a stop string's or an abort's), computed but unreleased, and hand
no frame, so N2 reads no trailing mark. The link's canonical order is the steps' order, each step's frames in its rows'
order (Model Runner V2's); N2's counts read only which step hands each frame, so they don't depend on the order within
a step, which IO-TIE's frame-to-(request, position) map shares. Each request samples at least `output_len` tokens, and
no row reaches past its last known token, or the advice is malformed and every window refuses.

**The window's view** (`view`), the unit's public inputs, computed by the verifier from the advice and the declared
syncs:
- `handed[t]`, the frames whose step's release bucket is the window's bucket t; `release` is non-decreasing, as the
  egress certifier is handed frames in canonical order;
- `queued`, the egress certifier's queue at the window's start: the frames handed before it less those the certifier
  releases before it (`grid.released` at R from window 0, which starts empty);
- `floor[t]`, the request frames the window's ingress counts must have released by its bucket t: frame k of request j,
  handed at h in window v, must be handed no earlier than j's anchor (the bucket that releases its last ingress frame)
  + δ + ⌊λ·q/p⌋ under window v's declared sync (δ, p/q), which is `schedule.eligible` with v's sync; so the link's
  ingress must have released the first N_j = Σ_{i ≤ j} ingress_frames_i request frames by bucket h − δ − ⌊λ·q/p⌋, and
  `floor` is the most of those that fall in the window's buckets, less B_w = Σ_{v < w} received[v];
- `received`, `received[w]`.

**What 1 means.** Every ingress count is at most RIN and bucket 0's at least OWN; with bucket 0's OWN own frames taken
off, the ingress counts' prefix sums are at least `floor` in every bucket and sum to `received`; and the egress counts
are the release, FIFO at R, of `handed` from a queue of `queued` (`derive`).

**Carry-over is stated by the advice**, not forbidden. The egress queue a window starts with is a function of the
advice (`queued`), and so are the counts it releases; a request's response may span any windows. The ingress carry is
`received`: each window holds its own total to it exactly, so by induction from window 0 (B_0 = 0) on a link whose
every window is accepted (`service.LinkWindows.network_verdict`'s AND), B_w is the request frames the ingress
certifier released before window w.

**Soundness.** With the advice fixed, the counts are determined. On an accepted window w, the egress counts are
`derive(view(advice, w))`, a function of the advice alone: the ingress counts and the syncs only accept or refuse, and
change no count. Each window reports the advice it read (`digest`, its `advice` output), and the network verdict
refuses a window that names another than the first accepted window's. So on a link whose every window is accepted, the
egress counts of all its windows are a function of one advice, and the egress timing channel of accepted runs carries
at most the advice's bits (`advice.bits`, the accepted-runs bound's A_c), beside the log2 #Σ_sync bits a window the
sync charge already counts. Nothing else is left free: not the frames, not the ingress timing, not when the server
actually wrote. And the advice is held to the ingress head: no frame is handed before its request's frames were
released by the ingress certifier, δ + ⌊λ/ρ⌋ earlier.

**Premises** of "an honest advised link is accepted", which N2's claim doesn't rest on: the link's canonical order is
`frames`' (the prover's framer numbers events so, one per released token); each step's frames are handed to the egress
certifier in one bucket, its `release` (the prover's framer writes a step's events together, away from a bucket's
edge, and the proxy hands each frame when it is ready, neither shaping nor faulting a late one), or the advice takes
each step's release bucket from the certified egress record (`releases`), which reproduces it when every frame of a
step is handed by the bucket that releases the step's first; the steps' releases fall in the link's windows, each
with a declared sync; requests are listed in their ingress order with their ingress frames; each window's declared
sync has δ and ρ that every frame handed in it meets (a small δ, a ρ at least a stream's rate); with the salt on, the
own frame counts as bucket 0's, so a window that starts with RIN or more request frames carried in under-counts by one
until the own frame is released, which only a hand at its exact eligibility notices; and both ends share T and τ_b.

**Cost** (`size`): about 230 ANDs a bucket, the release check and the ingress floor: 137,990 at T = 600, R = 6,
RIN = 1, OWN = 0 and 139,253 at R = 16, RIN = 4, OWN = 1, a twentieth of one head check (3,229,371).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, add, and_reduce, const_bits, mux, sub
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier.grid import released
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
class Frame:
    """One egress frame of the advice, in the link's canonical order: the step that samples it, its request (an index
    into the advice's `requests`) and its logical index λ, its place in the response."""

    step: int
    request: int
    index: int


@dataclass(frozen=True)
class WindowView:
    """The unit's public inputs for one window (`view`): the frames handed in each bucket, the request frames the
    ingress counts must have released by each bucket, the egress queue at the window's start, and the request frames
    the window's ingress releases."""

    handed: tuple[int, ...]
    floor: tuple[int, ...]
    queued: int
    received: int


# --- the reference ----------------------------------------------------------------------------------------------------

def _nat(x: Any, what: str, least: int = 0) -> int:
    if type(x) is not int or x < least:
        raise ValueError(f"the advice's {what} is an int ≥ {least}, not {x!r}")
    return x


def frames(advice) -> list[Frame]:
    """The advice's egress frames in the link's canonical order: each step's released tokens, in its rows' order, a row
    sampling its request's next token when `start + n` is the prompt's length plus the tokens sampled before it, and
    the sample released while the request has released fewer than its `output_len`. Raises `ValueError` on a malformed
    advice: a row past its request's last known token, or a request that samples fewer than its `output_len`."""
    requests = list(advice.requests)
    prompt = [_nat(r.prompt_len, "prompt_len") for r in requests]
    length = [_nat(r.output_len, "output_len") for r in requests]
    sampled, out = [0] * len(requests), []
    for s, step in enumerate(advice.steps):
        for row in step.rows:
            j = row.request
            if type(j) is not int or not 0 <= j < len(requests):
                raise ValueError(f"step {s}: a row names request {j!r}, not one of the advice's {len(requests)}")
            end = _nat(row.start, "start") + _nat(row.n, "n", 1)
            known = prompt[j] + sampled[j]
            if end > known:
                raise ValueError(f"step {s}: request {j}'s row reaches position {end}, past its {known} known tokens")
            if end == known:
                if sampled[j] < length[j]:
                    out.append(Frame(s, j, sampled[j]))
                sampled[j] += 1
    short = [j for j in range(len(requests)) if sampled[j] < length[j]]
    if short:
        raise ValueError(f"requests {short[:8]} sample fewer tokens than their output_len")
    return out


def hands(advice, framed: Sequence[Frame] | None = None) -> list[int]:
    """Each frame's hand bucket, its step's `release`. Raises `ValueError` unless `release` has one bucket a step,
    non-decreasing."""
    rel = [_nat(b, "release bucket") for b in advice.release]
    if len(rel) != len(advice.steps):
        raise ValueError(f"the advice has {len(advice.steps)} steps and {len(rel)} release buckets")
    if any(b < a for a, b in zip(rel, rel[1:])):
        raise ValueError("the advice's release buckets decrease: the egress certifier is handed frames in order")
    return [rel[f.step] for f in (frames(advice) if framed is None else framed)]


def _received(advice) -> list[int]:
    got = [_nat(n, "received count") for n in advice.received]
    need = sum(_nat(r.ingress_frames, "ingress_frames", 1) for r in advice.requests)
    if sum(got) != need:
        raise ValueError(f"the advice's windows receive {sum(got)} request frames and its requests are {need}")
    return got


def view(advice, window: int, T: int, R: int, syncs: Sequence[ClockSync]) -> WindowView:
    """The unit's public inputs for `window` (the module docstring) at T buckets a window and egress allocation R, from
    the advice and every window's declared sync. Raises `ValueError` on a malformed advice, a window the advice's
    `received` doesn't cover, a frame handed in a window with no declared sync or before any request frame could be
    released, and a value past a u32."""
    if not all(type(x) is int and x >= 1 for x in (T, R)) or type(window) is not int or window < 0:
        raise ValueError("a view has T ≥ 1, R ≥ 1 and a window ≥ 0")
    framed = frames(advice)
    handed_at, got = hands(advice, framed), _received(advice)
    if window >= len(got):
        raise ValueError(f"the advice receives in {len(got)} windows, not window {window}")
    need, total = [], 0
    for r in advice.requests:
        total += r.ingress_frames
        need.append(total)
    lo, hi = window * T, (window + 1) * T
    floor_at: dict[int, int] = {}
    for f, h in zip(framed, handed_at):
        v = h // T
        if v >= len(syncs):
            raise ValueError(f"a frame is handed in window {v}, which declares no sync")
        sync = syncs[v]
        b = h - sync.delta - f.index * sync.rho.denominator // sync.rho.numerator
        if b < 0:
            raise ValueError(f"request {f.request}'s frame {f.index} is handed in bucket {h}, before its request")
        if b < hi:
            floor_at[b] = max(floor_at.get(b, 0), need[f.request])
    base, floor, g = sum(got[:window]), [], 0
    for b in sorted(k for k in floor_at if k < lo):
        g = max(g, floor_at[b])
    for t in range(lo, hi):
        g = max(g, floor_at.get(t, 0))
        floor.append(max(0, g - base))
    enqueued = [0] * hi
    for h in handed_at:
        if h < hi:
            enqueued[h] += 1
    queued = sum(enqueued[:lo]) - sum(released(R, enqueued[:lo]))
    out = WindowView(tuple(enqueued[lo:hi]), tuple(floor), queued, got[window])
    if max([*out.handed, *out.floor, out.queued, out.received]) > U32_MAX:
        raise ValueError("a view's counts are u32s")
    return out


def digest(advice) -> str:
    """The SHA-512 of what N2 reads of the advice (the module docstring's fields), the `advice` output each window of an
    advised link reports, by which `service.LinkWindows.network_verdict` holds every window to one advice."""
    body = {"format": DIGEST, "requests": [[r.prompt_len, r.output_len, r.ingress_frames] for r in advice.requests],
            "steps": [[[row.request, row.start, row.n] for row in step.rows] for step in advice.steps],
            "release": list(advice.release), "received": list(advice.received)}
    return hashlib.sha512(canonical_json(body)).hexdigest()


def derive(handed: Sequence[int], queued: int, R: int) -> list[int]:
    """The egress counts: the release, FIFO at R a bucket and left-packed, of `handed` from a queue of `queued`."""
    return released(R, list(handed), queued)


def releases(step_frames: Sequence[int], counts: Sequence[int]) -> list[int]:
    """Each step's release bucket as a certified egress record gives it: the bucket that releases the step's first
    frame, from the record's counts over the session's buckets (window 0's first on), its frames being the steps' in
    canonical order, `step_frames[s]` a step. A step with no frame takes the previous step's bucket (0 for the first).
    `view`'s counts then reproduce the record whenever every frame of a step was handed by the bucket that releases the
    step's first. Raises `ValueError` when the record releases fewer frames than the steps hold."""
    cum, out, k, last = 0, [], 0, 0
    bucket = iter(enumerate(counts))
    t = -1
    for n in step_frames:
        if n:
            while cum <= k:
                try:
                    t, c = next(bucket)
                except StopIteration:
                    raise ValueError("the record releases fewer frames than the advice's steps hold") from None
                cum += c
            last = t
        out.append(last)
        k += n
    return out


def reference(T: int, R: int, RIN: int, OWN: int, ingress: Sequence[int], egress: Sequence[int],
              handed: Sequence[int], floor: Sequence[int], queued: int, received: int) -> bool:
    """`NetworkAdvisedCounts_v1`'s output: every ingress count at most RIN and bucket 0's at least OWN, the ingress
    counts less the OWN own frames at least `floor` in every prefix and summing to `received`, and the egress counts
    `derive`'s of `handed` from `queued`."""
    if not all(len(x) == T for x in (ingress, egress, handed, floor)) or any(c > RIN for c in ingress) or \
            ingress[0] < OWN:
        return False
    cum, q = -OWN, queued
    for t in range(T):
        cum += ingress[t]
        a = q + handed[t]
        e = min(a, R)
        if cum < floor[t] or egress[t] != e:
            return False
        q = a - e
    return cum == received


# --- the circuit ------------------------------------------------------------------------------------------------------

STATICS = ("T", "R", "RIN", "OWN")
PART_STATICS = STATICS[:-1]


def _check_statics(S) -> None:
    if not all(type(x) is int and x >= 1 for x in (S.T, S.R, S.RIN)) or S.OWN not in (0, 1) or S.T * S.RIN > U32_MAX:
        raise ValueError(f"T, R and RIN are ≥ 1 with T·RIN a u32, and OWN 0 or 1, not {vars(S)}")


def _widths(S) -> SimpleNamespace:
    """The ingress count's bits (RIN), the ingress prefix's (T·RIN), the egress count's (R) and the queue's, which holds
    `queued` and every u32 `handed` without overflow."""
    return SimpleNamespace(IN=S.RIN.bit_length(), C=(S.T * S.RIN).bit_length(), E=S.R.bit_length(),
                           Q=(U32_MAX * (S.T + 1)).bit_length())


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
    W = _widths(S)
    state = Array(W.Q + W.C, BIT)
    return (("state", state), ("ingress", COUNT), ("egress", COUNT), ("handed", COUNT), ("floor", COUNT)), \
        Tuple(state, BIT)


def _build_bucket(C, S, state, *e, last: bool = False):
    """One bucket: the queue less the egress release, min(queue + handed, R), which the egress count must be, and the
    ingress prefix sum, which must be at least the floor, its count at most RIN. The last bucket's queue is read by
    nothing, so it keeps only the prefix sum."""
    W = _widths(S)
    c, g, h, f = (_u32(e[4 * k:4 * k + 4]) for k in range(4))
    q, cum = state[:W.Q], state[W.Q:]
    ok_in = and_reduce(C, [b.inv() for b in c[W.IN:]] + [_le_const(C, c[:W.IN], S.RIN)])
    cum = add(C, cum, c[:W.IN], width=W.C)
    ok_floor = and_reduce(C, [b.inv() for b in f[W.C:]] + [_lt(C, cum, f[:W.C], W.C).inv()])
    a = add(C, q, h, width=W.Q)
    out = mux(C, _lt(C, a, const_bits(S.R, W.Q), W.Q), a[:W.E], const_bits(S.R, W.E))
    ok_out = and_reduce(C, [b.inv() for b in g[W.E:]] + _same(C, g[:W.E], out))
    ok = [and_reduce(C, [ok_in, ok_floor, ok_out])]
    return [cum, ok] if last else [sub(C, a, out, W.Q)[0] + cum, ok]


def _sig_last(S):
    W = _widths(S)
    params, _ = _sig_bucket(S)
    return params, Tuple(Array(W.C, BIT), BIT)


def _build_last(C, S, state, *e):
    return _build_bucket(C, S, state, *e, last=True)


def _sig_end(S):
    W = _widths(S)
    return (("cum", Array(W.C, BIT)), ("received", COUNT)), BIT


def _build_end(C, S, cum, *received):
    """The window's request frames are `received`."""
    return [and_reduce(C, _same(C, list(cum) + [ZERO] * (32 - len(cum)), _u32(received)))]


NetworkAdvisedOwn = _part("NetworkAdvisedOwn", PART_STATICS, _sig_own, _build_own,
                          "bucket 0's ingress count less the certifier's own frame, and whether it is 1 … RIN")
NetworkAdvisedBucket = _part("NetworkAdvisedBucket", PART_STATICS, _sig_bucket, _build_bucket,
                             "one bucket: the egress count is the release of the queue and the frames handed, and the "
                             "ingress prefix sum is at least the floor")
NetworkAdvisedLast = _part("NetworkAdvisedLast", PART_STATICS, _sig_last, _build_last,
                           "the window's last bucket, as `NetworkAdvisedBucket`, keeping only the ingress prefix sum")
NetworkAdvisedEnd = _part("NetworkAdvisedEnd", PART_STATICS, _sig_end, _build_end,
                          "the window's ingress releases the advice's request frames")


def _sig(S):
    _check_statics(S)
    counts = Array(S.T, COUNT)
    return (("ingress", counts), ("egress", counts), ("handed", counts), ("floor", counts), ("queued", COUNT),
            ("received", COUNT)), BIT


def _body(B, S, ingress, egress, handed, floor, queued, received):
    _check_statics(S)
    W, K = _widths(S), _Consts(B)
    part = {d.name: bind(d, **{k: getattr(S, k) for k in PART_STATICS})
            for d in (NetworkAdvisedOwn, NetworkAdvisedBucket, NetworkAdvisedLast, NetworkAdvisedEnd)}
    ingress, egress, handed, floor = (_refs(x) for x in (ingress, egress, handed, floor))
    own = []
    if S.OWN:
        first = _refs(B.call(part["NetworkAdvisedOwn"], _coll(COUNT, ingress[:32])))
        ingress, own = first[:32] + ingress[32:], first[32:]
    q = _refs(queued)
    state = [q[8 * (3 - j // 8) + j % 8] for j in range(32)] + K.bits(0, W.Q - 32 + W.C)
    oks, n = [], 32 * (S.T - 1)
    if S.T > 1:
        out = _refs(B.scan(part["NetworkAdvisedBucket"], _coll(Array(W.Q + W.C, BIT), state),
                           [_coll(Array(S.T - 1, COUNT), x[:n]) for x in (ingress, egress, handed, floor)]))
        state, oks = out[:W.Q + W.C], out[W.Q + W.C:]
    last = _refs(B.call(part["NetworkAdvisedLast"], _coll(Array(W.Q + W.C, BIT), state),
                        *(_coll(COUNT, x[n:]) for x in (ingress, egress, handed, floor))))
    end = B.call(part["NetworkAdvisedEnd"], _coll(Array(W.C, BIT), last[:W.C]), received)
    bits = own + oks + last[W.C:] + _refs(end)
    return B.call(BL.and_gate(len(bits)), _coll(Array(len(bits), BIT), bits))


NetworkAdvisedCounts = CompositeDefinition(
    "NetworkAdvisedCounts", 1, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="1 iff the ingress counts are at most RIN, bucket 0's at least OWN, and with its OWN own frames taken off their "
        "prefix sums are at least `floor` and sum to `received`, and the egress counts are the release, FIFO at R, of "
        "`handed` from a queue of `queued` (`advised.reference`)")
NetworkAdvisedCounts.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins (`gate`)
#: the circuit digests circuit-check pins (`head_check.pinned_digest`)
PINS = Path(__file__).with_name("advised.circuit-check.json")


def unit(T: int, R: int, RIN: int, OWN: int):
    return bind(NetworkAdvisedCounts, T=T, R=R, RIN=RIN, OWN=OWN)


@lru_cache(maxsize=None)
def gate(T: int, R: int, RIN: int, OWN: int) -> PrimitiveDefinition:
    """`unit(…)` as one gate of N2's advised proof unit (`service.LinkWindows.unit`) that names the Definition and its
    pinned circuit digest, `Gate[NetworkAdvisedCounts_v1{…}@<digest>]_v2`, as `head_check.gate` holds the head check."""
    d = unit(T, R, RIN, OWN)
    return gate_of(d, pinned_digest(d, PINS), "N2's advised counts")


# --- the arguments ----------------------------------------------------------------------------------------------------

def _count_bits(counts: Sequence[int]) -> list[int]:
    data = b"".join(n.to_bytes(4, "big") for n in counts)
    return [(data[k] >> b) & 1 for k in range(len(data)) for b in range(8)]


def arguments(ingress: Sequence[int], egress: Sequence[int], v: WindowView) -> list[list[int]]:
    """The unit's arguments in its signature's order, each its leaves' bits, 0 or 1."""
    return [_count_bits(ingress), _count_bits(egress), _count_bits(v.handed), _count_bits(v.floor),
            _count_bits([v.queued]), _count_bits([v.received])]


def evaluate(key: tuple[int, int, int, int], cases: Sequence[tuple[Sequence[int], Sequence[int], WindowView]]) \
        -> list[int]:
    """The unit's output on each case, (ingress, egress, view) of one size class `key` = (T, R, RIN, OWN), in one
    evaluation."""
    args = [list(rows) for rows in zip(*(arguments(*c) for c in cases))]
    return [int(b) for b in evaluate_bits(unit(*key), *args).reshape(-1).tolist()]


def size(T: int, R: int, RIN: int, OWN: int) -> int:
    """The unit's ANDs."""
    return ands(unit(T, R, RIN, OWN))


__all__ = ["AdvisedProfile", "Frame", "NetworkAdvisedCounts", "PINS", "WindowView", "arguments", "derive", "digest",
           "evaluate", "frames", "gate", "hands", "reference", "releases", "size", "unit", "view"]
