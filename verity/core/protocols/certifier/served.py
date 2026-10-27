"""N2's served profile: on a served link, whose egress end has no public occupancy profile, the egress counts derived
from the ingress counts under the declared clock sync, zero-knowledge. A Boolean Definition, a gated part of N2's
served proof block (`service`'s `NetworkServedLinkWindow` v4), which runs on every link-window; its one output is the ok
bit. That unit's Program holds it as one gate that names its circuit digest (`gate`), pinned in this module's sidecar
and held to the Definition by circuit-check; N2's verifier of record reads its output from the unit's proof, and the
in-the-clear check (`LinkWindows.clear`) runs its `reference`, which the tests hold to the circuit.

    NetworkServedCounts_v1{T, FOUT, FIN, R, RIN, OWN}
        (ingress: Array<T, Array<4, Byte>>, egress: Array<T, Array<4, Byte>>, sync: Array<3, Array<16>>) -> Value<1>
        Byte = Array<8, Value<1>>

**Statics: the size class, all public.** T is both ends' buckets a window, FOUT = F the frames of every response (FIXLEN),
FIN the frames of every request (a power of two), R the egress allocation, RIN the ingress allocation and OWN the
ingress certifier's own frames at the head of every window's first bucket: 1 with its salt frame (a PoUW window of one
certifier window), else 0.

**Inputs.** `ingress` and `egress` are the two heads' counts as `RecordHead.encode` writes them (u32 big-endian), cut
from the heads N2's head checks open, so they are witness. `sync` is the window's declared clock sync (δ, p, q), each
u16 least significant bit first with ρ = p/q: public, the advice the audit already reads and charges (log2 #Σ_sync bits
a window).

**What 1 means.** Every ingress count is at most RIN, bucket 0's at least OWN, p and q are nonzero, and the egress
counts are `derive`'s, of the ingress counts with bucket 0's OWN own frames taken off:
- the ingress counts complete a request at every FIN-th frame, and each request's anchor is the bucket that releases its
  last frame, as the proxy anchors a stream;
- each request is answered by F frames, request after request in canonical order, the k-th frame of a response at
  logical index k (`schedule.LogicalFrame`), handed at `schedule.handing_buckets`' h = max(eligible, h before) under
  the sync;
- the egress certifier releases them FIFO, at most R a bucket (`grid.released`);
- and the window is idle at both ends: the requests end on a frame boundary (the ingress counts sum to a multiple of
  FIN), every frame is handed by the window's last bucket, and every frame is released by then (the egress queue ends
  empty).

**Carry-over.** Nothing carries between windows: each window's derivation starts with an empty egress queue, no request
frame pending and no earlier hand, and checks that it ends the same way. So by induction from the run's first window,
whose certifier starts empty (and from a certifier back from maintenance, which carries nothing), on a link whose every
window is accepted (`service.LinkWindows.network_verdict`'s AND) each window's egress counts are the Program's under the
premises below, and they depend on that window's ingress counts and sync alone. A proxy meets it with its admission rule
(`Admission`, `active.Bucketer(served=…)`): it admits a request only while the window, with it, still ends idle
(`admits`), and holds the rest, in order, for the next window, so each response is released in its request's window. A
request whose response would straddle a window's end makes the window refuse.

**Premises** of "the egress counts are the Program's": every response is F frames (FIXLEN: every request runs to its
max_tokens, one token a frame), every ingress frame is a request's but the certifier's own salt frame at each window's
head (OWN), every request is FIN frames (a fixed request size), the canonical order is request-major (each response's
frames consecutive, responses in request order), both ends share T and τ_b, and one declared sync applies to the
window's anchors. N2's claim doesn't rest on them: on an accepted
window the egress counts are this function of the ingress counts and the sync, whatever the frames hold.

**The circuit**, a closed form of `derive`. With G = ⌊(F − 1)·q/p⌋ a response's span, request j anchored at b_j
(sorted) has its last frame handed at b_j + δ + G, and its frame λ at max(b_j + δ + ⌊λ·q/p⌋, b_{j−1} + δ + G). So
the frames handed by bucket t are A(t) = F·N(t − δ − G) + CP(t − δ − a*), where N(x) is the requests anchored at or
before x (0 before the window), a* the first bucket after t − δ − G that anchors a request (the term 0 when a* > t − δ)
and CP(d) = ⌈(d + 1)·p/q⌉ the frames of one response handed d buckets after its first. The certifier releases
R(t) = min(R(t − 1) + R, A(t)), and the check is that the egress counts' prefix sums are R(t), and that R(T − 1) is
F·N(T − 1). In the circuit S = δ + G is clamped to T, which changes nothing in the window; the vectors N and
next-anchor are shifted by T − S with a barrel shifter, and CP is a division by q in each bucket.

**Cost** (`size`): 678,795 ANDs at T = 600, F = 1024, F_in = 1, R = 6, R_in = 1, of which the release check is 794 a
bucket (476,400), the shift 147,273, the arrivals 48 a bucket and the next anchors 20 a step; 688,210 at F_in = 2,
R = 16, R_in = 4. OWN = 1 adds 63: 678,858 and 688,273.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, add, and_reduce, compress_columns, const_bits, multiply, mux, \
    or_reduce, sub
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier.head_check import (BIT, BYTE, _coll, _Consts, _part, _refs, ands, gate_of,
                                                        pinned_digest)
from verity.core.protocols.certifier.grid import released
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, handing_buckets

#: δ, p and q are each u16 in the unit's `sync` input
SYNC_BITS = 16
SYNC_MAX = (1 << SYNC_BITS) - 1


@dataclass(frozen=True)
class ServedProfile:
    """A served link's public sizes: every response's frames (FIXLEN), every request's frames (a power of two), the
    link's windows, which `service.LinkWindows.network_verdict` takes by default, and `own`, the ingress certifier's own
    frames at the head of every window's first bucket: 1 with its salt on (`active.Proxy(salt=…)`, a PoUW window of
    one certifier window), else 0."""

    response_frames: int
    request_frames: int
    windows: int
    own: int = 0

    def __post_init__(self) -> None:
        if not all(type(x) is int for x in (self.response_frames, self.request_frames, self.windows, self.own)) or \
                self.response_frames < 1 or self.windows < 1 or self.request_frames < 1 or \
                self.request_frames & (self.request_frames - 1) or self.own not in (0, 1):
            raise ValueError("a served profile has F ≥ 1 frames a response, a power of two of frames a request, one "
                             f"or more windows and 0 or 1 own frames a window, not {self!r}")


# --- the reference ----------------------------------------------------------------------------------------------------

def anchors(ingress: Sequence[int], request_frames: int) -> list[int]:
    """Each whole request's anchor in the window: the bucket whose release completes its `request_frames` frames."""
    out, done, total = [], 0, 0
    for t, c in enumerate(ingress):
        total += c
        while (done + 1) * request_frames <= total:
            out.append(t)
            done += 1
    return out


def derive(ingress: Sequence[int], sync: ClockSync, profile: ServedProfile, allocation: int) -> list[int] | None:
    """The window's egress counts from its ingress counts under the declared sync: each request answered by
    `profile.response_frames` frames, handed by `schedule.handing_buckets` and released by `grid.released` at the egress
    allocation, from an idle start. The first `profile.own` frames of bucket 0 are the ingress certifier's own (its
    salt frame) and answer no request. None when they aren't there, or the window doesn't end idle: a request cut by
    the window's end, a frame handed after its last bucket, or one still queued then."""
    T, F, FIN = len(ingress), profile.response_frames, profile.request_frames
    if any(type(c) is not int or c < 0 for c in ingress):
        raise ValueError("ingress counts are non-negative ints")
    if profile.own:
        if not ingress or ingress[0] < profile.own:
            return None
        ingress = [ingress[0] - profile.own, *ingress[1:]]
    total = sum(ingress)
    if total % FIN or total // FIN * F > T * allocation:
        return None
    frames = [LogicalFrame(b"", a, k) for a in anchors(ingress, FIN) for k in range(F)]
    enqueued = [0] * T
    for h in handing_buckets(frames, [sync], T):
        if h >= T:
            return None
        enqueued[h] += 1
    counts = released(allocation, enqueued)
    return counts if sum(counts) == len(frames) else None


@dataclass(frozen=True)
class Admission:
    """A served link's admission rule at its proxy (`active.Bucketer(served=…)`): requests of the profile's F_in frames,
    each admitted only while its window, with it, still ends idle (`admits`) under `sync(window)` at the egress
    allocation. `sync` gives the window's declared sync where the proxy holds it, else the slowest of Σ_sync (largest
    δ, smallest ρ): a later hand only lowers the release curve, so a window idle under it is idle under any sync of
    Σ_sync."""

    profile: ServedProfile
    allocation: int
    sync: Callable[[int], ClockSync]


def admits(released_counts: Sequence[int], queued: int, sync: ClockSync, profile: ServedProfile, allocation: int,
           ingress_allocation: int, T: int) -> bool:
    """Whether the window still ends idle (`derive` is not None) with `released_counts` the ingress counts of its
    buckets so far and `queued` frames, the request's being admitted included, released FIFO at `ingress_allocation` a
    bucket from the next bucket on: false when one would be released after the window's last bucket."""
    counts = list(released_counts) + [0] * (T - len(released_counts))
    for k in range(queued):
        t = len(released_counts) + k // ingress_allocation
        if t >= T:
            return False
        counts[t] += 1
    return derive(counts, sync, profile, allocation) is not None


def sync_words(sync: ClockSync) -> tuple[int, int, int]:
    """The sync as the unit's `sync` input: δ, and ρ = p/q in lowest terms, each u16."""
    words = (sync.delta, sync.rho.numerator, sync.rho.denominator)
    if not all(0 <= x <= SYNC_MAX for x in words):
        raise ValueError(f"a served unit's sync has δ, p and q below 2^{SYNC_BITS}, not {sync!r}")
    return words


def reference(T: int, F: int, FIN: int, R: int, RIN: int, OWN: int, ingress: Sequence[int], egress: Sequence[int],
              sync: tuple[int, int, int]) -> bool:
    """`NetworkServedCounts_v1`'s output: every ingress count at most RIN, p and q nonzero, and the egress counts
    `derive`'s, with OWN own frames at the head of bucket 0."""
    delta, p, q = sync
    if len(ingress) != T or len(egress) != T or not p or not q or any(c > RIN for c in ingress):
        return False
    return derive(ingress, ClockSync(delta, Fraction(p, q)), ServedProfile(F, FIN, 1, OWN), R) == list(egress)


# --- the circuit ------------------------------------------------------------------------------------------------------

def _check_statics(S) -> None:
    if not all(type(x) is int and x >= 1 for x in (S.T, S.FOUT, S.FIN, S.R, S.RIN)) or S.FIN & (S.FIN - 1) or \
            S.T > SYNC_MAX or S.OWN not in (0, 1):
        raise ValueError(f"T < 2^{SYNC_BITS}, F, R and RIN are ≥ 1, FIN a power of two and OWN 0 or 1, not {vars(S)}")


def _widths(S) -> SimpleNamespace:
    K = S.FIN.bit_length() - 1
    wc = (S.T * S.RIN).bit_length()
    n_max = (S.T * S.RIN) >> K
    we = S.R.bit_length()
    wa = (S.FOUT * n_max + S.FOUT).bit_length()
    return SimpleNamespace(K=K, IN=S.RIN.bit_length(), C=wc, N=max(1, wc - K), T=S.T.bit_length(),
                           G=(3 * S.T).bit_length(), F=max(1, (S.FOUT - 1).bit_length()), E=we, A=wa,
                           S=max(wa, (S.T * ((1 << we) - 1) + S.R).bit_length()) + 1)


STATICS = ("T", "FOUT", "FIN", "R", "RIN", "OWN")
PART_STATICS = STATICS[:-1]


def _u32(e, at: int = 0) -> list:
    """The 32 bits, least significant first, of the u32 big-endian in bytes e[at … at + 3]."""
    return [e[at + 3 - j // 8][j % 8] for j in range(32)]


def _same(C, a, b) -> list:
    """The bits that are 1 iff a = b, both padded to one width."""
    n = max(len(a), len(b))
    a, b = list(a) + [ZERO] * (n - len(a)), list(b) + [ZERO] * (n - len(b))
    return [(x ^ y).inv() for x, y in zip(a, b)]


def _lt(C, a, b, w) -> object:
    return sub(C, a, b, w)[1]


def _le_const(C, x, k: int) -> object:
    if k >= (1 << len(x)) - 1:
        return ONE
    return _lt(C, const_bits(k, len(x)), x, len(x)).inv()


def _min(C, a, b, w) -> list:
    a, b = (list(v) + [ZERO] * (w - len(v)) for v in (a, b))
    return mux(C, _lt(C, a, b, w), a, b)


def _mulc(C, x, k: int, w: int) -> list:
    """x·k for a constant k, its low w bits."""
    cols = [[] for _ in range(w)]
    for j in range(k.bit_length()):
        if k >> j & 1:
            for i, b in enumerate(x):
                if i + j < w:
                    cols[i + j].append(b)
    return (compress_columns(C, cols) + [ZERO] * w)[:w]


def _divide(C, y, q, n: int) -> list:
    """⌊y/q⌋'s n bits, least significant first, when y < q·2^n (restoring division); garbage otherwise."""
    wq = len(q)
    y = list(y) + [ZERO] * max(0, n + wq - len(y))
    rem, out = y[n:n + wq], []
    for i in reversed(range(n)):
        rem = [y[i]] + rem
        diff, borrow = sub(C, rem, q, wq + 1)
        out.append(borrow.inv())
        rem = mux(C, borrow.inv(), diff, rem)[:wq]
    return out[::-1]


# The parts. Each is traced once per size class, and the body below wires them: the arrivals and the next anchors are
# scans, the shift one part, the buckets a scan carrying the egress counts' prefix sum.

def _w_info(W) -> int:
    return 2 * W.T + 1 + SYNC_BITS + 2 + 1


def _sig_sync(S):
    W = _widths(S)
    return (("sync", Array(3, Array(SYNC_BITS, BIT))),), Array(_w_info(W), BIT)


def _build_sync(C, S, delta, p, q):
    """u = T − S, Gc − 1 and whether Gc = 0, p + q − 1, and whether p and q are nonzero, where G = ⌊(F − 1)·q/p⌋ and
    S = δ + G and δ are clamped to T, and Gc = S − δ."""
    T, F, W = S.T, S.FOUT, _widths(S)
    num = _mulc(C, q, F - 1, SYNC_BITS + (F - 1).bit_length())
    G = _divide(C, num, p, len(num))
    big = _lt(C, delta, const_bits(T, SYNC_BITS), SYNC_BITS).inv()
    dc = mux(C, big, const_bits(T, W.T), delta[:W.T])
    s = add(C, dc, G, width=max(W.T, len(G)) + 1)
    Sc = mux(C, _lt(C, s, const_bits(T, len(s)), len(s)), s[:W.T], const_bits(T, W.T))
    Gc = sub(C, Sc, dc, W.T)[0]
    u = sub(C, const_bits(T, W.T), Sc, W.T)[0]
    gm1, none = sub(C, Gc, [ONE], W.T)
    k1 = sub(C, add(C, p, q, width=SYNC_BITS + 1, carry_out=True), [ONE], SYNC_BITS + 2)[0]
    return [u + gm1 + [none] + k1 + [C.AND(or_reduce(C, p), or_reduce(C, q))]]


def _sig_own(S):
    return (("count", Array(4, BYTE)),), Tuple(Array(4, BYTE), BIT)


def _build_own(C, S, *count):
    """Bucket 0's ingress count less the certifier's own frame, and whether the count is 1 … RIN."""
    W = _widths(S)
    c = _u32(count)
    rest, borrow = sub(C, c, [ONE], 32)
    ok = and_reduce(C, [borrow.inv()] + [b.inv() for b in c[W.IN:]] + [_le_const(C, c[:W.IN], S.RIN)])
    return [[rest[8 * (3 - k) + b] for b in range(8)] for k in range(4)] + [[ok]]


def _sig_arrive(S):
    W = _widths(S)
    return (("cum", Array(W.C, BIT)), ("count", Array(4, BYTE))), Tuple(Array(W.C, BIT), Array(W.N + 2, BIT))


def _build_arrive(C, S, cum, *count):
    """One ingress bucket: the prefix sum C_t, and N_t = ⌊C_t / FIN⌋, whether N_t > N_{t−1} (the bucket anchors a
    request) and whether the count is at most RIN."""
    W = _widths(S)
    c = _u32(count)
    ok = and_reduce(C, [b.inv() for b in c[W.IN:]] + [_le_const(C, c[:W.IN], S.RIN)])
    after = add(C, cum, c[:W.IN], width=W.C)
    n_before, n = cum[W.K:W.K + W.N], after[W.K:W.K + W.N]
    return [after, n + [or_reduce(C, [a ^ b for a, b in zip(n, n_before)]), ok]]


def _sig_next(S):
    W = _widths(S)
    return (("next", Array(W.G, BIT)), ("anchors", BIT)), Tuple(Array(W.G, BIT), Array(W.G, BIT))


def _build_next(C, S, nxt, anchors):
    """next(z) from next(z + 1): 0 when bucket z anchors a request, else one more."""
    W = _widths(S)
    out = [C.AND(anchors[0].inv(), b) for b in add(C, nxt, [ONE], width=W.G)]
    return [out, out]


def _sig_shift(S):
    W = _widths(S)
    return (("vec", Array(2 * S.T, Array(W.N + W.G, BIT))), ("info", Array(_w_info(W), BIT))), \
        Array(S.T, Array(W.N + W.G, BIT))


def _build_shift(C, S, *e):
    """out[t] = vec[t + u], u = T − S: a barrel shifter, its largest stage first, so each stage keeps only the entries
    a smaller shift can still reach."""
    W, T = _widths(S), S.T
    cur, u = list(e[:2 * T]), e[2 * T][:W.T]
    for k in reversed(range(W.T)):
        sh = 1 << k
        at = (lambda i: cur[i] if i < len(cur) else [ZERO] * (W.N + W.G))
        cur = [mux(C, u[k], at(i + sh), at(i)) for i in range(T + sh - 1)]
    return cur[:T]


def _sig_bucket(S):
    W = _widths(S)
    return (("released", Array(W.S, BIT)), ("entry", Array(W.N + W.G, BIT)), ("count", Array(4, BYTE)),
            ("info", Array(_w_info(W), BIT)), ("sync", Array(3, Array(SYNC_BITS, BIT)))), \
        Tuple(Array(W.S, BIT), BIT)


def _build_bucket(C, S, released_before, entry, *e):
    """One egress bucket t, reading (N(t − S), next(t − S + 1)): the frames handed by t, A = F·N(t − S) + CP(d) when a
    request anchored after t − S has its first frame due by t (d buckets ago), and the check that the egress counts'
    prefix sum is min(its value at t − 1 + R, A)."""
    W = _widths(S)
    count, info, (_, p, q) = e[:4], e[4], e[5:]
    gm1, none, k1 = info[W.T:2 * W.T], info[2 * W.T], info[2 * W.T + 1:2 * W.T + 1 + SYNC_BITS + 2]
    n_x, gap = entry[:W.N], entry[W.N:]
    d, after = sub(C, gm1 + [ZERO] * (W.G - W.T), gap, W.G)
    live = C.AND(after.inv(), none.inv())
    y = add(C, multiply(C, d[:W.T], p), k1, width=W.T + SYNC_BITS + 2)
    cp = [C.AND(live, b) for b in _divide(C, y, q, W.F)]
    A = add(C, _mulc(C, n_x, S.FOUT, W.A), cp, width=W.A)
    due = _min(C, A, add(C, released_before, const_bits(S.R, W.S), width=W.S), W.S)
    c = _u32(count)
    released_after = add(C, released_before, c[:W.E], width=W.S)
    return [released_after, [and_reduce(C, [b.inv() for b in c[W.E:]] + _same(C, released_after, due))]]


def _sig_end(S):
    W = _widths(S)
    return (("released", Array(W.S, BIT)), ("cum", Array(W.C, BIT))), BIT


def _build_end(C, S, released_total, cum):
    """The window ends idle: the ingress counts end on a request's last frame, and every frame is released."""
    W = _widths(S)
    return [and_reduce(C, [b.inv() for b in cum[:W.K]]
                       + _same(C, released_total, _mulc(C, cum[W.K:W.K + W.N], S.FOUT, W.S)))]


NetworkServedSync = _part("NetworkServedSync", PART_STATICS, _sig_sync, _build_sync,
                          "the declared sync's shift u = T − min(δ + G, T), Gc − 1, Gc = 0, p + q − 1 and p, q ≠ 0")
NetworkServedOwn = _part("NetworkServedOwn", PART_STATICS, _sig_own, _build_own,
                         "bucket 0's ingress count less the certifier's own frame, and whether it is 1 … RIN")
NetworkServedArrive = _part("NetworkServedArrive", PART_STATICS, _sig_arrive, _build_arrive,
                            "one ingress bucket: the prefix sum, N_t, whether it anchors a request, count ≤ RIN")
NetworkServedNext = _part("NetworkServedNext", PART_STATICS, _sig_next, _build_next,
                          "next(z) from next(z + 1): the buckets to the first that anchors a request")
NetworkServedShift = _part("NetworkServedShift", PART_STATICS, _sig_shift, _build_shift,
                           "the (N, next) vector shifted by u = T − S: bucket t reads entry t + u")
NetworkServedBucket = _part("NetworkServedBucket", PART_STATICS, _sig_bucket, _build_bucket,
                            "one egress bucket: the frames handed by t and the release check on the counts' prefix sum")
NetworkServedEnd = _part("NetworkServedEnd", PART_STATICS, _sig_end, _build_end,
                         "the window ends idle: no request cut, every frame released")


def _sig(S):
    _check_statics(S)
    counts = Array(S.T, Array(4, BYTE))
    return (("ingress", counts), ("egress", counts), ("sync", Array(3, Array(SYNC_BITS, BIT)))), BIT


def _body(B, S, ingress, egress, sync):
    _check_statics(S)
    T, W, K = S.T, _widths(S), _Consts(B)
    part = {d.name: bind(d, **{k: getattr(S, k) for k in PART_STATICS})
            for d in (NetworkServedSync, NetworkServedOwn, NetworkServedArrive, NetworkServedNext, NetworkServedShift,
                      NetworkServedBucket, NetworkServedEnd)}
    own = []
    if S.OWN:
        counts = _refs(ingress)
        first = _refs(B.call(part["NetworkServedOwn"], _coll(Array(4, BYTE), counts[:32])))
        ingress, own = _coll(Array(T, Array(4, BYTE)), first[:32] + counts[32:]), first[32:]
    info = B.call(part["NetworkServedSync"], sync)
    arrivals = _refs(B.scan(part["NetworkServedArrive"], _coll(Array(W.C, BIT), K.bits(0, W.C)), [ingress]))
    cum, ys = arrivals[:W.C], arrivals[W.C:]
    y = W.N + 2
    N = [ys[t * y:t * y + W.N] for t in range(T)]
    anchors = [ys[t * y + W.N] for t in reversed(range(T))] + K.bits(0, T - 1)
    oks = [ys[t * y + W.N + 1] for t in range(T)]
    nexts = _refs(B.scan(part["NetworkServedNext"], _coll(Array(W.G, BIT), K.bits(T, W.G)),
                         [_coll(Array(2 * T - 1, BIT), anchors)]))[W.G:]
    # scan step i is z = T − 1 − i; entry i of the vector is (N(i − T), next(i − T + 1))
    nxt = {T - 1 - i: nexts[i * W.G:(i + 1) * W.G] for i in range(2 * T - 1)} | {T: K.bits(T, W.G)}
    vec = [(K.bits(0, W.N) if i < T else N[i - T]) + nxt[i - T + 1] for i in range(2 * T)]
    shifted = B.call(part["NetworkServedShift"], _coll(Array(2 * T, Array(W.N + W.G, BIT)), [r for v in vec for r in v]),
                     info)
    buckets = _refs(B.scan(part["NetworkServedBucket"], _coll(Array(W.S, BIT), K.bits(0, W.S)), [shifted, egress],
                           [info, sync]))
    end = B.call(part["NetworkServedEnd"], _coll(Array(W.S, BIT), buckets[:W.S]), _coll(Array(W.C, BIT), cum))
    bits = own + oks + buckets[W.S:] + [_refs(info)[-1], _refs(end)[0]]
    return B.call(BL.and_gate(len(bits)), _coll(Array(len(bits), BIT), bits))


NetworkServedCounts = CompositeDefinition(
    "NetworkServedCounts", 1, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="1 iff the ingress counts are at most RIN and the egress counts are the shaper's release of FOUT frames a "
        "request of FIN frames under the declared sync (δ, p/q), from and to an idle window, the first OWN frames of "
        "bucket 0 the certifier's own (`served.derive`)")


NetworkServedCounts.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins (`gate`)
#: the circuit digests circuit-check pins (`head_check.pinned_digest`)
PINS = Path(__file__).with_name("served.circuit-check.json")


def unit(T: int, F: int, FIN: int, R: int, RIN: int, OWN: int):
    return bind(NetworkServedCounts, T=T, FOUT=F, FIN=FIN, R=R, RIN=RIN, OWN=OWN)


@lru_cache(maxsize=None)
def gate(T: int, F: int, FIN: int, R: int, RIN: int, OWN: int) -> PrimitiveDefinition:
    """`unit(…)` as one gate of N2's served proof block (`service.LinkWindows.unit`) that names the Definition and its
    pinned circuit digest, `Gate[NetworkServedCounts_v1{…}@<digest>]_v2`, as `head_check.gate` holds the head check."""
    d = unit(T, F, FIN, R, RIN, OWN)
    return gate_of(d, pinned_digest(d, PINS), "N2's served counts")


# --- the arguments ----------------------------------------------------------------------------------------------------

def _count_bits(counts: Sequence[int]) -> list[int]:
    data = b"".join(n.to_bytes(4, "big") for n in counts)
    return [(data[k] >> b) & 1 for k in range(len(data)) for b in range(8)]


def arguments(ingress: Sequence[int], egress: Sequence[int], sync: tuple[int, int, int]) -> list[list[int]]:
    """The unit's arguments in its signature's order, each its leaves' bits, 0 or 1."""
    return [_count_bits(ingress), _count_bits(egress), [(x >> j) & 1 for x in sync for j in range(SYNC_BITS)]]


def evaluate(key: tuple[int, int, int, int, int, int], cases: Sequence[tuple[Sequence[int], Sequence[int],
                                                                          tuple[int, int, int]]]) -> list[int]:
    """The unit's output on each case, (ingress, egress, sync) of one size class `key` = (T, F, FIN, R, RIN, OWN), in
    one evaluation."""
    args = [list(rows) for rows in zip(*(arguments(*c) for c in cases))]
    return [int(b) for b in evaluate_bits(unit(*key), *args).reshape(-1).tolist()]


def size(T: int, F: int, FIN: int, R: int, RIN: int, OWN: int) -> int:
    """The unit's ANDs."""
    return ands(unit(T, F, FIN, R, RIN, OWN))


__all__ = ["NetworkServedCounts", "PINS", "SYNC_BITS", "ServedProfile", "anchors", "arguments", "derive", "evaluate",
           "gate", "reference", "size", "sync_words", "unit"]
