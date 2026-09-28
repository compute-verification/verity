"""A colluding sender's covert channel through the online prover's declared clock sync, and a decoder outside.

The egress bound charges `log2 #Σ_sync` bits per window for the declared sync (7 bits at 16 × 8). That is an UPPER
bound. With an online prover (`online.Prover`, rules `last`, `cover-K-M[-fallback]`) the declared sync for window w is
computed at bucket w·T from the frames read so far, so a server inside the boundary steers it by WHEN it writes. This
module measures what a colluding sender and an outside receiver actually achieve through that declaration while every
window stays accepted.

**The channel.** A frame anchored (ingress-released) in window u at index 0, written so it is ready `g` buckets after
its anchor, forces the online prover's summary of window u to need `δ = g` to meet it (`online._Link.cost`). So the
sender sets each window u's frontier `g_u` by writing all of u's frames `g_u` buckets late. The next window's
declaration reads that frontier: under `last` (`cover-1-0`) window u+1 declares exactly `δ = g_u`; under `cover-K-M`
it declares `δ = max(g_{u-K+1..u}) + M` (rounded into Σ_sync). The receiver outside reads the declared sync (the wire's
occupancy, or the published advice) and recovers the frontier the sender chose.

**Acceptance.** Window u's frames are met by window u's own declared sync only if `g_u ≤ δ_declared(u)` (a frame ready
later than its hand time is a `mismatch`, fail-closed). `δ_declared(u)` is set by earlier frontiers through the rule, so
the sender draws each `g_u` from `[1, δ_declared(u)]`: every window then stays accepted. This is the whole lever — the
sender cannot raise a frontier above what the rule already granted without being rejected.

`encode_writes` runs this closed loop against a faithful replica of the proxy's own `online.Prover` (same arrival order,
same anchors and ready buckets), so its predicted declarations equal the proxy's; the harness verifies that on the
injected clock. `measure` reports the achieved rate from the proxy's logged declarations: the mutual information between
the symbol the sender sent and the sync the receiver read, the symbol error rate of a concrete decoder, and the bits a
decoder recovers per window. Every number is `≤ log2 #Σ_sync`; a measurement above it would be a soundness finding.

Stdlib only; it imports the certifier's `schedule` and `online` and nothing from the benchmark harness, so the tests and
`active_replay.py` share it.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from verity.core.protocols.certifier import online
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet


def encode_writes(keys: Sequence[tuple[int, int, int, int]], sessions: Sequence[list], sync_set: SyncSet,
                  window_buckets: int, windows: int, rule: online.Rule | None, envelope: ClockSync, seed: int,
                  horizon: int, fixed: Sequence[ClockSync] = ()) -> dict:
    """The covert sender's writes over one configuration, and what they transmit.

    `keys` is the Program's canonical order, `(ready, anchor, session rank, index)` per position, as
    `active_replay.program_keys` gives it; `anchor` is the ingress-release bucket. Each window's frames are written
    `g` buckets after their anchor, where `g` is a random symbol drawn from `[1, δ_declared]`, the feasible frontier
    that keeps the window accepted. The declarations are predicted by a replica of the proxy's `online.Prover`, fed the
    frames in the proxy's read order (by write bucket, then position). With `rule` None the declarations are `fixed`
    (syncs chosen before the run, `envelope` or `model`): the same sender then steers nothing, the control.

    Returns `writes` (session name → `[[bucket, position], …]`, as `active_replay`'s server reads them), `symbols` (the
    frontier `g` sent per window), `caps` (the feasible ceiling per window) and `declared` (the predicted sync per
    window, as `[delta, str(rho)]`)."""
    by_window: dict[int, list[int]] = defaultdict(list)
    for position, (_ready, anchor, _rank, _k) in enumerate(keys):
        by_window[anchor // window_buckets].append(position)
    prover = online.Prover(sync_set, window_buckets, rule, envelope, links=1) if rule is not None else None
    rng = random.Random(seed)
    unread: list[tuple[int, int, int, int]] = []  # (write bucket, position, anchor, index): the proxy reads in this order
    writes: dict[str, list[list[int]]] = defaultdict(list)
    symbols, caps, declared = [], [], []

    def feed(limit: int) -> None:
        while unread and unread[0][0] < limit:
            b, position, anchor, k = heapq.heappop(unread)
            prover.arrive(0, position, LogicalFrame(position.to_bytes(8, "big"), anchor, k), b + 1)

    for w in range(windows):
        if prover is None:
            sync = fixed[w]
        else:
            feed(w * window_buckets)
            sync = prover.declare()
        declared.append([sync.delta, str(sync.rho)])
        cap = max(1, sync.delta)
        g = rng.randint(1, cap)
        symbols.append(g)
        caps.append(cap)
        for position in by_window.get(w, []):
            _ready, anchor, rank, k = keys[position]
            b = min(anchor + g - 1, horizon - 2)
            b = max(b, anchor)
            writes[sessions[rank][0]].append([b, position])
            heapq.heappush(unread, (b, position, anchor, k))
    return {"writes": {name: ws for name, ws in writes.items()}, "symbols": symbols, "caps": caps,
            "declared": declared}


class LiveSender:
    """The covert sender in a live server's shim, which can only delay what the model writes: a stream anchored in
    window w writes nothing before bucket `anchor + g_w − 1` (`hold`), so window w's frontier is `max(g_w, the model's
    own)`, and `g_w` is drawn from `[1, δ_declared(w)]` as `encode_writes` draws it. `δ_declared(w)` comes from a
    replica of the proxy's prover fed every event as the shim wrote it (`wrote`), at the bucket after it was written;
    the shim takes a stream's anchor to be the bucket its request reached it in. `symbol(w)` must be called no earlier
    than bucket w·T, when every event that window's declaration reads has been written."""

    def __init__(self, sync_set: SyncSet, window_buckets: int, rule: online.Rule | None, envelope: ClockSync,
                 seed: int, fixed: Sequence[ClockSync] = ()) -> None:
        self.T, self.fixed = window_buckets, list(fixed)
        self.prover = online.Prover(sync_set, window_buckets, rule, envelope, links=1) if rule is not None else None
        self.rng = random.Random(seed)
        self.symbols: list[int] = []
        self.caps: list[int] = []
        self.declared: list[list] = []
        self.written: list[tuple[int, int, int, int]] = []  # (bucket, position, anchor, index), in position order
        self.fed = 0

    def symbol(self, window: int) -> int:
        while len(self.symbols) <= window:
            w = len(self.symbols)
            if self.prover is None:
                sync = self.fixed[min(w, len(self.fixed) - 1)]
            else:
                while self.fed < len(self.written) and self.written[self.fed][0] < w * self.T:
                    b, position, anchor, k = self.written[self.fed]
                    self.prover.arrive(0, position, LogicalFrame(position.to_bytes(8, "big"), anchor, k), b + 1)
                    self.fed += 1
                sync = self.prover.declare()
            self.declared.append([sync.delta, str(sync.rho)])
            self.caps.append(max(1, sync.delta))
            self.symbols.append(self.rng.randint(1, self.caps[-1]))
        return self.symbols[window]

    def hold(self, anchor: int) -> int:
        """The first bucket a stream anchored at `anchor` may write in."""
        return anchor + self.symbol(anchor // self.T) - 1

    def wrote(self, position: int, anchor: int, k: int, bucket: int) -> None:
        self.written.append((bucket, position, anchor, k))

    def log(self, windows: int) -> dict:
        if windows:
            self.symbol(windows - 1)
        return {"symbols": self.symbols[:windows], "caps": self.caps[:windows], "predicted": self.declared[:windows]}


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def _mutual_information(pairs: Sequence[tuple]) -> float:
    """The plug-in mutual information I(X;Y) in bits of a sample of (x, y) pairs."""
    if not pairs:
        return 0.0
    xs, ys = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    joint = Counter(pairs)
    return max(0.0, _entropy(list(xs.values())) + _entropy(list(ys.values())) - _entropy(list(joint.values())))


def _sync_index(sync: Sequence, sync_set: SyncSet) -> int:
    """A declared sync `[delta, str(rho)]`'s index in Σ_sync, or -1 if out of set (the envelope may be)."""
    from fractions import Fraction

    delta, rho = int(sync[0]), Fraction(sync[1])
    if delta not in sync_set.deltas or rho not in sync_set.rhos:
        return -1
    return sync_set.deltas.index(delta) * len(sync_set.rhos) + sorted(sync_set.rhos).index(rho)


def margin_of(rule_name: str) -> int:
    """The δ margin M a rule adds over the frontiers it reads: 0 for `last`, M for `cover-K-M[-fallback]`."""
    online.rule(rule_name)
    return 0 if rule_name == "last" else int(rule_name.split("-")[2])


def _null_mi(pairs: Sequence[tuple], shuffles: int = 20) -> float:
    """The plug-in MI's bias on this sample: its mean with the symbols shuffled against the declarations."""
    if not pairs:
        return 0.0
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    rng, total = random.Random(0), 0.0
    for _ in range(shuffles):
        rng.shuffle(xs)
        total += _mutual_information(list(zip(xs, ys)))
    return total / shuffles


def measure(symbols: Sequence[int], declared: Sequence[Sequence], sync_set: SyncSet, delay: int = 1,
            margin: int = 0) -> dict:
    """`measure_runs` on one run."""
    return measure_runs([(symbols, declared)], sync_set, delay, margin)


def measure_runs(runs: Sequence[tuple[Sequence[int], Sequence[Sequence]]], sync_set: SyncSet, delay: int = 1,
                 margin: int = 0) -> dict:
    """The achieved covert rate from the sender's per-window symbols and the declarations the receiver read, pooled
    over runs (each a configuration, whose prover starts with no history).

    Pairs symbol `g_w` with the declaration at window `w + delay` of the same run (the first window that reads the
    frontier `g_w`). Reports the plug-in mutual information with the declared δ and with the full declared sync, and
    its shuffle null (the estimator's bias on this many pairs); the symbol error rate of the decoder
    `ĝ = δ_declared(w + delay) − M` (exact under `last`, where M = 0); and its decoded rate, the mutual information
    between the symbol and `ĝ` less its shuffle null (a decoder that guesses scores 0, not its lucky hits). Every rate
    is in bits per window and is at most `log2 #Σ_sync`."""
    pairs_delta, pairs_full, pairs_decoded = [], [], []
    for symbols, declared in runs:
        for w in range(max(0, len(symbols) - delay)):
            d = int(declared[w + delay][0])
            pairs_delta.append((symbols[w], d))
            pairs_full.append((symbols[w], _sync_index(declared[w + delay], sync_set)))
            pairs_decoded.append((symbols[w], d - margin))
    n = len(pairs_delta)
    errors = sum(x != y for x, y in pairs_decoded)
    return {
        "window_decisions": n,
        "mutual_information_delta_bits": _mutual_information(pairs_delta),
        "mutual_information_full_bits": _mutual_information(pairs_full),
        "mutual_information_null_bits": _null_mi(pairs_full),
        "mutual_information_corrected_bits": max(0.0, _mutual_information(pairs_full) - _null_mi(pairs_full)),
        "declared_delta_entropy_bits": _entropy(list(Counter(d for _, d in pairs_delta).values())),
        "declared_full_entropy_bits": _entropy(list(Counter(d for _, d in pairs_full).values())),
        "symbol_entropy_bits": _entropy(list(Counter(x for x, _ in pairs_delta).values())),
        "symbol_error_rate": errors / n if n else 0.0,
        "decoded_bits_per_window": max(0.0, _mutual_information(pairs_decoded) - _null_mi(pairs_decoded)),
        "distinct_declared_delta": len({d for _, d in pairs_delta}),
        "distinct_declared_full": len({d for _, d in pairs_full}),
        "max_bits": sync_set.bits,
    }


def stream_seed(seed: int, name: str) -> int:
    """A configuration's own seed: pooled configurations must carry independent messages, or the estimator would read
    their shared prefix as signal."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def control_symbols(windows: int, seed: int) -> list[int]:
    """A random message for a sender that transmits nothing: independent of the declarations, so its measured rate is
    the estimator's null (the control)."""
    rng = random.Random(seed ^ 0xC0FFEE)
    return [rng.randrange(1, 16) for _ in range(windows)]


def pool(summaries: Sequence[dict], sync_set: SyncSet) -> dict:
    """`measure_runs` over the runs of several `active_replay.py` summaries (one per trace, say), per strategy, with the
    link-windows each strategy had accepted, raw and committed."""
    out: dict[str, dict] = {}
    for strategy in sorted({s for doc in summaries for s in doc.get("covert", {})}):
        mine = [doc["covert"][strategy] for doc in summaries if strategy in doc.get("covert", {})]
        runs = [tuple(r) for m in mine for r in m["runs"]]
        out[strategy] = measure_runs(runs, sync_set, margin=mine[0]["margin"])
        rows = [(doc["egress"], doc["ingress"]) if strategy == "honest" else (doc["adversarial"][strategy],)
                for doc in summaries if strategy in doc.get("covert", {})]
        out[strategy]["link_windows"] = sum(x["link_windows"] for r in rows for x in r)
        out[strategy]["accepted"] = {k: sum(x["accepted"][k] for r in rows for x in r) for k in ("raw", "committed")}
        out[strategy]["configurations"] = len(runs)
        if any("declared_are_predicted" in m for m in mine):
            out[strategy]["declared_are_predicted"] = all(m.get("declared_are_predicted", False) for m in mine)
    return out


def steady_state(sync_set: SyncSet, window_buckets: int, rule_name: str, envelope: ClockSync, windows: int,
                 seed: int) -> dict:
    """`encode_writes`' sender over one link of `windows` windows on a noiseless channel (one frame per window, at its
    start, so its frontier is exactly the symbol), measured by `measure_runs`: the rate past the start, where every
    `active_replay` configuration declares the envelope. `envelope` names the fixed control."""
    T = window_buckets
    keys = [(w * T + 1, w * T, w, 0) for w in range(windows)]
    rule = None if rule_name == "envelope" else online.rule(rule_name)
    enc = encode_writes(keys, [[f"s{w}"] for w in range(windows)], sync_set, T, windows, rule, envelope, seed,
                        windows * T, [envelope] * windows)
    return measure_runs([(enc["symbols"], enc["declared"])], sync_set, margin=margin_of(rule_name) if rule else 0)


def _line(m: dict) -> str:
    return (f"MI {m['mutual_information_full_bits']:.3f} bits/window (null {m['mutual_information_null_bits']:.3f}, "
            f"corrected {m['mutual_information_corrected_bits']:.3f}), δ-MI {m['mutual_information_delta_bits']:.3f}, "
            f"decoded {m['decoded_bits_per_window']:.3f}, SER {m['symbol_error_rate']:.3f}, "
            f"{m['window_decisions']} window-decisions")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from fractions import Fraction
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="covert.py", description="pool active_replay.py summaries' covert measurements, "
                                 "or measure the sender's steady state under each --steady rule")
    ap.add_argument("summaries", nargs="*", type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steady", action="append", default=[], metavar="RULE", help="last, cover-K-M[-fallback] or envelope")
    ap.add_argument("--windows", type=int, default=5000, help="with --steady: the link's windows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    report = json.loads(args.calibration.read_text())
    cs = report["clock_sync"]
    sync_set = SyncSet(tuple(cs["deltas"]), tuple(Fraction(r) for r in cs["rhos"]))
    if args.steady:
        d = cs["declared"]
        envelope = ClockSync(max(d["delta"]), min(Fraction(r) for r in d["rho"]))
        result = {rule: steady_state(sync_set, report["grid"]["window_buckets"], rule, envelope, args.windows,
                                     stream_seed(args.seed, rule)) for rule in args.steady}
        for rule, m in result.items():
            print(f"{rule}: {_line(m)}")
    else:
        result = pool([json.loads(p.read_text()) for p in args.summaries], sync_set)
        for s, m in result.items():
            print(f"{s}: {_line(m)}, accepted {m['accepted']} of {m['link_windows']}")
    args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
