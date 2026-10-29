"""The calls' difftest data: random windows of the lead's calls, and `calls.Calls`'s answers on each, for
`CallsDifftestMain.lean`.

    python scripts/calls_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/calls/instances.jsonl`, one window per line as `NetworkCertifier.Calls.CallsDifftest.rawOfJson` reads it,
and `DIR/calls/python.jsonl`, the same line's results from the checkout's `verity.core.protocols.certifier.calls`: each
event's answer (an A root's release as hex, the calls a bucket refuses, a words root's verdict, the close's verdicts,
`"refusal"` for a `CallRefusal` and `"no coins"` for an A root whose coins weren't flipped), and the window's calls as
`Calls.summary` gives them without the release's digest, whose bytes the answers hold. The Lean audit runs
`CallsDifftestMain.lean` on both files (`runs` in `lean-audit.json`, which pins their SHA-256), and
`verity/core/protocols/certifier/tests/test_network_certifier_calls_lean_difftest.py` checks that this script still
writes them.

A window is an egress cap and up to 31 events: A roots of calls 0 … 7, buckets and words roots, and in nine windows of
ten the close, which a quarter of them follow with more events. Shapes are small (most at most 40 rows, 16 columns and K 150; some up to 200 rows,
four coin tiles; some K 806 … 900, two row words a row), and now and then one the certifier refuses (a dimension 0, or
past 2^40 MMA tiles) or one at exactly 2^40 tiles, which it admits and whose flip the entropy can't hold. The
certifier's entropy is the calls' flips in order: random row bytes, G words below `G_BOUND` (now and then 0 or
`G_BOUND − 1`) with rejected words planted before them (now and then `G_BOUND` or 2^32 − 1), and random lane bytes;
now and then it ends inside a flip, after which no call's coins are flipped, and otherwise it may run on past the last.
Caps run from 0 to many bits a MAC, and a third of the buckets bring an open call's count exactly to its bits or one
past them. Refused events: a second A root of a call, a root not of 64 bytes, a bucket of more frames than its
allocation, a words root of a call not open, and any event after the close.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier.calls import G_BOUND, CallRefusal, CallShape, Calls, EgressCap

COUNT = 1_000
#: Shapes at the verifier's bound: exactly 2^40 MMA tiles (admitted) and one past (refused) at S = 2^40 and 2^40 + 1,
#: and one past at 2^29 tiles a step over 2^11 + 1 steps.
EDGES = ((16, 8, 21 + 28 * (2**40 - 1)), (16, 8, 21 + 28 * 2**40), (17, 8, 21 + 28 * (2**40 - 1)),
         (2**20, 2**16, 21 + 28 * 2**11))


class Entropy:
    """The certifier's entropy: `data`, read in order."""

    def __init__(self, data: bytes) -> None:
        self.data, self.at = data, 0

    def __call__(self, k: int) -> bytes:
        out = self.data[self.at:self.at + k]
        self.at += k
        return out


def _shape(rng: random.Random) -> tuple[int, int, int]:
    r = rng.random()
    if r < 0.04:
        dims = [rng.randint(1, 40), rng.randint(1, 16), rng.randint(1, 150)]
        dims[rng.randrange(3)] = 0
        return tuple(dims)
    if r < 0.075:
        return rng.choice(EDGES)
    if r < 0.16:
        return rng.choice([64, 65, 128, 129, rng.randint(41, 200)]), rng.randint(1, 9), rng.randint(1, 60)
    if r < 0.26:
        return rng.randint(1, 20), rng.randint(1, 8), rng.choice([806, 834, 835, rng.randint(806, 900)])
    return (rng.choice([1, 15, 16, 17, rng.randint(1, 40)]), rng.choice([1, 7, 8, 9, rng.randint(1, 16)]),
            rng.choice([1, 21, 22, 49, 50, rng.randint(1, 150)]))


def _g_words(rng: random.Random, n: int) -> bytes:
    """n accepted G words, with rejected ones planted before them."""
    words = []
    for _ in range(n):
        while rng.random() < 0.08:
            words.append(rng.choice([G_BOUND, 2**32 - 1, rng.randrange(G_BOUND, 2**32)]))
        words.append(rng.choice([0, G_BOUND - 1]) if rng.random() < 0.05 else rng.randrange(G_BOUND))
    return struct.pack(f"<{len(words)}I", *words)


def _flip_bytes(rng: random.Random, shape: CallShape) -> bytes:
    """The entropy one flip of `shape` reads, exactly."""
    return (rng.randbytes(4 * shape.m * shape.H) + _g_words(rng, shape.n)
            + rng.randbytes(3 * shape.S * shape.tiles * shape.n))


def _bucket_of(rng: random.Random, bits: int) -> tuple[int, list[int]]:
    """A bucket of exactly `bits` bits: an allocation r, its r.bit_length() timing bits, and frames holding the rest."""
    if bits == 0:
        return 0, []
    r = rng.choice([1, 1, 2, 3, rng.randint(1, 64)])
    while r.bit_length() > bits:
        r //= 2
    rest = bits - r.bit_length()
    k = rng.randint(1 if rest else 0, r)
    cuts = sorted(rng.randint(0, rest) for _ in range(max(k - 1, 0)))
    return r, [b - a for a, b in zip([0, *cuts], [*cuts, rest])] if k else []


class _Window:
    """A window's events and the entropy its flips read, built as the certifier will answer them: which calls are
    registered, which are open (each one's `cap · rows · n · K`, its count and whether it is refused), whether the
    entropy has ended and whether the window is closed."""

    def __init__(self, rng: random.Random, cap: EgressCap) -> None:
        self.rng, self.cap = rng, cap
        self.events: list[list] = []
        self.chunks: list[bytes] = []
        self.ids: set[int] = set()
        self.open: dict[int, list] = {}
        self.spent = self.closed = False

    def a_root(self) -> None:
        rng = self.rng
        fresh = [c for c in range(8) if c not in self.ids]
        call = rng.choice(sorted(self.ids)) if self.ids and (not fresh or rng.random() < 0.06) else rng.choice(fresh)
        root = rng.randbytes(rng.choice([64] * 30 + [0, 63, 65]))
        rows, cols, K = _shape(rng)
        self.events.append(["a", call, root.hex(), rows, cols, K])
        if self.closed or call in self.ids or len(root) != 64:
            return
        try:
            shape = CallShape(rows, cols, K)
        except CallRefusal:
            return
        self.ids.add(call)
        if self.spent:
            return
        if shape.release_bytes > 2**20:
            self.spent = True
            return
        flip = _flip_bytes(rng, shape)
        if rng.random() < 0.04:
            self.chunks.append(flip[:rng.randrange(len(flip))])
            self.spent = True
            return
        self.chunks.append(flip)
        self.open[call] = [self.cap.cap * rows * shape.n * K, 0, False]

    def bucket(self) -> None:
        rng = self.rng
        live = [c for c, s in sorted(self.open.items()) if not s[2]]
        if live and rng.random() < 0.35:
            limit, count, _ = self.open[rng.choice(live)]
            r, frames = _bucket_of(rng, limit // self.cap.fwd_macs - count + rng.choice([0, 1]))
        else:
            r = rng.choice([0, 1, 2, 3, 7, 8, rng.randint(0, 64)])
            frames = [rng.choice([0, 1, 8, rng.randint(0, 512)]) for _ in range(rng.randint(0, r))]
            if rng.random() < 0.04:
                frames.append(rng.randint(0, 64))
        self.events.append(["b", r, frames])
        if self.closed or len(frames) > r:
            return
        for s in self.open.values():
            if not s[2]:
                s[1] += r.bit_length() + sum(frames)
                s[2] = s[1] * self.cap.fwd_macs > s[0]

    def words_root(self) -> None:
        rng = self.rng
        call = rng.choice(sorted(self.open)) if self.open and rng.random() < 0.85 else rng.randrange(8)
        root = rng.randbytes(rng.choice([64] * 30 + [0, 32, 65]))
        self.events.append(["w", call, root.hex()])
        if not self.closed and len(root) == 64:
            self.open.pop(call, None)

    def close(self) -> None:
        self.events.append(["c"])
        self.closed = True
        self.open.clear()


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    cap = EgressCap(rng.choice([0, 1, 2, 5, rng.randint(0, 50), rng.randint(0, 5000)]),
                    rng.choice([1, 1, 3, 1000, rng.randint(1, 10**6)]))
    w = _Window(rng, cap)
    for _ in range(rng.randint(0, 26)):
        rng.choices([w.a_root, w.bucket, w.words_root], [0.3, 0.5, 0.2])[0]()
    if rng.random() < 0.9:
        w.close()
        for _ in range(rng.choice([0] * 6 + [1, 4])):
            rng.choice([w.a_root, w.bucket, w.words_root, w.close])()
    if not w.spent:
        w.chunks.append(rng.randbytes(rng.choice([0, 0, 1, 7, 64])))
    return {"cap": cap, "entropy": b"".join(w.chunks), "events": w.events}


def lean_input(inst: dict) -> dict:
    return {"cap": inst["cap"].cap, "fwd_macs": inst["cap"].fwd_macs, "entropy": inst["entropy"].hex(),
            "events": inst["events"]}


def _answer(calls: Calls, event: list) -> dict | str:
    try:
        match event:
            case ["a", call, root, rows, cols, K]:
                try:
                    return {"release": calls.a_root(call, bytes.fromhex(root), CallShape(rows, cols, K)).hex()}
                except CallRefusal:
                    raise
                except ValueError:
                    return "no coins"
            case ["b", r, frames]:
                return {"refused": calls.bucket(r, frames)}
            case ["w", call, root]:
                return {"verdict": calls.words_root(call, bytes.fromhex(root)).value}
            case ["c"]:
                return {"verdicts": [[call, v.value] for call, v in calls.close().items()]}
    except CallRefusal:
        return "refusal"
    raise ValueError(f"not an event: {event!r}")


def python_results(inst: dict) -> dict:
    calls = Calls(inst["cap"], Entropy(inst["entropy"]))
    answers = [_answer(calls, ev) for ev in inst["events"]]
    return {"answers": answers, "summary": [{k: v for k, v in c.items() if k != "release"} for c in calls.summary()]}


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/calls``."""
    d = out / "calls"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"calls/instances.jsonl": d / "instances.jsonl", "calls/python.jsonl": d / "python.jsonl"}
    with open(paths["calls/instances.jsonl"], "w") as fi, open(paths["calls/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst = instance(seed)
            fi.write(line(lean_input(inst)))
            fp.write(line(python_results(inst)))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.core.protocols.certifier is audited")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.core.protocols.certifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.core.protocols.certifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
