"""The difftest's data: random instances, and `certificate.py`'s results on each, for `NetworkCheck/Difftest.lean`.

    python scripts/difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/difftest/instances.jsonl`, one instance per line as `NetworkCheck.Difftest.evaluate` reads it, and
`DIR/difftest/python.jsonl`, the same line's results from the checkout's `verity.core.protocols.certifier.certificate`
(and `verity.ml.commitments.identity`, `verity.ml.crypto.lms` through it), with `null` where Python raises `ValueError`.
The Lean audit runs `NetworkCheck/DifftestMain.lean` on both files (`runs` in `lean-audit.json`, which pins their SHA-256),
and `verity/core/protocols/certifier/tests/test_network_check_lean_difftest.py` checks that this script still writes
them.

An instance has grid parameters (T·r from 1, so D = 0, to past 32, B up to 300 bytes, so a leaf's frame spans blocks,
τ_b and Q up to and past 2^32), Σ_sync or none (deltas and ρ large, integer and fractional), and a grid that is
left-packed or not, with a row of another length, another row count or a frame of another size. Its queries: the grid's
real frames or another list (too many, a frame of another size), every k to open and some out of range, `verify_frame`
on each opening and on tampered ones (a byte of the frame, the root or a sibling flipped, another k, k past 2^D or
negative, a path one short or one long, a frame of another size), heads as `record_head` gives them and with a field out
of range (counts, the window, T, τ_b, a digest's length, a link past 2^16 bytes), links with non-ASCII and control
characters, signed digests (`certificate_digest`) of the record's digest under drawn salts, and of other lengths, and
certificate messages over the signed digest with every field valid and each made invalid (sessions of 31 and 33 bytes
among them).
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier import certificate as cert
from verity.core.protocols.certifier.certificate import Direction, RecordHead
from verity.core.protocols.certifier.grid import GridParams, Reason, Status
from verity.core.protocols.certifier.schedule import SyncSet
from verity.ml.commitments.identity import canonical_json_bytes

COUNT = 2_000
REASONS = [None] + [r.value for r in Reason]
LINKS = ["eth0", "", "uplink/0", "é-link", "链路", "🛰️sat", "tab\there", "nul\x00", "q\"uote\\", "x" * 300]


def _maybe(f, *args):
    """``f(*args)``, or None where Python raises `ValueError`."""
    try:
        return f(*args)
    except ValueError:
        return None


def _hex(b: bytes | None) -> str | None:
    return None if b is None else b.hex()


def _params(rng: random.Random) -> GridParams:
    shape = rng.random()
    if shape < 0.1:
        buckets, allocation = 1, 1
    elif shape < 0.2:
        buckets, allocation = rng.choice([(2, 4), (4, 4), (8, 2), (16, 2), (4, 8)])
    elif shape < 0.25:
        buckets, allocation = rng.randint(9, 17), rng.randint(2, 3)
    else:
        buckets, allocation = rng.randint(1, 6), rng.randint(1, 4)
    frame_bytes = rng.choice([1, 2, 3, 8, 16, 64, 89, 90, 91, 128, 300]) if rng.random() < 0.3 else rng.randint(1, 12)
    bucket_ms = rng.choice([1, 100, 2 ** 32 - 1, 2 ** 32]) if rng.random() < 0.2 else rng.randint(1, 1000)
    queue = rng.choice([1, 1 << 16, 2 ** 32, 2 ** 70]) if rng.random() < 0.3 else rng.randint(1, 1 << 20)
    return GridParams(buckets, allocation, frame_bytes, bucket_ms, queue)


def _sync_set(rng: random.Random) -> SyncSet | None:
    if rng.random() < 0.3:
        return None
    deltas = rng.sample(range(50), rng.randint(1, 3)) + ([10 ** 20] if rng.random() < 0.1 else [])
    rhos = {Fraction(rng.randint(1, 12), rng.randint(1, 12)) for _ in range(rng.randint(1, 4))}
    if rng.random() < 0.1:
        rhos.add(Fraction(10 ** 25 + 1, 3))
    return SyncSet(tuple(deltas), tuple(sorted(rhos, key=lambda r: rng.random())))


def _grid(rng: random.Random, p: GridParams) -> list[list[bytes | None]]:
    """A grid of T rows of r slots, left-packed, or, now and then, not, with a row of another length, another row count
    or a frame of another size."""
    grid = []
    for _ in range(p.buckets):
        n = rng.randint(0, p.allocation) if rng.random() < 0.7 else rng.choice([0, p.allocation])
        grid.append([rng.randbytes(p.frame_bytes) for _ in range(n)] + [None] * (p.allocation - n))
    fault = rng.random()
    if fault < 0.04 and p.allocation > 1:
        b, s = rng.randrange(p.buckets), rng.randrange(1, p.allocation)
        grid[b] = [None] * s + [rng.randbytes(p.frame_bytes)] + [None] * (p.allocation - s - 1)
    elif fault < 0.06:
        rng.choice(grid).append(None if rng.random() < 0.5 else rng.randbytes(p.frame_bytes))
    elif fault < 0.08:
        if rng.random() < 0.5 and len(grid) > 1:
            grid.pop()
        else:
            grid.append([None] * p.allocation)
    elif fault < 0.10:
        full = [row for row in grid if row[0] is not None]
        if full:
            rng.choice(full)[0] = rng.randbytes(p.frame_bytes + rng.choice([-1, 1]) if p.frame_bytes > 1 else 2)
    return grid


def _frames(rng: random.Random, p: GridParams, real: list[bytes] | None) -> list[bytes]:
    """The grid's real frames, or another list: random, as many as the window has slots, one more, or with a frame of
    another size."""
    roll = rng.random()
    slots = p.buckets * p.allocation
    if real is not None and roll < 0.8:
        return list(real)
    if roll < 0.88 and slots <= 64:
        frames = [rng.randbytes(p.frame_bytes) for _ in range(slots + (rng.random() < 0.5))]
    else:
        frames = [rng.randbytes(p.frame_bytes) for _ in range(rng.randint(0, min(slots, 12)))]
    if frames and rng.random() < 0.3:
        frames[rng.randrange(len(frames))] = rng.randbytes(max(0, p.frame_bytes + rng.choice([-1, 1])))
    return frames


def _flip(rng: random.Random, b: bytes) -> bytes:
    if not b:
        return b"\x00"
    i = rng.randrange(len(b))
    return b[:i] + bytes([b[i] ^ (1 << rng.randrange(8))]) + b[i + 1:]


def _verifies(rng: random.Random, p: GridParams, frames: list[bytes], root: bytes | None,
              opened: dict[int, tuple[bytes, ...]]) -> list[list]:
    """`verify_frame` queries: each opening as given, and tampered ones."""
    d = cert.depth(p)
    out = []
    for k, path in opened.items():
        frame, path = frames[k], list(path)
        out.append([root, k, frame, path])
        kind = rng.randrange(10)
        if kind == 0:
            out.append([root, k, _flip(rng, frame), path])
        elif kind == 1:
            out.append([_flip(rng, root), k, frame, path])
        elif kind == 2 and path:
            i = rng.randrange(len(path))
            out.append([root, k, frame, path[:i] + [_flip(rng, path[i])] + path[i + 1:]])
        elif kind == 3:
            out.append([root, rng.choice([k ^ 1, k + 1, max(k - 1, 0), (k + 1) % max(len(frames), 1)]), frame, path])
        elif kind == 4:
            out.append([root, rng.choice([k + (1 << d), -1, -(1 << d), 2 ** 64 + k]), frame, path])
        elif kind == 5:
            out.append([root, k, frame, path[:-1] if path and rng.random() < 0.5 else path + [cert.empty(d)]])
        elif kind == 6:
            out.append([root, k, frame[:-1] if len(frame) > 1 else frame + b"\x00", path])
        elif kind == 7:
            out.append([root, k, frame, path[::-1]])
    if rng.random() < 0.3:
        out.append([rng.randbytes(64), rng.randrange(1 << d), rng.randbytes(p.frame_bytes),
                    [rng.randbytes(64) for _ in range(d)]])
    return [[r.hex() if r is not None else rng.randbytes(64).hex(), k, f.hex(), [s.hex() for s in path]]
            for r, k, f, path in out]


def _head_json(h: dict) -> dict:
    return {"link": h["link"], "direction": h["direction"].value, "window": h["window"], "T": h["buckets"],
            "r": h["allocation"], "B": h["frame_bytes"], "tau": h["bucket_ms"], "status": h["status"],
            "deadline": h["deadline"], "parameters": h["parameters"].hex(), "counts": list(h["counts"]),
            "root": h["root"].hex()}


def _head_result(h: dict) -> list[str] | None:
    try:
        head = RecordHead(h["link"], h["direction"], h["window"], h["buckets"], h["allocation"], h["frame_bytes"],
                          h["bucket_ms"], Status(None if h["status"] is None else Reason(h["status"])), h["deadline"],
                          h["parameters"], tuple(h["counts"]), h["root"])
        return [head.encode().hex(), head.digest().hex()]
    except ValueError:
        return None


def _heads(rng: random.Random, base: dict) -> list[dict]:
    """The head as `record_head` gives it, and copies with one field changed, in range or out of it."""
    out = [dict(base)]
    for _ in range(rng.randint(1, 3)):
        h = dict(base)
        kind = rng.randrange(12)
        if kind == 0:
            h["counts"] = list(h["counts"]) + [0] if rng.random() < 0.5 else list(h["counts"])[:-1]
        elif kind == 1:
            c = list(h["counts"])
            c[rng.randrange(len(c))] = rng.choice([h["allocation"] + 1, -1])
            h["counts"] = c
        elif kind == 2:
            h["window"] = rng.choice([0, 2 ** 64 - 1, 2 ** 64, -1, 2 ** 63])
        elif kind == 3:
            h["bucket_ms"] = rng.choice([0, 2 ** 32 - 1, 2 ** 32, -1])
        elif kind == 4:
            h["frame_bytes"] = rng.choice([0, 2 ** 32 - 1, 2 ** 32])
        elif kind == 5:
            h["parameters"] = rng.randbytes(rng.choice([0, 63, 65]))
        elif kind == 6:
            h["root"] = rng.randbytes(rng.choice([0, 63, 65]))
        elif kind == 7 and rng.random() < 0.05:
            h["link"] = rng.choice(["a" * 65535, "a" * 65536, "é" * 32768, "é" * 32767])
        elif kind == 8:
            h["allocation"] = rng.choice([2 ** 32, 2 ** 32 - 1, max(h["counts"])])
        elif kind == 9:
            h["status"] = rng.choice(REASONS)
            h["deadline"] = not h["deadline"]
        elif kind == 10:
            h["direction"] = Direction.INGRESS if h["direction"] is Direction.EGRESS else Direction.EGRESS
        else:
            h["link"] = rng.choice(LINKS)
        out.append(h)
    return out


def _messages(rng: random.Random, key_id: bytes, session: bytes, link: str, direction: Direction, window: int,
              digest: bytes | None) -> list[list]:
    """`certificate_message` on the signed digest, and with one field changed, in range or out of it: a session of 31
    or 33 bytes among them, which Python and Lean must both refuse."""
    digest = digest if digest is not None else rng.randbytes(64)
    out = [[key_id, session, link, direction, window, digest]]
    for _ in range(rng.randint(0, 2)):
        m = list(out[0])
        kind = rng.randrange(6)
        if kind == 0:
            m[0] = rng.randbytes(rng.choice([0, 15, 17, 32]))
        elif kind == 1:
            m[5] = rng.randbytes(rng.choice([0, 32, 63, 65]))
        elif kind == 2:
            m[4] = rng.choice([0, 2 ** 64 - 1, 2 ** 64, -1])
        elif kind == 3:
            m[2] = rng.choice(LINKS)
        elif kind == 4:
            m[3] = Direction.INGRESS if m[3] is Direction.EGRESS else Direction.EGRESS
        else:
            m[1] = rng.randbytes(rng.choice([0, 31, 33, 32, 64]))
        out.append(m)
    return out


def instance(seed: int) -> tuple[dict, dict]:
    """Instance `seed` as `NetworkCheck.Difftest.evaluate` reads it, and the Python's results on it."""
    rng = random.Random(seed)
    p, ss = _params(rng), _sync_set(rng)
    grid = _grid(rng, p)
    link = rng.choice(LINKS) if rng.random() < 0.3 else f"link{rng.randrange(100)}"
    direction = rng.choice(list(Direction))
    window = rng.choice([0, 1, 2 ** 32, 2 ** 64 - 1]) if rng.random() < 0.2 else rng.randrange(10_000)
    status = rng.choice(REASONS) if rng.random() < 0.4 else None
    deadline = rng.random() < 0.8
    key_id = rng.randbytes(16)
    session = rng.randbytes(32)
    grid_t = tuple(tuple(row) for row in grid)
    real = _maybe(cert.real_frames, grid_t, p)
    frames = _frames(rng, p, real)
    root = _maybe(cert.frames_root, frames, p)
    ks = sorted(set(rng.sample(range(-1, len(frames) + 1), min(len(frames) + 2, 3))))
    opens = [_maybe(cert.open_frame, frames, p, k) for k in ks]
    opened = {k: path for k, path in zip(ks, opens, strict=True) if path is not None}
    verifies = _verifies(rng, p, frames, root, opened)
    parameters = cert.parameters_digest(p, ss)
    head = _maybe(lambda: cert.record_head(link, direction, window, grid_t, Status(None if status is None else
                                                                                   Reason(status)),
                                           p, parameters, deadline=deadline))
    head_result = None if head is None else _maybe(lambda: [head.encode().hex(), head.digest().hex()])
    base = {"link": link, "direction": direction, "window": window, "buckets": p.buckets, "allocation": p.allocation,
            "frame_bytes": p.frame_bytes, "bucket_ms": p.bucket_ms, "status": status, "deadline": deadline,
            "parameters": parameters, "counts": list(head.counts) if head else [0] * p.buckets,
            "root": head.root if head else rng.randbytes(64)}
    heads = _heads(rng, base)
    record = bytes.fromhex(head_result[1]) if head_result is not None else rng.randbytes(64)
    digests = [[rng.randbytes(cert.SALT_BYTES), record]] + [
        [rng.randbytes(rng.choice([0, 63, 64, 65])), rng.randbytes(rng.choice([0, 64, 65]))]
        for _ in range(rng.randint(0, 2))]
    signed = cert.certificate_digest(*digests[0]) if head_result is not None else None
    messages = _messages(rng, key_id, session, link, direction, window, signed)
    nodes = [[rng.randbytes(rng.choice([0, 1, 63, 64, 65, 130])), rng.randbytes(rng.choice([0, 64, 200]))]
             for _ in range(rng.randint(0, 2))]
    manifest = {
        "buckets": p.buckets, "allocation": p.allocation, "frame_bytes": p.frame_bytes, "bucket_ms": p.bucket_ms,
        "queue_capacity": p.queue_capacity,
        "sync_set": None if ss is None else {"deltas": list(ss.deltas), "rhos": [str(r) for r in ss.rhos]}}
    inst = {
        "params": {"T": p.buckets, "r": p.allocation, "B": p.frame_bytes, "tau": p.bucket_ms, "Q": p.queue_capacity},
        "sync_set": None if ss is None else {"deltas": list(ss.deltas),
                                             "rhos": [[r.numerator, r.denominator] for r in ss.rhos]},
        "grid": [[_hex(x) for x in row] for row in grid], "frames": [f.hex() for f in frames],
        "link": link, "direction": direction.value, "window": window, "status": status, "deadline": deadline,
        "opens": ks, "verifies": verifies, "heads": [_head_json(h) for h in heads],
        "digests": [[salt.hex(), r.hex()] for salt, r in digests],
        "messages": [[i.hex(), s.hex(), l, d.value, w, g.hex()] for i, s, l, d, w, g in messages],
        "nodes": [[a.hex(), b.hex()] for a, b in nodes]}
    d = cert.depth(p)
    results = {
        "depth": d, "empty": [cert.empty(lv).hex() for lv in range(d + 1)],
        "real_frames": None if real is None else [f.hex() for f in real],
        "leaves": [_hex(_maybe(cert.frame_leaf, f, p.frame_bytes)) for f in frames], "root": _hex(root),
        "opens": [None if path is None else [s.hex() for s in path] for path in opens],
        "verifies": [cert.verify_frame(bytes.fromhex(r), p, k, bytes.fromhex(f), [bytes.fromhex(s) for s in path])
                     for r, k, f, path in verifies],
        "manifest": canonical_json_bytes(manifest).decode(), "parameters": parameters.hex(),
        "record_head": head_result, "heads": [_head_result(h) for h in heads],
        "digests": [cert.certificate_digest(salt, r).hex() for salt, r in digests],
        "messages": [_hex(_maybe(cert.certificate_message, i, s, l, d_, w, g)) for i, s, l, d_, w, g in messages],
        "nodes": [cert.node(a, b).hex() for a, b in nodes]}
    return inst, results


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/difftest``."""
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    with (open(paths["difftest/instances.jsonl"], "w", encoding="utf-8") as fi,
          open(paths["difftest/python.jsonl"], "w", encoding="utf-8") as fp):
        for seed in range(count):
            inst, results = instance(seed)
            fi.write(line(inst))
            fp.write(line(results))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.core.protocols.certifier is tested")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.core.protocols.certifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.core.protocols.certifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
