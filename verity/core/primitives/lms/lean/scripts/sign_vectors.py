"""The sign difftest's data: LMS keys from seeds and their signatures, and `lms.py`'s results on each, for `Lms/SignDifftest.lean`.

    python scripts/sign_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/sign/instances.jsonl`, one instance per line as `Lms.SignDifftest.evaluate` reads it (a parameter set (N, W,
H, toy), how the key is made, `keygen` from a seed or `key` from an identifier and a seed, and the leaves and messages to
sign at), and `DIR/sign/python.jsonl`, the same line's results from the checkout's `verity.ml.crypto.lms`: whether
`Params` admits the set, the key's public key (`null` where `keygen` or `key` refuses its seed or identifier), and each
`sign_at`'s signature, encoded (`null` where it refuses the leaf), with `verify`'s verdict on it. The keys are at N = 64
(W = 1, 2, 4, heights 1 to 3) and at `TOY_N` (every W, heights 1 to 4), from seeds and identifiers of the right length
and of others, and at parameter sets `Params` refuses; the leaves are every leaf of the key or a sample, and 2^H and past
it; the messages are 0 to 100 bytes and certificate messages. The Lean audit runs `Lms/SignDifftestMain.lean` on both
files (`runs` in `lean-audit.json`, which pins their SHA-256), and `verity/ml/commitments/tests/
test_lms_sign_lean_difftest.py` checks that this script still writes them.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import verity.ml.crypto
from verity.ml.crypto import lms

COUNT = 80
#: (N, W, toy) by weight, and each one's heights: a key's tree takes 2^H · P · (2^W − 1) hashes, and the audit runs the
#: difftest interpreted, so the tall trees are the toy N's, and W = 8 is the toy N's only (at N = 64 its tree is
#: 2^H · 66 · 255 hashes; `Lms.Difftest` verifies at N = 64, W = 8)
SETS = [((64, 1, False), 8), ((64, 2, False), 6), ((64, 4, False), 3), ((8, 1, True), 6), ((8, 2, True), 6),
        ((8, 4, True), 6), ((8, 8, True), 3)]
HEIGHTS = {(64, 1): [1, 2, 3], (64, 2): [1, 2], (64, 4): [1], (8, 1): [1, 2, 3, 4], (8, 2): [1, 2, 3, 4], (8, 4): [1, 2, 3],
           (8, 8): [1, 2]}
INVALID = [(32, 1, 2, False), (8, 1, 2, False), (64, 3, 2, False), (64, 1, 0, False), (16, 2, 2, True)]


def _maybe(f, *args):
    try:
        return f(*args)
    except ValueError:
        return None


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    if rng.random() < 0.05:
        n, w, h, toy = rng.choice(INVALID)
    else:
        (n, w, toy), = rng.choices([s for s, _ in SETS], [k for _, k in SETS])
        h = rng.choice(HEIGHTS[n, w])
    make = rng.choice(["keygen", "keygen", "key"])
    size = n if rng.random() < 0.92 else rng.choice([0, n - 1, n + 1, 64 if n != 64 else 8])
    ident = 16 if rng.random() < 0.92 else rng.choice([0, 15, 17])
    top = 2 ** h
    leaves = list(range(top)) if top <= 4 or rng.random() < 0.4 else rng.sample(range(top), 3)
    leaves += rng.sample([top, top + 1, 2 ** 32 - 1], rng.randint(0, 2))
    signs = []
    for q in leaves:
        if rng.random() < 0.3:
            message = lms.certificate_message(rng.randbytes(16), rng.randbytes(lms.SESSION_BYTES),
                                              rng.choice(["responses", "requests", ""]),
                                              rng.choice(lms.DIRECTIONS), rng.randrange(2 ** 64), rng.randbytes(64))
        else:
            message = rng.randbytes(rng.choice([0, 1, 31, 64, 100]))
        signs.append([q, message.hex()])
    return {"params": [n, w, h, toy], "make": make, "seed": rng.randbytes(size).hex(),
            "I": rng.randbytes(ident).hex() if make == "key" else None, "signs": signs}


def python_results(inst: dict) -> dict:
    n, w, h, toy = inst["params"]
    p = _maybe(lambda: lms.Params(n, w, h, toy=toy))
    if p is None:
        return {"params": False, "public": None, "signatures": None}
    seed = bytes.fromhex(inst["seed"])
    key = _maybe(lms.keygen, p, seed) if inst["make"] == "keygen" else _maybe(lms.key, p, bytes.fromhex(inst["I"]), seed)
    if key is None:
        return {"params": True, "public": None, "signatures": None}
    public, out = lms.public_key(key), []
    for q, message in inst["signs"]:
        m = bytes.fromhex(message)
        sig = _maybe(lms.sign_at, key, q, m)
        out.append(None if sig is None else [sig.encode().hex(), lms.verify(p, public, m, sig)])
    return {"params": True, "public": public.hex(), "signatures": out}


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/sign``."""
    d = out / "sign"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"sign/instances.jsonl": d / "instances.jsonl", "sign/python.jsonl": d / "python.jsonl"}
    with open(paths["sign/instances.jsonl"], "w") as fi, open(paths["sign/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst = instance(seed)
            fi.write(line(inst))
            fp.write(line(python_results(inst)))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.ml.crypto.lms is tested")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.ml.crypto.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.ml.crypto comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
