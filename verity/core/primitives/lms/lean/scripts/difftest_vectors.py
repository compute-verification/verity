"""The difftest's data: LMS keys, signatures and tampered ones, and `lms.py`'s results on each, for `Lms/Difftest.lean`.

    python scripts/difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/difftest/instances.jsonl`, one instance per line as `Lms.Difftest.evaluate` reads it (a parameter set
(N, W, H, toy), a public key, a message and a signature's bytes), and `DIR/difftest/python.jsonl`, the same line's
results from the checkout's `verity.ml.crypto.lms`: whether `Params` admits the set, whether `Signature.decode` reads the
signature, the `digits` of its message hash, `root_of`, and `verify`, with `null` where Python raises `ValueError`. The
Lean audit runs `Lms/DifftestMain.lean` on both files (`runs` in `lean-audit.json`, which pins their SHA-256), and
`verity/ml/commitments/tests/test_lms_lean_difftest.py` checks that this script still writes them.

The keys are `keygen`'s, at every W, at N = 64 and at `TOY_N`, of heights 1 to 5. At heights 11, 20 (`DEPLOYED`'s) and 30,
whose keygen takes minutes to hours, a key's tree is a stand-in: `sign_at`'s one-time signature is the seed's, its path
nodes are random, and the public key is the root they give, which is all `verify` reads. The signatures are `sign_at`'s,
on messages of 0 to 100 bytes and on certificate messages, then tampered: a bit of C, of a chain value, of a path node or
of the message flipped, another leaf, a leaf past 2^H, another key's or another message's signature, a signature one byte
short or long, a public key with a bit flipped or of another length, and parameter sets `Params` refuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import verity.ml.crypto
from verity.ml.crypto import lms

COUNT = 600
#: (N, W, toy) of every parameter set `Params` admits, by weight: verification at N = 64, W = 8 takes about 8,000 hashes,
#: so it is sampled least
SETS = [((64, 1, False), 30), ((64, 2, False), 12), ((64, 4, False), 10), ((64, 8, False), 1), ((8, 1, True), 8),
        ((8, 2, True), 6), ((8, 4, True), 6), ((8, 8, True), 4)]
HEIGHTS = {8: [1, 2], 4: [1, 2, 3], 2: [1, 2, 3, 5], 1: [1, 2, 3, 5]}
TALL = [20, 20, 20, 11, 30]
INVALID = [(32, 1, 4, False), (8, 1, 4, False), (64, 3, 4, False), (64, 1, 0, False), (64, 1, 31, False),
           (16, 2, 2, True), (64, 16, 1, False)]
TAMPER = ["none", "none", "none", "C", "y", "path", "message", "q", "q-past", "other-key", "other-message", "short",
          "long", "pk-bit", "pk-length", "params"]


def _maybe(f, *args):
    try:
        return f(*args)
    except ValueError:
        return None


class _StandIn:
    """A tree's nodes as `sign_at` reads them (`tree[r ^ 1]`): random, from the key's seed and r."""

    def __init__(self, seed: bytes, n: int) -> None:
        self.seed, self.n = seed, n

    def __getitem__(self, r: int) -> bytes:
        return hashlib.sha512(b"stand-in" + self.seed + r.to_bytes(8, "big")).digest()[:self.n]


@lru_cache(maxsize=None)
def _key(n: int, w: int, h: int, toy: bool, which: int) -> tuple[lms.PrivateKey, bytes | None]:
    """Key `which` of the set: `keygen`'s below height 6, else one whose tree is a stand-in (its public key is None until
    a signature gives its root)."""
    p = lms.Params(n, w, h, toy=toy)
    seed = hashlib.sha512(f"lms-difftest/{n}/{w}/{h}/{toy}/{which}".encode()).digest()[:n]
    if h <= 5:
        k = lms.keygen(p, seed)
        return k, lms.public_key(k)
    I = hashlib.sha512(b"verity/lms/v0/I" + seed).digest()[:16]
    return lms.PrivateKey(p, I, seed, _StandIn(seed, n)), None  # type: ignore[arg-type]


def _sign(key: lms.PrivateKey, public: bytes | None, q: int, message: bytes) -> tuple[bytes, bytes]:
    """(public key, signature) of `message` at leaf q; a stand-in's public key is the root its signature gives."""
    sig = lms.sign_at(key, q, message)
    if public is None:
        public = key.I + lms.root_of(key.params, key.I, message, sig)
    return public, sig.encode()


def _flip(rng: random.Random, b: bytes, lo: int = 0, hi: int | None = None) -> bytes:
    if not b:
        return b"\x01"
    i = rng.randrange(lo, len(b) if hi is None else hi)
    return b[:i] + bytes([b[i] ^ (1 << rng.randrange(8))]) + b[i + 1:]


def _message(rng: random.Random) -> bytes:
    if rng.random() < 0.3:
        return lms.certificate_message(rng.randbytes(16), rng.randbytes(lms.SESSION_BYTES), f"link{rng.randrange(9)}",
                                       rng.choice(lms.DIRECTIONS), rng.randrange(1 << 20), rng.randbytes(64))
    return rng.randbytes(rng.choice([0, 1, 31, 32, 64, 100]))


def instance(seed: int) -> dict:
    """Instance `seed`: a parameter set, a public key, a message and a signature's bytes."""
    rng = random.Random(seed)
    (n, w, toy), = rng.choices([s for s, _ in SETS], [x for _, x in SETS])
    tall = rng.random() < 0.15
    h = rng.choice(TALL) if tall else rng.choice(HEIGHTS[w])
    if tall:
        n, w, toy = (64, 1, False) if h == 20 and rng.random() < 0.7 else (n, w, toy)
    key, public = _key(n, w, h, toy, rng.randrange(2))
    p = key.params
    q = rng.choice([0, 2 ** h - 1]) if rng.random() < 0.3 else rng.randrange(2 ** h)
    message = _message(rng)
    public, sig = _sign(key, public, q, message)
    params = [n, w, h, toy]
    tamper = rng.choice(TAMPER)
    off_y, off_path = 4 + n, 4 + n * (1 + p.P)
    if tamper == "C":
        sig = _flip(rng, sig, 4, off_y)
    elif tamper == "y":
        sig = _flip(rng, sig, off_y, off_path)
    elif tamper == "path":
        sig = _flip(rng, sig, off_path)
    elif tamper == "message":
        message = _flip(rng, message) if rng.random() < 0.7 else message + b"\x00"
    elif tamper == "q":
        sig = ((q + rng.randrange(1, 2 ** h)) % 2 ** h if h > 1 else 1 - q).to_bytes(4, "big") + sig[4:]
    elif tamper == "q-past":
        sig = rng.choice([2 ** h + q, 2 ** 32 - 1, 2 ** h]).to_bytes(4, "big") + sig[4:]
    elif tamper == "other-key":
        other, other_public = _key(n, w, h, toy, 2)
        sig = _sign(other, other_public, q, message)[1]
    elif tamper == "other-message":
        sig = _sign(key, public, q, message + b"!")[1]
    elif tamper == "short":
        sig = sig[:-1]
    elif tamper == "long":
        sig = sig + b"\x00"
    elif tamper == "pk-bit":
        public = _flip(rng, public)
    elif tamper == "pk-length":
        public = rng.choice([public[:-1], public + b"\x00", public[:16], b""])
    elif tamper == "params":
        params = list(rng.choice(INVALID))
    return {"params": params, "public": public.hex(), "message": message.hex(), "signature": sig.hex(), "tamper": tamper}


def python_results(inst: dict) -> dict:
    """`lms.py`'s results on the instance, in the shape `Lms.Difftest.evaluate` gives them."""
    n, w, h, toy = inst["params"]
    public, message = bytes.fromhex(inst["public"]), bytes.fromhex(inst["message"])
    out = {"params": True, "decode": None, "digits": None, "root": None, "verify": None}
    try:
        p = lms.Params(n, w, h, toy=toy)
    except ValueError:
        return out | {"params": False}
    sig = _maybe(lms.Signature.decode, p, bytes.fromhex(inst["signature"]))
    if sig is None:
        return out | {"decode": False}
    root = _maybe(lms.root_of, p, public[:16], message, sig)
    return out | {"decode": True, "digits": lms.digits(p, lms._message_hash(p, public[:16], sig.q, sig.C, message)),
                  "root": None if root is None else root.hex(), "verify": _maybe(lms.verify, p, public, message, sig)}


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/difftest``."""
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    with open(paths["difftest/instances.jsonl"], "w") as fi, open(paths["difftest/python.jsonl"], "w") as fp:
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
