"""The setup difftest's data: generated instances and the Python reference's answers on each, for `PousCheck.SetupDifftest`.

    python scripts/setup_difftest.py <verity checkout> --out DIR

Writes, under `DIR/difftest/`:

- `instances.jsonl`, one instance per line, and `python.jsonl`, the same line's answers from the checkout's
  `verity.experimental.protocols.pous` (`schemes/p2.py`, `setup.py`, `protocol.py`) and `verity.ml.commitments.identity`:
  - `maps`: `d(z)`, `σ_K(r)` and `D(z) = (d(σ_K(d(z))) − t) mod p` by `_Field` (`sq`, `sigma`), on canonical values at
    the ladder's widths and at w = 16,448, edges among them (0, 1, p − 1, K all ones, `r ⊕ K ≥ p`);
  - `keys`: `sha512_keys(params, expand_prefix(params, salt), s, j)`, K and t, at w = 24 with c0 = 65,533 (t rejects a
    candidate about once in 256) and the ladder's widths and w = 16,448; and a salt of another length, which
    `expand_prefix` refuses;
  - `decode`: `decode_segment` of encoded segments (`encode_segment`) and of each mutation below: the payload, or the kind
    of `ValueError`;
  - `commit` and `tagged`: `setup.commit(params, W)` with the manifest, and `tagged_sha512(tag, payload)` with plain
    SHA-512 of the payload, which Lean absorbs in the given pieces;
  - `setup`: `protocol.verifier_key(P2(params), commitment, salt, mode=PUBLIC, encoded=C*)`: accepted with `vk`'s root,
    or the kind of `ValueError`;
- `p2-vectors.json`, a copy of the spec's §7 vectors (`benchmarks/pous/p2_v1/vectors.json`), which Lean checks against
  §7's SHA-512, entry hashes and all.

The setup and decode instances are each width's honest encoding and its mutations: a bit flipped, a block set to p or to
all ones, a block one byte short, a block dropped, blocks or segments swapped, a segment missing or extra, an empty C*,
another salt (of the right length and not), a commitment with a byte flipped and one to another W. At w = 16,448 they are
the scheme's pinned small instance (`tests/fixtures/schemes/p2-16448-v4.json`) and one fresh encoding of 2 segments of 2
blocks, about 10 s of the generator's 15.

The generator keeps to parameters where `_Field.square`'s two folds reduce fully (`4·c0² < 2^w`): below that it is not
`a² mod p` (w = 24 with c0 = 65,533, say), and Lean's `square` is the spec's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import verity.experimental.protocols.pous
from verity.experimental.protocols.pous import protocol, setup
from verity.experimental.protocols.pous.schemes import p2 as P2
from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SPEC_VECTORS = REPO / "benchmarks" / "pous" / "p2_v1" / "vectors.json"
PINNED = REPO / "verity" / "experimental" / "protocols" / "pous" / "tests" / "fixtures" / "schemes" / "p2-16448-v4.json"

LADDER = [(64, 189), (96, 17), (128, 173), (192, 237), (256, 189), (384, 317), (512, 569), (1024, 105)]
"""The spec's reduced widths (§1), each `2^w − c0` prime and ≡ 3 mod 4, so `encode_segment`'s roots exist."""
FULL = (16448, 21065)
REJECTING = (24, 65533)
"""A width where t rejects a candidate with probability c0 / 2^w ≈ 2^-8: only for keys (its folds do not reduce)."""


def _hex(x: int, n: int) -> str:
    return x.to_bytes(n, "big").hex()


def _params(p: P2.P2Params) -> dict:
    return {"w": p.w, "c0": p.c0, "blocks_per_segment": p.blocks_per_segment, "segments": p.segments}


def _kind(e: ValueError) -> str:
    """The kind of a `ValueError` from `decode_segment`, `expand_prefix` or `verifier_key_from_encoding`, as Lean's
    `Setup.Refusal.kind` names it."""
    m = str(e)
    for needle, kind in [("C* has more than", "segments"), ("segments, the parameters say", "segments"),
                         ("is outside 0..", "segments"), ("of C* is not", "shape"), ("a segment of C is", "shape"),
                         ("the salt is", "salt"), ("is not below p", "noncanonical"), ("decodes to more than", "payload"),
                         ("does not match the commitment", "commitment")]:
        if needle in m:
            return kind
    if re.fullmatch(r"block \d+ is not \d+ bytes", m):
        return "width"
    raise ValueError(f"an unclassified refusal: {m}")


def maps(rng: random.Random, w: int, c0: int, count: int):
    f, n = P2._field(P2.P2Params(w, c0, 1)), w // 8
    p, ones = f.p, (1 << w) - 1
    edges = [(0, 0, 0), (1, ones, 0), (p - 1, ones, p - 1), (p - 1, 0, 1), (0, ones, 5), ((p - 1) // 2, 1, 2)]
    for i in range(count):
        if i < len(edges):
            z, k, t = edges[i]
        else:
            z, k, t = rng.randrange(p), rng.getrandbits(w), rng.randrange(p)
        r = rng.randrange(p) if i % 3 else p - 1 - rng.randrange(c0)           # some r with r ⊕ K ≥ p
        if i % 7 == 0:
            k = ones ^ r ^ rng.randrange(c0)                                     # r ⊕ K just below 2^w: rejected
        inst = {"kind": "maps", "w": w, "c0": c0, "z": _hex(z, n), "r": _hex(r, n), "k": _hex(k, n), "t": _hex(t, n)}
        out = {"d": _hex(f.sq(z), n), "sigma": _hex(f.sigma(r, k), n),
               "decode": _hex((f.sq(f.sigma(f.sq(z), k)) - t) % p, n)}
        yield inst, out


def _t_index(params: P2.P2Params, prefix: bytes, s: int, j: int, t: int) -> int:
    """Which candidate of `stream_t` t is (the spec's `t_rejection_index`), which `sha512_keys` does not return."""
    nb, base = params.block_bytes, prefix + s.to_bytes(8, "big") + j.to_bytes(8, "big")
    a = 0
    while (x := int.from_bytes(P2.stream(base, b"t", (a + 1) * nb)[a * nb:], "big")) >= params.p:
        a += 1
    if x != t:
        raise AssertionError("sha512_keys's t is not the first candidate below p")
    return a


def keys(rng: random.Random, w: int, c0: int, count: int):
    params, n = P2.P2Params(w, c0, 1 << 16, 1 << 20), w // 8
    for i in range(count):
        salt = rng.randbytes(24 if i % 50 else 32)
        s, j = rng.randrange(1 << 20), rng.randrange(1 << 16)
        inst = {"kind": "keys", "w": w, "c0": c0, "salt": salt.hex(), "s": s, "j": j}
        try:
            prefix = P2.expand_prefix(params, salt)
            k, t = P2.sha512_keys(params, prefix, s, j)
            out = {"K": _hex(k, n), "t": _hex(t, n), "t_index": _t_index(params, prefix, s, j, t)}
        except ValueError as e:
            out = {"error": _kind(e)}
        yield inst, out


def honest(rng: random.Random, params: P2.P2Params) -> tuple[bytes, bytes, list[list[bytes]]]:
    w, salt = rng.randbytes(params.w_bytes), rng.randbytes(24)
    return w, salt, [P2.encode_segment(params, None, salt, s, w[s * params.segment_bytes:(s + 1) * params.segment_bytes])
                     for s in range(params.segments)]


def mutations(rng: random.Random, params: P2.P2Params, w: bytes, salt: bytes, enc: list[list[bytes]]):
    """`(label, commitment, salt, C*)`: the honest setup, then each mutation of it."""
    commitment, nb, p = setup.commit(params, w), params.block_bytes, params.p

    def at(s: int, j: int, block: bytes) -> list[list[bytes]]:
        return [[block if (a, b) == (s, j) else x for b, x in enumerate(seg)] for a, seg in enumerate(enc)]

    s, j = rng.randrange(params.segments), rng.randrange(params.blocks_per_segment)
    bit = rng.randrange(8 * nb)
    flipped = (int.from_bytes(enc[s][j], "big") ^ (1 << bit)).to_bytes(nb, "big")
    other = rng.randbytes(params.w_bytes)
    yield "honest", commitment, salt, enc
    yield "flip", commitment, salt, at(s, j, flipped)
    yield "p", commitment, salt, at(s, j, p.to_bytes(nb, "big"))
    yield "ones", commitment, salt, at(s, j, b"\xff" * nb)
    yield "short", commitment, salt, at(s, j, enc[s][j][1:])
    yield "dropped", commitment, salt, [seg[:-1] if a == s else seg for a, seg in enumerate(enc)]
    if params.blocks_per_segment > 1:
        yield "swap-blocks", commitment, salt, [seg[::-1] if a == s else seg for a, seg in enumerate(enc)]
    if params.segments > 1:
        yield "swap-segments", commitment, salt, enc[::-1]
    yield "missing", commitment, salt, enc[:-1]
    yield "extra", commitment, salt, enc + [enc[0]]
    yield "empty", commitment, salt, []
    yield "salt", commitment, rng.randbytes(24), enc
    yield "salt-32", commitment, salt + bytes(8), enc
    yield "commitment", bytes([commitment[0] ^ 1]) + commitment[1:], salt, enc
    yield "other-w", setup.commit(params, other), salt, enc


def setups(rng: random.Random, params: P2.P2Params, w: bytes, salt: bytes, enc: list[list[bytes]]):
    scheme = P2.P2(params)
    for label, commitment, s_, c in mutations(rng, params, w, salt, enc):
        inst = {"kind": "setup", "label": label, "params": _params(params), "commitment": commitment.hex(),
                "salt": s_.hex(), "encoded": [[b.hex() for b in seg] for seg in c]}
        try:
            vk = protocol.verifier_key(scheme, commitment, s_, mode=protocol.PUBLIC, encoded=c)
            out = {"verdict": "ok", "root": vk.root.hex()}
        except ValueError as e:
            out = {"error": _kind(e)}
        yield inst, out
        for seg in range(len(c)) if label in ("honest", "flip", "p", "ones", "short", "dropped", "salt-32") else []:
            dinst = {"kind": "decode", "label": label, "params": _params(params), "salt": s_.hex(), "s": seg,
                     "blocks": [b.hex() for b in c[seg]]}
            try:
                dout = {"payload": P2.decode_segment(params, None, s_, seg, c[seg]).hex()}
            except ValueError as e:
                dout = {"error": _kind(e)}
            yield dinst, dout
    try:
        P2.decode_segment(params, None, salt, params.segments, enc[0])
        raise AssertionError("decode_segment decoded a segment past the parameters' count")
    except ValueError as e:
        yield ({"kind": "decode", "label": "outside", "params": _params(params), "salt": salt.hex(),
                "s": params.segments, "blocks": [b.hex() for b in enc[0]]}, {"error": _kind(e)})


def commits(rng: random.Random, count: int):
    for i in range(count):
        w, c0 = rng.choice(LADDER + [FULL])
        params = P2.P2Params(w, c0, rng.randint(1, 5), rng.randint(1, 3))
        data = rng.randbytes(params.w_bytes)
        yield ({"kind": "commit", "params": _params(params), "w": data.hex()},
               {"commitment": setup.commit(params, data).hex(), "manifest": canonical_json_bytes(params.manifest()).decode()})
        tag = rng.choice(["verity-pous/w-commitment/v2", "t", "verity/x" * rng.randint(1, 40)])
        payload = rng.randbytes(rng.choice([0, 1, 111, 112, 127, 128, 129, 255, 256, rng.randrange(2000)]))
        cuts = sorted(rng.randrange(len(payload) + 1) for _ in range(rng.randrange(4)))
        pieces = [b - a for a, b in zip([0] + cuts, cuts + [len(payload)])]
        yield ({"kind": "tagged", "tag": tag, "payload": payload.hex(), "pieces": pieces},
               {"digest": tagged_sha512(tag, payload).hex(), "sha512": hashlib.sha512(payload).hexdigest()})


def instances():
    rng = random.Random(20261007)
    for w, c0 in LADDER + [FULL]:
        yield from maps(rng, w, c0, 40 if w < 16448 else 12)
    yield from keys(rng, *REJECTING, 2048)
    for w, c0 in LADDER + [FULL]:
        yield from keys(rng, w, c0, 60 if w < 16448 else 8)
    yield from commits(rng, 60)
    for w, c0 in LADDER:
        for _ in range(3):
            params = P2.P2Params(w, c0, rng.randint(1, 4), rng.randint(1, 3))
            yield from setups(rng, params, *honest(rng, params))
    pinned = json.loads(PINNED.read_text())["small"]
    params = P2.P2Params(**{k: pinned["params"][k] for k in ("w", "c0", "blocks_per_segment", "segments")})
    enc = [[bytes.fromhex(b) for b in seg] for seg in pinned["blocks"]]
    salt = bytes.fromhex(pinned["salt"])
    w = hashlib.shake_256(b"verity-pous/vectors/scheme/p2-16448/v4/w").digest(params.w_bytes)   # pous_vectors._bytes
    yield from setups(rng, params, w, salt, enc)
    params = P2.P2Params(*FULL, 2, 2)
    yield from setups(rng, params, *honest(rng, params))


def line(x: dict) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n"


def write(out: Path) -> dict[str, Path]:
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl",
             "difftest/p2-vectors.json": d / "p2-vectors.json"}
    count = 0
    with open(paths["difftest/instances.jsonl"], "w") as fi, open(paths["difftest/python.jsonl"], "w") as fp:
        for inst, ans in instances():
            fi.write(line(inst))
            fp.write(line(ans))
            count += 1
    shutil.copyfile(SPEC_VECTORS, paths["difftest/p2-vectors.json"])
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.experimental.protocols.pous is the reference")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    here = Path(verity.experimental.protocols.pous.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.experimental.protocols.pous comes from {here}, not from the checkout {a.verity}")
    for name, path in write(a.out).items():
        print(f"{name}: {sum(1 for _ in open(path)) if name.endswith('.jsonl') else path.stat().st_size} "
              f"{'lines' if name.endswith('.jsonl') else 'bytes'}")


if __name__ == "__main__":
    main()
