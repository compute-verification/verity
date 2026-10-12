"""The difftest's data: instances, and the Python verifier's answers on each, for `PousCheck/DifftestMain.lean`.

    python scripts/difftest_vectors.py <verity checkout> --out DIR

Writes `DIR/difftest/instances.jsonl`, one instance per line, and `DIR/difftest/python.jsonl`, the same line's answers
from the checkout's Python, each line canonical JSON (sorted keys, no whitespace). `PousCheck.Difftest` recomputes every
answer from the instance and fails on any line that differs. The Lean audit runs it (`runs` in `lean-audit.json`, which
pins both files' SHA-256).

`merkle` instances are trees of random blocks under either leaf (the spec's `reference.leaf`, the generic protocol's
`verifier.leaf_hash`): every leaf, the root, the SHA-512 of every audit path (canonical JSON of the list), each path's
exact length, and `verify_path` (`wire.verify_path`, `verifier.verify_path`) on each leaf's honest path and on a flipped,
shortened, lengthened, swapped (leaf for sibling), odd-length or misindexed one, and on an index past the tree.

`randomness` instances are `verity.ml.randomness` derivations, streams and uniforms over random parts, and P2's audit id,
mode-0 key, indices and nonces and mode-1 round indices (`reference.audit_key`, `round_key`).

`audit` instances are sessions for `wire.verify_audit`, whose answer is its `FINAL` frame, the `WireError` code it raises,
or the `ValueError` it refuses its inputs with: honest setups (public or private, committing to their `W` or not) and
audits (mode 0 or 1, decode off, on, or on and compared with the held `W`, some blocks' payloads too wide) at the reduced
widths of `vectors.POINTS`, the verifier holding its own root of a private setup (and sometimes of a public one) and,
sometimes, `W`; each with one adversarial edit (a wrong or non-canonical block, a wrong path or path count, a non-derived
or outside index, a wrong nonce, round, epoch or audit id, a replayed challenge or response, a late or exactly-on-time
answer, a flipped, truncated or extended frame, a dropped tail or an extra frame, a wrong verifier secret, a seen pair, a
bad AUDIT_OPEN field or length, a missing or wrong beacon, a non-canonical or invalid manifest, a broken setup frame, a
wrong or missing private root, a wrong, short or missing held `W` under flag bit 1, an elapsed time past u64);
deployment-width audits (B = 2^19, k = 105) over one private store; the spec's audit vectors (§7), each checked here
against its pinned FINAL; and named sessions at edges the two once read apart (`redteam_audits`). The `sha512` instance is §7's pins of the audit files and frames, its answer the pins as the spec
states them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace


def _python(repo: Path):
    sys.path.insert(0, str(repo / "benchmarks" / "pous"))
    from p2_v1 import reference, wire
    from verity.core.protocols.pous import verifier
    return reference, wire, verifier


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _merkle(rng: random.Random) -> dict:
    kind = rng.choice(["spec", "package"])
    salt = rng.randbytes(32 if kind == "spec" else 24)
    bps = rng.choice([1, 1, 2, 3, 4, 5, 7, 8])
    segments = rng.randint(1, max(1, 24 // bps))
    width = rng.choice([1, 8, 63, 64, 111, 112, 128, 255, rng.randint(1, 300)])
    if rng.random() < 0.05:
        width, segments = 2056, 1
    blocks = [rng.randbytes(width) for _ in range(bps * segments)]
    samples = [(rng.randrange(1 << 64), rng.randrange(1 << 16), rng.randbytes(rng.randint(0, 200))) for _ in range(3)]
    return {"kind": "merkle", "leaf": kind, "salt": salt.hex(), "blocks_per_segment": bps,
            "blocks": [b.hex() for b in blocks], "samples": [[s, v, b.hex()] for s, v, b in samples], "seed": rng.randrange(1 << 32)}


def _merkle_answers(py, inst: dict) -> dict:
    reference, wire, verifier = py
    salt, bps = bytes.fromhex(inst["salt"]), inst["blocks_per_segment"]
    blocks = [bytes.fromhex(b) for b in inst["blocks"]]
    if inst["leaf"] == "spec":
        def leaf(s, v, b):
            return reference.leaf(salt, s, v, b)
        verify = wire.verify_path
    else:
        def leaf(s, v, b):
            return verifier.leaf_hash(SimpleNamespace(blocks_per_segment=1 << 16, block_bytes=len(b)), salt, s * (1 << 16) + v, b)
        verify = verifier.verify_path
    leaves = [leaf(i // bps, i % bps, b) for i, b in enumerate(blocks)]
    n = len(leaves)
    tree = verifier.MerkleTree(leaves)
    root = reference.root(leaves)
    assert root == tree.root
    paths = [tree.path(i) for i in range(n)]
    assert all(list(p) == reference.path(leaves, i) for i, p in enumerate(paths))
    rng = random.Random(inst["seed"])
    checks = []
    for i in range(n):
        p = list(paths[i])
        cases = [("honest", 0, 0)]
        if p:
            cases += [("flipped", rng.randrange(len(p)), rng.randrange(8)), ("short", 0, 0), ("swapped", 0, 0),
                      ("odd-sibling", 0, 0)]
        cases += [("long", 0, 0), ("index", rng.randrange(n + 2), 0), ("past", n + rng.randrange(3), 0)]
        for name, a, b in cases:
            j, lf, q = check_input(name, i, a, b, leaves, p, root)
            checks.append([name, i, a, b, verify(root, n, j, lf, q)])
    return {"leaves": [x.hex() for x in leaves], "root": root.hex(),
            "paths_sha512": hashlib.sha512(canonical([[s.hex() for s in p] for p in paths]).encode()).hexdigest(),
            "path_lengths": [wire.path_length(n, i) for i in range(n)], "checks": checks,
            "samples": [leaf(s, v, bytes.fromhex(b)).hex() for s, v, b in inst["samples"]]}


def check_input(name: str, i: int, a: int, b: int, leaves: list[bytes], p: list[bytes], root: bytes):
    """A check's index, leaf and path, from its name and arguments (`PousCheck.Difftest.checkInput`)."""
    if name == "flipped":
        q = list(p)
        q[a] = bytes([q[a][0] ^ (1 << b)]) + q[a][1:]
        return i, leaves[i], q
    if name == "short":
        return i, leaves[i], p[:-1]
    if name == "swapped":
        return i, p[0], [leaves[i]] + p[1:]
    if name == "odd-sibling":
        return i, leaves[i], [p[0][:63]] + p[1:]
    if name == "long":
        return i, leaves[i], p + [root]
    if name in ("index", "past"):
        return a, leaves[i], p
    return i, leaves[i], p


def _part(rng: random.Random) -> list:
    tag = rng.choice("bsi")
    if tag == "b":
        return ["b", rng.randbytes(rng.choice([0, 1, 7, 32, 64, rng.randint(0, 100)])).hex()]
    if tag == "s":
        return ["s", "".join(rng.choice("abz-/0é€\U0001f600\0") for _ in range(rng.randint(0, 12)))]
    return ["i", rng.choice([0, 1, 255, 256, 1 << 64, rng.randrange(1 << rng.randint(1, 300))])]


def _value(part: list):
    return bytes.fromhex(part[1]) if part[0] == "b" else part[1]


def _randomness(rng: random.Random) -> dict:
    word = "abcdefghijklmnopqrstuvwxyz0123456789-"
    domain = "verity/{}/{}/v{}".format("".join(rng.choice(word) for _ in range(rng.randint(1, 10))),
                                      "".join(rng.choice(word) for _ in range(rng.randint(1, 16))), rng.randint(0, 99))
    keys = sorted({"".join(rng.choice("abcAZ_é€\U0001f600") for _ in range(rng.randint(0, 6))) for _ in range(rng.randint(0, 6))})
    index = lambda: [_part(rng) for _ in range(rng.randint(0, 4))]  # noqa: E731
    sizes = [1, 2, 3, 7, 1 << 19, (1 << 64) - 1, 1 << 64, rng.randint(1, 1 << 20), rng.randrange(1, 1 << rng.randint(1, 300))]
    rounds = rng.randint(0, 6)
    return {"kind": "randomness", "source": rng.randbytes(rng.choice([1, 32, rng.randint(1, 80)])).hex(), "domain": domain,
            "context": [[k, _part(rng)] for k in keys],
            "streams": [[index(), rng.choice([0, 1, 32, 63, 64, 65, 128, rng.randint(0, 300)])] for _ in range(3)],
            "uniforms": [[rng.choice(sizes), index()] for _ in range(4)],
            "p2": {"source": rng.randbytes(32).hex(), "epoch": rng.randbytes(64).hex(), "root": rng.randbytes(64).hex(),
                   "salt": rng.randbytes(32).hex(), "counter": rng.choice([0, 1, rng.randrange(1 << 64)]),
                   "blocks": rng.choice([1, 2, 7, 1 << 19, rng.randint(1, 1 << 24)]), "rounds": rounds,
                   "beacons": [rng.randbytes(32).hex() for _ in range(rng.randint(0, 3))]}}


def _randomness_answers(py, inst: dict) -> dict:
    reference = py[0]
    from verity.ml.randomness import derive
    key = derive(bytes.fromhex(inst["source"]), inst["domain"], {k: _value(v) for k, v in inst["context"]})
    p2 = inst["p2"]
    src, epoch, root, salt = (bytes.fromhex(p2[k]) for k in ("source", "epoch", "root", "salt"))
    aid, akey = reference.audit_key(src, epoch, root, salt, p2["counter"])
    blocks = p2["blocks"]
    return {"key": key.value.hex(),
            "streams": [key.stream(tuple(_value(p) for p in i), n).hex() for i, n in inst["streams"]],
            "uniforms": [key.uniform(n, tuple(_value(p) for p in i)) for n, i in inst["uniforms"]],
            "audit_id": aid.hex(), "audit_key": akey.value.hex(),
            "indices": [akey.uniform(blocks, j) for j in range(p2["rounds"])],
            "nonces": [akey.stream(("nonce", j), 32).hex() for j in range(p2["rounds"])],
            "round_indices": [reference.round_key(bytes.fromhex(b), aid, epoch, root, j).uniform(blocks, j)
                              for j, b in enumerate(p2["beacons"])]}


# --- audits -----------------------------------------------------------------------------------------------------------------

SMALL_POINTS = ((64, 189), (96, 17), (128, 173))
"""`vectors.POINTS`' reduced widths: P2 primes, so that `reference.encode` can make honest blocks."""


def _flip(data: bytes, at: int, bit: int = 0) -> bytes:
    out = bytearray(data)
    out[at] ^= 1 << bit
    return bytes(out)


class _Session:
    """An honest setup and audit, as editable frames: `setup`, `open`, `rounds` (`[challenge, response]` per round),
    `extra`, `elapsed`, and what the verifier holds (`source`, `beacons`, `seen`, `private_root`, `w_bytes`)."""

    def __init__(self, py, rng: random.Random, params, commitment: bytes, salt: bytes, root: bytes, block, path,
                 public_blocks, mode: int, k: int, flags: int, delta: int, rtt: int):
        R, Wi = py[0], py[1]
        self.py, self.rng, self.params, self.block, self.path = py, rng, params, block, path
        self.manifest = R.canonical_json_bytes(params.manifest())
        self.commitment, self.salt, self.root = commitment, salt, root
        self.epoch = R.epoch_id(params, commitment, salt, root)
        self.public = public_blocks is not None
        self.setup = [Wi.envelope(Wi.SETUP_COMMIT, len(self.manifest).to_bytes(4, "big") + self.manifest + commitment),
                      Wi.envelope(Wi.SETUP_SALT, commitment + salt)]
        self.setup += [Wi.envelope(Wi.SETUP_BLOCK, commitment + salt + i.to_bytes(8, "big") + len(b).to_bytes(4, "big") + b)
                       for i, b in enumerate(public_blocks or [])]
        self.setup.append(self.root_frame(params.blocks, root, 1 if self.public else 0))
        self.source = rng.randbytes(rng.choice([16, 32, 64]))
        self.counter = rng.choice([0, 1, rng.randrange(1 << 64)])
        self.aid = R.audit_id(self.source, self.epoch, self.counter)
        self.mode, self.k, self.flags, self.delta, self.rtt = mode, k, flags, delta, rtt
        key = R.audit_key(self.source, self.epoch, root, salt, self.counter)[1] if mode == 0 else None
        self.beacons = None if mode == 0 else {}
        self.indices, self.nonces, self.rounds = [], [], []
        for j in range(k):
            if key is not None:
                index, nonce = key.uniform(params.blocks, j), key.stream(("nonce", j), 32)
            else:
                nonce = rng.randbytes(32)
                self.beacons[j] = nonce
                index = R.round_key(nonce, self.aid, self.epoch, root, j).uniform(params.blocks, j)
            self.indices.append(index)
            self.nonces.append(nonce)
            self.rounds.append([self.challenge(j, index, nonce), self.response(j, index, block(index), path(index))])
        self.open = self.open_frame()
        self.elapsed = [rng.randint(0, delta + rtt) for _ in range(k)]
        self.extra: list[bytes] = []
        self.seen: list[tuple[bytes, bytes]] = []
        self.private_root = None if self.public else root
        self.w_bytes: bytes | None = None

    def root_frame(self, count: int, root: bytes, mode: int) -> bytes:
        Wi = self.py[1]
        return Wi.envelope(Wi.SETUP_ROOT, self.commitment + self.salt + count.to_bytes(8, "big") + root + bytes((mode,)))

    def open_frame(self, **change) -> bytes:
        f = {"epoch": self.epoch, "aid": self.aid, "mode": self.mode, "k": self.k, "delta": self.delta, "rtt": self.rtt,
             "flags": self.flags, "counter": self.counter} | change
        return self.py[1].open_message(f["epoch"], f["aid"], f["mode"], f["k"], f["delta"], f["rtt"], f["flags"], f["counter"])

    def challenge(self, j: int, index: int, nonce: bytes, epoch: bytes | None = None, aid: bytes | None = None) -> bytes:
        return self.py[1].challenge_message(epoch or self.epoch, aid or self.aid, j, index, nonce)

    def response(self, j: int, index: int, block: bytes, path, epoch: bytes | None = None, aid: bytes | None = None) -> bytes:
        return self.py[1].response_message(epoch or self.epoch, aid or self.aid, j, index, block, path)

    def frames(self) -> list[bytes]:
        return self.setup + ([] if self.open is None else [self.open]) + [f for r in self.rounds for f in r] + self.extra

    def instance(self, name: str) -> dict:
        return {"kind": "audit", "name": name, "frames": [f.hex() for f in self.frames()], "elapsed_ns": self.elapsed,
                "verifier_source": None if self.source is None else self.source.hex(),
                "beacons": None if self.beacons is None else [[j, b.hex()] for j, b in sorted(self.beacons.items())],
                "seen": [[e.hex(), a.hex()] for e, a in self.seen],
                "private_root": None if self.private_root is None else self.private_root.hex(),
                "w_bytes": None if self.w_bytes is None else self.w_bytes.hex()}


def _small_session(py, rng: random.Random, **force) -> _Session:
    R, Wi = py[0], py[1]
    w, c0 = rng.choice(SMALL_POINTS)
    n = w // 8
    profile = rng.choice([f"p2-{w - 1}b", f"p2-{n - 1}B", f"p2-{rng.randint(1, w - 1)}b", f"p2-{rng.randint(1, n - 1)}B"])
    bits = Wi.payload_bits(profile, w)
    bps = rng.choice([1, 1, 2, 3, 4, 8])
    params = R.Params(w, c0, max(1, bits // 8), bps, rng.randint(1, max(1, 16 // bps)), profile)
    bad = force.get("bad_payloads", 0.0)
    payloads = [rng.randrange(1 << bits, params.p) if rng.random() < bad else rng.randrange(1 << bits)
                for _ in range(params.blocks)]
    held = R.pack_payloads([m & ((1 << bits) - 1) for m in payloads], bits)
    commitment = R.w_commitment(params, held) if rng.random() < 0.9 else rng.randbytes(64)
    salt = R.setup_salt(rng.randbytes(32), params, commitment)
    blocks = []
    for i, m in enumerate(payloads):
        kv, t, _ = R.keys(salt, params, commitment, i // bps, i % bps)
        blocks.append(R.encode(params, m, kv, t).to_bytes(n, "big"))
    leaves = [R.leaf(salt, i // bps, i % bps, b) for i, b in enumerate(blocks)]
    paths = [R.path(leaves, i) for i in range(len(leaves))]
    public = force.get("public", rng.random() < 0.6)
    delta = rng.choice([R.DELTA_NS, rng.randint(0, 10**6)])
    rtt = rng.choice([0, R.MAX_RTT_ALLOWANCE_NS, rng.randint(0, 10**6)])
    flags = force.get("flags", rng.choice([0, 1, 1, 3]))
    s = _Session(py, rng, params, commitment, salt, R.root(leaves), blocks.__getitem__, paths.__getitem__,
                 blocks if public else None, force.get("mode", rng.choice([0, 0, 1])),
                 force.get("k", rng.randint(1, 6)), flags, delta, rtt)
    s.held = held
    if flags & 2 or rng.random() < 0.2:
        s.w_bytes = held
    if public and rng.random() < 0.3:
        s.private_root = s.root
    return s


def _manifest_variant(rng: random.Random, s: _Session) -> bytes:
    """A manifest a verifier must refuse or that names another store: non-canonical spellings, wrong fields or types,
    out-of-range geometry, unknown profiles, no P2 prime."""
    m = dict(s.params.manifest())
    canon = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()  # noqa: E731
    w = m["w"]
    edits = [
        lambda: canon(m).replace(b",", b", ", 1), lambda: json.dumps(m, separators=(",", ":")).encode(),
        lambda: canon(m | {"extra": 1}), lambda: canon({k: v for k, v in m.items() if k != "segments"}),
        lambda: canon(m | {"scheme": "p3"}), lambda: canon(m).replace(f'"w":{w}'.encode(), f'"w":{w}.0'.encode()),
        lambda: canon(m | {"w": str(w)}), lambda: canon(m | {"w": True}), lambda: canon(m | {"segments": 0}),
        lambda: canon(m | {"segments": 1 << 64}), lambda: canon(m | {"segments": (1 << 64) - 1}),
        lambda: canon(m | {"blocks_per_segment": 0}), lambda: canon(m | {"blocks_per_segment": (1 << 16) + 1}),
        lambda: canon(m | {"blocks_per_segment": 1 << 16}), lambda: canon(m | {"w": 0}), lambda: canon(m | {"w": -64}),
        lambda: canon(m | {"w": 16_480}), lambda: canon(m | {"w": 16_448}), lambda: canon(m | {"w": 72}),
        lambda: canon(m | {"w": 4}), lambda: canon(m | {"c0": m["c0"] + 2}), lambda: canon(m | {"c0": 0}),
        lambda: canon(m | {"c0": -3}), lambda: canon(m | {"c0": (1 << w) + 1}), lambda: canon(m | {"c0": (1 << w) - 3}),
        lambda: canon(m | {"profile": rng.choice(["p2-0b", f"p2-{w}b", f"p2-{w // 8}B", "p2-b", "p2-B", "P2-7B", "p2-7",
                                                  "p2-07B", "p2-+7B", "p2-7Bb", "p2-\u00b3B", "p2-2054B", "", "p2--7B"])}),
        lambda: canon(m | {"profile": 7}), lambda: canon(m).replace(b'"c0"', b'"c0":1,"c0"', 1),
        lambda: canon(m).replace(b"p2", b"\\u0070\\u0032", 1), lambda: canon(m) + b" ", lambda: b"\xef\xbb\xbf" + canon(m),
        lambda: canon(m)[:-1] + b"\xff}", lambda: canon(m).replace(b"{", b"[", 1), lambda: b"[" + canon(m) + b"]",
        lambda: canon(m | {"segments": m["segments"] + 1}), lambda: canon(m).replace(f'"w":{w}'.encode(), f'"w":0{w}'.encode()),
        lambda: canon(m | {"profile": m["profile"] + "x" * 600}),
        lambda: canon(m).replace(f'"w":{w}'.encode(), f'"w":{w}e0'.encode()),
        lambda: canon(m).replace(f'"w":{w}'.encode(), b'"w":-0'),
        lambda: canon(m).replace(f'"w":{w}'.encode(), b'"w":1e999'),
        lambda: canon(m).replace(b'"profile":"', b'"profile":"\\ud800', 1),
    ]
    return rng.choice(edits)()


def _mutate(rng: random.Random, s: _Session) -> str:
    """One adversarial edit of an honest session, and its name."""
    R, Wi = s.py[0], s.py[1]
    q, B, k = s.params, s.params.blocks, s.k
    j = rng.randrange(k)
    index, nonce = s.indices[j], s.nonces[j]
    limit = s.delta + s.rtt
    ch, rs = s.rounds[j]
    name = rng.choice(["honest", "honest", "late", "on-deadline", "wrong-block", "noncanonical-block", "flip-sibling",
                       "path-count", "non-derived-index", "outside-index", "wrong-nonce", "challenge-round",
                       "response-round", "response-index", "response-identity", "challenge-identity", "replay-response",
                       "replay-challenge", "flip-byte", "truncate", "extend", "drop-tail", "extra-frame", "wrong-source",
                       "seen", "open-field", "open-length", "beacon", "manifest", "manifest", "setup", "setup", "setup",
                       "private-root", "no-private-root", "held-w", "held-w", "no-held-w", "elapsed-u64"])
    if name == "late":
        s.elapsed[j] = limit + rng.choice([1, rng.randint(1, 10**6)])
    elif name == "on-deadline":
        s.elapsed[j] = limit
    elif name == "wrong-block":
        other = s.block((index + rng.randrange(1, B)) % B) if B > 1 and rng.random() < 0.5 else \
            rng.randrange(min(q.p, 1 << q.w)).to_bytes(q.width_bytes, "big")
        s.rounds[j][1] = s.response(j, index, other, s.path(index))
    elif name == "noncanonical-block":
        s.rounds[j][1] = s.response(j, index, rng.randrange(q.p, 1 << q.w).to_bytes(q.width_bytes, "big"), s.path(index))
    elif name == "flip-sibling":
        p = list(s.path(index))
        if p:
            x = rng.randrange(len(p))
            p[x] = _flip(p[x], rng.randrange(64), rng.randrange(8))
        s.rounds[j][1] = s.response(j, index, s.block(index), p)
    elif name == "path-count":
        p = list(s.path(index))
        p = p[:-1] if p and rng.random() < 0.5 else p + [rng.randbytes(64)]
        s.rounds[j][1] = s.response(j, index, s.block(index), p)
    elif name in ("non-derived-index", "outside-index"):
        other = (index + rng.randrange(1, B)) % B if name == "non-derived-index" and B > 1 else B + rng.randrange(3)
        s.rounds[j][0] = s.challenge(j, other, nonce)
        if other < B:
            s.rounds[j][1] = s.response(j, other, s.block(other), s.path(other))
    elif name == "wrong-nonce":
        s.rounds[j][0] = s.challenge(j, index, _flip(nonce, rng.randrange(32), rng.randrange(8)))
    elif name == "challenge-round":
        s.rounds[j][0] = s.challenge(rng.choice([j + 1, (j + 1) % k, j + 256]) % 65536, index, nonce)
    elif name == "response-round":
        s.rounds[j][1] = s.response(rng.choice([j + 1, (j - 1) % 65536]), index, s.block(index), s.path(index))
    elif name == "response-index":
        s.rounds[j][1] = s.response(j, (index + 1) % (1 << 64), s.block(index), s.path(index))
    elif name in ("response-identity", "challenge-identity"):
        epoch, aid = (_flip(s.epoch, rng.randrange(64)), None) if rng.random() < 0.5 else (None, _flip(s.aid, rng.randrange(32)))
        if name == "response-identity":
            s.rounds[j][1] = s.response(j, index, s.block(index), s.path(index), epoch, aid)
        else:
            s.rounds[j][0] = s.challenge(j, index, nonce, epoch, aid)
    elif name == "replay-response" and j > 0:
        s.rounds[j][1] = s.rounds[j - 1][1]
    elif name == "replay-challenge" and j > 0:
        s.rounds[j][0] = s.rounds[j - 1][0]
    elif name in ("flip-byte", "truncate", "extend"):
        frames = s.frames()
        at = rng.randrange(len(frames))
        f = frames[at]
        if name == "flip-byte":
            g = _flip(f, rng.randrange(len(f)), rng.randrange(8))
        elif name == "truncate":
            cut = rng.randint(1, min(len(f), 80))
            g = f[:-cut] if rng.random() < 0.5 else Wi.envelope(f[5], f[10:-cut] if cut < len(f) - 10 else b"")
        else:
            more = rng.randbytes(rng.randint(1, 80))
            g = f + more if rng.random() < 0.5 else Wi.envelope(f[5], f[10:] + more)
        _replace(s, at, g)
    elif name == "drop-tail":
        frames = s.frames()
        _keep(s, rng.randrange(1, len(frames)))
    elif name == "extra-frame":
        s.extra.append(rng.choice([s.rounds[0][0], s.rounds[-1][1], s.open, rng.randbytes(20)]))
    elif name == "wrong-source":
        s.source = rng.choice([None, b"", rng.randbytes(32), _flip(s.source, 0)])
    elif name == "seen":
        others = [(rng.randbytes(64), rng.randbytes(32)) for _ in range(rng.randint(0, 2))]
        s.seen = others + ([(s.epoch, s.aid)] if rng.random() < 0.7 else [(s.epoch, _flip(s.aid, 0))])
    elif name == "open-field":
        field = rng.choice(["mode", "k", "k", "flags", "flags", "counter", "epoch", "aid", "delta"])
        value = {"mode": rng.choice([1 - s.mode, 2, 255]), "k": rng.choice([0, 106, 65535, k + 1, max(1, k - 1)]),
                 "flags": rng.choice([2, 3, 4, 0x81, 0xFF, 1 - (s.flags & 1)]), "counter": (s.counter + 1) % (1 << 64),
                 "epoch": _flip(s.epoch, 3), "aid": _flip(s.aid, 5),
                 "delta": rng.choice([0, (1 << 64) - 1])}[field]
        s.open = s.open_frame(**{field: value})
    elif name == "open-length":
        s.open = Wi.envelope(Wi.AUDIT_OPEN, s.open[10:] + b"\x00" if rng.random() < 0.5 else s.open[10:-1])
    elif name == "beacon" and s.beacons is not None:
        choice = rng.choice(["missing", "short", "other", "long"])
        if choice == "missing":
            del s.beacons[j]
        else:
            s.beacons[j] = {"short": nonce[:31], "other": _flip(nonce, 1), "long": nonce + b"\x00"}[choice]
    elif name == "manifest":
        m = _manifest_variant(rng, s)
        s.setup[0] = Wi.envelope(Wi.SETUP_COMMIT, len(m).to_bytes(4, "big") + m + s.commitment)
    elif name == "setup":
        name = "setup-" + _setup_mutation(rng, s)
    elif name == "private-root":
        s.private_root = rng.choice([_flip(s.root, rng.randrange(64), rng.randrange(8)), rng.randbytes(64), b""])
    elif name == "no-private-root":
        s.private_root = None
    elif name == "held-w":
        s.open = s.open_frame(flags=3)
        bits = Wi.payload_bits(q.profile, q.w)
        at = index * bits + rng.randrange(bits)
        s.w_bytes = rng.choice([_flip(s.held, at // 8, 7 - at % 8), s.held[:len(s.held) // 2], b""])
    elif name == "no-held-w":
        s.open = s.open_frame(flags=3)
        s.w_bytes = None
    elif name == "elapsed-u64":
        s.elapsed[j] = (1 << 64) + rng.choice([0, rng.randrange(1 << 70)])
    else:
        name = "honest"
    return name


def _replace(s: _Session, at: int, frame: bytes) -> None:
    if at < len(s.setup):
        s.setup[at] = frame
    elif at == len(s.setup):
        s.open = frame
    elif at - len(s.setup) - 1 < 2 * len(s.rounds):
        r = at - len(s.setup) - 1
        s.rounds[r // 2][r % 2] = frame
    else:
        s.extra[at - len(s.setup) - 1 - 2 * len(s.rounds)] = frame


def _keep(s: _Session, count: int) -> None:
    """Keep the session's first `count` frames."""
    frames = s.frames()[:count]
    s.setup, rest = frames[:len(s.setup)], frames[len(s.setup):]
    s.open = rest[0] if rest else None
    rounds = rest[1:]
    s.rounds = [rounds[i:i + 2] for i in range(0, len(rounds), 2)]
    s.extra = []


def _setup_mutation(rng: random.Random, s: _Session) -> str:
    Wi = s.py[1]
    q, D, S = s.params, Wi.D, Wi.S
    blocks = [i for i, f in enumerate(s.setup) if f[5] == Wi.SETUP_BLOCK]
    name = rng.choice(["salt-commitment", "salt-length", "root-length", "root-commitment", "root-salt", "root-count",
                       "root-mode", "root-value", "missing-salt", "missing-root", "open-for-root", "private-partial",
                       "public-missing"] + (["block-swap", "block-drop", "block-duplicate", "block-commitment",
                                             "block-salt", "block-length", "block-noncanonical", "block-truncated",
                                             "block-precedence"] if len(blocks) > 1 else []))
    salt_at, root_at = 1, len(s.setup) - 1
    if name == "salt-commitment":
        s.setup[salt_at] = Wi.envelope(Wi.SETUP_SALT, _flip(s.commitment, 7) + s.salt)
    elif name == "salt-length":
        s.setup[salt_at] = Wi.envelope(Wi.SETUP_SALT, s.commitment + s.salt + b"\x00")
    elif name == "root-length":
        s.setup[root_at] = s.setup[root_at][:-1]
        s.setup[root_at] = Wi.envelope(Wi.SETUP_ROOT, s.setup[root_at][10:])
    elif name == "root-commitment":
        s.setup[root_at] = Wi.envelope(Wi.SETUP_ROOT, _flip(s.setup[root_at][10:], 1))
    elif name == "root-salt":
        s.setup[root_at] = Wi.envelope(Wi.SETUP_ROOT, _flip(s.setup[root_at][10:], D + 1))
    elif name == "root-count":
        s.setup[root_at] = s.root_frame(rng.choice([q.blocks + 1, q.blocks - 1, 0]), s.root, 1 if s.public else 0)
    elif name == "root-mode":
        s.setup[root_at] = s.root_frame(q.blocks, s.root, rng.choice([2, 255, 0 if s.public else 1]))
    elif name == "root-value":
        s.setup[root_at] = s.root_frame(q.blocks, _flip(s.root, rng.randrange(64)), 1 if s.public else 0)
    elif name == "missing-salt":
        _keep(s, 1)
    elif name == "missing-root":
        s.setup = s.setup[:-1]
    elif name == "open-for-root":
        s.setup[root_at] = s.open
    elif name == "private-partial" and blocks:
        del s.setup[blocks[-1]]
        s.setup[-1] = s.root_frame(q.blocks, s.root, 0)
    elif name == "public-missing" and blocks:
        del s.setup[blocks[-1]]
    elif name == "block-swap":
        a, b = rng.sample(blocks, 2)
        s.setup[a], s.setup[b] = s.setup[b], s.setup[a]
    elif name == "block-drop":
        del s.setup[rng.choice(blocks[:-1])]
    elif name == "block-duplicate":
        at = rng.choice(blocks)
        s.setup.insert(at, s.setup[at])
    elif name in ("block-commitment", "block-salt", "block-length", "block-noncanonical", "block-truncated"):
        at = rng.choice(blocks)
        body = s.setup[at][10:]
        if name == "block-commitment":
            body = _flip(body, 0)
        elif name == "block-salt":
            body = _flip(body, D)
        elif name == "block-length":
            body = body[:D + S + 8] + (q.width_bytes + 1).to_bytes(4, "big") + body[D + S + 12:]
        elif name == "block-noncanonical":
            body = body[:D + S + 12] + rng.randrange(q.p, 1 << q.w).to_bytes(q.width_bytes, "big")
        else:
            body = body[:-1]
        s.setup[at] = Wi.envelope(Wi.SETUP_BLOCK, body)
    elif name == "block-precedence":
        a, b = blocks[0], blocks[-1]
        s.setup[a] = Wi.envelope(Wi.SETUP_BLOCK, _flip(s.setup[a][10:], 0))
        body = s.setup[b][10:]
        s.setup[b] = Wi.envelope(Wi.SETUP_BLOCK, body[:D + S + 12] + q.p.to_bytes(q.width_bytes, "big"))
    return name


def _small_audit(py, rng: random.Random) -> dict:
    s = _small_session(py, rng, bad_payloads=rng.choice([0.0, 0.0, 0.5]))
    return s.instance(_mutate(rng, s))


_DEPLOYMENT: dict = {}


def _deployment_store(py) -> dict:
    """The deployment's store, B = 2^19 blocks of 2,056 bytes, as a private (mode 0) setup: 16 distinct blocks repeated,
    so that each leaf is one compression after its block's row digest, and the tree's every level kept for the paths."""
    if not _DEPLOYMENT:
        R, Wi = py[0], py[1]
        rng = random.Random("deployment")
        q = R.DEPLOYMENT
        commitment = rng.randbytes(64)
        salt = R.setup_salt(rng.randbytes(32), q, commitment)
        pool = [rng.randrange(q.p).to_bytes(q.width_bytes, "big") for _ in range(16)]
        rows = [R.row_digest(b) for b in pool]
        bps = q.blocks_per_segment
        head = b"\x00" + salt
        level = [hashlib.sha512(head + (i // bps).to_bytes(8, "big") + (i % bps).to_bytes(2, "big") + rows[i % 16]).digest()
                 for i in range(q.blocks)]
        assert level[12345] == R.leaf(salt, 12345 // bps, 12345 % bps, pool[12345 % 16])
        levels = [level]
        while len(level) > 1:
            level = [R.node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
            levels.append(level)
        root = level[0]

        def path(i: int) -> list[bytes]:
            return [levels[h][(i >> h) ^ 1] for h in range(len(levels) - 1)]

        assert all(Wi.verify_path(root, q.blocks, i, levels[0][i], path(i)) for i in (0, 1, 777, q.blocks - 1))
        _DEPLOYMENT.update(params=q, commitment=commitment, salt=salt, root=root, block=lambda i: pool[i % 16], path=path)
    return _DEPLOYMENT


def _full_audit(py, rng: random.Random, name: str) -> dict:
    """A deployment-shaped audit: k = 105, Δ = 0.5 ms, an RTT allowance, mode 0, decode off, and one named edit."""
    R = py[0]
    d = _deployment_store(py)
    rtt = {"rtt-over-cap": R.MAX_RTT_ALLOWANCE_NS + 29}.get(name, rng.choice([0, 10_000, R.MAX_RTT_ALLOWANCE_NS]))
    s = _Session(py, rng, d["params"], d["commitment"], d["salt"], d["root"], d["block"], d["path"], None, 0,
                 R.K_AUDIT if name != "k-104" else 104, 0x01 if name == "decode" else 0, R.DELTA_NS, rtt)
    j = rng.randrange(s.k)
    if name == "late":
        s.elapsed[j] = s.delta + s.rtt + 1
    elif name == "flip-sibling":
        p = list(s.path(s.indices[j]))
        x = rng.randrange(len(p))
        p[x] = _flip(p[x], 9)
        s.rounds[j][1] = s.response(j, s.indices[j], s.block(s.indices[j]), p)
    return s.instance(f"deployment-{name}")


def _vector_audits(py) -> list[dict]:
    """The spec's audit vectors (§7): the 31 negative and boundary transcripts and the two-round end-to-end one, and a
    deployment-width session around `full_width_response` (its encoded block, decoded on flag bit 0)."""
    R, Wi = py[0], py[1]
    for path, pin in ((R.AUDIT_VECTORS, R.AUDIT_VECTORS_SHA512), (R.NEGATIVE_VECTORS, R.NEGATIVE_VECTORS_SHA512)):
        if hashlib.sha512(path.read_bytes()).hexdigest() != pin:
            raise SystemExit(f"{path.name} is not the file the spec pins")
    out = []
    for case in json.loads(R.NEGATIVE_VECTORS.read_bytes())["cases"]:
        frames = [bytes.fromhex(f) for f in case["frames"]]
        seen = []
        if case["seen_audit_pair"]:
            body = next(f for f in frames if f[5] == Wi.AUDIT_OPEN)[10:]
            seen = [[body[:Wi.D].hex(), body[Wi.D:Wi.D + Wi.S].hex()]]
        out.append({"kind": "audit", "name": "vector-" + case["name"], "frames": case["frames"],
                    "elapsed_ns": case["elapsed_ns"], "verifier_source": case["verifier_source"], "beacons": None,
                    "seen": seen, "private_root": None, "w_bytes": None, "expected_final": case["expected_final"]})
    v = json.loads(R.AUDIT_VECTORS.read_bytes())["small_end_to_end"]
    frames = [m["bytes"] for m in v["setup_messages"]] + [v["audit_open"]]
    frames += [f for x in v["exchanges"] for f in (x["challenge"], x["response"])]
    out.append({"kind": "audit", "name": "vector-small-end-to-end", "frames": frames, "elapsed_ns": v["elapsed_ns"],
                "verifier_source": v["audit_source"], "beacons": None, "seen": [], "private_root": None, "w_bytes": None,
                "expected_final": v["final"]})
    full = json.loads(R.AUDIT_VECTORS.read_bytes())["full_width_response"]
    q = R.Params(R.DEPLOYMENT.w, R.DEPLOYMENT.c0, 2054, blocks_per_segment=1, segments=1)
    manifest, commitment = full["manifest"].encode(), bytes.fromhex(full["w_commitment"])
    salt, root = bytes.fromhex(full["setup_salt"]), bytes.fromhex(full["verification_root"])
    epoch, aid = bytes.fromhex(full["epoch_id"]), bytes.fromhex(full["audit_id"])
    source, counter = bytes.fromhex(full["audit_source"]), full["audit_counter"]
    key = R.audit_key(source, epoch, root, salt, counter)[1]
    for mode in (0, 1):
        setup = Wi.setup_messages(manifest, commitment, salt, [bytes.fromhex(full["block"])], root)
        if mode == 0:
            setup = setup[:2] + [Wi.envelope(Wi.SETUP_ROOT, commitment + salt + (1).to_bytes(8, "big") + root + b"\x00")]
        frames = setup + [Wi.open_message(epoch, aid, 0, 1, R.DELTA_NS, R.MAX_RTT_ALLOWANCE_NS, 0x01, counter),
                          Wi.challenge_message(epoch, aid, 0, 0, key.stream(("nonce", 0), 32)), bytes.fromhex(full["response"])]
        assert q.blocks == 1
        out.append({"kind": "audit", "name": f"vector-full-width-response-setup-mode-{mode}", "frames": [f.hex() for f in frames],
                    "elapsed_ns": [400_000], "verifier_source": source.hex(), "beacons": None, "seen": [],
                    "private_root": root.hex() if mode == 0 else None, "w_bytes": None})
    return out


def _unchecked_params(R, w: int, c0: int, payload_bytes: int, bps: int, segments: int, profile: str):
    """`reference.Params` without its refusals, for a store the wire must refuse."""
    q = object.__new__(R.Params)
    for k, v in {"w": w, "c0": c0, "payload_bytes": payload_bytes, "blocks_per_segment": bps, "segments": segments,
                 "profile": profile}.items():
        object.__setattr__(q, k, v)
    return q


def _small_p_session(py, rng: random.Random, *, public: bool, flags: int) -> _Session:
    """An honest session over a store `reference.Params` refuses: w = 32, profile `p2-24b`, and p the largest prime
    ≡ 3 mod 4 below 2^24, so that not every 3-byte payload is below p (c0 = 2^32 − p ≡ 1 mod 4)."""
    R = py[0]
    w, bits, bps, segments = 32, 24, 2, 2
    p = 16777199
    q = _unchecked_params(R, w, (1 << w) - p, bits // 8, bps, segments, "p2-24b")
    payloads = [rng.randrange(p) for _ in range(q.blocks)]
    held = R.pack_payloads(payloads, bits)
    commitment = R.w_commitment(q, held)
    salt = R.setup_salt(rng.randbytes(32), q, commitment)
    blocks = []
    for i, m in enumerate(payloads):
        kv, t, _ = R.keys(salt, q, commitment, i // bps, i % bps)
        blocks.append(R.encode(q, m, kv, t).to_bytes(w // 8, "big"))
    leaves = [R.leaf(salt, i // bps, i % bps, b) for i, b in enumerate(blocks)]
    paths = [R.path(leaves, i) for i in range(len(leaves))]
    s = _Session(py, rng, q, commitment, salt, R.root(leaves), blocks.__getitem__, paths.__getitem__,
                 blocks if public else None, 0, 3, flags, R.DELTA_NS, 0)
    s.w_bytes = held if flags & 2 else None
    return s


def _with_manifest(s: _Session, manifest: bytes) -> None:
    s.setup[0] = s.py[1].envelope(s.py[1].SETUP_COMMIT, len(manifest).to_bytes(4, "big") + manifest + s.commitment)


def redteam_audits(py) -> list[dict]:
    """Named sessions at the edges the two verifiers once read apart: a store whose payloads are not all below p (honest
    sessions over it, public and private, under each flag, and p negative), a store of 2^79 blocks under SETUP_ROOT block
    counts no u64 can make its own, a mode-1 beacon list naming a round twice, and manifest numbers Python's `json` reads
    as floats (an exponent of a million digits' value, `NaN`, `±Infinity`, either side of the double range's end)."""
    R = py[0]
    out = []
    for public in (True, False):
        for flags in (0, 1, 3):
            s = _small_p_session(py, random.Random(f"redteam/p/{public}/{flags}"), public=public, flags=flags)
            out.append(s.instance(f"redteam-payload-not-below-p-{'public' if public else 'private'}-flags-{flags}"))
    s = _small_p_session(py, random.Random("redteam/p/truncated"), public=True, flags=1)
    s.rounds = [s.rounds[0][:1]]
    out.append(s.instance("redteam-payload-not-below-p-truncated"))
    s = _small_p_session(py, random.Random("redteam/p/negative"), public=True, flags=1)
    _with_manifest(s, R.canonical_json_bytes(s.params.manifest() | {"c0": (1 << 32) + 1}))
    out.append(s.instance("redteam-p-negative"))
    for count in (0, (1 << 64) - 1, 1 << 15):
        s = _small_session(py, random.Random(f"redteam/blocks/{count}"), public=False, mode=0, flags=0)
        _with_manifest(s, R.canonical_json_bytes(s.params.manifest() | {"blocks_per_segment": 1 << 16, "segments": 1 << 63}))
        s.setup[-1] = s.root_frame(count, s.root, 0)
        out.append(s.instance(f"redteam-blocks-past-u64-count-{count}"))
    for order in ("wrong-first", "wrong-last"):
        s = _small_session(py, random.Random(f"redteam/beacons/{order}"), mode=1, k=3, flags=0)
        inst = s.instance(f"redteam-beacon-round-twice-{order}")
        j, right = inst["beacons"][1]
        pair = [[j, right], [j, _flip(bytes.fromhex(right), 0).hex()]]
        inst["beacons"] = inst["beacons"][:1] + (pair[::-1] if order == "wrong-first" else pair) + inst["beacons"][2:]
        out.append(inst)
    for name, edit in [("exponent-1e1000000", b'"w":1e1000000'), ("exponent-1e16", b'"w":1e16'), ("nan", b'"w":NaN'),
                       ("infinity", b'"w":Infinity'), ("minus-infinity", b'"w":-Infinity'),
                       ("past-double", b'"w":1.7976931348623159e308'), ("below-double-max", b'"w":1.7976931348623158e308')]:
        s = _small_session(py, random.Random(f"redteam/float/{name}"), public=True, mode=0, flags=0)
        _with_manifest(s, R.canonical_json_bytes(s.params.manifest()).replace(f'"w":{s.params.w}'.encode(), edit))
        out.append(s.instance(f"redteam-manifest-{name}"))
    return out


SPEC = "verity/core/protocols/pous/p2/PROTOCOL.md"


def _pins(py) -> dict:
    """§7's SHA-512 pins of the audit files and frames, which `PousCheck` recomputes from the bytes."""
    R = py[0]
    av = json.loads(R.AUDIT_VECTORS.read_bytes())
    small, full = av["small_end_to_end"], av["full_width_response"]
    items = [["audit-vectors.json SHA-512", R.AUDIT_VECTORS.read_bytes().hex()],
             ["audit-negative-vectors.json SHA-512", R.NEGATIVE_VECTORS.read_bytes().hex()],
             ["w=64 audit-open frame", small["audit_open"]], ["w=64 round-0 challenge", small["exchanges"][0]["challenge"]],
             ["w=64 round-0 response", small["exchanges"][0]["response"]], ["w=64 final frame", small["final"]],
             ["w=16,448 response frame", full["response"]]]
    return {"kind": "sha512", "spec": SPEC, "items": items}


def _pins_answers(py, inst: dict) -> dict:
    """The pins as §7 of the spec states them."""
    text = (Path(py[0].__file__).resolve().parents[3] / inst["spec"]).read_text()
    out = {}
    for label, _ in inst["items"]:
        found = re.findall(r"^- (?:`([^`]+)`|([^:`]+)) ?(SHA-512)?: `([0-9a-f]{128})`$", text, re.M)
        wanted = [h for file, plain, sha, h in found if (f"{file} SHA-512" if file else plain.strip()) == label]
        if len(wanted) != 1:
            raise SystemExit(f"§7 has {len(wanted)} pins for {label}")
        out[label] = wanted[0]
    return out


def _storage(w: int, c0: int, bps: int, segments: int, k: int, delta: int, rtt: int) -> dict | None:
    """The generic protocol's gate and profile for P2 at a store: `TimedAudit` covers the audit when it runs it
    without `research`, and with `research` its certificate and gaps make the `StorageProfile`. None when `P2Params`
    refuses the store."""
    from verity.core.protocols.pous.protocol import StorageProfile, TimedAudit
    from verity.core.protocols.pous.schemes.p2 import P2, P2Params
    from verity.core.protocols.pous.verifier import VerifierKey
    from verity.ml.randomness import derive
    try:
        params = P2Params(w, c0, bps, segments)
    except ValueError:
        return None
    scheme = P2(params)
    vk, key = VerifierKey(params, bytes(24), bytes(64), ()), derive(b"difftest", "verity/pous/difftest/v1", {})

    def challenger(research: bool) -> TimedAudit:
        return TimedAudit(scheme, vk, key, k=k, deadline_ns=delta, rtt_allowance_ns=rtt, root_only=True,
                                research=research, experimental=True)

    try:
        challenger(False)
        covered = True
    except ValueError:
        covered = False
    mc = challenger(True)
    profile = StorageProfile(scheme.name, params.manifest(), params.blocks, params.m, mc.k, mc.limit_ns, mc.certificate,
                             mc.gaps, scheme.experimental)
    return {"covered": covered, "profile": profile.to_json()}


def _assess(py, frames: list[bytes], final: bytes) -> dict:
    """What the session's record establishes: the `FINAL`'s verdict and, on success, the gate and profile at the
    manifest's store, with `k`, `Δ` and `RTT` the `FINAL`'s."""
    Wi = py[1]
    verdict, _, records = Wi.parse_final(final)
    if not verdict:
        return {"verdict": False, "accepted": False}
    m = json.loads(frames[0][14:-Wi.D])
    o = Wi.D + Wi.S + 4 + 10
    delta, rtt = int.from_bytes(final[o:o + 8], "big"), int.from_bytes(final[o + 8:o + 16], "big")
    gate = _storage(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], records, delta, rtt)
    if gate is None:
        return {"verdict": True, "accepted": False, "profile": None}
    return {"verdict": True, "covered": gate["covered"], "accepted": gate["covered"], "profile": gate["profile"]}


def _storage_instance(rng: random.Random) -> dict:
    """A store and an audit's `k`, `Δ` and `RTT` near the certificate's edges: the registered point or one coordinate off
    it, `k` around 105, `Δ + RTT` around one root call."""
    w, c0, bps = rng.choice([(16448, 21065, 1 << 16)] * 4 + [(16448, 21069, 1 << 16), (16416, 21065, 1 << 16),
                             (16448, 21065, 1 << 15), (64, 189, 8), (16448, 1 << 16 | 1, 1 << 16), (16448, 65533, 1 << 16)])
    segments = rng.choice([8, 8, 8, 1, 7, 9, 1 << 40])
    k = rng.choice([105, 105, 104, 106, 1, 0, 65535])
    limit = rng.choice([854_028, 854_029, 854_000, 0, 1, 854_029 * 2, 1_708_057, 10**12, rng.randint(0, 2 * 854_029)])
    delta = rng.choice([500_000, 0, limit, rng.randint(0, limit)])
    delta = min(delta, limit)
    return {"kind": "storage", "w": w, "c0": c0, "blocks_per_segment": bps, "segments": segments, "k": k, "delta": delta,
            "rtt": limit - delta}


def _storage_answers(py, inst: dict) -> dict | None:
    return _storage(inst["w"], inst["c0"], inst["blocks_per_segment"], inst["segments"], inst["k"], inst["delta"],
                    inst["rtt"])


def verifier(inst: dict) -> dict:
    """`verify_audit`'s keyword arguments from an instance: what the verifier holds besides the frames."""
    def raw(key: str) -> bytes | None:
        return None if inst[key] is None else bytes.fromhex(inst[key])
    return {"verifier_source": raw("verifier_source"),
            "beacon_inputs": None if inst["beacons"] is None else {j: bytes.fromhex(b) for j, b in inst["beacons"]},
            "seen_audits": {(bytes.fromhex(e), bytes.fromhex(a)) for e, a in inst["seen"]},
            "private_root": raw("private_root"), "w_bytes": raw("w_bytes")}


def _audit_answers(py, inst: dict) -> dict:
    Wi = py[1]
    try:
        final = Wi.verify_audit([bytes.fromhex(f) for f in inst["frames"]], inst["elapsed_ns"], **verifier(inst))
    except Wi.WireError as error:
        return {"error": error.code}
    except ValueError as error:
        return {"refused": str(error)}
    if "expected_final" in inst and final.hex() != inst["expected_final"]:
        raise SystemExit(f"{inst['name']}: the reference verifier no longer gives the vector's FINAL")
    return {"final": final.hex(), "storage": _assess(py, [bytes.fromhex(f) for f in inst["frames"]], final)}


DEPLOYMENT_AUDITS = ("honest", "honest", "late", "flip-sibling", "decode", "rtt-over-cap", "k-104")


def instances(py) -> list[dict]:
    out = [_merkle(random.Random(f"merkle/{seed}")) for seed in range(400)]
    out += [_randomness(random.Random(f"randomness/{seed}")) for seed in range(200)]
    out += [_small_audit(py, random.Random(f"audit/{seed}")) for seed in range(600)]
    out += [_full_audit(py, random.Random(f"deployment/{i}"), name) for i, name in enumerate(DEPLOYMENT_AUDITS)]
    out += [_storage_instance(random.Random(f"storage/{seed}")) for seed in range(200)]
    return out + _vector_audits(py) + redteam_audits(py) + [_pins(py)]


ANSWERS = {"merkle": _merkle_answers, "randomness": _randomness_answers, "audit": _audit_answers, "sha512": _pins_answers,
           "storage": _storage_answers}


def answers(py, inst: dict) -> dict:
    return ANSWERS[inst["kind"]](py, inst)


def write(repo: Path, out: Path) -> dict[str, Path]:
    py = _python(repo)
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    insts = instances(py)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    paths["difftest/instances.jsonl"].write_text("".join(canonical(i) + "\n" for i in insts))
    paths["difftest/python.jsonl"].write_text("".join(canonical(answers(py, i)) + "\n" for i in insts))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose Python verifier is the reference")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    py = _python(a.verity)
    for module in py:
        here = Path(module.__file__).resolve()
        if not here.is_relative_to(a.verity.resolve()):
            raise SystemExit(f"{module.__name__} comes from {here}, not from the checkout {a.verity}")
    write(a.verity, a.out)


if __name__ == "__main__":
    main()
