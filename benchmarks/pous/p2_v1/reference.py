"""P2 v4 on the CPU, as `verity/core/protocols/pous/p2/PROTOCOL.md` (P2 specification v4) states it: the
reference the harness's GPU kernels are gated against, and the one the wire verifier (`wire.py`) and the vectors
(`vectors.py`) are built on. Its maps and key streams are the registered scheme's
(`verity.core.protocols.pous.schemes.p2`, held to `PousCheck.P2` by its difftest), under the spec's names and with the
spec's refusals of non-canonical inputs; what is the spec's own here is its setup framing, `Params` with its refusals
(which `PousCheck.Wire` mirrors), the verification tree and the audit's derivations.

v4 is v1.1 with one hash, SHA-512 with its whole 64-byte output, everywhere: the
W commitment, the verification tree, the epoch id and the keys, which are counter-mode SHA-512 where v1 read SHAKE256.
`verity.ml.randomness` is the spec's derivation (`frame`, `derive`) and
`verity.ml.commitments.identity.canonical_json_bytes` its manifest encoding. It is pinned by `vectors.json` beside it
(`VECTORS_SHA512`, the hash in the spec's §7).

- `Params`: `p = 2^w − c0 ≡ 3 mod 4`, the payload profile (`p2-2054B` at `DEPLOYMENT`: 2,054 bytes of each 2,056-byte block)
  and the store's shape (blocks per segment, segments).
- `setup_salt`, `expand_prefix`: §4's post-commitment salt and the spec's key prefix, which frames the model commitment and
  a 32-byte salt where the registered scheme's frames its 24-byte salt alone.
- `stream`, `keys`, `keys_after`: §4's per-block `K` and `t` after a prefix (`schemes.p2.stream` and `sha512_keys_indexed`:
  counter-mode SHA-512 over a prefix of whole 128-byte blocks; `t` is rejection-sampled below `p`).
- `d`, `rho`, `sigma`, `encode`, `decode`: §3's maps on `schemes.p2`'s field, `E_j(m) = ρ(σ_K(ρ(m + t)))` and
  `D_j(c) = d(σ_K(d(c))) − t`. They and the keys read only `w`, `c0` and `p`, so they take a `schemes.p2.P2Params` too.
- `manifest`, `w_commitment`, `row_digest`, `leaf`, `node`, `root`, `epoch_id`, `audit_key`, `round_key`: the setup
  commitment, the RFC 6962 verification tree over 0x00-tagged leaves of each block's `sha512/row/v2` digest, and the
  audit's challenge derivation (mode 0 and mode 1).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from verity.core.protocols.pous.schemes import p2 as P2
from verity.ml.commitments.identity import canonical_json_bytes
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix
from verity.ml.randomness import Key, derive, frame

HERE = Path(__file__).resolve().parent
VECTORS = HERE / "vectors.json"
VECTORS_SHA512 = "959911fd371111fe20c4cee6fc532ff205bbf377962c73ac539a4d2de9b22121287e6e996d9ae4bfbef9f26fcd67ac78620873702c8b2899bc41a21bdd76e853"
AUDIT_VECTORS = HERE / "audit-vectors.json"
AUDIT_VECTORS_SHA512 = "5f8ed7f279ccd2f2d5bb60607fb5d7bec82bf982ac85da268f76cb8fbb16d36424e7b99eaa386508840aa2bd3c3b501512bde86b930871a769cd470a14ab176e"
NEGATIVE_VECTORS = HERE / "audit-negative-vectors.json"
NEGATIVE_VECTORS_SHA512 = "f57340a0965628051bd0c938c4c7e2db5a7133982ae1a523a1221a3fbb273bfbcb1ebc1d29680b61b47f86d45e7c98aa4a5ac6311545c19c78728d857596a687"
"""The spec's §7 pins: `vectors.py` writes the three files and rewrites these."""
PROTOCOL = P2.PROTOCOL_ID
SETUP_DOMAIN = "verity/pous/p2-setup/v4"
AUDIT_ID_DOMAIN = "verity/pous/p2-audit-id/v4"
CHALLENGE_DOMAIN = "verity/pous/p2-audit-challenge/v4"
ROUND_DOMAIN = "verity/pous/p2-audit-round/v4"
KEYS_TAG = P2.KEYS_TAG
EPOCH_TAG = b"verity/pous/p2/epoch/v4\0"
TAGGED_SHA512 = b"verity/tagged-sha512/v1\0"
W_COMMITMENT_LABEL = b"verity-pous/w-commitment/v2"
DIGEST_BYTES = 64
SALT_BYTES = 32
K_AUDIT = P2.CERT_K
"""The deployment audit's challenges (spec §5, F6)."""
DELTA_NS = P2.DELTA_NS
MAX_RTT_ALLOWANCE_NS = P2.RTT_ALLOWANCE_CAP_NS
"""Δ and the most round-trip allowance the operating point permits (spec §6): v1.1's cap, so
that `Δ + RTT ≤ 854 µs` keeps SeqRoot's 2× margin with 8 cores of one chiplet cooperating
(v1 permitted 1 ms)."""


@dataclass(frozen=True)
class Params:
    """A width, its prime, the payload profile (whole bytes, the leading two bytes of every block zero) and the store."""

    w: int
    c0: int
    payload_bytes: int
    blocks_per_segment: int = 1 << 16
    segments: int = 8
    profile: str = "p2-2054B"

    def __post_init__(self) -> None:
        if self.w < 8 or self.w % 32 or self.c0 < 1 or self.c0 % 4 != 1 or self.p % 4 != 3:
            raise ValueError("P2 needs w a multiple of 32 bits and c0 = 1 mod 4 with p = 2^w - c0 = 3 mod 4")
        if not 0 < self.payload_bytes < self.width_bytes:
            raise ValueError("a payload is fewer bytes than a block")
        if self.p <= 1 << 8 * self.payload_bytes:
            raise ValueError("every payload is below p")

    @property
    def p(self) -> int:
        return (1 << self.w) - self.c0

    @property
    def width_bytes(self) -> int:
        return self.w // 8

    @property
    def prime_id(self) -> str:
        return f"p2-p{self.w}-c{self.c0}"

    @property
    def blocks(self) -> int:
        return self.blocks_per_segment * self.segments

    def with_segments(self, segments: int) -> Params:
        return Params(self.w, self.c0, self.payload_bytes, self.blocks_per_segment, segments, self.profile)

    def manifest(self) -> dict[str, int | str]:
        return {"scheme": "p2", "w": self.w, "c0": self.c0, "profile": self.profile,
                "blocks_per_segment": self.blocks_per_segment, "segments": self.segments}


DEPLOYMENT = Params(16_448, 21_065, 2_054)


def setup_salt(source: bytes, params: Params, model: bytes) -> bytes:
    """The 256-bit setup salt, drawn after the commitment to `W` (`model`) from the verifier's or a beacon's `source`."""
    return derive(source, SETUP_DOMAIN, {"model": model, "prime": params.prime_id, "protocol": PROTOCOL}).value


def expand_prefix(salt: bytes, params: Params, model: bytes) -> bytes:
    """Every block's key-stream input before its `u64be(segment) ‖ u64be(local block)`: the framed setup, zero-padded to
    whole 128-byte blocks, so that one midstate serves every block and each 64-byte chunk is one compression after it."""
    if len(salt) != SALT_BYTES or len(model) != DIGEST_BYTES:
        raise ValueError("the setup salt is 32 bytes and the model commitment 64")
    body = KEYS_TAG + frame(PROTOCOL) + frame(params.prime_id) + frame(model) + frame(salt)
    return body + bytes(-len(body) % 128)


stream = P2.stream


def keys(salt: bytes, params: Params, model: bytes, segment: int, block: int) -> tuple[int, int, int]:
    """Block `(segment, block)`'s full-width mask `K`, its tweak `t` (uniform below `p`) and `t`'s candidate index."""
    return keys_after(expand_prefix(salt, params, model), params, segment, block)


def keys_after(prefix: bytes, params: Params | P2.P2Params, segment: int, block: int) -> tuple[int, int, int]:
    """`keys` after any framing's `prefix`: the spec's (`expand_prefix`), or the registered scheme's, whose prefix frames
    the protocol's 24-byte salt and no model commitment (`verity...schemes.p2.expand_prefix`)."""
    return P2.sha512_keys_indexed(params, prefix, segment, block)


def d(params: Params | P2.P2Params, z: int) -> int:
    if not 0 <= z < params.p:
        raise ValueError("d's input is not canonical")
    return P2._field(params).sq(z)


def rho(params: Params | P2.P2Params, y: int) -> int:
    """`d⁻¹`: the parity-canonical root of ±y."""
    if not 0 <= y < params.p:
        raise ValueError("ρ's input is not canonical")
    return P2._field(params).rho(y)


def sigma(params: Params | P2.P2Params, r: int, k: int) -> int:
    if not 0 <= r < params.p or not 0 <= k < 1 << params.w:
        raise ValueError("σ's inputs are not canonical")
    return P2._field(params).sigma(r, k)


def encode(params: Params | P2.P2Params, m: int, k: int, t: int) -> int:
    if not 0 <= m < params.p or not 0 <= t < params.p:
        raise ValueError("the payload and the tweak are canonical")
    return rho(params, sigma(params, rho(params, (m + t) % params.p), k))


def decode(params: Params | P2.P2Params, c: int, k: int, t: int) -> int:
    if not 0 <= c < params.p:
        raise ValueError("the codeword is not canonical")
    return (d(params, sigma(params, d(params, c), k)) - t) % params.p


def w_commitment(params: Params, w: bytes) -> bytes:
    """The commitment to `W` (spec §4's `w_commitment`): tagged SHA-512 over the canonical manifest and `W`, 64 bytes
    (`verity.ml.commitments.identity.tagged_sha512`'s frame, hashed here in pieces so that `W` is not copied)."""
    manifest = canonical_json_bytes(params.manifest())
    h = hashlib.sha512(TAGGED_SHA512 + len(W_COMMITMENT_LABEL).to_bytes(4, "big") + W_COMMITMENT_LABEL)
    h.update((8 + len(manifest) + len(w)).to_bytes(8, "big") + len(manifest).to_bytes(8, "big") + manifest)
    h.update(w)
    return h.digest()


def pack_payloads(payloads: list[int], bits: int) -> bytes:
    """`W` from its payloads (spec §4): packed continuously, the first payload and the most significant bit first,
    zero-padded; at a whole number of bytes per payload, their concatenation."""
    packed = 0
    for value in payloads:
        if not 0 <= value < 1 << bits:
            raise ValueError("payload does not fit its profile width")
        packed = packed << bits | value
    pad = -len(payloads) * bits % 8
    return (packed << pad).to_bytes((len(payloads) * bits + pad) // 8, "big")


def row_digest(data: bytes) -> bytes:
    """A block's `sha512/row/v2` digest, `SHA512(P(8·len) ‖ block)`: the row a block's decode circuit hashes."""
    return hashlib.sha512(sha512_row_v2_prefix(8 * len(data)) + data).digest()


def leaf(salt: bytes, segment: int, block: int, data: bytes) -> bytes:
    """`SHA512(0x00 ‖ setup salt ‖ u64be(segment) ‖ u16be(local block) ‖ row_digest(block))`: 107 bytes, one compression."""
    return hashlib.sha512(b"\x00" + salt + segment.to_bytes(8, "big") + block.to_bytes(2, "big") + row_digest(data)).digest()


def node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(b"\x01" + left + right).digest()


def root(leaves: list[bytes]) -> bytes:
    """The RFC 6962 tree's root: split at the largest power of two below the count, nodes `SHA512(0x01 ‖ l ‖ r)`."""
    if not leaves:
        raise ValueError("a verification tree has at least one leaf")

    def mth(lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            return leaves[lo]
        k = 1 << ((hi - lo - 1).bit_length() - 1)
        return node(mth(lo, lo + k), mth(lo + k, hi))

    return mth(0, len(leaves))


def path(leaves: list[bytes], index: int) -> list[bytes]:
    """Leaf `index`'s RFC 9162 inclusion path, siblings from the leaf toward the root."""
    if len(leaves) == 1:
        return []
    k = 1 << ((len(leaves) - 1).bit_length() - 1)
    if index < k:
        return path(leaves[:k], index) + [root(leaves[k:])]
    return path(leaves[k:], index - k) + [root(leaves[:k])]


def epoch_id(params: Params, model: bytes, salt: bytes, verification_root: bytes) -> bytes:
    manifest = canonical_json_bytes(params.manifest())
    return hashlib.sha512(EPOCH_TAG + len(manifest).to_bytes(4, "big") + manifest + model + salt
                          + params.blocks.to_bytes(8, "big") + verification_root).digest()


def audit_id(source: bytes, epoch: bytes, counter: int) -> bytes:
    return derive(source, AUDIT_ID_DOMAIN, {"counter": counter, "epoch": epoch}).stream("id", 32)


def audit_key(source: bytes, epoch: bytes, verification_root: bytes, salt: bytes, counter: int) -> tuple[bytes, Key]:
    """Mode 0: the audit's id and the key whose `uniform(blocks, j)` is round `j`'s index and `stream(("nonce", j), 32)`
    its nonce."""
    aid = audit_id(source, epoch, counter)
    return aid, derive(source, CHALLENGE_DOMAIN, {"audit": aid, "epoch": epoch, "root": verification_root, "salt": salt})


def round_key(beacon: bytes, aid: bytes, epoch: bytes, verification_root: bytes, j: int) -> Key:
    """Mode 1: round `j`'s key from its 32-byte beacon value; its `uniform(blocks, j)` is the round's index."""
    return derive(beacon, ROUND_DOMAIN, {"audit": aid, "epoch": epoch, "root": verification_root, "round": j})


def conformance(full_encode: bool = False) -> dict[str, object]:
    """`vectors.json` against this reference: the file's hash is the spec's, and every vector's salt, keys and decode (and its
    encode, at full width only with `full_encode`: two roots in Python take seconds) come out as pinned."""
    import json

    raw = VECTORS.read_bytes()
    doc = json.loads(raw)
    checks = {"file_sha512": hashlib.sha512(raw).hexdigest() == VECTORS_SHA512}
    for v in doc["vectors"]:
        w, c0 = int(v["w"]), int(v["c0"])
        params = Params(w, c0, 2054 if (w, c0) == (DEPLOYMENT.w, DEPLOYMENT.c0) else w // 8 - 1)
        model, n = bytes.fromhex(v["model_commitment"]), params.width_bytes
        salt = setup_salt(bytes.fromhex(v["source"]), params, model)
        k, t, attempt = keys(salt, params, model, int(v["segment"]), int(v["block"]))
        m, c = int(v["m"], 16), int(v["c"], 16)
        ok = (salt.hex() == v["setup_salt"] and k.to_bytes(n, "big").hex() == v["K"] and t.to_bytes(n, "big").hex() == v["t"]
              and attempt == int(v["t_rejection_index"]) and decode(params, c, k, t) == m)
        if full_encode or w < DEPLOYMENT.w:
            ok = ok and encode(params, m, k, t) == c
        checks[v["id"]] = ok
    for s in doc["sigma_rejection_vectors"]:
        w, c0 = int(s["w"]), int(s["c0"])
        params = Params(w, c0, 2054 if (w, c0) == (DEPLOYMENT.w, DEPLOYMENT.c0) else w // 8 - 1)
        checks[s["id"]] = sigma(params, int(s["r"], 16), int(s["K"], 16)) == int(s["sigma"], 16)
    return {"passed": all(checks.values()), "vector": str(VECTORS.name), "checks": checks}
