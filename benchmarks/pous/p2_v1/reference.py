"""P2 v1 on the CPU, as the frozen specification states it (the POUS store's `docs/efficient-crypto/p2-spec-draft.md`, v1,
29 Sep 2026): the reference the harness's GPU kernels are gated against.

A port of the P2 lane's `internal/efficient-crypto/p2-bounty-kit/p2_ref.py` onto core: `verity.randomness.frame` and
`derive` are the spec's framing and derivation, and `verity.commitments.identity.canonical_json_bytes` its manifest
encoding. Big integers are Python's (gmpy2's `powmod` when it is installed). It is pinned by `vectors.json` beside it, the
kit's `p2-test-vectors/v1` (SHA-256 `781fec77…`, the file hash in the spec's §7).

- `Params`: `p = 2^w − c0 ≡ 3 mod 4`, the payload profile (`p2-2054B` at `DEPLOYMENT`: 2,054 bytes of each 2,056-byte block)
  and the store's shape (blocks per segment, segments).
- `setup_salt`, `keys`: §4's post-commitment salt and per-block `K` and `t` (SHAKE256 with the v1 framing; `t` is
  rejection-sampled below `p`).
- `d`, `rho`, `sigma`, `encode`, `decode`: §3's maps, `E_j(m) = ρ(σ_K(ρ(m + t)))` and `D_j(c) = d(σ_K(d(c))) − t`.
- `manifest`, `w_commitment`, `leaf`, `root`, `epoch_id`, `audit_key`: the setup commitment, the RFC 6962 verification
  tree over 0x00-tagged leaves, and the audit's challenge derivation (mode 0, `verity/pous/audit-challenge/v2`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from verity.commitments.identity import canonical_json_bytes
from verity.randomness import Key, derive, frame

HERE = Path(__file__).resolve().parent
VECTORS = HERE / "vectors.json"
VECTORS_SHA256 = "781fec77f4fcadb45f10673457ecd9b63d26cf826a2ebaed5da795539d8a5fab"
PROTOCOL = "p2/v1"
SETUP_DOMAIN = "verity/pous/p2-setup/v1"
EXPAND_LABEL = "verity/pous/p2/expand/v1"
W_COMMITMENT_LABEL = b"verity-pous/w-commitment/v1"
K_AUDIT = 105
"""The deployment audit's challenges (spec §5, F6)."""
DELTA_NS = 500_000
MAX_RTT_ALLOWANCE_NS = 350_000
"""Δ and the most round-trip allowance the operating point permits (spec §6): option A of the proposed v1.1, pending
Daniel's ruling, so that `Δ + RTT ≤ 850 µs` keeps SeqRoot's 2× margin with 8 cores of one chiplet cooperating
(`verity_pous.schemes.p2.RTT_ALLOWANCE_CAP_NS`; v1 permitted 1 ms)."""

try:
    import gmpy2

    def _powmod(x: int, e: int, m: int) -> int:
        return int(gmpy2.powmod(x, e, m))
except ImportError:
    def _powmod(x: int, e: int, m: int) -> int:
        return pow(x, e, m)


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
    """Every block's SHAKE256 input before its `u64be(segment) ‖ u64be(local block) ‖ purpose`."""
    if len(salt) != 32 or len(model) != 32:
        raise ValueError("the setup salt and the model commitment are 32 bytes each")
    return frame(EXPAND_LABEL) + frame(PROTOCOL) + frame(params.prime_id) + frame(model) + frame(salt)


def keys(salt: bytes, params: Params, model: bytes, segment: int, block: int) -> tuple[int, int, int]:
    """Block `(segment, block)`'s full-width mask `K`, its tweak `t` (uniform below `p`) and `t`'s candidate index."""
    n = params.width_bytes
    base = expand_prefix(salt, params, model) + segment.to_bytes(8, "big") + block.to_bytes(8, "big")
    k = int.from_bytes(hashlib.shake_256(base + b"K").digest(n), "big")
    stream = hashlib.shake_256(base + b"t")
    attempt = 0
    while True:
        candidate = int.from_bytes(stream.digest((attempt + 1) * n)[attempt * n:], "big")
        if candidate < params.p:
            return k, candidate, attempt
        attempt += 1


def d(params: Params, z: int) -> int:
    if not 0 <= z < params.p:
        raise ValueError("d's input is not canonical")
    square = z * z % params.p
    return square if z % 2 == 0 else (-square) % params.p


def rho(params: Params, y: int) -> int:
    """`d⁻¹`: one exponentiation by `(p + 1) / 4`, then the parity-canonical member."""
    if not 0 <= y < params.p:
        raise ValueError("ρ's input is not canonical")
    if y == 0:
        return 0
    s = _powmod(y, (params.p + 1) // 4, params.p)
    square = s * s % params.p
    if square == y:
        return s if s % 2 == 0 else params.p - s
    if square == (-y) % params.p:
        return s if s % 2 else params.p - s
    raise ArithmeticError("p is not a prime congruent to 3 mod 4")


def sigma(params: Params, r: int, k: int) -> int:
    if not 0 <= r < params.p or not 0 <= k < 1 << params.w:
        raise ValueError("σ's inputs are not canonical")
    x = r ^ k
    return x if x < params.p else r


def encode(params: Params, m: int, k: int, t: int) -> int:
    if not 0 <= m < params.p or not 0 <= t < params.p:
        raise ValueError("the payload and the tweak are canonical")
    return rho(params, sigma(params, rho(params, (m + t) % params.p), k))


def decode(params: Params, c: int, k: int, t: int) -> int:
    if not 0 <= c < params.p:
        raise ValueError("the codeword is not canonical")
    return (d(params, sigma(params, d(params, c), k)) - t) % params.p


def w_commitment(params: Params, w: bytes) -> bytes:
    """The commitment to `W` (spec §4's `w_commitment`): tagged SHA-256 over the canonical manifest and `W`."""
    manifest = canonical_json_bytes(params.manifest())
    h = hashlib.sha256(b"veritor/tagged-sha256/v1\0" + len(W_COMMITMENT_LABEL).to_bytes(4, "big") + W_COMMITMENT_LABEL)
    h.update((8 + len(manifest) + len(w)).to_bytes(8, "big") + len(manifest).to_bytes(8, "big") + manifest)
    h.update(w)
    return h.digest()


def leaf(salt: bytes, segment: int, block: int, data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + salt + segment.to_bytes(8, "big") + block.to_bytes(2, "big") + data).digest()


def root(leaves: list[bytes]) -> bytes:
    """The RFC 6962 tree's root: split at the largest power of two below the count, nodes `SHA256(0x01 ‖ l ‖ r)`."""
    if not leaves:
        raise ValueError("a verification tree has at least one leaf")

    def mth(lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            return leaves[lo]
        k = 1 << ((hi - lo - 1).bit_length() - 1)
        return hashlib.sha256(b"\x01" + mth(lo, lo + k) + mth(lo + k, hi)).digest()

    return mth(0, len(leaves))


def epoch_id(params: Params, model: bytes, salt: bytes, verification_root: bytes) -> bytes:
    manifest = canonical_json_bytes(params.manifest())
    return hashlib.sha256(b"verity/pous/p2/epoch/v1\0" + len(manifest).to_bytes(4, "big") + manifest + model + salt
                          + params.blocks.to_bytes(8, "big") + verification_root).digest()


def audit_key(source: bytes, epoch: bytes, verification_root: bytes, salt: bytes, counter: int) -> tuple[bytes, Key]:
    """Mode 0: the audit's id and the key whose `uniform(blocks, j)` is round `j`'s index."""
    audit_id = derive(source, "verity/pous/audit-id/v1", {"counter": counter, "epoch": epoch}).stream("id", 32)
    return audit_id, derive(source, "verity/pous/audit-challenge/v2",
                            {"audit": audit_id, "epoch": epoch, "root": verification_root, "salt": salt})


def conformance(full_encode: bool = False) -> dict[str, object]:
    """`vectors.json` against this reference: the file's hash is the spec's, and every vector's salt, keys and decode (and its
    encode, at full width only with `full_encode`: two roots in Python take seconds) come out as pinned."""
    import json

    raw = VECTORS.read_bytes()
    doc = json.loads(raw)
    checks = {"file_sha256": hashlib.sha256(raw).hexdigest() == VECTORS_SHA256}
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
