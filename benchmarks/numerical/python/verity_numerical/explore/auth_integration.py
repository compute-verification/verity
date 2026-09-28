"""What binding a numerical proof's x / w / y to the Verity commitment roots costs, per option (auth-integration lane,
2026-09-22).  Report: ``backends/numerical/reports/auth_integration.md``; JSON: ``reports/auth_integration.json``.

Every ledger number so far says ``authentication = excluded``.  This module makes the *complete* number knowable: it
prices five ways of authenticating the proved operands against ``verity.ml.commitments`` roots and plugs the best into
``explore.tensor`` through its ``authentication`` knob, so that ``price_design(..., zk="malicious",
authentication=<x>)`` is the complete Verity proof per design.  Everything is a cost-model ESTIMATE on the same footing
as ``tensor.md`` (limb MACs at H100 INT8 peak x 0.172, SIMT field mults, hash bytes at a stated rate); the only
measurements used are the hash-gpu lane's SHA-256/Blake3/Poseidon2 rates.

Options (report section 2; ``OPTIONS``)
========================================

(a) ``in_circuit_sha256``   the SP1 way: the proof re-hashes the whole ``leaf/v1`` leaf (3 072 / 4 096 B rows) and folds
                            the node/lift path in-circuit.  Priced with the best lookup-based SHA-256 AIR in the
                            literature (Stwo-style, 7 840 cells per compression) and, as the upper bracket, the R1CS
                            circuit (26 170 nonlinear constraints).  No protocol change.
(b) ``verifier_openings``   the prover hands the verifier the leaf/v1 openings; the proof pins its committed words to
                            the now-public values by public linear constraints.  The verifier learns the tensors: only
                            acceptable for tensors that are public (a policy knob: ``weights_public``).
(c) ``leaf_v2_pcs_hash``    protocol change: rows/columns become leaves under the proof system's hash (Poseidon2 over
                            BabyBear, width 16 or 24, salted); the tensor digest is a Poseidon2 Merkle root over row
                            leaves and is disclosed; the ids tree above stays SHA-256 and is folded natively by the
                            verifier.  The proof still hashes the sampled rows in-circuit, but algebraically.
(d) ``algebraic``           protocol change ``leaf/v2`` (``reference.leaf_v2``, ``LEAF_V2.md``): the tensor is a Ligero
                            tableau (RS-encoded rows with k - l randomizers, Merkle over columns); the leaf digest is the
                            chunk roots' hash; the proof binds its witness rows by a cross-tableau IRS + linear test on
                            its existing Ligero round.  Nothing is hashed in-circuit.
(e)                         the ZK verdict per option is in ``PRIVACY`` and the report.

Sharing (:func:`sharing_from_bench`).  Leaf hashing / row binding is paid once per *distinct* leaf, and the frozen
input set has a structure: 24 captured activation rows x 144 weight rows each (3 456 real VUs) + 640 corners; 2 598
distinct weight rows in 18 weight matrices; every real VU's ``y`` is one word of one of 24 output rows.  Three regimes
are priced: ``bench`` (that structure extrapolated to 4 096 real VUs), ``decode_step`` (one token, 4 096 coordinates of
one projection: x and y shared by all, w all distinct) and ``worst`` (nothing shared).
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from verity_numerical.explore import authentication as auth_v1
from verity_numerical.explore import fields as fl
from verity_numerical.security import accounting as acc

__all__ = [
    "AUTH_LEVELS",
    "CITATIONS",
    "GADGETS",
    "OPTIONS",
    "PRIVACY",
    "REPO",
    "SHARINGS",
    "AuthDelta",
    "Gadget",
    "Sharing",
    "capture_cost",
    "delta",
    "main",
    "option_table",
    "report",
    "sharing_from_bench",
    "soundness_with_auth",
]

REPO = Path(__file__).resolve().parents[5]
BENCH = REPO / "fixtures" / "bench-instances" / "v1"
K_TARGET, B_TARGET = 1536, 4096
UNITS_PER_VU = K_TARGET // 16
WORD_BYTES = 2
ELEMENT_BYTES = 4                    # BabyBear element on the wire / in a column leaf
DIGEST_BYTES = 32

AUTH_LEVELS = ("none", "in_circuit_sha256", "leaf_v2_pcs_hash", "algebraic")
OPTIONS = ("in_circuit_sha256", "verifier_openings", "leaf_v2_pcs_hash", "algebraic")

CITATIONS: dict[str, str] = {
    "sha256_lookup_air": "T. Coratger, 'SHA-256 AIR for Stwo' design note (hackmd.io/@tcoratger/HJFGugU0Ze, read 2026-09-22): lookup-based "
                         "M31 layout, schedule 48 x (22 trace + 8 lookups) + compression 64 x (52 trace + 21 lookups) = 4 384 trace cells + "
                         "1 728 LogUp interactions ~ 7 840 cells per compression (L counted as 2T).",
    "sha256_p3_air": "Plonky3 p3-sha256-air 0.7.0-rc.1 (docs.rs, read 2026-09-22): one compression per row, 7 488 columns, bit-decomposed "
                     "sigma/Ch/Maj, packed 16-bit-limb additions, max constraint degree 3, no lookups.",
    "sha256_r1cs": auth_v1.CITATIONS["sha256_r1cs"],
    "sp1_sha_air": auth_v1.CITATIONS["sp1_sha_air"],
    "poseidon2_babybear": "Plonky3 p3-baby-bear poseidon2.rs (read 2026-09-22): BabyBear Poseidon2 S-box x^7, R_F = 8, R_P = 13 (width 16) / "
                          "21 (width 24); hash-gpu lane's fused CUDA kernel matches Plonky3's test vectors (reports/hash_gpu.md).",
    "hash_gpu": "backends/numerical/reports/hash_gpu.md, run r20260922-054241-b966 (RTX 4090): full-tree SHA-256 446 GB/s (B shape), "
                "Blake3 359 GB/s, Poseidon2/BabyBear w16 57.7 GB/s, w24 34.2 GB/s; generic 32 B-leaf trees SHA-256 146 GB/s.",
    "ligero_zk": "Ames-Hazay-Ishai-Venkitasubramaniam, Ligero (ePrint 2022/1608) Lemma 4.15: with k - l randomizers per row any t <= k - l "
                 "opened columns are uniform; the linear/quadratic test responses are hidden by one blinding row each.",
    "leaf_v1": auth_v1.CITATIONS["leaf_scheme"],
    "bench_sharing": "fixtures/bench-instances/v1/vu-k1536.index.json + seeds/seeds.json: 3 456 real VUs = 24 seeds x 144 weight rows; "
                     "2 598 distinct (op, j) weight rows over 18 weight matrices; 640 corners with their own operands.",
    "tensor_model": "backends/numerical/reports/tensor.md (explore.tensor): H100 INT8 dense 989.5 T MAC/s x utilisation 0.172; RS n=16384/k=4096 "
                    "as a two-stage matrix DFT: 4096 x 128 + 16384 x 256 base MACs per row (x L^2 = 16 limb MACs on BabyBear).",
}


# ---------------------------------------------------------------------------------------------------------------------
# Hash gadgets in circuit
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Gadget:
    """One hash invocation (a SHA-256 compression or a Poseidon2 permutation) as proof work.

    ``gates_a``: sumcheck elements A-GKR's GKR sweeps per invocation (one per constraint / trace cell);
    ``lookups_a``: logUp leaves (fractional-GKR) per invocation; ``committed_a``: elements A commits (hints);
    ``committed_b``: elements B-Ligero's Ligero commits (trace cells, helper columns x extension degree for lookups);
    ``products_b``: quadratic rows' worth of products; ``payload_bytes``: message bytes absorbed per invocation;
    ``layers``: multiplicative depth per invocation (A's sequential depth if the chain is not hinted)."""

    name: str
    gates_a: int
    lookups_a: int
    committed_a: int
    committed_b_base: int
    helper_lookups_b: int
    products_b: int
    payload_bytes: int
    layers: int
    citation: str

    def committed_b(self, d: int) -> int:
        return self.committed_b_base + self.helper_lookups_b * d


SHA256_LOOKUP_AIR = Gadget("SHA-256, lookup-based AIR (Stwo-style)", gates_a=4_384, lookups_a=1_728, committed_a=4_384,
                           committed_b_base=4_384, helper_lookups_b=1_728, products_b=4_384, payload_bytes=64, layers=2 * 64,
                           citation="sha256_lookup_air")
SHA256_P3_AIR = Gadget("SHA-256, Plonky3 p3-sha256-air (no lookups)", gates_a=7_488, lookups_a=0, committed_a=7_488,
                       committed_b_base=7_488, helper_lookups_b=0, products_b=7_488, payload_bytes=64, layers=3 * 64,
                       citation="sha256_p3_air")
SHA256_R1CS = Gadget("SHA-256, R1CS (26 170 nonlinear)", gates_a=26_170, lookups_a=0, committed_a=10_496,
                     committed_b_base=3 * 26_170, helper_lookups_b=0, products_b=26_170, payload_bytes=64, layers=3 * 64,
                     citation="sha256_r1cs")


def _poseidon2(width: int, partial: int, name: str) -> Gadget:
    sboxes = 8 * width + partial
    muls = 4 * sboxes                              # x^7: x^2, x^4, x^3, x^7
    lin = (8 + partial) * width                   # one output wire per state element per round (MDS is linear)
    rate = width - 8                               # capacity 8 elements (248 bits) as in Plonky3's BabyBear sponges
    return Gadget(name, gates_a=muls + lin, lookups_a=0, committed_a=0, committed_b_base=3 * muls, helper_lookups_b=0,
                  products_b=muls, payload_bytes=rate * WORD_BYTES, layers=(8 + partial) * 3, citation="poseidon2_babybear")


POSEIDON2_BB16 = _poseidon2(16, 13, "Poseidon2/BabyBear width 16 (rate 8 words)")
POSEIDON2_BB24 = _poseidon2(24, 21, "Poseidon2/BabyBear width 24 (rate 16 words)")

GADGETS: dict[str, Gadget] = {
    "sha256_lookup_air": SHA256_LOOKUP_AIR, "sha256_p3_air": SHA256_P3_AIR, "sha256_r1cs": SHA256_R1CS,
    "poseidon2_bb16": POSEIDON2_BB16, "poseidon2_bb24": POSEIDON2_BB24,
}


# ---------------------------------------------------------------------------------------------------------------------
# Sharing structure of a batch
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sharing:
    """Distinct leaves / rows a B-VU batch touches.  ``x_leaves``: distinct activation rows (one per token per GEMM);
    ``w_rows``: distinct weight rows; ``y_leaves``: distinct output rows; ``w_matrices``: distinct weight matrices;
    ``w_chunks``: ``R -> touched chunks of R rows`` for the algebraic layout (``sum_rows``: rows in touched chunks)."""

    name: str
    B: int
    x_leaves: int
    w_rows: int
    y_leaves: int
    w_matrices: int
    w_chunks: dict[int, tuple[int, int]] = field(default_factory=dict)
    note: str = ""

    def per_vu(self, role: str) -> float:
        n = {"x": self.x_leaves, "w": self.w_rows, "y": self.y_leaves}[role]
        return n / self.B

    def touched_rows(self, R: int) -> int:
        """Tableau rows the algebraic openings cover: x/y tensors (one chunk of <= R rows each; a decode-step activation
        tensor holds one row per token, taken as 1 here) plus the weight chunks."""
        if R in self.w_chunks:
            w_rows = self.w_chunks[R][1]
        else:
            w_rows = min(self.w_rows * R, self.w_matrices * 2048)     # bracket: every row its own chunk vs whole matrices
        return self.x_leaves + self.y_leaves + w_rows

    def touched_chunks(self, R: int) -> int:
        return self.x_leaves + self.y_leaves + (self.w_chunks[R][0] if R in self.w_chunks else self.w_rows)


def sharing_from_bench(B: int = B_TARGET, Rs: Sequence[int] = (1, 16, 64, 256, 4096)) -> Sharing:
    """The frozen input set's real structure, extrapolated from its 3 456 real VUs to ``B`` real VUs (the 640
    corners are synthetic and have no Verity leaves)."""
    index = json.loads((BENCH / "vu-k1536.index.json").read_text())
    seeds = json.loads((BENCH / "seeds" / "seeds.json").read_text())["seeds"]
    names = list(seeds)
    real = [r for r in index["rows"] if r[1][0] == "real"]
    x_ids = {r[2] for r in real}
    w_keys = {(seeds[names[r[2]]]["op"], r[3]) for r in real}
    mats = {seeds[names[r[2]]]["op"] for r in real}
    scale = B / len(real)
    chunks: dict[int, tuple[int, int]] = {}
    for R in Rs:
        touched = {(op, j // R) for op, j in w_keys}
        rows = 0
        for op, c in touched:
            N = next(s["N"] for s in seeds.values() if s["op"] == op)
            rows += min(R, N - c * R)
        chunks[R] = (round(len(touched) * scale), round(rows * scale))
    return Sharing("bench", B, x_leaves=round(len(x_ids) * scale), w_rows=round(len(w_keys) * scale), y_leaves=round(len(x_ids) * scale),
                   w_matrices=len(mats), w_chunks=chunks,
                   note=f"{len(real)} real VUs: {len(x_ids)} activation rows x 144 weight rows; {len(w_keys)} distinct weight rows in "
                        f"{len(mats)} matrices; scaled x{scale:.3f} to {B} real VUs")


def _decode_step(B: int = B_TARGET) -> Sharing:
    return Sharing("decode_step", B, 1, B, 1, 1, {R: (-(-B // R), B) for R in (1, 16, 64, 256, 4096)},
                   note="one token, all B coordinates of one projection (N >= B): x and the output row shared by every VU")


def _worst(B: int = B_TARGET) -> Sharing:
    return Sharing("worst", B, B, B, B, B, {R: (B, B * R) for R in (1, 16, 64, 256, 4096)},
                   note="one coordinate per token per matrix: nothing shared (chunk rows bracketed as B x R)")


SHARINGS: dict[str, Sharing] = {"bench": sharing_from_bench(), "decode_step": _decode_step(), "worst": _worst()}

#: leaf/v1 leaf structure of the K=1536 obligation (``explore.authentication`` reads it from the fixture).
V1 = auth_v1.read_openings()


# ---------------------------------------------------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class AuthDelta:
    """What one option adds to one design's batch, in ``explore.tensor``'s units (per B-VU batch)."""

    option: str
    gadget: str
    sharing: str
    weights_public: bool
    committed_elements: float = 0.0          # extra committed trace elements (A: hints; B: trace cells + helpers)
    gkr_elements: float = 0.0                # A: extra checker-GKR sumcheck elements (one per constraint)
    lookup_leaves: float = 0.0               # A: extra logUp leaves; B: extra helper products
    products_b: float = 0.0                  # B: extra quadratic-row products
    matmul_macs: float = 0.0                 # extra limb MACs outside the above (row combinations over tensor rows)
    simt_field_mults: float = 0.0
    prover_hash_bytes: float = 0.0           # extra native hashing by the prover
    depth_delta: int = 0
    proof_bytes: float = 0.0                 # extra prover -> verifier bytes
    verifier_hash_bytes: float = 0.0
    verifier_field_ops: float = 0.0
    prg_bytes: float = 0.0
    notes: list[str] = field(default_factory=list)

    def per_vu(self, B: int = B_TARGET) -> dict:
        return {k: v / B for k, v in asdict(self).items() if isinstance(v, int | float) and not isinstance(v, bool)}


def _v1_compressions(sharing: Sharing, roles: Sequence[str], include_path: bool = True) -> tuple[float, dict[str, float]]:
    """leaf/v1 SHA-256 compressions per VU (leaf + node/lift path) after sharing, by role."""
    by: dict[str, float] = {}
    for leaf in V1.leaves:
        if leaf.role not in roles:
            continue
        comps = leaf.leaf_compressions + (leaf.path_compressions if include_path else 0)
        by[leaf.role] = comps * sharing.per_vu(leaf.role)
    return sum(by.values()), by


def _public_linear_words(sharing: Sharing, roles: Sequence[str]) -> float:
    """Words per VU pinned to public values by linear constraints (option (b) for a public tensor)."""
    return sum(leaf.words_read * sharing.per_vu(leaf.role) for leaf in V1.leaves if leaf.role in roles)


def delta(option: str, candidate: str, d: int, B: int = B_TARGET, sharing: Sharing | str = "bench",
          weights_public: bool = False, gadget: str | None = None, ligero: acc.LigeroParams | None = None,
          L: int = 4, R: int = 64, code_n: int = fl.N_CODE) -> AuthDelta:
    """The batch delta of ``option`` for candidate ``A`` or ``B`` (extension degree ``d``, ``L`` limbs per base element)."""
    sh = SHARINGS[sharing] if isinstance(sharing, str) else sharing
    if option == "none":
        return AuthDelta("none", "", sh.name, weights_public)
    if option not in OPTIONS:
        raise ValueError(f"unknown option {option!r}; one of {OPTIONS}")
    private_roles = ("x", "y") if weights_public else ("x", "w", "y")
    public_roles = ("w",) if weights_public else ()
    out = AuthDelta(option, gadget or "", sh.name, weights_public)
    lig = ligero or acc.LigeroParams(n=fl.N_CODE, k=fl.K_CODE, l=fl.L_PAYLOAD, t=191)

    if option == "verifier_openings":
        # everything public: the verifier re-hashes leaf/v1 leaves natively; the proof pins each word by a public
        # linear constraint (A: one more linkage constraint in the geometric batch; B: one linear constraint)
        comps, _ = _v1_compressions(sh, ("x", "w", "y"))
        words = _public_linear_words(sh, ("x", "w", "y"))
        out.gadget = "sha256 native (verifier)"
        out.simt_field_mults = words * B * d                   # rho^k * word accumulations, F_c x F_t
        out.verifier_hash_bytes = comps * B * 64
        out.verifier_field_ops = words * B * d
        out.proof_bytes = sum((leaf.value_bytes + leaf.depth * DIGEST_BYTES) * sh.per_vu(leaf.role) for leaf in V1.leaves) * B
        out.notes.append("x, W rows and whole output rows disclosed to the verifier: NOT private (NON_ZK at best); fine for public tensors only")
        return out

    # the public part (weights, if public) always goes through verifier-side openings + public linear constraints
    if public_roles:
        comps_pub, _ = _v1_compressions(sh, public_roles)
        words_pub = _public_linear_words(sh, public_roles)
        out.simt_field_mults += words_pub * B * d
        out.verifier_hash_bytes += comps_pub * B * 64
        out.verifier_field_ops += words_pub * B * d
        out.proof_bytes += sum((leaf.value_bytes + leaf.depth * DIGEST_BYTES) * sh.per_vu(leaf.role) for leaf in V1.leaves if leaf.role in public_roles) * B
        out.notes.append("weights public: W rows opened to the verifier (leaf/v1, native SHA-256) and pinned by public linear constraints")

    if option == "in_circuit_sha256":
        g = GADGETS[gadget or "sha256_lookup_air"]
        out.gadget = g.name
        comps, by = _v1_compressions(sh, private_roles)
        inv = comps * B
        out.gkr_elements = inv * g.gates_a
        out.lookup_leaves = inv * g.lookups_a
        out.committed_elements = inv * (g.committed_a if candidate == "A" else g.committed_b(d))
        out.products_b = inv * g.products_b
        out.depth_delta = 2 * g.layers if candidate == "A" else 0     # chaining states hinted: one leaf block + one node in depth
        out.notes.append(f"{comps:.2f} SHA-256 compressions per VU after sharing ({', '.join(f'{k} {v:.2f}' for k, v in by.items())}); "
                         "whole leaf/v1 leaves + node/lift paths in circuit; no protocol change")
        return out

    if option == "leaf_v2_pcs_hash":
        g = GADGETS[gadget or "poseidon2_bb16"]
        out.gadget = g.name
        perms = 0.0
        by = {}
        for leaf in V1.leaves:
            if leaf.role not in private_roles:
                continue
            words = leaf.value_bytes // WORD_BYTES
            rows_in_tensor = {"x": 256, "w": 2048, "y": 256}[leaf.role]        # tokens per step / matrix rows (assumption)
            per_leaf = -(-(words + 1) // (g.payload_bytes // WORD_BYTES)) + math.ceil(math.log2(rows_in_tensor))   # + salt, + row tree
            by[leaf.role] = per_leaf * sh.per_vu(leaf.role)
            perms += by[leaf.role]
        inv = perms * B
        out.gkr_elements = inv * g.gates_a
        out.committed_elements = inv * (g.committed_a if candidate == "A" else g.committed_b(d))
        out.products_b = inv * g.products_b
        out.depth_delta = 2 * g.layers if candidate == "A" else 0
        out.verifier_hash_bytes = sum(leaf.depth * 2 * 64 * sh.per_vu(leaf.role) for leaf in V1.leaves if leaf.role in private_roles) * B
        out.proof_bytes = sum((DIGEST_BYTES + leaf.depth * DIGEST_BYTES) * sh.per_vu(leaf.role) for leaf in V1.leaves if leaf.role in private_roles) * B
        out.notes.append(f"{perms:.2f} Poseidon2 permutations per VU after sharing ({', '.join(f'{k} {v:.2f}' for k, v in by.items())}): salted row "
                         "leaves + row tree in circuit; tensor digest disclosed, ids-tree path folded natively; protocol change (leaf/v2-hash)")
        return out

    # algebraic (leaf/v2 Ligero tableau): cross-tableau IRS + linear test on the proof's own Ligero round
    dl = d * L
    # rows the verifier reads (claimed rows: whole x / w rows, the y row of the word) vs rows inside the opened column
    # leaves (every row of a touched chunk: bytes to send and hash, not field work)
    claimed_rows = (sh.x_leaves + sh.y_leaves) + (0 if weights_public else sh.w_rows)
    rows = sh.touched_rows(R) if not weights_public else sh.x_leaves + sh.y_leaves
    touched_chunks = sh.touched_chunks(R) if not weights_public else sh.x_leaves + sh.y_leaves
    out.gadget = f"ligero-rs n={lig.n} k={lig.k} l={lig.l} t={lig.t} R={R}"
    # prover: two row combinations over the claimed tensor rows (IRS: r'_i U_i; linear: alpha_i U_i, then one pointwise
    # multiply by the shared interpolant G(X) of the geometric position coefficients rho^j -- one NTT for the whole test,
    # PROTOCOL A 12.2's rho_geo trick), F_t x F_c
    out.matmul_macs = 2 * claimed_rows * lig.n * L * dl + (fl.K_CODE * 128 + fl.N_CODE * 256) * L * dl
    # the equality constraints join the linear test: the proof-side words of every VU x d (the geometric coefficients)
    words = sum(leaf.words_read for leaf in V1.leaves if leaf.role in private_roles)     # per VU, not amortised: every VU's own claim
    out.simt_field_mults += words * B * d
    # proof bytes: t columns of every touched chunk (R rows x 4 B) + paths, one combined IRS polynomial (k elements)
    depth_chunk = math.ceil(math.log2(lig.n))
    out.proof_bytes += lig.t * (rows * ELEMENT_BYTES + touched_chunks * depth_chunk * DIGEST_BYTES) + lig.k * ELEMENT_BYTES
    # verifier: fold t column leaves per touched chunk (hash R x 4 B each + path), evaluate the two combinations at Q
    out.verifier_hash_bytes += lig.t * (rows * ELEMENT_BYTES + touched_chunks * depth_chunk * 64)
    out.verifier_field_ops += 2 * lig.t * claimed_rows * d + 2 * lig.k * d
    out.prg_bytes = 0.0        # the tensor randomizers are the committer's (capture time), not the prover's
    out.notes.append(f"{claimed_rows} claimed tensor rows; {rows} rows in {touched_chunks} chunks (R={R}) opened at t={lig.t} columns; nothing hashed in circuit; "
                     "depth +0 (rides the existing Ligero round); protocol change leaf/v2 (LEAF_V2.md)")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Privacy verdicts, soundness, capture-time cost
# ---------------------------------------------------------------------------------------------------------------------

PRIVACY: dict[str, dict[str, str]] = {
    "in_circuit_sha256": {
        "verdict": "private", "sees": "roots (in O) and the verdict; the leaf bytes and digests stay inside the proof",
        "simulatable_because": "no new prover message: the hash constraints are checker constraints and inherit the design's HVZK/step-0 argument",
        "assumption": "none beyond the design's (CRH of H); no hiding assumption on SHA-256 since digests are never disclosed",
    },
    "verifier_openings": {
        "verdict": "NOT private (public tensors only)", "sees": "the opened leaves in the clear: x rows, W rows, whole output rows",
        "simulatable_because": "n/a -- acceptable only for tensors that are public by policy (open-weights models: W); never for activations",
        "assumption": "n/a",
    },
    "leaf_v2_pcs_hash": {
        "verdict": "private (computational)", "sees": "the tensor digest (Poseidon2 Merkle root over salted row leaves), the SHA-256 ids path, the verdict",
        "simulatable_because": "row leaves are H(row || salt) with a fresh 31-bit x 8 salt: a hiding commitment under Poseidon2 as a "
                               "(pseudo)random function; the digest is then independent of the rows; the in-circuit part inherits the design's ZK",
        "assumption": "hiding of salted Poseidon2 (beyond CRH): an extra assumption to state; salt per row must be fresh per commitment",
    },
    "algebraic": {
        "verdict": "private (statistical, per commitment lifetime)",
        "sees": "t opened columns of every touched chunk, two blinded combination polynomials (v: degree < k, q: degree < 2k-1), the chunk roots, the verdict",
        "simulatable_because": "t <= k - l columns of rows with fresh randomizers are uniform (Ligero Lemma 4.15); v and q are one-time-padded by the "
                               "proof's blinding rows; Q and the coefficients are committed at step 0 (HM96), so a malicious verifier gains nothing "
                               "from adaptivity; the ids-tree path is over digests only",
        "assumption": "the randomizer budget: the SUM of t over every opening of one commitment must stay <= k - l (weights re-randomised per "
                      "session or public; l chosen for the number of proofs per runtime commitment); HM96 binding for step 0 as in ZK_CONSTRUCTION.md",
    },
}


def soundness_with_auth(target_log2: int = -128, B: int = B_TARGET) -> dict:
    """The accountants' budgets (BabyBear^6, the tensor model's parameters) with the algebraic option's extra terms:
    the equality constraints and the tensor rows join the existing linear / IRS tests, so no new *query* term appears;
    the new *field* terms are Schwartz-Zippel over the enlarged constraint count and one batching coefficient per
    tensor row."""
    out = {}
    for cand in ("A", "B"):
        dgr, t, _ = fl.first_small_field_configuration(target_log2, fl.BABYBEAR, cand, K_TARGET, B)
        Ft = fl.BABYBEAR.trace
        Fc = Ft.extension(dgr)
        lig = acc.LigeroParams(n=fl.N_CODE, k=fl.K_CODE, l=fl.L_PAYLOAD, t=t)
        sfc = fl.small_field_census(Ft)
        tables = sfc.tables()
        if cand == "A":
            p = acc.CandidateA(K=K_TARGET, B=B, trace_field=Ft, challenge_field=Fc, ligero=lig, hash_queries=fl.HASH_QUERIES, tables=tables,
                               inputs_per_unit=sfc.columns_committed_a, query_wires_per_unit=sfc.queries_on_computed_wires,
                               coin_commitment_bits=160, leaf_commitment_bits=160)
            bud = acc.account_candidate_a(p)
        else:
            p = acc.CandidateB(K=K_TARGET, B=B, trace_field=Ft, challenge_field=Fc, ligero=lig, hash_queries=fl.HASH_QUERIES, tables=tables,
                               witness_per_unit=sfc.columns_committed_b, linear_per_unit=sfc.lin_total + 2 * sfc.mul,
                               quadratic_per_unit=sfc.mul, coin_commitment_bits=160, leaf_commitment_bits=160)
            bud = acc.account_candidate_b(p)
        before = acc.log2(bud.total())
        sh = SHARINGS["bench"]
        eq_constraints = B * sum(leaf.words_read for leaf in V1.leaves)
        rows = sh.touched_rows(64)
        bud.add("batching", f"auth: {eq_constraints} equality constraints join the linear test (Schwartz-Zippel over the geometric coefficient)",
                Fraction(eq_constraints - 1, Fc.size))
        bud.add("encoding_opening", f"auth: {rows} tensor tableau rows join the IRS combination (one uniform coefficient each: field term)",
                Fraction(rows + 1, Fc.size))
        bud.add("commitment_hash", "auth: leaf/v2 chunk roots and column leaves under the same H (already inside the generic q^2/2^h term)", 0)
        after = acc.log2(bud.total())
        out[cand] = {"extension_degree": dgr, "t": t, "soundness_log2_before": round(before, 2), "soundness_log2_after": round(after, 2),
                     "delta_bits": round(after - before, 4), "meets_target": bud.meets(target_log2),
                     "added_terms": [{"component": tm.component, "name": tm.name, "log2": round(tm.log2, 1)} for tm in bud.terms[-3:]],
                     "shared_query_terms": "IRS (1-e/n)^t and linear ((k+l)/n)^t are the design's own; the auth rows and constraints ride them",
                     "note": "in-circuit options add constraints only (no new soundness term); verifier_openings adds none"}
    return out


def capture_cost(tokens: int = 256, layers: int = 28, hidden: int = 1536, intermediate: int = 8960, qkv_n: int = 2048,
                 params: float = 1.54e9, L: int = 4, hash_rate: float = 446e9, peak_macs: float = 989.5e12 / 2 * 2, util: float = 0.172,
                 code_n: int = fl.N_CODE, code_k: int = fl.K_CODE, l_payload: int = fl.L_PAYLOAD) -> dict:
    """What the vLLM-side committer computes per decode step under leaf/v2 (option (d)), Qwen2.5-1.5B shapes:
    per layer the four GEMM input rows (hidden, hidden, hidden, intermediate) and four output rows (qkv N, hidden,
    2 x intermediate, hidden) per token; weights once per (re)commitment.  Encoding as the tensor model's matrix DFT
    (``rs_matrix_ntt``), hashing at the hash-gpu SHA-256 rate; ``peak_macs`` is the H100 INT8 dense MAC rate."""
    rows_per_token = layers * sum(-(-c // l_payload) for c in (hidden, hidden, hidden, intermediate, qkv_n, hidden, 2 * intermediate, hidden))
    words_per_token = layers * (3 * hidden + intermediate + qkv_n + hidden + 2 * intermediate + hidden)
    macs_per_row = (code_k * 128 + code_n * 256) * L * L
    macs_per_row_brakedown = code_n * 20 * L * L
    act_rows = rows_per_token * tokens
    weight_rows = params / hidden * (-(-hidden // l_payload))
    rate = 989.5e12 * util          # MAC/s at the realistic utilisation (H100)

    def cost(rows: float) -> dict:
        hashed = rows * code_n * ELEMENT_BYTES
        return {"tableau_rows": rows, "encode_limb_macs": rows * macs_per_row, "encode_seconds_h100_0.172": rows * macs_per_row / rate,
                "encode_seconds_brakedown": rows * macs_per_row_brakedown / (rate * 0.092 / 0.172),
                "hash_bytes": hashed, "hash_seconds_sha256_4090": hashed / hash_rate, "hash_seconds_sha256_h100_scaled": hashed / (hash_rate * 3.32),
                "prg_bytes": rows * (code_k - l_payload) * ELEMENT_BYTES, "leaf_v1_sha256_seconds_same_bytes": rows * l_payload * WORD_BYTES / hash_rate}

    return {"tokens": tokens, "words_per_token": words_per_token, "tableau_rows_per_token": rows_per_token,
            "activations_per_step": cost(act_rows), "weights_per_commitment": cost(weight_rows),
            "decode_step_reference_ms": "a 1.5B-parameter decode step on an H100 is ~10-20 ms at this batch (order of magnitude, not measured here)",
            "note": "rate 1/4 code as the proofs use; n=8192 halves every line; the Brakedown line is the gather-bound expander at 0.092 utilisation"}


# ---------------------------------------------------------------------------------------------------------------------
# Complete numbers through explore.tensor
# ---------------------------------------------------------------------------------------------------------------------

def option_table(target_log2: int = -128, util: float = 0.172, hash_rate_key: str = "blake3_simt", sharing: str = "bench",
                 designs: Sequence[str] = ("A", "B"), weights_public_variants: Sequence[bool] = (False, True)) -> list[dict]:
    """Every (design x option x gadget x weights_public) at the H100 realistic point: the malicious-verifier ZK number
    with and without authentication, its delta, proof bytes, verifier work, depth."""
    from verity_numerical.explore import tensor as t

    rows = []
    fs = t.FIELDS["babybear"]
    for dn in designs:
        design = t.DESIGNS[dn]
        base_nonzk = t.price_design(design, fs, target_log2, zk="none")
        base = t.price_design(design, fs, target_log2, zk="malicious")
        hr = t.HASH_RATES[hash_rate_key]["bytes_per_s"]
        tb0 = base_nonzk.times(t.GPUS["H100"], "H100", util, hr, fs=fs)
        tb = base.times(t.GPUS["H100"], "H100", util, hr, fs=fs)
        for opt, gadgets in (("in_circuit_sha256", ("sha256_lookup_air", "sha256_r1cs")), ("verifier_openings", (None,)),
                             ("leaf_v2_pcs_hash", ("poseidon2_bb16", "poseidon2_bb24")), ("algebraic", (None,))):
            for g in gadgets:
                for wp in (weights_public_variants if opt != "verifier_openings" else (True,)):
                    p = t.price_design(design, fs, target_log2, zk="malicious", authentication=opt, auth_gadget=g, auth_sharing=sharing,
                                       weights_public=wp)
                    tt = p.times(t.GPUS["H100"], "H100", util, hr, fs=fs)
                    dlt = p.auth_delta
                    rows.append({
                        "design": dn, "option": opt, "gadget": dlt.gadget, "weights_public": wp, "sharing": sharing,
                        "overhead_nonzk": tb0["overhead.vs_native_peak"], "overhead_zk": tb["overhead.vs_native_peak"],
                        "overhead_complete": tt["overhead.vs_native_peak"],
                        "delta_vs_zk": tt["t.total"] / tb["t.total"] - 1.0, "delta_vs_nonzk": tt["t.total"] / tb0["t.total"] - 1.0,
                        "committed_elements_delta_per_vu": dlt.committed_elements / t.B_TARGET,
                        "proof_bytes_delta_per_vu": dlt.proof_bytes / t.B_TARGET,
                        "verifier_hash_bytes_per_vu": dlt.verifier_hash_bytes / t.B_TARGET, "verifier_field_ops_per_vu": dlt.verifier_field_ops / t.B_TARGET,
                        "depth": p.depth, "depth_delta": dlt.depth_delta, "privacy": PRIVACY[opt]["verdict"] if not (wp and opt != "verifier_openings") else
                        PRIVACY[opt]["verdict"] + " (x, y); W public",
                        "protocol_change": opt in ("leaf_v2_pcs_hash", "algebraic"), "notes": dlt.notes,
                    })
    return rows


def report() -> dict:
    tbl = option_table()
    sens = []
    from verity_numerical.explore import tensor as t
    fs = t.FIELDS["babybear"]
    for dn in ("A", "B"):
        base = t.price_design(t.DESIGNS[dn], fs, -128, zk="malicious").times(t.GPUS["H100"], "H100", 0.172, t.HASH_RATES["blake3_simt"]["bytes_per_s"], fs=fs)
        for sh in SHARINGS:
            for opt in ("in_circuit_sha256", "algebraic"):
                for wp in (False, True):
                    p = t.price_design(t.DESIGNS[dn], fs, -128, zk="malicious", authentication=opt, auth_sharing=sh, weights_public=wp)
                    tt = p.times(t.GPUS["H100"], "H100", 0.172, t.HASH_RATES["blake3_simt"]["bytes_per_s"], fs=fs)
                    sens.append({"design": dn, "sharing": sh, "option": opt, "weights_public": wp, "overhead_complete": tt["overhead.vs_native_peak"],
                                 "delta_vs_zk": tt["t.total"] / base["t.total"] - 1.0, "proof_bytes_delta_per_vu": p.auth_delta.proof_bytes / t.B_TARGET})
    best = {}
    for dn in ("A", "B"):
        cands = [r for r in tbl if r["design"] == dn and not r["weights_public"] and r["privacy"].startswith("private")]
        best[dn] = min(cands, key=lambda r: r["overhead_complete"])
    # algebraic proof bytes vs chunk size R (bench sharing, private weights): column leaves grow with R, paths shrink
    r_sweep = []
    for R in (1, 16, 64, 256, 4096):
        dl = delta("algebraic", "A", 6, B_TARGET, "bench", False, None, None, 4, R)
        r_sweep.append({"R": R, "proof_bytes_per_vu": dl.proof_bytes / B_TARGET, "verifier_hash_bytes_per_vu": dl.verifier_hash_bytes / B_TARGET,
                        "touched_chunks": SHARINGS["bench"].touched_chunks(R), "rows_in_touched_chunks": SHARINGS["bench"].touched_rows(R)})
    return {
        "lane": "auth-integration", "date": "2026-09-22", "B": B_TARGET, "K": K_TARGET, "target": "2^-128", "hardware": "H100-SXM model @0.172, Blake3 50 GB/s",
        "sharing": {k: asdict(v) | {"w_chunks": {str(R): list(v) for R, v in v.w_chunks.items()}} for k, v in SHARINGS.items()},
        "gadgets": {k: asdict(g) for k, g in GADGETS.items()},
        "options": tbl, "sharing_sensitivity": sens, "best_private_per_design": best, "algebraic_chunk_size_sweep": r_sweep,
        "soundness": soundness_with_auth(), "capture_cost": capture_cost(), "privacy": PRIVACY,
        "breakthrough_criterion": {"rule": "a viable private option within 20% of the numerical-only (non-ZK, auth excluded) number",
                                   "met": {dn: best[dn]["delta_vs_nonzk"] < 0.20 for dn in best}},
        "citations": CITATIONS,
        "assumptions": [
            "in-circuit SHA-256 priced as the Stwo-style lookup AIR (7 840 cells / compression; 4 384 constraints + 1 728 lookups) and the R1CS circuit (26 170) as bracket; a cell is one sumcheck element (A) / one committed element (B)",
            "Poseidon2/BabyBear: 4 multiplications per x^7 S-box, one linear wire per state element per round; rate = width/2 words (one 16-bit word per element); salt 8 elements per row leaf",
            "algebraic: two row combinations + one size-n interpolant per touched tensor row (matrix-DFT price), t=191 columns of every touched chunk of R=64 rows opened; verifier folds them natively",
            "bench sharing = the frozen instance set's real structure scaled to 4 096 real VUs; decode_step and worst bracket it",
            "leaf_v2_pcs_hash rows-per-tensor for the row tree: 256 tokens (activations), 2 048 rows (weights): assumptions",
            "weights_public: W rows opened to the verifier under leaf/v1 and pinned by public linear constraints (no privacy for W by policy)",
        ],
    }


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    rep = report()
    print(f"sharing bench: {SHARINGS['bench'].note}")
    print(f"{'design':6} {'option':18} {'gadget':42} {'Wpub':>4} {'ovh nonZK':>10} {'ovh ZK':>10} {'complete':>10} {'d vs ZK':>8} {'bytes/VU':>9} {'depth':>5}  privacy")
    for r in rep["options"]:
        print(f"{r['design']:6} {r['option']:18} {r['gadget'][:42]:42} {'yes' if r['weights_public'] else 'no':>4} {_fmt(r['overhead_nonzk']):>10} "
              f"{_fmt(r['overhead_zk']):>10} {_fmt(r['overhead_complete']):>10} {r['delta_vs_zk']:>+8.1%} {_fmt(r['proof_bytes_delta_per_vu']):>9} "
              f"{r['depth']:>5}  {r['privacy']}")
    print("\nbest private:", {k: (v["option"], _fmt(v["overhead_complete"]), f"{v['delta_vs_nonzk']:+.2%}") for k, v in rep["best_private_per_design"].items()})
    print("soundness:", json.dumps(rep["soundness"], indent=1))
    if "--write" in args:
        out = REPO / "backends" / "numerical" / "reports" / "auth_integration.json"
        out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":       # pragma: no cover
    main()
