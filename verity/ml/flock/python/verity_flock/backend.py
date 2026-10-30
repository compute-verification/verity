"""The Flock backend family: pure binary-field proofs of the committed relation, hash leaves included, with no link.

One configured backend is one Table 1 row. :data:`FLOCK_FRAME_V3_BLAKE3` proves, for every VU of a GEMM-coordinate
subcircuit, that the output word committed in frame-v3 tree ``y`` is the subcircuit applied to the x row and W column
committed as ``blake3-keyed/row/v2`` row leaves in trees ``a`` and ``b``. The keyed-BLAKE3 row digests are computed in
the proof (Flock's BLAKE3 compression table, chunk chains wired in the circuit, chunk values public); the SHA-256
frame-v3 trees and the BLAKE3 parent nodes are computed natively by the verifier from the instance set. The transition
units are the pinned lowering of :mod:`verity_flock.lowering`, wired in the same circuit to the message words their
bytes are hashed from. Coins are the live verifier's (flock-128-r2: Fast100, two reps on one root).

Prover and verifier are ``flock-pure`` (``verity/ml/flock/live/src/bin/flock-pure.rs``, built into a flock b684b12
checkout with ``flock-link-b684b12.patch``; ``verity/ml/flock/pod/20-pure.sh``).
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from verity.ml.circuits.silicon.models import ADA_E4M3_M16N8K32, AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16, HOPPER_E4M3_K32
from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, Target

from . import lowering

__all__ = ["BACKENDS", "FLOCK_FRAME_V3_BLAKE3", "FLOCK_VLLM_V1", "Flock", "GemmSub", "Supported", "Unavailable", "Unsupported"]

#: Pipelines with a pinned lowering, by the model object the target names (imported: a rename fails at import time).
_LOWERED = {HOPPER_BF16_M16N8K16: "bf16-hopper", ADA_E4M3_M16N8K32: "fp8-ada", HOPPER_E4M3_K32: "fp8-hopper",
            AMPERE_BF16_M16N8K16: "bf16-ampere"}
_TARGETS = {BF16_HOPPER.name: "bf16-hopper", FP8_ADA_MMA.name: "fp8-ada", FP8_HOPPER_WGMMA.name: "fp8-hopper",
            FIRST.name: "bf16-ampere"}
_HERE = Path(__file__).resolve().parent
_LOWERING_SOURCES = ("lowering.py", "unit.py", "gf2.py")


@dataclass(frozen=True)
class GemmSub:
    """A GEMM-coordinate subcircuit: one target's tensor-core step chained over K (the Table 2 line's relation)."""

    target: Target
    k: int = 1536


@dataclass(frozen=True)
class Supported:
    relation: str


@dataclass(frozen=True)
class Unsupported:
    reason: str


@dataclass(frozen=True)
class Unavailable:
    reason: str


def lowering_version() -> str:
    """sha256 over the lowering's source files (``lowering.py``, ``unit.py``, ``gf2.py``), in that order."""
    h = hashlib.sha256()
    for name in _LOWERING_SOURCES:
        h.update(name.encode() + b"\0" + (_HERE / name).read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Flock:
    """A configured Flock backend. ``scheme`` is the core commitment scheme its statements bind."""

    scheme: str = "frame-v3/blake3-keyed/row/v2"
    profile: str = "flock-128-r2"
    binary: str = "flock-pure"
    #: ``union-wiring``: flock-pure's union circuit with Flock's wiring argument (CPU only; drill-down). ``block``:
    #: flock-gpu-link's ``flock-pure-block`` (one block R1CS per VU chunk, operand equality and the chain as in-block copy
    #: rows; Flock-CUDA and CPU), the configuration Table 2 cells come from.
    statement: str = "union-wiring"
    security: dict = field(default_factory=dict, compare=False, hash=False)

    def supports(self, sub: GemmSub):
        relation = _LOWERED.get(sub.target.pipeline)
        if relation is None or _TARGETS.get(sub.target.name) != relation:
            return Unsupported(f"{sub.target.name}: no pinned Flock lowering of its transition unit")
        pipe = lowering.PIPES[relation]
        row_bytes = sub.k * pipe.word_bits // 8
        if self.scheme == "vllm-v1":
            if row_bytes != 32 * (sub.k // pipe.k):
                return Unsupported(f"K = {sub.k}: flock-vllm-block needs every unit to read 32 bytes of each operand row")
        elif sub.k % pipe.k or row_bytes % 64 or row_bytes <= 1024:
            return Unsupported(f"K = {sub.k}: needs a multiple of {pipe.k} whose {pipe.word_bits}-bit rows are whole "
                               "BLAKE3 blocks and more than one chunk")
        if sub.k != 1536:
            return Unsupported(f"K = {sub.k}: the instance sets are K = 1536")
        if shutil.which(self.binary) is None:
            return Unavailable(f"{self.binary} isn't built on this machine (verity/ml/flock/pod/20-pure.sh)")
        return Supported(relation)

    def lower(self, sub: GemmSub) -> dict:
        """The unit's constraint system and the shape parameters ``flock-pure`` builds the circuit from. Refuses a
        lowering whose bytes are not the pinned ones."""
        s = self.supports(sub)
        if isinstance(s, Unsupported):
            raise ValueError(s.reason)
        relation = _LOWERED[sub.target.pipeline]
        pipe = lowering.PIPES[relation]
        text = lowering.netlist(relation)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != lowering.PINS[relation]:
            raise ValueError(f"{relation}: lowering {digest} is not the pinned {lowering.PINS[relation]}")
        return {"format": lowering.FORMAT, "relation": relation, "netlist": text, "sha256": digest, "k": sub.k,
                "units": sub.k // pipe.k, "row_bytes": sub.k * pipe.word_bits // 8, "epilogue": pipe.epilogue,
                "lowering_version": lowering_version()}

    def record(self) -> dict:
        """The Table 1 entry: what this configuration guarantees."""
        if self.scheme == "vllm-v1":
            return self._record_vllm_v1()
        return {
            "family": "Flock",
            "configuration": f"Flock(scheme={self.scheme}, profile={self.profile}, statement={self.statement})",
            "statement": ("full relation: x rows, W columns and outputs committed under frame-v3 (trees a, b over "
                          "blake3-keyed/row/v2 row digests, tree y over word leaves); the verifier sees the chunk "
                          "chaining values of every row and the output words, never the operands"),
            "commitment_scheme": {"id": "frame-v3", "leaf": "blake3-keyed/row/v2 (rows), u16/u32 word leaves (y)",
                                  "in_proof": "every row's chunk compressions (keyed BLAKE3), chained in the circuit",
                                  "native": "BLAKE3 parent nodes and the SHA-256 frame-v3 trees, by the verifier"},
            "hashes": [
                {"hash": "BLAKE3 (keyed)", "role": "statement commitment (row digests)", "assumption": "collision resistance", "class": "standard"},
                {"hash": "SHA-256", "role": "statement commitment (frame-v3 trees)", "assumption": "collision resistance", "class": "standard"},
                {"hash": "BLAKE3 (CPU) / SHA-256 (Flock-CUDA)", "role": "proof-internal commitment (Ligerito Merkle trees)",
                 "assumption": "collision resistance", "class": "standard"},
                {"hash": "SHA-256", "role": "live-coin binding (each round's digest, recorded before its coin)",
                 "assumption": "collision resistance", "class": "standard"},
            ],
            "zk_class": "NON_ZK_PROOF",
            "verifier_model": "interactive: live verifier coins (flock-live R1-R8), designated verifier, not transferable",
            "soundness": {"profile": self.profile, "per_rep": "2^-97.8 (Fast100, red-team-flock, no grinding credit)",
                          "whole_proof": "2^-195.5 (two reps on one root, independent live coins)",
                          "dominant_term": "Ligerito proximity queries per rep, squared over the reps",
                          "accountant": "lanes/flock-128/evidence/accounting.py"},
            "proximity": {"regime": "up to the Johnson bound", "status": "proven (Haböck ePrint 2025/2110; BCHKS25)"},
            "setup": "transparent", "interaction": "interactive: every Flock challenge is a live coin round trip (858 per two-rep session at 256 VUs)",
            "assumptions": ["GF(2^128) and GF(2^256) extension fields of Flock b684b12",
                            "the wiring argument's product GKR over the cell space (Flock b684b12)"],
            "pin": {"lowering": dict(lowering.PINS), "lowering_version": lowering_version(),
                    "flock": "succinctlabs/flock b684b12 + verity/ml/flock/flock-link-b684b12.patch"},
        }


    def _record_vllm_v1(self) -> dict:
        return {
            "family": "Flock",
            "configuration": f"Flock(scheme={self.scheme}, profile={self.profile}, statement={self.statement})",
            "statement": ("full relation: x rows, W columns and outputs committed under vllm-v1 (trees a, b over position "
                          "leaves of the rows, tree y over position leaves of the output words' little-endian bytes; every "
                          "root the step-root binding of a verifier-held StepDomain, PROTOCOL.md section 9); the verifier sees "
                          "each row's leaf digest and the output words, never the operands"),
            "commitment_scheme": {"id": "vllm-v1", "leaf": "pos-leaf: SHA-256(verity/pos-leaf/v0 || u64be(len) || row) (rows and y words)",
                                  "in_proof": "every row's position-leaf SHA-256 compressions (prefix, IV and padding constant), chained in the circuit",
                                  "native": "the vllm-v1 node / lift trees and the step-root bindings, by the verifier; the y leaves",
                                  "domains": "ligero.vllm_tree port domains (the same as B-Ligero's +vllm-v1), domain digests in the statement"},
            "hashes": [
                {"hash": "SHA-256", "role": "statement commitment (vllm-v1 leaves, nodes, lifts, step roots)", "assumption": "collision resistance", "class": "standard"},
                {"hash": "BLAKE3 (CPU) / SHA-256 (Flock-CUDA)", "role": "proof-internal commitment (Ligerito Merkle trees)",
                 "assumption": "collision resistance", "class": "standard"},
                {"hash": "SHA-256", "role": "live-coin binding (each round's digest, recorded before its coin)",
                 "assumption": "collision resistance", "class": "standard"},
            ],
            "zk_class": "NON_ZK_PROOF",
            "verifier_model": "interactive: live verifier coins (flock-live R1-R8), designated verifier, not transferable",
            "soundness": {"profile": self.profile, "per_rep": "2^-97.8 (Fast100, red-team-flock, no grinding credit)",
                          "whole_proof": "2^-195.5 (two reps on one root, independent live coins); red-team review of "
                                         "verity/flock-vllm-block/v1 pending",
                          "dominant_term": "Ligerito proximity queries per rep, squared over the reps",
                          "accountant": "lanes/flock-128/evidence/accounting.py"},
            "proximity": {"regime": "up to the Johnson bound", "status": "proven (Haböck ePrint 2025/2110; BCHKS25)"},
            "setup": "transparent", "interaction": "interactive: every Flock challenge is a live coin round trip",
            "assumptions": ["GF(2^128) and GF(2^256) extension fields of Flock b684b12"],
            "pin": {"lowering": dict(lowering.PINS), "lowering_version": lowering_version(),
                    "flock": "succinctlabs/flock b684b12 + verity/ml/flock/flock-link-b684b12.patch + flock-gpu-link-b684b12.patch"},
        }


FLOCK_FRAME_V3_BLAKE3 = Flock()
FLOCK_BLOCK = Flock(binary="flock-pure-gpu", statement="block")
FLOCK_VLLM_V1 = Flock(scheme="vllm-v1", binary="flock-vllm-v1", statement="block")
BACKENDS = (FLOCK_FRAME_V3_BLAKE3, FLOCK_BLOCK, FLOCK_VLLM_V1)
