"""What authenticating the boundary values against Verity's leaf-Merkle roots costs A-GKR and B-Ligero.

A relation-only proof (``authentication = excluded``, both PROTOCOL.md files) is not a Verity proof: the batch's
``x``, ``W[j,:]`` and ``y`` must be bound to the object's roots (``verity.ml.circuits.targets.target``: "boundary words are
authenticated as packed words against the object's Merkle roots").  This module reads the real K=1536 obligation
(``fixtures/typed-obligation-v0/stage2--qkv-proj-decode-0-k1536.json``) for the leaf structure -- nothing about
leaf sizes, tree depths or hash framing is hard-coded -- and costs four ways of paying for that binding.

What is authenticated per input (read from the fixture by :func:`read_openings`)
====================================================================================

Three whole-tensor leaves under the ``verity-vllm/leaf/v1`` scheme (``verity.ml.commitments.leaves``), opened as packed
leaves (``verity_sp1.proofs.packed``): the checker hashes the *whole* leaf and reads words at statement-fixed
indices; range openings do not exist yet (``leaves.py`` docstring).

* ``x``: the activation row, one leaf of the runtime tree (``ru/<session>``, 368 leaves, depth 9), layout
  ``Array<1536, u16>``: 3072 value bytes; the obligation reads all 1536 words.
* ``W[j,:]``: one weight row, one leaf of the ``weights`` tree (928 325 leaves, depth 20), layout
  ``Array<1, Array<1536, u16>>``: 3072 value bytes; all 1536 words read.
* ``y``: the *output row* leaf (``Array<2048, u16>``, 4096 value bytes) of the runtime tree; the obligation reads
  the single word at index ``j`` (1014).  The whole row is hashed to read one word.

Leaf hash = SHA-256 over ``u32be(len tag) || tag || (u64be(len part) || part)*`` with parts ``(canonical_id,
dtype, shape_text, value)``; node/lift hashes are SHA-256 over the same framing with 32-byte digest parts
(``leaves.H``, ``DIGEST_BYTES = 32``).  All 3073 references of the obligation resolve into these 3 leaves; a batch
of coordinates of one output row shares the ``x`` and ``y`` leaves (the same ``x`` leaf is read by every
coordinate ``j``), so leaf hashing amortises across the batch by the sharing structure, which is a parameter here.

What is private (the reading this module costs first, and the alternative)
==========================================================================

*Reading P (protocol as written).*  The opened values are the prover's private input: ``verity_sp1.proofs.wire``
calls the witness "the guest's private input"; the SP1 guest commits only ``h_O || h_L || verdict`` (65 bytes,
``binding.BOUND_PUBLIC_VALUES_LEN``), so the verifier sees the roots (through ``O``) and the verdict, never ``x``,
``W`` or ``y``.  Both candidate protocols say so outright: ``archive/gkr/PROTOCOL.md`` section 1 marks the per-unit words
``x_u`` **not public**; the ZK requirement is exactly that the transcript reveal nothing about the opened values
beyond what ``O`` states (leaf identities, ranks, layouts, roots are public in ``O``).  The leaf *digests* are
committed material as well: disclosing them to the verifier is a hiding assumption on SHA-256 beyond the collision
resistance the protocols assume (section 7 of both), recorded on the options that do it.

*Reading C (values in the clear).*  ``verity_sp1.proofs.transparent`` and ``packed.check_packed_openings`` are
reference checkers that see the openings in the clear and are declared "not zero-knowledge; it is the reference".
Nothing in ``verity.ml.commitments`` forbids handing the opened leaves to a verifier; a backend that does so is a
``NON_ZK_PROOF`` at best under ``archive/AGENTS.md``'s labels.  Option (b) is costed under this reading and marked
incompatible with reading P.

Options (per input and per batch; see :func:`cost_option`)
============================================================

(a) Merkle openings verified inside the proof: SHA-256 of every distinct leaf (whole tensor row) plus the
    node/lift chain to the root, as circuit constraints.
(b) Prover commits the input vectors in the PCS; the verifier receives the opened leaves and paths in the clear,
    recomputes SHA-256 natively and checks "committed word = public value" (a linear constraint per word).
(c) Recommitment: in-circuit SHA-256 of the *leaf* only, its digest disclosed; the verifier folds the Merkle path
    natively.  Saves the path compressions; discloses leaf digests (hiding assumption).
(d) Protocol change: Verity re-commits at the runtime with Poseidon2 over Goldilocks (t=12, R_F=8, R_P=22,
    ``p3-goldilocks``); leaf and path hashed in circuit.  This changes ``verity.ml.commitments`` -- the README's
    "Commitments" section owns the leaf scheme -- and every recorded root; a protocol decision, not a backend
    optimisation.

Cost model (all constants are cited or derived in :data:`CITATIONS`; every figure marked est. is an estimate)
=============================================================================================================

* SHA-256 compression as an arithmetic circuit: 26 170 nonlinear R1CS constraints per two-to-one compression
  (faulhornlabs/hash-circuits; circomlib ``Sha256(256)`` is 29 380 incl. padding).  A multiplication gate of a
  layered circuit per nonlinear constraint (A-GKR's GKR counts gates; B-Ligero's Ligero commits 3 aligned
  slots per product, ``census.columns_committed_b``'s rule).
* A-GKR commits inputs and hints only: for SHA-256 the hints are the bit decompositions the circuit needs --
  every 32-bit addition output re-decomposed (64 rounds x 4 additions + 8 final + 48 schedule = 312 additions,
  32 bits each) plus the 512 message bits of the block: 10 496 committed elements per compression (derivation,
  est.; the booleanity products are inside the 26 170).
* AIR reference points: SP1's ``ShaCompressChip`` spends 80 rows per compression; RISC Zero's accelerator 68
  cycles per 64-byte block; SP1's *software* SHA-256 in the measured guest: 138 164 cycles for the fixture's
  246 compressions (``merkle.auth``, run ``r20260921-224405-22dc``) = 562 cycles per compression.
* Poseidon2-Goldilocks t=12: 8 full rounds x 12 S-boxes + 22 partial rounds = 118 S-boxes ``x^7`` at 4
  multiplications each = 472 multiplications per permutation (derivation from the round counts); rate 8 elements;
  3 BF16 words packed per element (48 bits < 64: no wrap) = 24 words per permutation; a 2-to-1 node is one
  permutation (two 4-element digests fit the rate).  A's committed extra: none (packing is linear, S-boxes are
  gates); B's: 3 slots per multiplication.
* Depth (A-GKR's latency axis): Merkle-Damgard chains the compressions of one leaf; a layered GKR either
  runs the chain layer by layer (est. 3 layers per SHA-256 round, 192 per compression) or commits every chaining
  state as hints (256 bits per block) and hashes blocks in parallel.  Both are reported.  B-Ligero's depth is
  unchanged (local constraints).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from verity.ml.commitments.leaves import (
    DIGEST_BYTES,
    LEAF_TAG,
    LIFT_TAG,
    NODE_TAG,
    path_shape,
)
from verity_sp1.proofs.packed import decode_packed_witness

from verity_numerical.security import accounting
from verity_numerical.security.census import census_vu

__all__ = [
    "CITATIONS",
    "FIXTURE",
    "OPTIONS",
    "POSEIDON2_GOLDILOCKS",
    "SHA256",
    "SHARING",
    "HashCircuit",
    "LeafOpening",
    "Openings",
    "OptionCost",
    "Sharing",
    "batch_table",
    "build_report",
    "cost_option",
    "instance_compressions",
    "main",
    "read_openings",
    "relation_side",
    "sha256_compressions",
]

REPO = Path(__file__).resolve().parents[5]
FIXTURE = REPO / "fixtures" / "typed-obligation-v0" / "stage2--qkv-proj-decode-0-k1536.json"

SHA256_BLOCK = 64
SHA256_PAD = 9                      # 0x80 + u64 length
FRAMING_TAG_PREFIX = 4              # u32be(len(tag))
FRAMING_PART_PREFIX = 8             # u64be(len(part))

CITATIONS: dict[str, str] = {
    "sha256_r1cs": "faulhornlabs/hash-circuits README (read 2026-09-21): SHA-256 two-to-one compression 26 170 nonlinear "
                   "R1CS constraints; circomlib Sha256(256) 29 380 (RareSkills circom tutorial). https://github.com/faulhornlabs/hash-circuits",
    "sp1_sha_air": "sp1-core ShaCompressChip docs: 'each SHA compress syscall takes up 80 rows' (8 init, 64 compression, 8 finalize). "
                   "https://docs.rs/sp1-core/latest/sp1_core/syscall/precompiles/sha256/struct.ShaCompressChip.html",
    "risc0_sha": "RISC Zero Guest Optimization Guide: SHA-256 accelerator '68 cycles per 64-byte block and 6 cycles to initialize'; pure Rust ~2 000. "
                 "https://dev.risczero.com/api/zkvm/optimization",
    "sp1_measured": "archive/sp1/README.md, run r20260921-224405-22dc: merkle.auth (leaf hashing + path folding) 138 164 cycles of 10 128 736 (1.3%); "
                    "parse.statement 3.71M + merkle.table 4.56M (82%); gates 0.43M (4.2%).",
    "poseidon2_goldilocks": "p3-goldilocks poseidon2.rs: Goldilocks Poseidon2 alpha=7, R_F=8, R_P=22 (widths 8, 12, 16); Plonky3 Poseidon2Air Goldilocks "
                            "WIDTH=12, SBOX_DEGREE=7, HALF_FULL_ROUNDS=4, PARTIAL_ROUNDS=22. https://docs.rs/p3-goldilocks/latest/src/p3_goldilocks/poseidon2.rs.html",
    "poseidon2_r1cs": "faulhornlabs/hash-circuits: Poseidon2 (t=3, BN254) two-to-one compression ~240 nonlinear R1CS constraints (order-of-magnitude cross-check).",
    "leaf_scheme": "verity/ml/commitments/leaves.py: H(tag,*parts)=sha256(u32be(len tag)||tag||(u64be(len part)||part)*); "
                   "leaf=(id,dtype,shape,value); node/lift; DIGEST_BYTES=32; path_shape/tree_depth.",
    "packed": "archive/sp1/python/verity_sp1/proofs/packed.py: 'the guest authenticates the whole packed leaf against the committed root, then reads the "
              "word at the statement-fixed index'; 'a leaf is hashed once however many positions read words of it'; canonical_id/dtype/shape are the prover's to supply.",
    "privacy_witness": "archive/sp1/python/verity_sp1/proofs/wire.py:465 'The canonical bytes of a witness (the guest's private input)'; "
                       "binding.py: public values = h_O || h_L || verdict (65 bytes), nothing else.",
    "privacy_protocol": "archive/gkr/PROTOCOL.md section 1: x_u (per-unit words, hints, row columns) public? **no**; section 7: only collision resistance "
                        "of H is assumed for soundness, hiding beyond CRH is an extra assumption; archive/direct/PROTOCOL.md sections 1, 7, 8 likewise.",
    "privacy_reference": "archive/sp1/python/verity_sp1/proofs/transparent.py:10: the reference checker is 'not zero-knowledge; it is the reference every other "
                         "backend must agree with' -- it sees openings in the clear.",
    "readme_commitments": "README.md 'Commitments': verity.ml.commitments owns application-level layouts, roots, openings; proof-system-internal commitments "
                          "are the backend's.  Changing the leaf hash is a verity.ml.commitments (protocol) change.",
    "relation_census": "archive/gkr/PROTOCOL.md 9.4 / archive/direct/PROTOCOL.md 9.4: committed elements at K=1536, B=4096: A 181.4M, B 779.6M "
                       "(accounting.census_estimate_a/b); per-unit census security.census.census_vu(1536).",
}


# ---------------------------------------------------------------------------------------------------------------------
# The fixture: what is opened
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LeafOpening:
    """One opened leaf of the K=1536 obligation, as the witness carries it."""

    role: str                   # x | w | y
    tree: str                   # root id in the object
    tree_leaf_count: int
    rank: int
    depth: int                  # path steps to the root
    lifts: int                  # of which lifted (odd trailing) steps
    canonical_id: str
    dtype: str
    shape_text: str
    value_bytes: int
    words_read: int             # references of the obligation resolving into this leaf
    word_bits: int

    @property
    def preimage_bytes(self) -> int:
        """Bytes SHA-256 absorbs for the leaf hash under ``leaves.H``'s framing."""
        parts = (self.canonical_id.encode(), self.dtype.encode(), self.shape_text.encode())
        return (FRAMING_TAG_PREFIX + len(LEAF_TAG.encode())
                + sum(FRAMING_PART_PREFIX + len(p) for p in parts)
                + FRAMING_PART_PREFIX + self.value_bytes)

    @property
    def public_prefix_bytes(self) -> int:
        """The framing before the value: tag, id, dtype, shape, value length -- all fixed by the object ``O``."""
        return self.preimage_bytes - self.value_bytes

    @property
    def leaf_compressions(self) -> int:
        return sha256_compressions(self.preimage_bytes)

    @property
    def public_prefix_compressions(self) -> int:
        """Whole blocks of the preimage that hold only public framing: their chaining state is a public constant."""
        return self.public_prefix_bytes // SHA256_BLOCK

    @property
    def path_compressions(self) -> int:
        return (self.depth - self.lifts) * NODE_COMPRESSIONS + self.lifts * LIFT_COMPRESSIONS


@dataclass(frozen=True)
class Openings:
    """The three leaves of one instance plus the batch-level facts read from the object."""

    name: str
    K: int
    leaves: tuple[LeafOpening, ...]
    references: int
    roots: dict[str, int]       # root id -> leaf count

    def leaf(self, role: str) -> LeafOpening:
        return next(leaf for leaf in self.leaves if leaf.role == role)

    def summary(self) -> dict[str, Any]:
        return {
            "fixture": self.name, "K": self.K, "references": self.references, "roots": self.roots,
            "digest_bytes": DIGEST_BYTES, "hash": "SHA-256 (verity-vllm/leaf/v1 framing)",
            "node_compressions": NODE_COMPRESSIONS, "lift_compressions": LIFT_COMPRESSIONS,
            "leaves": [{**asdict(leaf), "preimage_bytes": leaf.preimage_bytes, "leaf_compressions": leaf.leaf_compressions,
                        "public_prefix_compressions": leaf.public_prefix_compressions,
                        "path_compressions": leaf.path_compressions} for leaf in self.leaves],
            "instance_compressions_no_sharing": instance_compressions(self, Sharing("none", 1, 1, 1)),
        }


def sha256_compressions(message_bytes: int) -> int:
    """Merkle-Damgard block count: the message, 0x80 and the 8-byte length, padded to 64-byte blocks."""
    return -(-(message_bytes + SHA256_PAD) // SHA256_BLOCK)


def _framed_len(tag: str, *part_lens: int) -> int:
    return FRAMING_TAG_PREFIX + len(tag.encode()) + sum(FRAMING_PART_PREFIX + n for n in part_lens)


NODE_COMPRESSIONS = sha256_compressions(_framed_len(NODE_TAG, DIGEST_BYTES, DIGEST_BYTES))
LIFT_COMPRESSIONS = sha256_compressions(_framed_len(LIFT_TAG, DIGEST_BYTES))


def read_openings(path: Path = FIXTURE) -> Openings:
    """The leaf/opening structure of the fixture: object tables for identity, the recorded witness for the bytes."""
    doc = json.loads(Path(path).read_text())
    obj = doc["object"]
    roots = {r["id"]: int(r["count"]) for r in obj["roots"]}
    leaves_by_id = {v["id"]: v for v in obj["leaves"]}
    reads: dict[str, int] = {}
    for ref in obj["references"]:
        reads[ref["leaf"]] = reads.get(ref["leaf"], 0) + 1
    witness = decode_packed_witness(bytes.fromhex(doc["witness"]))
    by_id = {leaf.canonical_id: leaf for tree in witness.trees for leaf in tree}
    subcircuit = obj["subcircuits"][0]
    K = int(subcircuit["definition"]["static"]["K"])
    obligation = obj["obligations"][0]
    out_ref = obligation["boundary"]["out"][0]
    out_leaf_id = next(ref["leaf"] for ref in obj["references"] if ref["ref"] == out_ref)
    openings = []
    for leaf_id, entry in leaves_by_id.items():
        packed = by_id[leaf_id]
        count = roots[entry["root"]]
        shape = path_shape(count, int(entry["leaf"]))
        layout = entry["layout"]
        while "array" in layout and "array" in layout["array"][1]:
            layout = layout["array"][1]
        word_bits = int(layout["array"][1]["value"])
        if leaf_id == out_leaf_id:
            role = "y"
        elif entry["root"] == "weights" or leaf_id.startswith("w="):
            role = "w"
        else:
            role = "x"
        openings.append(LeafOpening(
            role=role, tree=entry["root"], tree_leaf_count=count, rank=int(entry["leaf"]), depth=len(shape),
            lifts=sum(1 for has_sibling in shape if not has_sibling), canonical_id=packed.canonical_id,
            dtype=packed.dtype, shape_text=packed.shape_text, value_bytes=len(packed.value),
            words_read=reads[leaf_id], word_bits=word_bits,
        ))
    order = {"x": 0, "w": 1, "y": 2}
    openings.sort(key=lambda leaf: order[leaf.role])
    return Openings(name=doc["name"], K=K, leaves=tuple(openings), references=len(obj["references"]), roots=roots)


# ---------------------------------------------------------------------------------------------------------------------
# Sharing and hash circuits
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sharing:
    """How many instances of the batch read one physical leaf, per role.  Leaf hashing is paid once per distinct leaf
    (``packed.py``: 'a leaf is hashed once however many positions read words of it')."""

    name: str
    x: int
    w: int
    y: int

    def share(self, role: str) -> int:
        return getattr(self, role)


def _sharing_for(B: int) -> dict[str, Sharing]:
    """Three regimes.  ``none``: every instance its own leaves (worst case, e.g. one coordinate per token).
    ``bench24``: the bench dataset's structure -- 24 (x, output-row) pairs, the batch's coordinates spread over them,
    every W row distinct.  ``one_row``: all B coordinates of one output row of one token (B <= N = 2048)."""
    per_seed = -(-B // 24)
    return {
        "none": Sharing("none", 1, 1, 1),
        "bench24": Sharing("bench24", per_seed, 1, per_seed),
        "one_row": Sharing("one_row", B, 1, B),
    }


SHARING = _sharing_for(4096)


def instance_compressions(openings: Openings, sharing: Sharing, include_path: bool = True,
                          skip_public_prefix: bool = False) -> float:
    """SHA-256 compressions charged to one instance after amortising shared leaves."""
    total = 0.0
    for leaf in openings.leaves:
        comps = leaf.leaf_compressions - (leaf.public_prefix_compressions if skip_public_prefix else 0)
        if include_path:
            comps += leaf.path_compressions
        total += comps / sharing.share(leaf.role)
    return total


@dataclass(frozen=True)
class HashCircuit:
    """A hash primitive as circuit cost per invocation (one compression or one permutation)."""

    name: str
    mul_gates: int                  # multiplication gates (A: GKR gates; B: products)
    committed_a: int                # extra committed elements for A-GKR per invocation (hints)
    layers: int                     # multiplicative depth per invocation (est.)
    bytes_per_invocation: int       # payload absorbed per invocation
    node_invocations: int           # invocations per Merkle node
    citation: str

    @property
    def committed_b(self) -> int:
        """B-Ligero commits 3 aligned slots per product (``census.columns_committed_b``'s rule)."""
        return 3 * self.mul_gates


SHA256_ADDITIONS_PER_COMPRESSION = 64 * 4 + 8 + 48
SHA256 = HashCircuit(
    name="SHA-256 (protocol hash, verity-vllm/leaf/v1)",
    mul_gates=26_170,
    committed_a=SHA256_ADDITIONS_PER_COMPRESSION * 32 + 512,
    layers=3 * 64,
    bytes_per_invocation=SHA256_BLOCK,
    node_invocations=NODE_COMPRESSIONS,
    citation="sha256_r1cs",
)

POSEIDON2_WIDTH, POSEIDON2_RATE, POSEIDON2_RF, POSEIDON2_RP = 12, 8, 8, 22
POSEIDON2_SBOXES = POSEIDON2_RF * POSEIDON2_WIDTH + POSEIDON2_RP
POSEIDON2_WORDS_PER_ELEMENT = 3     # 3 x 16-bit words = 48 bits, no wrap in Goldilocks
POSEIDON2_GOLDILOCKS = HashCircuit(
    name="Poseidon2-Goldilocks t=12 (protocol change, option d)",
    mul_gates=POSEIDON2_SBOXES * 4,
    committed_a=0,
    layers=(POSEIDON2_RF + POSEIDON2_RP) * 3,
    bytes_per_invocation=POSEIDON2_RATE * POSEIDON2_WORDS_PER_ELEMENT * 2,
    node_invocations=1,
    citation="poseidon2_goldilocks",
)


# ---------------------------------------------------------------------------------------------------------------------
# The options
# ---------------------------------------------------------------------------------------------------------------------

OPTIONS = ("a", "b", "c", "d")


@dataclass(frozen=True)
class OptionCost:
    option: str
    title: str
    compatible_with_private_values: bool
    hiding_assumption: str
    sharing: str
    B: int
    # per input (amortised)
    hash_invocations_per_instance: float
    hash_invocations_by_type: dict[str, float]
    mul_gates_per_instance: float
    committed_a_per_instance: float
    committed_b_per_instance: float
    verifier_hash_invocations_per_instance: float
    communication_bytes_per_instance: float
    depth_layers_chained: int           # A: hash circuit layers on the critical path if chains are not hinted
    depth_layers_hinted: int            # A: with chaining states committed as hints (parallel blocks)
    committed_a_hint_chaining_per_instance: float
    # batch totals
    mul_gates_batch: float
    committed_a_batch: float
    committed_b_batch: float
    communication_bytes_batch: float
    # against the relation
    relation_mul_gates_batch: float
    relation_committed_a_batch: float
    relation_committed_b_batch: float
    fraction_mul_gates_a: float
    fraction_committed_a: float
    fraction_committed_b: float
    field_ops_a_est: float
    field_ops_b_est: float
    fraction_field_ops_a: float
    fraction_field_ops_b: float
    notes: tuple[str, ...]

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationSide:
    """The relation-only census the options are compared with (``census_vu`` and the accountants' estimates)."""

    B: int
    K: int
    mul_gates_batch: int
    committed_a_batch: int
    committed_b_batch: int
    field_ops_a: float
    field_ops_b: float

    @property
    def field_ops_per_committed_b(self) -> float:
        return self.field_ops_b / self.committed_b_batch

    @property
    def field_ops_per_mul_a(self) -> float:
        return self.field_ops_a / self.mul_gates_batch


def relation_side(B: int = 4096, K: int = 1536) -> RelationSide:
    """Relation-only totals at the primary operating point (F_p^3, t=192), from the accountants."""
    GL = accounting.Field.prime("Goldilocks", accounting.GOLDILOCKS)
    Fc = GL.extension(3)
    lig = accounting.LigeroParams(n=2**14, k=2**12, l=3840, t=192)
    a = accounting.census_estimate_a(accounting.CandidateA(K=K, B=B, trace_field=GL, challenge_field=Fc, ligero=lig))
    b = accounting.census_estimate_b(accounting.CandidateB(K=K, B=B, trace_field=GL, challenge_field=Fc, ligero=lig))
    vu = census_vu(K)
    return RelationSide(
        B=B, K=K, mul_gates_batch=vu.mul * B,
        committed_a_batch=int(a["census.committed_elements"]), committed_b_batch=int(b["census.committed_elements"]),
        field_ops_a=a["census.field_ops"], field_ops_b=b["census.field_ops"],
    )


def _leaf_invocations(leaf: LeafOpening, h: HashCircuit, include_leaf: bool, include_path: bool,
                      skip_public_prefix: bool) -> tuple[float, float]:
    """(leaf-hash invocations, path invocations) of one leaf under hash ``h``."""
    if h is SHA256:
        leaf_inv = leaf.leaf_compressions - (leaf.public_prefix_compressions if skip_public_prefix else 0)
        path_inv = leaf.path_compressions
    else:
        words = leaf.value_bytes // (leaf.word_bits // 8)
        leaf_inv = -(-words // (POSEIDON2_RATE * POSEIDON2_WORDS_PER_ELEMENT))
        path_inv = leaf.depth * h.node_invocations
    return (leaf_inv if include_leaf else 0.0), (path_inv if include_path else 0.0)


def cost_option(option: str, openings: Openings, sharing: Sharing, B: int = 4096,
                rel: RelationSide | None = None) -> OptionCost:
    """Cost one option for one sharing regime at batch size ``B``."""
    if option not in OPTIONS:
        raise ValueError(f"unknown option {option!r}")
    rel = rel or relation_side(B, openings.K)
    h = POSEIDON2_GOLDILOCKS if option == "d" else SHA256
    in_circuit_leaf = option in ("a", "c", "d")
    in_circuit_path = option in ("a", "d")
    by_type: dict[str, float] = {}
    verifier_inv = 0.0
    comm = 0.0
    chained_layers = 0
    hinted_chain_elements = 0.0
    for leaf in openings.leaves:
        share = sharing.share(leaf.role)
        leaf_inv, path_inv = _leaf_invocations(leaf, h, in_circuit_leaf, in_circuit_path, skip_public_prefix=False)
        by_type[f"{leaf.role}:leaf"] = by_type.get(f"{leaf.role}:leaf", 0.0) + leaf_inv / share
        by_type[f"{leaf.role}:path"] = by_type.get(f"{leaf.role}:path", 0.0) + path_inv / share
        full_leaf, full_path = _leaf_invocations(leaf, h, True, True, False)
        if option == "b":
            verifier_inv += (full_leaf + full_path) / share
            comm += (leaf.value_bytes + leaf.depth * DIGEST_BYTES) / share
        elif option == "c":
            verifier_inv += full_path / share
            comm += (DIGEST_BYTES + leaf.depth * DIGEST_BYTES) / share
        if in_circuit_leaf:
            chained_layers = max(chained_layers, int(full_leaf) * h.layers + (int(full_path) * h.layers if in_circuit_path else 0))
            hinted_chain_elements += (full_leaf * (DIGEST_BYTES * 8 if h is SHA256 else 4)) / share
    inv = sum(by_type.values())
    mul = inv * h.mul_gates
    ca = inv * h.committed_a
    cb = inv * h.committed_b
    if option == "b":
        # every committed word is pinned to a public value: a linear constraint each, no extra committed element
        mul, ca, cb = 0.0, 0.0, 0.0
    hinted_layers = (h.layers * (1 + (1 if in_circuit_path else 0))) if in_circuit_leaf else 0
    notes: list[str] = []
    if option == "a":
        notes.append("whole leaf + node/lift chain in circuit; the first public-framing blocks of each leaf "
                     f"({sum(leaf.public_prefix_compressions for leaf in openings.leaves)} compressions per instance before sharing) "
                     "have a public chaining state and could be skipped (not credited here)")
    if option == "b":
        notes.append("verifier sees x, W[j,:] and the whole output row in the clear: incompatible with reading P; "
                     "NON_ZK_PROOF at best; the proof adds one linear 'committed word = public word' constraint per word "
                     f"({openings.references} per instance)")
    if option == "c":
        notes.append("leaf digests disclosed to the verifier (hiding of the row under SHA-256 preimage resistance: an assumption "
                     "beyond CRH, PROTOCOL.md section 7 of both candidates); the verifier folds the Merkle paths natively")
    if option == "d":
        notes.append("changes verity.ml.commitments (verity-vllm/leaf/v1 -> a Poseidon2 leaf scheme) and every recorded root, conformance "
                     "vector and the SP1 guest: a protocol decision (README 'Commitments'), not a backend optimisation")
    frac_mul = mul * B / rel.mul_gates_batch
    frac_ca = ca * B / rel.committed_a_batch
    frac_cb = cb * B / rel.committed_b_batch
    fo_a = mul * B * rel.field_ops_per_mul_a
    fo_b = cb * B * rel.field_ops_per_committed_b
    return OptionCost(
        option=option, title=_TITLES[option], compatible_with_private_values=option != "b",
        hiding_assumption=_HIDING[option], sharing=sharing.name, B=B,
        hash_invocations_per_instance=inv, hash_invocations_by_type=by_type,
        mul_gates_per_instance=mul, committed_a_per_instance=ca, committed_b_per_instance=cb,
        verifier_hash_invocations_per_instance=verifier_inv, communication_bytes_per_instance=comm,
        depth_layers_chained=chained_layers, depth_layers_hinted=hinted_layers,
        committed_a_hint_chaining_per_instance=hinted_chain_elements,
        mul_gates_batch=mul * B, committed_a_batch=ca * B, committed_b_batch=cb * B, communication_bytes_batch=comm * B,
        relation_mul_gates_batch=rel.mul_gates_batch, relation_committed_a_batch=rel.committed_a_batch,
        relation_committed_b_batch=rel.committed_b_batch,
        fraction_mul_gates_a=frac_mul, fraction_committed_a=frac_ca, fraction_committed_b=frac_cb,
        field_ops_a_est=fo_a, field_ops_b_est=fo_b,
        fraction_field_ops_a=fo_a / rel.field_ops_a, fraction_field_ops_b=fo_b / rel.field_ops_b,
        notes=tuple(notes),
    )


_TITLES = {
    "a": "(a) SHA-256 Merkle openings inside the proof",
    "b": "(b) openings checked by the verifier outside the proof (values public)",
    "c": "(c) recommitment: in-circuit SHA-256 of the leaf only, path outside",
    "d": "(d) protocol change: Poseidon2-Goldilocks leaves, all inside",
}
_HIDING = {
    "a": "none beyond the protocols' CRH: only roots (in O) and the verdict leave the proof",
    "b": "n/a: x, W, y are public to the verifier",
    "c": "leaf digests disclosed: hiding of a 3072/4096-byte row under SHA-256 preimage resistance (beyond CRH)",
    "d": "none beyond CRH of Poseidon2 (a new assumption set: Poseidon2 replaces SHA-256 as the protocol hash)",
}


def batch_table(openings: Openings | None = None, B: int = 4096, sharings: Sequence[Sharing] | None = None) -> list[OptionCost]:
    openings = openings or read_openings()
    rel = relation_side(B, openings.K)
    sharings = sharings or list(_sharing_for(B).values())
    return [cost_option(o, openings, s, B, rel) for s in sharings for o in OPTIONS]


def build_report(B: int = 4096) -> dict[str, Any]:
    openings = read_openings()
    rel = relation_side(B, openings.K)
    rows = batch_table(openings, B)
    sp1_cycles_per_compression = 138_164 / instance_compressions(openings, Sharing("none", 1, 1, 1))
    verdict = _killing_question(rows)
    return {
        "lane": "explore-auth (wave 1)", "date": "2026-09-22", "fixture": str(FIXTURE.relative_to(REPO)),
        "what_is_authenticated": openings.summary(),
        "what_is_private": {
            "reading_P_private_values": {
                "statement": "x, W[j,:], y are the prover's private input; the verifier learns the roots (through O), h_O, h_L and the verdict. "
                             "Leaf identities, ranks, layouts and counts are public in O.",
                "code": [CITATIONS["privacy_witness"], CITATIONS["privacy_protocol"], CITATIONS["packed"]],
                "consequence": "the ZK requirement is over the opened values (and the leaf digests, which are committed material); "
                               "option (b) is excluded; (c) adds a hiding assumption on SHA-256 for disclosed leaf digests.",
            },
            "reading_C_values_in_clear": {
                "statement": "the reference checkers (transparent.py, packed.check_packed_openings) see openings in the clear and say they are not ZK; "
                             "nothing in verity.ml.commitments forbids a verifier that sees values, but such a backend is NON_ZK_PROOF at best.",
                "code": [CITATIONS["privacy_reference"]],
            },
            "costed_first": "reading P",
        },
        "hash_circuits": {
            "sha256": {**asdict(SHA256), "committed_b": SHA256.committed_b, "citation": CITATIONS[SHA256.citation],
                       "air_reference": [CITATIONS["sp1_sha_air"], CITATIONS["risc0_sha"]],
                       "sp1_software_measured_cycles_per_compression": sp1_cycles_per_compression,
                       "committed_a_derivation": f"{SHA256_ADDITIONS_PER_COMPRESSION} additions x 32 bits + 512 message bits (est.)"},
            "poseidon2_goldilocks": {**asdict(POSEIDON2_GOLDILOCKS), "committed_b": POSEIDON2_GOLDILOCKS.committed_b,
                                     "citation": CITATIONS["poseidon2_goldilocks"], "cross_check": CITATIONS["poseidon2_r1cs"],
                                     "sboxes": POSEIDON2_SBOXES, "words_per_permutation": POSEIDON2_RATE * POSEIDON2_WORDS_PER_ELEMENT,
                                     "mul_gates_derivation": "118 S-boxes x^7 at 4 multiplications (x^2, x^4, x^3, x^7) = 472 (derivation)"},
        },
        "relation_side": asdict(rel),
        "sharing_regimes": {s.name: asdict(s) for s in _sharing_for(B).values()},
        "options": [r.row() for r in rows],
        "killing_question": verdict,
        "citations": CITATIONS,
        "assumptions": [
            "a nonlinear R1CS constraint of the cited SHA-256 circuit is one multiplication gate of a layered circuit (A) and one Ligero product, 3 slots (B)",
            "Candidate A's committed elements for SHA-256 are the bit decompositions of every 32-bit addition output plus the message bits (10 496 per compression, est.)",
            "the bench dataset's 24 seeds are 24 (x, output-row) pairs with all W rows distinct ('bench24'); 'none' and 'one_row' bracket it",
            "field-operation fractions scale the accountants' relation-side rates (F_t mult-eq. per mul gate for A, per committed element for B); estimates",
            "Poseidon2 packs 3 BF16 words per Goldilocks element; a 2-to-1 node is one permutation; no range openings (leaves.py: not carried yet)",
        ],
    }


def _killing_question(rows: Sequence[OptionCost]) -> dict[str, Any]:
    by = {(r.option, r.sharing): r for r in rows}
    compatible = {k: r for k, r in by.items() if r.compatible_with_private_values}
    dominates = {f"{o}/{s}": {"A_committed_x": round(r.fraction_committed_a, 2), "A_mul_x": round(r.fraction_mul_gates_a, 2),
                              "B_committed_x": round(r.fraction_committed_b, 2)}
                 for (o, s), r in compatible.items()}
    return {
        "question": "does authentication dominate again for either candidate under the options compatible with the privacy model?",
        "answer": "yes for every SHA-256 option compatible with reading P ((a) and (c)): at B=4096 authentication is a multiple of the whole "
                  "relation-only census for both candidates in every sharing regime; only the protocol change (d) brings it to the same order as "
                  "the relation (A: gates ~ 1x, no extra committed elements but ~5 700 chained layers; B: committed +~60%).",
        "why_sp1_differed": "SP1 pays ~562 guest cycles per SHA-256 compression natively and ~10 M cycles for statement machinery + a relation "
                            "that a zkVM executes as ordinary instructions; a purpose-built circuit makes the relation ~1000x cheaper while "
                            "SHA-256 stays at ~26k gates per compression, so the ratio inverts.",
        "ratios_vs_relation": dominates,
        "overnight_sweep_should_assume": "relation-only numbers stay labelled authentication=excluded; add a separate authentication line costed as "
                                         "(c) with SHA-256 (the compatible baseline; W-row leaf dominates, 50 compressions per distinct row) and (d) "
                                         "as the protocol proposal; (b) only for NON_ZK diagnostics. Neither (a) nor (c) is affordable at B=4096 "
                                         "without weight-row leaf amortisation across proofs.",
    }


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f}G"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.1f}"


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    sizes = [int(a) for a in args if a.isdigit()]
    B = sizes[0] if sizes else 4096
    openings = read_openings()
    print(f"fixture {openings.name}: K={openings.K}, {openings.references} references into {len(openings.leaves)} leaves")
    for leaf in openings.leaves:
        print(f"  {leaf.role}: {leaf.value_bytes} B value, {leaf.preimage_bytes} B preimage -> {leaf.leaf_compressions} compressions; "
              f"tree {leaf.tree} ({leaf.tree_leaf_count} leaves) depth {leaf.depth} ({leaf.lifts} lifts) -> {leaf.path_compressions} compressions; "
              f"{leaf.words_read} words read")
    rel = relation_side(B, openings.K)
    print(f"relation-only at B={B}: mul {_fmt(rel.mul_gates_batch)}, committed A {_fmt(rel.committed_a_batch)}, B {_fmt(rel.committed_b_batch)}")
    head = f"{'option':8} {'sharing':8} {'hash/inst':>10} {'mul/inst':>10} {'A comm/inst':>12} {'B comm/inst':>12} " \
           f"{'A comm x':>9} {'A mul x':>8} {'B comm x':>9} {'comm B/inst':>12} {'priv':>5}"
    print(head)
    for r in batch_table(openings, B):
        print(f"{r.option:8} {r.sharing:8} {_fmt(r.hash_invocations_per_instance):>10} {_fmt(r.mul_gates_per_instance):>10} "
              f"{_fmt(r.committed_a_per_instance):>12} {_fmt(r.committed_b_per_instance):>12} "
              f"{r.fraction_committed_a:>9.2f} {r.fraction_mul_gates_a:>8.2f} {r.fraction_committed_b:>9.2f} "
              f"{_fmt(r.communication_bytes_per_instance):>12} {'yes' if r.compatible_with_private_values else 'NO':>5}")
    if "--write" in args:
        out = REPO / "backends" / "numerical" / "reports" / "authentication.json"
        out.write_text(json.dumps(build_report(B), indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":       # pragma: no cover
    main()
