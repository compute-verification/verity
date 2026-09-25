"""Application-level commitments: binding Verity values and authenticating their openings.

Two commitment schemes, both wire-pinned by the SP1 kernel: transparent SHA-256 Merkle trees over
verifier-derived position domains, one word per leaf (``merkle``, with ``multiproof`` opening many leaves of one
tree at once), and trees over whole runtime tensor leaves under the ``verity-vllm/leaf/v1`` framing (``leaves``),
the scheme the packed-word obligations of the serving path authenticate against.  ``indexed`` holds the domains,
``identity`` the canonical-manifest digests every identity here is built from, ``limits`` the verifier's limits.

Two named schemes sit behind ``scheme.CommitmentScheme`` and are selectable by configured backends: frame-v3
(``merkle.FrameV3``) and ``vllm_production``, the framing vLLM's production Commit path emits (id, position and
``fa2h`` leaves, node/lift trees, raw-SHA root bindings), with its spec and conformance vectors beside it.

These establish that the values supplied to a check are the values previously bound.  They say nothing about the
relation between those values; that is a verification obligation (``verity.verification``) a backend establishes.
A proof system's own commitments (a STARK's trace or FRI commitments) are not here.

Promoted from veritor (eb865ce2): ``protocol/merkle.py``, ``protocol/multiproof.py``, ``core/indexed.py``,
``core/identity.py``, ``core/limits.py``, and the ``Commitment`` / ``Opening`` records of ``protocol/messages.py``.
Every hash frame and tag is a wire constant mirrored by the SP1 kernel.
"""

from .identity import Digest, JSONValue, canonical_json_bytes, identity_digest, tagged_sha256, validate_digest
from .indexed import ExplicitIndexedDomain, IndexedDomain, RangeIndexedDomain, iter_domain, iter_members
from .limits import VerificationLimits
from .merkle import Commitment, CommitmentDomain, MerkleTree, Opening, merkle_depth, validate_commitment, verify_opening
from .multiproof import MultiproofCost, multiproof, multiproof_cost, multiproof_from_paths, verify_multiproof

__all__ = [
    "Commitment",
    "CommitmentDomain",
    "Digest",
    "ExplicitIndexedDomain",
    "IndexedDomain",
    "JSONValue",
    "MerkleTree",
    "MultiproofCost",
    "Opening",
    "RangeIndexedDomain",
    "VerificationLimits",
    "canonical_json_bytes",
    "identity_digest",
    "iter_domain",
    "iter_members",
    "merkle_depth",
    "multiproof_cost",
    "multiproof_from_paths",
    "multiproof",
    "tagged_sha256",
    "validate_commitment",
    "validate_digest",
    "verify_multiproof",
    "verify_opening",
]
