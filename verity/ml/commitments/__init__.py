"""Application-level commitments: binding Verity values and authenticating their openings.

Two commitment schemes, both wire-pinned by the SP1 kernel: transparent SHA-256 Merkle trees over
verifier-derived position domains, one word per leaf (``merkle``; ``verity.ml.commitments.multiproof`` opens many
leaves of one tree at once), and trees over whole runtime tensor leaves under the ``verity-vllm/leaf/v1`` framing (``leaves``),
the scheme the packed-word obligations of the serving path authenticate against.  ``indexed`` holds the domains,
``identity`` the canonical-manifest digests every identity here is built from, ``limits`` the verifier's limits, and
``layout`` a serving step's Layout JSON (the rows a sequence slot commits), its checks and its digest.

Two named schemes sit behind ``scheme.CommitmentScheme`` and are selectable by backends, each with a spec
and conformance vectors beside it: ``frame_v3`` (word leaves and the ``sha256/row/v1`` / ``blake3-keyed/row/v2`` row
leaves of ``rowleaf``) and ``vllm_v1``, the framing vLLM's serving Commit path emits (id, position and ``fa2h``
leaves, node/lift trees, raw-SHA root bindings).  ``CommitmentScheme.leaf_layout`` is what a backend proves: every
leaf digest is ``H(prefix || value || suffix)`` with H SHA-256 or BLAKE3 (``blake3`` is a plain-Python reference);
verifiers check the trees natively.  Both schemes have an opt-in SHA-512 variant, ``frame-v3-sha512`` and ``vllm-v1-sha512``
(the same framing through SHA-512: a ``hash="sha512"`` domain; each spec's last section).  A ``hash="blake3"`` domain is
``frame-b3``, a tree mode for latency whose leaves and nodes are BLAKE3 keyed by the domain id (``merkle``'s docstring); the
SP1 kernel does not mirror it.  ``hm96`` is the hiding leaf rule over
a SHA-256 or SHA-512 leaf digest (``hm96-sha256/v1``, ``hm96-sha512/v1``: Halevi–Micali, a stored salt per leaf;
statistically hiding given uniform salts, with the pinned key except for a 2^-64 fraction of keys; binding under the hash's
collision resistance); frame-v3 names it in a tree leaf's schema (``hm96-sha512/row/v2``).

These establish that the values supplied to a check are the values previously bound.  They say nothing about the
relation between those values; that is a verification obligation (``verity_sp1.proofs``) a backend establishes.
A proof system's own commitments (a STARK's trace or FRI commitments) are not here.

Promoted from veritor (eb865ce2): ``protocol/merkle.py``, ``protocol/multiproof.py``, ``core/indexed.py``,
``core/identity.py``, ``core/limits.py``, and the ``Commitment`` / ``Opening`` records of ``protocol/messages.py``.
Every hash frame and tag is a wire constant mirrored by the SP1 kernel.
"""

from .identity import Digest, JSONValue, canonical_json_bytes, identity_digest, tagged_sha256, validate_digest
from .indexed import ExplicitIndexedDomain, IndexedDomain, RangeIndexedDomain, iter_domain, iter_members
from .limits import VerificationLimits
from .merkle import Commitment, CommitmentDomain, MerkleTree, Opening, merkle_depth, validate_commitment, verify_opening

__all__ = [
    "Commitment",
    "CommitmentDomain",
    "Digest",
    "ExplicitIndexedDomain",
    "IndexedDomain",
    "JSONValue",
    "MerkleTree",
    "Opening",
    "RangeIndexedDomain",
    "VerificationLimits",
    "canonical_json_bytes",
    "identity_digest",
    "iter_domain",
    "iter_members",
    "merkle_depth",
    "tagged_sha256",
    "validate_commitment",
    "validate_digest",
    "verify_opening",
]
