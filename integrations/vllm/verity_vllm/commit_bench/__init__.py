"""CPERF 2026-09-17 prototypes for the commitment-cost study (``docs/vllm-poc/commitment-cost-study-2026-09-17.md``).

Nothing here is on the production path.  :mod:`fasttree` re-implements the frozen hash framing of
:mod:`verity_vllm.hashing` / :mod:`verity_vllm.commit` with packed (pre-resolved) identities, a pluggable hash
backend (SHA-256, BLAKE3, no-op) and zero-copy leaf inputs, and asserts root equality with the production
``MerkleTree`` wherever the semantics are unchanged.  :mod:`bench` is the measurement runner (waterfall, ablations,
leaf-granularity sweep, backends, process-parallel trees); :mod:`prov` records provenance.
"""
