"""Integrity: whether the streamed executions are what their circuits compute (`verity.primitives.registration`).
Shared by every construction: `sampling` (the draw and its bound) and `profile` (the `IntegrityProfile` each
establishes, which coverage reads). The constructions, each sampling at its own scale and neither importing the other:
`replay` (whole executions, run again by a sealed replayer) and `proofs` (regions, proved by a proof system)."""
