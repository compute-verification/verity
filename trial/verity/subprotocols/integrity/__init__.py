"""Integrity: establishing that a computation is what its circuit computes. Shared by every construction:
`registration` (the prover's commitment to its computation), `sampling` (the draw and its bound) and `profile` (the
`IntegrityProfile` each establishes). The constructions, each sampling at its own scale and neither importing the
other: `replay` (whole executions, run again by a sealed replayer) and `proofs` (regions, proved by a proof system)."""
