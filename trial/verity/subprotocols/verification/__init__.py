"""Verification: establishing that a computation is what its circuit computes. `units` (what a unit is, how its values
are committed, and `ProofSystem`, the interface of whatever proves one), `sampling` (the protocol: the prover registers
its computation as units, the verifier draws some and has each drawn one proved), `replay` (a proof system that opens
the unit and recomputes it) and `proofs` (proof systems proper; `sp1` is heavy, its own distribution)."""
