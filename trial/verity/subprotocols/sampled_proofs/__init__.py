"""Sampled proofs: the prover commits to its computation as units and to every unit's inputs and outputs; the verifier
draws units and has each drawn one proved; if all verify, few units are wrong. `registration` (what is fixed before the
draw), `law` (the draw and its bound), `audit` (the audit and the `IntegrityProfile` it establishes), `replay`, and
`sp1` (heavy, its own distribution). A construction supplies only `verify_unit`."""
