"""Sampled proofs: the prover commits to its computation as units and to every unit's inputs and outputs; the verifier
draws units and has each drawn one proved; if all verify, few units are wrong. `verifier` and `prover` (the two
parties), `events` (what they send, and the `IntegrityProfile`), `registration` (what is fixed before the draw), `law`
(the draw and its bound), `construction` (how a unit is proved: `Replay`; `sp1` is heavy, its own distribution)."""
