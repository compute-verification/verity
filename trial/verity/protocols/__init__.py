"""The protocols, flat. Each establishes its own statement in its own events: from the prover, under named cryptographic
or physical assumptions, or from other protocols' statements and physical facts. A protocol imports
`verity.primitives`, and another protocol only for the events it reads; one whose construction holds others composes
them, and its guarantee is proved from theirs.

The products: `remote` (integrity by proofs, `pouw`, `compute` against declared capacity, `nci`) and `onsite`
(the same with measured capacity and a replayer, plus `pous` and `memory`, `transparency` and `network`)."""
