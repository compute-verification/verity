/-!
# C-Flock's definitions (trusted)

The vocabulary C-Flock's guarantees and named assumptions are stated in: the soundness model's games, programs,
circuits, layouts and statements, and level3's definitions, each under the name it has in `Proofs.Flock` before it
moved (`FlockSoundness.X`, `FlockLevel3.X`). They build on Mathlib alone: a module here imports Mathlib, core's
definitions and other modules of `Definitions.Flock`, never an assumption, a guarantee, a proof or ArkLib (`layers` in
`lean-audit.json`).

This tree, `Specs.Flock.Assumptions` and `Specs.Flock.Guarantees` are C-Flock's trusted text. `Proofs.Flock` holds the
proofs and every lemma.
-/
