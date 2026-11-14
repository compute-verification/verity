[2026-11-14T21:21Z] a0bfe719e40e
model: Claude Opus 5.5 (claude-opus-5-5-high); subscriptions removed: none held.

NAMES2, 01:21 PM PST: pushed, `architecture` at 9d3fe3d46831. Files moved 0, split 3, left 3.
- Split: `Proofs/InferenceOnly/Lemmas.lean` and the root `Proofs/InferenceOnly.lean` (`Inference` to `Proofs/Slowdown/Lemmas.lean`, its proof to `Proofs/Slowdown.lean`; `ReadBound` stays); `Proofs/InferenceOnly/ReadsIn.lean` (`threaded_reads_by` and the `ThreadedReads` example to `Proofs/Slowdown/ReadsBy.lean`); `Definitions/InferenceOnly.lean` (`slowdown` to `Definitions/Slowdown/Factor.lean`, now `Slowdown.slowdown`). Texts unchanged; full names renamed by `tools/move/slowdown_split_map.toml` (the policy README's two). No lock records these four. The policy root reaches the bound's proofs through `public import Proofs.Slowdown`, since `Proofs/lean-audit.json`'s roots name `Proofs.InferenceOnly` and not `Proofs.Slowdown` (for the regeneration).
- The accounting split put its coverage floor in `Definitions/Slowdown/` with `InferenceOnly` namespaces; `slowdown_namespace_map.toml` now renames those seven (`CapabilityFloor`, `…Honest`, `…Nontrivial`, guarantee and proof, and the assumption `CoversEfficientMethods`) to `Slowdown.*`. Lock renames: 466 now in `lanes/training/architecture/slowdown-lock-moves.json` (11 new).
- Built: all 29 Definitions modules over `Definitions.InferenceOnly` and the coverage ones, with the local `SiteSubtree` patch only; Proofs unbuilt.
- Question (name or place): `Accounting.Slowdown.*` declarations now sit in `Slowdown/` too (`Tasks.lean`, `CoverageToy.lean`, the toy in `Proofs/Slowdown/Coverage.lean`). My reading: they keep `Accounting.Slowdown`, as the split's rule says, since `Slowdown.*` would collide (`Accounting.Slowdown.slowdown` against `Slowdown.slowdown`, `Reads`).
- Left: `Proofs/InferenceOnly/Cover.lean` (umbrella), `Proofs/InferenceOnly/Pull.lean` (`pull`, used by both).

