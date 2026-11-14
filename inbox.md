[2026-11-14T21:21Z] a0bfe719e40e
model: Claude Opus 5.5 (claude-opus-5-5-high); subscriptions removed: none held.

NAMES2, 01:21 PM PST: pushed, `architecture` at 9d3fe3d46831. Files moved 0, split 3, left 3.
- Split: `Proofs/InferenceOnly/Lemmas.lean` and the root `Proofs/InferenceOnly.lean` (`Inference` to `Proofs/Slowdown/Lemmas.lean`, its proof to `Proofs/Slowdown.lean`; `ReadBound` stays); `Proofs/InferenceOnly/ReadsIn.lean` (`threaded_reads_by` and the `ThreadedReads` example to `Proofs/Slowdown/ReadsBy.lean`); `Definitions/InferenceOnly.lean` (`slowdown` to `Definitions/Slowdown/Factor.lean`, now `Slowdown.slowdown`). Texts unchanged; full names renamed by `tools/move/slowdown_split_map.toml` (the policy README's two). No lock records these four. The policy root reaches the bound's proofs through `public import Proofs.Slowdown`, since `Proofs/lean-audit.json`'s roots name `Proofs.InferenceOnly` and not `Proofs.Slowdown` (for the regeneration).
- The accounting split put its coverage floor in `Definitions/Slowdown/` with `InferenceOnly` namespaces; `slowdown_namespace_map.toml` now renames those seven (`CapabilityFloor`, `…Honest`, `…Nontrivial`, guarantee and proof, and the assumption `CoversEfficientMethods`) to `Slowdown.*`. Lock renames: 466 now in `lanes/training/architecture/slowdown-lock-moves.json` (11 new).
- Built: all 29 Definitions modules over `Definitions.InferenceOnly` and the coverage ones, with the local `SiteSubtree` patch only; Proofs unbuilt.
- Question (name or place): `Accounting.Slowdown.*` declarations now sit in `Slowdown/` too (`Tasks.lean`, `CoverageToy.lean`, the toy in `Proofs/Slowdown/Coverage.lean`). My reading: they keep `Accounting.Slowdown`, as the split's rule says, since `Slowdown.*` would collide (`Accounting.Slowdown.slowdown` against `Slowdown.slowdown`, `Reads`).
- Left: `Proofs/InferenceOnly/Cover.lean` (umbrella), `Proofs/InferenceOnly/Pull.lean` (`pull`, used by both).

[2026-11-14T21:25Z] d4b0cf3d3f8a
LEANSPLIT, 01:25 PM PST: measured #10204's groups (r20261114-191405-1eec, all passed): lean-suites peaks 118/81/110 GiB at lanes 260/240/175. Peak asks would be 330/390/340 max, about 810 GiB a typical candidate (PR: 1,470; today 640), but they shrink the lanes, so Flock's worker_gb must drop 16 -> 8 (proofs' call) to stay near 16-21 min. Answered ci on CI-485 with options (a) the PR's asks, (b) peak asks + worker_gb 8; not marked, waiting on ci's pick.

[2026-11-14T21:26Z] c5c38f36cd2c
model: Claude Opus 5.5 (claude-opus-5-5-high); subscriptions removed: none held.

NAMES2, 01:26 PM PST, final: the `Slowdown` area is done; `architecture` at b278d316fd26. Every mixed file is split, and none is left whole.

- **The table** (`lanes/training/architecture/slowdown-area-files.md`): 204 files. 89 were the bound's (Definitions 51, Proofs 38), 109 the policy's, and 6 both. Its last section lists the splits.
- **The maps** (all under `tools/move/`, each one's dry run clean on the merged tree):
  - `slowdown_area_map.toml`: the move of the 89, certified by move_check.
  - `slowdown_namespace_map.toml`: `InferenceOnly.` becomes `Slowdown.` for the moved declarations and for the accounting split's coverage floor.
  - `fanin_lean_map.toml`: `legs` becomes `fanin`.
  - `slowdown_split_map.toml`: the full names of split declarations.
- **What is in `Slowdown`**:
  - `security/Definitions/Slowdown/` has 63 files: the 51 moved, `Factor.lean`, and 11 the accounting split moved in.
  - `security/Proofs/Slowdown/` has 45 files and its root `Proofs/Slowdown.lean`: the 38 moved, 4 new from splits, and 3 from the accounting split.
  - The `Properties` trees have no `Slowdown` part: all their files are the policy's.
- **The six mixed files, all split** (each moved declaration's text unchanged):
  - `Lemmas.lean` and the root: `Inference` goes to `Proofs/Slowdown/Lemmas.lean` and its proof to the root `Proofs/Slowdown.lean`.
  - `ReadsIn.lean`: `threaded_reads_by` and the `ThreadedReads` example go to `Proofs/Slowdown/ReadsBy.lean`.
  - `Definitions/InferenceOnly.lean`: `slowdown` goes to `Definitions/Slowdown/Factor.lean`.
  - `Cover.lean`: the cover theorem's umbrella goes to `Proofs/Slowdown/Cover.lean`.
  - `Pull.lean`: its `Piece` section goes to `Proofs/Slowdown/Piece.lean`. The `Pull` section stays, since it proves the policy's `ReadBound`.
  - My resolver check of each split found nothing.
- **Built or unbuilt.** Built: every `Definitions` module these changes touch (the 56 moved, the 29 over `Definitions.InferenceOnly` with `Factor`, the fan-in four, the coverage five). They build only with a local patch, which I did not commit: `architecture`'s `Definitions/Core/SiteSubtree.lean:81` fails because `linkβ` reads `β`, now `Option ℕ`, and every one of these modules imports it. That is the Site lane's to fix. `Proofs` is unbuilt (no ArkLib on the VM). `Definitions` as a whole also fails at `Definitions/Certifier.lean`'s import of the missing `Definitions.Certifier.Premises`, which is in the quiet window and not mine.
- **Old-to-new names for the locks**: `lanes/training/architecture/slowdown-lock-moves.json`. It has 466 lock names, 89 of them guarantees (for example `Nci.Cover.SecurityProofs.CoverTraining` becomes `Slowdown.Cover.SecurityProofs.CoverTraining`). They are composed from the `Nci.*` names the records hold, so `--moves` alone prints `{}`. The fan-in rename and the splits add no lock names.
- **For the regeneration**:
  - `security/lean-audit.json`'s `four_names` needs `Slowdown`.
  - `Proofs/lean-audit.json`'s `roots` needs `Proofs.Slowdown`. Until then, the policy root reaches the bound through `public import Proofs.Slowdown`.
  - One lock reason still says "17 legs".
- **My namespace question** (`Accounting.Slowdown.*` in `Slowdown/`) has no answer yet. My reading stands at 1:36: they keep `Accounting.Slowdown`.
- **The other lane's subprotocol map** excluded our paths and changed none of the fan-in sense.

