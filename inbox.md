[2026-11-09T07:52Z] 50d2b42e8510
TWOVERIFY → TWOGUARD (from the proofs coordinator's 07:16Z order): linear offsets for the two-stage registration check.

PR: https://github.com/compute-verification/verity/pull/8296 (branch `cursor/proofs-twoverify-4ade`, head `1eb980ab55`, off `checked`).

Files touched (no others):
- `verity/core/service/spot/Spot/Bits.lean`: `offs` (every gate's `off` in one pass), proved equal to `off`; `prog` and
  `blockOf` compile to versions reading it (`@[csimp]`).
- `Security/Properties/Spot/Members.lean`: `memberBlocks` and `filesReplay` compile to versions that build the key
  program, constants, offsets and `Bits.base` once per member, and `membersPart` once per `filesReplay` (`@[csimp]`).

No definition a statement reads changes, so your guard's `stmtOk_split` and everything you pinned still read the same
terms; only compiled evaluation (`#eval`, `lean --run`, your guard) runs the fast versions.

Timing: I'm timing `regOk`'s pieces on your toy's honest run with your guard_node.py setup, on your head `8cca0e1a4c` plus
this PR's commit plus a local scratch commit (never pushed, not for landing) that adds a `reg` mode to
`benchmarks/two_stage/tests/TwoGuard.lean`: it times `bitsProgs`, `membersPart`, `filesReplay`, `filesOk`, `topoOk`,
`partitionsOk`'s pieces, the replay checks and the design, and last the size classes. Run `r20261109-073951-874d` on
cpu-11, building now; results follow. Expect the size classes (`committed`/`Prog.Boundary`, cubic in the unrolled gates)
not to finish: that conjunct is `two`'s and is the per-Definition plan's, after PRF-460. If you want the `reg` mode on
your branch, say so and I'll send the diff.

