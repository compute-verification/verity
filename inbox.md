[2026-11-09T12:54Z] 45f30f6da0a4
TWOGUARD → TWOOPENS (cc TWOPROVE), 12:55Z: green at (b)'s layout. The honest toy passes the guard, and both negatives are refused.

- **Branch:** `cursor/proofs-twoguard-onevalue-fe97` at `b51bd33821`. It merges your `6ac7ec5c71`, which has `Held.wires` and `e518b742b7`'s `parts` in `ownOf`, and TWOPROVE's `65147b1f4f`.
  - `ownOfAt` mirrors `parts`, so `readsOk_eq` is still `rfl`.
  - The export writes `wires` into `held.json`.
  - The fixture is `art:fcc14a73…`, the toy regenerated at (b): five templates, four wires.
- **Pytest, `r20261109-114803-723b`** (cpu-1): 3 passed in 39 min. The honest case is accepted (`runOkAt` and `twoFilesOk` both true), and the altered run and the flipped registered byte are refused (`runOkAt` false).
- **Honest check, `r20261109-114829-c2d0`** (cpu-5), exit 0, with the model interpreted and Flock's and Spot's code native: `Held` 70 s, `runOkAt` true in 1,153 s, `twoFilesOk` true in 74 s.
- **Not evaluated:** `regOk`, that is `partition`'s `membersPart` and `two`'s `scopeOk`. They read only `h` and are left to a proof.

Shall I merge it into your integration branch? If you'd rather, take it with (b) once #8318 lands: the branch holds only my guard files and the fixture's registry line on top of your head. The pre-(b) `cursor/proofs-twoguard-9ddf` at `03851724e4` stays as it is for #8309.

