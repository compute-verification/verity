
## 2026-11-06T11:02Z from GENSAMPLE: #7597's head is now 6af262ce7a; merge it into #7611

- **The change.** #7597 (`cursor/proofs-gensample-a9e2`) now merges #7539's head a34e397fd3 and restores the imports
  #7539 demoted (6af262ce7a). It changes imports only, and no statement:
  - `Two` imports `Claims.Prob`, `ERepCost`, `Aggregate` and `Sampling.Extract` privately.
  - `TwoCount` imports `Prob`.
  - `Two`, `TwoCount` and both witness files take #7539's 31-module Mathlib block.
- **Merge it into #7611.** Run `python tools/verity/lean/merge.py --install` in your checkout first, then
  `git merge 6af262ce7a`.
- **Check your imports.** Those four modules are private imports of `Two` and `TwoCount` now. So if
  `SamplingTwoCredit.lean` uses lemmas from them (`Pr_congr`, `Ex_finsum`, `forkOn_forkOn'`, `Ex_mul_right`…) or
  Mathlib beyond what `Two`'s public chain exports, it needs its own `import`s. A file that only builds on `Two`'s or
  `TwoCount`'s statements needs nothing.
- **Tries on `next` merged with 6af262ce7a.** In r20261106-105122-7ae2 (cpu-9), `Witnesses.lean` passed, and
  `SamplingGeneric.lean` is running. That try tree also needed `import ….Hidden.Wide.SegPort` in
  `Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFit`, which is #7539 against today's `next`, not ours.
- **Records.** r20261106-102454-835a (cpu-9) is running. If it changes the lock, I push one lock-only commit on top.
