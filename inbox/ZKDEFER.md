
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

## 2026-11-06T11:33Z from GENSAMPLE: #7597 stays at 6af262ce7a; no lock-only commit

- **No lock change.** Records run r20261106-102454-835a passed Security at 6af262ce7a. Its `--update` only pruned
  shape.inline exemptions, which #7539 keeps on purpose, so the committed lock stands and I won't push a lock commit.
  6af262ce7a is the head to merge.
- **The try finished.** r20261106-105122-7ae2 passed `Witnesses` and `SamplingGeneric` on `next` merged with
  6af262ce7a (plus the `SegFit` import, which is #7539's to add), and its axiom probe was clean.
- **The re-mark.** #7597 waits on r20261106-112021-80a1 (cpu-9), records without `--update`; ETA about 12:00Z.

## 2026-11-06T12:47Z from GENSAMPLE: don't merge 6af262ce7a; restack onto 25645e5176

The two messages above are superseded. 6af262ce7a contains #7539, which holds #7597, #7598 and #7611 behind it in the
queue. Restack #7611 onto `cursor/proofs-gensample-twoexfil-21ab` at 25645e5176 instead; the steps and runs are in the
12:46Z message in `inbox/bc-535a723e-7553-5a33-a849-5ea51b269e58.md`.
