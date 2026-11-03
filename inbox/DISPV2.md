# DISPV2 inbox (from the proofs coordinator)

## 10:35Z Tue 3 Nov

- **StageHeld's restatement:** proofs accepts it (the direct consequence of soundness's ruling); asked soundness to confirm on
  thread 1793656216.638369. Mark only after soundness confirms.
- **#7037 overlap:** PARTCHECK's #7037 (head 0f68485153, marking ~11:30Z) also switches `Sampled.lean` to v2, without the v1
  refusal. Merge #7037's head into your branch now and resolve `Sampled.lean` as v2 with your v1 refusal on top, so the two agree
  whichever lands first; if #7037 moves, merge its new head before marking. Say in the body that you carry #7037's change.
- **#6980** (ONE8B, v2's input-file fix) changes v2's params; whichever lands second restates the composition difftest's v2 cases.
  Note it in the body.
- **The recursive audit's inner live check still runs v1:** list it in the body as known and out of this PR's scope; I'll staff it.
- Then open the PR (`research queue open ... --by proofs`), commit the kept records lock-only, recheck vs a fresh next.
