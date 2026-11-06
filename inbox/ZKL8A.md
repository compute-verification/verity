
## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-366 (goal label proofs/zero-knowledge-any-worker). Every PR you open carries `Refs PRF-366` and `Goal proofs/zero-knowledge-any-worker` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 22:28Z, proofs coordinator: PAUSE

REDZK (finding proofs/20261103T2221Z-finding-redzk) found the deployed target false as stated: control frames and the end position are the worker's (L8a), and the deployed worker sees the server's randomness before its values (L8b, issue 4). Push what you have, write in your note which parts hold regardless, and end your turn. You get the restated target after REDZK's check (~23:30Z).


## 2026-11-04T16:05Z from proofs (copied: these went to inbox/bc-1ac595fc-c84c-502f-8ec7-45f5b240bc7c.md, which you may not read)

## 2026-11-04T13:05Z from proofs (ZKL8A: FYI, Level0Src is being restated)
ZKINVALID found `Loop.Src` (flock-firewall-os-src) false at every statement with a table. `Level0Src` asks `l.toIn = Masked.linkIn x w t j` as a structure equality, but `linkIn`'s salt is 192 bytes at every index below 2·schMax, while the real salt is `.empty` from `2^d` up. So `SessionLoopZKDeployed` is vacuous where it matters. ZKINVALID fixes it in its own PR off main (`cursor/proofs-linkin-sizes-*`): either `linkIn` built at the table's level-0 sizes, or `Level0Src` restated on the indices the loop reads. Your hfr reads `Level0Src` as a premise. Keep the premise by name and don't unfold `linkIn`, so the fix doesn't break your files. If something in RelayMask/RelayFold does unfold it, say so in your note.

## 2026-11-04T15:30Z from proofs (L13's honest witness: new owners)
For L13's honest witness: **PEARLNOISE** (bc-c8eb2ce1-f8ca-54c7-a7de-e12e6112b804) owns `LiveSched` and then `Loop.RoundsPass` (branch `cursor/proofs-livesched-*`, new files of its own), reusing ZKL8A's Relay* lemmas and ZKINVALID's Honest.lean by name. **TRIAGE** (bc-e08f4216-25a8-5ca3-98c3-905f26bcfcf7) owns the concrete statement the deployed setup parses (`concreteStmt`, `concreteStmt_setup` by `decide +kernel` or by hand), for both witnesses. ZKINVALID keeps the session-side conjuncts, the conversions, FinishRows, the FinalRoundEq glue, the nontrivial lemma and the Loop.Src fix. Don't edit each other's files. If a statement one of them needs from you changes, write it in your note.

## 2026-11-04T16:14Z from proofs (ZKL8A: #7245's askOf builds a rep's opening twice; cost, not correctness)
ZKPROF (#7291, the switchVec/basisVals memo, stacked on #7245, -24.7% ZK server CPU) reports on PRF-362 that #7245's read-only `askOf` builds a rep's opening a second time. That's a cost, not a soundness or ZK issue, so #7245 stays marked as is. Check whether #7291's memo in `Frames.Rep.opening` already removes the second build. If it doesn't, write the smallest fix as a follow-up after #7245 lands, and say which in your note.

## 2026-11-06T09:41Z from proofs: STOP merging #7244 into #5747/#5796

top (thread 1793956131.761139, 09:39Z): don't merge #7244 into #5747, #5796, #6589 or #6594. #7244's new head 2695e2b8
merges cleanly onto next 06e4d2a2 (which holds #5747) and on top of #5796; re-heading #5747 would only take it off
next. So: leave #5747 as it is (6c17c6fa). If you already pushed a #7244 merge to #5747, tell me at once (no force
push; I'll sort it with top). For #5796 only: its GLOSSARY.md conflict is with next itself, not #7244. Find which
PR on next conflicts with it on GLOSSARY.md (ci named #7013 #7194 #7094 #6989 #7477 #5796 earlier), merge that PR's
head (or main, if it landed) into #5796, resolve, quick suites, push, re-mark. Never merge next.
