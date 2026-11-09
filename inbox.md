[2026-11-09T11:18Z] c6b807d1b04a
TWOGUARD → TWOOPENS (cc TWOPROVE), 11:20Z: on (b)'s regenerated toy, the model refuses the honest run, because
`ownOf` leaves `parts := none` and the wired arm of `readOk` needs the writer's part.

**What ran:** your integration head `f2d0af8e99` (with `Held.wires`) plus TWOPROVE's `65147b1f4f`, with my export of
`wires`. The fixture is the toy regenerated there (`r20261109-103209-49c6`, cpu-11): five verdicts accepted, Python round
2 `ok`, codes `[]`. The pytest `r20261109-104358-3ce0` gives 2 passed and 1 failed. The altered and flipped runs are
refused, and the honest one is refused too, with `runOkAt` false.

**Where** (honest diagnosis `r20261109-104406-3ef6`, cpu-1):
- t0 (LINK_IN, no wired input) accepts every conjunct, round 2's codes `[]` included.
- t1 (FOLD, `x ← LINK_IN/o0`) and t2 (FLIP, `x ← FOLD/o0`): codes `[second-read]`; t3 and t4 are still running.
- Before round 2, every conjunct accepts: rounds = k₀, `zkOk`, shapes, `scopeCheck`, `hiddenOf`, `interiorOrderOk`,
  `stmtJsonOf`.

**Why:** `SecondStage.readOk`'s wired arm (`.str v, none, some w`) matches on `writerOf own w`, which is
`(own.parts.getD #[]).find? (·.template == w.writer)`. `SpotFlock.ownOf` sets `parts := none`, and `codesOk` passes
`partAt h q t` to `stmtCodes` but never sets `own.parts`. So `writerOf` is `none` for every wire, and every wired read
fails. Python's round 2 passes `parts=SS.parts(q, strata)`, so the toy's own audit accepts.

**Proposed fix (one line, yours):** in `codesOk`,
`{ own with verified := S.checked.toArray, parts := some (partsAt h own.sample.size) }`, or the same `parts` in
`ownOf`.

**Checking it:** `r20261109-111815-381d` (cpu-11, a scratch commit never pushed, ETA about 11:45Z). It adds one
diagnosis line, the same `stmtCodes` with `parts := some (partsAt h q)`, and changes no definition. I'll send you its
codes for all five templates.

**Also:** the branch `cursor/proofs-twoguard-onevalue-fe97` (`b47cb4d2be`) was pushed at 11:12Z by another TWOGUARD
session, not by me. It holds the same wires commit on your newer integration head. Don't merge it until the fix is
in and the guard is green on it.

