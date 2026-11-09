[2026-11-09T11:17Z] 1f56a1227a0e
TWOPROVE → TWOGUARD, TWOOPENS (11:18Z): #8318's final head is 37e4f81caa, marked ready at 11:15Z. It is 65147b1f4f plus a merge of origin/checked a190c406c7 (docstring conflicts from #8308 in second_stage.py, Pieces.lean and SecondStage.lean, and #7918's rate-only first stage), with no other change. The toy audit and the suites at 37e4f81caa pass (r20261109-105529-247f, r20261109-105552-deed). Merge 37e4f81caa rather than 65147b1f4f if you take (b) before it lands.

[2026-11-09T11:18Z] a93dd5229d7e
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

[2026-11-09T11:19Z] e48f4e12f610
TWOGUARD → TWOOPENS (cc TWOPROVE), 11:25Z: at (b)'s layout the model refuses the honest toy at its first wired reader. The cause is `parts := none` in the model's `Own`. The fix is one field in Code.lean, yours.

- **Run:** pytest `r20261109-104358-3ce0` (cpu-11) on the toy regenerated at (b), `r20261109-103209-49c6`. The Python round 2 gave `{"ok": true, "codes": []}` and all five Lean verdicts were accepted, yet the honest case fails: `runOkAt` is false. The altered and flipped cases are refused, as they should be.
- **Where:** the honest diagnosis `r20261109-104406-3ef6` (cpu-1, still running) passes everything at template 0, LINK_IN. At template 1, FOLD, whose `x` is wired to LINK_IN's `o0`, the codes as `codesOk` reads them are `[second-read]`. Every other check there passes (rounds, `zkOk`, shapes, `scopeCheck`, `hiddenOf`, `interiorOrderOk`, `pinOk`).
- **Why:** `SecondStage.readOk`'s wired arm (`.str v, none, some w`) needs `writerOf own w`, which is `(own.parts.getD #[]).find? (·.template == w.writer)`. `SpotFlock.ownOf` sets `parts := none`, and `codesOk` doesn't fill it, so every wired read fails. `audit.round_two` passes the parts, which is why the Python accepts.
- **Fix:** in `codesOk`, give `stmtCodes` `{ own with verified := S.checked.toArray, parts := some (partsAt h own.sample.size) }`. It already computes `partsAt h own.sample.size` for `posOf`. My mirror calls `codesOk` itself, so it follows with no change. If you'd rather set it in `ownOf`, note that `partsAt` is defined after `ownOf`, and tell me so I mirror it.
- **Confirmation in flight:** `r20261109-111815-381d` (cpu-11, a scratch diagnosis at `585de1cd51`) prints each template's codes with the parts set that way.

My (b) branch is `cursor/proofs-twoguard-onevalue-fe97` at `b47cb4d2be`: my `03851724e4`, merged with your `cursor/proofs-sessions-onevalue-3e7a` (`36f1b429b7`), plus the wires export. Once your fix is on a head, I'll merge it and rerun the three cases on the (b) fixture (5,110,008 B, sha256 `f8bfdbf5…`).

[2026-11-09T11:21Z] 466683cb870a
Coordinator 11:22Z, PRIORITY: TWOGUARD found a third model gap. At (b)'s layout the model refuses the honest toy at every wired read ([second-read] at FOLD and FLIP) because SpotFlock.ownOf sets parts := none and codesOk never fills it, while SecondStage.readOk looks up the writer's part in own.parts (writerOf). Python passes because audit.round_two supplies parts. Fix (TWOGUARD's): codesOk gives stmtCodes parts := some (partsAt h own.sample.size), which it already computes for posOf. Put it in your integration branch now and tell TWOGUARD (bc-f49559ec-24ca-58ec-873e-33564093fe97) and SLOTSAT the head; SLOTSAT's instance must also have parts filled.

