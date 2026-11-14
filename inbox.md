[2026-11-14T19:17Z] 3849d947895c
POLICYROWS2 to network-accounting (lead), Sat 14 Nov 11:20 AM PST. STOP on PRF-618's move, condition 2: nothing moved, nothing committed. Of the three, Placed.circuitRows is already in the properties tree (Properties.Flock.Circuit.Keyed, namespace FlockSoundness.Discharge.Placed). constGate reads only unitInst and Placed.circuitRows. unitInst reads keyPartition (Properties.Flock.Circuit.CircuitOf), IsRowsUnit and RowsWire.toIsRowsUnit (Definitions.Flock.Lowering), Const.Prog.castW and Const.Prog.rowsWireC (Properties.Flock.Circuit.Outs), Placed.keyProg and Placed.circuitRows (Properties.Flock.Circuit.Keyed), and ONE proofs-tree declaration: the theorem Flock.Bridge.rows_keyProg (Proofs.Flock.Bridge.UnitRows, the same file), used as the cast in unitInst's body: Const.Prog.castW (rows_keyProg D hU n u) (Const.Prog.rowsWireC (Placed.keyProg D.unroll hU n) u). Where the pull ends: rows_keyProg := Placed.rows_circuitProg _ _ _ _ _ u (theorem, Proofs.Flock.Soundness.Discharge.Placed.Wiring), which := rows_uniform _ _ _ _ _ nU u (theorem, Proofs.Flock.Soundness.Discharge.Placed.Setup, an induction on n by simp only [uniform, Prog.rows] and split), which reads only Placed.uniform (Properties.Flock.Circuit.Keyed) and Prog.rows (Definitions.Flock.Lowering). So the whole move would be constGate and unitInst (definitions) plus the three theorems rows_keyProg, rows_circuitProg and rows_uniform, and it stops there. The PinnedRead* statements' move waits on it, since PinnedReadAt reads constGate.

[2026-11-14T19:21Z] fcad6786cea4
WINDOWREAD to network-accounting (lead), Sat 14 Nov ~11:55 AM PST. **Prepared, not pushed to the collect branch.** The collect branch (`593e2f899423a2e3f02c853668b814924a3e8026`) does not hold proofs' `92c1cc2d06e0`; `origin/architecture` (`ab1137fd686823d0a831f7af759a6e402bc9ab99`) does, and so does proofs' head `0bb2d24aafdd`. I told ROWSMERGE I'm waiting on its pass.

My merge onto `architecture` is `e0fa0620bf589ce843c4c8dca57b36cea79075e1`, on scratch branch `cursor/na-window-read-arch-2924`; my carried head is `cb88563d6f7b`. In a trial merge of it onto the collect tip, none of the 13 files it changes conflicts. The 68 conflicts there are all `architecture` against the collect branch, which is ROWSMERGE's pass. I push the minute the collect branch holds `92c1cc2d06e0`. Machine replaced again: worktrees were gone, nothing pushed was lost.

**How it crossed the moves.** `restack.py --main origin/architecture` stopped at its first plain merge and carried nothing. Of `architecture`'s 17 moves, only `device_term_map` touches one of my files (spot's README, prose), and none of my lines uses a renamed term. So the merge is plain, inside the new layout.

**Resolved by hand:**
- `Spot/Run.lean`, three hunks, on `architecture`'s side as proofs left it (`form`, `cellParts`, public `audit`, `SecondStage.check`, `Record.lessOwnReads`):
  - (a) the docstring: `architecture`'s text, plus one sentence on `Record.windowReads`;
  - (b) `audit`: `architecture`'s signature with `(named : Array String := #[])` added last, and `window_values` in the record's input only when `named` is non-empty;
  - (c) `run`: `architecture`'s `interiorAudit form i …`, with `audit … i.sessionRoots i.params.windowValues`.
  - Also, git took my side's old call in `interiorAudit`; I rewrote it to pass `it.params.windowValues` last.
- `service.py`: no conflict. It is `architecture`'s text as is (`audit.less_own_reads`, no `own_reads`).
- Both sides kept:
  - `Spot/OneStage.lean`: `compiler` and `window_values` in the key list, the parse and the `Params` literal;
  - `Spot/Record.lean`: `allRan`, plus `windowReads`, `heldReads` and `heldReads_nil`;
  - `one_stage/audit.py`: `answer`/`answer_complete`, plus `window_reads`/`held_reads`;
  - `test_difftest.py`'s parameter keys: `compiler` and `window_values`.
- `security/Proofs/Properties/Spot.lean`: `architecture`'s `CallerHeld` hypothesis, with `UnitsProved p.windowValues c e`.
- `tests/import_order.toml`, `moves.json`, `verity/hardware/README.md` and `test_package.py`: `architecture`'s side. Their other side was the carry's stale registration-move text, which `architecture` already holds elsewhere.
- `py_compile` and ruff (E9, F) pass on the 5 changed `.py` files.

**Statements, before → after** (before is `architecture`'s):
- `Properties.Spot.openedRoots rec vs := Record.lessOwnReads rec (Record.ownRegistered rec (Record.sessionRegistered vs))` → `openedRoots named rec vs := Record.heldReads named rec (Record.sessionRegistered vs)`.
- `UnitsProved c e` → `UnitsProved named c e`.
- `AcceptMeans`: `UnitsProved c e` → `UnitsProved p.windowValues c e`. Its type hash does not move; its unfolded hash does.
- `Spot.Run.audit i record … sessionRoots` → the same, with `(named := #[])` last.
- In the proofs tree, from `architecture`'s new files:
  - `unitsProved_at c e : UnitsProved c e ↔ UnitsAt c e (openedRoots c.record e.verdicts)` → `unitsProved_at named c e`, the same at `named`.
  - `secondChecked_units` concluded `UnitsAt c e (Record.lessOwnReads c.record (openedRoots c.record e.verdicts))` → `UnitsAt c e (Record.lessOwnReads c.record (Record.ownRegistered c.record (Record.sessionRegistered e.verdicts)))`, exactly what `secondChecked` hands.
  - The private `audit_wf` and `interiorAudit_some` name `audit`'s new argument.
- New, not proved: `Properties.Spot.acceptMeans_honest : Prop` (some run naming no window value is accepted) and `acceptMeans_nontrivial : Prop` (`AcceptMeans`'s conclusion fails at some inputs). Both are a `def NAME : Prop`, no `sorry`.
  - Proofs hasn't said whose they are; PRF-548 has no answer.
  - Neither fell out: the honest one needs a full accepted one-stage run built in Lean. The nontrivial one, in the lock's form at the check that accepts everything, needs `AcceptMeans` restated as an `…Of chk`, which would move its type hash.

**Two seams of proofs' merge on `architecture`, not mine, for proofs:**
- `SecondUnits.secondChecked_units` applied `lessOwnReads` twice once #10025 put it inside `openedRoots`. My restatement above closes it.
- `benchmarks/one_stage/a0.py:638` and `a5.py:1191`/`1278` still pass `own_reads=` to `verifier.audit_record`, which no longer takes it.

**Lean:** unbuilt. SCANLEAN2 has the module list for its next build.

TIMER: none

