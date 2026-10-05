import FlockSoundness.Discharge.HiddenExec.Defs
import FlockSoundness.Discharge.Integrate
import FlockSoundness.Protocol.Discharge.HiddenExec.Tables

/-!
# The tables and sites of a one-table hidden-output session

Integrate's `DrawSetup`, `Table`, `tableAt` and `planAt` (`Integrate/Sites.lean`) for `Stmt.setupHidden`. The public
file holds every row's `b ‖ c` and no output word, so a table is `stmtOf` of the hidden-output statement at its regions,
read off its layout (`Refine.stmtOf`, which reads `Δ` and the regions `setupHidden` built), not `setupH`'s
`Layout.model`:

* `DrawSetupH I mPts S d`: `Stmt.setupHidden` on `I` accepts the draw `d`, which decodes to a draw `D` proving every unit
  of `S`; the statement's blocks hold `D`'s proved units, its `fast100` schedule has two levels or more, and its link
  points are `mPts` (`DrawSetup`'s fields, with `setupHidden` for `setupH`), and the statement is well formed (`wf`,
  which `setupHidden_wf` proves of every statement `setupHidden` returns).
* `TableH I mPts S`: a table, the hidden-output statement it is `stmtOf` of (at a draw `d`, at regions `regs`), and
  each drawn unit's VU and block.
* `DrawSetupH.table`: the table at `regionsOf st`, each unit at its rank `r` among `D.proved`, block `r / c.g`, VU
  `r % c.g`. `DrawSetupH.refused`: at a draw with no `DrawSetupH`, the reference setup's statement with Integrate's
  `contra` regions, which no witness satisfies.
* `tableAtH`, `planAtH`, `tabAtH`, `tablesAtH`: the draw's table, the one-table plan and its tables, as
  `Prog.flock_headline_live_on` takes them.

`drawSetupH_of_accepts` and `tableAtH_of_accepts` (`Accepted.lean`): at a draw flock-verify accepts under hidden
outputs, the table is `stmtOf` of the statement it accepted.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs Accepts recordDraw)
open FlockSoundness.Discharge.Integrate (DrawSetup)

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

variable {I mPts}

namespace DrawSetupH

variable {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetupH I mPts S d)

theorem rank_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S) (e : x.rank u = x.rank u') : u = u' := by
  have a := x.proved_rank h
  rw [e, x.proved_rank h'] at a
  exact Fin.ext (Option.some.inj a).symm

/-- **Distinct drawn units have distinct sites** in the draw's table. -/
theorem table_site_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S)
    (e : (x.table.o u h).val = (x.table.o u' h').val ∧ x.table.vu u h = x.table.vu u' h') : u = u' := by
  refine x.rank_inj h h' ?_
  obtain ⟨e1, e2⟩ := e
  have e1' : x.rank u / x.st.c.g = x.rank u' / x.st.c.g := e1
  have e2' : x.rank u % x.st.c.g = x.rank u' % x.st.c.g := e2
  rw [← Nat.div_add_mod' (x.rank u) x.st.c.g, ← Nat.div_add_mod' (x.rank u') x.st.c.g, e1', e2']

end DrawSetupH

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))

variable {Reg : Type}

end FlockSoundness.Discharge.HiddenExec
