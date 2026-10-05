import FlockSoundness.Discharge.Exec
import FlockSoundness.Discharge.Layout
import FlockSoundness.Discharge.Placed
import FlockSoundness.Protocol.Discharge.Integrate.Sites

/-!
# The tables and sites of a one-table session

`Prog.flock_headline_live` takes a table `T S R` at every draw `S` and each drawn unit's site in it (`Placed.Sites`).
This file builds both from the session's inputs `I : Inputs` (the verify command's flags: one circuit file, one public
file, one table), at the draw the coin server writes for `S` (`dj S`, the draw's JSON):

* `DrawSetup I mPts S d`: `Stmt.setupH` on `I` accepts the draw `d`, which decodes to a draw `D` proving every unit of
  `S`; the statement's blocks hold `D`'s proved units, its `fast100` schedule has two levels or more, and its link
  points are `mPts`.
* `DrawSetup.table`: its table, the statement `(setupH_layout ht h).model (regionsOf st)` (what `Placed.Accepted` reads,
  so `Accepted.hT` is `rfl`), and each unit `u`'s site: its rank `r` among `D.proved`, block `r / c.g` and VU `r % c.g`.
  Distinct units have distinct ranks and so distinct sites (`table_site_inj`).
* `DrawSetup.refused`: at a draw with no `DrawSetup`, a table no witness satisfies (`refused_unsat`): a reference
  setup's statement with two regions pinning the same bit to 0 and to 1.
* `tableAt`: the draw's table, `table` when the draw has a `DrawSetup` and `refused` otherwise; `tableAt_stmt` says it
  is setupH's statement wherever setupH accepts.
* `sitesAt`, `sitesAt_parse`: the `Sites` of the plan `fun S R => [tableAt … S]`, and `hparse` from the session's one
  circuit file.
* `sitesAt_bcSite`: e2e-layout's site fact `BcSite` at every satisfiable site: at a `DrawSetup` table by the rank, and at
  a refused table vacuously.
-/

namespace FlockSoundness.Discharge.Integrate

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

variable {I mPts} (ht : I.tags.typed = false)

namespace DrawSetup

variable {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetup I mPts S d)

theorem rank_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S) (e : x.rank u = x.rank u') : u = u' := by
  have a := x.proved_rank h
  rw [e, x.proved_rank h'] at a
  exact Fin.ext (Option.some.inj a).symm

/-- **Distinct drawn units have distinct sites** in the draw's table. -/
theorem table_site_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S)
    (e : ((x.table ht).o u h).val = ((x.table ht).o u' h').val ∧ (x.table ht).vu u h = (x.table ht).vu u' h') :
    u = u' := by
  refine x.rank_inj h h' ?_
  obtain ⟨e1, e2⟩ := e
  have e1' : x.rank u / x.st.c.g = x.rank u' / x.st.c.g := e1
  have e2' : x.rank u % x.st.c.g = x.rank u' % x.st.c.g := e2
  rw [← Nat.div_add_mod' (x.rank u) x.st.c.g, ← Nat.div_add_mod' (x.rank u') x.st.c.g, e1', e2']

end DrawSetup

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀))

variable {Reg : Type}

variable {c : Flock.Circuit}

/-- **Distinct drawn units have distinct sites** wherever flock-verify sets up the draw (red-team F1: the count judges
each site once). -/
theorem sitesAt_inj {S : Finset (Fin n)} (hS : Nonempty (DrawSetup I mPts S (dj S))) (R : Reg) {u u' : Fin n}
    (h : u ∈ S) (h' : u' ∈ S)
    (e : ((sitesAt ht dj x₀).o S R u h).val = ((sitesAt ht dj x₀).o S R u' h').val ∧
      (sitesAt ht dj x₀).vu S R u h = (sitesAt ht dj x₀).vu S R u' h') : u = u' := by
  have ht' : tableAt ht dj x₀ S = (Classical.choice hS).table ht := by unfold tableAt; rw [dite_eq_left_of_eq_true (eq_true hS)]
  refine (Classical.choice hS).table_site_inj ht h h' ?_
  have e' : ((tableAt ht dj x₀ S).o u h).val = ((tableAt ht dj x₀ S).o u' h').val ∧
      (tableAt ht dj x₀ S).vu u h = (tableAt ht dj x₀ S).vu u' h' := e
  rw [ht'] at e'
  exact e'

variable {pop : Public}

end FlockSoundness.Discharge.Integrate
