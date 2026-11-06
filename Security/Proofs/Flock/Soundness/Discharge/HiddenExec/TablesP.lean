import Proofs.Flock.Soundness.Discharge.HiddenExec.Tables

/-!
# The tables of a one-table hidden-output session at public inputs

`Tables.lean` at the verifier's public inputs `own` (`Exec.setupOfP I own`):

* `DrawSetupHP I own mPts S d`: `DrawSetupH` with `Stmt.setupHidden … own`;
* `TableHP I own mPts S`: `TableH` with that run, and its regions holding the statement's (`hregs`, which
  `AcceptedHP` reads);
* `DrawSetupHP.table`, `DrawSetupHP.refused`, `tableAtHP`, `planAtHP`, `tabAtHP`, `tablesAtHP`: `DrawSetupH.table`,
  `DrawSetupH.refused`, `tableAtH`, `planAtH`, `tabAtH`, `tablesAtH` at `own`.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.Integrate (DrawSetup)

variable {n : ℕ} (I : Inputs) (own : Option ByteArray) (mPts : ℕ)

/-- **A draw flock-verify sets up under hidden outputs at the verifier's public inputs `own`**: `DrawSetupH` with
`Stmt.setupHidden … own`. -/
structure DrawSetupHP (S : Finset (Fin n)) (d : Lean.Json) where
  st : Stmt
  h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program own = .ok st
  D : Draw.UnitDraw
  hD : Draw.UnitDraw.ofJson d = .ok D
  hS : ∀ u ∈ S, u.val ∈ D.proved
  hg : 0 < st.c.g
  hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl
  m : ℕ
  k0 : ℕ
  l₀ : Accounting.Level
  l₁ : Accounting.Level
  rest : List Accounting.Level
  ext : ℕ
  hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
  hm : 13 ≤ st.m
  hmPts : st.mPts = mPts
  wf : StmtWF st ∧ RegionsWF st

/-- **A hidden-output draw's table at `own` with its sites**: `TableH` at `Stmt.setupHidden … own`, its regions holding
the statement's. -/
structure TableHP (S : Finset (Fin n)) where
  T : TabSpec mPts
  st : Stmt
  d : Lean.Json
  h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program own = .ok st
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hregs : ∀ r ∈ st.regions.toList, regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs
  hwf : StmtWF st
  hT : T.S = stmtOf st hwf regs
  vu : ∀ u ∈ S, ℕ
  hvu : ∀ u (h : u ∈ S), vu u h < st.c.g
  o : ∀ u ∈ S, Fin (2 ^ T.S.nbl)

variable {I own mPts}

namespace DrawSetupHP

variable {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetupHP I own mPts S d)

/-- The table at statement regions `regs` (`DrawSetupH.specAt`). -/
noncomputable def specAt (regs : List (Model.Region x.st.c.kLog (x.st.m - x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) : TabSpec mPts where
  S := stmtOf x.st x.wf.1 regs
  m := x.m
  k0 := x.k0
  l₀ := x.l₀
  l₁ := x.l₁
  rest := x.rest
  ext := x.ext
  hsch := x.hsch
  hm := x.hm
  ptLocal := PT_LOCAL
  hlay := ⟨hregs, by
    obtain ⟨-, hr⟩ := x.wf
    show PT_LOCAL + (x.st.m - x.st.c.kLog) ≤ mPts
    have h1 := hr.mPts
    have h2 := hr.hm
    have h3 := x.hmPts
    omega⟩

/-- A drawn unit's rank among the draw's proved units. -/
def rank (u : Fin n) : ℕ := x.D.proved.toList.idxOf u.val

theorem rank_lt {u : Fin n} (h : u ∈ S) : x.rank u < x.D.proved.size := by
  have := List.idxOf_lt_length_of_mem (Array.mem_toList_iff.2 (x.hS u h))
  simpa [rank] using this

theorem proved_rank {u : Fin n} (h : u ∈ S) : x.D.proved[x.rank u]? = some u.val := by
  have hl : x.rank u < x.D.proved.toList.length := by simpa using x.rank_lt h
  have := List.getElem_idxOf (x := u.val) (xs := x.D.proved.toList) hl
  rw [Array.getElem?_eq_getElem (x.rank_lt h), ← Array.getElem_toList]
  exact congrArg some this

theorem rank_div_lt {u : Fin n} (h : u ∈ S) : x.rank u / x.st.c.g < 2 ^ (x.st.m - x.st.c.kLog) := by
  obtain ⟨-, hr⟩ := x.wf
  rw [Nat.div_lt_iff_lt_mul x.hg, hr.hm, Nat.add_sub_cancel_left, Nat.mul_comm]
  exact lt_of_lt_of_le (x.rank_lt h) x.hblk

/-- **The draw's table** (`DrawSetupH.table`): `stmtOf` of the statement at its regions, each unit at its rank `r`:
block `r / c.g`, template instance `r % c.g`. -/
noncomputable def table : TableHP I own mPts S where
  T := x.specAt (regionsOf x.st) (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1
  st := x.st
  d := d
  h := x.h
  regs := regionsOf x.st
  hregs _ hr := List.mem_map_of_mem hr
  hwf := x.wf.1
  hT := rfl
  vu u _ := x.rank u % x.st.c.g
  hvu _ _ := Nat.mod_lt _ x.hg
  o u h := ⟨x.rank u / x.st.c.g, x.rank_div_lt h⟩

/-- The regions of a refused draw's table: the reference setup's, and Integrate's `contra`. -/
theorem refused_wf : ∀ R ∈ regionsOf x.st ++ DrawSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog),
    R.free.Nodup ∧ R.free.length ≤ PT_LOCAL := by
  intro R hR
  rcases List.mem_append.1 hR with hR | hR
  · exact (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1 R hR
  · exact DrawSetup.contra_wf R hR

/-- **The table of a refused draw `S'`** (`DrawSetupH.refused`): the reference setup's statement with `contra`'s
regions, which no witness satisfies. -/
noncomputable def refused (S' : Finset (Fin n)) : TableHP I own mPts S' where
  T := x.specAt (regionsOf x.st ++ DrawSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog)) x.refused_wf
  st := x.st
  d := d
  h := x.h
  regs := regionsOf x.st ++ DrawSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog)
  hregs _ hr := List.mem_append_left _ (List.mem_map_of_mem hr)
  hwf := x.wf.1
  hT := rfl
  vu _ _ := 0
  hvu _ _ := x.hg
  o _ _ := ⟨0, Nat.two_pow_pos _⟩

theorem refused_unsat (S' : Finset (Fin n)) (z : Witness (x.refused S').T.S.m) :
    ¬ (x.refused S').T.S.Satisfies z :=
  DrawSetup.contra_unsat (S := (x.refused S').T.S) (fun _ hR => List.mem_append_right _ hR) z

end DrawSetupHP

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupHP I own mPts S₀ (dj S₀))

open Classical in
/-- **The table at draw `S`** at `own` (`tableAtH`). -/
noncomputable def tableAtHP (S : Finset (Fin n)) : TableHP I own mPts S :=
  if h : Nonempty (DrawSetupHP I own mPts S (dj S)) then (Classical.choice h).table else x₀.refused S

/-- **A refused draw's table is unsatisfiable** (`tableAtH_unsat`). -/
theorem tableAtHP_unsat {S : Finset (Fin n)} (hS : ¬ Nonempty (DrawSetupHP I own mPts S (dj S)))
    (z : Witness (tableAtHP dj x₀ S).T.S.m) : ¬ (tableAtHP dj x₀ S).T.S.Satisfies z := by
  revert z
  unfold tableAtHP
  rw [dite_eq_right_of_eq_false (eq_false hS)]
  exact x₀.refused_unsat S

variable {Reg : Type}

/-- The one-table plan at `own` (`planAtH`). -/
noncomputable abbrev planAtHP : Finset (Fin n) → Reg → List (TabSpec mPts) := fun S _ => [(tableAtHP dj x₀ S).T]

/-- Every drawn unit is in the plan's one table. -/
abbrev tabAtHP : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planAtHP (Reg := Reg) dj x₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

/-- The one-table plan's tables at `own` (`tablesAtH`). -/
noncomputable abbrev tablesAtHP : Finset (Fin n) → Reg → TabSpec mPts := fun S _ => (tableAtHP dj x₀ S).T

end FlockSoundness.Discharge.HiddenExec
