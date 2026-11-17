module

public import Proofs.Flock.Soundness.Discharge.Hidden.Commit
public import Proofs.Flock.Soundness.Discharge.Integrate.Sites
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Refine.Setup
public import Proofs.Flock.Soundness.Discharge.Integrate.Headline
public import Proofs.Flock.Soundness.Discharge.Integrate

/-! The tables and sites of a one-table hidden-output session, each table `stmtOf` of `Stmt.setupHidden`'s statement. -/

@[expose] public section

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs Accepts recordDraw)
open FlockSoundness.Discharge.Integrate (SampleSetup)

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

/-- **A sample verity-verify-proof sets up under hidden outputs**: `Integrate.SampleSetup` with `Stmt.setupHidden` for
`Stmt.setupH`. -/
structure SampleSetupH (S : Finset (Fin n)) (d : Lean.Json) where
  st : Stmt
  h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st
  D : Draw.UnitSample
  hD : Draw.UnitSample.ofJson d = .ok D
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

/-- **A hidden-output sample's table with its sites**: the table, the statement `setupHidden` accepted at a sample `d` on
the session's inputs, the regions at which the table is that statement's `stmtOf`, and each sampled proof block's template instance and block. -/
structure TableH (S : Finset (Fin n)) where
  T : TabSpec mPts
  st : Stmt
  d : Lean.Json
  h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hwf : StmtWF st
  hT : T.S = stmtOf st hwf regs
  vu : ∀ u ∈ S, ℕ
  hvu : ∀ u (h : u ∈ S), vu u h < st.c.g
  o : ∀ u ∈ S, Fin (2 ^ T.S.nbl)

variable {I mPts}

namespace SampleSetupH

variable {S : Finset (Fin n)} {d : Lean.Json} (x : SampleSetupH I mPts S d)

/-- The table at statement regions `regs`: `stmtOf` of the hidden-output statement, its schedule, and the link layout
at `PT_LOCAL`. -/
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

/-- A sampled proof block's rank among the sample's proved units. -/
def rank (u : Fin n) : ℕ := x.D.proved.toList.idxOf u.val

theorem rank_lt {u : Fin n} (h : u ∈ S) : x.rank u < x.D.proved.size := by
  have := List.idxOf_lt_length_of_mem (Array.mem_toList_iff.2 (x.hS u h))
  simpa [rank] using this

theorem proved_rank {u : Fin n} (h : u ∈ S) : x.D.proved[x.rank u]? = some u.val := by
  have hl : x.rank u < x.D.proved.toList.length := by simpa using x.rank_lt h
  have := List.getElem_idxOf (x := u.val) (xs := x.D.proved.toList) hl
  rw [Array.getElem?_eq_getElem (x.rank_lt h), ← Array.getElem_toList]
  exact congrArg some this

theorem rank_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S) (e : x.rank u = x.rank u') : u = u' := by
  have a := x.proved_rank h
  rw [e, x.proved_rank h'] at a
  exact Fin.ext (Option.some.inj a).symm

theorem rank_div_lt {u : Fin n} (h : u ∈ S) : x.rank u / x.st.c.g < 2 ^ (x.st.m - x.st.c.kLog) := by
  obtain ⟨-, hr⟩ := x.wf
  rw [Nat.div_lt_iff_lt_mul x.hg, hr.hm, Nat.add_sub_cancel_left, Nat.mul_comm]
  exact lt_of_lt_of_le (x.rank_lt h) x.hblk

/-- **The sample's table**: `stmtOf` of the hidden-output statement at its regions, and each sampled proof block at its rank `r`:
block `r / c.g`, template instance `r % c.g`. -/
noncomputable def table : TableH I mPts S where
  T := x.specAt (regionsOf x.st) (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1
  st := x.st
  d := d
  h := x.h
  regs := regionsOf x.st
  hwf := x.wf.1
  hT := rfl
  vu u _ := x.rank u % x.st.c.g
  hvu _ _ := Nat.mod_lt _ x.hg
  o u h := ⟨x.rank u / x.st.c.g, x.rank_div_lt h⟩

theorem table_site_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S)
    (e : (x.table.o u h).val = (x.table.o u' h').val ∧ x.table.vu u h = x.table.vu u' h') : u = u' := by
  refine x.rank_inj h h' ?_
  obtain ⟨e1, e2⟩ := e
  have e1' : x.rank u / x.st.c.g = x.rank u' / x.st.c.g := e1
  have e2' : x.rank u % x.st.c.g = x.rank u' % x.st.c.g := e2
  rw [← Nat.div_add_mod' (x.rank u) x.st.c.g, ← Nat.div_add_mod' (x.rank u') x.st.c.g, e1', e2']

theorem refused_wf : ∀ R ∈ regionsOf x.st ++ SampleSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog),
    R.free.Nodup ∧ R.free.length ≤ PT_LOCAL := by
  intro R hR
  rcases List.mem_append.1 hR with hR | hR
  · exact (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1 R hR
  · exact SampleSetup.contra_wf R hR

/-- **The table of a refused sample `S'`**: a reference setup `x`'s statement with `contra`'s regions, which no witness
satisfies (`refused_unsat`), every unit at block 0, template instance 0. -/
noncomputable def refused (S' : Finset (Fin n)) : TableH I mPts S' where
  T := x.specAt (regionsOf x.st ++ SampleSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog)) x.refused_wf
  st := x.st
  d := d
  h := x.h
  regs := regionsOf x.st ++ SampleSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog)
  hwf := x.wf.1
  hT := rfl
  vu _ _ := 0
  hvu _ _ := x.hg
  o _ _ := ⟨0, Nat.two_pow_pos _⟩

theorem refused_unsat (S' : Finset (Fin n)) (z : Witness (x.refused S').T.S.m) :
    ¬ (x.refused S').T.S.Satisfies z :=
  SampleSetup.contra_unsat (S := (x.refused S').T.S) (fun _ hR => List.mem_append_right _ hR) z

end SampleSetupH

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀))

open Classical in
/-- **The table at sample `S`**: the sample's table (`SampleSetupH.table`) when verity-verify-proof sets up the sample's JSON `dj S`
under hidden outputs, and otherwise the reference setup `x₀`'s refused table. -/
noncomputable def tableAtH (S : Finset (Fin n)) : TableH I mPts S :=
  if h : Nonempty (SampleSetupH I mPts S (dj S)) then (Classical.choice h).table else x₀.refused S

theorem tableAtH_stmt {S : Finset (Fin n)} (x : SampleSetupH I mPts S (dj S)) :
    ∃ y : SampleSetupH I mPts S (dj S), y.st = x.st ∧ tableAtH dj x₀ S = y.table := by
  refine ⟨Classical.choice ⟨x⟩, ?_, by unfold tableAtH; rw [dite_eq_left_of_eq_true (eq_true ⟨x⟩)]⟩
  have e := (Classical.choice (⟨x⟩ : Nonempty (SampleSetupH I mPts S (dj S)))).h.symm.trans x.h
  exact Except.ok.inj e

theorem tableAtH_cases (S : Finset (Fin n)) :
    (∃ x : SampleSetupH I mPts S (dj S), tableAtH dj x₀ S = x.table) ∨ tableAtH dj x₀ S = x₀.refused S := by
  unfold tableAtH
  split
  · exact .inl ⟨_, rfl⟩
  · exact .inr rfl

theorem tableAtH_unsat {S : Finset (Fin n)} (hS : ¬ Nonempty (SampleSetupH I mPts S (dj S)))
    (z : Witness (tableAtH dj x₀ S).T.S.m) : ¬ (tableAtH dj x₀ S).T.S.Satisfies z := by
  revert z
  unfold tableAtH
  rw [dite_eq_right_of_eq_false (eq_false hS)]
  exact x₀.refused_unsat S

variable {Reg : Type}

/-- The one-table plan: at sample `S`, the table `tableAtH … S`, for every registration. -/
noncomputable abbrev planAtH : Finset (Fin n) → Reg → List (TabSpec mPts) := fun S _ => [(tableAtH dj x₀ S).T]

/-- Every sampled proof block is in the plan's one table. -/
abbrev tabAtH : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planAtH (Reg := Reg) dj x₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

/-- The one-table plan's tables, as `Prog.flock_headline_live_on` takes them. -/
noncomputable abbrev tablesAtH : Finset (Fin n) → Reg → TabSpec mPts := fun S _ => (tableAtH dj x₀ S).T

end FlockSoundness.Discharge.HiddenExec
