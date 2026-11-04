import Proofs.Flock.Soundness.Discharge.Exec
import Proofs.Flock.Soundness.Discharge.Layout
import Proofs.Flock.Soundness.Discharge.Placed

/-!
# The tables and sites of a one-table session

`Prog.flock_headline_live` takes a table `T S R` at every draw `S` and each drawn unit's site in it (`Placed.Sites`).
This file builds both from the session's inputs `I : Inputs` (the verify command's flags: one circuit file, one public
file, one table), at the draw the challenger writes for `S` (`dj S`, the draw's JSON):

* `DrawSetup I mPts S d`: `Stmt.setupH` on `I` accepts the draw `d`, which decodes to a draw `D` proving every unit of
  `S`; the statement's blocks hold `D`'s proved units, its `fast100` schedule has two levels or more, and its link
  points are `mPts`.
* `DrawSetup.table`: its table, the statement `(setupH_layout ht h).model (regionsOf st)` (what `Placed.Accepted` reads,
  so `Accepted.hT` is `rfl`), and each unit `u`'s site: its rank `r` among `D.proved`, block `r / c.g` and template
  instance `r % c.g`. Distinct units have distinct ranks and so distinct sites (`table_site_inj`).
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

/-- **A draw flock-verify sets up**: on the session's inputs `I`, `Stmt.setupH` accepts the draw's JSON `d`, which
decodes to a draw `D` whose proved units include `S`'s; the statement's `c.g · 2^nbl` template instance slots hold `D`'s
proved units, its `fast100` schedule has at least two levels (`hsch`, `hm`, as `TabSpec` asks), and its link points are
`mPts`. -/
structure DrawSetup (S : Finset (Fin n)) (d : Lean.Json) where
  st : Stmt
  h : Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st
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

/-- **A draw's table with its sites**: the table, an acceptance whose statement is the table's, and each drawn unit's
template instance and block there. -/
structure Table (S : Finset (Fin n)) where
  T : TabSpec mPts
  acc : Placed.Accepted T.S
  vu : ∀ u ∈ S, ℕ
  hvu : ∀ u (h : u ∈ S), vu u h < acc.st.c.g
  o : ∀ u ∈ S, Fin (2 ^ T.S.nbl)

variable {I mPts} (ht : I.tags.typed = false)

namespace DrawSetup

variable {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetup I mPts S d)

theorem wf : StmtWF x.st ∧ RegionsWF x.st := setupH_wf x.h

/-- The table at statement regions `regs`: setupH's level-0 matrices and pin (`Layout.model`), its schedule, and the
link layout at `PT_LOCAL`. -/
noncomputable def specAt (regs : List (Model.Region x.st.c.kLog (x.st.m - x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) : TabSpec mPts where
  S := (setupH_layout ht x.h).model regs
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

theorem rank_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S) (e : x.rank u = x.rank u') : u = u' := by
  have a := x.proved_rank h
  rw [e, x.proved_rank h'] at a
  exact Fin.ext (Option.some.inj a).symm

theorem rank_div_lt {u : Fin n} (h : u ∈ S) : x.rank u / x.st.c.g < 2 ^ (x.st.m - x.st.c.kLog) := by
  obtain ⟨-, hr⟩ := x.wf
  rw [Nat.div_lt_iff_lt_mul x.hg, hr.hm, Nat.add_sub_cancel_left, Nat.mul_comm]
  exact lt_of_lt_of_le (x.rank_lt h) x.hblk

/-- **The draw's table**: setupH's statement at its regions, and each drawn unit at its rank `r`: block `r / c.g`,
template instance `r % c.g`. -/
noncomputable def table : Table mPts S where
  T := x.specAt ht (regionsOf x.st) (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1
  acc := ⟨I.tags, I.circuitFile, I.publicFile, I.coins, I.tables, some d, I.partition, I.program, x.st, ht, x.h,
    regionsOf x.st, rfl⟩
  vu u _ := x.rank u % x.st.c.g
  hvu _ _ := Nat.mod_lt _ x.hg
  o u h := ⟨x.rank u / x.st.c.g, x.rank_div_lt h⟩

/-- **Distinct drawn units have distinct sites** in the draw's table. -/
theorem table_site_inj {u u' : Fin n} (h : u ∈ S) (h' : u' ∈ S)
    (e : ((x.table ht).o u h).val = ((x.table ht).o u' h').val ∧ (x.table ht).vu u h = (x.table ht).vu u' h') :
    u = u' := by
  refine x.rank_inj h h' ?_
  obtain ⟨e1, e2⟩ := e
  have e1' : x.rank u / x.st.c.g = x.rank u' / x.st.c.g := e1
  have e2' : x.rank u % x.st.c.g = x.rank u' % x.st.c.g := e2
  rw [← Nat.div_add_mod' (x.rank u) x.st.c.g, ← Nat.div_add_mod' (x.rank u') x.st.c.g, e1', e2']

/-- Two regions with no free bits, both at in-block address 0, one pinning 0 and one 1. -/
def contra (kl nbl : ℕ) : List (Model.Region kl nbl) :=
  [⟨[], fun _ => false, fun _ _ => 0⟩, ⟨[], fun _ => false, fun _ _ => 1⟩]

theorem contra_wf {kl nbl : ℕ} : ∀ R ∈ contra kl nbl, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL := by
  intro R hR
  simp only [contra, List.mem_cons, List.not_mem_nil, or_false] at hR
  rcases hR with rfl | rfl <;> exact ⟨List.nodup_nil, Nat.zero_le _⟩

/-- No witness satisfies a statement with `contra`'s regions. -/
theorem contra_unsat {S : Model.Statement} (hS : contra S.kLog S.nbl ⊆ S.regions) (z : Witness S.m) :
    ¬ S.Satisfies z := by
  rintro ⟨-, -, hR⟩
  let r0 : Model.Region S.kLog S.nbl := ⟨[], fun _ => false, fun _ _ => 0⟩
  let r1 : Model.Region S.kLog S.nbl := ⟨[], fun _ => false, fun _ _ => 1⟩
  have m0 : r0 ∈ S.regions := hS (by simp [contra, r0])
  have m1 : r1 ∈ S.regions := hS (by simp [contra, r1])
  have h0 : r0.inBlock 0 = 0 := by simp [r0, Model.Region.inBlock]
  have h1 : r1.inBlock 0 = 0 := by simp [r1, Model.Region.inBlock]
  have hlt : (0 : ℕ) < 2 ^ S.kLog := Nat.two_pow_pos _
  have a := hR r0 m0 ⟨0, Nat.two_pow_pos _⟩ 0 (by simp [r0]) (by rw [h0]; exact hlt)
  have b := hR r1 m1 ⟨0, Nat.two_pow_pos _⟩ 0 (by simp [r1]) (by rw [h1]; exact hlt)
  have e : S.block z ⟨0, Nat.two_pow_pos _⟩ ⟨r0.inBlock 0, by rw [h0]; exact hlt⟩ =
      S.block z ⟨0, Nat.two_pow_pos _⟩ ⟨r1.inBlock 0, by rw [h1]; exact hlt⟩ := by
    congr 1
  rw [a, b] at e
  exact zero_ne_one (α := ZMod 2) e

/-- The regions of a refused draw's table: the reference setup's, and `contra`'s. -/
theorem refused_wf : ∀ R ∈ regionsOf x.st ++ contra x.st.c.kLog (x.st.m - x.st.c.kLog),
    R.free.Nodup ∧ R.free.length ≤ PT_LOCAL := by
  intro R hR
  rcases List.mem_append.1 hR with hR | hR
  · exact (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1 R hR
  · exact contra_wf R hR

/-- **The table of a refused draw `S'`**: a reference setup `x`'s statement with `contra`'s regions, which no witness
satisfies (`refused_unsat`), every unit at block 0, template instance 0. -/
noncomputable def refused (S' : Finset (Fin n)) : Table mPts S' where
  T := x.specAt ht (regionsOf x.st ++ contra x.st.c.kLog (x.st.m - x.st.c.kLog)) x.refused_wf
  acc := ⟨I.tags, I.circuitFile, I.publicFile, I.coins, I.tables, some d, I.partition, I.program, x.st, ht, x.h,
    regionsOf x.st ++ contra x.st.c.kLog (x.st.m - x.st.c.kLog), rfl⟩
  vu _ _ := 0
  hvu _ _ := x.hg
  o _ _ := ⟨0, Nat.two_pow_pos _⟩

theorem refused_unsat (S' : Finset (Fin n)) (z : Witness (x.refused ht S').T.S.m) :
    ¬ (x.refused ht S').T.S.Satisfies z :=
  contra_unsat (S := (x.refused ht S').T.S) (fun _ hR => List.mem_append_right _ hR) z

/-- **The contra-only table of a refused draw `S'`**: `refused` with `contra`'s two regions alone, so its region count
is 2 whatever the reference setup's; no witness satisfies it (`refusedC_unsat`). -/
noncomputable def refusedC (S' : Finset (Fin n)) : Table mPts S' where
  T := x.specAt ht (contra x.st.c.kLog (x.st.m - x.st.c.kLog)) contra_wf
  acc := ⟨I.tags, I.circuitFile, I.publicFile, I.coins, I.tables, some d, I.partition, I.program, x.st, ht, x.h,
    contra x.st.c.kLog (x.st.m - x.st.c.kLog), rfl⟩
  vu _ _ := 0
  hvu _ _ := x.hg
  o _ _ := ⟨0, Nat.two_pow_pos _⟩

theorem refusedC_unsat (S' : Finset (Fin n)) (z : Witness (x.refusedC ht S').T.S.m) :
    ¬ (x.refusedC ht S').T.S.Satisfies z :=
  contra_unsat (S := (x.refusedC ht S').T.S) (fun _ hR => hR) z

end DrawSetup

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀))

open Classical in
/-- **The table at draw `S`**: the draw's table (`DrawSetup.table`) when flock-verify sets up the draw's JSON `dj S`,
and otherwise the reference setup `x₀`'s refused table. -/
noncomputable def tableAt (S : Finset (Fin n)) : Table mPts S :=
  if h : Nonempty (DrawSetup I mPts S (dj S)) then (Classical.choice h).table ht else x₀.refused ht S

/-- **The table is setupH's statement wherever setupH accepts**: at a draw with a `DrawSetup` at `dj S`, the table's
statement is `Layout.model` of setupH's statement at its regions, and its acceptance's draw is `dj S`. -/
theorem tableAt_stmt {S : Finset (Fin n)} (x : DrawSetup I mPts S (dj S)) :
    ∃ y : DrawSetup I mPts S (dj S), y.st = x.st ∧ tableAt ht dj x₀ S = y.table ht := by
  refine ⟨Classical.choice ⟨x⟩, ?_, by unfold tableAt; rw [dite_eq_left_of_eq_true (eq_true ⟨x⟩)]⟩
  have e := (Classical.choice (⟨x⟩ : Nonempty (DrawSetup I mPts S (dj S)))).h.symm.trans x.h
  exact Except.ok.inj e

theorem tableAt_cases (S : Finset (Fin n)) :
    (∃ x : DrawSetup I mPts S (dj S), tableAt ht dj x₀ S = x.table ht) ∨ tableAt ht dj x₀ S = x₀.refused ht S := by
  unfold tableAt
  split
  · exact .inl ⟨_, rfl⟩
  · exact .inr rfl

/-- **A refused draw's table is unsatisfiable**: at a draw with no `DrawSetup`, no witness satisfies the table. -/
theorem tableAt_unsat {S : Finset (Fin n)} (hS : ¬ Nonempty (DrawSetup I mPts S (dj S)))
    (z : Witness (tableAt ht dj x₀ S).T.S.m) : ¬ (tableAt ht dj x₀ S).T.S.Satisfies z := by
  revert z
  unfold tableAt
  rw [dite_eq_right_of_eq_false (eq_false hS)]
  exact x₀.refused_unsat ht S

theorem tableAt_inputs (S : Finset (Fin n)) :
    (tableAt ht dj x₀ S).acc.tags = I.tags ∧ (tableAt ht dj x₀ S).acc.cf = I.circuitFile ∧
      (tableAt ht dj x₀ S).acc.tables = I.tables ∧ (tableAt ht dj x₀ S).acc.pf = I.publicFile := by
  rcases tableAt_cases ht dj x₀ S with ⟨x, e⟩ | e <;> rw [e] <;> exact ⟨rfl, rfl, rfl, rfl⟩

variable {Reg : Type}

/-- The one-table plan: at draw `S`, the table `tableAt … S`, for every registration. -/
noncomputable abbrev planAt : Finset (Fin n) → Reg → List (TabSpec mPts) := fun S _ => [(tableAt ht dj x₀ S).T]

/-- Every drawn unit is in the plan's one table. -/
abbrev tabAt : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planAt (Reg := Reg) ht dj x₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

/-- **The sites of the one-table plan**: each drawn unit's acceptance, template instance and block are its draw's
table's. -/
noncomputable def sitesAt : Placed.Sites (planAt (Reg := Reg) ht dj x₀) (tabAt ht dj x₀) where
  acc S _ _ _ := (tableAt ht dj x₀ S).acc
  vu S _ u h := (tableAt ht dj x₀ S).vu u h
  hvu S _ u h := (tableAt ht dj x₀ S).hvu u h
  o S _ u h := (tableAt ht dj x₀ S).o u h

variable {c : Flock.Circuit}

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesAt_parse (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesAt ht dj x₀).acc S R u h).tags ((sitesAt ht dj x₀).acc S R u h).cf
        ((sitesAt ht dj x₀).acc S R u h).tables = .ok c := by
  intro S R u h
  obtain ⟨e1, e2, e3, -⟩ := tableAt_inputs ht dj x₀ S
  show HmRow.parse (tableAt ht dj x₀ S).acc.tags (tableAt ht dj x₀ S).acc.cf (tableAt ht dj x₀ S).acc.tables = .ok c
  rw [e1, e2, e3]
  exact hc

/-- **Distinct drawn units have distinct sites** wherever flock-verify sets up the draw (the count judges each site
once). -/
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

/-- **The site fact at the draw's table**, with or without shared rows: at a draw flock-verify sets up, each drawn
unit's site is `BcSite`: the acceptance's draw is `dj S`, which decodes to `D`, and the unit's block and template
instance index its rank in `D.proved`. -/
theorem table_bcSiteS (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (hpop : HmRow.loadPublic I.tags c I.publicFile = .ok pop)
    {S : Finset (Fin n)} (x : DrawSetup I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    Layout.BcSite (x.table ht).acc pop ((x.table ht).o u h) ((x.table ht).vu u h) u.val := by
  have hcc : (x.table ht).acc.st.c = c := (x.table ht).acc.c_eq hc
  have hcx : x.st.c = c := hcc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmRow.loadPublic I.tags x.st.c I.publicFile = .ok pop
    rw [hcx]; exact hpop
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2 hc ho (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

set_option linter.unusedVariables false in
/-- **The site fact at the draw's table** (`table_bcSiteS`). -/
theorem table_bcSite (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (hpop : HmRow.loadPublic I.tags c I.publicFile = .ok pop) (hpt : pop.tables = none)
    {S : Finset (Fin n)} (x : DrawSetup I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    Layout.BcSite (x.table ht).acc pop ((x.table ht).o u h) ((x.table ht).vu u h) u.val :=
  table_bcSiteS ht dj hc ho hpop x h

/-- **`BcSite` at every satisfiable site** of the one-table plan: at a draw flock-verify sets up by `table_bcSite`; at a
refused draw no message satisfies the table (`refused_unsat`). -/
theorem sitesAt_bcSite (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (hpop : HmRow.loadPublic I.tags c I.publicFile = .ok pop) (hpt : pop.tables = none) :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) (M : Binding.Msg (planAt ht dj x₀) (tabAt ht dj x₀) S R u),
      Binding.Sat (planAt ht dj x₀) (tabAt ht dj x₀) S R u M →
        Layout.BcSite ((sitesAt ht dj x₀).acc S R u h) pop ((sitesAt ht dj x₀).o S R u h)
          ((sitesAt ht dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show Layout.BcSite (tableAt ht dj x₀ S).acc pop ((tableAt ht dj x₀ S).o u h) ((tableAt ht dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetup I mPts S (dj S))
  · have e : tableAt ht dj x₀ S = (Classical.choice hS).table ht := by unfold tableAt; rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact table_bcSite ht dj hc ho hpop hpt _ h
  · exact absurd hM (tableAt_unsat ht dj x₀ hS _)

end FlockSoundness.Discharge.Integrate
