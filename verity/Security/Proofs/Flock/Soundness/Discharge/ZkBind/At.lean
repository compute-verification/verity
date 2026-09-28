import Proofs.Flock.Soundness.Discharge.ZkBind.Shadow
import Proofs.Flock.Soundness.Discharge.ZkBind.Headline
import Proofs.Flock.Soundness.Discharge.ZkLink.TablesAt
import Proofs.Flock.Soundness.Discharge.Integrate.Headline

/-!
# The sites, the layout and the rows at the `--zk` tables

`zk_flock_countC_execOS_keyProg` takes the sites `si`, the parse `hparse`, the layout `lay`, the rows `rs` and
`HmRowComputes` `hHm`. Here they are built at item 3's `--zk` tables (`ZkLink.tablesAtZK dj y₀`: at each draw `S` the
table of a `DrawSetupZK` at the challenger's draw `dj S`, a refused table otherwise), as `Integrate.sitesAt`, `layAt`,
`rsAt` and `rsAt_computes` are at the public path's `tablesAt`:

* `tableSh`: the public table (`Integrate.Table`) at the same draw: the `DrawSetupZK`'s `DrawSetup` at `zkInputs I` and
  its `table`, or `y₀`'s refused one. Its statement is the `--zk` table's (`tableAtZK_S`: `Layout.model` of setupH's
  statement is `stmtOf` of it, `Integrate.model_eq_stmtOf`).
* `sitesZAt`: the `--zk` sites, each drawn unit at its rank in the draw's proved units (`sitesZAt_at`), and
  `sitesZAt_parse` from the session's one circuit file.
* `layZAt`, `rsZAt`: e2e-layout's `keyedLayout` and `keyedRowsOuts` at `tableSh`'s sites, moved to the `--zk` tables
  (`LayoutZ.ofShadow`, `HmRowsZ.ofShadow`), the decoders agreeing (`decAgreeAt`); `rsZAt_computes` is e2e-hm's
  `keyedRowsOuts_computes` moved along.

None of them takes a draw premise: like `tablesAtZK` they are defined at every draw. `sitesZAt_at` reads the draw's own
`DrawSetupZK`, which item 3 gives from `AcceptsZK` and `DrawOkZK I dj mPts S` at the session's own draw `S`.
-/

namespace FlockSoundness.Discharge.ZkBind

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec Integrate Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- An acceptance whose statement is `T`, as one whose statement is `T' = T`. -/
def recastAcc {T T' : Model.Statement} (a : Accepted T) (e : T = T') : Accepted T' := { a with hT := a.hT.trans e }

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}

theorem zkTyped (ht : I.tags.typed = false) : (zkInputs I).tags.typed = false := (zkTags_typed I).trans ht

include ht dj y₀

open Classical in
/-- **The public table at the `--zk` table's draw**: the draw's `DrawSetupZK`'s table, or `y₀`'s refused one. -/
noncomputable def tableSh (S : Finset (Fin n)) : Integrate.Table mPts S :=
  if h : Nonempty (DrawSetupZK I mPts S (dj S)) then (Classical.choice h).x.table (zkTyped ht)
  else y₀.x.refused (zkTyped ht) S

/-- **The `--zk` table's statement is the accepted model statement** (`stmtOf st` is `Layout.model` of setupH's
statement at its regions, `Integrate.model_eq_stmtOf`), at every draw. -/
theorem tableAtZK_S (S : Finset (Fin n)) : (tableAtZK dj y₀ S).S = (tableSh ht dj y₀ S).T.S := by
  unfold tableAtZK tableSh
  split_ifs with h
  · exact (table_stmtOf (zkTyped ht) _).symm
  · exact (model_eq_stmtOf (zkTyped ht) y₀.x.h y₀.x.wf.1 _).symm

/-- The one-table `--zk` plan: `tablesAtZK`. -/
noncomputable abbrev planZAt : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S R => [tablesAtZK dj y₀ S R]

/-- Every drawn unit is in the plan's one table. -/
abbrev tabZAt : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZAt (Reg := Reg) dj y₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

/-- The public plan at the same draws. -/
noncomputable abbrev planSh : Finset (Fin n) → Reg → List (TabSpec mPts) := fun S _ => [(tableSh ht dj y₀ S).T]

abbrev tabSh : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planSh (Reg := Reg) ht dj y₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

theorem planZAt_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planZAt dj y₀) (tabZAt dj y₀) S R u).S = (Partition.tabOf (planSh ht dj y₀) (tabSh ht dj y₀) S R u).S :=
  fun S _ _ => tableAtZK_S ht dj y₀ S

/-- The public plan's sites (`Integrate.sitesAt`'s, at `tableSh`). -/
noncomputable def sitesSh : Placed.Sites (planSh (Reg := Reg) ht dj y₀) (tabSh ht dj y₀) where
  acc S _ _ _ := (tableSh ht dj y₀ S).acc
  vu S _ u h := (tableSh ht dj y₀ S).vu u h
  hvu S _ u h := (tableSh ht dj y₀ S).hvu u h
  o S _ u h := (tableSh ht dj y₀ S).o u h

/-- **The `--zk` sites**: each drawn unit's acceptance, template instance and block are its draw's public table's. -/
noncomputable def sitesZAt : SitesZ (planZAt (Reg := Reg) dj y₀) (tabZAt dj y₀) where
  acc S R u _ := recastAcc (tableSh ht dj y₀ S).acc (planZAt_S ht dj y₀ S R u).symm
  vu S _ u h := (tableSh ht dj y₀ S).vu u h
  hvu S _ u h := (tableSh ht dj y₀ S).hvu u h
  o S R u h := Fin.cast (congrArg (fun T : Model.Statement => 2 ^ T.nbl) (planZAt_S ht dj y₀ S R u).symm)
    ((tableSh ht dj y₀ S).o u h)

variable {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)

include hc in
/-- **One circuit per session**: every `--zk` site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesZAt_parse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse ((sitesZAt ht dj y₀).acc S R u h).tags ((sitesZAt ht dj y₀).acc S R u h).cf
      ((sitesZAt ht dj y₀).acc S R u h).tables = .ok c := by
  intro S R u h
  show HmRow.parse (tableSh ht dj y₀ S).acc.tags (tableSh ht dj y₀ S).acc.cf (tableSh ht dj y₀ S).acc.tables = .ok c
  unfold tableSh
  by_cases hS : Nonempty (DrawSetupZK I mPts S (dj S))
  · rw [dite_eq_left_of_eq_true (eq_true hS)]; exact hc
  · rw [dite_eq_right_of_eq_false (eq_false hS)]; exact hc

include hc in
theorem sitesSh_parse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse ((sitesSh ht dj y₀).acc S R u h).tags ((sitesSh ht dj y₀).acc S R u h).cf
      ((sitesSh ht dj y₀).acc S R u h).tables = .ok c := by
  intro S R u h
  show HmRow.parse (tableSh ht dj y₀ S).acc.tags (tableSh ht dj y₀ S).acc.cf (tableSh ht dj y₀ S).acc.tables = .ok c
  unfold tableSh
  by_cases hS : Nonempty (DrawSetupZK I mPts S (dj S))
  · rw [dite_eq_left_of_eq_true (eq_true hS)]; exact hc
  · rw [dite_eq_right_of_eq_false (eq_false hS)]; exact hc

/-- **Each drawn unit's site at its draw's `DrawSetupZK`**: at a draw `S` that `verify --zk` sets up (`y`), each drawn
unit's acceptance is setupH's statement at `dj S`, its block and template instance its rank `r` among the draw's proved
units: block `r / c.g`, template instance `r % c.g`. -/
theorem sitesZAt_at {S : Finset (Fin n)} (y : DrawSetupZK I mPts S (dj S)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    ((sitesZAt (Reg := Reg) ht dj y₀).acc S R u h).st = y.x.st ∧
      ((sitesZAt (Reg := Reg) ht dj y₀).acc S R u h).draw = some (dj S) ∧
      (sitesZAt (Reg := Reg) ht dj y₀).vu S R u h = y.x.rank u % y.x.st.c.g ∧
      ((sitesZAt (Reg := Reg) ht dj y₀).o S R u h).val = y.x.rank u / y.x.st.c.g := by
  have hne : Nonempty (DrawSetupZK I mPts S (dj S)) := ⟨y⟩
  have hT : tableSh ht dj y₀ S = (Classical.choice hne).x.table (zkTyped ht) := by
    unfold tableSh; rw [dite_eq_left_of_eq_true (eq_true hne)]
  have hst : (Classical.choice hne).x.st = y.x.st := Except.ok.inj ((Classical.choice hne).x.h.symm.trans y.x.h)
  have hD : (Classical.choice hne).x.D = y.x.D := Except.ok.inj ((Classical.choice hne).x.hD.symm.trans y.x.hD)
  have hr : (Classical.choice hne).x.rank u = y.x.rank u := by simp only [DrawSetup.rank, hD]
  refine ⟨?_, ?_, ?_, ?_⟩
  · show (tableSh ht dj y₀ S).acc.st = _
    rw [hT]; exact hst
  · show (tableSh ht dj y₀ S).acc.draw = _
    rw [hT]; rfl
  · show (tableSh ht dj y₀ S).vu u h = _
    rw [hT]; show (Classical.choice hne).x.rank u % (Classical.choice hne).x.st.c.g = _
    rw [hr, hst]
  · show ((tableSh ht dj y₀ S).o u h).val = _
    rw [hT]; show (Classical.choice hne).x.rank u / (Classical.choice hne).x.st.c.g = _
    rw [hr, hst]

variable (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none) (hscope : Layout.scopeOk c = true)

/-- **The decoders agree**: the `--zk` sites' places and the public sites' places read one witness alike. -/
theorem decAgreeAt : DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt_S (Reg := Reg) ht dj y₀)
    (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
      ((sitesZAt ht dj y₀).placeKey hU (sitesZAt_parse ht dj y₀ hc) (Layout.outsOf c hU n)))
    (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planSh ht dj y₀) (tabSh ht dj y₀)
      (Layout.keyPlace (sitesSh_parse ht dj y₀ hc) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM hM' g hg
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, UnitPlace.decode]
  congr 1
  funext x
  congr 1
  have hM2 : witnessOf execArith (Partition.tabOf (planSh ht dj y₀) (tabSh ht dj y₀) S R u).S.m
      (Partition.tabOf (planSh ht dj y₀) (tabSh ht dj y₀) S R u).sch.k0
      (Partition.tabOf (planSh ht dj y₀) (tabSh ht dj y₀) S R u).sch.level₀.logCols M =
      witOf (planZAt_S ht dj y₀ S R u) (witZ execArith (planZAt dj y₀) (tabZAt dj y₀) S R u M') := hM
  rw [hM2]
  exact block_witOf _ _ _ _ _ _ rfl rfl

/-- **`lay` at the `--zk` tables**: e2e-layout's `keyedLayout` at the public sites, moved to the `--zk` tables. -/
noncomputable abbrev layZAt :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt (Reg := Reg) dj y₀) (tabZAt dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
        ((sitesZAt ht dj y₀).placeKey hU (sitesZAt_parse ht dj y₀ hc) (Layout.outsOf c hU n))) :=
  LayoutZ.ofShadow (planZAt_S ht dj y₀)
    (Layout.keyedLayout (sitesSh_parse ht dj y₀ hc) (Layout.outsOf c hU n) Hm.saltCol fun _ => pub)
    (decAgreeAt ht dj y₀ hc hU)

include hc hU hpub hpt hscope in
theorem sitesSh_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planSh ht dj y₀) (tabSh ht dj y₀) S R u),
      Binding.Sat (planSh ht dj y₀) (tabSh ht dj y₀) S R u M →
        Layout.BcSite ((sitesSh ht dj y₀).acc S R u h) pub ((sitesSh ht dj y₀).o S R u h)
          ((sitesSh ht dj y₀).vu S R u h) u.val := by
  intro S R u h M hM
  show Layout.BcSite (tableSh ht dj y₀ S).acc pub ((tableSh ht dj y₀ S).o u h) ((tableSh ht dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZK I mPts S (dj S))
  · have e : tableSh ht dj y₀ S = (Classical.choice hS).x.table (zkTyped ht) := by
      unfold tableSh; rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact table_bcSite (zkTyped ht) dj hc (Layout.scopeOk_spec hU hscope).2.1 hpub hpt _ h
  · have e : tableSh ht dj y₀ S = y₀.x.refused (zkTyped ht) S := by
      unfold tableSh; rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableSh ht dj y₀ S).T.S.Satisfies z := by
      rw [e]; exact y₀.x.refused_unsat (zkTyped ht) S
    exact absurd hM (hns _)

/-- **`rs` at the `--zk` tables**: e2e-layout's `keyedRowsOuts` at the public sites, its circuit conditions from
`scopeOk`, moved to the `--zk` tables. -/
noncomputable abbrev rsZAt : HmRowsZ (layZAt (Reg := Reg) ht dj y₀ hc hU (pub := pub)) :=
  HmRowsZ.ofShadow (planZAt_S ht dj y₀) (decAgreeAt ht dj y₀ hc hU)
    (Layout.keyedRowsOuts (sitesSh_parse ht dj y₀ hc) Hm.saltCol hc (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1
      (sitesSh_bcSite ht dj y₀ hc hU hpub hpt hscope))

/-- **`hHm` at the `--zk` tables' rows** (e2e-hm's `keyedRowsOuts_computes`, moved along): in a satisfying witness of a
drawn unit's `--zk` table, each row port's `b ‖ c` at the unit's site is the commit string of the row's bytes under
the salt at its `hm96` slot. -/
theorem rsZAt_computes : (rsZAt (Reg := Reg) ht dj y₀ hc hU hpub hpt hscope).Computes :=
  HmRowsZ.ofShadow_computes (planZAt_S ht dj y₀) (decAgreeAt ht dj y₀ hc hU) _
    (Hm.keyedRowsOuts_computes (sitesSh_parse ht dj y₀ hc) hc (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope)
      (sitesSh_bcSite ht dj y₀ hc hU hpub hpt hscope))

/-- **The compiled `--zk` count curve at the `--zk` tables, its places and binding built.**
`zk_flock_countC_execOS_keyProg` at item 3's tables (`tablesAtZK dj y₀`), with `si := sitesZAt`, `hparse :=
sitesZAt_parse`, the program's outputs the returned outputs' bits (`Layout.outsOf`), `lay := layZAt`, `rs := rsZAt`
and `hHm := rsZAt_computes`. Past the count curve's own hypotheses it takes the session's inputs: untyped tags (`ht`),
one circuit file parsing to `c` under the `--zk` tags (`hc`), a public file without shared-row tables (`hpub`,
`hpt`), and the circuit in `rs`'s scope (`hscope`). -/
theorem zk_flock_countC_execOS_at (Hs : Hashes 𝔽 Dig (List UInt8))
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planZAt dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith Hs
      (planZAt dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat execArith Hs (planZAt dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planZAt dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
        (XplurZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planZAt dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_keyProg execArith Hs (planZAt dj y₀) (tabZAt dj y₀) execArith_correct hU (sitesZAt ht dj y₀) (sitesZAt_parse ht dj y₀ hc)
    (Layout.outsOf c hU n) (layZAt ht dj y₀ hc hU) (rsZAt ht dj y₀ hc hU hpub hpt hscope)
    (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope) σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ZkBind
