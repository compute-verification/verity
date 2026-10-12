import Proofs.Flock.Soundness.Discharge.ZkHidden.Count
import Proofs.Flock.Soundness.Discharge.ZkBind.Shadow
import Proofs.Flock.Soundness.Discharge.ZkBind.BindX
import Proofs.Flock.Soundness.Discharge.ZkBind.At
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec

set_option Elab.async false

/-!
# The sites, the layout and the rows at the hidden-output `--zk` tables

lp-zkbind's `sitesZAt`, `layZAt`, `rsZAt` and `rsZAt_computes` (`ZkBind/At.lean`) at the hidden-output tables
`tablesAtZKH dj y₀`, as HiddenExec's `sitesAtH`, `layAtH`, `rsAtH` and `rsAtH_computes` are at the public path's
`tablesAtH`:

* `tableShH`: the public hidden-output table (`HiddenExec.TableH`) at the same draw: the `DrawSetupZKH`'s
  `DrawSetupH` at `zkInputs I` and its `table`, or `y₀`'s refused one. Its statement is the `--zk` table's
  (`tableAtZKH_S`, both `stmtOf` of setupHidden's statement at the same regions).
* `sitesShH`, `layShH`, `rsShH`, `rsShH_computes`: HiddenExec's `keyLayoutH` and `keyRowsH` at `tableShH`'s sites.
* `placeZH`: each drawn unit's place in its `--zk` table, `keyPlaceH`'s along `tableAtZKH_S`; `decAgreeH`: the two
  decoders read one witness alike.
* `layZH`, `rsZH`, `rsZH_computes`: the public layout and rows moved to the `--zk` tables (`LayoutZ.ofShadow`,
  `HmRowsZX.ofShadow`).
* `zk_flock_countC_execOS_atH`: #874's count curve at `planAtZKH` (`zk_flock_countC_execOS_crH`) with those.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind HiddenExec Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## A unit's place along an equality of statements -/

/-- A unit's place in `T`, as one in `T' = T`. -/
def placeCast {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {T' T : Statement} {u : Fin n} (e : T' = T)
    (p : UnitPlace P T u) : UnitPlace P T' u := by
  subst e
  exact p

theorem decode_placeCast {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {T' T : Statement} {u : Fin n}
    (e : T' = T) (p : UnitPlace P T u) (z : Witness T'.m) : (placeCast e p).decode z = p.decode (witOf e z) := by
  subst e
  rfl

/-! ## The public hidden-output table at each `--zk` table's draw -/

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}

open Classical in
/-- **The public hidden-output table at the `--zk` table's draw**: the draw's `DrawSetupZKH`'s table, or `y₀`'s
refused one. -/
noncomputable def tableShH (S : Finset (Fin n)) : TableH (zkInputs I) mPts S :=
  if h : Nonempty (DrawSetupZKH I mPts S (dj S)) then (Classical.choice h).x.table else y₀.x.refused S

/-- **The `--zk` table's statement is the public table's**, at every draw: both are `stmtOf` of the same setupHidden
statement at the same regions. -/
theorem tableAtZKH_S (S : Finset (Fin n)) : (tableAtZKH dj y₀ S).S = (tableShH dj y₀ S).T.S := by
  unfold tableAtZKH tableShH
  split_ifs <;> rfl

/-- The public plan at the same draws. -/
noncomputable abbrev planShH : Finset (Fin n) → Reg → List (TabSpec mPts) := fun S _ => [(tableShH dj y₀ S).T]

abbrev tabShH : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planShH (Reg := Reg) dj y₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

theorem planAtZKH_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S =
      (Partition.tabOf (planShH dj y₀) (tabShH dj y₀) S R u).S :=
  fun S _ _ => tableAtZKH_S dj y₀ S

variable (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOkH c = true)

/-- **The public plan's sites** (`sitesAtH`'s, at `tableShH`). -/
noncomputable def sitesShH : SitesH (planShH (Reg := Reg) dj y₀) (tabShH dj y₀) where
  acc S _ _ _ := (tableShH dj y₀ S).acc (zkTyped ht) hc (rowsFit_of_scopeH (zkTyped ht) hc hscope)
  vu S _ u h := (tableShH dj y₀ S).vu u h
  hvu S _ u h := (tableShH dj y₀ S).hvu u h
  o S _ u h := (tableShH dj y₀ S).o u h

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesShH_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShH dj y₀ ht hc hscope).acc S R u h).tags ((sitesShH dj y₀ ht hc hscope).acc S R u h).cf
        ((sitesShH dj y₀ ht hc hscope).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (parse_untyped (zkTyped ht) hc).1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable site** of the public plan: at a draw `verify --zk` sets up by `tableH_bcSite`;
at a refused draw no message satisfies the table (`refused_unsat`). -/
theorem sitesShH_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShH dj y₀) (tabShH dj y₀) S R u), Binding.Sat (planShH dj y₀) (tabShH dj y₀) S R u M →
        BcSiteH ((sitesShH dj y₀ ht hc hscope).acc S R u h) pub ((sitesShH dj y₀ ht hc hscope).o S R u h)
          ((sitesShH dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableShH dj y₀ S).acc (zkTyped ht) hc (rowsFit_of_scopeH (zkTyped ht) hc hscope)) pub ((tableShH dj y₀ S).o u h)
    ((tableShH dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZKH I mPts S (dj S))
  · have e : tableShH dj y₀ S = (Classical.choice hS).x.table := by
      unfold tableShH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSite (zkTyped ht) hc hscope dj hU hpub hpt _ h
  · have e : tableShH dj y₀ S = y₀.x.refused S := by
      unfold tableShH
      rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableShH dj y₀ S).T.S.Satisfies z := by
      rw [e]
      exact y₀.x.refused_unsat S
    exact absurd hM (hns _)

/-- **`lay` at the public plan**: `keyLayoutH` at `sitesShH`, the program's outputs the returned outputs' bits, the
registration the population's public file. -/
noncomputable abbrev layShH :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShH (Reg := Reg) dj y₀) (tabShH dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan**: `keyRowsH` at `sitesShH`, its circuit conditions from `scopeOk`, with `hsite` from
`sitesShH_bcSite`. -/
noncomputable abbrev rsShH : Binding.HmRowsX (layShH dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShH_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

theorem rsShH_computes : (rsShH dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShH_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-! ## The `--zk` places, layout and rows -/

/-- **Each drawn unit's place in its hidden-output `--zk` table**: `keyPlaceH`'s place in the public table, along
`planAtZKH_S`. -/
noncomputable def placeZH (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S u :=
  placeCast (planAtZKH_S dj y₀ S R u)
    (keyPlaceH (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeH :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
        (tabAtZKH dj y₀) (placeZH dj y₀ ht hc hscope hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZH, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables**: `layShH` moved along `planAtZKH_S`. -/
noncomputable abbrev layZH :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH (Reg := Reg) dj y₀)
      (tabAtZKH dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
        (placeZH dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planAtZKH_S dj y₀) (layShH dj y₀ ht hc hscope hU (pub := pub)) (decAgreeH dj y₀ ht hc hscope hU)

/-- **`rs` at the hidden-output `--zk` tables**: `rsShH` moved along `planAtZKH_S`. -/
noncomputable abbrev rsZH : HmRowsZX (layZH dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planAtZKH_S dj y₀) (decAgreeH dj y₀ ht hc hscope hU) (rsShH dj y₀ ht hc hscope hU hpub hpt)

/-- **`hHm` at the hidden-output `--zk` tables' rows** (`rsShH_computes`, moved along): in a satisfying witness of a
drawn unit's `--zk` table, each row's `b ‖ c` at the unit's site, input or output, is the commit string of the row's
bytes under the salt at its `hm96` slot. -/
theorem rsZH_computes : (rsZH dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planAtZKH_S dj y₀) (decAgreeH dj y₀ ht hc hscope hU) _
    (rsShH_computes dj y₀ ht hc hscope hU hpub hpt)

/-- **The compiled `--zk` count curve at the hidden-output `--zk` tables, its places and binding built**:
`zk_flock_countC_execOS_crH` with `place := placeZH` and `vb := rsZH.bindingZ rsZH_computes`. Past the count curve's
own hypotheses it takes the session's inputs: untyped tags (`ht`), one circuit file `HmOut.parse` accepts as `c` under
the `--zk` tags (`hc`), the circuit in `rs`'s scope (`hscope`), and a public file `HmOut.loadPublic` loads without
shared-row tables (`hpub`, `hpt`). -/
theorem zk_flock_countC_execOS_atH (Hs : Hashes 𝔽 Dig (List UInt8))
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKH dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith Hs
      (planAtZKH dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat execArith Hs (planAtZKH dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
        (XplurZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
          (tabAtZKH dj y₀)
          ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_crH execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) dj y₀
    execArith_correct σs ks hks St hS Kd pop Lb src hA3 (placeZH dj y₀ ht hc hscope hU)
    ((rsZH dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZH_computes dj y₀ ht hc hscope hU hpub hpt))
    hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ZkHidden
