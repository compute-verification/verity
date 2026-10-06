import Proofs.Flock.Soundness.Discharge.ZkReg.BindH
import Proofs.Flock.Soundness.Discharge.ZkRegV2.KeyH

set_option Elab.async false

/-!
# The rows at the hidden-output `--zk` tables, for `sha512/row/v2` rows

`ZkHidden/SoundBind.lean` and `ZkReg/BindH.lean` at a circuit `ZkRegV2.scopeOkHV2` admits: `sitesShHV2` is `sitesShH`
with each drawn table's acceptance `TableH.accV2`; `layShHV2`, `rsShHSV2` and `rsShHSV2_computes` are `layShH`,
`rsShHS` and `rsShHS_computes` with `keyLayoutHV2`, `keyRowsHV2` and `keyRowsH_computesV2` (the rows' `sha512/row/v2`
bytes); `placeZHV2`, `decAgreeHV2`, `layZHV2`, `rsZHSV2` and `rsZHSV2_computes` move them to the `--zk` tables; and
`zk_flock_countC_execOS_atHSV2` is the compiled `--zk` count curve at `rsZHSV2`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : ZkRegV2.scopeOkHV2 c = true)

/-- **The public plan's sites, at `sha512/row/v2` rows**: `sitesShH` with each drawn table's acceptance
`TableH.accV2`. -/
noncomputable def sitesShHV2 : SitesH (planShH (Reg := Reg) dj y₀) (tabShH dj y₀) where
  acc S _ _ _ := (tableShH dj y₀ S).accV2 (zkTyped ht) hc (ZkRegV2.scopeOkHV2_ports hscope)
  vu S _ u h := (tableShH dj y₀ S).vu u h
  hvu S _ u h := (tableShH dj y₀ S).hvu u h
  o S _ u h := (tableShH dj y₀ S).o u h

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesShH_parseV2 :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShHV2 dj y₀ ht hc hscope).acc S R u h).tags ((sitesShHV2 dj y₀ ht hc hscope).acc S R u h).cf
        ((sitesShHV2 dj y₀ ht hc hscope).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (parse_untyped (zkTyped ht) hc).1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hU hpub in
/-- **`BcSiteH` at every satisfiable site** of the public plan, at `sha512/row/v2` rows (`tableH_bcSiteSV2`). -/
theorem sitesShH_bcSiteSV2 :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShH dj y₀) (tabShH dj y₀) S R u), Binding.Sat (planShH dj y₀) (tabShH dj y₀) S R u M →
        BcSiteH ((sitesShHV2 dj y₀ ht hc hscope).acc S R u h) pub ((sitesShHV2 dj y₀ ht hc hscope).o S R u h)
          ((sitesShHV2 dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableShH dj y₀ S).accV2 (zkTyped ht) hc (ZkRegV2.scopeOkHV2_ports hscope)) pub
    ((tableShH dj y₀ S).o u h) ((tableShH dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZKH I mPts S (dj S))
  · have e : tableShH dj y₀ S = (Classical.choice hS).x.table := by
      unfold tableShH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteSV2 (zkTyped ht) hc (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_spec hU hscope).2.1
      dj hpub _ h
  · have e : tableShH dj y₀ S = y₀.x.refused S := by
      unfold tableShH
      rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableShH dj y₀ S).T.S.Satisfies z := by
      rw [e]
      exact y₀.x.refused_unsat S
    exact absurd hM (hns _)

/-- **`lay` at the public plan, at `sha512/row/v2` rows**: `keyLayoutHV2` at `sitesShHV2`. -/
noncomputable abbrev layShHV2 :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShH (Reg := Reg) dj y₀) (tabShH dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShH_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n))) :=
  keyLayoutHV2 (sitesShH_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_rows hscope)
    (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan, at `sha512/row/v2` rows**: `keyRowsHV2` at `sitesShHV2`, with `hsite` from
`sitesShH_bcSiteSV2`. -/
noncomputable abbrev rsShHSV2 : Binding.HmRows (layShHV2 dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsHV2 (sitesShH_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_rows hscope)
    (Layout.outsOut_outsOf c hU n) (sitesShH_bcSiteSV2 dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (ZkRegV2.scopeOkHV2_spec hU hscope).1
      (ZkRegV2.scopeOkHV2_spec hU hscope).2.2)

theorem rsShHSV2_computes : (rsShHSV2 dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  keyRowsH_computesV2 (sitesShH_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_rows hscope)
    (Layout.outsOut_outsOf c hU n) (sitesShH_bcSiteSV2 dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (ZkRegV2.scopeOkHV2_spec hU hscope).1
      (ZkRegV2.scopeOkHV2_spec hU hscope).2.2)

/-! ## The `--zk` places, layout and rows -/

/-- **Each drawn unit's place in its hidden-output `--zk` table, at `sha512/row/v2` rows**: `keyPlaceH`'s place at
`sitesShHV2`, along `planAtZKH_S`. -/
noncomputable def placeZHV2 (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S u :=
  placeCast (planAtZKH_S dj y₀ S R u)
    (keyPlaceH (sitesShH_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeHV2 :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
        (tabAtZKH dj y₀) (placeZHV2 dj y₀ ht hc hscope hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShH_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHV2, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables, at `sha512/row/v2` rows**: `layShHV2` moved along `planAtZKH_S`. -/
noncomputable abbrev layZHV2 :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH (Reg := Reg) dj y₀)
      (tabAtZKH dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
        (placeZHV2 dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planAtZKH_S dj y₀) (layShHV2 dj y₀ ht hc hscope hU (pub := pub))
    (decAgreeHV2 dj y₀ ht hc hscope hU)

/-- **`rs` at the hidden-output `--zk` tables, at `sha512/row/v2` rows**: `rsShHSV2` moved along `planAtZKH_S`. -/
noncomputable abbrev rsZHSV2 : HmRowsZ (layZHV2 dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planAtZKH_S dj y₀) (decAgreeHV2 dj y₀ ht hc hscope hU) (rsShHSV2 dj y₀ ht hc hscope hU hpub)

/-- **`hHm` at the hidden-output `--zk` tables' `sha512/row/v2` rows** (`rsShHSV2_computes`, moved along). -/
theorem rsZHSV2_computes : (rsZHSV2 dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planAtZKH_S dj y₀) (decAgreeHV2 dj y₀ ht hc hscope hU) _
    (rsShHSV2_computes dj y₀ ht hc hscope hU hpub)

/-- **The compiled `--zk` count curve at the hidden-output `--zk` tables' `sha512/row/v2` rows**:
`zk_flock_countC_execOS_crH` with `place := placeZHV2` and `vb := rsZHSV2.bindingZ rsZHSV2_computes`. -/
theorem zk_flock_countC_execOS_atHSV2 (Hs : Hashes 𝔽 Dig (List UInt8))
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
          ((rsZHSV2 dj y₀ ht hc hscope hU hpub).bindingZ (rsZHSV2_computes dj y₀ ht hc hscope hU hpub))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
          (tabAtZKH dj y₀)
          ((rsZHSV2 dj y₀ ht hc hscope hU hpub).bindingZ (rsZHSV2_computes dj y₀ ht hc hscope hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_crH execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) dj y₀
    execArith_correct σs ks hks St hS Kd pop Lb src hA3 (placeZHV2 dj y₀ ht hc hscope hU)
    ((rsZHSV2 dj y₀ ht hc hscope hU hpub).bindingZ (rsZHSV2_computes dj y₀ ht hc hscope hU hpub))
    hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ZkReg
