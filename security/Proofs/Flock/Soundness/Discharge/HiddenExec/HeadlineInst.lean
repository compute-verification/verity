module

public import Proofs.Flock.Soundness.Discharge.Exec.LiveOn
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Inst
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites
public import Proofs.Flock.Soundness.Discharge.Inst.KeysRow
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
import Proofs.Flock.Soundness.Discharge.HiddenExec.Headline
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.Layout.QSort

@[expose] public section

set_option Elab.async false

/-! The headline at the executable over hidden-output sessions at the template instance's program `instProg c n`, with no condition on the unit's input groups or on where the outputs are returned. -/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.Layout (portRowW)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hrf : Hidden.Any.RowsFitS c)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀))
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) {Reg : Type}

include ht hc in
theorem checkFactsH : CheckFacts c := by
  have P := parse_facts (parse_untyped ht hc).1
  obtain ⟨ls, hls⟩ := P.check
  exact check_facts P.typed hls

include hpub in
theorem tableH_bcSiteI {S : Finset (Fin n)} (x : SampleSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.acc ht hc hrf) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    rw [hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2_of (checkFactsH ht hc) (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

include hpub in
theorem sitesAtH_bcSiteI :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtH dj x₀) (tabAtH dj x₀) S R u), Binding.Sat (planAtH dj x₀) (tabAtH dj x₀) S R u M →
        BcSiteH ((sitesAtH ht hc hrf dj x₀).acc S R u h) pub ((sitesAtH ht hc hrf dj x₀).o S R u h)
          ((sitesAtH ht hc hrf dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableAtH dj x₀ S).acc ht hc hrf) pub ((tableAtH dj x₀ S).o u h) ((tableAtH dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (SampleSetupH I mPts S (dj S))
  · have e : tableAtH dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteI ht hc hrf dj hpub _ h
  · exact absurd hM (tableAtH_unsat dj x₀ hS _)

include ht hc x₀ in
theorem instKeysRowH (hU : Built c.unit) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true) :
    InstKeysRow c (portRowW c) := by
  have hcx : x₀.st.c = c := setupHidden_c_of_parse hc x₀.h
  have L := flat_layout ht x₀.h
  subst hcx
  exact instKeysRow_of_parse (st := Hidden.flatTwin x₀.st) L (parse_untyped ht hc).1 hpr hw

/-- **`pp` at the hidden tables, at `instProg`** (`SitesH.progPlaces_instProg` at `sitesAtH`). -/
noncomputable abbrev ppHI : ProgPlaces (instProg c n) (outsOfI c n) (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) :=
  SitesH.progPlaces_instProg (sitesAtH_c ht hc hrf dj x₀) hrf.1 (outsOfI c n)

/-- **`lay` at the hidden tables, at `instProg`**: `instLayoutH` at `sitesAtH`, the program's outputs the output words'
bits, the registration the population's public file. -/
noncomputable abbrev layAtHI :
    Binding.Layout ((instProg c n).partition (outsOfI c n)) (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
      (((instProg c n).partition (outsOfI c n)).placeDecoder (planAtH dj x₀) (tabAtH dj x₀)
        (ppHI ht hc hrf dj x₀).derived.place) :=
  instLayoutH (sitesAtH_c ht hc hrf dj x₀) hrf (outsOfI c n) (fun _ => pub)

/-- **`rs` at the hidden tables, at `instProg`**: `instRowsH` at `sitesAtH`, with `hsite` from `sitesAtH_bcSiteI` and
`InstKeysRow` from `instKeysRowH`. -/
noncomputable abbrev rsAtHI (hU : Built c.unit) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true) :
    Binding.HmRowsX (layAtHI ht hc hrf dj x₀ (pub := pub) (Reg := Reg)) :=
  instRowsH (sitesAtH_c ht hc hrf dj x₀) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (pinFit_of_rowsS ht hc hrf.1)
    (sitesAtH_bcSiteI ht hc hrf dj x₀ hpub) (instKeysRowH ht hc dj x₀ hU hpr hw)

theorem rsAtHI_computes (hU : Built c.unit) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true) :
    (rsAtHI ht hc hrf dj x₀ hpub hU hpr hw (Reg := Reg)).Computes :=
  instRowsH_computes (sitesAtH_c ht hc hrf dj x₀) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (pinFit_of_rowsS ht hc hrf.1)
    (checkFactsH ht hc).upv_pos (sitesAtH_bcSiteI ht hc hrf dj x₀ hpub) (instKeysRowH ht hc dj x₀ hU hpr hw)

theorem flock_headline_exec_hidden_inst {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    (hU : Built c.unit) (hpr : Layout.PortReads c hU)
    (hw : ProvedScope.wiresAcyclic c = true)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dec one fin capOf
      (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((instProg c n).partition (outsOfI c n)) H512 enc512 Lp
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtHI ht hc hrf dj x₀ hpub hU hpr hw (Reg := Reg)).binding
        (rsAtHI_computes ht hc hrf dj x₀ hpub hU hpr hw))
      (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 Lp
      (planAtH dj x₀ ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 Lp (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((instProg c n).partition (outsOfI c n)).incorrect
          (Xpub {instOneG (c := c) (n := n)} {instZeroG (c := c) (n := n)}
            (XplurL ((instProg c n).partition (outsOfI c n)) H512 enc512 Lp (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtHI ht hc hrf dj x₀ hpub hU hpr hw (Reg := Reg)).binding
                (rsAtHI_computes ht hc hrf dj x₀ hpub hU hpr hw))
              (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((instProg c n).partition (outsOfI c n)).linkBoundE
          (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtHI ht hc hrf dj x₀ hpub hU hpr hw (Reg := Reg)).binding
            (rsAtHI_computes ht hc hrf dj x₀ hpub hU hpr hw)) k Rw M ρ t' :=
  flock_headline_hidden dj x₀ (instProg c n) (outsOfI c n) σs ks hks St hS Kd pop Lb src hA3
    (ppHI ht hc hrf dj x₀)
    {instOneG (c := c) (n := n)} (constCols_instProgH (sitesAtH_c ht hc hrf dj x₀) hrf.1 (outsOfI c n))
    {instZeroG (c := c) (n := n)} (zeroCols_instProgH (sitesAtH_c ht hc hrf dj x₀) hrf.1 (outsOfI c n))
    (layAtHI ht hc hrf dj x₀ (pub := pub)) (rsAtHI ht hc hrf dj x₀ hpub hU hpr hw)
    (rsAtHI_computes ht hc hrf dj x₀ hpub hU hpr hw) hk hRw hM hρ hr ht' K wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.HiddenExec
