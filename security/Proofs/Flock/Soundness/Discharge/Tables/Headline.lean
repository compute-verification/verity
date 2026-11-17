module

public import Proofs.Flock.Soundness.Discharge.Tables.Session
public import Proofs.Flock.Soundness.Binding.RowsX
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Audit.SamplingDesign
import Proofs.Flock.Soundness.Discharge.Exec.BindingE2EL
import Proofs.Flock.Soundness.Discharge.Exec.Headline
import Proofs.Flock.Soundness.Discharge.Exec.HeadlineL

/-! The headline at the executable for sessions of `J` tables, whose error term
`miss(K) + ksAvgStrict + linkBoundE` does not grow with `J`. -/

@[expose] public section

namespace FlockSoundness.Discharge.Tables

open Game Model Accounting Refine Audit Audit.Partition Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {N' n : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N')) {Reg : Type} {mPts : ℕ}

theorem _root_.FlockSoundness.Prog.flock_headline_live_tables
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs) (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf
      (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOfJ dg plan N wr)) τ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have h := p.flock_headline_execL outs plan tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst
    zeros hZero lay rs hHm hk hRw hM hρ hr ht K I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
  rwa [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h

theorem execOS_subset_miss_le (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src) (c : ℕ) :
    (SamplingDesign.execOS (n := n) src (.subset kd) n).miss c ≤ (SamplingDesign.subset n kd hkd).miss c :=
  iSup₂_mono fun B _ => SamplingDesign.execOS_subset_escape_le n kd hkd Lb src hA3 B

theorem _root_.FlockSoundness.Prog.flock_headline_live_tables_subset
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (hK : K ≤ n) (I : Inputs)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf (SamplingDesign.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)) τ ≤
      ((n - K).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        StrictCR.ksAvgStrict plan (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := SamplingDesign.execOS (n := n) src (.subset kd) n)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := flock_e2e_count_hm96L (p.partition outs) H512 enc512 leaf512 plan tab (execAccept := fun o => o.1 = true)
    pp.derived ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := flock_headline_exec_L plan I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) _ hH
  rw [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h
  refine h.trans (add_le_add (add_le_add ?_ le_rfl) le_rfl)
  rw [← SamplingDesign.subset_miss hkd hK]
  exact execOS_subset_miss_le kd hkd Lb src hA3 K

theorem _root_.FlockSoundness.Prog.flock_headline_live_tables_subset_miss
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf (SamplingDesign.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict plan (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := SamplingDesign.execOS (n := n) src (.subset kd) n)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := flock_e2e_count_hm96L (p.partition outs) H512 enc512 leaf512 plan tab (execAccept := fun o => o.1 = true)
    pp.derived ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := flock_headline_exec_L plan I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) _ hH
  rw [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h
  exact h.trans (add_le_add (add_le_add (execOS_subset_miss_le kd hkd Lb src hA3 K) le_rfl) le_rfl)

theorem _root_.FlockSoundness.Prog.flock_headline_live_tables_subset_missX
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (plan S R).length mPts → Transcript)
    (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Opens 𝔽 Dig × Salts (List UInt8))
    (dg : ByteArray → ByteArray)
    (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → ByteArray → List Msg)
    (one : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → ℕ → Bool)
    (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (plan S R).length → Fin 2 → Array ByteArray → List Msg)
    (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (plan S R).length → Cap Dig) (N : ℕ)
    (hdec : DecodesOnJ H512 enc512 leaf512 plan I wr opn dec one fin capOf (SamplingDesign.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
      (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω)
        (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 plan
        (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm)
              (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg plan N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict plan (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := SamplingDesign.execOS (n := n) src (.subset kd) n)
          plan tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := flock_e2e_count_hm96LX (p.partition outs) H512 enc512 leaf512 plan tab
    (execAccept := fun o => o.1 = true) pp.derived ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := flock_headline_exec_L plan I (liveOfJ dg plan N wr) τ
    (simAuditJ H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_tables H512 enc512 leaf512 plan I wr opn dg dec one fin capOf N hdec τ) _ hH
  rw [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h
  exact h.trans (add_le_add (add_le_add (execOS_subset_miss_le kd hkd Lb src hA3 K) le_rfl) le_rfl)

end FlockSoundness.Discharge.Tables
