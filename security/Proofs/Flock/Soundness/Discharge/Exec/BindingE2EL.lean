module

public import Proofs.Flock.Soundness.Discharge.Exec.E2EL
public import Proofs.Flock.Soundness.Discharge.Exec.StrictCRL
public import Proofs.Flock.Soundness.Binding.E2E
public import Proofs.Flock.Soundness.Binding.RowsX

/-! The end-to-end theorem with the value binding of `hm96-sha512/row/v1` rows, whose one gap is `HmRowComputes` (`hHm`) and which takes no A3. -/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Refine Binding Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

/-- It takes no A3: `miss_L` is the sampling design's own, and `Prog.flock_headline` bounds the live sample's under A3.
The serving roots' rewinding `tr` is a named hypothesis, since the refinement `ExecRefines` it stands for is not proved. -/
theorem flock_e2e_count_hm96L
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H E Lf (plan (L.draw ω) (reg σ)) j
      (batchStratL H E Lf plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSessionL H E Lf plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (XplurL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' + ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((flock_e2e_countL P H E Lf plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (ksAvgBE_le_strictL H E Lf plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

theorem flock_e2e_count_hm96LX
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H E Lf (plan (L.draw ω) (reg σ)) j
      (batchStratL H E Lf plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSessionL H E Lf plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (XplurL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' + ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((flock_e2e_countL P H E Lf plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (ksAvgBE_le_strictL H E Lf plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

/-- It reads the sampling design only through `ε_ks` and `ε_bind`, so it holds at the live sample with no A3. -/
theorem flock_e2e_sampled_hm96L
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H E Lf (plan (L.draw ω) (reg σ)) j
      (batchStratL H E Lf plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSessionL H E Lf plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (XplurL P H E Lf plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) + P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((flock_e2e_sampledL P H E Lf plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht σ hCR).trans
      (add_le_add (ksAvgBE_le_strictL H E Lf plan k (reg σ) (cont σ) hKS) le_rfl))
    (tr.le hT))

end FlockSoundness.Discharge.Exec
