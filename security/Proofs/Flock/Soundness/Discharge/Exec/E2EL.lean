module

public import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockPublicL
public import Proofs.Flock.Soundness.E2E
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-! The end-to-end theorem as a skeleton whose gaps are named hypotheses (`hExec`, `dp`, `hConst`, `hZero`, `vb`), about the
protocol as Lean defines it: the executable's verdict enters only when `hExec` is discharged. No theorem carries
`GateRows.rows_sound` to a correct unit's committed values. -/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

section

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

/-- The committed values are 1 on `ones`, 0 on `zeros` and the plurality elsewhere (`Xpub`); `hExec`, `dp`, `hConst`, `hZero` and `vb` are unproved gaps, taken as named hypotheses. -/
theorem flock_e2e_countL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (flock_batched_count_publicL P H E Lf plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht K σ hCR)

theorem flock_e2e_sampledL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (flock_batched_sampled_publicL P H E Lf plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht σ hCR)

end

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

variable {D Sl : Type} [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {L : SamplingDesign n}

theorem _root_.FlockSoundness.Prog.flock_e2e_countL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros (XplurL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  FlockSoundness.Discharge.Exec.flock_e2e_countL (p.partition outs) H E Lf plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht K σ hCR

theorem _root_.FlockSoundness.Prog.flock_e2e_sampledL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.partition outs).incorrect
          (Xpub ones zeros (XplurL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  FlockSoundness.Discharge.Exec.flock_e2e_sampledL (p.partition outs) H E Lf plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht σ hCR

end FlockSoundness.Discharge.Exec
