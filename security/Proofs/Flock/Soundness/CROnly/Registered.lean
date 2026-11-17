module

public import Proofs.Flock.Soundness.CROnly.Defs
public import Proofs.Flock.Soundness.E2E
public import Proofs.Flock.Soundness.Registered.Defs
public import Proofs.Flock.Soundness.Registered.E2E
public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.CROnly.Link

/-! The end-to-end theorems with registered reads, under `cr/sha-512` alone. -/

@[expose] public section

namespace FlockSoundness.CROnly

open Game Model Accounting Extract LinkFinder Refine Audit Audit.Partition Binding Registered
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

theorem flock_e2e_sampled_hm96_reads
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    (rd : Reads lay rs)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT)
    (path : Bool × Finset (Fin n) → lay.Pos → ByteArray) (hchk : ∀ o, execAccept o → rd.Checked (reg σ) (path o))
    {qR : ℝ} (hR : (StrictCR.oneRun (audit L Reg (batchedSession H E plan)) σ fun o => rd.pick (reg σ) (path o)).CR
      H512 qR) :
    prob (fun o => execAccept o ∧
        (¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2 ∨
          rd.Off (reg σ)))
        (audit L Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) + ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) :=
  Registered.flock_e2e_sampled_hm96_reads P H E plan tab dp ones hConst zeros hZero lay rs hHm rd hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT path hchk hR

theorem flock_e2e_count_hm96_reads
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    (rd : Reads lay rs)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT)
    (path : Bool × Finset (Fin n) → lay.Pos → ByteArray) (hchk : ∀ o, execAccept o → rd.Checked (reg σ) (path o))
    {qR : ℝ} (hR : (StrictCR.oneRun (audit L Reg (batchedSession H E plan)) σ fun o => rd.pick (reg σ) (path o)).CR
      H512 qR) :
    prob (fun o => execAccept o ∧
        (K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card ∨
          rd.Off (reg σ)))
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) + ENNReal.ofReal (tr.bound qT) +
        ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) :=
  Registered.flock_e2e_count_hm96_reads P H E plan tab dp ones hConst zeros hZero lay rs hHm rd hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le K σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT path hchk hR

end FlockSoundness.CROnly
