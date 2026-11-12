module

public import Definitions.Claims.RecursionGeneric
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.RecGeneric.SampledProof
public import Proofs.Flock.RecGeneric.Lift
public import Proofs.Flock.RecGeneric.BadDev

/-!
# L1, proved

The inner stage's failure where every outer extraction succeeds (`Claims.PrBadGood`) is at most the inner protocol's
own failure against the plurality prover `P_O`'s aborting runs, plus `N_I` bad samples and a rejecting judged rerun
(`Comp.PrBadGood_le`). The inner protocol's knowledge soundness bounds the first by its terms at `P_O`'s runs
(`Proto.KSAbortLin`), each read on the composite's reruns plus the reruns it asks about where a sample is bad or `E_O`
gives witnesses outside `R_O` (`Comp.PrAt_le_lift`), at most `d_i` times one rerun's chance (`RunFinder.ReadsLe`). A
bad sample costs the gated link (`Comp.Pr_badO_le_link`) and a rejecting judged rerun costs `B_I`
(`Comp.Pr_innerRejects_le_fork`). So the inner stage pays `Comp.err` past the outer terms (`Comp.PrBadGood_le_err`),
and with `E_O`'s failure at each of `E^rep`'s `N_I` reruns (`Claims.servedERep_bad_le_good`, `Comp.OuterKSAt`) the
composite pays `Comp.err` (`Comp.PrBad_le_err`), which is L1, `Definitions.Claims.RecursionGeneric`'s `CompositionKS`
(`compositionKS`). With B1, L1 gives the recursive audit through the sampled compiler
(`Flock.SecurityProofs.RecursiveSampledKS`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

namespace Comp

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
variable {K : Comp Coin} {I : Proto Coin W} {S : Server} {H : Msg → Msg} {Ω : Type} {P : Prover Ω} {x : Msg}

/-- **The inner stage pays `Comp.err` past the outer terms**: where every outer extraction succeeds, the composite
fails at most at `κ_Π`, the inner terms on the composite's reruns, `PrOut` at each rerun they read, `B_I`, and the
gated link at weight `β`. -/
theorem PrBadGood_le_err (hWF : I.WF) (hKS : I.KSAbortLin H) (hCF : I.LastCoinFree H) (hB : S.Binds)
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) [Fintype Ω] [Nonempty Ω]
    (m : Fin I.nT → ℕ) (M : ℕ) :
    PrBadGood H K.V (K.R I) K.RO K.EO (K.dec (S.IVStar I.V I.ans) H) (fun x => I.waits (K.xI x))
        (fun x => I.k (K.xI x)) (fun x => I.cap (K.xI x)) (fun v => I.opens (S.openView v))
        (fun x => I.assemble (K.xI x)) (fun x => I.check (K.xI x)) x P ≤
      I.κ (K.xI x) +
        ∑ i, (I.terms i).w (K.xI x) *
          (((I.terms i).F (m i)).PrAt H (K.innerModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
            (I.terms i).tail (K.xI x) (m i)) +
        K.reads I x * K.PrOut H P x +
        (forkPlan (Coin := Coin) I.V.k).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
        K.β I x * ((linkPlan (Coin := Coin) I.V.k M).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
          1 / ((M : ℝ) + 1) + K.PrOut H P x) := by
  obtain ⟨⟨decC, hdec⟩, -, hw0, -, hRd⟩ := hWF
  have h1 := PrBadGood_le (K := K) (I := I) (S := S) (H := H) (P := P) (x := x) hCF decC hdec hK.replays hK.refuses
    hK.rounds
  have h2 := hKS (K.xI x) Ω (K.innerPO I S H P x decC) ((Fin (K.V.k x) → Coin) × K.EO.C x) (K.keepAO I S H P x) m
  have h3 : ∀ i, ((I.terms i).F (m i)).PrAt H
      (runModel H I.V (K.innerPO I S H P x decC) (K.xI x) (K.keepAO I S H P x)) (K.xI x) ≤
      ((I.terms i).F (m i)).PrAt H (K.innerModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
        (I.terms i).d (K.xI x) *
          (Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2) + K.PrOut H P x) := by
    intro i
    refine (PrAt_le_lift _ hCF decC hdec hK.rounds hK.replays hK.callsTrue hK.callsLog).trans (add_le_add le_rfl ?_)
    refine (hRd i (m i) Ω (K.V.k x) (K.EO.C x) (K.innerModel (S.IVStar I.V I.ans) H P x) (K.xI x)
      (fun ω u => K.BadO I S H P x ω u ∨ K.OutRO H P x ω u)).trans ?_
    exact mul_le_mul_of_nonneg_left (Claims.Pr_or_le _ _) (Nat.cast_nonneg _)
  have h4 : ∑ i, (I.terms i).w (K.xI x) *
        (((I.terms i).F (m i)).PrAt H
          (runModel H I.V (K.innerPO I S H P x decC) (K.xI x) (K.keepAO I S H P x)) (K.xI x) +
          (I.terms i).tail (K.xI x) (m i)) ≤
      ∑ i, (I.terms i).w (K.xI x) *
          (((I.terms i).F (m i)).PrAt H (K.innerModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
            (I.terms i).tail (K.xI x) (m i)) +
        K.reads I x *
          (Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2) + K.PrOut H P x) := by
    rw [Comp.reads, Finset.sum_mul, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun i _ => ?_
    exact (mul_le_mul_of_nonneg_left (add_le_add (h3 i) le_rfl) (hw0 i (K.xI x))).trans_eq (by ring)
  have h5 := Pr_badO_le_link (K := K) (P := P) hB hCF hK M
  have h6 := Pr_innerRejects_le_fork (K := K) (I := I) (S := S) (P := P) hK
  have hr0 : 0 ≤ K.reads I x :=
    Finset.sum_nonneg fun i _ => mul_nonneg (hw0 i (K.xI x)) (Nat.cast_nonneg _)
  have hNr : 0 ≤ (K.NI I x : ℝ) + K.reads I x := add_nonneg (Nat.cast_nonneg _) hr0
  have h7 := mul_le_mul_of_nonneg_left h5 hNr
  rw [Comp.β]
  linarith [h1, h2, h4, h6, h7]

/-- **L1 at one statement**: where `E_O` is knowledge sound at `x` and the compiler meets its contract there, the
composite fails at most at `Comp.err`: `E^rep`'s `N_I` reruns each pay `E_O`'s failure (`servedERep_bad_le_good`), and
the inner stage the rest (`PrBadGood_le_err`). -/
theorem PrBad_le_err (hWF : I.WF) (hKS : I.KSAbortLin H) (hCF : I.LastCoinFree H) (hB : S.Binds)
    (hO : K.OuterKSAt H x)
    (hK : K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x) [Fintype Ω] [Nonempty Ω]
    (m : Fin I.nT → ℕ) (M : ℕ) :
    PrBad H K.V (K.R I) (K.E I S) x P ≤ K.err I S H P x m M := by
  have hE : PrBad H K.V (K.R I) (K.E I S) x P =
      PrBad H K.V (K.R I) (servedERep K.V K.EO (K.dec (S.IVStar I.V I.ans) H) (fun x => I.waits (K.xI x))
        (fun x => I.k (K.xI x)) (fun x => I.cap (K.xI x)) (fun v => I.opens (S.openView v))
        (fun x => I.assemble (K.xI x)) (fun x => I.check (K.xI x))) x P := by
    unfold PrBad
    exact Pr_congr fun s => by rw [E_eval]
  have hsplit := servedERep_bad_le_good H K.V (K.R I) K.RO K.EO (K.dec (S.IVStar I.V I.ans) H)
    (fun x => I.waits (K.xI x)) (fun x => I.k (K.xI x)) (fun x => I.cap (K.xI x))
    (fun v => I.opens (S.openView v)) (fun x => I.assemble (K.xI x)) (fun x => I.check (K.xI x)) x P
  have hA : (1 + (I.waits (K.xI x) : ℝ) * (I.k (K.xI x) : ℝ)) = K.NI I x := by
    rw [Comp.NI, Nat.cast_add, Nat.cast_one, Nat.cast_mul]
  rw [hA] at hsplit
  have hin := PrBadGood_le_err (P := P) hWF hKS hCF hB hK m M
  have hA0 : (0 : ℝ) ≤ K.NI I x := Nat.cast_nonneg _
  have hp0 : 0 ≤ PrB H K.V P x := PrB_nonneg H K.V P x
  have hsq : Real.sqrt (K.γ I x * PrB H K.V P x) = (K.NI I x : ℝ) * Real.sqrt (K.γO x * PrB H K.V P x) := by
    rw [Comp.γ, mul_assoc, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hA0]
  have hmul := mul_le_mul_of_nonneg_left (hO Ω P) hA0
  have hκ : K.κ I x = I.κ (K.xI x) + K.NI I x * K.κO x := rfl
  rw [hE, Comp.err, hsq, hκ]
  linarith [hsplit, hin, hmul]

end Comp

/-- **L1** (`CompositionKS`): at every statement where `E_O` is knowledge sound and the compiler meets its contract,
the composed system fails at most at `Comp.err` (`Comp.PrBad_le_err`). -/
theorem compositionKS : CompositionKS := by
  intro Coin W _ _ _ H I S K hWF hKS hCF hB x hO _ _ hK Ω _ _ P m M
  exact Comp.PrBad_le_err hWF hKS hCF hB hO hK m M

end Claims.RecGeneric

/-- **The recursive audit through the sampled compiler** (`Flock.Guarantees.RecursiveSampledKS`): B1 at L1. -/
theorem Flock.SecurityProofs.RecursiveSampledKS : Flock.Guarantees.RecursiveSampledKS :=
  Claims.RecGeneric.recursiveSampledKS_of_L1 Claims.RecGeneric.compositionKS
