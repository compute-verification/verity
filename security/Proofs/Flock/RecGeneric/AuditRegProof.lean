module

public import Proofs.Flock.RecGeneric.AuditReg
public import Proofs.Flock.RecGeneric.StatementProof
public import Proofs.Core.Game.Prob

/-!
# The recursive audit at the registered design, proved

A strategy of the registered-design audit is one of the fixed-design audit at the design it registered, with the same
outcome probabilities (`Spot.Sampling.prob_auditReg`). There `RecursiveSampledKS` bounds acceptance by `P_a(F)` plus the
error; at a cutoff `δ ≥ P_a(F)` that bounds the event, and below it the event is empty.
-/

@[expose] public section

open Claims Claims.RecGeneric FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.SampledGeneric in
/-- **The recursive audit at the registered design** (`Flock.Guarantees.RecursiveAuditBoundReg`): `RecursiveSampledKS`
at the design the strategy registered. -/
theorem Flock.SecurityProofs.RecursiveAuditBoundReg : Flock.Guarantees.RecursiveAuditBoundReg := by
  intro Coin W _ _ _ H I Sv K mI Mo n Reg Dh Hc L B Correct stmt reads opens M hWF hKS hLC hSv hO hJ hM σ hCR δ
  have hb := Flock.SecurityProofs.RecursiveSampledKS Coin W H I Sv K mI Mo n Reg Dh Hc (L σ.1) B Correct stmt reads
    opens M hWF hKS hLC hSv hO hJ hM (atReg σ) hCR
  refine (prob_auditReg L _ _ σ).trans_le ?_
  by_cases hE : (L σ.1).missProb (incorrect B Correct (reg (atReg σ))
      (((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials (L σ.1)).plur (reg (atReg σ))
        (cont (atReg σ)))) ≤ δ
  · refine (Verity.Game.prob_mono (fun o h => h.1) _ _).trans (hb.trans ?_)
    rw [add_assoc]
    exact add_le_add_right hE _
  · exact ((Verity.Game.prob_mono (q := fun _ => False) (fun o h => hE h.2) _ _).trans
      (Verity.Game.prob_false _ _).le).trans (zero_le _)
