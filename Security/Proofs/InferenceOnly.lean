import Definitions.InferenceOnly.Guarantees
import Proofs.InferenceOnly.Inequalities
import Proofs.InferenceOnly.LoomisWhitney
import Proofs.Core
import Proofs.InferenceOnly.Chain
import Proofs.InferenceOnly.Throttling
import Proofs.InferenceOnly.Support
import Proofs.InferenceOnly.Reads
import Proofs.InferenceOnly.Pull
import Proofs.InferenceOnly.Threads
import Proofs.InferenceOnly.Layered
import Proofs.InferenceOnly.Contained
import Proofs.InferenceOnly.Fit
import Proofs.InferenceOnly.FitKeep
import Proofs.InferenceOnly.FitSlotsWitness
import Proofs.InferenceOnly.WeightsSlotsWitness
import Proofs.InferenceOnly.FitRate
import Proofs.InferenceOnly.FitWitness
import Proofs.InferenceOnly.Dampener
import Proofs.InferenceOnly.Info
import Proofs.InferenceOnly.Transfers
import Proofs.InferenceOnly.Fmt
import Proofs.InferenceOnly.Fmt2
import Proofs.InferenceOnly.Training
import Proofs.InferenceOnly.Packed
import Proofs.InferenceOnly.PackedTraining
import Proofs.InferenceOnly.Remote
import Proofs.InferenceOnly.RemotePeriod
import Proofs.InferenceOnly.Verdict
import Proofs.InferenceOnly.DescVerdict
import Proofs.InferenceOnly.Activations
import Proofs.InferenceOnly.Covering
import Proofs.InferenceOnly.CoveringWitness
import Proofs.InferenceOnly.CoveringTwoCosts
import Proofs.InferenceOnly.CoveringTwoCostsWitness
import Proofs.InferenceOnly.CoveringPriced
import Proofs.InferenceOnly.CoveringPricedWitness
import Proofs.InferenceOnly.Cover
import Proofs.InferenceOnly.Ldl

/-!
# Proofs of NCI's guarantees

The read bound intersects each IU of the containing declaration with the contained circuit's gates (`pull`,
`Proofs.InferenceOnly.Pull`), each piece reading no more novel information than its IU.
-/

namespace InferenceOnly.SecurityProofs

open Verity Finset

theorem ReadBound : Guarantees.ReadBound := by
  intro Γ α c d M ⟨n, P, hP⟩ hB
  have hsize : ∀ u, c.D.size ((pull M P).gates u) ≤ Γ := fun u => (card_pull_gates_le u).trans (hP u).1
  have hle : Reads Γ c ≤ ∑ u, novelReads c ((pull M P).gates u) := Nat.sInf_le ⟨n, pull M P, hsize, rfl⟩
  calc (Reads Γ c : ℝ) ≤ ∑ u, (novelReads c ((pull M P).gates u) : ℝ) := by exact_mod_cast hle
    _ ≤ ∑ u, (novelReads d (P.gates u) : ℝ) :=
        Finset.sum_le_sum fun u _ => by exact_mod_cast novelReads_pull M hB P u
    _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) := Finset.sum_le_sum fun u _ => (hP u).2
    _ = α * d.D.size d.D.computed := by
        rw [← Finset.mul_sum]
        congr 1
        exact_mod_cast P.sum_card_gates

theorem Inference : Guarantees.Inference := by
  intro Γ α d n P prompt hsize hreads hprompt
  have hc : Complies Γ α d := by
    refine ⟨n, P, fun u => ⟨hsize u, le_trans ?_ (hprompt u)⟩⟩
    unfold novelReads
    refine Nat.cast_le.2 (Finset.sum_le_sum_of_subset fun a ha => ?_)
    rw [Finset.mem_sdiff] at ha
    exact (Finset.mem_union.1 (hreads u ha.1)).resolve_left ha.2
  refine ⟨hc, le_trans (iInf₂_le d ⟨Verity.SecurityProofs.ContainsImplements d d (Containment.refl d.D)
    fun _ _ h => h, hc⟩) ENNReal.div_self_le_one⟩

end InferenceOnly.SecurityProofs
