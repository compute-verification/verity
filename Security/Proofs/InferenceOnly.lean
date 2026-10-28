import Definitions.InferenceOnly.Guarantees
import Proofs.InferenceOnly.Inequalities
import Proofs.InferenceOnly.LoomisWhitney
import Proofs.Core
import Proofs.InferenceOnly.Chain
import Proofs.InferenceOnly.Throttling
import Proofs.InferenceOnly.Support
import Proofs.InferenceOnly.Reads
import Proofs.InferenceOnly.Pull
import Proofs.InferenceOnly.Rate
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
import Proofs.InferenceOnly.PartitionCovers
import Proofs.InferenceOnly.CompliesCK
import Proofs.InferenceOnly.Ldl
import Proofs.InferenceOnly.LdlWitnesses
import Proofs.InferenceOnly.Hardness
import Proofs.InferenceOnly.Ldl.Witnesses
import Proofs.InferenceOnly.Hardness.Witnesses

/-!
# Proofs of NCI's guarantees

The read bound takes the policy at its rate (`compliesRate_of_complies`) and intersects each inference block of the
containing declaration with the reference circuit's gates (`pull`, `Proofs.InferenceOnly.Pull`), each piece reading
no more fresh input bits than its block.
-/

namespace InferenceOnly.SecurityProofs

open Verity Finset

theorem ReadBound : Guarantees.ReadBound := by
  intro c k r d M hd hB
  obtain ⟨n, P, hP⟩ := compliesRate_of_complies hd
  have hsize : ∀ u, r.D.size ((pull M P).gates u) ≤ c := fun u => (card_pull_gates_le u).trans (hP u).1
  have hle : Reads c r ≤ ∑ u, novelReads r ((pull M P).gates u) := Nat.sInf_le ⟨n, pull M P, hsize, rfl⟩
  calc (Reads c r : ℝ) ≤ ∑ u, (novelReads r ((pull M P).gates u) : ℝ) := by exact_mod_cast hle
    _ ≤ ∑ u, (novelReads d (P.gates u) : ℝ) :=
        Finset.sum_le_sum fun u _ => by exact_mod_cast novelReads_pull M hB P u
    _ ≤ ∑ u, (k / c : ℝ) * (d.D.size (P.gates u) : ℝ) := Finset.sum_le_sum fun u _ => (hP u).2
    _ = (k / c : ℝ) * d.D.size d.D.computed := by
        rw [← Finset.mul_sum]
        congr 1
        exact_mod_cast P.sum_card_gates

theorem Inference : Guarantees.Inference := by
  intro c k d B P prompt hB hcard hreads hprompt
  have hc : Complies c k d := by
    refine ⟨B, P, ⟨hB, hcard, fun u => ?_⟩⟩
    have hsub : d.unregistered (P.gates u) ⊆ prompt u := fun a ha => by
      rw [Declaration.unregistered, Finset.mem_sdiff] at ha
      exact (Finset.mem_union.1 (hreads u ha.1)).resolve_left ha.2
    have hw : d.D.width (prompt u) = (prompt u).card := by
      simp [Circuit.width, hB.one_bit]
    exact (Finset.card_le_card hsub).trans (hw.symm.le.trans (hprompt u))
  refine ⟨hc, le_trans (iInf₂_le d ⟨Verity.SecurityProofs.ContainsImplements d d (Containment.refl d.D)
    fun _ _ h => h, hc⟩) ENNReal.div_self_le_one⟩

end InferenceOnly.SecurityProofs
