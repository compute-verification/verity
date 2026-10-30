import Definitions.Nci.Guarantees
import Proofs.Nci.Inequalities
import Proofs.Nci.LoomisWhitney
import Proofs.Core
import Proofs.Nci.Chain
import Proofs.Nci.Throttling
import Proofs.Nci.Support
import Proofs.Nci.Reads
import Proofs.Nci.Pull
import Proofs.Nci.Rate
import Proofs.Nci.Threads
import Proofs.Nci.Layered
import Proofs.Nci.Contained
import Proofs.Nci.Fit
import Proofs.Nci.FitKeep
import Proofs.Nci.FitSlotsWitness
import Proofs.Nci.WeightsSlotsWitness
import Proofs.Nci.FitRate
import Proofs.Nci.FitWitness
import Proofs.Nci.Dampener
import Proofs.Nci.Info
import Proofs.Nci.Transfers
import Proofs.Nci.Fmt
import Proofs.Nci.Fmt2
import Proofs.Nci.Training
import Proofs.Nci.Packed
import Proofs.Nci.PackedTraining
import Proofs.Nci.Remote
import Proofs.Nci.RemotePeriod
import Proofs.Nci.Capability
import Proofs.Nci.Verdict
import Proofs.Nci.DescVerdict
import Proofs.Nci.Activations
import Proofs.Nci.Covering
import Proofs.Nci.CoveringWitness
import Proofs.Nci.CoveringTwoCosts
import Proofs.Nci.CoveringTwoCostsWitness
import Proofs.Nci.CoveringPriced
import Proofs.Nci.CoveringPricedWitness
import Proofs.Nci.Cover
import Proofs.Nci.PartitionCovers
import Proofs.Nci.CompliesCK
import Proofs.Nci.Ldl

/-!
# Proofs of NCI's guarantees

The read bound takes the policy at its rate (`compliesRate_of_complies`) and intersects each inference block of the
containing declaration with the reference circuit's gates (`pull`, `Proofs.Nci.Pull`), each piece reading
no more fresh input bits than its block.
-/

namespace Nci.SecurityProofs

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
  intro c k d B P prompt hB hS
  obtain ⟨hcard, hreads, hprompt⟩ := hS
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

end Nci.SecurityProofs
