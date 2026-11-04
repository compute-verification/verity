module

public import Definitions.Nci.Guarantees
public import Proofs.Nci.Inequalities
public import Proofs.Nci.LoomisWhitney
public import Proofs.Core
public import Proofs.Nci.Chain
public import Proofs.Nci.Throttling
public import Proofs.Nci.Support
public import Proofs.Nci.Reads
public import Proofs.Nci.Pull
public import Proofs.Nci.Threads
public import Proofs.Nci.Layered
public import Proofs.Nci.Contained
public import Proofs.Nci.Fit
public import Proofs.Nci.FitKeep
public import Proofs.Nci.FitSlotsWitness
public import Proofs.Nci.WeightsSlotsWitness
public import Proofs.Nci.FitRate
public import Proofs.Nci.FitWitness
public import Proofs.Nci.Dampener
public import Proofs.Nci.Info
public import Proofs.Nci.Transfers
public import Proofs.Nci.Fmt
public import Proofs.Nci.Fmt2
public import Proofs.Nci.Training
public import Proofs.Nci.Packed
public import Proofs.Nci.PackedTraining
public import Proofs.Nci.Remote
public import Proofs.Nci.RemotePeriod
public import Proofs.Nci.Capability
public import Proofs.Nci.Verdict
public import Proofs.Nci.DescVerdict
public import Proofs.Nci.Activations
public import Proofs.Nci.Covering
public import Proofs.Nci.CoveringWitness
public import Proofs.Nci.CoveringTwoCosts
public import Proofs.Nci.CoveringTwoCostsWitness
public import Proofs.Nci.CoveringPriced
public import Proofs.Nci.CoveringPricedWitness
public import Proofs.Nci.Cover
public import Proofs.Nci.PartitionCovers
public import Proofs.Nci.CompliesCK
public import Proofs.Nci.Ldl

/-!
# Proofs of NCI's guarantees

The read bound intersects each IU of the containing declaration with the contained circuit's gates (`pull`,
`Proofs.Nci.Pull`), each piece reading no more novel information than its IU.
-/

@[expose] public section

namespace Nci.SecurityProofs

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

end Nci.SecurityProofs
