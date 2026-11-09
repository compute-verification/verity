module

public import Proofs.InferenceOnly.Inequalities
public import Proofs.InferenceOnly.LoomisWhitney
public import Proofs.InferenceOnly.Cut
public import Proofs.InferenceOnly.CutWitness
public import Proofs.Core
public import Proofs.InferenceOnly.Chain
public import Proofs.InferenceOnly.Throttling
public import Proofs.InferenceOnly.Support
public import Proofs.InferenceOnly.Reads
public import Proofs.InferenceOnly.Pull
public import Proofs.InferenceOnly.Rate
public import Proofs.InferenceOnly.Threads
public import Proofs.InferenceOnly.Layered
public import Proofs.InferenceOnly.Contained
public import Proofs.InferenceOnly.Fit
public import Proofs.InferenceOnly.FitKeep
public import Proofs.InferenceOnly.FitSlotsWitness
public import Proofs.InferenceOnly.WeightsSlotsWitness
public import Proofs.InferenceOnly.FitRate
public import Proofs.InferenceOnly.FitWitness
public import Proofs.InferenceOnly.Dampener
public import Proofs.InferenceOnly.Info
public import Proofs.InferenceOnly.Transfers
public import Proofs.InferenceOnly.Fmt
public import Proofs.InferenceOnly.Fmt2
public import Proofs.InferenceOnly.Training
public import Proofs.InferenceOnly.Packed
public import Proofs.InferenceOnly.PackedTraining
public import Proofs.InferenceOnly.Remote
public import Proofs.InferenceOnly.RemotePeriod
public import Proofs.InferenceOnly.Capability
public import Proofs.InferenceOnly.Verdict
public import Proofs.InferenceOnly.DescVerdict
public import Proofs.InferenceOnly.Activations
public import Proofs.InferenceOnly.Covering
public import Proofs.InferenceOnly.CoveringWitness
public import Proofs.InferenceOnly.CoveringTwoCosts
public import Proofs.InferenceOnly.CoveringTwoCostsWitness
public import Proofs.InferenceOnly.Cover
public import Proofs.InferenceOnly.PartitionCovers
public import Proofs.InferenceOnly.CompliesCK
public import Proofs.InferenceOnly.Witnesses
public import Proofs.InferenceOnly.Formalism
public import Proofs.InferenceOnly.Lemmas

/-!
# Proofs of NCI's guarantees

The read bound intersects each inference block of the containing declaration with the reference circuit's gates
(`pull`, `Proofs.InferenceOnly.Pull`), each piece reading no more fresh input bits than its block.
-/

@[expose] public section

namespace InferenceOnly.SecurityProofs

open Verity Finset

theorem ReadBound : Guarantees.ReadBound := by
  intro Γ α r d M hd hB
  obtain ⟨n, P, hP⟩ := hd
  have hsize : ∀ u, r.D.size ((pull M P).gates u) ≤ Γ := fun u => (card_pull_gates_le u).trans (hP u).1
  have hle : Reads Γ r ≤ ∑ u, novelReads r ((pull M P).gates u) := Nat.sInf_le ⟨n, pull M P, hsize, rfl⟩
  calc (Reads Γ r : ℝ) ≤ ∑ u, (novelReads r ((pull M P).gates u) : ℝ) := by exact_mod_cast hle
    _ ≤ ∑ u, (novelReads d (P.gates u) : ℝ) :=
        Finset.sum_le_sum fun u _ => by exact_mod_cast novelReads_pull M hB P u
    _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) := Finset.sum_le_sum fun u _ => (hP u).2
    _ = α * d.D.size d.D.computed := by
        rw [← Finset.mul_sum]
        congr 1
        exact_mod_cast P.sum_card_gates

theorem Inference : Guarantees.Inference := by
  intro c k d B P prompt hB hS
  obtain ⟨hcard, hreads, hprompt⟩ := hS
  have hc : Complies c k d := by
    refine ⟨B, P, ⟨hB, fun u => (hcard u).le, fun u => ?_⟩⟩
    have hsub : d.unregistered (P.gates u) ⊆ prompt u := fun a ha => by
      rw [RegisteredCircuit.unregistered, Finset.mem_sdiff] at ha
      exact (Finset.mem_union.1 (hreads u ha.1)).resolve_left ha.2
    have hw : d.D.width (prompt u) = (prompt u).card := by
      simp [Circuit.width, hB.one_bit]
    exact (Finset.card_le_card hsub).trans (hw.symm.le.trans (hprompt u))
  refine ⟨hc, le_trans (iInf₂_le d ⟨Verity.SecurityProofs.ContainsImplements d d (Containment.refl d.D)
    fun _ _ h => h, hc⟩) ENNReal.div_self_le_one⟩

end InferenceOnly.SecurityProofs
