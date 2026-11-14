module

public import Proofs.Slowdown.Cut
public import Proofs.Slowdown.CutWitness
public import Proofs.Slowdown.Untied
public import Proofs.Slowdown.CutCharged
public import Proofs.Core
public import Proofs.InferenceOnly.Pull
public import Proofs.InferenceOnly.Rate
public import Proofs.Slowdown.Threads
public import Proofs.Slowdown.Layered
public import Proofs.Slowdown.Fit
public import Proofs.Slowdown.FitKeep
public import Proofs.Slowdown.FitSlotsWitness
public import Proofs.Slowdown.WeightsSlotsWitness
public import Proofs.Slowdown.FitRate
public import Proofs.Slowdown.FitWitness
public import Proofs.Slowdown.Slowdown
public import Proofs.Slowdown.Remote
public import Proofs.Slowdown.RemoteMonitoringPeriod
public import Proofs.Slowdown.Capability
public import Proofs.InferenceOnly.Verdict
public import Proofs.InferenceOnly.DescVerdict
public import Proofs.Slowdown.Covering
public import Proofs.Slowdown.CoveringWitness
public import Proofs.Slowdown.CoveringTwoCosts
public import Proofs.Slowdown.CoveringTwoCostsWitness
public import Proofs.InferenceOnly.Cover
public import Proofs.InferenceOnly.PartitionCovers
public import Proofs.InferenceOnly.CompliesCK
public import Proofs.Slowdown.Witnesses
public import Proofs.Slowdown.Formalism
public import Proofs.InferenceOnly.Lemmas

/-!
# Proofs of the inference-only policy's guarantees

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
  have hle : Reads Γ r ≤ ∑ u, freshInput r ((pull M P).gates u) := Nat.sInf_le ⟨n, pull M P, hsize, rfl⟩
  calc (Reads Γ r : ℝ) ≤ ∑ u, (freshInput r ((pull M P).gates u) : ℝ) := by exact_mod_cast hle
    _ ≤ ∑ u, (freshInput d (P.gates u) : ℝ) :=
        Finset.sum_le_sum fun u _ => by exact_mod_cast freshInput_pull M hB P u
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
