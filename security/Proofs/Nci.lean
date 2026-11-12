module

public import Proofs.Nci.Cut
public import Proofs.Nci.CutWitness
public import Proofs.Nci.Untied
public import Proofs.Nci.CutCharged
public import Proofs.Core
public import Proofs.Nci.Pull
public import Proofs.Nci.Rate
public import Proofs.Nci.Threads
public import Proofs.Nci.Layered
public import Proofs.Nci.Fit
public import Proofs.Nci.FitKeep
public import Proofs.Nci.FitSlotsWitness
public import Proofs.Nci.WeightsSlotsWitness
public import Proofs.Nci.FitRate
public import Proofs.Nci.FitWitness
public import Proofs.Nci.Dampener
public import Proofs.Nci.Remote
public import Proofs.Nci.RemotePeriod
public import Proofs.Nci.Capability
public import Proofs.Nci.Verdict
public import Proofs.Nci.DescVerdict
public import Proofs.Nci.Covering
public import Proofs.Nci.CoveringWitness
public import Proofs.Nci.CoveringTwoCosts
public import Proofs.Nci.CoveringTwoCostsWitness
public import Proofs.Nci.Cover
public import Proofs.Nci.PartitionCovers
public import Proofs.Nci.CompliesCK
public import Proofs.Nci.Witnesses
public import Proofs.Nci.Formalism
public import Proofs.Nci.Lemmas

/-!
# Proofs of NCI's guarantees

The read bound intersects each inference block of the containing declaration with the reference circuit's gates
(`pull`, `Proofs.Nci.Pull`), each piece reading no more fresh input bits than its block.
-/

@[expose] public section

namespace Nci.SecurityProofs

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

end Nci.SecurityProofs
