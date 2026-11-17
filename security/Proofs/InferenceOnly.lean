module

public import Proofs.Slowdown
public import Proofs.Core
public import Proofs.InferenceOnly.Pull
public import Proofs.InferenceOnly.Rate
public import Proofs.InferenceOnly.PartitionCovers
public import Proofs.InferenceOnly.CompliesCK
public import Proofs.InferenceOnly.Lemmas

/-!
# Proofs of the inference-only policy's guarantees

The read bound intersects each inference block of the containing declaration with the reference training algorithm's gates
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

end InferenceOnly.SecurityProofs
