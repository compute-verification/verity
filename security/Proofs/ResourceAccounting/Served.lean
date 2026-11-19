module

public import Properties.ResourceAccounting.Served
public import Proofs.Compose.Compose.ServiceBudget
public import Proofs.Flock.Claims.Sampling.Select.Bounds
public import Proofs.ResourceAccounting.Lemmas.Compose

/-!
# The served window's sample misses a set no more often than a uniform permutation's subset (`SampleMissesSubset`)

`Properties.Served.SampleMissesSubset` from two bounds that meet at `C(N − |B|, K)/C(N, K)`, `B` the tiles of `U` below
`N`:

* **the code** (`Properties.Spot.SecurityProofs.one_bound`): `sampleUnits (.subset K) N` from `L` uniform bytes
  misses `B`'s numbers, and so `U`, with probability at most that ratio;
* **the model** (`Budget.miss_eq`): a uniform permutation's `K`-subset misses `B` with probability exactly that ratio,
  the core's `Verity.SamplingDesign.missProb_permPrefix_eq`.
-/

@[expose] public section

open Claims renaming Ex → expect

namespace Properties.Served.SecurityProofs

open Pouw.RelativeSecurity
open ResourceAccounting.SecurityProofs.Compose.Budget (miss_eq)
open ResourceAccounting.SecurityProofs.TS.Stages (prob_prod)

theorem prob_eq_pr {α : Type} [Fintype α] (p : α → Prop) : prob p = Claims.Pr p := by
  classical
  rw [prob_eq_card, Claims.Pr, Claims.prUniform]

/-- The tiles of `Fin N` whose numbers are in `U`. -/
def tilesOf (N : ℕ) (U : Finset ℕ) : Finset (Fin N) := Finset.univ.filter fun t => t.val ∈ U

theorem mem_tilesOf {N : ℕ} {U : Finset ℕ} {t : Fin N} : t ∈ tilesOf N U ↔ t.val ∈ U := by
  simp [tilesOf]

/-- **`SampleMissesSubset`**: the code's subset sample misses `U` with probability at most `C(N − |B|, K)/C(N, K)`, `B`
the tiles of `U` (`one_bound` on `B`'s numbers), which the permutation's subset misses `B` with exactly
(`Budget.miss_eq`). -/
theorem sampleMissesSubset : SampleMissesSubset := by
  intro K N L U hK
  classical
  have hr : (tilesOf N U).map Fin.valEmbedding ⊆ Finset.range N := fun u hu => by
    obtain ⟨t, -, rfl⟩ := Finset.mem_map.1 hu
    exact Finset.mem_range.2 t.isLt
  have hU : ∀ u ∈ (tilesOf N U).map Fin.valEmbedding, u ∈ U := fun u hu => by
    obtain ⟨t, ht, rfl⟩ := Finset.mem_map.1 hu
    exact mem_tilesOf.1 ht
  have hcode := (Properties.Spot.SecurityProofs.one_bound N _ hr L).1 K hK
  rw [Finset.card_map] at hcode
  refine le_trans ?_ (hcode.trans ?_)
  · rw [prob_eq_pr]
    refine Claims.Pr_mono fun src h => ?_
    obtain ⟨⟨units, read⟩, hd, hm⟩ := h
    exact ⟨units, read, hd, fun u hu => hm u (hU u hu)⟩
  · exact (miss_eq K N (tilesOf N U) hK).ge.trans (le_of_eq (congrArg prob (funext fun β =>
      propext (forall₂_congr fun t _ => not_congr mem_tilesOf))))

end Properties.Served.SecurityProofs
