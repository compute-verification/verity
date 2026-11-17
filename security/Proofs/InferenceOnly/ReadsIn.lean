module

public import Definitions.InferenceOnly.Reads
public import Proofs.InferenceOnly.Pull

/-!
# Reads in a set of gates, and the pull into the contained circuit

* `freshInput_le_readsIn`: what a set reads novelly is at most all it reads, `readsIn d R univ`.
* `freshInput_pull_le_readsIn`: through a containment `M : c ≼ d`, a piece of a unit of `d` reads novelly at most what
  its unit reads in `M`'s image, baseline included, so no condition on the baselines is needed: a value the piece reads
  is a gate of `c` outside the piece or an input of `c`, and `M` sends either to a gate of its image that the unit reads.
-/

@[expose] public section

open Slowdown

namespace InferenceOnly.SecurityProofs

open Verity Finset

theorem freshInput_le_readsIn (d : RegisteredCircuit) (R : Finset (Fin d.D.N)) : freshInput d R ≤ readsIn d R univ := by
  unfold freshInput readsIn
  rw [inter_univ]
  exact sum_le_sum_of_subset sdiff_subset

/-- **The pull into the contained circuit**: through a containment `M : c ≼ d`, the piece of `c` that a unit of `d`
induces (`pull`) reads at most what the unit reads in `M`'s image, baseline included. -/
theorem freshInput_pull_le_readsIn {c d : RegisteredCircuit} (M : Containment c.D d.D) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : freshInput c ((pull M P).gates u) ≤ readsIn d (P.gates u) (univ.map ⟨M.emb, M.emb_inj⟩) := by
  unfold freshInput readsIn
  calc c.D.width (c.D.into ((pull M P).gates u) \ c.θ) ≤ c.D.width (c.D.into ((pull M P).gates u)) :=
        sum_le_sum_of_subset sdiff_subset
    _ = d.D.width ((c.D.into ((pull M P).gates u)).map ⟨M.emb, M.emb_inj⟩) := width_map (M := M) _
    _ ≤ d.D.width (d.D.into (P.gates u) ∩ univ.map ⟨M.emb, M.emb_inj⟩) := by
        refine sum_le_sum_of_subset fun b hb => ?_
        obtain ⟨a, -, rfl⟩ := mem_map.1 hb
        exact mem_inter.2 ⟨into_pull_map u hb, mem_map_of_mem _ (mem_univ a)⟩

end InferenceOnly.SecurityProofs
