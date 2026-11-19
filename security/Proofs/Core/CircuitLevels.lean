module

public import Definitions.Core.CircuitLevels

/-! # Levels at a circuit: a level's blocks are its partition's nonempty units -/

@[expose] public section

namespace Verity

open Finset

namespace Partition

variable {C : Circuit} {n : ℕ}

theorem map_block (P : Partition C n) (u : Fin n) :
    (P.block u).map (Function.Embedding.subtype _) = P.gates u := by
  ext g
  simp only [block, gates, mem_map, mem_filter, mem_univ, true_and, Function.Embedding.subtype_apply]
  refine ⟨fun ⟨g', hg', h⟩ => h ▸ hg', fun hg => ⟨⟨g, ?_⟩, hg, rfl⟩⟩
  cases hi : C.isInput g
  · rfl
  · exact absurd ((P.input_iff g).2 hi) (by rw [hg]; exact Option.some_ne_none u)

/-- **A level's blocks are its partition's units**: as gates of `C`, the blocks of `P.blocks` are the nonempty
`P.gates u`. -/
theorem mem_blocks_gates (P : Partition C n) (b : Finset C.Computed) :
    b ∈ P.blocks.parts ↔ ∃ u, (P.gates u).Nonempty ∧ b.map (Function.Embedding.subtype _) = P.gates u := by
  rw [mem_blocks]
  constructor
  · rintro ⟨hb, u, rfl⟩
    exact ⟨u, P.map_block u ▸ hb.map, P.map_block u⟩
  · rintro ⟨u, hu, hb⟩
    rw [← P.map_block u] at hb hu
    obtain rfl := Finset.map_injective _ hb
    exact ⟨Finset.map_nonempty.1 hu, u, rfl⟩

end Partition

theorem Levels.ofPartitions_levels {C : Circuit} (ch : Partition.Chain C (Partition.whole C)) :
    (Levels.ofPartitions ch).levels = ch.levels := rfl

theorem Partition.Chain.levels_cons {C : Circuit} {nc nf : ℕ} {Pc : Partition C nc} (Pf : Partition C nf)
    (R : Pf.Refines Pc) (rule : Fin nc → SamplingDesign (Fin nf)) (rest : Partition.Chain C Pf) :
    (Partition.Chain.cons Pf R rule rest).levels = ⟨Pf.blocks, R.design rule⟩ :: rest.levels := rfl

end Verity
