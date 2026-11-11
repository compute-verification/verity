module

public import Proofs.Flock.SpotFlock.Reveals

/-!
# What SPOT's reveal opens

Each opening a unit's reveal makes is at a gate its proof block needs opened (`pos_mem_needs`), so every position a
run's extractions open lies in `opensSet`, the gates some proof block of the partition needs opened: a set the
statement fixes before the run. It is the `Spos` of `Claims.SamplingGeneric.Compiled.pr_regHit_le_canon_pos`.
-/

@[expose] public section

namespace SpotFlock.Reveal

open Claims Claims.Sampling SamplingDeploy

variable (part : AuditStatement → Option Partition) (two : AuditStatement → Option Design.Two)
  (ir : Seen → ℕ → List Msg) (op : AuditStatement → ℕ → Seen → Claims.Flock.Wit → ℕ → List Msg → Opening Bool)

/-- **The positions a run can open** at `s`: every gate some proof block of the partition needs opened. -/
def opensSet (s : AuditStatement) : Finset ℕ :=
  (Finset.range (((part s).map fun P => P.blocks.length).getD 0)).biUnion fun u => (needs part s u).toFinset

theorem pos_mem_needs (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    ∀ p ∈ reveal part two ir op s u v wF, p.2.pos ∈ needs part s u := by
  intro p hp
  obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hp
  exact hg

theorem pos_mem_opensSet (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    ∀ p ∈ reveal part two ir op s u v wF, p.2.pos ∈ opensSet part s := by
  intro p hp
  have hn := pos_mem_needs part two ir op s u v wF p hp
  refine Finset.mem_biUnion.2 ⟨u, ?_, List.mem_toFinset.2 hn⟩
  rw [Finset.mem_range]
  unfold needs at hn
  cases hP : part s with
  | none => simp [hP] at hn
  | some P =>
    simp only [hP, Option.map_some, Option.getD_some] at hn ⊢
    cases hb : P.blocks[u]? with
    | none => simp [hb] at hn
    | some b => exact (List.getElem?_eq_some_iff.1 hb).1

/-- `opensSet` has at most the blocks' needs, summed. -/
theorem card_opensSet_le (s : AuditStatement) :
    (opensSet part s).card ≤
      ∑ u ∈ Finset.range (((part s).map fun P => P.blocks.length).getD 0), (needs part s u).length :=
  Finset.card_biUnion_le.trans (Finset.sum_le_sum fun u _ => List.toFinset_card_le _)

end SpotFlock.Reveal
