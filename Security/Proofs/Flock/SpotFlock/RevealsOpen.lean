module

public import Proofs.Flock.SpotFlock.Reveals
public import Proofs.Flock.SpotFlock.Code

/-!
# What SPOT's reveal opens

Each opening a unit's reveal makes is at a gate its proof block needs opened (`pos_mem_needs`), so every position a
run's extractions open lies in `opensSet`, the gates some proof block of the partition needs opened: a set the
statement fixes before the run. It is the `Spos` of `Claims.SamplingGeneric.Compiled.pr_regHit_le_canon_pos`, and
`code_reveal_pos` states it at the deployed code's reveal.
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

namespace SpotFlock

variable (h : Held)

/-- **The deployed code's reveal opens only `opensSet`**: every opening SPOT[Flock]'s two-stage code reveals for a unit
is at a gate some proof block of the partition needs opened. -/
theorem code_reveal_pos (s : AuditStatement) (u : ℕ) (v : Seen) (w : Claims.Flock.Wit) :
    ∀ p ∈ (code h).reveal s u v w, p.2.pos ∈ Reveal.opensSet (partition h) s :=
  Reveal.pos_mem_opensSet (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u v w

end SpotFlock
