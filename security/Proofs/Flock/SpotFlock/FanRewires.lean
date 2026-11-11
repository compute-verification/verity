module

public import Proofs.Flock.SpotFlock.AddressesWired
public import Proofs.Flock.SamplingGeneric.WiredScope

/-!
# A connected instance of the select's shape meets the wired program's condition

`fanProg` (`AddressesWired`) is three members in one replay block: two writers of one output each and a reader whose
first and third inputs are wired to the first writer's output and whose second is wired to the second's. It has the
two things the served select's wiring has and a one-to-one wiring lacks: one written gate read at two gates of one
proof block, and several writers read by one reader.

* `touches_mem_of`: the gates a proof block holds or reads are one member's, decided on its gates and their reads.
* `fanProg_rewires`: **the wires rewire the program** (`Prog.Rewires`, by `Prog.Rewires.of_earlier`), so the wired
  program is in the two-stage scope wherever the members' programs are (`Prog.Rewires.inScopeTwo`) and a proof block
  correct side by side is correct, cut, at the wired program on values that agree where it reads one gate twice
  (`Prog.Rewires.exists_correct`).
* `fanProg_not_injective`: the wires send two gates to one, so no condition that asks the map to be one to one
  holds here.
-/

@[expose] public section

namespace Claims.Sampling

/-- **The gates a proof block holds or reads are member `t`'s**, where its own gates are and every gate one of them
reads is: `Prog.Touches` read off the block's gates and their reads. -/
theorem touches_mem_of {C : Prog} {u : List ℕ} {mem : ℕ → ℕ} {t : ℕ} (h1 : ∀ i ∈ u, mem i = t)
    (h2 : ∀ g ∈ u, ∀ i ∈ ((C.gates[g]?).map BGate.reads).getD [], mem i = t) :
    ∀ i, C.Touches u i → mem i = t := by
  intro i hi
  rcases hi with hi | ⟨g, hg, x, hx, hi⟩
  · exact h1 i hi
  · refine h2 g hg i ?_
    rw [hx]
    exact hi

end Claims.Sampling

namespace SpotFlock

open Claims Claims.Sampling

/-- The member a gate of `fanProg` is in: the first writer's gates `0, 1`, the second's `2, 3`, the reader's the
rest. -/
def fanMember (i : ℕ) : ℕ := if i < 2 then 0 else if i < 4 then 1 else 2

/-- The wires move the reader's three inputs and no other gate. -/
theorem fanWire_moves {i : ℕ} (h : fanWire i ≠ i) : i = 4 ∨ i = 5 ∨ i = 6 := by
  unfold fanWire at h
  by_cases h4 : i = 4
  · exact Or.inl h4
  by_cases h5 : i = 5
  · exact Or.inr (Or.inl h5)
  by_cases h6 : i = 6
  · exact Or.inr (Or.inr h6)
  rw [if_neg h4, if_neg h5, if_neg h6] at h
  exact (h rfl).elim

/-- **The wires rewire the three members' programs**, each member one proof block: `Prog.Rewires.of_earlier` at the
members' order. -/
theorem fanProg_rewires : fanProg.Rewires ⟨[[[[0, 1], [2, 3], [4, 5, 6, 7]]]]⟩ fanWire fanKeep := by
  refine Prog.Rewires.of_earlier (mem := fanMember) ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro i h
    rcases fanWire_moves h with rfl | rfl | rfl <;> decide
  · intro i h
    rcases fanWire_moves h with rfl | rfl | rfl <;> exact ⟨by decide, fun c => by cases c <;> decide⟩
  · intro i h
    rcases fanWire_moves h with rfl | rfl | rfl <;> decide
  · intro i j h
    unfold fanMember at h
    split_ifs at h <;> omega
  · intro i h
    rcases fanWire_moves h with rfl | rfl | rfl <;> decide
  · intro i h
    rcases fanWire_moves h with rfl | rfl | rfl <;> decide
  · intro u hu
    have hu' : u ∈ ([[0, 1], [2, 3], [4, 5, 6, 7]] : List (List ℕ)) := hu
    rcases List.mem_cons.1 hu' with rfl | hu'
    · exact ⟨⟨1, by decide, by decide⟩, 0, touches_mem_of (by decide) (by decide)⟩
    rcases List.mem_cons.1 hu' with rfl | hu'
    · exact ⟨⟨3, by decide, by decide⟩, 1, touches_mem_of (by decide) (by decide)⟩
    · obtain rfl := List.mem_singleton.1 hu'
      exact ⟨⟨7, by decide, by decide⟩, 2, touches_mem_of (by decide) (by decide)⟩

/-- **The wires send two gates to one**: the reader's first and third inputs both go to the first writer's
output. -/
theorem fanProg_not_injective : fanWire 4 = fanWire 6 ∧ (4 : ℕ) ≠ 6 := by decide

end SpotFlock
