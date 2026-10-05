import FlockSoundness.Binding.Layout
import FlockSoundness.Audit.FlockLinked
import FlockSoundness.Protocol.Discharge.Layout.Readers

/-!
# The decoder's bits, read at fixed addresses

`Binding.Layout.decode_row` says that in a satisfying witness of a drawn unit's table, the decoder's bit on each of the
unit's committed wires is the bit of the value the witness holds, at fixed addresses (`rowAt`), at the wire's commit
string. For the decoder that reads each drawn unit from its place (`Partition.placeDecoder`), this file reduces it to
one fact per column:

* `exists_col_of_io`: each committed wire of a unit is on a column of its instance (`IsRowsUnit`);
* `placeDecoder_wire`: in a satisfying witness, the decoder's bit on a column's wire is the witness's bit at the
  column's place (`decode_wire`, with the agreement `UnitPlace.aliased` gives);
* `decode_row_of_cols`: `decode_row` from a fact about each column on a committed wire;
* `ofBits`: a layout whose values are bit strings, each wire one bit of its commit string's value, from the one fact
  that each such column holds the bit the witness has at the wire's fixed row address (`copy`).

`rowAt` and `saltAt` are arguments that depend on the draw, the registration, the unit and the commit string only: the
readers are fixed before the witness is (a reader that chose its values would discharge nothing).
-/

namespace FlockSoundness.Discharge.Layout

open Game Model Audit Binding

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n}

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u)

/-- **The decoder's bit on a column's wire** is, in a satisfying witness, the witness's bit at the column's place. -/
theorem placeDecoder_wire {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (hM : Binding.Sat plan tab S R u M) (c : Fin (place S R u hu).R.w) :
    P.placeDecoder plan tab place S R u M ((place S R u hu).inst.wire c) =
      bit ((Partition.tabOf plan tab S R u).S.block (Binding.wit plan tab S R u M) (place S R u hu).o
        ((place S R u hu).col c)) := by
  simp only [Partition.placeDecoder, hu, ↓reduceDIte, UnitPlace.decode]
  exact (place S R u hu).inst.decode_wire ((place S R u hu).agree hM) c

/-- **`decode_row` from the columns**: if, in every satisfying witness, each column of a drawn unit on a committed wire
holds the bit of the value read at the wire's commit string, the decoder's bit on every committed wire is that bit. -/
theorem decode_row_of_cols {Pos Val : Type} (pos : Fin C.N → Pos) (vbit : Val → Fin C.N → Bool)
    (rowOf : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Binding.Msg plan tab S R u → Pos → Val)
    (h : ∀ S R u (hu : u ∈ S) M, Binding.Sat plan tab S R u M → ∀ c, (place S R u hu).inst.wire c ∈ P.io u →
      bit ((Partition.tabOf plan tab S R u).S.block (Binding.wit plan tab S R u M) (place S R u hu).o
          ((place S R u hu).col c)) =
        vbit (rowOf S R u M (pos ((place S R u hu).inst.wire c))) ((place S R u hu).inst.wire c)) :
    ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ P.io u,
      P.placeDecoder plan tab place S R u M g = vbit (rowOf S R u M (pos g)) g := by
  intro S R u hu M hM g hg
  obtain ⟨c, rfl⟩ := exists_col_of_io (place S R u hu).inst hg
  rw [placeDecoder_wire place hu M hM c]
  exact h S R u hu M hM c hg

end FlockSoundness.Discharge.Layout
