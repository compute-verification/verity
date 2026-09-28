import Proofs.Flock.Soundness.Binding.Layout
import Proofs.Flock.Soundness.Audit.FlockLinked

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

/-- **Each committed wire of a unit is on a column of its instance**: the unit's own gates are its computed rows'
columns, and the gates they read are columns of those rows. -/
theorem exists_col_of_io {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R) {g : Fin C.N} (hg : g ∈ P.io u) :
    ∃ c, I.wire c = g := by
  classical
  simp only [Partition.io, Finset.mem_filter] at hg
  rcases hg.2 with hu | ⟨h, hh, hgh⟩
  · obtain ⟨i, hi⟩ := I.onto g hu
    exact ⟨_, hi⟩
  · obtain ⟨i, rfl⟩ := I.onto h hh
    rw [I.op i] at hgh
    simp only [Op.args, List.mem_append, List.mem_map] at hgh
    rcases hgh with ⟨c, -, rfl⟩ | ⟨c, -, rfl⟩
    · exact ⟨c, rfl⟩
    · exact ⟨c, rfl⟩

/-- **A committed wire outside a unit is read by one of its rows**: on a column some computed row of the instance
reads. -/
theorem exists_read_of_io {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R) {g : Fin C.N} (hg : g ∈ P.io u)
    (hout : P.unit g ≠ some u) : ∃ i, ∃ c ∈ R.a i ++ R.b i, I.wire c = g := by
  classical
  simp only [Partition.io, Finset.mem_filter] at hg
  rcases hg.2 with hu | ⟨h, hh, hgh⟩
  · exact absurd hu hout
  · obtain ⟨i, rfl⟩ := I.onto h hh
    rw [I.op i] at hgh
    simp only [Op.args, List.mem_append, List.mem_map] at hgh
    rcases hgh with ⟨c, hc, rfl⟩ | ⟨c, hc, rfl⟩
    · exact ⟨i, c, List.mem_append_left _ hc, rfl⟩
    · exact ⟨i, c, List.mem_append_right _ hc, rfl⟩

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

/-- A bit string of width `W` read from `k ≤ W` bits, `false` past them. -/
def padBits {W k : ℕ} (b : Fin k → Bool) : Fin W → Bool := fun i => if h : i.val < k then b ⟨i.val, h⟩ else false

/-- **A value of at most `W` bits, with its width**: `sha512/row/v1` hashes a row's word count with its bytes, so the
value carries it. -/
abbrev BitVal (W : ℕ) : Type := Fin (W + 1) × (Fin W → Bool)

/-- Width `k`, capped at `W`. -/
def widthOf (W k : ℕ) : Fin (W + 1) := ⟨min k W, Nat.lt_succ_of_le (Nat.min_le_right k W)⟩

/-- **A layout of bit-string values**, from the decoder that reads each drawn unit from its place. Each commit string
`q` holds a value of `rowBits q ≤ W` bits with that width (`BitVal`), read at the fixed addresses `rowAt` (its salt at
`saltAt`), and wire `g` is bit `idx g` of its commit string's value. `copy`, the one fact: in a satisfying witness, each
column of a drawn unit on a committed wire `g` holds the witness's bit at `rowAt` of `g`'s commit string and bit. -/
noncomputable def ofBits (Pos : Type) [Fintype Pos] [DecidableEq Pos] [Inhabited Pos] (W : ℕ) (pos : Fin C.N → Pos)
    (idx : Fin C.N → Fin W) (regLeaf : Reg → Pos → Refine.Dig) (rowBits : Pos → ℕ)
    (hidx : ∀ g, (idx g).val < rowBits (pos g))
    (rowAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → (q : Pos) → Fin (rowBits q) →
      Fin (2 ^ (Partition.tabOf plan tab S R u).S.m))
    (saltAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Pos → Fin 192 → Fin 8 →
      Fin (2 ^ (Partition.tabOf plan tab S R u).S.m))
    (copy : ∀ S R u (hu : u ∈ S) M, Binding.Sat plan tab S R u M → ∀ c, (place S R u hu).inst.wire c ∈ P.io u →
      (Partition.tabOf plan tab S R u).S.block (Binding.wit plan tab S R u M) (place S R u hu).o ((place S R u hu).col c) =
        Binding.wit plan tab S R u M (rowAt S R u (pos ((place S R u hu).inst.wire c))
          ⟨idx ((place S R u hu).inst.wire c), hidx _⟩)) :
    Binding.Layout P plan tab (P.placeDecoder plan tab place) where
  Pos := Pos
  Val := BitVal W
  pos := pos
  bit v g := v.2 (idx g)
  regLeaf := regLeaf
  rowBits := rowBits
  rowOf q b := (widthOf W (rowBits q), padBits b)
  rowAt := rowAt
  saltAt := saltAt
  decode_row := by
    refine decode_row_of_cols place pos (fun v g => v.2 (idx g))
      (fun S R u M q => (widthOf W (rowBits q), padBits (bitsAt (Binding.wit plan tab S R u M) (rowAt S R u q)))) ?_
    intro S R u hu M hM c hc
    simp only [padBits, hidx, ↓reduceDIte, bitsAt]
    rw [copy S R u hu M hM c hc]

end FlockSoundness.Discharge.Layout
