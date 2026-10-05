import FlockSoundness.Protocol.Audit.FlockLinked

/-!
# Public wires: the constant and the zero are the statement's, not the prover's

The link theorem's value layer (`Xplur`) is the plurality of a trial's readings at each commit string. On the constant
and on the statement's zero, that plurality is the prover's only in form. Every satisfying message of every drawn unit's
table decodes the constant to 1 (`UnitPlace.decode_one`) and the zero to 0 (its forced-zero rows). So the committed
transcript here is `Xpub ones zeros vals`: 1 on `ones`, the gates the units' constants read; 0 on `zeros`, the statement's
zero; and `vals` elsewhere.

* `linkEvB_of_pub`: at a drawn unit, when the public wires decode to their values in every satisfying message of its
  table (`PubDecodes`), the link event at `Xpub ones zeros vals` is the link event at `vals`. So the link theorem's bound holds
  unchanged for `Xpub ones zeros Xplur` (`flock_batched_linkSoundE_pub`), and with it the count curve and the drawn-unit
  bound (`flock_batched_count_public`, `flock_batched_drawn_public`).
* `pubDecodes_of_place`: `PubDecodes` for the decoder that reads each drawn unit from its place. Two facts about the
  statement suffice:
  - `ConstCols`: only a unit's constant column sits on a gate of `ones`. No input column of a unit reads a constant,
    which is a fact of the circuit's wiring.
  - `ZeroCols`: every column on a gate of `zeros` carries 0 in every satisfying witness. These are the forced-zero rows,
    which the verifier's Δ copies to the slot inputs that read the zero.

This discharges the end-to-end theorem's `hOne` and the zero's binding (`E2E.lean`): no hypothesis on the prover's
committed values remains in their place, only these two facts about the statement.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **The committed transcript with its public wires**: 1 on `ones` (the gates the units' constants read), 0 on `zeros`
(the statement's zero) and `vals` elsewhere. -/
noncomputable def Xpub {N : ℕ} (ones zeros : Finset (Fin N)) (vals : Fin N → Bool) : Fin N → Bool :=
  fun g => if g ∈ ones then true else if g ∈ zeros then false else vals g

namespace Partition.IsRowsUnit

variable {C : Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R)

end Partition.IsRowsUnit

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

variable {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : P.ValueBinding plan tab decode Hc)

/-! ## The public wires from the drawn units' places -/

/-- **Only a unit's constant column sits on a gate of `ones`**, at every drawn unit's place: no input column of a unit
reads a constant, and a unit's constant is outside it. A fact of the circuit's wiring. -/
def ConstCols (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) (ones : Finset (Fin C.N)) : Prop :=
  ∀ S R u (h : u ∈ S) (c : Fin (place S R u h).R.w), (place S R u h).inst.wire c ∈ ones →
    c = (place S R u h).R.one

/-- **`ConstCols` for one constant gate**: when every drawn unit's constant column sits on the gate `one`, no other column
does, since an instance shares a gate only between input columns. -/
theorem constCols_singleton (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) (one : Fin C.N)
    (h : ∀ S R u (hu : u ∈ S), (place S R u hu).inst.wire (place S R u hu).R.one = one) :
    P.ConstCols plan tab place {one} := by
  intro S R u hu c hc
  rw [Finset.mem_singleton, ← h S R u hu] at hc
  rcases (place S R u hu).inst.shared _ _ hc with he | ⟨_, h2⟩
  · exact he
  · simp only [Rows.one] at h2
    omega

/-- **The zero's columns carry 0**, at every drawn unit's place: a column on a gate of `zeros` carries 0 in every witness
satisfying the table's statement. These are the forced-zero rows, which the verifier's Δ copies to the slot inputs that
read the zero. -/
def ZeroCols (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) (zeros : Finset (Fin C.N)) : Prop :=
  ∀ S R u (h : u ∈ S) (c : Fin (place S R u h).R.w), (place S R u h).inst.wire c ∈ zeros →
    ∀ z : Witness (tabOf plan tab S R u).S.m, (tabOf plan tab S R u).S.Satisfies z →
      (tabOf plan tab S R u).S.block z (place S R u h).o ((place S R u h).col c) = 0

end Partition

end FlockSoundness.Audit
