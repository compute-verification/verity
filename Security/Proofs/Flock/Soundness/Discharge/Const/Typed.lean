import Proofs.Flock.Soundness.Discharge.Placed.Typed
import Proofs.Flock.Soundness.Discharge.Const.ConstCols

/-!
# `hConst` on the typed path

At a program of units' places from its tables' classes (`Placed.UProg.progPlaces_of_classes`, and
`progPlaces_of_setupH` with each class from `setupH_flatTableClass`), each unit's instance is the program's own
(`UProg.instOf`, carried to the rows circuit's rows by `Placed.UProg.instRows`), which wires the constant column to the
constant input (`UProg.pairs_oneGate`). So only a unit's constant column sits on `oneOf hk`, as main's
`UProg.constCols_of_classes` says at `derivedPlaces_of_classes`.
-/

namespace FlockSoundness.Discharge.Const.UProg

open FlockSoundness Audit

theorem castInst_wire_one {C : Audit.Circuit} {m : ℕ} {P : Audit.Partition C m} {v : Fin m} {R₁ R₂ : Rows}
    (e : R₁ = R₂) (i : P.IsRowsUnit v R₁) : (Placed.castInst e i).wire R₂.one = i.wire R₁.one := by
  subst e
  rfl

variable {k one : ℕ} (hk : one < k) {outs : List ℕ} (p : FlockSoundness.UProg k one outs)

/-- **The rows circuit's instance of a unit wires its constant to the constant input.** -/
theorem instRows_wire_one (u : Fin p.n) :
    (Placed.UProg.instRows hk p u).wire ((p.toRows hk).rows u).one = p.oneOf hk :=
  (castInst_wire_one (P := (p.toRows hk).partition p.outsC) (Placed.UProg.rows_toRows hk p u).symm
    (p.instOf hk u)).trans (Fin.ext (p.pairs_oneGate hk u))

variable {Reg : Type} {mPts : ℕ}

/-- **`hConst` at a program of units' places from its tables' classes**: only a unit's constant column sits on the
constant input. -/
theorem constCols_progPlaces_of_classes (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (cls : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S → TableClass (Partition.tabOf plan tab S R u).S)
    (slot : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), Fin (cls S R u h).slots)
    (o : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), (p.spec hk u).Reads (cls S R u h))
    (hcp : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u)) :
    ((p.toRows hk).partition p.outsC).ConstCols plan tab
      (Placed.UProg.progPlaces_of_classes hk p plan tab cls slot o hr hcp).derived.place {p.oneOf hk} :=
  constCols_of_oneWire _ _ (instRows_wire_one hk p)

/-- **`hConst` at a program of units' places from setupH's accept on the typed path.** -/
theorem constCols_progPlaces_of_setupH (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (acc : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S →
      Placed.TypedAccepted (Partition.tabOf plan tab S R u).S)
    (slot : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), Fin (acc S R u h).cls.slots)
    (o : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), (p.spec hk u).Reads (acc S R u h).cls)
    (hcp : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S),
      (acc S R u h).cls.Copies (slot S R u h) (p.srcOf u)) :
    ((p.toRows hk).partition p.outsC).ConstCols plan tab
      (Placed.UProg.progPlaces_of_setupH hk p plan tab acc slot o hr hcp).derived.place {p.oneOf hk} :=
  constCols_of_oneWire _ _ (instRows_wire_one hk p)

end FlockSoundness.Discharge.Const.UProg
