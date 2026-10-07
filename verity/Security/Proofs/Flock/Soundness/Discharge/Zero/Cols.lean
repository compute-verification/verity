import Proofs.Flock.Soundness.E2E
import Proofs.Flock.Soundness.Types.ProgramPlaces

/-!
# The zero's columns are input columns (`hZero`, e2e)

`ZeroCols place zeros` reads every column of a drawn unit's place that sits on a gate of `zeros`. When those gates are no
unit's (the statement's zero is a program input) and no unit's constant column sits on one, only input columns do: a
computed column is a gate of the unit itself (`IsRowsUnit.mem`).

- `zeroCols_of_inputs`: `ZeroCols` from its input columns, at any drawn units' places.
- `Prog.zeroCols_of_inputs`: the same at the places `ProgPlaces` gives the program's units, the `hZero` that
  `Prog.flock_headline` takes.
- `TableClass.block_eq_zero`: at a table class's slot, an input whose copy position is one of the class's forced-zero rows
  carries 0, in every block of a satisfying witness.
-/

namespace FlockSoundness.Discharge.Zero

open Audit Model

/-- **A column on a gate that is no unit's, other than the constant's, is an input column**: a computed column is a gate
of the unit. -/
theorem lt_nIn_of_unit_none {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (I : P.IsRowsUnit u R) {c : Fin R.w} (hu : P.unit (I.wire c) = none) (h1 : c ≠ R.one) : c.val < R.nIn := by
  by_contra hc
  have hw : c.val < R.nIn + R.nComp + 1 := c.isLt
  have h2 : c.val ≠ R.nIn + R.nComp := fun he => h1 (Fin.ext he)
  have hci : R.comp ⟨c.val - R.nIn, by omega⟩ = c :=
    Fin.ext (show R.nIn + (c.val - R.nIn) = c.val by omega)
  have hm := I.mem ⟨c.val - R.nIn, by omega⟩
  rw [hci, hu] at hm
  cases hm

section Partition

variable {C : Audit.Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **`ZeroCols` from the input columns**: when the gates of `zeros` are no unit's and no drawn unit's constant column
sits on one, the columns on them are input columns, so it suffices that those carry 0. -/
theorem zeroCols_of_inputs (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u)
    (zeros : Finset (Fin C.N)) (hzu : ∀ g ∈ zeros, P.unit g = none)
    (hone : ∀ S R u (h : u ∈ S), (place S R u h).inst.wire (place S R u h).R.one ∉ zeros)
    (hin : ∀ S R u (h : u ∈ S) (c : Fin (place S R u h).R.w), c.val < (place S R u h).R.nIn →
      (place S R u h).inst.wire c ∈ zeros → ∀ z : Witness (Partition.tabOf plan tab S R u).S.m,
        (Partition.tabOf plan tab S R u).S.Satisfies z →
        (Partition.tabOf plan tab S R u).S.block z (place S R u h).o ((place S R u h).col c) = 0) :
    P.ZeroCols plan tab place zeros := by
  intro S R u h c hc
  refine hin S R u h c (lt_nIn_of_unit_none (place S R u h).inst (hzu _ hc) ?_) hc
  rintro rfl
  exact hone S R u h hc

end Partition

section Prog

variable {N n : ℕ} {p : Prog N n} {outs : Finset (Fin N)} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}

/-- **`hZero` at the program's places from its input columns**: the gates of `zeros` are program inputs (no unit's), no
unit's constant column sits on one, and each drawn unit's input column on one carries 0 in every satisfying witness. -/
theorem Prog.zeroCols_of_inputs (pp : ProgPlaces p outs plan tab) (zeros : Finset (Fin N))
    (hzu : ∀ g ∈ zeros, p.unit g = none) (hone : ∀ u, (pp.inst u).wire (p.rows u).one ∉ zeros)
    (hin : ∀ S R u (h : u ∈ S) (c : Fin (p.rows u).w), c.val < (p.rows u).nIn → (pp.inst u).wire c ∈ zeros →
      ∀ z : Witness (Partition.tabOf plan tab S R u).S.m, (Partition.tabOf plan tab S R u).S.Satisfies z →
        (Partition.tabOf plan tab S R u).S.block z (pp.o S R u h) (pp.col S R u h c) = 0) :
    (p.partition outs).ZeroCols plan tab pp.derived.place zeros :=
  Zero.zeroCols_of_inputs (p.partition outs) plan tab _ zeros hzu (fun _ _ u _ => hone u) hin

/-- **`hZero` for one zero gate**: `zer` is a program input, every unit's constant column sits on a gate `one ≠ zer`, and
each drawn unit's input column on `zer` carries 0 in every satisfying witness. -/
theorem Prog.zeroCols_singleton (pp : ProgPlaces p outs plan tab) {zer one : Fin N} (hzu : p.unit zer = none)
    (hz1 : zer ≠ one) (hone : ∀ u, (pp.inst u).wire (p.rows u).one = one)
    (hin : ∀ S R u (h : u ∈ S) (c : Fin (p.rows u).w), c.val < (p.rows u).nIn → (pp.inst u).wire c = zer →
      ∀ z : Witness (Partition.tabOf plan tab S R u).S.m, (Partition.tabOf plan tab S R u).S.Satisfies z →
        (Partition.tabOf plan tab S R u).S.block z (pp.o S R u h) (pp.col S R u h c) = 0) :
    (p.partition outs).ZeroCols plan tab pp.derived.place {zer} :=
  Prog.zeroCols_of_inputs pp {zer} (fun _ hg => (Finset.mem_singleton.1 hg) ▸ hzu)
    (fun u h => hz1 ((Finset.mem_singleton.1 h).symm.trans (hone u)))
    (fun S R u h c hc hw => hin S R u h c hc (Finset.mem_singleton.1 hw))

end Prog

/-- **An input copying a forced-zero row carries 0**: at slot `g` of a table class, an input whose copy position is one
of the class's forced-zero rows carries 0 in every block of a satisfying witness. -/
theorem TableClass.block_eq_zero {St : Statement} (cls : TableClass St) (g : Fin cls.slots) (c : Fin cls.rows.w)
    (hc : c.val < cls.rows.nIn) (hz : cls.copyPos g c.val ∈ cls.zeros) {z : Witness St.m} (hsat : St.Satisfies z)
    (o : Fin (2 ^ St.nbl)) : St.block z o (cls.col g c) = 0 := by
  rw [cls.copy g c hc z hsat o]
  exact cls.zero _ hz z hsat o

end FlockSoundness.Discharge.Zero
