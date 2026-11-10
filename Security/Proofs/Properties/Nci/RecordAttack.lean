module

public import Proofs.Properties.Nci.Record

/-!
# Instances against the circuit record's properties

A scratch file of instances, never merged: what the statements of `Properties.Nci.Record` say and leave out.

* **A split leaf is refused.** Under `Aligned` a writer's leaf is no wider than the reader's leaf it lands in
  (`aligned_width`), so a wire whose writer has a leaf wider than every leaf of the row that reads it is not `Aligned`
  (`not_aligned_of_wider`), and no circuit is `Served`.
* **A wire from a writer the Program has no unit for.** `Served` holds the wiring's templates to the Program by their
  ids alone, and `Spot.Wires.check` bounds a wire's writer by the units the wiring declares. Where a row has a writer
  and that writer is no unit of the Program, or has no such output row, the row's gates are wired and a read of one
  lands on no gate (`lands_nowhere`): the wired circuit reads nothing for it.
* **A held row that has a writer** puts no gate in θ (`held_wired_names_nothing`): the wired circuit has none of its
  gates. `Good` asks that an opening's root and domain rows be held, and never that they have no writer.
* **The pins and the record's text.** `Carries`, the four premises of the rate rule and `Complies` are the same
  statements whatever pins the check reports (`carries_repin`, `owns_repin`, `complies_repin`), and `Served` is the
  same whatever record text it reports (`served_retext`).
* **The premises on the check hold of a check that reads nothing** (`refuseAll_premises`): they are consistent, and
  that an accepting check meets them is not shown here.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)
open Flock.Program (Prog)
open Flock.TemplateQuery (TRange)

/-! ### Wires -/

/-- Under `Aligned` a writer's leaf is no wider than the reader's leaf it lands in. -/
theorem aligned_width {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    (h : Aligned P rs W S) {r : Row} {l' g' lo' hi' : ℕ} (hw : WriterLeaf P rs S r l' g' lo' hi') :
    ∃ l g lo hi, RowLeaf P rs W r l g lo hi ∧ hi' - lo' ≤ hi - lo := by
  obtain ⟨l, g, lo, hi, hleaf, h1, h2⟩ := h r l' g' lo' hi' hw
  exact ⟨l, g, lo, hi, hleaf, by omega⟩

/-- A wire whose writer has a leaf wider than every leaf of the row that reads it is not `Aligned`. -/
theorem not_aligned_of_wider {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {r : Row} {l' g' lo' hi' w : ℕ} (hw : WriterLeaf P rs S r l' g' lo' hi')
    (hrow : ∀ l g lo hi, RowLeaf P rs W r l g lo hi → hi - lo ≤ w) (hwide : w < hi' - lo') :
    ¬ Aligned P rs W S := by
  intro h
  obtain ⟨l, g, lo, hi, hleaf, hle⟩ := aligned_width h hw
  have := hrow l g lo hi hleaf
  omega

/-- How a read lands: on the gate itself where no wire covers it, else through a gate that feeds it. -/
theorem lands_inv {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary} {o a : ℕ}
    (h : Lands P rs W S o a) :
    (¬ Wired P rs W S o ∧ a = o) ∨ ∃ g, Feeds P rs W S g o ∧ Lands P rs W S g a := by
  cases h with
  | here _ hnw => exact Or.inl ⟨hnw, rfl⟩
  | wire _ g _ hf hl => exact Or.inr ⟨g, hf, hl⟩

/-- **A wire from a writer the Program has no unit for.** A leaf of a row that has a writer, where no gate is a leaf
of that writer's output row: a read of it lands on no gate. -/
theorem lands_nowhere {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary} {r : Row}
    {l g lo hi : ℕ} (hown : OwnRows P rs W) (hleaf : RowLeaf P rs W r l g lo hi)
    (hw : (S.writerOf r.tid r.unit r.port r.row).isSome)
    (hno : ∀ l' g' lo' hi', ¬ WriterLeaf P rs S r l' g' lo' hi') (a : ℕ) : ¬ Lands P rs W S g a := by
  intro h
  rcases lands_inv h with ⟨hnw, -⟩ | ⟨g', hf, -⟩
  · exact hnw ⟨r, l, lo, hi, hw, hleaf⟩
  · obtain ⟨r', l1, l', lo1, hi1, lo', hi', hleaf', hwl, -, -⟩ := hf
    obtain ⟨hr, -⟩ := hown.2 r l lo hi r' l1 lo1 hi1 g hleaf hleaf'
    rw [← hr] at hwl
    exact hno l' g' lo' hi' hwl

/-- **A held row that has a writer puts no gate in θ**: its gates are wired, and the wired circuit has none of
them. -/
theorem held_wired_names_nothing {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {D : Verity.Circuit} {at_ : Fin D.N → ℕ} (hlay : LaysOut P rs W S D at_) {r : Row}
    (hw : (S.writerOf r.tid r.unit r.port r.row).isSome) (x : Fin D.N) (l lo hi : ℕ) :
    ¬ RowLeaf P rs W r l (at_ x) lo hi := fun h =>
  ((hlay.2.1 (at_ x)).1 ⟨x, rfl⟩).2 ⟨r, l, lo, hi, hw, h⟩

/-! ### The pins and the record's text -/

/-- The check `F`, reporting the pins `ps`. -/
def repin (F : Form) (ps : F.Case → List String) : Form :=
  { F with inst := fun c => { F.inst c with pins := ps c } }

/-- The check `F`, reporting the record text `s`. -/
def retext (F : Form) (s : F.Case → String) : Form :=
  { F with inst := fun c => { F.inst c with record := s c } }

/-- `Carries` holds no pin to the instance. -/
theorem carries_repin (F : Form) (ps : F.Case → List String) (j : Json) (c : F.Case) :
    Carries (repin F ps) j c ↔ Carries F j c := Iff.rfl

theorem owns_repin (F : Form) (ps : F.Case → List String) : Owns (repin F ps) ↔ Owns F := Iff.rfl

theorem laidOut_repin (F : Form) (ps : F.Case → List String) : LaidOut (repin F ps) ↔ LaidOut F := Iff.rfl

theorem counts_repin (F : Form) (ps : F.Case → List String) : Counts (repin F ps) ↔ Counts F := Iff.rfl

theorem rates_repin (F : Form) (ps : F.Case → List String) : Rates (repin F ps) ↔ Rates F := Iff.rfl

theorem complies_repin (F : Form) (ps : F.Case → List String) : Complies (repin F ps) ↔ Complies F := Iff.rfl

/-- `Served` reads no record text. -/
theorem served_retext (F : Form) (s : F.Case → String) (c : F.Case) (d : Verity.RegisteredCircuit) :
    Served (retext F s) c d ↔ Served F c d := Iff.rfl

theorem counts_retext (F : Form) (s : F.Case → String) : Counts (retext F s) ↔ Counts F := Iff.rfl

/-! ### A check that reads nothing -/

/-- A check that refuses every instance. -/
def refuseAll : Form where
  Case := Unit
  read := fun _ _ => .error "refused"
  inst := fun _ => { program := "", record := "", gamma := 0, num := 0, den := 1, pins := [] }
  wiring := fun _ => { templates := [], cuts := [], wires := [] }
  slot := fun _ _ _ => none
  held := fun _ _ => none
  summarize := fun _ => { gates := 0, fresh := 0 }
  rate := fun _ _ _ _ => some "refused"
  layout := fun _ => none

theorem refuseAll_read {j tr : Json} {c : refuseAll.Case} (h : refuseAll.read j tr = .ok c) : False := by
  have h' : (Except.error "refused" : Except String Unit) = Except.ok c := h
  cases h'

theorem refuseAll_rate {Γ num den : ℕ} {s : Summary} (h : refuseAll.rate Γ num den s = none) : False := by
  have h' : (some "refused" : Option String) = none := h
  cases h'

/-- **The premises on the check hold of a check that reads nothing.** -/
theorem refuseAll_premises :
    Owns refuseAll ∧ LaidOut refuseAll ∧ Counts refuseAll ∧ Rates refuseAll ∧ Provenance refuseAll ∧
      LayoutReads refuseAll :=
  ⟨fun _ _ _ h => (refuseAll_read h).elim, fun _ _ _ h => (refuseAll_read h).elim,
    fun _ _ _ h => (refuseAll_read h).elim, fun _ _ _ _ h => (refuseAll_rate h).elim,
    fun _ _ _ _ _ h => (refuseAll_read h).elim, fun _ _ _ _ => rfl⟩

end NciCheck.Properties.Record.Attack
