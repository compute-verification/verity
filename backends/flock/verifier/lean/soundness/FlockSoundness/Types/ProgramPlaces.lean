import FlockSoundness.Types.Program

/-!
# Each drawn unit's place, for a program of units (verified-lowering phase 1, S4d)

`E2E.lean`'s `DerivedPlaces` asks, for every draw and every drawn unit, the unit's place in its table: its rows, its
instance in the committed circuit, a block and the columns there at which the table's statement holds the rows. For a
program of units (`Types.Program`) the rows and the instance are the program's own:
- `UProg.spec u` is unit `u`'s spec, `UProg.rowsOf u` its composed rows and `UProg.instOf u` their instance in `p.P`;
- `UProg.derivedPlaces` is `DerivedPlaces` from a block and columns per drawn unit at which its table holds its rows
  (`RowsAt`).

`TableClass` is what a table's accepted statement gives (1d and 1e): the verifier's derivation of its unit class and the
class's rows placed at each slot. `UProg.derivedPlaces_of_classes` places each drawn unit at a slot of its table's class,
when the unit reads the class's derivation inputs (`Reads`): `deriveChecked` is a function, so they have the same rows.
-/

open Flock Flock.DeriveAll

namespace FlockSoundness

open Audit Types.Dag Compose Model

/-- A block, and the columns there at which statement `St` holds rows `R`. -/
structure RowsAt (St : Statement) (R : Rows) where
  o : Fin (2 ^ St.nbl)
  col : Fin R.w → Fin (2 ^ St.kLog)
  placed : Placement St R col

/-- The same place, for equal rows. -/
def RowsAt.cast {St : Statement} {R R' : Rows} (e : R = R') (a : RowsAt St R) : RowsAt St R' := e ▸ a

/-- **A table's unit class, placed at each slot**: the verifier's run of `deriveChecked` on the class's layouts, and
statement `St`'s level-0 matrices holding the class's composed rows at each slot's columns. -/
structure TableClass (St : Statement) where
  types : String → Option CircuitType
  ls : Array Layout
  info : String → Option TableInfo
  words : String → Option (Array ℕ)
  unit : ℕ
  done : Array D
  hd : deriveChecked types ls info words unit = .ok done
  slots : ℕ
  col : Fin slots → Fin (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)).w →
    Fin (2 ^ St.kLog)
  placed : ∀ g, Placement St (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)) (col g)

namespace TableClass

variable {St : Statement} (c : TableClass St)

/-- The class's composed rows. -/
abbrev rows : Rows := Rows.compose (ofBlock c.words c.done (c.done.getD c.unit default)) (ofBlock_ordered c.hd)

/-- Slot `g` of block `o`. -/
def atSlot (g : Fin c.slots) (o : Fin (2 ^ St.nbl)) : RowsAt St c.rows := ⟨o, c.col g, c.placed g⟩

end TableClass

/-- A unit reads a table class's derivation inputs: its types, layouts, table info and words, and layout. -/
def UnitSpec.Reads {St : Statement} (s : UnitSpec) (c : TableClass St) : Prop :=
  s.types = c.types ∧ s.ls = c.ls ∧ s.info = c.info ∧ s.words = c.words ∧ s.idx = c.unit

theorem compose_congr {L L' : Logical} (e : L = L') (h : Ordered L) (h' : Ordered L') :
    Rows.compose L h = Rows.compose L' h' := by
  subst e; rfl

/-- **A unit that reads a class's inputs has the class's rows**: `deriveChecked` is a function. -/
theorem UnitSpec.R_eq_of_reads {St : Statement} {s : UnitSpec} {c : TableClass St} (h : s.Reads c) : s.R = c.rows := by
  obtain ⟨ht, hl, hi, hw, hu⟩ := h
  have hs := s.hd
  rw [ht, hl, hi, hw, hu, c.hd] at hs
  have hdn : s.done = c.done := (Except.ok.inj hs).symm
  exact compose_congr (by rw [hw, hdn, hu]) _ _

/-- A unit pair's spec. -/
def UnitPair.toSpec {C CB : Audit.Circuit} {n : ℕ} {P : Partition C n} {PB : Partition CB n} {m : Fin CB.N → Fin C.N}
    {u : Fin n} (U : UnitPair P PB m u) : UnitSpec :=
  ⟨U.types, U.ls, U.info, U.words, U.idx, U.done, U.hd, U.t, U.ht, U.loOf, U.F, U.hF⟩

namespace UProg

variable {k one : ℕ}

/-- Unit `u`'s spec. -/
noncomputable def spec (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) : UnitSpec :=
  (p.pairs hk u).toSpec

/-- Unit `u`'s composed rows. -/
noncomputable abbrev rowsOf (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) : Rows :=
  (p.spec hk u).R

/-- Their instance in the program's rows circuit. -/
noncomputable def instOf (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) :
    (p.P hk).IsRowsUnit u (p.rowsOf hk u) :=
  (p.pairs hk u).rows

theorem spec_last (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs)
    (inj : ∀ i < s.t.inBits, ∀ j < s.t.inBits, src i = src j → i = j) :
    (snoc p s src valid inj).spec hk (Fin.last p.n) = s := by
  unfold spec
  rw [pairs, Fin.lastCases_last]
  rfl

theorem spec_castSucc (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs)
    (inj : ∀ i < s.t.inBits, ∀ j < s.t.inBits, src i = src j → i = j) (v : Fin p.n) :
    (snoc p s src valid inj).spec hk v.castSucc = p.spec hk v := by
  unfold spec
  rw [pairs, Fin.lastCases_castSucc]
  rfl

variable {Reg : Type} {mPts : ℕ}

/-- **Each drawn unit's place, from its rows' place in its table.** -/
noncomputable def derivedPlaces (hk : one < k) {outs : List ℕ} (p : UProg k one outs)
    (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (at_ : ∀ S R u, u ∈ S → RowsAt (Partition.tabOf plan tab S R u).S (p.rowsOf hk u)) :
    (p.P hk).DerivedPlaces plan tab (p.rowsOf hk) where
  place S R u h := ⟨p.rowsOf hk u, p.instOf hk u, (at_ S R u h).o, (at_ S R u h).col, (at_ S R u h).placed⟩
  rows _ _ _ _ := rfl

/-- **Each drawn unit's place, from its table's class**: unit `u`, drawn in `S` under `R`, at slot `slot` of block `o`
of its table's class `cls`, whose derivation inputs it reads. -/
noncomputable def derivedPlaces_of_classes (hk : one < k) {outs : List ℕ} (p : UProg k one outs)
    (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (cls : ∀ S R u, u ∈ S → TableClass (Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (o : ∀ S R u, u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ S R u (h : u ∈ S), (p.spec hk u).Reads (cls S R u h)) :
    (p.P hk).DerivedPlaces plan tab (p.rowsOf hk) :=
  p.derivedPlaces hk plan tab fun S R u h =>
    ((cls S R u h).atSlot (slot S R u h) (o S R u h)).cast (UnitSpec.R_eq_of_reads (hr S R u h)).symm

end UProg

end FlockSoundness
