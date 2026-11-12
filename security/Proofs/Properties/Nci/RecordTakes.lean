module

public import Properties.Nci.Record

/-!
# What the record's lemmas take of the Program and the wires

Statements about a decoded Program, the root's ranges and the wires `Spot.Wires.check` accepts. The landing bound and
the wired circuit take each as a hypothesis, and each is proved of the record check's own form on its own: that a gate
is as wide as the leaf that names it (`LeafWide`); that what a unit returns and what its gates read are the unit's
own gates and rows (`ReturnsOwn`, `OperandsOwn`); that a gate inside a unit is computed and every computed gate is
inside one (`UnitsComputed`, `GatesOwn`); that the wires rank the units (`Ranked`); that a wired row holds the output
it is wired to (`OutFits`); and that a unit's parameters have their own names (`PortsNamed`). `landing` is the gates
of the wired circuit on which a row's reads land.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row fieldsOf)

/-- **A gate is as wide as the leaf that names it**: a leaf of a reader row, and a leaf of a unit's output row, is
a gate whose value has the leaf's bits. -/
def LeafWide (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi →
      ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value (hi - lo)) ∧
    ∀ tid u sc out l g lo hi, UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi →
      ∃ p ops sc', P.gate g = .ok (p, ops, sc') ∧ (P.defs[p]!).ret = .value (hi - lo)

/-- **What a unit returns is its own**: a leaf of a unit's output row is a gate of the unit or a leaf of one of
the unit's reader rows. -/
def ReturnsOwn (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  ∀ tid u sc out l g lo hi, UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi →
    (sc.off ≤ g ∧ g < sc.off + (P.defs[sc.d]!).gates) ∨
      ∃ r l' lo' hi', r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l' g lo' hi'

/-- **What a gate of a unit reads is the unit's own**: an operand of a gate inside a unit is an earlier gate of
the unit or a leaf of one of the unit's reader rows. -/
def OperandsOwn (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  ∀ tid u sc g p ops sc', UnitAt P rs tid u sc → sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates →
    P.gate g = .ok (p, ops, sc') → ∀ o ∈ ops, (sc.off ≤ o ∧ o < g) ∨
      ∃ r l lo hi, r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l o lo hi

/-- **A gate inside a unit is computed.** -/
def UnitsComputed (P : Prog) (rs : Array TRange) : Prop :=
  ∀ tid u sc g p ops sc', UnitAt P rs tid u sc → sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates →
    P.gate g = .ok (p, ops, sc') → Flock.Program.isInputId (P.defs[p]!).key = false

/-- **Every gate has its place**: a gate of the Program holds one value; an input gate reads nothing; and a computed
gate lies in a unit. -/
def GatesOwn (P : Prog) (rs : Array TRange) : Prop :=
  ∀ g, g < gates P → ∃ p ops sc w, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value w ∧
    (Flock.Program.isInputId (P.defs[p]!).key = true → ops = #[]) ∧
    (Flock.Program.isInputId (P.defs[p]!).key = false →
      ∃ tid u sc', UnitAt P rs tid u sc' ∧ sc'.off ≤ g ∧ g < sc'.off + (P.defs[sc'.d]!).gates)

/-- **The wires rank the units**: a row's writer ranks below the row's unit. -/
def Ranked (S : Spot.Wires.Summary) (ρ : String × ℤ → ℕ) : Prop :=
  ∀ (r : Row) wt wu out, S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) →
    ρ (wt, wu) < ρ (r.tid, (r.unit : ℤ))

/-- **A wired row holds its output**: the output row a reader row is wired to has no more bits than the row. -/
def OutFits (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  ∀ (r : Row) wt wu out sc sc' i f, S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) → 0 ≤ wu →
    UnitAt P rs r.tid r.unit sc → UnitAt P rs wt wu.toNat sc' → i < (P.defs[sc.d]!).params.size →
    (P.defs[sc.d]!).params[i]!.1 = r.port → out = "o" ++ toString f →
    f < (fieldsOf (P.defs[sc'.d]!).ret).length →
    (outRow (P.defs[sc'.d]!) f).2.foldl (· + ·) 0 ≤
      (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.foldl (· + ·) 0

/-- **A unit's parameters have their own names.** -/
def PortsNamed (P : Prog) (rs : Array TRange) : Prop :=
  ∀ tid u sc i i', UnitAt P rs tid u sc → i < (P.defs[sc.d]!).params.size → i' < (P.defs[sc.d]!).params.size →
    (P.defs[sc.d]!).params[i]!.1 = (P.defs[sc.d]!).params[i']!.1 → i = i'

/-! ## Where a row's reads land -/

open Classical in
/-- The gates of `D` on which a read of a leaf of the row `r` lands. -/
noncomputable def landing (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    {D : Verity.Circuit} (at_ : Fin D.N → ℕ) (r : Row) : Finset (Fin D.N) :=
  Finset.univ.filter fun a => ∃ l o lo hi, RowLeaf P rs W r l o lo hi ∧ Lands P rs W S o (at_ a)

end NciCheck.Properties.Record.SecurityProofs
