module

public import Properties.InferenceOnly.Record

/-! The statements about a decoded Program, the root's ranges and the wires that the landing bound and the wired circuit
take as hypotheses. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope fieldsOf)
open Flock.TemplateQuery (TRange)
open Spot.RecordText (Row)

def LeafWide (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi →
      ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value (hi - lo)) ∧
    ∀ tid u sc out l g lo hi, UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi →
      ∃ p ops sc', P.gate g = .ok (p, ops, sc') ∧ (P.defs[p]!).ret = .value (hi - lo)

def ReturnsOwn (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  ∀ tid u sc out l g lo hi, UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi →
    (sc.off ≤ g ∧ g < sc.off + (P.defs[sc.d]!).gates) ∨
      ∃ r l' lo' hi', r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l' g lo' hi'

def OperandsOwn (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  ∀ tid u sc g p ops sc', UnitAt P rs tid u sc → sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates →
    P.gate g = .ok (p, ops, sc') → ∀ o ∈ ops, (sc.off ≤ o ∧ o < g) ∨
      ∃ r l lo hi, r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l o lo hi

def UnitsComputed (P : Prog) (rs : Array TRange) : Prop :=
  ∀ tid u sc g p ops sc', UnitAt P rs tid u sc → sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates →
    P.gate g = .ok (p, ops, sc') → Flock.Program.isInputId (P.defs[p]!).key = false

def GatesOwn (P : Prog) (rs : Array TRange) : Prop :=
  ∀ g, g < gates P → ∃ p ops sc w, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value w ∧
    (Flock.Program.isInputId (P.defs[p]!).key = true → ops = #[]) ∧
    (Flock.Program.isInputId (P.defs[p]!).key = false →
      ∃ tid u sc', UnitAt P rs tid u sc' ∧ sc'.off ≤ g ∧ g < sc'.off + (P.defs[sc'.d]!).gates)

def Ranked (S : Spot.Wires.Summary) (ρ : String × ℤ → ℕ) : Prop :=
  ∀ (r : Row) wt wu out, S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) →
    ρ (wt, wu) < ρ (r.tid, (r.unit : ℤ))

def OutFits (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  ∀ (r : Row) wt wu out sc sc' i f, S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) → 0 ≤ wu →
    UnitAt P rs r.tid r.unit sc → UnitAt P rs wt wu.toNat sc' → i < (P.defs[sc.d]!).params.size →
    (P.defs[sc.d]!).params[i]!.1 = r.port → out = "o" ++ toString f →
    f < (fieldsOf (P.defs[sc'.d]!).ret).length →
    (outRow (P.defs[sc'.d]!) f).2.foldl (· + ·) 0 ≤
      (inRow (P.defs[sc.d]!) i (W.cutsOf r.tid r.port) r.row).2.foldl (· + ·) 0

def PortsNamed (P : Prog) (rs : Array TRange) : Prop :=
  ∀ tid u sc i i', UnitAt P rs tid u sc → i < (P.defs[sc.d]!).params.size → i' < (P.defs[sc.d]!).params.size →
    (P.defs[sc.d]!).params[i]!.1 = (P.defs[sc.d]!).params[i']!.1 → i = i'

open Classical in
noncomputable def landing (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    {D : Verity.Circuit} (at_ : Fin D.N → ℕ) (r : Row) : Finset (Fin D.N) :=
  Finset.univ.filter fun a => ∃ l o lo hi, RowLeaf P rs W r l o lo hi ∧ Lands P rs W S o (at_ a)

end InferenceOnlyCheck.Properties.Record.SecurityProofs
