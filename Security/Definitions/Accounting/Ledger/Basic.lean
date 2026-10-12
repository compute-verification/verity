import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Definitions.Accounting.Device

/-!
# The ledger (trusted)

The physical model of resource accounting (the console's draft, `internal/physical-model-and-pins.md`, 11 Oct) in its
smallest form.

* **The actual circuit** is everything the declared hardware computes over the period. Its **annotation** says which
  leaf of the hardware runs each gate and when, which leaf holds each value between uses, and which values move between
  leaves. Only its counts matter here (`Annotation`): the gates of each type a leaf runs at each tick, the bits it holds
  at the end of each tick, and the bits that move from one leaf to another at each tick. The world is a leaf like any
  other, outside every declared part.
* **Parts** are sets of leaves, nested or disjoint, as the hardware declaration names them (`Parts`): an SM, a GPU's
  cores and its HBM, a GPU, a server, a rack, a site.
* **A region** is a part over a set of ticks (`Region`): usually a window of wall-clock time (`Region.ofWindow`), and
  for an audit graded at its challenges, the challenge ticks.
* **The ledger** of a region (`Annotation.ledger`) has four entries (`Entry`): the gates of each type run in the part
  over the region's ticks; the bits the part holds, in bit-ticks (one bit held for one tick is one bit-tick, so a
  one-tick region's entry is the bits held at that moment); and the bits crossing the part's boundary inward and
  outward over the region's ticks. A ledger is a function of its entries, so ledgers are compared and added entry by
  entry.
* **A meter** is a weighted sum of entries (`Meter.eval`): one entry, or work at a price per gate type. Pins,
  requirements and capacities are read through a meter, so that "capacity minus pin" is one subtraction.

Nothing but the ledger reads the annotation's counts. What completeness, identity and latency are about (the content
that crosses a boundary, who answered, chains of dependent gates) is not in the counts; those assumptions take it as
their own parameters (`Definitions.Accounting.Ledger.Physical`).
-/

namespace Accounting.Ledger

/-- An entry of a region's ledger, with gate types `G`. -/
inductive Entry (G : Type)
  /-- the gates of type `g` run in the part over the region's ticks -/
  | gates (g : G)
  /-- the bits the part holds over the region's ticks, in bit-ticks -/
  | held
  /-- the bits crossing the part's boundary inward over the region's ticks -/
  | crossIn
  /-- the bits crossing the part's boundary outward over the region's ticks -/
  | crossOut
  deriving DecidableEq

end Accounting.Ledger

namespace Accounting

/-- A region's ledger, with gate types `G`: a count for each entry. Ledgers are ordered and added entry by entry. -/
abbrev Ledger (G : Type) : Type := Ledger.Entry G → ℕ

namespace Ledger

/-- A meter: a weight for each entry. A meter's value on a ledger is the weighted sum of its entries. -/
abbrev Meter (G : Type) : Type := Entry G → ℕ

variable {G : Type}

/-- A meter's value on a ledger: each entry times its weight, summed. -/
def Meter.eval [Fintype G] (m : Meter G) (L : Ledger G) : ℕ :=
  ∑ g, m (.gates g) * L (.gates g) + m .held * L .held + m .crossIn * L .crossIn + m .crossOut * L .crossOut

/-- The meter that reads one entry. -/
def Meter.entry [DecidableEq G] (e : Entry G) : Meter G := fun e' => if e' = e then 1 else 0

/-- Work at price `p`: the gates of each type, each at its price. -/
def Meter.priced (p : G → ℕ) : Meter G
  | .gates g => p g
  | _ => 0

/-- A meter reads no crossing: its value adds over disjoint parts. -/
def Meter.NoCrossing (m : Meter G) : Prop := m .crossIn = 0 ∧ m .crossOut = 0

/-- A nested partition: the parts the hardware declaration names, each a set of leaves, any two nested or disjoint. -/
structure Parts (Λ : Type) where
  /-- the declared parts -/
  parts : Set (Finset Λ)
  /-- any two declared parts are nested or disjoint -/
  nested : ∀ p ∈ parts, ∀ q ∈ parts, p ⊆ q ∨ q ⊆ p ∨ Disjoint p q

/-- A region: a part over a set of ticks. -/
structure Region (Λ : Type) where
  /-- the part, a set of leaves -/
  part : Finset Λ
  /-- the ticks, usually a window's -/
  ticks : Finset ℕ

/-- A part over a window of wall-clock time. -/
def Region.ofWindow {Λ : Type} (p : Finset Λ) (w : Window) : Region Λ := ⟨p, w.ticks⟩

/-- The counts of an annotated actual circuit over leaves `Λ`, with gate types `G`. -/
structure Annotation (Λ G : Type) where
  /-- the gates of type `g` leaf `l` runs at tick `t` -/
  gates : Λ → ℕ → G → ℕ
  /-- the bits leaf `l` holds at the end of tick `t` -/
  held : Λ → ℕ → ℕ
  /-- the bits that move from leaf `l` to leaf `l'` at tick `t` -/
  link : Λ → Λ → ℕ → ℕ

/-- The actual circuit made of shares `A a`, one for each account `a`: every count summed over the accounts. -/
def Annotation.sum {Λ α : Type} [Fintype α] (A : α → Annotation Λ G) : Annotation Λ G where
  gates l t g := ∑ a, (A a).gates l t g
  held l t := ∑ a, (A a).held l t
  link l l' t := ∑ a, (A a).link l l' t

variable {Λ : Type} [Fintype Λ] [DecidableEq Λ]

/-- A region's ledger: the gates its part runs, the bit-ticks it holds and the bits crossing its boundary each way,
summed over its ticks. A bit that moves between two leaves of the part crosses nothing. -/
def Annotation.ledger (A : Annotation Λ G) (ρ : Region Λ) : Ledger G
  | .gates g => ∑ t ∈ ρ.ticks, ∑ l ∈ ρ.part, A.gates l t g
  | .held => ∑ t ∈ ρ.ticks, ∑ l ∈ ρ.part, A.held l t
  | .crossIn => ∑ t ∈ ρ.ticks, ∑ l ∈ ρ.partᶜ, ∑ l' ∈ ρ.part, A.link l l' t
  | .crossOut => ∑ t ∈ ρ.ticks, ∑ l ∈ ρ.part, ∑ l' ∈ ρ.partᶜ, A.link l l' t

end Ledger

end Accounting
