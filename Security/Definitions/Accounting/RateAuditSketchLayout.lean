import Definitions.Accounting.RateAuditSketch

/-!
# The sketch rate audit for a prover whose memory holds any function of the store (trusted)

`RateAuditSketch.lean`'s window, tag and digest, for a prover outside its verbatim class. What the prover holds in
memory is laid out before the signs as any function of the store, as `RateAudit.lean`'s device memory is, and the chip
reads it one bit at an address. That is the remote audit's prover under conservation: one
undifferentiated resource, every tier of whose memory is memory here, every device of which that computes is the chip,
and every bit of which it reads is counted. The class here also holds nothing on chip, takes in nothing over the side
link but the certified stream's sectors, and rebuilds nothing, so an arbitrary prover's class contains it. Its chip
queries the oracle `H(R, t, d)` at the window's fresh seed, as `RateAuditSketch.lean`'s does.
-/

namespace Accounting.RateSketch

open Pous (Bits)

/-- The store: the data's `N·m` sectors, without the certified stream. -/
def Data.store {N m ρ : ℕ} {X : Type} (D : Data N m ρ X) : Fin N × Fin m → X := fun p => D (.inl p)

/-- A chip program of a prover whose memory holds any function of the store, ending in a value of type `α`. Each step
waits on one answer: a tag from the oracle, one bit of memory at an address, or one sector of the certified stream over
the side link. -/
inductive LayoutChip (N ρ h : ℕ) (X Dg : Type) (α : Type) : Type where
  | done : α → LayoutChip N ρ h X Dg α
  | tag : (Fin ρ → X) × Fin N × Dg → (Bits h → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α
  | read : ℕ → (Bool → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α
  | link : Fin ρ → (X → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α

namespace LayoutChip

variable {N ρ h : ℕ} {X Dg α : Type}

/-- The program's value under oracle `O`, memory `M` and certified stream `R`. -/
def run (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) : LayoutChip N ρ h X Dg α → α
  | done a => a
  | tag x k => (k (O x)).run O M R
  | read a k => (k (M a)).run O M R
  | link r k => (k (R r)).run O M R

/-- The oracle queries, in order. -/
def queries (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) :
    LayoutChip N ρ h X Dg α → List ((Fin ρ → X) × Fin N × Dg)
  | done _ => []
  | tag x k => x :: (k (O x)).queries O M R
  | read a k => (k (M a)).queries O M R
  | link r k => (k (R r)).queries O M R

/-- The bits read from memory: one per `read` step. -/
def reads (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) : LayoutChip N ρ h X Dg α → ℕ
  | done _ => 0
  | tag x k => (k (O x)).reads O M R
  | read a k => (k (M a)).reads O M R + 1
  | link r k => (k (R r)).reads O M R

/-- The sectors taken in over the side link: one per `link` step. -/
def linkSteps (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) : LayoutChip N ρ h X Dg α → ℕ
  | done _ => 0
  | tag x k => (k (O x)).linkSteps O M R
  | read a k => (k (M a)).linkSteps O M R
  | link r k => (k (R r)).linkSteps O M R + 1

end LayoutChip

/-- A prover whose memory is laid out before the signs as any function of the store, and its chip program from the
signs. -/
structure LayoutDevice (N m ρ h : ℕ) (X Sig Dg : Type) where
  mem : (Fin N × Fin m → X) → ℕ → Bool
  chip : Sig → LayoutChip N ρ h X Dg (Fin N → Bits h)

namespace LayoutDevice

variable {N m ρ h : ℕ} {X Sig Dg : Type}

/-- The tags the device commits to at signs `σ`, oracle `O` and data `D`. -/
def tags (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Fin N → Bits h :=
  (A.chip σ).run O (A.mem D.store) D.stream

/-- The oracle queries it makes. -/
def queries (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    List ((Fin ρ → X) × Fin N × Dg) :=
  (A.chip σ).queries O (A.mem D.store) D.stream

/-- The bits it reads from memory in the window: the reads conservation counts. -/
def reads (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : ℕ :=
  (A.chip σ).reads O (A.mem D.store) D.stream

/-- The sectors it takes in over the side link. -/
def linkSteps (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : ℕ :=
  (A.chip σ).linkSteps O (A.mem D.store) D.stream

end LayoutDevice

end Accounting.RateSketch
