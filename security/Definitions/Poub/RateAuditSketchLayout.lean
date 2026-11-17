module

public import Definitions.Poub.RateAuditSketch
import Mathlib.Basic.ENNReal.Real

/-! The sketch rate audit for a prover whose memory holds any function of the store. -/

@[expose] public section

namespace ResourceAccounting.RateSketch

open Pous (Bits)

def Data.store {N m ρ : ℕ} {X : Type} (D : Data N m ρ X) : Fin N × Fin m → X := fun p => D (.inl p)

inductive LayoutChip (N ρ h : ℕ) (X Dg : Type) (α : Type) : Type where
  | done : α → LayoutChip N ρ h X Dg α
  | tag : (Fin ρ → X) × Fin N × Dg → (Bits h → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α
  | read : ℕ → (Bool → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α
  | link : Fin ρ → (X → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α

namespace LayoutChip

variable {N ρ h : ℕ} {X Dg α : Type}

def run (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) : LayoutChip N ρ h X Dg α → α
  | done a => a
  | tag x k => (k (O x)).run O M R
  | read a k => (k (M a)).run O M R
  | link r k => (k (R r)).run O M R

def queries (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) :
    LayoutChip N ρ h X Dg α → List ((Fin ρ → X) × Fin N × Dg)
  | done _ => []
  | tag x k => x :: (k (O x)).queries O M R
  | read a k => (k (M a)).queries O M R
  | link r k => (k (R r)).queries O M R

def reads (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) : LayoutChip N ρ h X Dg α → ℕ
  | done _ => 0
  | tag x k => (k (O x)).reads O M R
  | read a k => (k (M a)).reads O M R + 1
  | link r k => (k (R r)).reads O M R

def linkSteps (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) : LayoutChip N ρ h X Dg α → ℕ
  | done _ => 0
  | tag x k => (k (O x)).linkSteps O M R
  | read a k => (k (M a)).linkSteps O M R
  | link r k => (k (R r)).linkSteps O M R + 1

end LayoutChip

structure LayoutDevice (N m ρ h : ℕ) (X Sig Dg : Type) where
  mem : (Fin N × Fin m → X) → ℕ → Bool
  chip : Sig → LayoutChip N ρ h X Dg (Fin N → Bits h)

namespace LayoutDevice

variable {N m ρ h : ℕ} {X Sig Dg : Type}

def tags (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Fin N → Bits h :=
  (A.chip σ).run O (A.mem D.store) D.stream

def queries (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    List ((Fin ρ → X) × Fin N × Dg) :=
  (A.chip σ).queries O (A.mem D.store) D.stream

def reads (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : ℕ :=
  (A.chip σ).reads O (A.mem D.store) D.stream

def linkSteps (A : LayoutDevice N m ρ h X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : ℕ :=
  (A.chip σ).linkSteps O (A.mem D.store) D.stream

end LayoutDevice

end ResourceAccounting.RateSketch
