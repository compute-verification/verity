module

public import Proofs.Flock.Claims.Sampling.Select.Layout
public import Proofs.Flock.Claims.Sampling.Select.Scope
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import VCVio.Prelude.Core

/-! The refusals the readable scope needs (`Claims.Sampling.ScopeMatches`, `Refuses`), one per clause of its program part. -/

@[expose] public section

open Spot

namespace Claims.Sampling

open Properties.Spot Properties.Spot.SecurityProofs

variable {Coin Col T : Type}

def Partition.Covers (P : Partition) (C : Prog) : Prop :=
  (∀ g ∈ P.gates, g < C.gates.length) ∧
    ∀ g < C.gates.length, (∀ b, C.gates[g]? ≠ some (.const b)) → ¬ C.Unread g → g ∈ P.gates

def Partition.Cuts (P : Partition) (C : Prog) : Prop :=
  P.gates.Nodup ∧ P.Covers C ∧ (∀ u ∈ P.blocks, u ≠ []) ∧
    ∀ u ∈ P.blocks, ∃ w, sizeClass C u = some w ∧ committed C u ≤ w

theorem Partition.Partitions.cuts {P : Partition} {C : Prog} (h : P.Partitions C) : P.Cuts C :=
  ⟨h.1, ⟨h.2.1, h.2.2.1⟩, h.2.2.2.1, h.2.2.2.2⟩

theorem partitions_iff_cuts (P : Partition) (C : Prog) : P.Partitions C ↔ P.Cuts C :=
  ⟨Partition.Partitions.cuts, fun ⟨hnd, ⟨hin, hcov⟩, hne, hw⟩ => ⟨hnd, hin, hcov, hne, hw⟩⟩

def QueryCheck (D : Deploy Coin Col T) : Prop :=
  ∀ x, D.scopeCheck x = true ↔ ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt

def Evaluated (D : Deploy Coin Col T) (Q : Prog → Partition → Prop) : Prop :=
  ∀ x s Pt, D.parse x = some s → D.partition s = some Pt → Q s.prog Pt

def RefusesUnordered (D : Deploy Coin Col T) : Prop := Evaluated D fun C _ => C.Topological

def RefusesOverlap (D : Deploy Coin Col T) : Prop := Evaluated D fun _ P => P.gates.Nodup

def RefusesUncovered (D : Deploy Coin Col T) : Prop := Evaluated D fun C P => P.Covers C

def RefusesEmpty (D : Deploy Coin Col T) : Prop := Evaluated D fun _ P => ∀ u ∈ P.blocks, u ≠ []

def RefusesWide (D : Deploy Coin Col T) : Prop :=
  Evaluated D fun C P => ∀ u ∈ P.blocks, ∃ w, sizeClass C u = some w ∧ committed C u ≤ w

theorem programScope_of_refusals (D : Deploy Coin Col T) (hq : QueryCheck D) (h₁ : RefusesUnordered D)
    (h₂ : RefusesOverlap D) (h₃ : RefusesUncovered D) (h₄ : RefusesEmpty D) (h₅ : RefusesWide D) : ProgramScope D := by
  intro x
  rw [hq x]
  constructor
  · rintro ⟨s, Pt, hx, hP⟩
    exact ⟨s, _, hx, hP, h₁ x s _ hx hP,
      (partitions_iff_cuts _ _).2 ⟨h₂ x s _ hx hP, h₃ x s _ hx hP, h₄ x s _ hx hP, h₅ x s _ hx hP⟩⟩
  · rintro ⟨s, Pt, hx, hP, -, -⟩
    exact ⟨s, Pt, hx, hP⟩

theorem scopeMatches_of_refusals [Fintype Coin] [Nonempty Coin] (D : Deploy Coin Col T) (hslots : DrawSlots D)
    (hq : QueryCheck D) (h₁ : RefusesUnordered D) (h₂ : RefusesOverlap D) (h₃ : RefusesUncovered D)
    (h₄ : RefusesEmpty D) (h₅ : RefusesWide D) : ScopeMatches D :=
  scopeMatches_of D hslots (programScope_of_refusals D hq h₁ h₂ h₃ h₄ h₅)

theorem refuses_of_refusals [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T)
    (hslots : DrawSlots D) (hq : QueryCheck D) (h₁ : RefusesUnordered D) (h₂ : RefusesOverlap D)
    (h₃ : RefusesUncovered D) (h₄ : RefusesEmpty D) (h₅ : RefusesWide D) (hchecks : ChecksScope H D) :
    Refuses H D :=
  refuses_of H D (scopeMatches_of_refusals D hslots hq h₁ h₂ h₃ h₄ h₅) hchecks

theorem refusals_of_scopeMatches [Fintype Coin] (D : Deploy Coin Col T) (hq : QueryCheck D) (h : ScopeMatches D) :
    RefusesUnordered D ∧ RefusesOverlap D ∧ RefusesUncovered D ∧ RefusesEmpty D ∧ RefusesWide D := by
  have key : ∀ x s Pt, D.parse x = some s → D.partition s = some Pt →
      s.prog.Topological ∧ Pt.Cuts s.prog := by
    intro x s Pt hx hP
    obtain ⟨s', Pt', hx', hP', htop, hpart, -⟩ := (h x).2 ((hq x).2 ⟨s, Pt, hx, hP⟩)
    obtain rfl := Option.some.inj (hx'.symm.trans hx)
    obtain rfl := Option.some.inj (hP'.symm.trans hP)
    exact ⟨htop, hpart.cuts⟩
  exact ⟨fun x s Pt hx hP => (key x s Pt hx hP).1, fun x s Pt hx hP => (key x s Pt hx hP).2.1,
    fun x s Pt hx hP => (key x s Pt hx hP).2.2.1, fun x s Pt hx hP => (key x s Pt hx hP).2.2.2.1,
    fun x s Pt hx hP => (key x s Pt hx hP).2.2.2.2⟩

theorem scopeMatches_iff_refusals [Fintype Coin] [Nonempty Coin] (D : Deploy Coin Col T) (hslots : DrawSlots D)
    (hq : QueryCheck D) :
    ScopeMatches D ↔
      RefusesUnordered D ∧ RefusesOverlap D ∧ RefusesUncovered D ∧ RefusesEmpty D ∧ RefusesWide D :=
  ⟨refusals_of_scopeMatches D hq, fun ⟨h₁, h₂, h₃, h₄, h₅⟩ => scopeMatches_of_refusals D hslots hq h₁ h₂ h₃ h₄ h₅⟩

end Claims.Sampling
