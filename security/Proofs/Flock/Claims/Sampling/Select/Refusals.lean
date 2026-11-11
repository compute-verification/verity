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

/-!
# The refusals the readable scope needs (`Claims.Sampling.ScopeMatches`, `Refuses`)

Where the scope check is the verifier's own evaluation of the statement (`QueryCheck`: it takes a statement exactly when
the statement parses and its query evaluates), the readable scope's program part (`ProgramScope`) is one refusal per
clause of it, each a named Prop over the statements the verifier evaluates:

* `RefusesUnordered`: the program is topologically ordered.
* `RefusesOverlap`: no gate is in two proof blocks.
* `RefusesUncovered`: the proof blocks' gates are the program's, and every gate but a constant and an unread input is
  among them.
* `RefusesEmpty`: every proof block has a gate.
* `RefusesWide`: every proof block's committed values (the claims' `committed`, on its own gates) fit its public size
  class (`unitClass`).

The five give `ProgramScope` (`programScope_of_refusals`), and so, on partitions of the shape of the verifier's own
layouts (`DrawSlots`), `ScopeMatches` (`scopeMatches_of_refusals`) and, with `ChecksScope`, `Refuses`
(`refuses_of_refusals`). Each is needed: `ScopeMatches` at such a scope check gives all five
(`refusals_of_scopeMatches`, `scopeMatches_iff_refusals`).
-/

@[expose] public section

open Spot

namespace Claims.Sampling

open Properties.Spot Properties.Spot.SecurityProofs

variable {Coin Col T : Type}

/-- The proof blocks' gates are `C`'s, and every gate of `C` but a constant and an unread input is among them. -/
def Partition.Covers (P : Partition) (C : Prog) : Prop :=
  (∀ g ∈ P.gates, g < C.gates.length) ∧
    ∀ g < C.gates.length, (∀ b, C.gates[g]? ≠ some (.const b)) → ¬ C.Unread g → g ∈ P.gates

/-- **A cut of `C` into proof blocks**: no gate in two of `P`'s proof blocks, which cover `C` but its constants and
unread inputs, each with a gate, and each committing values that fit its public size class. -/
def Partition.Cuts (P : Partition) (C : Prog) : Prop :=
  P.gates.Nodup ∧ P.Covers C ∧ (∀ u ∈ P.blocks, u ≠ []) ∧
    ∀ u ∈ P.blocks, ∃ w, unitClass C u = some w ∧ committed C u ≤ w

/-- A partition of `C` cuts it into proof blocks. -/
theorem Partition.Partitions.cuts {P : Partition} {C : Prog} (h : P.Partitions C) : P.Cuts C :=
  ⟨h.1, ⟨h.2.1, h.2.2.1⟩, h.2.2.2.1, h.2.2.2.2⟩

/-- **A partition partitions `C` exactly when it cuts it.** -/
theorem partitions_iff_cuts (P : Partition) (C : Prog) : P.Partitions C ↔ P.Cuts C :=
  ⟨Partition.Partitions.cuts, fun ⟨hnd, ⟨hin, hcov⟩, hne, hw⟩ => ⟨hnd, hin, hcov, hne, hw⟩⟩

/-- **The scope check is the verifier's evaluation**: it takes a statement exactly when the statement parses and its
query evaluates. -/
def QueryCheck (D : Deploy Coin Col T) : Prop :=
  ∀ x, D.scopeCheck x = true ↔ ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt

/-- `Q` holds of every statement the verifier evaluates: its program and the partition its query evaluates to. -/
def Evaluated (D : Deploy Coin Col T) (Q : Prog → Partition → Prop) : Prop :=
  ∀ x s Pt, D.parse x = some s → D.partition s = some Pt → Q s.prog Pt

/-- **The verifier refuses a program out of order**: every program it evaluates is topologically ordered. -/
def RefusesUnordered (D : Deploy Coin Col T) : Prop := Evaluated D fun C _ => C.Topological

/-- **The verifier refuses overlapping proof blocks**: no gate is in two proof blocks of a partition it evaluates. -/
def RefusesOverlap (D : Deploy Coin Col T) : Prop := Evaluated D fun _ P => P.gates.Nodup

/-- **The verifier refuses a partition that leaves a gate out**: the proof blocks' gates are the program's, and every
gate but a constant and an unread input is among them (a constant, public, may be left out, and so may an input that no
gate reads and the program doesn't return). -/
def RefusesUncovered (D : Deploy Coin Col T) : Prop := Evaluated D fun C P => P.Covers C

/-- **The verifier refuses an empty proof block**: every proof block of a partition it evaluates has a gate. -/
def RefusesEmpty (D : Deploy Coin Col T) : Prop := Evaluated D fun _ P => ∀ u ∈ P.blocks, u ≠ []

/-- **The verifier refuses a proof block wider than its size class**: every proof block of a partition it evaluates has a
public size class that holds its committed values (`unitClass`; `committed`, `Prog.Boundary` on the unit's own gates). -/
def RefusesWide (D : Deploy Coin Col T) : Prop :=
  Evaluated D fun C P => ∀ u ∈ P.blocks, ∃ w, unitClass C u = some w ∧ committed C u ≤ w

/-- **The program part from the five refusals**: where the scope check is the verifier's evaluation (`QueryCheck`), the
scope check is the readable scope's program part. -/
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

/-- **`ScopeMatches` from the five refusals** (`programScope_of_refusals`, `scopeMatches_of`). -/
theorem scopeMatches_of_refusals [Fintype Coin] [Nonempty Coin] (D : Deploy Coin Col T) (hslots : DrawSlots D)
    (hq : QueryCheck D) (h₁ : RefusesUnordered D) (h₂ : RefusesOverlap D) (h₃ : RefusesUncovered D)
    (h₄ : RefusesEmpty D) (h₅ : RefusesWide D) : ScopeMatches D :=
  scopeMatches_of D hslots (programScope_of_refusals D hq h₁ h₂ h₃ h₄ h₅)

/-- **`Refuses` from the five refusals** and the verifier running its scope check (`ChecksScope`, `refuses_of`). -/
theorem refuses_of_refusals [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T)
    (hslots : DrawSlots D) (hq : QueryCheck D) (h₁ : RefusesUnordered D) (h₂ : RefusesOverlap D)
    (h₃ : RefusesUncovered D) (h₄ : RefusesEmpty D) (h₅ : RefusesWide D) (hchecks : ChecksScope H D) :
    Refuses H D :=
  refuses_of H D (scopeMatches_of_refusals D hslots hq h₁ h₂ h₃ h₄ h₅) hchecks

/-- **Each refusal is needed**: where the scope check is the verifier's evaluation (`QueryCheck`), `ScopeMatches` holds
only if the verifier refuses all five. -/
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

/-- **The readable scope is the verifier's exactly when it refuses the five**: on its own partitions (`DrawSlots`) and
where the scope check is its evaluation (`QueryCheck`). -/
theorem scopeMatches_iff_refusals [Fintype Coin] [Nonempty Coin] (D : Deploy Coin Col T) (hslots : DrawSlots D)
    (hq : QueryCheck D) :
    ScopeMatches D ↔
      RefusesUnordered D ∧ RefusesOverlap D ∧ RefusesUncovered D ∧ RefusesEmpty D ∧ RefusesWide D :=
  ⟨refusals_of_scopeMatches D hq, fun ⟨h₁, h₂, h₃, h₄, h₅⟩ => scopeMatches_of_refusals D hslots hq h₁ h₂ h₃ h₄ h₅⟩

end Claims.Sampling
