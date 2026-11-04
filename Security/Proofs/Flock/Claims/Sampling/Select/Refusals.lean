import Proofs.Flock.Claims.Sampling.Select.Scope
import Proofs.Flock.Claims.Sampling.Select.Layout

/-!
# The refusals the readable scope needs (`Claims.Sampling.ScopeMatches`, `Refuses`)

Where the scope check is the verifier's own evaluation of the statement (`QueryCheck`: it takes a statement exactly when
the statement parses and its query evaluates), the readable scope's program part (`ProgramScope`) is one refusal per
clause of it, each a named Prop over the statements the verifier evaluates:

* `RefusesUnordered`: the program is topologically ordered.
* `RefusesOverlap`: no gate is in two proof units.
* `RefusesUncovered`: the proof units' gates are the program's, and every gate but a constant is among them.
* `RefusesEmpty`: every proof unit has a gate.
* `RefusesWide`: every proof unit's committed values (the claims' `committed`, on its own gates) fit its public size
  class (`unitClass`).

The five give `ProgramScope` (`programScope_of_refusals`), and so, on partitions of the shape of the verifier's own
layouts (`DrawSlots`), `ScopeMatches` (`scopeMatches_of_refusals`) and, with `ChecksScope`, `Refuses`
(`refuses_of_refusals`). Each is needed: `ScopeMatches` at such a scope check gives all five
(`refusals_of_scopeMatches`, `scopeMatches_iff_refusals`).
-/

open SampledProofs

namespace Claims.Sampling

open Properties.SampledProofs Properties.SampledProofs.SecurityProofs

variable {Coin Col T : Type}

/-- The proof units' gates are `C`'s, and every gate of `C` but a constant is among them. -/
def Partition.Covers (P : Partition) (C : Prog) : Prop :=
  (∀ g ∈ P.gates, g < C.gates.length) ∧ ∀ g < C.gates.length, (∀ b, C.gates[g]? ≠ some (.const b)) → g ∈ P.gates

/-- **A cut of `C` into proof units**: no gate in two of `P`'s proof units, which cover `C` but its constants, each with
a gate, and each committing values that fit its public size class. -/
def Partition.Cuts (P : Partition) (C : Prog) : Prop :=
  P.gates.Nodup ∧ P.Covers C ∧ (∀ u ∈ P.blocks, u ≠ []) ∧
    ∀ u ∈ P.blocks, ∃ w, unitClass C u = some w ∧ committed C u ≤ w

/-- A partition of `C` cuts it into proof units. -/
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

/-- **The verifier refuses overlapping proof units**: no gate is in two proof units of a partition it evaluates. -/
def RefusesOverlap (D : Deploy Coin Col T) : Prop := Evaluated D fun _ P => P.gates.Nodup

/-- **The verifier refuses a partition that leaves a gate out**: the proof units' gates are the program's, and every
gate but a constant is among them (a constant, public, may be left out). -/
def RefusesUncovered (D : Deploy Coin Col T) : Prop := Evaluated D fun C P => P.Covers C

/-- **The verifier refuses an empty proof unit**: every proof unit of a partition it evaluates has a gate. -/
def RefusesEmpty (D : Deploy Coin Col T) : Prop := Evaluated D fun _ P => ∀ u ∈ P.blocks, u ≠ []

/-- **The verifier refuses a proof unit wider than its size class**: every proof unit of a partition it evaluates has a
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
