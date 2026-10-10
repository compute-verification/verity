import Proofs.Flock.Sampled.Defs

/-!
# F2's statements: sampling with prover-chosen batching, the partition hidden

Skeletons, stated and not yet proved; `Proofs.Flock.Sampled.NonVacuity` shows each one's hypotheses hold together.

* `Coverage`: the outer proof's coverage check and slot binding's conclusion put each drawn unit in exactly one slot, of
  its committed type, on its committed values, against the committed hidden partition; and the auditor, which checks
  only the sessions' padded shapes, has room for the draw.
* `SlotBinding`: each held slot's opening of the trace root binds it to its unit's committed record, or the run's two
  openings give a SHA-512 collision.
* `ShapeMenu`, `Menu.Room`, `MenuComplete`: the menu's shapes are staged by Lean's `VStar.statements` and its padded
  sessions take only them; and when the padded sessions have room for any `k` typed units, an honest prover can batch
  every draw.
* `SlotsSound`: the statement the sessions prove puts every drawn unit's committed record in its type's relation, or
  slot binding breaks.
* `Composition`: with F1's conclusion (`Recursion`, an interface), the audit accepts with the committed trace's wrong
  units in any family `𝓑` with probability at most the law's escape over `𝓑` plus F1's error, or a run breaks.
-/

namespace FlockSampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

/-- **Coverage.** When the outer proof's coverage check passes on a batching whose held slots are on their units'
committed types and values, each drawn unit sits in exactly one slot, of its committed type, on its committed values;
and the sessions' slots, which the auditor counts from their public shapes alone, number at least the draw. -/
def Coverage : Prop :=
  ∀ {N : ℕ} {T V : Type} (X : Trace N T V) (S : Finset (Fin N)) (β : Batching N T V),
    CoverCheck S β → Bound X β → Covers X S β ∧ S.card ≤ (β.view.map Shape.block).sum

/-- **Slot binding.** Under a commitment whose records determine their contents, with `X` the trace it commits: when
every held slot's record opens the root at its unit's position, every held slot is on its unit's committed type and
values, or some held slot's opening and the committed trace's give a SHA-512 collision (`SlotBreak`). -/
def SlotBinding : Prop :=
  ∀ {N : ℕ} {T V : Type} (C : Commit N T V) (X : Trace N T V) (path₀ : Fin N → ByteArray) (h : Hidden N T V),
    C.Injective → C.Committed X path₀ → C.SlotOpens h → Bound X h.β ∨ SlotBreak C X path₀ h

/-- **The shape menu.** V* stages every shape on the menu (`VStar.statements` at its inner shape is a result), and an
audit's padded sessions take only shapes on the menu. -/
structure ShapeMenu (M : Menu) : Prop where
  staged : ∀ s ∈ M.shapes, ∃ vs, M.vstar s = .ok vs
  padded : ∀ k, ∀ s ∈ M.padded k, s ∈ M.shapes

/-- **The padded sessions have room**: any `k` units, of any types, fit the menu's padded sessions for `k`, one type to a
session and within its shape's nonzero bound. -/
def Menu.Room {T V : Type} (M : Menu) (Ty : Types T V) : Prop :=
  ∀ (k : ℕ) (ty : Fin k → T), ∃ β : Batching k T Unit, β.view = M.padded k ∧ CoverCheck Finset.univ β ∧
    (∀ p i, β.held p = some i → β.typ p.1 = ty i) ∧ ∀ b, Ty.nnz (β.typ b) ≤ (β.shape b).nnz

/-- **The menu admits every draw** (completeness). At a menu V* stages whose padded sessions have room, for every
committed trace and every draw whose units it holds, the honest prover has a batching into the padded sessions for the
draw's size, each staged by V*, that passes the coverage check, is on the committed types and values, and keeps each
session's type within its shape's nonzero bound. -/
def MenuComplete : Prop :=
  ∀ {N : ℕ} {T V : Type} [Inhabited V] (M : Menu) (Ty : Types T V), ShapeMenu M → M.Room Ty →
    ∀ (X : Trace N T V) (S : Finset (Fin N)), (∀ i ∈ S, (X i).isSome = true) →
      ∃ β : Batching N T V, β.view = M.padded S.card ∧ (∀ s ∈ β.view, ∃ vs, M.vstar s = .ok vs) ∧
        CoverCheck S β ∧ Bound X β ∧ ∀ b, Ty.nnz (β.typ b) ≤ (β.shape b).nnz

/-- **The slots are sound.** Under a commitment whose records determine their contents, with `X` the trace it commits:
when what the sessions hold hidden satisfies the statement they prove at the draw `S`, no drawn unit is wrong in `X`, or
slot binding breaks. -/
def SlotsSound : Prop :=
  ∀ {N : ℕ} {T V : Type} (Ty : Types T V) (C : Commit N T V) (X : Trace N T V) (path₀ : Fin N → ByteArray)
    (S : Finset (Fin N)) (h : Hidden N T V),
    C.Injective → C.Committed X path₀ → Proves Ty C S h → Disjoint S (wrong Ty X) ∨ SlotBreak C X path₀ h

/-- **Composition with F1.** For F2's audit with the law `L`, the menu `M` and sessions `session`, with F1's conclusion
for those sessions (`I`) and, at each registration, a commitment whose records determine their contents and the trace
`X R` it commits: for every family `𝓑` of wrong-unit sets and every prover `σ`, a break of `σ`'s sessions at some draw
(F1's), or an outcome `σ` reaches whose sessions verify while a held slot's opening collides with the committed trace's
(`SlotBreak`), or
$$\Pr[\text{the auditor accepts} \wedge \mathrm{wrong}(X) \in 𝓑]
  \le \sup_{B \in 𝓑} \mathrm{escape}_L(B) + \mathbb{E}_\omega[\varepsilon].$$ -/
def Composition : Prop :=
  ∀ {N : ℕ} {T V Reg O : Type} (L : Law N) (Ty : Types T V) (M : Menu) (session : Finset (Fin N) → Reg → Game O)
    (com : Reg → Commit N T V) (view : O → List Shape) (I : Recursion session Ty com) (X : Reg → Trace N T V)
    (path₀ : Reg → Fin N → ByteArray),
    (∀ R, (com R).Injective) → (∀ R, (com R).Committed (X R) (path₀ R)) →
    ∀ (𝓑 : Finset (Fin N) → Prop) (σ : Strategy (sampledAudit L session)),
      (∃ ω, I.Break (L.draw ω) (regOf σ) (contOf σ ω)) ∨
      (∃ ω o, Reaches (session (L.draw ω) (regOf σ)) (contOf σ ω) o ∧ I.verifies o = true ∧
        SlotBreak (com (regOf σ)) (X (regOf σ)) (path₀ (regOf σ)) (I.hid (L.draw ω) (regOf σ) o)) ∨
      prob (fun x => Accepts M view I.verifies x.2 x.1 ∧ 𝓑 (wrong Ty (X (regOf σ)))) (sampledAudit L session) σ ≤
        (⨆ (B : Finset (Fin N)) (_ : 𝓑 B), L.escape B) + avg fun ω => I.ε (L.draw ω) (regOf σ) (contOf σ ω)

end FlockSampled
