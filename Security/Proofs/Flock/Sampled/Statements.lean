import Proofs.Flock.Sampled.Defs

/-!
# F2's statements: sampling with prover-chosen batching, the partition hidden

Skeletons, stated and not yet proved; `Proofs.Flock.Sampled.NonVacuity` shows each one's hypotheses hold together.

* `Coverage`: the outer proof's coverage check and slot binding's conclusion put each drawn unit in exactly one slot, of
  its committed type, on its committed values, against the committed hidden partition; and the auditor, which checks
  only the sessions' padded shapes, has room for the draw.
* `SlotBinding`: each held slot's opening of the trace root binds it to its unit's committed record, or the run's two
  openings give a SHA-512 collision.
* `ShapeMenu`, `Menu.Room`, `MenuComplete`: an audit's padded sessions take only the menu's shapes; they have room for
  any `k` units whose types are placed within the menu's bounds; and then, at a menu V* stages (`Menu.Staged`, an
  obligation), an honest prover can batch every draw.
* `SlotsSound`: the statement the sessions prove puts every drawn unit's committed record in its type's relation, or
  slot binding breaks.
* `Composition`: at each registration, with F1's conclusion (`Recursion`, an interface), the audit accepts with
  probability at most the law's escape over the committed trace's wrong units plus F1's error, or a run breaks.
-/

namespace FlockSampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

/-- **Coverage.** When the outer proof's coverage check passes on a batching whose held slots are on their units'
committed types and values, each drawn unit sits in exactly one slot, of its committed type, on its committed values;
and the sessions' slots, which the auditor counts from their public shapes alone, number at least the draw. -/
def Coverage : Prop :=
  ∀ {N : ℕ} {T V : Type} (X : Trace N T V) (S : Finset (Fin N)) (β : Batching N T V),
    CoverCheck S β → Bound X β → Covers X S β ∧ S.card ≤ (β.view.map Shape.slots).sum

/-- **Slot binding.** Under a commitment whose records determine their contents (`Commit.Injective`, the type in the
leaf as a fixed-width field), with `X` the trace it commits: when every held slot's record opens the root at its unit's
position, every held slot is on its unit's committed type and values, or some held slot's opening and the committed
trace's give a SHA-512 collision (`SlotBreak`; F1's `opens_binding` computes it). -/
def SlotBinding : Prop :=
  ∀ {N : ℕ} {T V : Type} (C : Commit N T V) (X : Trace N T V) (path₀ : Fin N → ByteArray) (h : Hidden N T V),
    C.Injective → C.Committed X path₀ → C.SlotOpens h → Bound X h.β ∨ SlotBreak C X path₀ h

/-- **The shape menu**: an audit's padded sessions take only shapes on the menu. -/
def ShapeMenu (M : Menu) : Prop := ∀ k, ∀ s ∈ M.padded k, s ∈ M.shapes

/-- **The padded sessions have room**: any `k` units, whose types `U` are placed within the menu's bounds, fit the menu's
padded sessions for `k`, each in a slot of its type at its type's shape, each session's type table within its shape's
type bound. -/
def Menu.Room {T V : Type} (M : Menu) (Ty : Types T V) : Prop :=
  ∀ (k : ℕ) (ty : Fin k → T) (U : Finset T) (place : T → Shape), (∀ i, ty i ∈ U) → M.Within Ty U place →
    ∃ β : Batching k T Unit, β.view = M.padded k ∧ CoverCheck Finset.univ β ∧
      (∀ p i, β.held p = some i → β.slotTyp p = ty i ∧ β.shape p.1 = place (ty i)) ∧
      ∀ b, (β.table b).card ≤ M.types (β.shape b)

/-- **The menu admits every draw** (completeness). At a menu V* stages whose padded sessions take only its shapes and
have room, for every committed trace whose types at the draw are placed within the menu's bounds, and every draw whose
units it holds, the honest prover has a batching into the padded sessions for the draw's size, each staged by V*, that
passes the coverage check, is on the committed types and values, keeps each held slot's type within its shape's nonzero
bound and each session's type table within its shape's type bound. -/
def MenuComplete : Prop :=
  ∀ {N : ℕ} {T V : Type} [Inhabited V] (M : Menu) (Ty : Types T V), M.Staged → ShapeMenu M → M.Room Ty →
    ∀ (X : Trace N T V) (S : Finset (Fin N)) (U : Finset T) (place : T → Shape),
      (∀ i ∈ S, ∃ t v, X i = some (t, v) ∧ t ∈ U) → M.Within Ty U place →
      ∃ β : Batching N T V, β.view = M.padded S.card ∧ (∀ s ∈ β.view, ∃ vs, M.vstar s = .ok vs) ∧
        CoverCheck S β ∧ Bound X β ∧ (∀ p i, β.held p = some i → Ty.nnz (β.slotTyp p) ≤ (β.shape p.1).nnz) ∧
        ∀ b, (β.table b).card ≤ M.types (β.shape b)

/-- **The slots are sound.** Under a commitment whose records determine their contents, with `X` the trace it commits:
when what the sessions hold hidden satisfies the statement they prove at the draw `S`, no drawn unit is wrong in `X`, or
slot binding breaks. -/
def SlotsSound : Prop :=
  ∀ {N : ℕ} {T V : Type} (Ty : Types T V) (C : Commit N T V) (X : Trace N T V) (path₀ : Fin N → ByteArray)
    (S : Finset (Fin N)) (h : Hidden N T V),
    C.Injective → C.Committed X path₀ → Proves Ty C S h → Disjoint S (wrong Ty X) ∨ SlotBreak C X path₀ h

/-- **Composition with F1**, per registration: the prover has registered `R` first (its trace root among it), then the
auditor draws, then the sessions run. For F2's audit at `R` with the law `L`, the menu `M` and sessions `session`, with
F1's conclusion for those sessions (`I`), a commitment at `R` whose records determine their contents and the trace `X` it
commits (fixed by `R` alone, so before the draw): for every prover `σ` of the draw's continuation, a break of `σ`'s
sessions at some draw (F1's), or an outcome `σ` reaches whose sessions verify while a held slot's opening collides with
the committed trace's (`SlotBreak`), or
$$\Pr[\text{the auditor accepts}] \le \mathrm{escape}_L(\mathrm{wrong}(X)) + \mathbb{E}_\omega[\varepsilon].$$
A game that starts with the registration gets the family form, over the registrations whose wrong units lie in a family,
from this one at each. -/
def Composition : Prop :=
  ∀ {N : ℕ} {T V Reg O : Type} (L : Law N) (Ty : Types T V) (M : Menu) (session : Finset (Fin N) → Reg → Game O)
    (com : Reg → Commit N T V) (view : O → List Shape) (I : Recursion session Ty com) (R : Reg) (X : Trace N T V)
    (path₀ : Fin N → ByteArray),
    (com R).Injective → (com R).Committed X path₀ →
    ∀ σ : Strategy (sampledAudit L session R),
      (∃ ω, I.Break (L.draw ω) R (contOf σ ω)) ∨
      (∃ ω o, Reaches (session (L.draw ω) R) (contOf σ ω) o ∧ I.verifies o = true ∧
        SlotBreak (com R) X path₀ (I.hid (L.draw ω) R o)) ∨
      prob (fun x => Accepts M view I.verifies x.2 x.1) (sampledAudit L session R) σ ≤
        L.escape (wrong Ty X) + avg fun ω => I.ε (L.draw ω) R (contOf σ ω)

end FlockSampled
