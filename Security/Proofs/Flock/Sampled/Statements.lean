import Proofs.Flock.Sampled.Defs

/-!
# F2's statements: sampling with prover-chosen batching, the partition hidden

`Proofs.Flock.Sampled.NonVacuity` shows each one's hypotheses hold together, and `Proofs.Flock.Sampled.Proofs` proves
each one.

* `Coverage`: the outer proof's coverage check and slot binding's conclusion put each drawn unit in exactly one slot, of
  its committed type, on its committed values, against the committed hidden partition; and the auditor, which checks
  only the sessions' padded shapes, has room for the draw.
* `SlotBinding`: each held slot's opening of the trace root binds it to its unit's committed record, or the run's two
  openings give a SHA-512 collision. Each slot has all its shape's unit positions, the record filling the first and
  dummy copies of the unit the rest, so the unit layout V* reads per slot is public.
* `ShapeMenu`, `Menu.Fits`, `Menu.RoomAt`, `Menu.Room`: an audit's padded sessions take only the menu's shapes; a draw's
  units fit them by type; the draws that don't fit have probability at most δ under the law (F6's quantile session
  counts); and the deterministic sufficient condition, room for every draw whose types are placed within the menu's
  bounds, which gives δ = 0 (`Menu.Room.roomAt`, in `Proofs.Flock.Sampled.NonVacuity`).
* `Menu.Honest`, `MenuComplete`, `MenuCompleteAt`: an honest batching of a draw; at a menu V* stages and the live
  verifier accepts (`Menu.Staged`, `Menu.InRange`, obligations), one exists at every draw that fits; so the auditor
  draws one the honest prover can't batch with probability at most δ.
* `SlotsSound`: the statement the sessions prove puts every drawn unit's committed record in its type's relation, or
  slot binding breaks.
* `Composition`: at each registration, with F1's conclusion (`Recursion`, an interface), the audit accepts with
  probability at most the law's escape over the committed trace's wrong units plus F1's error, or a run breaks.
* `ZeroKnowledge`, `ZeroKnowledgeAt`: with F1's zero knowledge (`Recursion.hides`), an honest audit's view at a draw is
  simulated from the reveal list alone: the menu, the draw's size, the padded sessions and the owed outputs; and under
  the law, except on the honest prover's abort, of probability at most δ, which reveals the type mix.
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
trace's give a SHA-512 collision (`SlotBreak`; F1's `opens_binding` computes it). A slot is padded (top's ruling for
candidate 3's hiding): it has all its shape's unit positions (`Shape.positions`), whatever its type, so U_σ, the unit
layout V* reads per slot, is the menu entry's and public. The record a held slot is bound to fills the first
`Ty.instances t` of them, the statement checks that it fits (`Types.FitsIn`), and dummy copies of the unit, wired to
nothing, fill the rest; the openings bind the record, and the dummies bind nothing. -/
def SlotBinding : Prop :=
  ∀ {N : ℕ} {T V : Type} (C : Commit N T V) (X : Trace N T V) (path₀ : Fin N → ByteArray) (h : Hidden N T V),
    C.Injective → C.Committed X path₀ → C.SlotOpens h → Bound X h.β ∨ SlotBreak C X path₀ h

/-- **The shape menu**: an audit's padded sessions take only shapes on the menu. -/
def ShapeMenu (M : Menu) : Prop := ∀ k, ∀ s ∈ M.padded k, s ∈ M.shapes

/-- **The draw's units fit the padded sessions**, by type: some batching of the draw `S` into the menu's padded sessions
for `|S|` passes the coverage check, puts each held slot on its unit's committed type, fitting its shape (`Types.FitsIn`:
the nonzero bound, and the unit positions with dummy copies in the rest), and keeps each session's type table within its
shape's type bound. It is an honest batching without the values; a placement of the types at shapes with enough sessions
at each (F6's) gives one. -/
def Menu.Fits {N : ℕ} {T V : Type} (M : Menu) (Ty : Types T V) (X : Trace N T V) (S : Finset (Fin N)) : Prop :=
  ∃ β : Batching N T Unit, β.view = M.padded S.card ∧ CoverCheck S β ∧
    (∀ p i, β.held p = some i → (∃ v, X i = some (β.slotTyp p, v)) ∧ Ty.FitsIn (β.slotTyp p) (β.shape p.1)) ∧
    ∀ b, (β.table b).card ≤ M.types (β.shape b)

/-- **The padded sessions have room at all but δ of the draws**: under the law `L`, the draws of `X`'s units that don't
fit `M.padded |S|` (`Menu.Fits`) have probability at most δ. F6's quantile session counts give δ from each shape's
count of drawn units at 1 − 2⁻⁴⁰; `Menu.Room` gives δ = 0. -/
def Menu.RoomAt {N : ℕ} {T V : Type} (M : Menu) (Ty : Types T V) (L : Law N) (X : Trace N T V) (δ : ℝ≥0∞) : Prop :=
  prCoin (fun ω => ¬ M.Fits Ty X (L.draw ω)) ≤ δ

/-- **The padded sessions have room at every draw**, deterministically: any draw `S`, whose units' types `ty` lie in
`U`, placed within the menu's bounds, fits the menu's padded sessions for `|S|`, each unit in a slot of its type at its
type's shape, each session's type table within its shape's type bound. A sufficient condition for `Menu.RoomAt` at
δ = 0 (`Menu.Room.roomAt`); F6's quantile session counts don't meet it. -/
def Menu.Room {T V : Type} (M : Menu) (Ty : Types T V) : Prop :=
  ∀ {N : ℕ} (S : Finset (Fin N)) (ty : Fin N → T) (U : Finset T) (place : T → Shape), (∀ i ∈ S, ty i ∈ U) →
    M.Within Ty U place →
    ∃ β : Batching N T Unit, β.view = M.padded S.card ∧ CoverCheck S β ∧
      (∀ p i, β.held p = some i → β.slotTyp p = ty i ∧ β.shape p.1 = place (ty i)) ∧
      ∀ b, (β.table b).card ≤ M.types (β.shape b)

/-- **An honest batching** of the draw `S` of the trace `X`: into the menu's padded sessions for `|S|`, each staged by
V* and in the live verifier's range; passing the coverage check; on the committed types and values; each held slot's
type fitting its shape, padded to its unit positions; each session's type table within its shape's type bound. -/
def Menu.Honest {N : ℕ} {T V : Type} (M : Menu) (Ty : Types T V) (X : Trace N T V) (S : Finset (Fin N))
    (β : Batching N T V) : Prop :=
  β.view = M.padded S.card ∧ (∀ s ∈ β.view, ∃ vs, M.vstar s = .ok vs) ∧ (∀ s ∈ β.view, M.rangeCheck s = .ok ()) ∧
    CoverCheck S β ∧ Bound X β ∧ (∀ p i, β.held p = some i → Ty.FitsIn (β.slotTyp p) (β.shape p.1)) ∧
    ∀ b, (β.table b).card ≤ M.types (β.shape b)

/-- **The menu admits each draw that fits** (completeness, per draw). At a menu V* stages and the live verifier accepts
(`Menu.InRange`), whose padded sessions take only its shapes, at every draw whose units fit the padded sessions by type
(`Menu.Fits`), the honest prover has an honest batching of it. -/
def MenuComplete : Prop :=
  ∀ {N : ℕ} {T V : Type} [Inhabited V] (M : Menu) (Ty : Types T V) (X : Trace N T V) (S : Finset (Fin N)),
    M.Staged → M.InRange → ShapeMenu M → M.Fits Ty X S → ∃ β : Batching N T V, M.Honest Ty X S β

/-- **The menu admits all but δ of the draws** (completeness under the law), the form an end-to-end statement reads:
at such a menu, with room at all but δ of `X`'s draws (`Menu.RoomAt`), the auditor draws one the honest prover can't
batch, and aborts on, with probability at most δ. -/
def MenuCompleteAt : Prop :=
  ∀ {N : ℕ} {T V : Type} [Inhabited V] (M : Menu) (Ty : Types T V) (L : Law N) (X : Trace N T V) (δ : ℝ≥0∞),
    M.Staged → M.InRange → ShapeMenu M → M.RoomAt Ty L X δ →
      prCoin (fun ω => ¬ ∃ β : Batching N T V, M.Honest Ty X (L.draw ω) β) ≤ δ

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

/-- **Zero knowledge at the reveal list.** With F1's conclusion for the sessions (`I`, its (ii) as `hides`): at every
registration `R` and draw `S`, for every honest witness `h` (sessions at the menu's padded shapes for `|S|`, proving the
statement at `S`), the auditor's view of the honest audit is close, in F1's sense, to the simulated one, which reads only
the reveal list `reveal M S (I.out R)`: the menu (its shapes, nonzero bounds and type bounds), the draw (the auditor's own
coins, so `k = |S|`), the padded sessions `M.padded |S|` and the owed outputs. It reads nothing of the batching, the
slots' types, their values, their instance counts or the types' placement: each slot's unit layout is its shape's
(`Shape.positions`, dummy copies where its type has fewer instances), so the padded sessions give it. And since the real
view holds the registration's root, the trace's leaves hide each unit's type as well as its values (the type field
salted and hidden with them). -/
def ZeroKnowledge : Prop :=
  ∀ {N : ℕ} {T V Reg O : Type} (session : Finset (Fin N) → Reg → Game O) (Ty : Types T V) (com : Reg → Commit N T V)
    (I : Recursion session Ty com) (M : Menu) (R : Reg) (S : Finset (Fin N)) (h : Hidden N T V),
    h.β.view = M.padded S.card → Proves Ty (com R) S h →
      I.Close (reveal M S (I.out R)) (I.real (reveal M S (I.out R)) R h) (I.sim (reveal M S (I.out R)))

/-- **Zero knowledge under the law**, the one place δ enters it. At a registration `R` whose committed trace `X` (under
`com R`, with the openings `path₀`) holds only records in their types' relations, at a menu V* stages and the live
verifier accepts, with room at all but δ of the draws: the honest prover aborts, having no honest batching of the draw, with probability at most δ; and at
every other draw, for every honest batching, the auditor's view of the honest audit, its slots opened at `path₀`
(`Hidden.honest`), is close in F1's sense to the simulated one, which reads only the reveal list. An honest abort is not
simulated from the reveal list: it reveals the type mix, that the draw's types don't fit the padded sessions, which the
hidden partition decides, so δ bounds that leak as well as the completeness error. `Close` keeps F1's form; δ stays
outside it. -/
def ZeroKnowledgeAt : Prop :=
  ∀ {N : ℕ} {T V Reg O : Type} [Inhabited V] (session : Finset (Fin N) → Reg → Game O) (Ty : Types T V)
    (com : Reg → Commit N T V) (I : Recursion session Ty com) (M : Menu) (L : Law N) (R : Reg) (X : Trace N T V)
    (path₀ : Fin N → ByteArray) (δ : ℝ≥0∞),
    M.Staged → M.InRange → ShapeMenu M → (com R).Committed X path₀ → (∀ i t v, X i = some (t, v) → Ty.holds t v) →
    M.RoomAt Ty L X δ →
      prCoin (fun ω => ¬ ∃ β : Batching N T V, M.Honest Ty X (L.draw ω) β) ≤ δ ∧
      ∀ (ω : L.Ω) (β : Batching N T V), M.Honest Ty X (L.draw ω) β →
        I.Close (reveal M (L.draw ω) (I.out R))
          (I.real (reveal M (L.draw ω) (I.out R)) R (Hidden.honest β path₀)) (I.sim (reveal M (L.draw ω) (I.out R)))

end FlockSampled
