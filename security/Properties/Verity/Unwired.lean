module

public import Properties.Verity.Game
public import Definitions.Compose.JointLive
public import Spot.RecordText
public import Spot.OneStageZk

/-!
# The unwired route: what its positions owe

A registered class is served as its unwired Program with explicit positions: each wired input row of a proof block
names its writer, the writer's output field and the leaf there. Two named `Prop`s, stated and not proved.

`Vocab.PositionsCarry`, owed a proof from the per-block check and the sample: on a play the sampled proofs accept, all
but an `ε` share of the units hold, at each input position, the value the writer's output is committed to at the
position named.

`Vocab.CutIsRegistered`, open until the registration's check exists: the unwired Program and positions the sampled
proofs are verified against are the cut of the registered wired Program and its circuit record. Both read the same positions `posOf`, and the
registered wired Program (`wired`) and the file's labels (`Labels`) as given, which that check is to define.
-/

@[expose] public section

namespace Verity

open ResourceAccounting Core

/-- Of reader `(template, unit, port, row)`, the position it names: writer template, output field and leaf. -/
abbrev Positions := String → ℕ → String → ℕ → Option (String × String × ℕ)

/-- Unit `u`'s callee in `P`: the Definition of the root node that holds it. -/
def calleeAt (P : Flock.Program.Prog) (u : ℕ) : Option Flock.Program.Defn :=
  (Flock.HmLinks.unitAt (Flock.HmLinks.nodes P) u).map fun (k, _) =>
    P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!

/-- **`Pu` with `pos` is the cut of `Pw` and its circuit record `doc`**: the wiring `doc` states over `Pw`
(`Spot.RecordText.wiring`), the same callee at every unit, and exactly one position for each wired reader row, naming
that wire's writer, output field and leaf. -/
def CutOf (Pw : Flock.Program.Prog) (doc : Lean.Json) (Pu : Flock.Program.Prog) (pos : Positions) : Prop :=
  ∃ W, Spot.RecordText.wiring Pw doc = .ok W ∧ (∀ u, calleeAt Pu u = calleeAt Pw u) ∧
    (∀ w ∈ W.wires, ∀ (u row v : ℕ), w.leafAt u row = some v → pos w.reader u w.port row = some (w.writer, w.out, v)) ∧
    ∀ t u port row x, pos t u port row = some x →
      ∃ w ∈ W.wires, ∃ v, w.reader = t ∧ w.port = port ∧ w.leafAt u row = some v ∧ x = (w.writer, w.out, v)

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- The committed file's labels, which the registration's check is to define: a reader's input gate's row
`(template, unit, port, row)` and a writer's output gate's leaf `(template, field, leaf)`, the leaf a position in that
template's tree of the field over all its units. -/
structure Labels where
  row : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option (String × ℕ × String × ℕ)
  leaf : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option (String × String × ℕ)

/-- Every input gate of a unit has its row label, so no wired reader row escapes `Carries` by going unlabelled. -/
def Labels.Total (L : Labels V) : Prop :=
  ∀ θ F (g : Fin (V.circuitOf θ F).N), (V.circuitOf θ F).isInput g = true → ((V.partsOf θ F).unit g).isSome →
    (L.row θ F g).isSome

/-- A leaf label names at most one gate. -/
def Labels.Injective (L : Labels V) : Prop :=
  ∀ θ F (h h' : Fin (V.circuitOf θ F).N) x, L.leaf θ F h = some x → L.leaf θ F h' = some x → h = h'

/-- Unit `u` carries its inputs in `X`: at each committed gate it reads, the position `posOf` names for its row is the
leaf of a writer unit's output gate that is committed to a value, and every value it is committed to is the input's,
bit for bit. A writer with no opening counts in `unbound` only for itself, so its reader does not carry. -/
def Carries (L : Labels V) (posOf : ByteArray → Array ByteArray → Positions)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (X : (V.circuitOf θ F).Trace) (u : Fin (V.units θ F)) : Prop :=
  ∀ g ∈ (V.partsOf θ F).io u, ∀ t w port row, L.row θ F g = some (t, w, port, row) →
    ∀ x, posOf θ F t w port row = some x →
      ∃ (h : Fin (V.circuitOf θ F).N) (v : Fin (V.units θ F)), L.leaf θ F h = some x ∧
        (V.partsOf θ F).unit h = some v ∧ h ∈ (V.partsOf θ F).io v ∧
        (∃ b : BitVec ((V.circuitOf θ F).w h), V.Opens θ F c h b) ∧
        ∀ b : BitVec ((V.circuitOf θ F).w h), V.Opens θ F c h b → ∀ i, (X g).getLsbD i = b.getLsbD i

open Classical in
/-- The units `X` doesn't bind or that don't carry their inputs. -/
noncomputable def uncarried (L : Labels V) (posOf : ByteArray → Array ByteArray → Positions)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (X : (V.circuitOf θ F).Trace) : Finset (Fin (V.units θ F)) :=
  Finset.univ.filter fun u => u ∈ unbound (V.partsOf θ F) (V.Opens θ F c) X ∨ ¬ V.Carries L posOf θ F c X u

/-- **The positions carry** (`SpotSound`'s form over the positions `posOf`, the ones `CutIsRegistered` reads): on a
play the sampled proofs accept, all but at most an `ε` share of the units are bound and carry their inputs, but for
the sampled proofs' error. It requires `L` total and injective (`Labels.Total`, `Labels.Injective`): else a labelling
that labels no gate makes every unit carry. -/
def PositionsCarry (k : Verity.Registration.Composite) (L : Labels V)
    (posOf : ByteArray → Array ByteArray → Positions) : Prop :=
  L.Total → L.Injective → ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧
        ¬ ((V.uncarried L posOf θ F c (V.extract θ F c σ)).card : ℝ) ≤ (V.certOf θ).ε * V.units θ F)
      (V.game θ F c) σ ≤ V.error .sampled θ F c σ

/-- **The unwired Program is the cut of the registered one**: on an accepted case, the Program the circuit file
decodes to, with the positions `posOf` reads, is the cut (`CutOf`) of the registered wired Program `wired` and the
circuit record the record names by its text's digest. -/
def CutIsRegistered (k : Verity.Registration.Composite) (wired : ByteArray → Option Flock.Program.Prog)
    (posOf : ByteArray → Array ByteArray → Positions) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (o : Exchange), V.Verdict k .sampled o →
    ∀ p r e, PolicyCase k o.sent p r e →
      ∀ (Pw : Flock.Program.Prog) (text : String) (doc : Lean.Json), wired θ = some Pw →
        InferenceOnlyVerifier.Theta.strAt (InferenceOnlyVerifier.Theta.getD r.record "window") "circuit" =
            some (Flock.hexOf (Flock.Sha512.hash text.toUTF8)) →
          Flock.JsonParse.parse text = .ok doc →
            ∀ Pu, Flock.Program.decode (V.circuitFile θ F) = .ok Pu → CutOf Pw doc Pu (posOf θ F)

end Vocab

end Verity
