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
proofs are verified against are the cut of the registered wired Program. Both read the registered wired Program and
the positions as given (`wired`, `named`, `posOf`), which that check is to define.
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

/-- **`Pu` with `pos` is the cut of `Pw` under `W`**: the same callee at every unit, and exactly one position for each
wired reader row, naming that wire's writer, output field and leaf. -/
def CutOf (Pw Pu : Flock.Program.Prog) (W : Spot.Wires.Wiring) (pos : Positions) : Prop :=
  (∀ u, calleeAt Pu u = calleeAt Pw u) ∧
    (∀ w ∈ W.wires, ∀ (u row v : ℕ), w.leafAt u row = some v → pos w.reader u w.port row = some (w.writer, w.out, v)) ∧
    ∀ t u port row x, pos t u port row = some x →
      ∃ w ∈ W.wires, ∃ v, w.reader = t ∧ w.port = port ∧ w.leafAt u row = some v ∧ x = (w.writer, w.out, v)

namespace Vocab

variable {d : ℕ} {α : Type} (V : Vocab d α)

/-- Unit `u` carries its inputs in `X`: at each committed gate it reads, whose position `named` gives, the value is
the value the writer unit's gate at that position is committed to, bit for bit. -/
def Carries (named : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option Place)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (X : (V.circuitOf θ F).Trace) (u : Fin (V.units θ F)) : Prop :=
  ∀ g ∈ (V.partsOf θ F).io u, ∀ q, named θ F g = some q →
    ∀ (h : Fin (V.circuitOf θ F).N) (v : Fin (V.units θ F)), V.place θ F c h = some q →
      (V.partsOf θ F).unit h = some v → h ∈ (V.partsOf θ F).io v →
        ∀ b, V.Opens θ F c h b → ∀ i, (X g).getLsbD i = b.getLsbD i

open Classical in
/-- The units `X` doesn't bind or that don't carry their inputs. -/
noncomputable def uncarried (named : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option Place)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (X : (V.circuitOf θ F).Trace) : Finset (Fin (V.units θ F)) :=
  Finset.univ.filter fun u => u ∈ unbound (V.partsOf θ F) (V.Opens θ F c) X ∨ ¬ V.Carries named θ F c X u

/-- **The positions carry** (`SpotSound`'s form over the registered reads' positions `named`): on a play the sampled
proofs accept, all but at most an `ε` share of the units are bound and carry their inputs, but for the sampled proofs'
error. -/
def PositionsCarry (k : Verity.Registration.Composite)
    (named : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option Place) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧
        ¬ ((V.uncarried named θ F c (V.extract θ F c σ)).card : ℝ) ≤ (V.certOf θ).ε * V.units θ F)
      (V.game θ F c) σ ≤ V.error .sampled θ F c σ

/-- **The unwired Program is the cut of the registered one**: on an accepted case, the Program the circuit file
decodes to, with the positions `posOf` reads, is the cut (`CutOf`) of the registered wired Program `wired` under the
wiring the record's text states. -/
def CutIsRegistered (k : Verity.Registration.Composite) (wired : ByteArray → Option Flock.Program.Prog)
    (posOf : ByteArray → Array ByteArray → Positions) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (o : Exchange), V.Verdict k .sampled o →
    ∀ p r e, PolicyCase k o.sent p r e →
      ∀ (Pw : Flock.Program.Prog) (text : String) (doc : Lean.Json) (W : Spot.Wires.Wiring), wired θ = some Pw →
        InferenceOnlyVerifier.Theta.strAt (InferenceOnlyVerifier.Theta.getD r.record "window") "circuit" =
            some (Flock.hexOf (Flock.Sha512.hash text.toUTF8)) →
          Flock.JsonParse.parse text = .ok doc → Spot.RecordText.wiring Pw doc = .ok W →
            ∀ Pu, Flock.Program.decode (V.circuitFile θ F) = .ok Pu → CutOf Pw Pu W (posOf θ F)

end Vocab

end Verity
