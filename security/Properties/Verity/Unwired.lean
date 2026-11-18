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
proofs are verified against are the cut of the registered wired Program and its circuit record. Both read the same
positions `posOf` and the registered wired Program (`wired`) as given, which that check is to define. The labels are the
cut Program's (`Labels.ofProgram`), placed on the committed circuit's gates by `place`, for which their totality and
injectivity are open (`OfProgramTotal`, `OfProgramInjective`).
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

namespace Unwired

open Flock

/-- Unit `u` of `P`: its root node, its member there and its callee. -/
def unitCallee (P : Program.Prog) (u : ℕ) : Option (ℕ × ℕ × Program.Defn) :=
  (HmLinks.unitAt (HmLinks.nodes P) u).map fun (k, m) => (k, m, P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!)

/-- The first leaf of parameter `i` of `ps`: the leaves of the parameters before it. -/
def paramStart (ps : Array (String × Program.Ty)) (i : ℕ) : ℕ :=
  (ps.extract 0 i).foldl (fun (a : ℕ) (q : String × Program.Ty) => a + Program.Ty.leaves q.2) 0

/-- Gate `g`'s input labels in `P`: `(template, u, p, 0, j)` for each unit `u` whose parameter `p` reads `g` at its
leaf `j`, one row of `p` per unit. -/
def inputLabels (P : Program.Prog) (g : ℕ) : List (String × ℕ × String × ℕ × ℕ) :=
  (List.range (HmLinks.count (HmLinks.nodes P))).flatMap fun u =>
    match unitCallee P u with
    | none => []
    | some (k, m, f) =>
      match Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m with
      | .error _ => []
      | .ok ops => (List.range f.params.size).flatMap fun i =>
          ((List.range (Program.Ty.leaves f.params[i]!.2)).filter fun j => ops[paramStart f.params i + j]? == some g).map
            fun j => (f.key, u, f.params[i]!.1, 0, j)

/-- Gate `g`'s output label in `P`: `(template, o{j}, leaf, word)` where `g` is word `word` of field `j` of the unit
whose leaf in the template's field tree is `leaf` (`HmLinks.treeLeaf`); none where no unit outputs `g`. -/
def outputLabel (P : Program.Prog) (g : ℕ) : Option (String × String × ℕ × ℕ) := do
  let ns := HmLinks.nodes P
  let v ← HmLinks.ownerAt ns g
  let (k, m) ← HmLinks.unitAt ns v
  let (key, rank) ← HmLinks.treeLeaf P ns v
  let outs ← (HmLinks.outGates P k m).toOption
  let i ← outs.findIdx? (· == g)
  let fs := Program.fieldsOf (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).ret
  let start := fun (j : ℕ) => (fs.take j).foldl (fun (a : ℕ) (t : Program.Ty) => a + t.leaves) 0
  let j ← (List.range fs.length).find? fun j => i < start (j + 1)
  pure (key, s!"o{j}", rank, i - start j)

end Unwired

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- The committed file's labels: a reader's input gate's row and word `(template, unit, port, row, word)` and a writer's
output gate's leaf and word `(template, field, leaf, word)`, the leaf a position in that template's tree of the field
over all its units. -/
structure Labels where
  row : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option (String × ℕ × String × ℕ × ℕ)
  leaf : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option (String × String × ℕ × ℕ)

/-- The labels of the cut Program `prog` (`Unwired.inputLabels`, `Unwired.outputLabel`) at the Program gate `place`
gives each of the committed circuit's gates. -/
def Labels.ofProgram (prog : ByteArray → Array ByteArray → Option Flock.Program.Prog)
    (place : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option ℕ) : Labels V where
  row θ F g := do
    let P ← prog θ F
    let n ← place θ F g
    (Unwired.inputLabels P n).head?
  leaf θ F h := do
    let P ← prog θ F
    let n ← place θ F h
    Unwired.outputLabel P n

/-- Every input gate of a unit has its row label, so no wired reader row escapes `Carries` by going unlabelled. -/
def Labels.Total (L : Labels V) : Prop :=
  ∀ θ F (g : Fin (V.circuitOf θ F).N), (V.circuitOf θ F).isInput g = true → ((V.partsOf θ F).unit g).isSome →
    (L.row θ F g).isSome

/-- A leaf label names at most one gate. -/
def Labels.Injective (L : Labels V) : Prop :=
  ∀ θ F (h h' : Fin (V.circuitOf θ F).N) x, L.leaf θ F h = some x → L.leaf θ F h' = some x → h = h'

/-- Unit `u` carries its inputs in `X`: at each committed gate it reads, the position `posOf` names for its row, at the
gate's word, is the leaf of a writer unit's output gate that is committed to a value, and every value it is committed to is the input's,
bit for bit. A writer with no opening counts in `unbound` only for itself, so its reader does not carry. -/
def Carries (L : Labels V) (posOf : ByteArray → Array ByteArray → Positions)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (X : (V.circuitOf θ F).Trace) (u : Fin (V.units θ F)) : Prop :=
  ∀ g ∈ (V.partsOf θ F).io u, ∀ t w port row word, L.row θ F g = some (t, w, port, row, word) →
    ∀ x, posOf θ F t w port row = some x →
      ∃ (h : Fin (V.circuitOf θ F).N) (v : Fin (V.units θ F)), L.leaf θ F h = some (x.1, x.2.1, x.2.2, word) ∧
        (V.partsOf θ F).unit h = some v ∧ h ∈ (V.partsOf θ F).io v ∧
        (∃ b : BitVec ((V.circuitOf θ F).w h), V.Opens θ F c h b) ∧
        ∀ b : BitVec ((V.circuitOf θ F).w h), V.Opens θ F c h b → ∀ i, (X g).getLsbD i = b.getLsbD i

open Classical in
/-- The units `X` doesn't bind or that don't carry their inputs. -/
noncomputable def uncarried (L : Labels V) (posOf : ByteArray → Array ByteArray → Positions)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (X : (V.circuitOf θ F).Trace) : Finset (Fin (V.units θ F)) :=
  Finset.univ.filter fun u => u ∈ unbound (V.partsOf θ F) (V.Opens θ F c) X ∨ ¬ V.Carries L posOf θ F c X u

/-- The cut Program the circuit file decodes to, the one `CutIsRegistered` reads. -/
def cutProgram (θ : ByteArray) (F : Array ByteArray) : Option Flock.Program.Prog :=
  (Flock.Program.decode (V.circuitFile θ F)).toOption

/-- `Labels.ofProgram` at the cut Program labels every input gate of a unit (`Labels.Total`). Open: it is about
`place`, the map from the committed circuit's gates to the Program's that the circuit file's layout is to define. -/
def OfProgramTotal (place : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option ℕ) : Prop :=
  (Labels.ofProgram V V.cutProgram place).Total

/-- `Labels.ofProgram` at the cut Program names each leaf and word at most once (`Labels.Injective`). Open, as
`OfProgramTotal`. -/
def OfProgramInjective (place : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option ℕ) : Prop :=
  (Labels.ofProgram V V.cutProgram place).Injective

/-- **The positions carry** (`SpotSound`'s form over the positions `posOf`, the ones `CutIsRegistered` reads, and the
cut Program's labels `Labels.ofProgram`): on a play the sampled proofs accept, all but at most an `ε` share of the units
are bound and carry their inputs, but for the sampled proofs' error. -/
def PositionsCarry (k : Verity.Registration.Composite)
    (place : (θ : ByteArray) → (F : Array ByteArray) → Fin (V.circuitOf θ F).N → Option ℕ)
    (posOf : ByteArray → Array ByteArray → Positions) : Prop :=
  V.OfProgramTotal place → V.OfProgramInjective place →
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧
        ¬ ((V.uncarried (Labels.ofProgram V V.cutProgram place) posOf θ F c (V.extract θ F c σ)).card : ℝ) ≤
          (V.certOf θ).ε * V.units θ F)
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
