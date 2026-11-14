module

public import Properties.Verity.Game
public import Properties.Spot.Wires
public import Spot.FilePart
public import NciCheck.RecordText

/-!
# What ties the inference-only policy's clause to the committed files

The game's clause for the inference-only policy (`Verity.Vocab.Holds` at `.inferenceOnly`) is on the Program the
policy's compiled check judges, and is proved of the case as sent. It says nothing of the committed files until two
statements hold, each a named `Prop` here, stated and not proved. Remote's soundness (`Verity.RemoteSoundness`) takes
both as hypotheses.

`GateARow`, `Γ`'s unit: in the game's circuit of a committed file (`Flock.Bridge.circuitOfFile`) every unit has one
gate for each of its file's rows. The policy's `Γ` counts rows of the committed circuit a block, and PoUW's claim
counts that circuit's gates (`Verity.Vocab.computed`), so with it the two count one thing.

`Vocab.WiresSound`, the wires: on an accepted play the wiring the policy's record names is the wiring the committed
units are held to. The committed circuit's units each commit their own rows, and nothing in it joins a unit's input
row to the output row the record wires it to. The statement reads the wiring on the commitment:

* **the record's wiring** (`RecordWiring`): the wires of the text whose SHA-512 the record's `window.circuit` is,
  as SPOT's reader reads them (`NciCheck.RecordText.wiring`);
* **a row's root** (`RecordRoot`, `rowName`): the root the record's `roots` hold for the row's name, a port's own
  name or, of a port cut into several rows, the port's name, `#` and the row;
* **a unit's leaf under a root** (`Vocab.ReadsLeaf`, `Vocab.WritesLeaf`): the registered position (`Vocab.place`) of
  an input gate the unit reads, or of a gate of the unit, a wiring's unit `u` being the committed circuit's unit `u`;
* **a wire holds** (`Vocab.SameRow`): every opening the row-leaf verifier accepts of the reader's leaf and every one
  it accepts of the writer's leaf hold the same bits.

A unit is then held (`Properties.Spot.WiredCorrect`) when the transcript binds it (`Accounting.unbound`) and no wire
into it fails (`Properties.Spot.miswired`), and the claim (`Vocab.WiresHold`) is sampled proofs' at twice its share:
at most a `2ε` share of the units not held. The target is the claim at twice sampled proofs' error.
-/

@[expose] public section

namespace Verity

open Accounting
open scoped ENNReal

/-- **`Γ`'s unit: the committed circuit has one gate a row.** At all committed files that parse
(`Flock.Bridge.parsedOf`), every unit of the game's circuit of them (`Flock.Bridge.circuitOfFile`, its units
`partsOfFile`) has as many gates as its circuit file has rows a unit: the `m` computed rows the verifier counts from
the parsed file (`Spot.FilePart.ofCircuit`), the unit net's rows stacked `units_per_vu` times. A unit's gates are its
computed gates: the input gates are in no unit.

It is a count over the game's circuit and reads no play. What it does not say: that a row is one gate of the sampling
claims' program, which lowers a row to three gates (`Spot.FilePart.block`); or anything of files that don't parse,
whose circuit has no unit. -/
def GateARow : Prop :=
  ∀ (tags : Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (pf : ByteArray)
    (r : Flock.Bridge.Parsed), Flock.Bridge.parsedOf tags cf tables pf = some r →
    ∀ u : Fin (Flock.Bridge.unitsOfFile tags cf tables pf),
      (Flock.Bridge.circuitOfFile tags cf tables pf).size ((Flock.Bridge.partsOfFile tags cf tables pf).gates u) =
        (Spot.FilePart.ofCircuit r.D.unroll r.n).m

/-- **The record names the wiring `W`**: the record's `window.circuit` is the SHA-512 of a text, the circuit
record's, that parses to a document whose cuts and wires SPOT's reader reads as `W`
(`NciCheck.RecordText.wiring`: no template and no order, which no wire's rows read). -/
def RecordWiring (record : Lean.Json) (W : Spot.Wires.Wiring) : Prop :=
  ∃ (text : String) (doc : Lean.Json),
    NciCheck.Baseline.strAt (NciCheck.Baseline.getD record "window") "circuit" =
        some (Flock.hexOf (Flock.Sha512.hash text.toUTF8)) ∧
      Flock.JsonParse.parse text = .ok doc ∧ NciCheck.RecordText.wiring doc = .ok W

/-- **The record names `R` as the root of its row `name`**: one of its `roots` has the port `name` and the root
`R`, in hex. -/
def RecordRoot (record : Lean.Json) (name : String) (R : Hash.Digest) : Prop :=
  ∃ roots, record.getObjValD "roots" = .arr roots ∧ ∃ e ∈ roots,
    e.getObjValD "port" = .str name ∧ e.getObjValD "root" = .str (Flock.hexOf ⟨R.bytes.toArray⟩)

/-- **A reader row's name** in the record: the port's own name, or, of a port the wiring cuts into several rows
(`Spot.Wires.Wiring.cutsOf`), the port's name, `#` and the row. -/
def rowName (W : Spot.Wires.Wiring) (template port : String) (row : ℕ) : String :=
  if W.cutsOf template port = 1 then port else port ++ "#" ++ toString row

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **A leaf unit `u` reads under the root `R`**: the registered position of an input gate that a gate of the
committed circuit's unit `u` reads, when its root is `R`. -/
def ReadsLeaf (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (R : Hash.Digest) (u : ℕ) (p : Place) : Prop :=
  p.root = R ∧ ∃ (g h : Fin (V.circuitOf θ F).N) (v : Fin (V.units θ F)), V.place θ F c g = some p ∧
    (V.circuitOf θ F).isInput g = true ∧ (V.partsOf θ F).unit h = some v ∧ v.val = u ∧ g ∈ (V.circuitOf θ F).reads h

/-- **A leaf unit `u` writes under the root `R`**: the registered position of a gate of the committed circuit's
unit `u`, when its root is `R`. -/
def WritesLeaf (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (R : Hash.Digest) (u : ℕ) (p : Place) : Prop :=
  p.root = R ∧ ∃ (g : Fin (V.circuitOf θ F).N) (v : Fin (V.units θ F)), V.place θ F c g = some p ∧
    (V.partsOf θ F).unit g = some v ∧ v.val = u

/-- **The reader's row is its writer's row under the commitment**, for a reader row `x = (template, unit, port,
row)` and a writer's output row `y = (template, unit, output)` of the wiring `W`: at the roots the record names for
the two rows (`RecordRoot`), every opening the row-leaf verifier accepts under SHA-512 of a leaf the reader's unit
reads and every one it accepts of a leaf the writer's unit writes hold the same bit wherever both hold one. A row no
root of the record names, or a leaf with no accepted opening, is compared with nothing. -/
def SameRow (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (record : Lean.Json) (W : Spot.Wires.Wiring)
    (x : String × ℕ × String × ℕ) (y : String × ℤ × String) : Prop :=
  ∀ (Rx Ry : Hash.Digest) (p q : Place) (v : ℕ) (o o' : Hash.RowLeaf.Opening),
    RecordRoot record (rowName W x.1 x.2.2.1 x.2.2.2) Rx → RecordRoot record y.2.2 Ry →
    V.ReadsLeaf θ F c Rx x.2.1 p → (v : ℤ) = y.2.1 → V.WritesLeaf θ F c Ry v q →
    Hash.RowLeaf.accepts sha512 sha512 p.verifier p.root p.rank o →
    Hash.RowLeaf.accepts sha512 sha512 q.verifier q.root q.rank o' →
    ∀ (i : ℕ) (b b' : Bool), o.row[i]? = some b → o'.row[i]? = some b' → b = b'

open Classical in
/-- **The units the wiring `W` doesn't hold** in the transcript `X`: those that are not correct in the wired
population (`Properties.Spot.WiredCorrect`), a unit being correct on its own rows when `X` binds it
(`Accounting.unbound` under the commitment's openings) and a wire holding as `SameRow` compares its two rows. A
wiring's unit `u` of any template is the committed circuit's unit `u`. -/
noncomputable def unheld (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (record : Lean.Json)
    (W : Spot.Wires.Wiring) (X : (V.circuitOf θ F).Trace) : Finset (Fin (V.units θ F)) :=
  Finset.univ.filter fun u => ¬ ∀ template : String,
    Properties.Spot.WiredCorrect W (V.SameRow θ F c record W)
      (fun _ => u ∉ unbound (V.partsOf θ F) (V.Opens θ F c) X) (template, u.val)

/-- **The wires' claim** in the play of `σ` that ended with `o`, at composite `k`: for every case of the policy in
the play's messages (`PolicyCase`) and every wiring its record names (`RecordWiring`), at most a `2ε` share of the
committed units are not held in the extractor's transcript (`unheld`): unbound in it, or reading a row that is not
the row the wiring gives it. `ε` is sampled proofs' published share.

What it does not say:

* **Anything of a record whose rows are its members'.** A record of several members names a member's roots by the
  member's name and numbers a template's units inside the member; this reads a root by the row's name alone and a
  unit by its index in the committed circuit, so of such a record it says sampled proofs' claim at twice its share
  and no more.
* **That a unit's rows are the Program's gates.** It holds a wired row to its writer's row; that a unit's file
  computes its template is not in it.
* **That the wiring is well formed.** It reads the wires of the record's text as they stand, without
  `Spot.Wires.check`, whose templates the text does not hold.
* **Anything of a composite whose policy check is another one.** The case is the one `NciCheck.Protocol.protocol`
  decodes (`PolicyCase`); at a composite whose list has another policy check in its place no case decodes, and the
  claim holds of every play. -/
def WiresHold (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (o : Exchange) : Prop :=
  ∀ p r e, PolicyCase k o.sent p r e → ∀ W, RecordWiring r.record W →
    ((V.unheld θ F c r.record W (V.extract θ F c σ)).card : ℝ) ≤ 2 * (V.certOf θ).ε * V.units θ F

/-- **The wires' target at composite `k`**: at all `θ`, `F` and `c`, for every strategy, the probability that
sampled proofs' verdict at `k` accepts while the wires' claim fails (`WiresHold`) is at most twice sampled proofs'
error at that strategy. No compiled check holds a sampled unit's wired rows to their writers' rows today
(`Spot.Wires.Summary.drawnOk` states that check, and no protocol calls it), so no theorem proves it. -/
def WiresSound (k : Verity.Registration.Composite) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.WiresHold k θ F c σ o) (V.game θ F c) σ ≤
      2 * V.error .sampled θ F c σ

end Vocab

end Verity
