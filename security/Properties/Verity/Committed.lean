module

public import Properties.Verity.Game
public import Spot.RecordText

/-!
# What ties the inference-only policy's clause to the committed files

The game's clause for the inference-only policy (`Verity.Vocab.Holds` at `.inferenceOnly`) is on the Program the
policy's compiled check judges, and is proved of the case as sent. It says nothing of the committed files until two
statements hold, one proved and one a named `Prop` stated and not proved.

`Flock.Bridge.GateARow`, `Γ`'s unit, proved (`Flock.Bridge.gateARow`): in the game's circuit of a committed file
(`Flock.Bridge.circuitOfFile`) every unit has one gate for each of its file's rows. The policy's `Γ` counts rows of the
committed circuit a block, and PoUW's claim counts that circuit's gates (`Verity.Vocab.computed`), so with it the two
count one thing.

`Vocab.ReadsByPath`, the wires: a wire is one commitment. Its writer's output field is one tree (one root for each
writer template and output field), and a reader's wired row is a reference with a path into that tree, at the leaf
the wire's closed form gives (`Circuits.Wire.leafAt`). Flock's instance opens the reader's row there, under the
statement's roots, as `Sessions.opens` opens every read: one value at one position, so a reader's row is its
writer's row and no comparison of two rows is left.
-/

@[expose] public section

namespace Verity

open ResourceAccounting Core

/-- **The record names the wiring `W`**: the record's `window.circuit` is the SHA-512 of a text, the circuit
record's, that parses to a document whose cuts and wires SPOT's reader reads as `W` (`Spot.RecordText.wiring`). -/
def RecordWiring (record : Lean.Json) (W : Spot.Wires.Wiring) : Prop :=
  ∃ (text : String) (doc : Lean.Json),
    InferenceOnlyVerifier.Theta.strAt (InferenceOnlyVerifier.Theta.getD record "window") "circuit" =
        some (Flock.hexOf (Flock.Sha512.hash text.toUTF8)) ∧
      Flock.JsonParse.parse text = .ok doc ∧ Spot.RecordText.wiring doc = .ok W

/-- **The record names `R` as the root of `name`**: one of its `roots` has the port `name` and the root `R`, in hex. -/
def RecordRoot (record : Lean.Json) (name : String) (R : Hash.Digest) : Prop :=
  ∃ roots, record.getObjValD "roots" = .arr roots ∧ ∃ e ∈ roots,
    e.getObjValD "port" = .str name ∧ e.getObjValD "root" = .str (Flock.hexOf ⟨R.bytes.toArray⟩)

/-- **A writer's output tree's name** in the record: the writer template, `/` and the output field. -/
def fieldRoot (template field : String) : String :=
  template ++ "/" ++ field

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **Reads by path open at the writer's leaf**, at composite `k`: on every play SPOT's verdict at `k` accepts, for
every case of the policy (`PolicyCase`) and every wiring its record names (`RecordWiring`), each reader unit `u`'s
wired row `row` reads, under the root the record names for its writer's output field (`fieldRoot`), a registered
place the writer unit `Circuits.Wire.leafAt` gives writes. It is `FlockProofs.Registered.check_ok` extended to wired
ports, lifted to the record: `Flock.Registered.check` holds every link row to a wired port (`checkLinked`) and opens it
at `Port.readAt`'s closed form, the writer's leaf, under its registered root. -/
def ReadsByPath (k : Verity.Registration.Composite) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (o : Exchange), V.Verdict k .sampled o →
    ∀ p r e, PolicyCase k o.sent p r e → ∀ W, RecordWiring r.record W →
      ∀ w ∈ W.wires, ∀ (u row v : ℕ), w.leafAt u row = some v →
        ∀ R, RecordRoot r.record (fieldRoot w.writer w.out) R →
          ∃ (q : Place) (g h : Fin (V.circuitOf θ F).N) (gu hv : Fin (V.units θ F)),
            q.root = R ∧ V.place θ F c h = some q ∧ (V.partsOf θ F).unit h = some hv ∧ hv.val = v ∧
            (V.partsOf θ F).unit g = some gu ∧ gu.val = u ∧ h ∈ (V.circuitOf θ F).reads g

end Vocab

end Verity
