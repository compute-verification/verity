module

public import Properties.Verity.Game
public import Properties.Spot.Labels
public import Spot.RecordText
public import Spot.OneStageZk

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

Two more, each a named `Prop` stated and not proved, are what Flock's checks give `ReadsByPath`: `LinksExact`, that a
link's leaves are one whole return field of one writer unit, from its first leaf (`Flock.HmLinks.checkUnit`), and
`LinkRowsByPath`, that each link row opens the registered root at its writer's leaf in the writer template's tree
(`Flock.Registered.check` on the reads `Flock.Registered.withTrees` gives).
-/

@[expose] public section

namespace Verity

open ResourceAccounting Core

/-- **The record names the wiring `W`**: the record's `window.circuit` is the SHA-512 of a text, the circuit
record's, that parses to a document whose wiring over a Program `P` is `W` (`Spot.RecordText.wiring`: the text's
cuts and the edges of `P`'s root, of a text that names `P` by its SHA-512). -/
def RecordWiring (record : Lean.Json) (W : Spot.Wires.Wiring) : Prop :=
  ∃ (text : String) (doc : Lean.Json) (P : Flock.Program.Prog),
    InferenceOnlyVerifier.Theta.strAt (InferenceOnlyVerifier.Theta.getD record "window") "circuit" =
        some (Flock.hexOf (Flock.Sha512.hash text.toUTF8)) ∧
      Flock.JsonParse.parse text = .ok doc ∧ Spot.RecordText.wiring P doc = .ok W

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
wired row `row` reads the value of its label (`Circuits.Wire.toLabel`, a `Verity.Label.wire`): under the root the
record names for the label's writer's output field (`fieldRoot`), a registered place the label's unit writes. It is `FlockProofs.Registered.check_ok` extended to tree
ports, lifted to the record: `Flock.Registered.check` holds every link row to a tree port (`checkLinked`) and opens it at
its writer's leaf in the writer template's tree, which the verifier's program gives (`Registered.withTrees`,
`HmLinks.treeLeaf`), under its registered root. -/
def ReadsByPath (k : Verity.Registration.Composite) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (o : Exchange), V.Verdict k .sampled o →
    ∀ p r e, PolicyCase k o.sent p r e → ∀ W, RecordWiring r.record W →
      ∀ w ∈ W.wires, ∀ (u row : ℕ) (reg writer field : String) (v : ℕ),
        w.toLabel reg u row = some (.wire reg writer field v) →
        ∀ R, RecordRoot r.record (fieldRoot writer field) R →
          ∃ (q : Place) (g h : Fin (V.circuitOf θ F).N) (gu hv : Fin (V.units θ F)),
            q.root = R ∧ V.place θ F c h = some q ∧ (V.partsOf θ F).unit h = some hv ∧ hv.val = v ∧
            (V.partsOf θ F).unit g = some gu ∧ gu.val = u ∧ h ∈ (V.circuitOf θ F).reads g

end Vocab

open Flock.HmLinks (nodes unitAt ownerAt count outGates) in
/-- **A link reads its writer's outputs exactly** (`Flock.HmLinks.checkUnit`): where a circuit's links parse and
`checkUnit` accepts reader `i` of the public file, every parameter leaf of the reader's unit that the program wires to
another unit's output lies in exactly one link, and every link's leaves are circuit inputs, or one whole return field
of one writer unit (`Flock.HmLinks.fieldSpans`), leaf for leaf: the link's offset is the field's first word in the
writer's output row (each field on `Flock.Program.wordsOf` words of its own), the field's words lie within the link
row's, and its leaves are read from the field's first leaf. -/
def LinksExact : Prop :=
  ∀ (c : Flock.Circuit) (P : Flock.Program.Prog) (links : Array Flock.HmLinks.Link) (popIdx : Array Nat) (i w : Nat),
    Flock.HmLinks.parse c = .ok links →
    Flock.HmLinks.checkUnit c P links popIdx (unitAt (nodes P)) (ownerAt (nodes P)) (count (nodes P)) i = .ok w →
    ∃ (u k m : Nat) (ops : Array Nat), popIdx[i]? = some u ∧ unitAt (nodes P) u = some (k, m) ∧
      Flock.Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m = .ok ops ∧
      (∀ t < ops.size, (ownerAt (nodes P) ops[t]!).isSome → ∃! l, l ∈ links ∧ l.lo ≤ t ∧ t < l.hi) ∧
      ∀ l ∈ links, l.hi ≤ ops.size ∧
        ((∀ t, l.lo ≤ t → t < l.hi → ownerAt (nodes P) ops[t]! = none) ∨
          ∃ (v kv mv : Nat) (og : Array Nat), ownerAt (nodes P) ops[l.lo]! = some v ∧
            unitAt (nodes P) v = some (kv, mv) ∧ outGates P kv mv = .ok og ∧
            ∃ f fw : Nat, (l.off, f, l.hi - l.lo, fw) ∈ Flock.HmLinks.fieldSpans P kv ∧
              l.off + fw ≤ (c.ports.getD l.port default).words ∧ ops.extract l.lo l.hi = og.extract f (f + (l.hi - l.lo)))

open Flock.HmLinks (nodes writerOf) in
/-- **Every link row is read by path at its writer's leaf** (`Flock.Registered.check`): where the verifier's registered
reads, their tree ports' reads given from its program (`Flock.Registered.withTrees`), accept a linked statement that
names each port once, every link row is a tree port of field `out` whose rows are its registered rows' width
(`Flock.Registered.Port.fits`), and every instance's row of it opens the registered root at the leaf of the instance's
unit's writer in the writer template's tree. -/
def LinkRowsByPath : Prop :=
  ∀ (c : Flock.Circuit) (pub : Flock.Public) (program : ByteArray) (P : Flock.Program.Prog)
    (own own' : Array Flock.Registered.Port) (links : Array Flock.HmLinks.Link) (t : Flock.Tables) (indices : Array Nat),
    Flock.Program.decode program = .ok P → Flock.Registered.namesOnce c = .ok () →
    Flock.Registered.withTrees c pub (some program) (some own) = .ok (some own') →
    Flock.Registered.check c pub (some own') true = .ok () →
    Flock.HmLinks.parse c = .ok links → pub.tables = some t → Flock.Registered.unitIndices pub = .ok indices →
    ∀ l ∈ links, ∃ reg ∈ own', ∃ tmpl : String, reg.name = (c.ports.getD l.port default).name ∧
      reg.tree = some (tmpl, Flock.Registered.OUT_FIELD) ∧ reg.fits (c.ports.getD l.port default) ∧
      ∀ i < pub.n, ∃ (v leaf : Nat) (positions : Array Nat) (paths : Array ByteArray),
        writerOf P (nodes P) (indices.getD i 0) l.lo = .ok (some v) ∧ Flock.HmLinks.treeLeaf P (nodes P) v = some (tmpl, leaf) ∧
        Flock.Registered.entry pub reg = .ok (positions, paths) ∧
        Flock.Registered.opens reg (Flock.Registered.tableRow pub t l.port (Flock.Registered.ref t c.ports.size l.port i))
          leaf (paths.getD (Flock.Registered.ref t c.ports.size l.port i) ByteArray.empty)

end Verity
