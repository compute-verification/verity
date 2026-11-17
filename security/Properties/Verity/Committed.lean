module

public import Properties.Verity.Game
public import Properties.Verity.GateARow
public import Properties.Spot.Wires
public import Spot.FilePart
public import Spot.RecordText
public import Spot.OneStageZk
public import VStarRecursive.Sampled

/-! What ties the inference-only policy's clause to the committed files: `GateARow` and `Vocab.WiresSound`, stated
and not proved. -/

@[expose] public section

namespace Verity

open ResourceAccounting
open scoped ENNReal

/-- **The record names the wiring `W`**: the record's `window.circuit` is the SHA-512 of a text, the circuit
record's, that parses to a document whose cuts and wires SPOT's reader reads as `W`
(`Spot.RecordText.wiring`: no template and no order, which no wire's rows read). -/
def RecordWiring (record : Lean.Json) (W : Spot.Wires.Wiring) : Prop :=
  ∃ (text : String) (doc : Lean.Json),
    InferenceOnlyCheck.Theta.strAt (InferenceOnlyCheck.Theta.getD record "window") "circuit" =
        some (Flock.hexOf (Flock.Sha512.hash text.toUTF8)) ∧
      Flock.JsonParse.parse text = .ok doc ∧ Spot.RecordText.wiring doc = .ok W

/-- **The record names `R` as the root of its row `name`**: one of its `roots` has the port `name` and the root
`R`, in hex. -/
def RecordRoot (record : Lean.Json) (name : String) (R : Hash.Digest) : Prop :=
  ∃ roots, record.getObjValD "roots" = .arr roots ∧ ∃ e ∈ roots,
    e.getObjValD "port" = .str name ∧ e.getObjValD "root" = .str (Flock.hexOf ⟨R.bytes.toArray⟩)

/-- **A reader row's name** in the record: the port's own name, or, of a port the wiring cuts into several rows
(`Spot.Wires.Wiring.cutsOf`), the port's name, `#` and the row. -/
def rowName (W : Spot.Wires.Wiring) (template port : String) (row : ℕ) : String :=
  if W.cutsOf template port = 1 then port else port ++ "#" ++ toString row

/-- **SPOT's case in what the prover sent composite `k`, at one-stage v2's parameters `p`**, decoded: one of the
subprotocols' cases is SPOT's component's, its parameters are `p` as the component reads them, and its commitment
is the registration `c`. -/
def SpotCase (k : Verity.Registration.Composite) (s : Sent) (p : Spot.OneStageZk.Params)
    (c : Spot.OneStage.Commitment) : Prop :=
  ∃ ps cm ev, Decoded s ps cm ev ∧
    ∃ x ∈ Properties.Composition.triples (s.components k) ps ev,
      x.1 = VStar.Recursive.Sampled.protocol.onBlobs (Core.Blobs.ofArray s.blobs) ∧
      (Core.canonicalJson x.2.1.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p) ∧
      Spot.OneStage.commitmentOf x.2.2.commitment = .ok c

/-- **A template's unit in the population**, at the partition's ranges `rs`: the wiring's unit `u` of template `t` is
unit `first + u` of the one range of `rs` that is `t`'s, and none where `rs` holds no range of `t` or several, or
`u` is outside its count. -/
def popUnit (rs : Array (String × ℕ × ℕ)) (t : String) (u : ℤ) : Option ℕ :=
  match (rs.filter (·.1 == t)).toList with
  | [(_, lo, n)] => if 0 ≤ u ∧ u < (n : ℤ) then some (lo + u.toNat) else none
  | _ => none

/-- **A member's row's name** in the registration: the member's name, `/` and the row's name, as each of the
registration's roots is the member's whose name, then `/`, its port begins with (`Spot.OneStageZk.rootsOf`). -/
def memberRow (member row : String) : String :=
  member ++ "/" ++ row

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
(`ResourceAccounting.unbound` under the commitment's openings) and a wire holding as `SameRow` compares its two rows. A
wiring's unit `u` of any template is the committed circuit's unit `u`. -/
noncomputable def unheld (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (record : Lean.Json)
    (W : Spot.Wires.Wiring) (X : (V.circuitOf θ F).Trace) : Finset (Fin (V.units θ F)) :=
  Finset.univ.filter fun u => ¬ ∀ template : String,
    Properties.Spot.WiredCorrect W (V.SameRow θ F c record W)
      (fun _ => u ∉ unbound (V.partsOf θ F) (V.Opens θ F c) X) (template, u.val)

/-- **The wires' claim** in the play of `σ` that ended with `o`, at composite `k`: for every case of the policy in
the play's messages (`PolicyCase`) and every wiring its record names (`RecordWiring`), at most a `2ε` share of the
committed units are not held in the extractor's transcript (`unheld`): unbound in it, or reading a row that is not
the row the wiring gives it. `ε` is SPOT's published share.

What it does not say:

* **Anything of a record whose rows are its members'.** A record of several members names a member's roots by the
  member's name and numbers a template's units inside the member; this reads a root by the row's name alone and a
  unit by its index in the committed circuit, so of such a record it says SPOT's claim at twice its share
  and no more. The per-member form's claim, at its own wiring, is `MemberWiresHold`.
* **That a unit's rows are the Program's gates.** It holds a wired row to its writer's row; that a unit's file
  computes its template is not in it.
* **That the wiring is well formed.** It reads the wires of the record's text as they stand, without
  `Spot.Wires.check`, whose templates the text does not hold.
* **Anything of a composite whose policy check is another one.** The case is the one `InferenceOnlyCheck.Protocol.protocol`
  decodes (`PolicyCase`); at a composite whose list has another policy check in its place no case decodes, and the
  claim holds of every play.
* **Anything of a record that names no wiring.** Where the record's `window.circuit` is the SHA-512 of no text
  that parses to a wiring (`RecordWiring` holds of no `W`), the claim holds of the play and says nothing, not even
  SPOT's share. -/
def WiresHold (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (o : Exchange) : Prop :=
  ∀ p r e, PolicyCase k o.sent p r e → ∀ W, RecordWiring r.record W →
    ((V.unheld θ F c r.record W (V.extract θ F c σ)).card : ℝ) ≤ 2 * (V.certOf θ).ε * V.units θ F

/-- **The wires' target at composite `k`**: at all `θ`, `F` and `c`, for every strategy, the probability that
SPOT's verdict at `k` accepts while the wires' claim fails (`WiresHold`) is at most twice SPOT's
error at that strategy. The compiled check holds a sampled proof block's wired rows to their writers' rows only in the
per-member form, at its own wiring (`Spot.OneStageZk.wiresHeld`). The one statement's parameters name no wiring
(`Spot.OneStageZk.paramsOfJson_wiring`), and nothing checks the record's, so no theorem proves it. -/
def WiresSound (k : Verity.Registration.Composite) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.WiresHold k θ F c σ o) (V.game θ F c) σ ≤
      2 * V.error .sampled θ F c σ

/-- **The reader's row is its writer's row under the per-member registration**: at each row's population unit, member
and root, every accepted opening of a leaf the reader's unit reads and of one the writer's unit writes hold the
same bit where both hold one; a row of no unit, member or root is compared with nothing. -/
def MemberSameRow (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (record : Lean.Json) (W : Spot.Wires.Wiring)
    (rs ms : Array (String × ℕ × ℕ)) (x : String × ℕ × String × ℕ) (y : String × ℤ × String) : Prop :=
  ∀ (gx gy : ℕ) (mx my : String × ℕ) (Rx Ry : Hash.Digest) (p q : Place) (o o' : Hash.RowLeaf.Opening),
    popUnit rs x.1 x.2.1 = some gx → Spot.OneStageZk.memberAt ms gx = some mx →
    popUnit rs y.1 y.2.1 = some gy → Spot.OneStageZk.memberAt ms gy = some my →
    RecordRoot record (memberRow mx.1 (rowName W x.1 x.2.2.1 x.2.2.2)) Rx →
    RecordRoot record (memberRow my.1 y.2.2) Ry →
    V.ReadsLeaf θ F c Rx gx p → V.WritesLeaf θ F c Ry gy q →
    Hash.RowLeaf.accepts sha512 sha512 p.verifier p.root p.rank o →
    Hash.RowLeaf.accepts sha512 sha512 q.verifier q.root q.rank o' →
    ∀ (i : ℕ) (b b' : Bool), o.row[i]? = some b → o'.row[i]? = some b' → b = b'

open Classical in
/-- **The population's units the wiring `W` doesn't hold** in the transcript `X`: those `X` doesn't bind and those
mis-wired as `MemberSameRow` compares rows. The population's unit `g` is the committed circuit's unit `g`. -/
noncomputable def memberUnheld (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (record : Lean.Json)
    (W : Spot.Wires.Wiring) (rs ms : Array (String × ℕ × ℕ)) (X : (V.circuitOf θ F).Trace) :
    Finset (Fin (V.units θ F)) :=
  Finset.univ.filter fun g => g ∈ unbound (V.partsOf θ F) (V.Opens θ F c) X ∨
    ∃ (t : String) (u : ℕ), popUnit rs t u = some g.val ∧
      (t, u) ∈ Properties.Spot.miswired W (V.MemberSameRow θ F c record W rs ms)

/-- **The wires' claim at the per-member form's wiring**: for every SPOT case whose parameters name members and a
wiring, at most a `2ε` share of the committed units are not held in the extractor's transcript (`memberUnheld`).
It does not say the wiring is the policy's record's, and says nothing without members or a wiring. -/
def MemberWiresHold (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (o : Exchange) : Prop :=
  ∀ (p : Spot.OneStageZk.Params) (sc : Spot.OneStage.Commitment), SpotCase k o.sent p sc →
    ∀ (ms : Array Spot.OneStageZk.Member) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary),
      p.members = some ms → p.wiring = some (W, S) →
      ∀ (prog : Spot.Program) (rs : Array Flock.TemplateQuery.TRange), Spot.Program.parse p.own.program = .ok prog →
        Spot.Partition.evaluate prog (Spot.getD p.own.partition "query") = .ok rs →
        ((V.memberUnheld θ F c sc.record W (rs.map fun r => (r.template, r.first, r.count))
            (ms.map fun m => (m.name, m.first, m.count)) (V.extract θ F c σ)).card : ℝ) ≤
          2 * (V.certOf θ).ε * V.units θ F

/-- **The wires' target at the per-member form's wiring, at composite `k`**: SPOT's verdict accepts while
`MemberWiresHold` fails with probability at most twice SPOT's error. No theorem lifts the compiled check's hold
on the sampled proof blocks' wired rows to the population, so none proves it. -/
def MemberWiresSound (k : Verity.Registration.Composite) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.MemberWiresHold k θ F c σ o) (V.game θ F c) σ ≤
      2 * V.error .sampled θ F c σ

end Vocab

end Verity
