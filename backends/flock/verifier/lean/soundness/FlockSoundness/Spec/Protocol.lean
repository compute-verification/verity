import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Basic.Real.Basic
import Flock.Zk
import Flock.ProvedScope

/-!
# C-Flock's spec: the protocol

The definitions C-Flock's named assumptions (`FlockSoundness.Spec.Assumptions`) and claims are stated over: what
`flock-verify verify` accepts, over the executable verifier's own definitions (`Flock.*`), and the interaction model
the assumptions' finders are games of. It imports the executable verifier and Mathlib alone (the `layers` rule in
`lean-audit.json`).

* `Inputs`, `Transcript`: what `flock-verify verify` reads, its flags' values and the session directory.
* `Accepts zk I t`: `Main.verifyCmd`'s verdict on `t`, with `--zk` when `zk`: `ProvedScope.check` on the statement, the
  session's statement set up from the inputs at the record's draw (`Stmt.setupTables`), under `--zk` every table's
  `Zk.shapeOk`, and the verdict (`Flock.verify`, under `--zk` `Flock.Zk.verify`). `verifyCmd` refuses more (a draw by a
  law no soundness claim draws by, registered reads, the identifier, a reused coin-tree key), which only shrinks the event.
* `Game`, `Strategy`, `prob`, `expect`, `prCoin`: a public-coin interaction, a deterministic prover, and the probability
  and expectation over the verifier's uniform coins.
-/

namespace FlockSpec

open Lean (Json)
open scoped ENNReal

/-- What `flock-verify verify` reads besides the session directory: `--statement`'s tags, the circuit and public files,
`--coins`, `--session-tables`, the lookup tables the circuit pins, and the verifier's partition, program, work,
stratified `K` and closure map. -/
structure Inputs where
  tags : Flock.Tags
  circuitFile : ByteArray
  publicFile : ByteArray
  coins : Flock.Coins
  count : ℕ
  tables : List (String × ByteArray)
  partition : Option Json
  program : Option ByteArray
  work : Option (ℕ × Array (String × ℕ × ℕ))
  stratified : Option ℕ
  closure : Option (Std.HashMap ℕ (Array ℕ))

/-- A session directory: the record's text (`session.json`) and the proof files in `spec.repStreams` order. -/
structure Transcript where
  record : String
  proofs : Array ByteArray

/-- The record's `unit_draw` when it is a draw (`Main.recordDraw`). -/
def recordDraw (record : String) : Option Json :=
  match Json.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some d
    | .error _ => none
  | .error _ => none

/-- The tags a session is set up and checked under: `--statement`'s, under `--zk` with the `--zk` identity's hooks at
the circuit's `mask_log` and the session's table count. -/
def tagsOf (zk : Bool) (I : Inputs) : Flock.Tags :=
  if zk then Flock.Zk.tags I.tags (Flock.Zk.maskLogOf I.circuitFile) I.count else I.tags

/-- The coins a session is set up on: `--coins`, under `--zk` the keyed coin tree of the session's tables. -/
def coinsOf (zk : Bool) (I : Inputs) : Flock.Coins :=
  if zk then .tree (Flock.Zk.coinSpec (I.tags.tableNames I.count)) else I.coins

/-- **`flock-verify verify` accepts the session directory `t`**, with `--zk` when `zk`. -/
def Accepts (zk : Bool) (I : Inputs) (t : Transcript) : Prop :=
  Flock.ProvedScope.check (tagsOf zk I) zk I.circuitFile I.publicFile I.tables I.count false = .ok () ∧
    ∃ spec sts, Flock.Stmt.setupTables (tagsOf zk I) I.circuitFile I.publicFile (coinsOf zk I) I.count I.tables
        (recordDraw t.record) I.partition I.program I.work I.stratified I.closure = .ok (spec, sts) ∧
      if zk then sts.forM Flock.Zk.shapeOk = .ok () ∧
        Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()
      else Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

/-! ## Interactions -/

/-- The verifier's side of a public-coin interaction: the prover sends a message of its choice (`send`), the verifier
draws a uniform coin after every earlier message is fixed (`coin`), or the interaction ends with an output (`ret`). -/
inductive Game (α : Type) : Type 1 where
  | ret : α → Game α
  | send (M : Type) : (M → Game α) → Game α
  | coin (C : Type) [Fintype C] [Nonempty C] : (C → Game α) → Game α

/-- A deterministic prover: a message at every `send` node, as a function of the history the node's position encodes. -/
def Strategy {α : Type} : Game α → Type
  | .ret _ => PUnit
  | .send _ k => (m : _) × Strategy (k m)
  | @Game.coin _ C _ _ k => (c : C) → Strategy (k c)

open Classical in
/-- The probability, over the verifier's independent uniform coins, that the output satisfies `p`. -/
noncomputable def prob {α : Type} (p : α → Prop) : (g : Game α) → Strategy g → ℝ≥0∞
  | .ret a, _ => if p a then 1 else 0
  | .send _ k, s => prob p (k s.1) s.2
  | @Game.coin _ C iF _ k, s => ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob p (k c) (s c)

/-- The expectation of `f` over the verifier's uniform coins. -/
noncomputable def expect {α : Type} (f : α → ℝ) : (g : Game α) → Strategy g → ℝ
  | .ret a, _ => f a
  | .send _ k, s => expect f (k s.1) s.2
  | @Game.coin _ C iF _ k, s => (∑ c : C, expect f (k c) (s c)) / (@Fintype.card C iF)

open Classical in
/-- The probability that a uniform element of a finite type satisfies `p`. -/
noncomputable def prCoin {C : Type} [Fintype C] (p : C → Prop) : ℝ≥0∞ :=
  ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * if p c then 1 else 0

end FlockSpec
