import Flock.HmRow
import Flock.Norm
import Flock.Verify
import Proofs.Flock.Soundness.Audit.OneStage

/-!
# flock-verify's acceptance, as an event of the audit

`Prog.flock_headline` bounds an event of the audit as Lean defines it, whose outcome is that audit's verdict and the
draw. What the verifier of record decides is flock-verify's verdict on the session directory a run leaves: the record the coin
server wrote (`session.json`) and the proof file it received for each table's rep. This file states that verdict as an
event, on an audit whose outcome carries the transcript.

* `Inputs`: what `flock-verify verify` reads besides the session directory, its flags' values.
* `Transcript`: the session directory, the record's text and the proof files in `spec.repStreams` order.
* `Accepts I t`: flock-verify's verdict on `t` (`Main.verifyCmd`, an `hm96Rows` statement): the session's statement set
  up from the inputs and the record's draw (`Stmt.setupTables`), and `Flock.verify` accepting the record and proofs
  against it. `verifyCmd` refuses more: a keyed coin tree's reused key (`CoinTree.checkFresh`), which only shrinks the
  event. A run of several sessions sets each drawn session up at its own draw.
* `auditLive L Reg live`: the audit as it runs, the registration, then the verifier's draw, then the live session
  `live S R` that ends in the transcript, and its outcome the registration, the draw and the transcript.
* `ExecRefines`: the open guarantee R11 at one strategy pair, the probability that flock-verify accepts in the live
  audit is at most the probability that the audit as Lean defines it accepts, plus a live event `bad` (for `bad = ⊥`,
  the skeleton's `hExec`).
-/

namespace FlockSoundness.Discharge.Exec

open Lean (Json)
open Game Audit
open scoped ENNReal

/-- What `flock-verify verify` reads besides the session directory: the statement's tags (`--statement`, as `verifyCmd`
passes them to `Stmt.setupTables`, so after any rewrite of the tags by `--coins`), the circuit and public files
(`--circuit`, `--public`), the coins (`--coins`), the tables per session (`--session-tables`), the lookup tables the
circuit pins, and the verifier's own partition, program, work, stratified `K` and closure map. -/
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

/-- A session directory: the record's text (`session.json`) and the proof files, in `spec.repStreams` order
(`proofs[2i + r]` is table `i`'s rep `r`). -/
structure Transcript where
  record : String
  proofs : Array ByteArray

/-- A record's `unit_draw`, when it is a draw (not absent, not null), rebuilt (`Flock.norm`): `Main.recordDraw`, which
the executable's root module defines and no library exports. -/
def recordDraw (record : String) : Option Json :=
  match Flock.JsonParse.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some (Flock.norm d)
    | .error _ => none
  | .error _ => none

/-- The session's statement flock-verify sets up for a record: `Main.buildSession` on an `hm96Rows` statement, at the
record's draw. -/
def setupOf (I : Inputs) (record : String) : Flock.V (Flock.SessionSpec × Array Flock.Stmt) :=
  Flock.Stmt.setupTables I.tags I.circuitFile I.publicFile I.coins I.count I.tables (recordDraw record) I.partition I.program
    I.work I.stratified I.closure

/-- **flock-verify accepts the transcript**: the session's statement sets up from the inputs and the record's draw,
and `Flock.verify` accepts the record and the proof files against each table's statement. -/
def Accepts (I : Inputs) (t : Transcript) : Prop :=
  ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
    Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

variable {n : ℕ} {Reg : Type}

/-- **The audit as it runs**: the prover registers `R`; the verifier draws `S` by `L`; the live session `live S R`
(the prover against the challenger) ends in the session directory. Output: the registration, the draw and the
transcript. -/
def auditLive (L : Law n) (Reg : Type) (live : Finset (Fin n) → Reg → Game Transcript) :
    Game (Reg × Finset (Fin n) × Transcript) :=
  .send Reg fun R => .coin L.Ω fun ω => (live (L.draw ω) R).map fun t => (R, L.draw ω, t)

/-- flock-verify accepts the live audit's transcript. -/
def LiveAccepts (I : Inputs) (o : Reg × Finset (Fin n) × Transcript) : Prop := Accepts I o.2.2

/-- **R11 at one strategy pair**: the event `acc` of the live game `G` played by `τ` is no likelier than the audit `Gm`
as Lean defines it, played by `σ`, accepts, up to the live event `bad`. -/
def ExecRefines {β : Type} (G : Game β) (τ : Strategy G) (acc bad : β → Prop) (Gm : Game (Bool × Finset (Fin n)))
    (σ : Strategy Gm) : Prop :=
  prob acc G τ ≤ prob (fun o => o.1 = true) Gm σ + prob bad G τ

end FlockSoundness.Discharge.Exec
