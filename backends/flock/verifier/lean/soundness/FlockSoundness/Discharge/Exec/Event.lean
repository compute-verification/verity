import Flock.HmRow
import Flock.Verify
import FlockSoundness.Audit.OneStage
import FlockSoundness.Protocol.Discharge.Exec.Event

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
  event, and a run of several sessions reuses a setup for a draw of the same canonical form.
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

variable {n : ℕ} {Reg : Type}

/-- flock-verify accepts the live audit's transcript. -/
def LiveAccepts (I : Inputs) (o : Reg × Finset (Fin n) × Transcript) : Prop := Accepts I o.2.2

/-- **R11 at one strategy pair**: the event `acc` of the live game `G` played by `τ` is no likelier than the audit `Gm`
as Lean defines it, played by `σ`, accepts, up to the live event `bad`. -/
def ExecRefines {β : Type} (G : Game β) (τ : Strategy G) (acc bad : β → Prop) (Gm : Game (Bool × Finset (Fin n)))
    (σ : Strategy Gm) : Prop :=
  prob acc G τ ≤ prob (fun o => o.1 = true) Gm σ + prob bad G τ

end FlockSoundness.Discharge.Exec
