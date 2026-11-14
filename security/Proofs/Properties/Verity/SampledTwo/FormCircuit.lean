module

public import Proofs.Properties.Verity.SampledTwo.FormAgreed
public import Proofs.Properties.Verity.SampledTwo.FormSetup
public import Proofs.Properties.Verity.SampledTwo.FormCount
public import Proofs.Properties.Verity.SampledTwo.FormStmts
public import Proofs.Properties.Verity.SampledTwo.FormLast
public import Proofs.Properties.Verity.SampledTwo.FormLink
public import Proofs.Properties.Verity.SampledTwo.FormPrint
public import Proofs.Properties.Verity.SampledTwo.FormDraw
public import Proofs.Properties.Verity.SampledTwo.FormBytes
public import Proofs.Properties.Verity.SampledTwo.FormMember
public import Proofs.Properties.Verity.SampledTwo.FirstEarly
public import Proofs.Properties.Verity.SampledTwo.Reached

/-!
# The messages' piece at the form's reading, at the circuit statement's table

`agreed_form` (`FormAgreed.lean`) takes the locator's table names `tab` and, of the sessions' records, three
hypotheses: `SessionCoins.LogRising`, `SessionCoins.RecordsCover` and `SessionCoins.MsgsOfRecords`. Every session of
the form is set up at the circuit statement, whose one table is `Flock.Tags.circuit.table`. This module states the
piece at that table, where the three are theorems, and with them the statements' count and the statements' round.

* `StmtsCount` is `stmtsCount_of` (`FormCount.lean`), from `HeldOfAt` and `PartsHeldAt`.
* `SessionCoins.LogRising` is `SessionCoins.logRising_of`, from the log the check read (`SessionCoins.located_rising`,
  `SessionCoins.interiorReads`) and `SessionCoins.InteriorLogged`.
* `SessionCoins.RecordsCover` is `SessionCoins.recordsCover_of`, from `StmtsRead` and `StmtsCount`.
* `SessionCoins.MsgsOfRecords` is `SessionCoins.msgsOfRecords_form` (`FormSetup.lean`), from the join's fields,
  `StmtsCount`, `SessionCoins.StmtsPrint` and `SessionCoins.StmtsFit`. That a session the verdict accepts has
  messages is `SessionLive.sessionMsgs_of_verify` (`SessionLive.lean`), with no hypothesis on the case: the circuit
  statement retains rounds, so the record the verdict decoded kept each round's bytes.
* `SessionCoins.StmtsPrint` is `stmtsPrint_of` (`FormPrint.lean`), with no hypothesis: the parameters are read from
  the tree `Core.canonicalJson` returns, which is printable.
* `StmtsFixed` is `stmtsFixed_of_each` (`StmtsRound.lean`), from `StmtsCount`, `StmtAtFixed` and each case being the
  one sent (`CaseIsSent`).
* `StmtAtFixed` is `stmtAtFixed_of` (`FormStmts.lean`), from `StmtsRead`, `StmtsCount`, `CaseIsSent`, and
  `MemberAtFixed` and `DrawAtFixed`.
* `MemberAtFixed` is `memberAtFixed_of_coins` (`FormMember.lean`), from `StmtsRead`, `StmtsCount`,
  `SessionCoins.StmtsFit`, `StageCoinsOne` and `MembersApart`: the model's check holds the two plays' statements'
  public files to one part of the second stage's population at one pair of stage coins, the form's check holds each
  member's public file to the member's units, and no two members of the parameters have the same units.
* `MembersApart` is `membersApart` (`FormMember.lean`), with no hypothesis: the parameters' reading refuses a member
  with no unit and a unit in two members' ranges.
* `DrawAtFixed` is `drawAtFixed_of_coins` (`FormDraw.lean`), from `StmtsRead`, `StmtsCount`,
  `SessionCoins.StmtsFit` and `StageCoinsOne`: the model's check holds the two plays' samples to one canonical
  text at one pair of stage coins, and a record's sample is fixed by its canonical text.
* `StageCoinsOne` is `stageCoinsOne_of_play` (`FormDraw.lean`), from the join's fields and `PlayCoinsOne`: an
  accepted play's statements pass the model's check at any stage coins that carry the bytes its samples read
  (`stmtOk_form`).
* `PlayCoinsOne` is `playCoinsOne_of_shown` (`FormBytes.lean`), at the run's locator, which puts only openings past
  the second stage's last round (`opensFrom_run`): from `FirstShownEarly` and `SecondShown`, each stage's sample
  being its family's after one digest (`CoinMap.StageDraws`) and `SessionCoins.InteriorLogged`. Each stage's bytes
  start the coins the log before the statements' round shows, so one of two plays' bytes starts the other's.
* `FirstShownEarly` is `firstShownEarly` (`FirstEarly.lean`), from `CaseIsSent` and `Good` holding the check to
  have read a log of both sessions and found it the device's (`ReadIsDevice`): the form refuses an interior audit on
  the boundary registration's window, and on two windows the first session's read holds no opening.
* `LastFixed` is `lastFixed_of` (`FormLast.lean`), from `StmtAtFixed`, the sessions the verdict accepted
  (`SessionCoins.verdictsAt_of`) and each session's evidence on the case (`SessionCoins.logCloses_run`).
* `LinkFixed` is `linkFixed_of` (`FormLink.lean`), from `StmtAtFixed`, the sessions the verdict accepted and the
  members' check on the log the check read (`SessionCoins.membersNamed`).
* `agreed_form`'s `hR`, that the frame's run on agreeing coins has all the verifier's messages where its messages are
  the check's wherever it has one, is `reaches_frame` (`Reached.lean`), from `PlayAccepts`, `SessionMsgs`
  (`reachedAccepts`) and the frame's statement being the one the check reads (`stmtAt`).

The reading's statements and messages are taken to be the case's (`hS`, `hM`), which holds by definition at
`SessionMsgs.sessionReading`.

**Not established here**: `StmtsRead`, `PartsHeldAt` and `PinHeldAt`, each a hypothesis.
`SessionCoins.StmtsFit` is `SessionCoins.stmtsFit_of` (`FormSetup.lean`): the form refuses evidence of `2^64`
statements or more, and a play one of whose sessions' statements is `2^64` bytes or more.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The messages fixed where the coins agree, at the form's reading and the circuit statement's table**:
`agreed_form` at the locator whose table names are all `Flock.Tags.circuit.table`, where the reading's statements
and messages are the case's. Its hypotheses on the sessions' records are theorems there: the log's rounds rise
(`SessionCoins.logRising_of`), the check read a record for every template (`SessionCoins.recordsCover_of`), and each
record is its session's, accepted at its statement's setup (`SessionCoins.msgsOfRecords_form`), what each member's
statement prints of the member reading back (`stmtsPrint_of`). The evidence holds one statement a template
(`stmtsCount_of`), so the statements' message is fixed where each template's statement is (`stmtsFixed_of_each`),
which it is where the member it names and the sample its record prints are (`stmtAtFixed_of`), both where
one pair of stage coins carries the bytes both plays' samples read (`memberAtFixed_of_coins`, `drawAtFixed_of_coins`,
`stageCoinsOne_of_play`), which one does where each stage's bytes start the coins the log shows before every
opening (`playCoinsOne_of_shown`), as the first stage's do at the cut's read (`firstShownEarly`); and where its
statement is fixed, a session's first message is fixed by the string shown at its first round (`linkFixed_of`) and
its last by its evidence (`lastFixed_of`). The frame's run on agreeing coins, its messages the check's wherever it
has one, has every message (`reaches_frame`), as the accepted play's own prover's run does. -/
theorem agreed_circuit (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (ℓ₁ ℓ₂ : String) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hL : L = SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) fun _ => Flock.Tags.circuit.table) (q : ℕ)
    (hne : ℓ₁ ≠ ℓ₂) (hG : ∀ o, Good o → ReadIsDevice o)
    (hDraw₁ : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind firstDraw) ℓ₁)
    (hSecond : SecondShown C Good)
    (hDraw : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hlog : SessionCoins.InteriorLogged C Good) (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ)
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg)
    (hM : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j)
    (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') (hof : HeldOfAt A C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hparts : PartsHeldAt A C)
    (hsent : CaseIsSent C) (hC : CaseCommits C) (hx : stmtOf c = some x) :
    MessagesFixedAgreed V A x θ F c σ (formFrame V A x θ F c σ C S dev Good L) (formReading A C S dev)
      (FormAccepts C) Good (formMiss V A x θ F c σ C S dev Good L q) := by
  have hcount : StmtsCount A C Good := stmtsCount_of Good hof hparts
  have hfit : SessionCoins.StmtsFit C Good := SessionCoins.stmtsFit_of C Good
  have hsz : SessionCoins.HeldSize := fun _ _ _ hh => held_size hh
  have hprint : SessionCoins.StmtsPrint C Good := stmtsPrint_of C Good
  have hcoins : PlayCoinsOne A C Good L := by
    rw [hL]
    exact playCoinsOne_of_shown (opensFrom_run A _ _) hof hparts (firstShownEarly C Good hG hsent) hDraw₁ hSecond hDraw
      hlog
  have hone : StageCoinsOne A C S Good L := stageCoinsOne_of_play hparts hof hst hrounds hscope hpin hcoins
  have hdraw : DrawAtFixed A C Good L := drawAtFixed_of_coins hS hst hcount hfit hone
  have hmemb : MemberAtFixed A C Good L := memberAtFixed_of_coins hS hst hcount hfit hone (membersApart C)
  have hfix : StmtAtFixed A C Good L := stmtAtFixed_of A C S Good L hS hsent hst hcount hmemb hdraw
  have hcov : SessionCoins.RecordsCover A C Good := SessionCoins.recordsCover_of hsz hS hst hcount
  have hclose : SessionCoins.LogCloses A C Good L := by
    rw [hL]
    exact SessionCoins.logCloses_run A C Good (CoinMap.stageLocate ℓ₁ ℓ₂) _ hsz hlog hcov
  have hverd : SessionCoins.VerdictsAt A C Good fun _ => Flock.Tags.circuit.table :=
    SessionCoins.verdictsAt_of hS hof hst hrounds hscope hcount hprint hfit
  have hLast : LastFixed A C S Good L := lastFixed_of A C S Good L _ hM hverd hclose hfix
  have hLink : LinkFixed A C S Good L := linkFixed_of A C S Good L hM hsz hverd hfix
  exact agreed_form V A x θ F c σ C S dev P' (fun _ => Flock.Tags.circuit.table) ℓ₁ ℓ₂ Good L hL q hne hSecond hDraw
    hsz hlog hθ
    (SessionCoins.logRising_of (SessionCoins.located_rising A C _ hsz) hlog (SessionCoins.interiorReads C hsz))
    hcov
    (SessionCoins.msgsOfRecords_form hS hM hof hst hrounds hscope hcount hprint hfit) hplay hsess hof hst hrounds
    hscope
    (stmtsFixed_of_each A C S Good L hS hsent hcount hfix) hLink hLast hC hx
    fun ω r hA _ hc hm =>
      reaches_frame (formFrame V A x θ F c σ C S dev Good L)
        (stmtAt V A x θ F c σ (formFrame V A x θ F c σ C S dev Good L) C S dev hC hx)
        (reachedAccepts A x C S dev P' (sourcesFit A C) hplay hsess) ω r hA hc hm

end Verity.SampledTwo
