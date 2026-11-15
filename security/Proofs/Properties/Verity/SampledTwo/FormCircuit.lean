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
public import Proofs.Properties.Verity.SampledTwo.SessionSent

/-!
# The messages' piece at the form's reading, at the circuit statement's table

`agreed_form` (`FormAgreed.lean`) takes the locator's table names `tab` and, of the sessions' records, three
hypotheses: `SessionCoins.LogRising`, `SessionCoins.RecordsCover` and `SessionCoins.MsgsOfRecords`. Every session of
the form is set up at the circuit statement, whose one table is `Flock.Tags.circuit.table`. At that table the three
are theorems, and with them the statements' count and the statements' round, each where this list names it; this
module proves `SessionMsgs` at the sessions' replaying provers (`sessionMsgs_form`).

* `PartsHeldAt` is `partsHeldAt_of_accepts` (`AcceptsTop.lean`), from `HeldOfAt`: an accepted play's interior passed
  the run's check of its strata against its replay block's parts (`partsSizedAt_of_formAccepts`).
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
  `MemberAtFixed` and `SampleAtFixed`.
* `MemberAtFixed` is `memberAtFixed_of_coins` (`FormMember.lean`), from `StmtsRead`, `StmtsCount`,
  `SessionCoins.StmtsFit`, `StageCoinsOne` and `MembersApart`: the model's check holds the two plays' statements'
  public files to one part of the second stage's population at one pair of stage coins, the form's check holds each
  member's public file to the member's units, and no two members of the parameters have the same units.
* `MembersApart` is `membersApart` (`FormMember.lean`), with no hypothesis: the parameters' reading refuses a member
  with no unit and a unit in two members' ranges.
* `SampleAtFixed` is `sampleAtFixed_of_coins` (`FormDraw.lean`), from `StmtsRead`, `StmtsCount`,
  `SessionCoins.StmtsFit` and `StageCoinsOne`: the model's check holds the two plays' samples to one canonical
  text at one pair of stage coins, and a record's sample is fixed by its canonical text.
* `PlayCoinsOne` is `playCoinsOne_of_shown` (`FormBytes.lean`), at the run's locator, which puts only openings past
  the second stage's last round (`opensFrom_run`): from `FirstShownEarly` and `SecondShown`, each stage's sample
  being its family's after one digest (`CoinMap.StageSamples`) and `SessionCoins.InteriorLogged`. Each stage's bytes
  start the coins the log before the statements' round shows, so one of two plays' bytes starts the other's.
* `FirstShownEarly` is `firstShownEarly` (`FirstEarly.lean`), from `CaseIsSent` and `Good` holding the check to
  have read a log of both sessions and found it the device's (`ReadIsDevice`): the form refuses an interior audit on
  the boundary registration's window, and on two windows the first session's read holds no opening.
* `LastFixed` is `lastFixed_of` (`FormLast.lean`), from `StmtAtFixed`, the sessions the verdict accepted
  (`SessionCoins.verdictsAt_of`) and each session's evidence on the case (`SessionCoins.logCloses_run`).
* `LinkFixed` is `linkFixed_of` (`FormLink.lean`), from `StmtAtFixed`, the sessions the verdict accepted and the
  members' check on the log the check read (`SessionCoins.membersNamed`).
* `SessionMsgs` is `sessionMsgs_form`, at the sessions' replaying provers (`SessionMsgs.replayProver`), which
  `agreed_form` is taken at: from the join's fields and `PlayAccepts` at those provers (`sessionMsgs_replay`,
  `SessionSent.lean`). A replaying prover's messages are its session's on the case, each but the last answered with
  a coin, so the sequential prover sends them in the session's rounds.
* `agreed_form`'s `hR`, that the frame's run on agreeing coins has all the verifier's messages where its messages are
  the check's wherever it has one, is `reaches_frame` (`Reached.lean`), from `PlayAccepts`, `SessionMsgs`
  (`reachedAccepts`) and the frame's statement being the one the check reads (`stmtAt`).

The reading's statements and messages are taken to be the case's (`hS`, `hM`), which holds by definition at
`SessionMsgs.sessionReading`.

**Not established here**: `StmtsRead`, `PinHeldAt` and `StageCoinsOne` (`FormDraw.lean`), each a hypothesis.
`SessionCoins.StmtsFit` is `SessionCoins.stmtsFit_of` (`FormSetup.lean`): the form refuses evidence of `2^64`
statements or more, and a play one of whose sessions' statements is `2^64` bytes or more.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The replaying provers' messages are the ones the check read, at the form** (`SessionMsgs` at
`SessionMsgs.replayProver`): `sessionMsgs_replay` where the statements' count, their fit and the sessions' verdicts
are the form's (`stmtsCount_of`, `SessionCoins.stmtsFit_of`, `SessionCoins.verdictsAt_of`), at every play. -/
theorem sessionMsgs_form {A : Live₂} {C : CaseOf} [Fact (SampledNoneAt C)] {S : SessionReading}
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg)
    (hM : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j) (hof : HeldOfAt A C)
    (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hplay : PlayAccepts A C S (SessionMsgs.replayProver C)) :
    SessionMsgs A C S (SessionMsgs.replayProver C) := by
  have hcount : StmtsCount A C fun _ => True :=
    stmtsCount_of _ hof (partsHeldAt_of_accepts hof (partsSizedAt_of_formAccepts hof))
  have hfit : SessionCoins.StmtsFit C fun _ => True := SessionCoins.stmtsFit_of C _
  exact sessionMsgs_replay hS hM hcount hfit
    (SessionCoins.verdictsAt_of hS hof hst hrounds hscope hcount (stmtsPrint_of C _) hfit) hplay

end Verity.SampledTwo
