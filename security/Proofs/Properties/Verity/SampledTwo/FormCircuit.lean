module

public import Proofs.Properties.Verity.SampledTwo.FormAgreed
public import Proofs.Properties.Verity.SampledTwo.FormSetup
public import Proofs.Properties.Verity.SampledTwo.FormCount

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
  `StmtsCount`, and `SessionCoins.StmtsPrint`, `SessionCoins.StmtsFit` and `SessionCoins.SessionsLive`.
* `StmtsFixed` is `stmtsFixed_of_each` (`StmtsRound.lean`), from `StmtsCount`, `StmtAtFixed` and each case being the
  one sent (`CaseIsSent`).

The reading's statements and messages are taken to be the case's (`hS`, `hM`), which holds by definition at
`SessionMsgs.sessionReading`.

**Not established here**: `StmtsRead`, `PartsHeldAt`, `SessionCoins.StmtsPrint`, `SessionCoins.StmtsFit`,
`SessionCoins.SessionsLive`, `StmtAtFixed`, `LinkFixed` and `LastFixed`, each a hypothesis, and `agreed_form`'s `hR`.
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
record is its session's, accepted at its statement's setup (`SessionCoins.msgsOfRecords_form`). The evidence holds one
statement a template (`stmtsCount_of`), so the statements' message is fixed where each template's statement is
(`stmtsFixed_of_each`). -/
theorem agreed_circuit (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (ℓ₁ ℓ₂ : String) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hL : L = SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) fun _ => Flock.Tags.circuit.table) (q : ℕ)
    (hne : ℓ₁ ≠ ℓ₂) (hSecond : SecondShown C Good)
    (hDraw : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hlog : SessionCoins.InteriorLogged C Good) (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ)
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg)
    (hM : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j)
    (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') (hof : HeldOfAt A C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hparts : PartsHeldAt A C)
    (hprint : SessionCoins.StmtsPrint C Good) (hfit : SessionCoins.StmtsFit C Good)
    (hlive : SessionCoins.SessionsLive C Good) (hsent : CaseIsSent C) (hEach : StmtAtFixed A C Good L)
    (hLink : LinkFixed A C S Good L) (hLast : LastFixed A C S Good L) (hC : CaseCommits C)
    (hx : stmtOf c = some x)
    (hR : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x),
      FormAccepts C ((formFrame V A x θ F c σ C S dev Good L).out ω r) →
      Good ((formFrame V A x θ F c σ C S dev Good L).out ω r) →
      ReadAgrees A x (formReading A C S dev) ((formFrame V A x θ F c σ C S dev Good L).out ω r) r →
      (∀ i < (Claims.run FlockDeploy.sha512 (formFrame V A x θ F c σ C S dev Good L).P A.L.V x ω r).msgs.length,
        (formReading A C S dev).msg ((formFrame V A x θ F c σ C S dev Good L).out ω r) i =
          msgAt (Claims.run FlockDeploy.sha512 (formFrame V A x θ F c σ C S dev Good L).P A.L.V x ω r) i) →
      A.L.V.k x ≤
        (Claims.run FlockDeploy.sha512 (formFrame V A x θ F c σ C S dev Good L).P A.L.V x ω r).msgs.length) :
    MessagesFixedAgreed V A x θ F c σ (formFrame V A x θ F c σ C S dev Good L) (formReading A C S dev)
      (FormAccepts C) Good (formMiss V A x θ F c σ C S dev Good L q) := by
  have hcount : StmtsCount A C Good := stmtsCount_of Good hof hparts
  have hsz : SessionCoins.HeldSize := fun _ _ _ hh => held_size hh
  exact agreed_form V A x θ F c σ C S dev P' (fun _ => Flock.Tags.circuit.table) ℓ₁ ℓ₂ Good L hL q hne hSecond hDraw
    hsz hlog hθ
    (SessionCoins.logRising_of (SessionCoins.located_rising A C _ hsz) hlog (SessionCoins.interiorReads C hsz))
    (SessionCoins.recordsCover_of hsz hS hst hcount)
    (SessionCoins.msgsOfRecords_form hS hM hof hst hrounds hscope hcount hprint hfit hlive) hplay hsess hof hst
    hrounds hscope (stmtsFixed_of_each A C S Good L hS hsent hcount hEach) hLink hLast hC hx hR

end Verity.SampledTwo
