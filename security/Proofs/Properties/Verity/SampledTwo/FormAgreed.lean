module

public import Proofs.Properties.Verity.SampledTwo.FormRounds
public import Proofs.Properties.Verity.SampledTwo.LastSlot

/-!
# The messages' piece at the form's reading, where the coins agree

`reached_form` (`FormRounds.lean`) bounds the plays on which a message the form's check reads differs from the run's
at a round the run reaches, among the accepted plays a predicate holds of. The projection's assembly
(`projection_agreed`) reads the messages' piece as `MessagesFixedAgreed`: the same event at every round, among the
accepted plays whose coins as the check read them agree with the run's. This module gives it at the form's reading.

* **The plays read at `x`** (`readGood`): the plays `Good` holds of that the form's reading reads at the statement `x`
  with coins that agree with some run's (`Messages.ReadAt`). The messages' piece asks nothing of another play
  (`Messages.messagesFixedAgreed_of_readAt`).
* **The frame and the term** (`formFrame`, `formMiss`): `Messages.frameLog` and `Messages.missLog` at the form's
  reading, the digests `formDg` and the strings `formRaw`, on those plays.
* **The piece** (`agreed_form`): `reached_form` on those plays. There the sessions' statements set up and their
  messages before a session's last are of tag `0` by the run of record's accept (`Messages.sessionMsgsAsk_run`), the
  frame's statement is the one the check reads (`stmtAt`), and every other hypothesis of `reached_form` holds of
  fewer plays as it holds of more.

**Not established here**: the hypotheses of `reached_form` this module passes on (`SessionCoins.RecordsCover`,
`SessionCoins.MsgsOfRecords`, `StmtsFixed`, `LinkFixed`, `LastFixed` among them), and that a run on agreeing coins
whose messages are the check's at every round it reaches has all the verifier's messages (`hR`), which the accept on
the rounds a run reaches gives the assembly.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The plays `Good` holds of that the form's reading reads at `x`, with coins that agree with some run's**
(`Messages.ReadAt`). -/
abbrev readGood (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) :
    Verity.Exchange → Prop :=
  fun o => Good o ∧ Messages.ReadAt A x (formReading A C S dev) o

/-- **The frame of the messages' piece at the form's reading**: the frame at the locator `L`'s coin map whose prover
is the one of the digests `formDg` and the strings `formRaw` (`Messages.frameLog`), on the plays `readGood` holds
of. -/
noncomputable abbrev formFrame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) :
    Frame V A x θ F c σ :=
  Messages.frameLog V A x θ F c σ L (formReading A C S dev) (FormAccepts C) (readGood A x C S dev Good)
    (formDg A L θ) (formRaw A C S dev)

/-- **The messages' term at the form's reading**: `Messages.missLog` at `formFrame`'s digests and strings, with the
blobs' pair as the collision two plays may give, each finder at `q` SHA-512 calls. -/
noncomputable abbrev formMiss (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (q : ℕ) : ℝ≥0∞ :=
  Messages.missLog V A x θ F c σ L (formReading A C S dev) (FormAccepts C) (readGood A x C S dev Good)
    (formDg A L θ) (formRaw A C S dev) Messages.blobsAlt q

/-- **The messages fixed where the coins agree, at the form's reading**: at `formFrame`, among the plays the form's
check accepts that `Good` holds of and whose coins as the check read them agree with the run's, a message the check
reads differs from the run's at some round on plays at most `formMiss` likely. `reached_form` is taken on the plays
`readGood` holds of: there `Messages.SessionMsgsAsk` is `Messages.sessionMsgsAsk_run`, from the run of record's
accept, its sessions' messages and the held copies' (`PlayAccepts`, `SessionMsgs`, `HeldOfAt`, `RoundsHeldAt`,
`ScopeHeldAt`), and the frame's statement is the one the check reads (`stmtAt`, from `CaseCommits` and the record the
commitment carries). `hR` is `Messages.messagesFixedAgreed_of_readAt`'s: a run on agreeing coins whose messages are
the check's at every round it reaches has all the verifier's messages. -/
theorem agreed_form (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hL : L = SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (q : ℕ) (hne : ℓ₁ ≠ ℓ₂)
    (hSecond : SecondShown C Good)
    (hDraw : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hsz : SessionCoins.HeldSize) (hlog : SessionCoins.InteriorLogged C Good)
    (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ) (hRise : SessionCoins.LogRising A C Good tab)
    (hcov : SessionCoins.RecordsCover A C Good) (hmsgs : SessionCoins.MsgsOfRecords A C S Good tab)
    (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') (hof : HeldOfAt A C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hStmts : StmtsFixed A C S Good L)
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
  have hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) (readGood A x C S dev Good) L := by
    subst hL
    exact Messages.sessionMsgsAsk_run A x C S dev P' Good (CoinMap.stageLocate ℓ₁ ℓ₂) tab hplay hsess hof hst
      hrounds hscope
  exact Messages.messagesFixedAgreed_of_readAt
    (stmtAt V A x θ F c σ (formFrame V A x θ F c σ C S dev Good L) C S dev hC hx) hR
    (reached_form V A x θ F c σ C S dev tab ℓ₁ ℓ₂ (readGood A x C S dev Good) L hL q hne
      (fun o hA hG => hSecond o hA hG.1) (fun o hA => hDraw o ⟨hA.1, hA.2.1⟩) hsz
      (fun o hA hG => hlog o hA hG.1) hst (fun o hA hG => hθ o hA hG.1) (fun o hA hG => hRise o hA hG.1)
      (fun o hA hG => hcov o hA hG.1) (fun o hA hG => hmsgs o hA hG.1) hAsk
      (fun o o' hA hG hA' hG' => hStmts o o' hA hG.1 hA' hG'.1)
      (fun o o' t hA hG hA' hG' => hLink o o' t hA hG.1 hA' hG'.1)
      (fun o o' t hA hG hA' hG' => hLast o o' t hA hG.1 hA' hG'.1))

end Verity.SampledTwo
