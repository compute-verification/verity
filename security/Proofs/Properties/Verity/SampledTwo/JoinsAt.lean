module

public import Proofs.Properties.Verity.SampledTwo.Joined
public import Proofs.Properties.Verity.SampledTwo.GoodCase
public import Proofs.Properties.Verity.SampledTwo.CoinStages
public import Proofs.Properties.Verity.SampledTwo.CoinBytes
public import Proofs.Properties.Verity.SampledTwo.StageAlone
public import Proofs.Properties.Verity.SampledTwo.DeviceOrder
public import Proofs.Properties.Verity.SampledTwo.CutAccept
public import Proofs.Properties.Verity.SampledTwo.SessionDev
public import Proofs.Properties.Verity.SampledTwo.SessionCut
public import Proofs.Properties.Verity.SampledTwo.SessionTreeless
public import Proofs.Properties.Verity.SampledTwo.Reached
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.AcceptsTop
public import Proofs.Properties.Verity.SampledTwo.DrawRoots

/-!
# The two-stage leg's projection at the form's accept, from the coin map's law

`joined` (`Joined.lean`) is the projection's bound over any acceptance. This module is the one place the two-stage
form's check comes in: the acceptance is the form's (`FormAccepts C`), the check read a log of both sessions, the log
it read is the device's and every fresh coin carries its 512 bits (`JoinGood`), and the locator is the run's
(`joinLocate`: the stages' families' fresh coins on rounds `0` to `stage`, the sessions' openings on theirs).

**One premise at the form's accept.** `joinedOfLaw` takes `JoinsAt`, a structure whose fields are the pieces'
remaining premises, each in its owner's words, and the coin map's law (`CoinMap.CoinMapHolds`), which speaks of the
period's game and the coin map alone and reads no accept. `joinedForm` (`JoinedForm.lean`) is `joinedOfLaw` at
COINMAP's proof of the sampling design (`CoinMap.coinMapHolds`), so it takes `JoinsAt` and nothing else, and a witness of it at an
accepted play is one term of that structure. The sampling design is a hypothesis here only so that this module does not import
its proof (`CoinMapAgree.lean`).

**What `joinedOfLaw` discharges** from the fields:

* the cut at the deployed two-session history, where the device's ledger orders the two sessions
  (`Sessions.cutOrder_dev₂`), with the composite's wiring in the cut's words (`Wired`, from `wiring` and `caseIsSent`:
  `accepts_of_form`), and `FreshWide` joined to its event at no cost, since every play of the game has it
  (`CoinMap.freshWide_play`, `cutBound_and`);
* the stages' bytes within their releases (`firstFits_stage`, `secondFits_stage`, at the run's locator by
  `firstFits_congr`, `secondFits_congr`) and the first and the second stage's coins (`coinsAgree_first_of`,
  `coinsAgree_second_of`, with COINMAP's `devFirst_stage` and `devSecond_stage`);
* the sessions' releases' agreement and coins (`SessionCoins.sessionsAgree_accepted`, `coinsAgree_sessions`), with
  CUT's `held_size` for `HeldSize`;
* no coin at the statements' round, the accept on the rounds a run reaches (`reachedAccepts`, with ACCEPTS's
  `sourcesFit`, an accepted run read at most the coins it has, and `playAccepts_form`, its run of record a play
  SPOT[Flock] accepts) and the statement read (`stmtAt`).

**The stages' pieces are taken at the case kept on the good plays** (`goodCase C JoinGood`, `GoodCase.lean`).
STAGECOINS's `CutReads`, `DevFirst` and `DevSecond` speak of every play the form accepts, and none of them is known
there: the cut reads no log where a session's record carries a break, and a family's coins follow one root only on a
log the check read. At the kept case they speak of the accepted plays `JoinGood` holds of. There `CutReads` is
`JoinGood`'s first part, the fresh coins' width `DevSecond` needs is its last, and `StageDraws` is asked only of those
plays. There each stage's one root is CUT's (`oneRoot_draws`, `oneRoot_second`, at `otherRoots`), so `StageDraws`
  takes the families alone (`stageDraws_first`, `stageDraws_second`, `DrawRoots.lean`). What the pieces conclude comes
back to `C` by `coinsAgree_goodCase`, `firstFits_goodCase` and `secondFits_goodCase`. So `joinedOfLaw` takes no
`CutReads` and no coins' premise for either stage.

**What the stages read is the form's accept's**, with `caseIsSent`, and no field. The check's first session is the
cut's (STAGECOINS's `sessionMatches_of`, from the form's refusal of an interior record whose read-back digest is not
the interior commitment's, `INTERIOR_DIGEST`), and the first stage read the cut's sample (`firstDrawn_of`, from its
refusal of a first-stage record whose read-back digest is not the commitment's), both taken at the kept case
(`sessionMatches_goodCase`, `caseIsSent_goodCase`). The first stage's coins are read under the parameters' key and an
accepted play on two windows shows its sessions' order (`firstReads`, `orderShown`, CUT's `CutAccept.lean`).

**Not established**, each a field of `JoinsAt`:

* **The device's ledger** (`ledgerInOrder`, HISTORY's `Sessions.LedgerInOrder` at the deployment's two setups): each
  session opens its own run once on one window, no line opens another run, and the second session's `end` is not
  before the first session's `reserve`. Taken as a hypothesis; a lemma of the modelled device after soundness's device
  pull request (PRF-490). The model's device takes its ledger lines as given, so nothing here proves it, and it is a
  field by that name, not a part of `JoinGood`.
* **The two families distinct** (`distinctFamilies`), until the form refuses equal labels (TWOLEG).
* **Each stage's sample names its family** (`family₁`, `family₂`): the boundary registration's challenger's family is
  `ℓ₁` and the interior audit's is `ℓ₂`, in the parameters of each accepted play.
* **On two windows each session shows the other stage's family after that stage's digest alone** (`otherRoots`,
  CUT's `OtherRoots`): the form's accept does not give it.
* **The messages** (`msgOpens`, `digestBefore`, MESSAGES): no check gives either; their docstrings say what each
  holds of. That on two windows the first session samples no challenge tree is a theorem of the form's accept
  (`SessionCoins.firstTreeless_of_check`, from the form's refusal of a tree's line in the first session's log and
  `caseIsSent`), no longer a field; it gives the interior's log on the play's (`SessionCoins.interiorLogged_read`).
* **The run of record's accept** (`playAccepts_form`'s premises): `heldOfAt` (HELDOF); `portsDistinctAt`;
  `stmtsRead`, `roundsHeldAt`, `scopeHeldAt`, `pinHeldAt`, `sessionsRun` (MESSAGES and the sessions' workers); and
  `sessionMsgs`. That an accepted play's records name their registrations' names and read back as the records the run
  held, that the held copies' design evaluates, that each accepted play's held first stage samples one fixed number
  of replay blocks, and that at each such number the held copies' parts are the interior program's strata's, are
  theorems (ACCEPTS's `recordsHeldAt_of`, `recordsRead`, `twoHolds_at`, `firstK_of_formAccepts`,
  `partsSizedAt_of_formAccepts`), no longer fields. The last is the run's refusal of an interior whose strata are not
  the parts its replay block's cells give (`interior-parts`), read at `heldOfAt`'s copies; with `heldOfAt` it gives the
  parts at each play's replay sample (`partsHeldAt_of_accepts`).
* **The instance's** (`wiring`, `caseIsSent`, `caseCommits`, `stmtX`, `signer`).

Every field is over the form's accept, so the theorem says nothing about a run of record until the form accepts a
play. `CoinStages.lean` is COINMAP's, read here at the head the join holds. At the deployed
history the cut meets the open hole `ProjectionOf.lean` names: a genuine challenger signature from an earlier period
makes the forger's term `⊤`. The join neither closes nor widens it.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The join's `Good`**: the check read a log of both sessions and the log it read is the device's
(`ReadIsDevice`), and every fresh coin the play's log holds carries its 512 bits (`CoinMap.FreshWide`). -/
def JoinGood (o : Verity.Exchange) : Prop := ReadIsDevice o ∧ CoinMap.FreshWide o.log

/-- **The run's locator at the stages' families** `ℓ₁` and `ℓ₂`: the stages' locator on rounds `0` to `stage`, the
sessions' on theirs. -/
noncomputable abbrev joinLocate (A : Live₂) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) : CoinMap.Locate :=
  SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab

/-- **The sessions' reading the join reads**: the messages' reading `Ms` with the sessions' coins and agreement at
the coin map's release of the run's locator. -/
noncomputable abbrev joinSessions (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (ℓ₁ ℓ₂ : String) : SessionReading :=
  SessionCoins.withCoins A C tab (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) Ms

/-! ## The pieces' premises at the kept case -/

/-- **The form accepts what the prover sent**, where the play's case is the one sent. -/
theorem accepts_of_form {C : CaseOf} (hC : CaseIsSent C) {o : Verity.Exchange} (ha : FormAccepts C o) :
    Accepts o.sent := by
  obtain ⟨k, hk, p, cm, e, pub, prof, hd, hacc⟩ := ha
  obtain ⟨hcase, hblobs⟩ := hC o k p cm e hk hd
  refine ⟨p, cm, e, pub, prof, hcase, ?_⟩
  rw [← hblobs]
  exact hacc

theorem caseIsSent_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} (h : CaseIsSent C) :
    CaseIsSent (goodCase C G) :=
  fun o k p cm e hk hd => h o k p cm e (goodCase_some hk).2 hd

/-- **The cut reads a log of every play the form accepts the kept case on**, where `G` says it read one. -/
theorem cutReads_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} (hG : ∀ o, G o → LogRead o) :
    CutReads (goodCase C G) :=
  fun o ha => hG o (formAccepts_goodCase.1 ha).2

theorem sessionMatches_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} (h : SessionMatches C) :
    SessionMatches (goodCase C G) :=
  fun o ha => h o (formAccepts_goodCase.1 ha).1

/-- **A stage's sample at the kept case**: `StageDraws` asked of the accepted plays `G` holds of is `StageDraws` at the
kept case's acceptance and sample. -/
theorem stageDraws_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} {f : Case → Option (String × ByteArray)}
    {ℓ : String} (h : CoinMap.StageDraws (fun o => FormAccepts C o ∧ G o) (fun o => (C o).bind f) ℓ) :
    CoinMap.StageDraws (FormAccepts (goodCase C G)) (fun o => (goodCase C G o).bind f) ℓ := by
  intro o ha label after hd
  obtain ⟨ha', hg⟩ := formAccepts_goodCase.1 ha
  exact h o ⟨ha', hg⟩ label after
    ((congrArg (fun z : Option Case => z.bind f) (goodCase_of hg)).symm.trans hd)

/-! ## The stages' bytes within their releases -/

/-- **The run of record's input and interior on an accepted play.** -/
theorem input_interior_of {C : CaseOf} {o : Verity.Exchange} (hA : FormAccepts C o) :
    ∃ inp it, inputOn C o = some inp ∧ inp.interior = some it := by
  obtain ⟨k, hk, hacc⟩ := hA
  obtain ⟨_, inp, -, hin, -, it, _, hit, -⟩ := input_of_accepts hacc
  refine ⟨inp, it, ?_, hit⟩
  show (C o).bind inputOfCase = some inp
  rw [hk]
  exact hin

/-- **The first stage's bytes fit in round 0's releases at the stages' locator.** -/
theorem firstFits_stage (A : Live₂) (C : CaseOf) (ℓ₁ ℓ₂ : String) (G Good : Verity.Exchange → Prop)
    (hG : ∀ o, Good o → G o ∧ CoinMap.FreshWide o.log) (hShow : FirstShown C G)
    (hDraw : CoinMap.StageDraws (FormAccepts C) (fun o => (C o).bind firstDraw) ℓ₁) :
    FirstFits C (CoinMap.stageLocate ℓ₁ ℓ₂) (FormAccepts C) Good := by
  intro o hA hGo
  obtain ⟨hg, hw⟩ := hG o hGo
  obtain ⟨inp, it, hin, hit⟩ := input_interior_of hA
  refine ⟨inp, hin, ?_⟩
  have hL := (sourcesFit A C o inp it hA hin hit).1
  rcases Nat.eq_zero_or_pos inp.source.size with h0 | hpos
  · rw [h0]
    exact Nat.zero_le _
  obtain ⟨read, hread⟩ := firstRead_of_accepts hA hin
  have hex : inp.source.extract 0 read = inp.source :=
    extract_of_size_le (sourceRead_holds C o hA inp read hin hread)
  obtain ⟨label, after, hd, hb⟩ := hShow o hA hg inp read hin hread (by rw [hex]; exact hpos)
  obtain ⟨hl, hone⟩ := hDraw o hA label after hd
  rw [hex, hl] at hb
  exact CoinMap.size_le_fit_first ℓ₁ ℓ₂ hone hw hb hL

/-- **The second stage's bytes fit in rounds `1` to `stage`'s releases at the stages' locator.** -/
theorem secondFits_stage (A : Live₂) (C : CaseOf) (ℓ₁ ℓ₂ : String) (hne : ℓ₁ ≠ ℓ₂) (G Good : Verity.Exchange → Prop)
    (hG : ∀ o, Good o → G o ∧ CoinMap.FreshWide o.log) (hShow : SecondShown C G)
    (hDraw : CoinMap.StageDraws (FormAccepts C) (fun o => (C o).bind secondDraw) ℓ₂) :
    SecondFits A C (CoinMap.stageLocate ℓ₁ ℓ₂) (FormAccepts C) Good := by
  intro o hA hGo
  obtain ⟨hg, hw⟩ := hG o hGo
  obtain ⟨inp, it, hin, hit⟩ := input_interior_of hA
  refine ⟨inp, it, hin, hit, fun j _ => ?_⟩
  obtain ⟨label, after, hd, hb⟩ := hShow o hA hg inp it hin hit
  obtain ⟨hl, hone⟩ := hDraw o hA label after hd
  rw [hl] at hb
  exact CoinMap.stageWidth_le_fit_second ℓ₁ ℓ₂ hne hone hw hb j

/-- **`FirstFits` at a locator that puts round 0's releases where another does.** -/
theorem firstFits_congr {C : CaseOf} {L L' : CoinMap.Locate} {Acc Good : Verity.Exchange → Prop}
    (hL : ∀ pre fr b, CoinMap.locAt L pre fr b = some 0 ↔ CoinMap.locAt L' pre fr b = some 0)
    (h : FirstFits C L' Acc Good) : FirstFits C L Acc Good := by
  intro o hA hG
  obtain ⟨inp, hin, hsz⟩ := h o hA hG
  exact ⟨inp, hin, by rw [SessionCoins.fitOf_congr hL [] o.log]; exact hsz⟩

/-- **`SecondFits` at a locator that puts rounds `1` to `stage`'s releases where another does.** -/
theorem secondFits_congr {A : Live₂} {C : CaseOf} {L L' : CoinMap.Locate} {Acc Good : Verity.Exchange → Prop}
    (hL : ∀ n, 0 < n → n ≤ A.h.stage →
      ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n)
    (h : SecondFits A C L' Acc Good) : SecondFits A C L Acc Good := by
  intro o hA hG
  obtain ⟨inp, it, hin, hit, hsz⟩ := h o hA hG
  refine ⟨inp, it, hin, hit, fun j hj => ?_⟩
  rw [SessionCoins.fitOf_congr (hL (j + 1) (Nat.succ_pos j) hj) [] o.log]
  exact hsz j hj

/-! ## The join's one premise -/

/-- **What the join takes at the form's accept**: the pieces' remaining premises, each in its owner's words, at the
parameters `θ`, the files `F`, the commitment `c` and the strategy `σ`, the instance's case `C`, the messages' reading
`Ms`, the stages' families `ℓ₁` and `ℓ₂`, the digests `Dg`, the deployment `P` with its layouts and cut, and the
sessions' provers `P'`. `joinedForm` (`JoinedForm.lean`) takes this and nothing else; the module docstring says what
each field waits on. -/
structure JoinsAt (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg)
    (P : Sessions.Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) : Prop where
  /-- On two windows, the two setups' ledger lines are two sessions' in the device's order (`Sessions.LedgerInOrder`).
  Taken as a hypothesis; a lemma of the modelled device after soundness's device pull request (PRF-490). -/
  ledgerInOrder : ∀ o : Verity.Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
    Sessions.LedgerInOrder (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o)
  /-- The two stages' families differ. -/
  distinctFamilies : ℓ₁ ≠ ℓ₂
  /-- The first stage's sample names `ℓ₁`, on the accepted plays `JoinGood` holds of: the family of the boundary
  registration's challenger, in the play's parameters (`firstDraw`). -/
  family₁ : ∀ o, FormAccepts C o ∧ JoinGood o → ∀ label after, (C o).bind firstDraw = some (label, after) → label = ℓ₁
  /-- The second stage's sample names `ℓ₂`, on the accepted plays `JoinGood` holds of: the family of the interior
  audit's challenger, in the play's parameters (`secondDraw`). -/
  family₂ : ∀ o, FormAccepts C o ∧ JoinGood o → ∀ label after, (C o).bind secondDraw = some (label, after) → label = ℓ₂
  /-- On two windows each session shows the other stage's family after that stage's digest alone (CUT's
  `OtherRoots`), on the accepted plays `JoinGood` holds of. -/
  otherRoots : ∀ o, FormAccepts C o ∧ JoinGood o → OtherRoots o.sent
  /-- The message the check reads at each round opens the digest the play records. Holds of a strategy whose message
  at each round is fixed before the round's coin; no check gives it: at the digests the device signs (the link's
  `after`, a rep round's SHA-512 of its retained bytes, the closing's SHA-512 of the session's evidence; the
  statements' round has no release) the message read is not the string hashed. -/
  msgOpens : Messages.MsgOpens A x
    (joinReading A C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂)) (FormAccepts C) JoinGood Dg
  /-- Each round's digest is fixed by the coins the history before it shows. Holds of a strategy whose message at
  each round is fixed before the round's coin; no check gives it: at `Dg` the SHA-512 of the message read, it is that
  fact taken as a premise. -/
  digestBefore : ∀ i < A.L.V.k x, Messages.DigestBefore
    (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
    (joinReading A C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂)) (FormAccepts C) JoinGood Dg i
  /-- Each accepted play's held copies are the ones `heldOf` builds from its input and interior. -/
  heldOfAt : HeldOfAt A C
  /-- Each accepted play's registrations list each port once. -/
  portsDistinctAt : PortsDistinctAt C
  /-- The reading's session statements are the ones the run's second stage checked. -/
  stmtsRead : StmtsRead A C (joinSessions A C Ms tab ℓ₁ ℓ₂)
  /-- Each session statement has its template's stage count. -/
  roundsHeldAt : RoundsHeldAt A C (joinSessions A C Ms tab ℓ₁ ℓ₂)
  /-- Each session statement is in scope. -/
  scopeHeldAt : ScopeHeldAt A C (joinSessions A C Ms tab ℓ₁ ℓ₂)
  /-- Each session statement is its template's reference statement but for the run's fields. -/
  pinHeldAt : PinHeldAt A C (joinSessions A C Ms tab ℓ₁ ℓ₂)
  /-- Flock accepts each session, on the provers `P'`. The sessions' layout (`SessionCoins.slotStream`, `roundOf`)
  places a session's rounds rep stream by rep stream, the order the deployed session's `custody` reads its recorded
  rounds in. That this field, `sessionMsgs` and the coins `SeqCoinsRead` fixes hold together at an honest play is not
  proved here: it needs each rep to have `SessionCoins.repsOf` rounds, one table a session, and `tab t` to be that
  table's name, and no field states those. -/
  sessionsRun : SessionsRun A C (joinSessions A C Ms tab ℓ₁ ℓ₂) P'
  /-- The sessions' messages are the ones the check read, in the layout's order (`sessionsRun`'s docstring). -/
  sessionMsgs : SessionMsgs A C (joinSessions A C Ms tab ℓ₁ ℓ₂) P'
  /-- The sampled verdict accepts, after `θ`, only where the form accepts the instance's case. -/
  wiring : ∀ o, o.params = θ → V.Verdict k .sampled o → FormAccepts C o
  /-- The instance's case is the one the prover sent. -/
  caseIsSent : CaseIsSent C
  /-- The instance's case reads the play's commitment. -/
  caseCommits : CaseCommits C
  /-- The statement `x` is the record the commitment `c` carries. -/
  stmtX : stmtOf c = some x
  /-- The vocabulary's signer is the deployment's. -/
  signer : V.signer θ = P.signer fs₁ fs₂ cut θ

/-- **The projection's bound at the form's accept, from the coin map's law**: `joined` at the form's acceptance,
`JoinGood` and the run's locator, at the cut's terms at the deployed two-session history, from the coin map's law
(`hHolds`) and the join's one premise (`JoinsAt`). The stages' pieces are taken at the case kept on the plays
`JoinGood` holds of. -/
theorem joinedOfLaw (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (P : Sessions.Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hHolds : CoinMap.CoinMapHolds (d := d) (α := α))
    (h : JoinsAt V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg P fs₁ fs₂ cut P') :
    ProjectionBound V k A x θ F c σ
      (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
      (CoinMap.gapOf V θ F c σ (joinLocate A tab ℓ₁ ℓ₂) (A.L.V.k x) +
        joinMiss V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg q +
        terms V (Sessions.devOf P fs₁ fs₂ cut) θ F c σ) := by
  have hG : ∀ o, JoinGood o → LogIsDevice o ∧ CoinMap.FreshWide o.log := fun o hg => ⟨hg.1.2, hg.2⟩
  have hLs : ∀ pre i s c' b, CoinMap.stageLocate ℓ₁ ℓ₂ pre (.opening i s c') b = none :=
    fun _ _ _ _ _ => rfl
  have hstage : ∀ n, n ≤ A.h.stage → ∀ pre fr b,
      CoinMap.locAt (joinLocate A tab ℓ₁ ℓ₂) pre fr b = some n ↔
        CoinMap.locAt (CoinMap.stageLocate ℓ₁ ℓ₂) pre fr b = some n :=
    fun n hn => SessionCoins.locAt_run_stage A (CoinMap.stageLocate ℓ₁ ℓ₂) tab hLs hn
  -- The stages' pieces, at the case kept on the plays `JoinGood` holds of.
  have hC' : CaseIsSent (goodCase C JoinGood) := caseIsSent_goodCase h.caseIsSent
  have hR' : CutReads (goodCase C JoinGood) := cutReads_goodCase fun o (hg : JoinGood o) => hg.1.1
  have hD' : FirstDrawn (goodCase C JoinGood) := firstDrawn_of _ hC'
  have hM' : SessionMatches (goodCase C JoinGood) := sessionMatches_goodCase (sessionMatches_of C h.caseIsSent)
  have hcase : ∀ o k p c e, C o = some k → Decoded k p c e → caseOf o.sent = .ok (p, c, e) :=
    fun o k p c e hk hd => (h.caseIsSent o k p c e hk hd).1
  have hA : ∀ o, FormAccepts C o ∧ JoinGood o → FormAccepts C o ∧ ReadIsDevice o ∧ OtherRoots o.sent :=
    fun o ho => ⟨ho.1, ho.2.1, h.otherRoots o ho⟩
  have hDraw₁ : CoinMap.StageDraws (FormAccepts (goodCase C JoinGood))
      (fun o => (goodCase C JoinGood o).bind firstDraw) ℓ₁ :=
    stageDraws_goodCase (stageDraws_first hcase hA h.family₁)
  have hDraw₂ : CoinMap.StageDraws (FormAccepts (goodCase C JoinGood))
      (fun o => (goodCase C JoinGood o).bind secondDraw) ℓ₂ :=
    stageDraws_goodCase (stageDraws_second hcase hA h.family₂)
  have hShow₁ : FirstShown (goodCase C JoinGood) LogIsDevice := firstShown_of _ hC' hR' hD'
  have hShow₂ : SecondShown (goodCase C JoinGood) LogIsDevice :=
    secondShown_of _ hC' hR' (secondAlone_of_matches _ hC' hM')
  have hDev₁ : DevFirst (goodCase C JoinGood) (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) := by
    intro o hA label after b hd hb h0 hL
    obtain ⟨e, he, hsrc⟩ := CoinMap.devFirst_stage ℓ₁ ℓ₂ hDraw₁ o hA label after b hd hb h0 hL
    exact ⟨e, (SessionCoins.dev_run_stage A (CoinMap.stageLocate ℓ₁ ℓ₂) tab hLs (Nat.zero_le _) o.log).trans he,
      hsrc⟩
  have hDev₂ : DevSecond A (goodCase C JoinGood) (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) := by
    intro o hA label after b hd hb hsz j hj hpos
    obtain ⟨e, he, hsrc⟩ := CoinMap.devSecond_stage ℓ₁ ℓ₂ h.distinctFamilies hDraw₂ A.h.stage o hA
      (formAccepts_goodCase.1 hA).2.2 label after b hd hb hsz j hj hpos
    exact ⟨e, (SessionCoins.dev_run_stage A (CoinMap.stageLocate ℓ₁ ℓ₂) tab hLs (n := j + 1) hj o.log).trans he,
      hsrc⟩
  have hFit₁ : FirstFits C (joinLocate A tab ℓ₁ ℓ₂) (FormAccepts C) JoinGood :=
    firstFits_goodCase (firstFits_congr (hstage 0 (Nat.zero_le _))
      (firstFits_stage A (goodCase C JoinGood) ℓ₁ ℓ₂ LogIsDevice JoinGood hG hShow₁ hDraw₁))
  have hFit₂ : SecondFits A C (joinLocate A tab ℓ₁ ℓ₂) (FormAccepts C) JoinGood :=
    secondFits_goodCase (secondFits_congr (fun n _ hn => hstage n hn)
      (secondFits_stage A (goodCase C JoinGood) ℓ₁ ℓ₂ h.distinctFamilies LogIsDevice JoinGood hG hShow₂ hDraw₂))
  have hFirst : CoinsAgree A (joinReading A C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂))
      (FormAccepts C) JoinGood .first :=
    coinsAgree_goodCase (coinsAgree_mono (fun o (hg : JoinGood o) => hg.1.2)
      (coinsAgree_first_of A (goodCase C JoinGood) (joinSessions A C Ms tab ℓ₁ ℓ₂)
        (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) LogIsDevice hShow₁ hDev₁))
  have hSecond : CoinsAgree A (joinReading A C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂))
      (FormAccepts C) JoinGood .second :=
    coinsAgree_goodCase (coinsAgree_mono (fun o (hg : JoinGood o) => hg.1.2)
      (coinsAgree_second_of A (goodCase C JoinGood) (joinSessions A C Ms tab ℓ₁ ℓ₂)
        (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) LogIsDevice hShow₂ hDev₂))
  -- The sessions' pieces, at the case itself.
  have hsz : SessionCoins.HeldSize := fun _ _ _ hh => held_size hh
  have hlog : SessionCoins.InteriorLogged C JoinGood := fun o hA hg =>
    SessionCoins.interiorLogged_read C h.caseIsSent
      (SessionCoins.firstTreeless_of_check C h.caseIsSent (fun _ _ _ _ _ _ _ _ => rfl)) o hA hg.1
  have hLoc := SessionCoins.locates_run A (CoinMap.stageLocate ℓ₁ ℓ₂) tab
  have hSess := SessionCoins.sessionsAgree_accepted A x C Ms tab (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) JoinGood _
    hsz hlog
    (SessionCoins.devElems_of A x tab (joinLocate A tab ℓ₁ ℓ₂) (readsAt A.h) (FormAccepts C) JoinGood hLoc
      (fun _ hr => hr) (SessionCoins.sessionsFit_of A C tab JoinGood hsz hlog))
  have hSessions := SessionCoins.coinsAgree_sessions A C Ms tab (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) JoinGood
    (SessionCoins.devLocal_of A tab _ hLoc) hlog
  -- The cut, at the deployed two-session history, with `FreshWide` joined to its event.
  have hw : Wired V k θ := fun o hθ hv => accepts_of_form h.caseIsSent (h.wiring o hθ hv)
  have hCut : CutBound V k θ F c σ (fun o => ReadIsDevice o ∧ CoinMap.FreshWide o.log)
      (terms V (Sessions.devOf P fs₁ fs₂ cut) θ F c σ) :=
    cutBound_and (fun o hr => CoinMap.freshWide_play V θ F c σ hr)
      (Sessions.cutOrder_dev₂ V k P fs₁ fs₂ cut θ F c σ hw (firstReads θ) (orderShown θ) h.ledgerInOrder h.signer)
  have hParts : PartsHeldAt A C := partsHeldAt_of_accepts h.heldOfAt (partsSizedAt_of_formAccepts h.heldOfAt)
  exact joined V k A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg q (FormAccepts C)
    JoinGood (terms V (Sessions.devOf P fs₁ fs₂ cut) θ F c σ) h.wiring
    (hHolds V A x θ F c σ _ _ (readsAt A.h))
    hFit₁ hFit₂ hSess hFirst hSecond hSessions
    (SessionCoins.stmtReadsNone A C _ (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)))
    h.msgOpens h.digestBefore hCut
    (reachedAccepts A x C _ (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) P' (sourcesFit A C)
      (playAccepts_form A C _ P' hParts h.heldOfAt h.portsDistinctAt h.stmtsRead h.roundsHeldAt
        h.scopeHeldAt h.pinHeldAt h.sessionsRun) h.sessionMsgs)
    (stmtAt V A x θ F c σ _ C _ (CoinMap.dev (joinLocate A tab ℓ₁ ℓ₂)) h.caseCommits h.stmtX)

end Verity.SampledTwo
