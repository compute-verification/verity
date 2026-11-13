module

public import Proofs.Properties.Verity.SampledTwo.Joined
public import Proofs.Properties.Verity.SampledTwo.CoinStages
public import Proofs.Properties.Verity.SampledTwo.CoinBytes
public import Proofs.Properties.Verity.SampledTwo.StageAlone
public import Proofs.Properties.Verity.SampledTwo.CutRead
public import Proofs.Properties.Verity.SampledTwo.SessionDev
public import Proofs.Properties.Verity.SampledTwo.Reached

/-!
# The two-stage leg's projection at the form's accept

`joined` (`Joined.lean`) is the projection's bound over any acceptance. This module is the one place the two-stage
form's check comes in: the acceptance is the form's (`FormAccepts C`), the log the check read is the device's and every
fresh coin carries its 512 bits (`JoinGood`), and the locator is the run's (`SessionCoins.runLocate` over the stages'
`CoinMap.stageLocate`: the stages' families' fresh coins on rounds `0` to `stage`, the sessions' openings on theirs).

* **The stages' bytes within their releases** (`firstFits_stage`, `secondFits_stage`): the bits the stages read start
  the coins their draws' families released (STAGECOINS's `FirstShown`, `SecondShown`), each draw is the locator's
  family after one root (`StageDraws`), every fresh coin carries 512 bits (`FreshWide`), and the first stage read at
  most one coin's bytes (`SourcesFit`); COINMAP's `size_le_fit_first` and `stageWidth_le_fit_second` then bound them
  by what fits. At the run's locator by `fitOf_congr` (`firstFits_congr`, `secondFits_congr`), since it is the stages'
  locator at the stages' rounds (`SessionCoins.locAt_run_stage`).
* **`FreshWide` costs the cut nothing**: every play of the game has it (`CoinMap.freshWide_play`), so it joins the
  cut's `ReadIsDevice` at the cut's `terms` (`cutBound_and`).
* **`joinedForm`**: `joined` at the form, with the coin map's law (`CoinMap.coinMapHolds`), the releases' agreement,
  the first stage's and the sessions' coins, no coin at the statements' round, the accept on the rounds a run reaches
  (`reachedAccepts`) and the statement read (`stmtAt`) discharged.

`joinedForm` still takes, each a named premise of a piece:

* the wiring (`hAcc`) and the cut (`hCut`, at `ReadIsDevice`: CUT's, whose own premise `PrintsInOrder` is HISTORY's
  to prove at the model's device);
* the case's premises the stages' pieces read (`CaseIsSent`, `CutReads`, `FirstDrawn`, `SessionMatches`), the two
  families distinct, `StageDraws` at both draws, and `SourcesFit`;
* the second stage's coins (`hSecond`) at `JoinGood`: STAGECOINS's `DevSecond` holds of the coin map at the stages'
  locator only where the log's fresh coins carry 512 bits (`CoinMap.devSecond_stage`), and it does not read `Good`;
* the sessions' (`HeldSize`, `InteriorLogged`), the messages' (`MsgOpens`, `DigestBefore`), the accept's
  (`PlayAccepts`, `SessionMsgs`) and the statement's (`CaseCommits`, `stmtOf c = some x`).

Every hypothesis is over the form's accept, so the theorem says nothing about a run of record until the form accepts a
play; its witness is `joinedForm` at such a play. `CoinMapAgree.lean` and `CoinStages.lean` are COINMAP's, read here at
the head the join holds. Where the cut is instantiated at the deployed composite it meets the open hole of the device's
history that `ProjectionOf.lean` names: a genuine challenger signature from an earlier period makes the forger's term
`⊤`. The join neither closes nor widens it.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The join's `Good`**: the log the check read is the device's (`ReadIsDevice`), and every fresh coin the play's log
holds carries its 512 bits (`CoinMap.FreshWide`). -/
def JoinGood (o : Verity.Exchange) : Prop := ReadIsDevice o ∧ CoinMap.FreshWide o.log

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
    (hDraw : CoinMap.StageDraws (FormAccepts C) (fun o => (C o).bind firstDraw) ℓ₁) (hfit : SourcesFit A C) :
    FirstFits C (CoinMap.stageLocate ℓ₁ ℓ₂) (FormAccepts C) Good := by
  intro o hA hGo
  obtain ⟨hg, hw⟩ := hG o hGo
  obtain ⟨inp, it, hin, hit⟩ := input_interior_of hA
  refine ⟨inp, hin, ?_⟩
  have hL := (hfit o inp it hA hin hit).1
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
  exact ⟨inp, hin, by rw [fitOf_congr hL [] o.log]; exact hsz⟩

/-- **`SecondFits` at a locator that puts rounds `1` to `stage`'s releases where another does.** -/
theorem secondFits_congr {A : Live₂} {C : CaseOf} {L L' : CoinMap.Locate} {Acc Good : Verity.Exchange → Prop}
    (hL : ∀ n, 0 < n → n ≤ A.h.stage →
      ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n)
    (h : SecondFits A C L' Acc Good) : SecondFits A C L Acc Good := by
  intro o hA hG
  obtain ⟨inp, it, hin, hit, hsz⟩ := h o hA hG
  refine ⟨inp, it, hin, hit, fun j hj => ?_⟩
  rw [fitOf_congr (hL (j + 1) (Nat.succ_pos j) hj) [] o.log]
  exact hsz j hj

/-- **The projection's bound at the form's accept**: `joined` at the form's acceptance, `JoinGood` and the run's
locator, with the coin map's law, the releases' agreement, the first stage's and the sessions' coins, no coin at the
statements' round, the accept on the rounds a run reaches and the statement read discharged, under the pieces'
remaining premises. -/
theorem joinedForm (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (terms : ℝ≥0∞) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hAcc : ∀ o, o.params = θ → V.Verdict k .sampled o → FormAccepts C o)
    (hCut : CutBound V k θ F c σ ReadIsDevice terms)
    (hC : CaseIsSent C) (hR : CutReads C) (hD : FirstDrawn C) (hM : SessionMatches C) (hne : ℓ₁ ≠ ℓ₂)
    (hDraw₁ : CoinMap.StageDraws (FormAccepts C) (fun o => (C o).bind firstDraw) ℓ₁)
    (hDraw₂ : CoinMap.StageDraws (FormAccepts C) (fun o => (C o).bind secondDraw) ℓ₂)
    (hfit : SourcesFit A C)
    (hSecond : CoinsAgree A
      (joinReading A C (SessionCoins.withCoins A C tab
        (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
        (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab))
      (FormAccepts C) JoinGood .second)
    (hsz : SessionCoins.HeldSize) (hlog : SessionCoins.InteriorLogged C LogIsDevice)
    (hOpens : Messages.MsgOpens A x
      (joinReading A C (SessionCoins.withCoins A C tab
        (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
        (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab))
      (FormAccepts C) JoinGood Dg)
    (hDigest : ∀ i < A.L.V.k x, Messages.DigestBefore
      (joinFrame V A x θ F c σ C (SessionCoins.withCoins A C tab
        (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
        (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) Dg)
      (joinReading A C (SessionCoins.withCoins A C tab
        (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
        (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab))
      (FormAccepts C) JoinGood Dg i)
    (hplay : PlayAccepts A C (SessionCoins.withCoins A C tab
      (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms) P')
    (hsess : SessionMsgs A C (SessionCoins.withCoins A C tab
      (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms) P')
    (hCC : CaseCommits C) (hx : stmtOf c = some x) :
    ProjectionBound V k A x θ F c σ
      (joinFrame V A x θ F c σ C (SessionCoins.withCoins A C tab
        (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
        (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) Dg)
      (CoinMap.gapOf V θ F c σ (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (A.L.V.k x) +
        joinMiss V A x θ F c σ C (SessionCoins.withCoins A C tab
          (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
          (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) Dg q + terms) := by
  have hG : ∀ o, JoinGood o → LogIsDevice o ∧ CoinMap.FreshWide o.log := fun o h => ⟨h.1.2, h.2⟩
  have hlog' : SessionCoins.InteriorLogged C JoinGood := fun o hA hg => hlog o hA hg.1.2
  have hLs : ∀ pre i s c b, CoinMap.stageLocate ℓ₁ ℓ₂ pre (.opening i s c) b = none :=
    fun _ _ _ _ _ => rfl
  have hstage : ∀ n, n ≤ A.h.stage → ∀ pre fr b,
      CoinMap.locAt (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) pre fr b = some n ↔
        CoinMap.locAt (CoinMap.stageLocate ℓ₁ ℓ₂) pre fr b = some n :=
    fun n hn => SessionCoins.locAt_run_stage A (CoinMap.stageLocate ℓ₁ ℓ₂) tab hLs hn
  have hFit₁ : FirstFits C (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (FormAccepts C) JoinGood :=
    firstFits_congr (hstage 0 (Nat.zero_le _))
      (firstFits_stage A C ℓ₁ ℓ₂ LogIsDevice JoinGood hG (firstShown_of C hC hR hD) hDraw₁ hfit)
  have hFit₂ : SecondFits A C (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (FormAccepts C)
      JoinGood :=
    secondFits_congr (fun n _ hn => hstage n hn)
      (secondFits_stage A C ℓ₁ ℓ₂ hne LogIsDevice JoinGood hG
        (secondShown_of C hC hR (secondAlone_of_matches C hC hM)) hDraw₂)
  have hDev₁ : DevFirst C (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) := by
    intro o hA label after b hd hb h0 hL
    rw [SessionCoins.dev_run_stage A (CoinMap.stageLocate ℓ₁ ℓ₂) tab hLs (Nat.zero_le _)]
    exact CoinMap.devFirst_stage ℓ₁ ℓ₂ hDraw₁ o hA label after b hd hb h0 hL
  have hFirst := coinsAgree_mono (fun o (h : JoinGood o) => h.1.2)
    (coinsAgree_first_of A C (SessionCoins.withCoins A C tab
      (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
      (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) LogIsDevice
      (firstShown_of C hC hR hD) hDev₁)
  have hLoc := SessionCoins.locates_run A (CoinMap.stageLocate ℓ₁ ℓ₂) tab
  have hSess := SessionCoins.sessionsAgree_accepted A x C Ms tab
    (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) JoinGood _ hsz hlog'
    (SessionCoins.devElems_of A x tab (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (readsAt A.h)
      (FormAccepts C) JoinGood hLoc (fun _ h => h) (SessionCoins.sessionsFit_of A C tab JoinGood hsz hlog'))
  have hSessions := SessionCoins.coinsAgree_sessions A C Ms tab
    (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) JoinGood
    (SessionCoins.devLocal_of A tab _ hLoc) hlog'
  have hCut' : CutBound V k θ F c σ (fun o => ReadIsDevice o ∧ CoinMap.FreshWide o.log) terms :=
    cutBound_and (fun o hr => CoinMap.freshWide_play V θ F c σ hr) hCut
  exact joined V k A x θ F c σ C
    (SessionCoins.withCoins A C tab (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) Ms)
    (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) Dg q (FormAccepts C) JoinGood terms hAcc
    ((CoinMap.coinMapHolds (d := d) (α := α)) V A x θ F c σ _ _ (readsAt A.h))
    hFit₁ hFit₂ hSess hFirst hSecond hSessions
    (SessionCoins.stmtReadsNone A C _ (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)))
    hOpens hDigest hCut'
    (reachedAccepts A x C _ (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) P' hfit
      hplay hsess)
    (stmtAt V A x θ F c σ _ C _ (CoinMap.dev (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) hCC hx)

end Verity.SampledTwo
