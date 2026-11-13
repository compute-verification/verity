module

public import Proofs.Properties.Verity.SampledTwo.JoinElems

/-!
# A case kept only on the plays a predicate holds of

Several of the stages' pieces are stated over every play the form accepts, with no word on the play's log:
`CutReads` (the cut reads a log of every accepted play), `DevFirst` and `DevSecond` (the coin map's release starts with
the stage's bytes), and through them `StageDraws` (the play's log shows a family's coins after one root). None of
them is known of every accepted play: the cut reads no log where a session's record carries a break
(`reads_or_break`), and the play's log shows one root for a family only where it is the log the check read. Each is
wanted only where the play is good (the check read a log, and the log it read is the device's), and that is all the
join uses them for.

`goodCase C G` is the case `C` kept on the plays `G` holds of and none elsewhere. The form accepts it exactly on the
plays it accepts `C` and `G` holds (`formAccepts_goodCase`), so a piece stated over every play the form accepts, taken
at `goodCase C G`, speaks of the good accepted plays only. On those plays the two cases have the same run of record's
input (`inputOn_goodCase`), so what the pieces conclude at `goodCase C G` holds at `C`:

* `coinsAgree_goodCase`: a group's coins;
* `firstFits_goodCase`, `secondFits_goodCase`: the stages' bytes within their rounds' releases.

Nothing here reads the form's check: `C` is any case and `G` any predicate.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

open Classical in
/-- **The case `C` on the plays `G` holds of**, and none elsewhere. -/
noncomputable def goodCase (C : CaseOf) (G : Verity.Exchange → Prop) (o : Verity.Exchange) : Option Case :=
  if G o then C o else none

theorem goodCase_of {C : CaseOf} {G : Verity.Exchange → Prop} {o : Verity.Exchange} (h : G o) :
    goodCase C G o = C o := by
  unfold goodCase
  exact if_pos h

theorem goodCase_some {C : CaseOf} {G : Verity.Exchange → Prop} {o : Verity.Exchange} {k : Case}
    (h : goodCase C G o = some k) : G o ∧ C o = some k := by
  unfold goodCase at h
  by_cases hg : G o
  · rw [if_pos hg] at h
    exact ⟨hg, h⟩
  · rw [if_neg hg] at h
    cases h

/-- **The form accepts the kept case exactly on the plays it accepts the case and `G` holds.** -/
theorem formAccepts_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} {o : Verity.Exchange} :
    FormAccepts (goodCase C G) o ↔ FormAccepts C o ∧ G o := by
  constructor
  · rintro ⟨k, hk, ha⟩
    obtain ⟨hg, hk'⟩ := goodCase_some hk
    exact ⟨⟨k, hk', ha⟩, hg⟩
  · rintro ⟨⟨k, hk, ha⟩, hg⟩
    exact ⟨k, (goodCase_of hg).trans hk, ha⟩

/-- **The run of record's input is the case's**, on a play `G` holds of. -/
theorem inputOn_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} {o : Verity.Exchange} (h : G o) :
    inputOn (goodCase C G) o = inputOn C o := by
  unfold inputOn
  rw [goodCase_of h]

/-- **The form's reading reads a play's coins from its run of record's input alone**: two cases with one input at a
play have one coin there at every round. -/
theorem formReading_coin (A : Live₂) {C C' : CaseOf} (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) (n : ℕ) :
    (formReading A C' S dev).coin o n = (formReading A C S dev).coin o n := by
  simp only [formReading, h]

/-- **The form's reading's agreement reads the run of record's input alone**: two cases with one input at a play have
one agreement there at every round. -/
theorem formReading_agree (A : Live₂) {C C' : CaseOf} (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) (n : ℕ) (c c' : FlockDeploy.Coin) :
    (formReading A C' S dev).agree o n c c' ↔ (formReading A C S dev).agree o n c c' := by
  simp only [formReading, h]

/-- **A group's coins at the case, from the kept case's**: `CoinsAgree` at `goodCase C G` and `G` gives it at `C` and
`G`. -/
theorem coinsAgree_goodCase {A : Live₂} {C : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin} {G : Verity.Exchange → Prop} {g : Group}
    (h : CoinsAgree A (formReading A (goodCase C G) S dev) (FormAccepts (goodCase C G)) G g) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) G g := by
  intro o hA hG i hr hg c hc c' hd
  have hin : inputOn (goodCase C G) o = inputOn C o := inputOn_goodCase hG
  have hr' : (formReading A (goodCase C G) S dev).reads i := hr
  have hc' : (formReading A (goodCase C G) S dev).coin o i = some c := (formReading_coin A S dev hin i).trans hc
  have key := h o (formAccepts_goodCase.2 ⟨hA, hG⟩) hG i hr' hg c hc' c'
    fun e he => (formReading_agree A S dev hin i e c').2 (hd e he)
  exact (formReading_agree A S dev hin i c c').1 key

/-- **`FirstFits` at the case, from the kept case's.** -/
theorem firstFits_goodCase {C : CaseOf} {L : CoinMap.Locate} {G : Verity.Exchange → Prop}
    (h : FirstFits (goodCase C G) L (FormAccepts (goodCase C G)) G) : FirstFits C L (FormAccepts C) G := by
  intro o hA hG
  obtain ⟨inp, hin, hsz⟩ := h o (formAccepts_goodCase.2 ⟨hA, hG⟩) hG
  exact ⟨inp, (inputOn_goodCase hG).symm.trans hin, hsz⟩

/-- **`SecondFits` at the case, from the kept case's.** -/
theorem secondFits_goodCase {A : Live₂} {C : CaseOf} {L : CoinMap.Locate} {G : Verity.Exchange → Prop}
    (h : SecondFits A (goodCase C G) L (FormAccepts (goodCase C G)) G) : SecondFits A C L (FormAccepts C) G := by
  intro o hA hG
  obtain ⟨inp, it, hin, hit, hsz⟩ := h o (formAccepts_goodCase.2 ⟨hA, hG⟩) hG
  exact ⟨inp, it, (inputOn_goodCase hG).symm.trans hin, hit, hsz⟩

end Verity.SampledTwo
