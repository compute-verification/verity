module

public import Proofs.Properties.Verity.SampledTwo.JoinElems

/-! A case kept only on the plays a predicate holds of (`goodCase`), so a piece stated over every accepted
play speaks of the good ones only. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims

open Classical in
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

theorem formAccepts_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} {o : Verity.Exchange} :
    FormAccepts (goodCase C G) o ↔ FormAccepts C o ∧ G o := by
  constructor
  · rintro ⟨k, hk, ha⟩
    obtain ⟨hg, hk'⟩ := goodCase_some hk
    exact ⟨⟨k, hk', ha⟩, hg⟩
  · rintro ⟨⟨k, hk, ha⟩, hg⟩
    exact ⟨k, (goodCase_of hg).trans hk, ha⟩

theorem inputOn_goodCase {C : CaseOf} {G : Verity.Exchange → Prop} {o : Verity.Exchange} (h : G o) :
    inputOn (goodCase C G) o = inputOn C o := by
  unfold inputOn
  rw [goodCase_of h]

theorem formReading_coin (A : Live₂) {C C' : CaseOf} (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) (n : ℕ) :
    (formReading A C' S dev).coin o n = (formReading A C S dev).coin o n := by
  simp only [formReading, h]

theorem formReading_agree (A : Live₂) {C C' : CaseOf} (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) {o : Verity.Exchange}
    (h : inputOn C' o = inputOn C o) (n : ℕ) (c c' : FlockDeploy.Coin) :
    (formReading A C' S dev).agree o n c c' ↔ (formReading A C S dev).agree o n c c' := by
  simp only [formReading, h]

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

theorem firstFits_goodCase {C : CaseOf} {L : CoinMap.Locate} {G : Verity.Exchange → Prop}
    (h : FirstFits (goodCase C G) L (FormAccepts (goodCase C G)) G) : FirstFits C L (FormAccepts C) G := by
  intro o hA hG
  obtain ⟨inp, hin, hsz⟩ := h o (formAccepts_goodCase.2 ⟨hA, hG⟩) hG
  exact ⟨inp, (inputOn_goodCase hG).symm.trans hin, hsz⟩

theorem secondFits_goodCase {A : Live₂} {C : CaseOf} {L : CoinMap.Locate} {G : Verity.Exchange → Prop}
    (h : SecondFits A (goodCase C G) L (FormAccepts (goodCase C G)) G) : SecondFits A C L (FormAccepts C) G := by
  intro o hA hG
  obtain ⟨inp, it, hin, hit, hsz⟩ := h o (formAccepts_goodCase.2 ⟨hA, hG⟩) hG
  exact ⟨inp, it, (inputOn_goodCase hG).symm.trans hin, hit, hsz⟩

end Verity.SampledTwo
