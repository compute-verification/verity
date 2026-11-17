module

public import Proofs.Properties.Verity.SampledTwo.StmtsRound
public import Proofs.Properties.Verity.SampledTwo.SessionLast
public import Proofs.Properties.Verity.SampledTwo.RoundOne

/-! The messages' bound at the form's reading, round by round (`reached_form`); `LinkFixed`, `LastFixed`
and the hypotheses the rounds' own theorems take are not established here. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

def byRound {β : Type} (A : Live₂) (zero one mid stmt : β) (first rep last : ℕ → β) (past : β) (i : ℕ) : β :=
  if i = 0 then zero
  else if i = 1 then one
  else if i ≤ A.h.stage then mid
  else if i = A.h.stage + 1 then stmt
  else match SpotFlock.slotOf A.h (i + 1) with
    | some t =>
      if i + 1 = SpotFlock.off A.h t + SpotFlock.k₀ A.h t then last t
      else if i = SpotFlock.off A.h t then first t
      else rep t
    | none => past

section ByRound

variable {β : Type} (A : Live₂) {zero one mid stmt : β} {first rep last : ℕ → β} {past : β}

theorem byRound_zero : byRound A zero one mid stmt first rep last past 0 = zero := by
  unfold byRound
  rw [ite_eq_left rfl]

theorem byRound_one : byRound A zero one mid stmt first rep last past 1 = one := by
  unfold byRound
  rw [ite_eq_right (show ¬ (1 : ℕ) = 0 by omega), ite_eq_left rfl]

theorem byRound_mid {i : ℕ} (h2 : 2 ≤ i) (h3 : i ≤ A.h.stage) :
    byRound A zero one mid stmt first rep last past i = mid := by
  unfold byRound
  rw [ite_eq_right (show ¬ i = 0 by omega), ite_eq_right (show ¬ i = 1 by omega), ite_eq_left h3]

theorem byRound_stmt : byRound A zero one mid stmt first rep last past (A.h.stage + 1) = stmt := by
  have hpos : 0 < A.h.stage := Nat.succ_pos _
  unfold byRound
  rw [ite_eq_right (show ¬ A.h.stage + 1 = 0 by omega), ite_eq_right (show ¬ A.h.stage + 1 = 1 by omega),
    ite_eq_right (show ¬ A.h.stage + 1 ≤ A.h.stage by omega), ite_eq_left rfl]

theorem byRound_first {t : ℕ} (ht : t < A.h.ms.length) (hK : 2 ≤ SpotFlock.k₀ A.h t) :
    byRound A zero one mid stmt first rep last past (SpotFlock.off A.h t) = first t := by
  have hso := SpotFlock.stage_le_off A.h t
  have hslot : SpotFlock.slotOf A.h (SpotFlock.off A.h t + 1) = some t :=
    SpotFlock.slotOf_eq A.h ht (by omega) (by omega)
  unfold byRound
  rw [ite_eq_right (show ¬ SpotFlock.off A.h t = 0 by omega), ite_eq_right (show ¬ SpotFlock.off A.h t = 1 by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t ≤ A.h.stage by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t = A.h.stage + 1 by omega), hslot]
  show (if SpotFlock.off A.h t + 1 = SpotFlock.off A.h t + SpotFlock.k₀ A.h t then last t
    else if SpotFlock.off A.h t = SpotFlock.off A.h t then first t else rep t) = first t
  rw [ite_eq_right (show ¬ SpotFlock.off A.h t + 1 = SpotFlock.off A.h t + SpotFlock.k₀ A.h t by omega),
    ite_eq_left rfl]

theorem byRound_rep {t j : ℕ} (ht : t < A.h.ms.length) (h0 : 0 < j) (hj : j + 1 < SpotFlock.k₀ A.h t) :
    byRound A zero one mid stmt first rep last past (SpotFlock.off A.h t + j) = rep t := by
  have hso := SpotFlock.stage_le_off A.h t
  have hslot : SpotFlock.slotOf A.h (SpotFlock.off A.h t + j + 1) = some t :=
    SpotFlock.slotOf_eq A.h ht (by omega) (by omega)
  unfold byRound
  rw [ite_eq_right (show ¬ SpotFlock.off A.h t + j = 0 by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t + j = 1 by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t + j ≤ A.h.stage by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t + j = A.h.stage + 1 by omega), hslot]
  show (if SpotFlock.off A.h t + j + 1 = SpotFlock.off A.h t + SpotFlock.k₀ A.h t then last t
    else if SpotFlock.off A.h t + j = SpotFlock.off A.h t then first t else rep t) = rep t
  rw [ite_eq_right (show ¬ SpotFlock.off A.h t + j + 1 = SpotFlock.off A.h t + SpotFlock.k₀ A.h t by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t + j = SpotFlock.off A.h t by omega)]

theorem byRound_last {t : ℕ} (ht : t < A.h.ms.length) (hK : 1 ≤ SpotFlock.k₀ A.h t) :
    byRound A zero one mid stmt first rep last past (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) = last t := by
  have hso := SpotFlock.stage_le_off A.h t
  have hslot : SpotFlock.slotOf A.h (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) + 1) = some t :=
    SpotFlock.slotOf_eq A.h ht (by omega) (by omega)
  unfold byRound
  rw [ite_eq_right (show ¬ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) = 0 by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) = 1 by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) ≤ A.h.stage by omega),
    ite_eq_right (show ¬ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) = A.h.stage + 1 by omega), hslot]
  show (if SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) + 1 = SpotFlock.off A.h t + SpotFlock.k₀ A.h t
    then last t
    else if SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) = SpotFlock.off A.h t then first t else rep t) =
      last t
  rw [ite_eq_left (show SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) + 1 =
    SpotFlock.off A.h t + SpotFlock.k₀ A.h t by omega)]

theorem byRound_past {i : ℕ} (h0 : i ≠ 0) (h1 : i ≠ 1) (h3 : ¬ i ≤ A.h.stage) (h4 : i ≠ A.h.stage + 1)
    (hs : SpotFlock.slotOf A.h (i + 1) = none) : byRound A zero one mid stmt first rep last past i = past := by
  unfold byRound
  rw [ite_eq_right h0, ite_eq_right h1, ite_eq_right h3, ite_eq_right h4, hs]

end ByRound

theorem round_cases (A : Live₂) (i : ℕ) :
    i = 0 ∨ i = 1 ∨ (2 ≤ i ∧ i ≤ A.h.stage) ∨ i = A.h.stage + 1 ∨
      (∃ t, t < A.h.ms.length ∧ (i = SpotFlock.off A.h t ∨
        (∃ j, 0 < j ∧ j + 1 < SpotFlock.k₀ A.h t ∧ i = SpotFlock.off A.h t + j) ∨
        (1 ≤ SpotFlock.k₀ A.h t ∧ i = SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)))) ∨
      (i ≠ 0 ∧ i ≠ 1 ∧ ¬ i ≤ A.h.stage ∧ i ≠ A.h.stage + 1 ∧ SpotFlock.slotOf A.h (i + 1) = none) := by
  by_cases h0 : i = 0
  · exact Or.inl h0
  by_cases h1 : i = 1
  · exact Or.inr (Or.inl h1)
  by_cases h3 : i ≤ A.h.stage
  · exact Or.inr (Or.inr (Or.inl ⟨by omega, h3⟩))
  by_cases h4 : i = A.h.stage + 1
  · exact Or.inr (Or.inr (Or.inr (Or.inl h4)))
  cases hs : SpotFlock.slotOf A.h (i + 1) with
  | none => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨h0, h1, h3, h4, rfl⟩))))
  | some t =>
    obtain ⟨ht, hlo, hhi⟩ := slotOf_some hs
    refine Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨t, ht, ?_⟩))))
    by_cases hj0 : i = SpotFlock.off A.h t
    · exact Or.inl hj0
    by_cases hjl : i + 1 = SpotFlock.off A.h t + SpotFlock.k₀ A.h t
    · exact Or.inr (Or.inr ⟨by omega, by omega⟩)
    · exact Or.inr (Or.inl ⟨i - SpotFlock.off A.h t, by omega, by omega, by omega⟩)

noncomputable abbrev dgOf (A : Live₂) (θ : ByteArray) (i : ℕ)
    (p : List Verity.Release × Option (Verity.Frame × ByteArray)) : Option (List Msg) :=
  byRound A (some []) (Messages.endAnswers p) (some []) (some [Verity.sha512Msg θ.data.toList])
    (fun _ => Messages.endAnswers p) (fun _ => Messages.endAnswers p)
    (fun t => (SessionCoins.closeOf t p.1).map fun b => [b.data.toList]) (some []) i

noncomputable abbrev formDg (A : Live₂) (L : CoinMap.Locate) (θ : ByteArray) (o : Verity.Exchange) (i : ℕ) :
    Option (List Msg) :=
  dgOf A θ i (Messages.logBefore L o i)

noncomputable abbrev formRaw (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) (i : ℕ) : Option (List Msg) :=
  byRound A (some []) ((interiorTagged C o).map fun t => [t]) (some []) (some [o.params.data.toList])
    (fun t => SessionCoins.linkShownAt C o t)
    (fun _ => ((formReading A C S dev).msg o i).map fun m => [m.tail])
    (fun t => (SessionCoins.evidenceAt C o t).map fun b => [b.data.toList]) (some []) i

def LinkFixed (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) : Prop :=
  ∀ o o' t, FormAccepts C o → Good o → FormAccepts C o' → Good o' → t < A.h.ms.length → o.params = o'.params →
    Messages.logBefore L o (SpotFlock.off A.h t) = Messages.logBefore L o' (SpotFlock.off A.h t) →
    SessionCoins.linkShownAt C o t = SessionCoins.linkShownAt C o' t →
      S.msg o t 0 = S.msg o' t 0 ∨ ∃ p, Messages.blobsAlt o o' (SpotFlock.off A.h t) = some p

def LastFixed (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) : Prop :=
  ∀ o o' t, FormAccepts C o → Good o → FormAccepts C o' → Good o' → t < A.h.ms.length → o.params = o'.params →
    Messages.logBefore L o (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) =
      Messages.logBefore L o' (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) →
    SessionCoins.evidenceAt C o t = SessionCoins.evidenceAt C o' t →
      S.msg o t (SpotFlock.k₀ A.h t - 1) = S.msg o' t (SpotFlock.k₀ A.h t - 1) ∨
        ∃ p, Messages.blobsAlt o o' (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) = some p

theorem explainsOr_session_first (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (θ : ByteArray) (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ) (hfix : LinkFixed A C S Good L)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) {t : ℕ} (ht : t < A.h.ms.length)
    (hK : 1 ≤ SpotFlock.k₀ A.h t)
    (hr : ∀ o, FormAccepts C o → Good o → raw o (SpotFlock.off A.h t) = SessionCoins.linkShownAt C o t) :
    Messages.ExplainsOr (formReading A C S dev) (FormAccepts C) Good raw (Messages.logBefore L) Messages.blobsAlt
      (SpotFlock.off A.h t) := by
  intro o o' hA hG hA' hG' hc hraw
  rw [hr o hA hG, hr o' hA' hG'] at hraw
  have hlt : SpotFlock.off A.h t < SpotFlock.off A.h t + SpotFlock.k₀ A.h t := by omega
  rw [Messages.formMsg_session A C S dev o ht (Nat.le_refl _) hlt,
    Messages.formMsg_session A C S dev o' ht (Nat.le_refl _) hlt, Nat.sub_self]
  exact hfix o o' t hA hG hA' hG' ht ((hθ o hA hG).trans (hθ o' hA' hG').symm) hc hraw

theorem explainsOr_session_last (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (θ : ByteArray) (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ) (hfix : LastFixed A C S Good L)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) {t : ℕ} (ht : t < A.h.ms.length)
    (hK : 1 ≤ SpotFlock.k₀ A.h t)
    (hr : ∀ o, FormAccepts C o → Good o → raw o (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) =
      (SessionCoins.evidenceAt C o t).map fun b => [b.data.toList]) :
    Messages.ExplainsOr (formReading A C S dev) (FormAccepts C) Good raw (Messages.logBefore L) Messages.blobsAlt
      (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) := by
  intro o o' hA hG hA' hG' hc hraw
  rw [hr o hA hG, hr o' hA' hG'] at hraw
  have hinj : Function.Injective fun b : ByteArray => [b.data.toList] :=
    fun _ _ h => ByteArray.ext (Array.ext' (List.cons.inj h).1)
  have hev : SessionCoins.evidenceAt C o t = SessionCoins.evidenceAt C o' t := Option.map_injective hinj hraw
  have hlt : SpotFlock.k₀ A.h t - 1 < SpotFlock.k₀ A.h t := by omega
  rw [Messages.formMsg_session A C S dev o ht (Nat.le_add_right _ _) (Nat.add_lt_add_left hlt _),
    Messages.formMsg_session A C S dev o' ht (Nat.le_add_right _ _) (Nat.add_lt_add_left hlt _),
    Nat.add_sub_cancel_left]
  exact hfix o o' t hA hG hA' hG' ht ((hθ o hA hG).trans (hθ o' hA' hG').symm) hc hev

theorem showsAt_form (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (tab : ℕ → String) (ℓ₁ ℓ₂ : String)
    (Good : Verity.Exchange → Prop) (θ : ByteArray) (L : CoinMap.Locate)
    (hL : L = SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (hne : ℓ₁ ≠ ℓ₂)
    (hSecond : SecondShown C Good)
    (hDraw : CoinMap.StageSamples (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hsz : SessionCoins.HeldSize) (hlog : SessionCoins.InteriorLogged C Good) (hst : StmtsRead A C S)
    (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ) (hRise : SessionCoins.LogRising A C Good tab)
    (hcov : SessionCoins.RecordsCover A C Good) (hmsgs : SessionCoins.MsgsOfRecords A C S Good tab)
    (hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) Good L) (i : ℕ) :
    Messages.ShowsAt (formReading A C S dev) (FormAccepts C) Good (formDg A L θ) (formRaw A C S dev) i := by
  subst hL
  have hLoc := SessionCoins.locates_run A (CoinMap.stageLocate ℓ₁ ℓ₂) tab
  have hOpen := SessionCoins.logOpened_run A C Good (CoinMap.stageLocate ℓ₁ ℓ₂) tab hsz hlog
    (SessionCoins.roundsRecorded_of hsz hcov hmsgs)
  have hAns := SessionCoins.logAnswers_run A C Good (CoinMap.stageLocate ℓ₁ ℓ₂) tab hsz hlog
  have hClose := SessionCoins.logCloses_run A C Good (CoinMap.stageLocate ℓ₁ ℓ₂) tab hsz hlog hcov
  rcases round_cases A i with rfl | rfl | ⟨h2, h3⟩ | rfl |
    ⟨t, ht, rfl | ⟨j, h0, hj, rfl⟩ | ⟨hK1, rfl⟩⟩ | ⟨h0, h1, h3, h4, hs⟩
  · exact showsAt_form_zero A C S dev (FormAccepts C) Good
      (formDg A (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ) (formRaw A C S dev)
      (fun _ _ _ => byRound_zero A) (fun _ _ _ => byRound_zero A)
  · exact showsAt_form_one A C S dev tab ℓ₁ ℓ₂ Good hne hSecond hDraw (SessionCoins.interiorReads C hsz) hlog
      (formDg A (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ) (formRaw A C S dev)
      (fun _ _ _ => byRound_one A) (fun _ _ _ => byRound_one A)
  · exact showsAt_form_mid A C S dev (FormAccepts C) Good
      (formDg A (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ) (formRaw A C S dev) h2 h3
      (fun _ _ _ => byRound_mid A h2 h3) (fun _ _ _ => byRound_mid A h2 h3)
  · exact showsAt_form_stmts A C S dev Good θ hst hθ
      (formDg A (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ) (formRaw A C S dev)
      (fun _ _ _ => byRound_stmt A) (fun _ _ _ => byRound_stmt A)
  · intro o hA hG
    have hK := (SessionCoins.last_msg_of hcov hmsgs ht hA hG).1
    exact showsAt_session_first A C S dev Good (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) tab hLoc
      hRise hOpen hAns hAsk (formDg A (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ)
      (formRaw A C S dev) ht hK (fun _ _ _ => byRound_first A ht hK) (fun _ _ _ => byRound_first A ht hK) o hA hG
  · exact showsAt_session_mid A C S dev Good (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) tab hLoc
      hRise hOpen hAns hAsk (SessionCoins.midDigests_of hmsgs)
      (formDg A (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ) (formRaw A C S dev) ht h0 hj
      (fun _ _ _ => byRound_rep A ht h0 hj) (fun _ _ _ => byRound_rep A ht h0 hj)
  · intro o hA hG
    have hK := (SessionCoins.last_msg_of hcov hmsgs ht hA hG).1
    exact showsAt_session_last A C S dev Good (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) hClose
      (formDg A (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ) (formRaw A C S dev) ht hK
      (fun _ hA' hG' => (SessionCoins.last_msg_of hcov hmsgs ht hA' hG').2)
      (fun _ _ _ => byRound_last A ht hK1) (fun _ _ _ => byRound_last A ht hK1) o hA hG
  · exact Messages.showsAt_none [] (fun o _ _ => form_msg_none A C S dev o h0 h1 h3 h4 hs)
      (fun _ _ _ => byRound_past A h0 h1 h3 h4 hs) (fun _ _ _ => byRound_past A h0 h1 h3 h4 hs)

theorem explainsOr_form (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (tab : ℕ → String) (ℓ₁ ℓ₂ : String)
    (Good : Verity.Exchange → Prop) (θ : ByteArray) (L : CoinMap.Locate)
    (hL : L = SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (hsz : SessionCoins.HeldSize)
    (hlog : SessionCoins.InteriorLogged C Good) (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ)
    (hcov : SessionCoins.RecordsCover A C Good) (hmsgs : SessionCoins.MsgsOfRecords A C S Good tab)
    (hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) Good L) (hStmts : StmtsFixed A C S Good L)
    (hLink : LinkFixed A C S Good L) (hLast : LastFixed A C S Good L) (i : ℕ) :
    Messages.ExplainsOr (formReading A C S dev) (FormAccepts C) Good (formRaw A C S dev) (Messages.logBefore L)
      Messages.blobsAlt i := by
  subst hL
  have hOpen := SessionCoins.logOpened_run A C Good (CoinMap.stageLocate ℓ₁ ℓ₂) tab hsz hlog
    (SessionCoins.roundsRecorded_of hsz hcov hmsgs)
  rcases round_cases A i with rfl | rfl | ⟨h2, h3⟩ | rfl |
    ⟨t, ht, rfl | ⟨j, h0, hj, rfl⟩ | ⟨hK1, rfl⟩⟩ | ⟨h0, h1, h3, h4, hs⟩
  · exact Messages.explainsOr_of Messages.blobsAlt (explains_form_zero A C S dev (FormAccepts C) Good
      (formRaw A C S dev) (Messages.logBefore (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)))
  · exact Messages.explainsOr_of Messages.blobsAlt (explains_form_one A C S dev (FormAccepts C) Good
      (formRaw A C S dev) (Messages.logBefore (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab))
      (fun _ _ _ => byRound_one A))
  · exact Messages.explainsOr_of Messages.blobsAlt (explains_form_mid A C S dev (FormAccepts C) Good
      (formRaw A C S dev) (Messages.logBefore (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) h2 h3)
  · exact explainsOr_form_stmts A C S dev Good (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) hStmts
      (formRaw A C S dev) (fun _ _ _ => byRound_stmt A)
  · intro o o' hA hG hA' hG'
    have hK := (SessionCoins.last_msg_of hcov hmsgs ht hA hG).1
    exact explainsOr_session_first A C S dev Good (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ hθ
      hLink (formRaw A C S dev) ht (by omega) (fun _ _ _ => byRound_first A ht hK) o o' hA hG hA' hG'
  · exact Messages.explainsOr_of Messages.blobsAlt (explains_session_mid A C S dev Good
      (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) hOpen hAsk (formRaw A C S dev)
      (Messages.logBefore (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab)) ht hj
      (fun _ _ _ => byRound_rep A ht h0 hj))
  · exact explainsOr_session_last A C S dev Good (SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) θ hθ
      hLast (formRaw A C S dev) ht hK1 (fun _ _ _ => byRound_last A ht hK1)
  · exact Messages.explainsOr_of Messages.blobsAlt
      (Messages.explains_const [] fun o _ _ => form_msg_none A C S dev o h0 h1 h3 h4 hs)

theorem reached_form (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (tab : ℕ → String) (ℓ₁ ℓ₂ : String)
    (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hL : L = SessionCoins.runLocate A (CoinMap.stageLocate ℓ₁ ℓ₂) tab) (q : ℕ) (hne : ℓ₁ ≠ ℓ₂)
    (hSecond : SecondShown C Good)
    (hDraw : CoinMap.StageSamples (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hsz : SessionCoins.HeldSize) (hlog : SessionCoins.InteriorLogged C Good) (hst : StmtsRead A C S)
    (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ) (hRise : SessionCoins.LogRising A C Good tab)
    (hcov : SessionCoins.RecordsCover A C Good) (hmsgs : SessionCoins.MsgsOfRecords A C S Good tab)
    (hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) Good L) (hStmts : StmtsFixed A C S Good L)
    (hLink : LinkFixed A C S Good L) (hLast : LastFixed A C S Good L) :
    Messages.MessagesReached V A x θ F c σ
      (Messages.frameLog V A x θ F c σ L (formReading A C S dev) (FormAccepts C) Good (formDg A L θ)
        (formRaw A C S dev))
      (formReading A C S dev) (FormAccepts C) Good
      (Messages.missLog V A x θ F c σ L (formReading A C S dev) (FormAccepts C) Good (formDg A L θ)
        (formRaw A C S dev) Messages.blobsAlt q) := by
  have hShow : Messages.ShowsPlaced A x (formReading A C S dev) (FormAccepts C) Good L := by
    subst hL
    exact Messages.showsPlaced_form A x C S dev Good (CoinMap.stageLocate ℓ₁ ℓ₂) tab hAsk
  exact Messages.reachedLog V A x θ F c σ L (formReading A C S dev) (FormAccepts C) Good hShow (formDg A L θ)
    (formRaw A C S dev) Messages.blobsAlt q (fun i _ p => dgOf A θ i p) (fun _ _ _ _ => rfl)
    (fun i _ => showsAt_form A C S dev tab ℓ₁ ℓ₂ Good θ L hL hne hSecond hDraw hsz hlog hst hθ hRise hcov hmsgs
      hAsk i)
    Messages.blobsAlt_collides
    (fun i _ => explainsOr_form A C S dev tab ℓ₁ ℓ₂ Good θ L hL hsz hlog hθ hcov hmsgs hAsk hStmts hLink hLast i)

end Verity.SampledTwo
