module

public import Proofs.Properties.Verity.SampledTwo.Cover

/-!
# SPOT[Flock] reads a coin only where `readsAt` says

`readsOnly`: `ReadsOnly` at every held copy. Each step of `SpotFlock.V` reads the coins of the rounds before it only at
rounds `readsAt` names (`next_congr`):

* rounds 0 to `stage` answer their own coin;
* round `stage + 1`'s statement check reads round 0's coin and the second stage's;
* a session's round reads the coins of that session's rounds so far, all before the verdict's round;
* the verdict reads, for each template `t`, the coins of at most `k₀ t` rounds of its session (`decidedAt` takes a
  prefix), which also come before the verdict's round.

None reads round `stage + 1`'s coin or the verdict round's. So two coin vectors equal at the rounds `readsAt` names give
the verifier the same steps, and one prover's runs on them are one run. `acceptsRun_of'` is `acceptsRun` with
`CoinsCover` and `ReadsOnly` discharged.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

theorem all_congr' {α : Type} {l : List α} {p q : α → Bool} (h : ∀ a ∈ l, p a = q a) : l.all p = l.all q := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.all_cons, h a (by simp), ih fun b hb => h b (by simp [hb])]

theorem any_congr' {α : Type} {l : List α} {p q : α → Bool} (h : ∀ a ∈ l, p a = q a) : l.any p = l.any q := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.any_cons, h a (by simp), ih fun b hb => h b (by simp [hb])]

/-- **A step of SPOT[Flock] reads the coins only at the rounds `readsAt` names.** -/
theorem next_congr (h : SpotFlock.Held) (x : Msg) (as : List Msg) {cs cs' : List FlockDeploy.Coin}
    (hl : cs.length = as.length) (hl' : cs'.length = as.length)
    (hr : ∀ n (hn : n < cs.length), readsAt h n → cs[n] = cs'[n]'(by omega)) :
    SpotFlock.next h x as cs = SpotFlock.next h x as cs' := by
  have hT3 := SpotFlock.stage_le_off h h.ms.length
  have hgetD : ∀ n, n < as.length → readsAt h n → cs.getD n default = cs'.getD n default := by
    intro n hn hrd
    simp only [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (show n < cs.length by omega),
      List.getElem?_eq_getElem (show n < cs'.length by omega), Option.getD_some]
    exact hr n (by omega) hrd
  have hdrop : ∀ t < h.ms.length, ∀ j ≤ SpotFlock.k₀ h t,
      (cs.drop (SpotFlock.off h t)).take j = (cs'.drop (SpotFlock.off h t)).take j := by
    intro t ht j hj
    have hle : SpotFlock.off h t + SpotFlock.k₀ h t ≤ SpotFlock.off h h.ms.length :=
      Claims.Sampling.sessOff_add_le (h.stage + 2) (SpotFlock.k₀ h) ht
    have h3 := SpotFlock.stage_le_off h t
    apply List.ext_getElem (by simp [hl, hl'])
    intro m h1 h2
    simp only [List.length_take, List.length_drop] at h1
    simp only [List.getElem_take, List.getElem_drop]
    exact hr _ (by omega) (And.intro (by omega) (by omega))
  unfold SpotFlock.next
  rw [hl, hl']
  by_cases h0 : as.length = 0 ∨ as.length ≠ as.length
  · rw [if_pos h0, if_pos h0]
  rw [if_neg h0, if_neg h0]
  have hpos : 0 < as.length := by
    rcases Nat.eq_zero_or_pos as.length with e | e
    · exact absurd (Or.inl e) h0
    · exact e
  by_cases h1 : as.length ≤ h.stage + 1
  · rw [if_pos h1, if_pos h1, hgetD (as.length - 1) (by omega) (And.intro (by omega) (by omega))]
  rw [if_neg h1, if_neg h1]
  by_cases h2 : as.length = h.stage + 2
  · have hs : SpotFlock.stageOf h cs = SpotFlock.stageOf h cs' := by
      funext i
      have := i.isLt
      exact hgetD (i.val + 1) (by omega) (And.intro (by omega) (by omega))
    rw [if_pos h2, if_pos h2, hgetD 0 hpos (And.intro (by omega) (by omega)), hs]
  rw [if_neg h2, if_neg h2]
  cases hs : SpotFlock.slotOf h as.length with
  | some t =>
    obtain ⟨ht, h4, h5⟩ := slotOf_some hs
    have e := hdrop t ht (SpotFlock.k₀ h t) le_rfl
    rw [List.take_of_length_le, List.take_of_length_le] at e
    · dsimp only
      unfold SpotFlock.sessStep
      rw [e]
    · simp only [List.length_drop]
      omega
    · simp only [List.length_drop]
      omega
  | none =>
    have ev : ∀ m, SpotFlock.verdict h m as cs = SpotFlock.verdict h m as cs' := by
      intro m
      unfold SpotFlock.verdict
      refine all_congr' fun t ht => any_congr' fun j hj => ?_
      simp only [List.mem_range] at ht hj
      unfold SpotFlock.decidedAt
      rw [hdrop t ht j (by omega)]
    dsimp only
    rw [ev]

/-- **SPOT[Flock] reads a coin only where `readsAt` says** (`ReadsOnly`), at every held copy. -/
theorem readsOnly (h : SpotFlock.Held) : ReadsOnly h := by
  intro H Ω P x ω r r' hr
  have hv : (fun as => (SpotFlock.V h).next x as ((List.range as.length).map (coinSeq r))) =
      (fun as => (SpotFlock.V h).next x as ((List.range as.length).map (coinSeq r'))) := by
    funext as
    refine next_congr h x as (by simp) (by simp) fun n hn hrd => ?_
    simp only [List.length_map, List.length_range] at hn
    simp only [List.getElem_map, List.getElem_range]
    unfold coinSeq
    by_cases hk : n < (SpotFlock.V h).k x
    · rw [dif_pos hk, dif_pos hk]
      exact hr ⟨n, hk⟩ hrd
    · rw [dif_neg hk, dif_neg hk]
  exact congrArg (fun vn => Claims.stepWith H P x ω vn ((SpotFlock.V h).k x) [] [] ⟨[], [], [], .rej⟩) hv

/-- **The form's accept is the run's accept**, `acceptsRun` with `CoinsCover` and `ReadsOnly` discharged. -/
theorem acceptsRun_of' (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hfit : SourcesFit A C) (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of A x C S dev P' hfit hplay hsess (readsOnly A.h)

end Verity.SampledTwo
