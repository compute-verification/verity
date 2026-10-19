import Proofs.Flock.Claims.ZK.Interact

/-!
# A run cut at a round

`cutT ask stop t n τ` is the transcript `τ` cut at round `t`: `n` rounds against `ask` of `τ`'s messages before `t`,
then the message `stop`. `stopAt pm t stop` is the prover `pm` stopped at round `t`, and its run is its own run cut
there (`interact_stopAt`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-- The cut prover: `τ`'s messages before round `t`, then the stop. -/
def cutMsg (stop : Msg) (t : ℕ) (τ : List (Msg × Out)) (h : List (Msg × Msg)) : Msg :=
  if h.length < t then ((τ[h.length]?).map Prod.fst).getD stop else stop

/-- **The transcript `τ` cut at round `t`**: `n` rounds of `τ`'s messages before `t`, then the stop, against `ask`. -/
def cutT (ask : List Msg → Out) (stop : Msg) (t n : ℕ) (τ : List (Msg × Out)) : List (Msg × Out) :=
  interact ask (cutMsg stop t τ) n []

/-- The prover `pm` stopped at round `t`. -/
def stopAt (pm : List (Msg × Msg) → Msg) (t : ℕ) (stop : Msg) (h : List (Msg × Msg)) : Msg :=
  if h.length < t then pm h else stop

theorem stopAt_of_lt {pm : List (Msg × Msg) → Msg} {t : ℕ} {stop : Msg} {h : List (Msg × Msg)} (hl : h.length < t) :
    stopAt pm t stop h = pm h := by
  simp [stopAt, hl]

theorem stopAt_of_ge {pm : List (Msg × Msg) → Msg} {t : ℕ} {stop : Msg} {h : List (Msg × Msg)}
    (hl : ¬ h.length < t) : stopAt pm t stop h = stop := by
  simp [stopAt, hl]

theorem cutMsg_of_lt {stop : Msg} {t : ℕ} {τ : List (Msg × Out)} {h : List (Msg × Msg)} (hl : h.length < t) :
    cutMsg stop t τ h = ((τ[h.length]?).map Prod.fst).getD stop := by
  simp [cutMsg, hl]

theorem cutMsg_of_ge {stop : Msg} {t : ℕ} {τ : List (Msg × Out)} {h : List (Msg × Msg)} (hl : ¬ h.length < t) :
    cutMsg stop t τ h = stop := by
  simp [cutMsg, hl]

theorem length_histOf_take (τ : List (Msg × Out)) (i : ℕ) : (histOf (τ.take i)).length = min i τ.length := by
  simp [histOf]

theorem histOf_take_take (τ : List (Msg × Out)) {i i' : ℕ} (h : i' ≤ i) :
    (histOf (τ.take i)).take i' = histOf (τ.take i') := by
  simp only [histOf, ← List.map_take, List.take_take, Nat.min_eq_left h]

/-- Round `i`'s message is the prover's on the history before it. -/
theorem interact_getElem_fst (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (i : ℕ), i < (interact ask pm n hist).length →
      ((interact ask pm n hist)[i]?).map Prod.fst = some (pm (hist ++ histOf ((interact ask pm n hist).take i)))
  | 0, _, _, hi => by simp [interact] at hi
  | n + 1, hist, i, hi => by
      simp only [interact] at hi ⊢
      cases hb : ask (hist.map Prod.fst ++ [pm hist]) with
      | msg m =>
          rw [hb] at hi
          cases i with
          | zero => simp [histOf]
          | succ i =>
              have := interact_getElem_fst ask pm n (hist ++ [(pm hist, m)]) i (by simpa using hi)
              rw [List.getElem?_cons_succ, this, List.take_succ_cons]
              simp [histOf, msgOf]
      | acc =>
          rw [hb] at hi
          simp only [List.length_singleton, Nat.lt_one_iff] at hi
          subst hi
          simp [histOf]
      | rej =>
          rw [hb] at hi
          simp only [List.length_singleton, Nat.lt_one_iff] at hi
          subst hi
          simp [histOf]

/-- The stopped prover's run agrees with the prover's before the stop. -/
theorem stopAt_take (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (t n : ℕ) :
    (interact ask (stopAt pm t stop) n []).take t = (interact ask pm n []).take t := by
  cases t with
  | zero => simp
  | succ j =>
      refine interact_take_congr ask pm (stopAt pm (j + 1) stop) n [] j fun i hi hlt => ?_
      have hl : (histOf ((interact ask pm n []).take i)).length = i := by
        rw [length_histOf_take]; omega
      simp only [List.nil_append]
      exact stopAt_of_lt (by rw [hl]; omega)

/-- **The stopped prover's run is its run cut**. -/
theorem interact_stopAt (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (stop : Msg) (t n : ℕ) :
    interact ask (stopAt pm t stop) n [] = cutT ask stop t n (interact ask pm n []) := by
  symm
  unfold cutT
  refine interact_congr ask (stopAt pm t stop) (cutMsg stop t (interact ask pm n [])) n [] fun i hi => ?_
  simp only [List.nil_append]
  have hl : (histOf ((interact ask (stopAt pm t stop) n []).take i)).length = i := by
    rw [length_histOf_take]; omega
  by_cases hit : i < t
  · rw [cutMsg_of_lt (by rw [hl]; exact hit), stopAt_of_lt (by rw [hl]; exact hit), hl]
    have htk := stopAt_take ask pm stop t n
    have e1 : (interact ask (stopAt pm t stop) n []).take i =
        ((interact ask (stopAt pm t stop) n []).take t).take i := by
      rw [List.take_take, Nat.min_eq_left hit.le]
    have e2 : (interact ask pm n []).take i = ((interact ask pm n []).take t).take i := by
      rw [List.take_take, Nat.min_eq_left hit.le]
    have hiτ : i < (interact ask pm n []).length := by
      have := congrArg List.length htk
      simp only [List.length_take] at this
      omega
    rw [interact_getElem_fst ask pm n [] i hiτ, e1, htk, ← e2]
    simp
  · rw [cutMsg_of_ge (by rw [hl]; exact hit), stopAt_of_ge (by rw [hl]; exact hit)]

end Claims.ZKServer
