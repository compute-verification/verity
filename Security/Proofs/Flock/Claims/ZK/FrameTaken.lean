import Proofs.Flock.Claims.ZK.FinishRound

/-!
# The frames took every round's coins on a live history

`LeanRelease.FrameAdmits`' first piece, for any `LeanRelease`: the coins of each round the frames read are ones the
frame that read them took without refusal.

* `LeanRelease.live_take`: a live history's prefixes are live.
* `LeanRelease.coins_taken_of_answered`: where the frame takes the coins after a history of at least three rounds
  (`answered`), the transport's coin check passed it (`opens`), and the frame of the rep that released its last round,
  at that rep's state (`prog`) and configuration (`cfgAt`), answered the coins (`Masker.answer`) without refusal, as
  `takeAt` routes them, whether the next round is that rep's or the next rep's first.
* **`LeanRelease.coins_taken_of_take`**: on any history whose last coins the frame takes (`refusesCoins = false`), this
  holds at every prefix of at least three rounds, as a refusal sticks (`refusesCoins_of_take`); `coins_taken_of_live`
  is it on a live history. `Hello`'s and the `Commit`'s coins (prefixes of one and two rounds) go to the session, not to
  a frame (`takeAt`), so nothing here is said of them.
-/

namespace Claims.ZKServer

open Claims

namespace LeanRelease

open _root_.Flock.Firewall.Frame (Masker)

theorem cfg_eq_cfgAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) :
    L.cfg x ρ h = L.cfgAt x ρ h.length := rfl

/-- **A live history's prefixes are live.** -/
theorem live_take {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true) (i : ℕ) :
    L.toSrv.live wk x w ρ κ (h.take i) = true := by
  unfold ZkServer.live
  refine List.all_eq_true.mpr fun j hj => ?_
  have hj' : j ≤ (h.take i).length := Nat.lt_succ_iff.mp (List.mem_range.mp hj)
  rw [List.length_take] at hj'
  have e : (h.take i).take j = h.take j := by
    rw [List.take_take]
    congr 1
    omega
  rw [e]
  exact L.toSrv.pass_of_live wk x w ρ κ h hl (by omega)

/-- **The frame of the rep that released the last round took its coins**: where the frame takes the coins after a
history of at least three rounds, the transport's coin check passed the history, and the rep of its last round, at
its state and configuration, answered the coins. Where the next round is that rep's, this is the state the next round
is checked at; else the rep's answer is discarded and the next rep starts at its first-round state (`takeAt`). -/
theorem coins_taken_of_answered {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (hn : 3 ≤ h.length) {m : Masker} (ha : L.answered x ρ h = .ok m) :
    L.opens x ρ h = true ∧ ∃ s, (L.prog x h).answer (L.cfgAt x ρ (h.length - 1)) (L.coinsOf x h) = .ok s := by
  obtain ⟨ho, -, ha⟩ := answered_ok ha
  refine ⟨ho, ?_⟩
  · unfold frameAnswer at ha
    split at ha
    · unfold takeAt at ha
      rw [if_neg (by omega)] at ha
      by_cases hs : L.rep x h.length = L.rep x (h.length - 1)
      · rw [if_pos (decide_eq_true hs)] at ha
        have hc : L.cfgAt x ρ (h.length - 1) = L.cfg x ρ h := by
          unfold cfg cfgAt
          rw [hs]
        rw [hc]
        exact ⟨m, (bind_rankStop_ok ha).1⟩
      · rw [if_neg (fun e => hs (of_decide_eq_true e))] at ha
        cases hp : (L.prog x h).answer (L.cfgAt x ρ (h.length - 1)) (L.coinsOf x h) with
        | error e =>
          rw [hp] at ha
          cases ha
        | ok s => exact ⟨s, rfl⟩
    · rename_i hg
      rw [List.getLast?_eq_none_iff] at hg
      subst hg
      simp at hn

/-- **On a history whose last coins the frame takes, the frames took every round's coins**: at every prefix of `i ≥ 3`
rounds the transport's coin check passed and the frame of round `i - 1`'s rep, at its state and configuration, answered
the coins without refusal. A refusal after a prefix would stick (`refusesCoins_of_take`), so the frame took the coins
after each prefix too, whether or not the run is live. -/
theorem coins_taken_of_take {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (hr : L.refusesCoins x ρ h = false) {i : ℕ} (h3 : 3 ≤ i) (hi : i ≤ h.length) :
    L.opens x ρ (h.take i) = true ∧
      ∃ s, (L.prog x (h.take i)).answer (L.cfgAt x ρ (i - 1)) (L.coinsOf x (h.take i)) = .ok s := by
  have hlen : (h.take i).length = i := by
    rw [List.length_take]
    omega
  have hr' : L.refusesCoins x ρ (h.take i) = false := by
    cases hb : L.refusesCoins x ρ (h.take i) with
    | false => rfl
    | true =>
      rw [L.refusesCoins_of_take x ρ h i hb] at hr
      cases hr
  unfold refusesCoins at hr'
  cases ha : L.answered x ρ (h.take i) with
  | error e =>
    rw [ha] at hr'
    cases hr'
  | ok m =>
    have hc := L.coins_taken_of_answered x ρ (h.take i) (by omega) ha
    rwa [hlen] at hc

/-- **On a live history the frames took every round's coins** (`FrameAdmits`' first piece): `coins_taken_of_take`,
which reads neither the run's liveness nor the history's length. -/
theorem coins_taken_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (_hl : L.toSrv.live wk x w ρ κ h = true) (_hle : h.length ≤ L.last x)
    (hr : L.refusesCoins x ρ h = false) {i : ℕ} (h3 : 3 ≤ i) (hi : i ≤ h.length) :
    L.opens x ρ (h.take i) = true ∧
      ∃ s, (L.prog x (h.take i)).answer (L.cfgAt x ρ (i - 1)) (L.coinsOf x (h.take i)) = .ok s :=
  L.coins_taken_of_take x ρ h hr h3 hi

end LeanRelease

end Claims.ZKServer
