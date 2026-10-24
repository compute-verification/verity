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
* **`LeanRelease.coins_taken_of_live`**: on a live history up to the `Finish` whose last coins the frame takes
  (`refusesCoins = false`), this holds at every prefix of at least three rounds. Below the history's length the prefix
  is a frame round of a live run (`refusesCoins_of_live`); at its length it is the premise. `Hello`'s and the `Commit`'s
  coins (prefixes of one and two rounds) go to the session, not to a frame (`takeAt`), so nothing here is said of them.
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
  cases ho : L.opens x ρ h with
  | false =>
    rw [L.answered_of_not_opens x ρ h ho] at ha
    cases ha
  | true =>
    refine ⟨rfl, ?_⟩
    rw [L.answered_of_opens x ρ h ho] at ha
    unfold frameAnswer at ha
    split at ha
    · unfold takeAt at ha
      rw [if_neg (by omega)] at ha
      by_cases hs : L.rep x h.length = L.rep x (h.length - 1)
      · rw [if_pos (decide_eq_true hs)] at ha
        have hc : L.cfgAt x ρ (h.length - 1) = L.cfg x ρ h := by
          unfold cfg cfgAt
          rw [hs]
        rw [hc]
        exact ⟨m, ha⟩
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

/-- **On a live history the frames took every round's coins** (`FrameAdmits`' first piece): on a live history up to
the `Finish` whose last coins the frame takes, at every prefix of `i ≥ 3` rounds the transport's coin check passed and
the frame of round `i - 1`'s rep, at its state and configuration, answered the coins without refusal. Below the
history's length the prefix is a live frame round (`refusesCoins_of_live`); at it, the premise. -/
theorem coins_taken_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true) (hle : h.length ≤ L.last x)
    (hr : L.refusesCoins x ρ h = false) {i : ℕ} (h3 : 3 ≤ i) (hi : i ≤ h.length) :
    L.opens x ρ (h.take i) = true ∧
      ∃ s, (L.prog x (h.take i)).answer (L.cfgAt x ρ (i - 1)) (L.coinsOf x (h.take i)) = .ok s := by
  have hlen : (h.take i).length = i := by
    rw [List.length_take]
    omega
  have hr' : L.refusesCoins x ρ (h.take i) = false := by
    rcases Nat.lt_or_ge i h.length with hlt | hge
    · refine L.refusesCoins_of_live wk x w ρ κ (h.take i) ?_ (L.live_take wk x w ρ κ h hl i)
      rw [hlen]
      unfold kindAt
      rw [if_neg (by omega), if_neg (by omega)]
    · rw [List.take_of_length_le hge]
      exact hr
  unfold refusesCoins at hr'
  cases ha : L.answered x ρ (h.take i) with
  | error e =>
    rw [ha] at hr'
    cases hr'
  | ok m =>
    have hc := L.coins_taken_of_answered x ρ (h.take i) (by omega) ha
    rwa [hlen] at hc

end LeanRelease

end Claims.ZKServer
