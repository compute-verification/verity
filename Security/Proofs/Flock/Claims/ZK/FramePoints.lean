import Proofs.Flock.Claims.ZK.FrameRank

/-!
# The frame's claim points are the session's (`FrameAdmits`)

At the opening's label the deployed frame's rank check reads a rep's claim points off the algebraic phase's coins as the
stream answered them, one flat array (`Frame.claimPoints`, `Masker.coins`); the session reads them off the rep's rounds
(`MaskRank.claimPoints`, `coin j e` element `e` of round `j`). In the phase's draw order the array holds round `0`'s
`K_SKIP` coins, round `1`'s `m − 13` outer ones, `z`, the `m − 6` rounds of `ρ`, `α`, `β`, the `k_log − 6` rounds of the
lincheck and `r'`, so:

* `extract_eq_map`: a run of an array that agrees with `f` is `f` over a range.
* **`claimPoints_frame`**: where the array holds round `1`'s coins from `K_SKIP`, `ρ`'s from `m − 6` and the lincheck's
  from `2 m − 10`, the frame's claim points are `MaskRank.claimPoints` at the rounds, as lists.
-/

namespace Claims.ZKServer.Rank

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (maskRank)

theorem extract_eq_map {α : Type} [Inhabited α] (a : Array α) {i n : ℕ} (hc : i + n ≤ a.size) (f : ℕ → α)
    (h : ∀ j < n, a[i + j]! = f j) : a.extract i (i + n) = (Array.range n).map f := by
  refine Array.ext (by simp; omega) fun j h₁ h₂ => ?_
  simp only [Array.getElem_extract, Array.getElem_map, Array.getElem_range]
  have hj : j < n := by simpa using h₂
  rw [← h j hj, getElem!_pos a (i + j) (by omega)]

theorem toList_extract_one {α : Type} (a : Array α) : (a.extract 1 a.size).toList = a.toList.drop 1 := by
  rw [Array.toList_extract, List.extract_eq_take_drop, List.take_of_length_le (by simp)]

/-- **The frame's claim points are the rounds'**: on an algebraic phase of `2 m + k_log − 15` coins holding round `1`'s
`m − 13` coins from `K_SKIP`, `ρ`'s rounds `3 …` from `m − 6` and the lincheck's rounds `5 + (m − 6) …` from `2 m − 10`,
`Frame.claimPoints` at the rep's layout returns `MaskRank.claimPoints` at the rounds' coins, as lists. -/
theorem claimPoints_frame {m kLog : ℕ} (k0 : ℕ) (hm : 13 ≤ m) (hk : 6 ≤ kLog) (a : Array F128)
    (hc : a.size = 2 * m + kLog - 15) (coin : ℕ → ℕ → F128)
    (h1 : ∀ e < m - 13, a[6 + e]! = coin 1 e) (hρ : ∀ j < m - 6, a[m - 6 + j]! = coin (3 + j) 0)
    (ht : ∀ j < kLog - 6, a[2 * m - 10 + j]! = coin (5 + (m - 6) + j) 0) :
    ∃ P, _root_.Flock.Firewall.Frame.claimPoints (_root_.Flock.Zk.PadLayout.new m kLog k0) a = .ok P ∧
      P.map Array.toList = FlockComplete.MaskRank.claimPoints m kLog coin := by
  have eρ : a.extract (m - 6) (m - 6 + (m - 6)) = (Array.range (m - 6)).map fun j => coin (3 + j) 0 :=
    extract_eq_map a (by omega) _ hρ
  have et : a.extract (2 * m - 10) (2 * m - 10 + (kLog - 6)) =
      (Array.range (kLog - 6)).map fun j => coin (5 + (m - 6) + j) 0 :=
    extract_eq_map a (by omega) _ ht
  have e1 : a.extract 6 (6 + (m - 13)) = (Array.range (m - 13)).map (coin 1) := extract_eq_map a (by omega) _ h1
  unfold _root_.Flock.Firewall.Frame.claimPoints
  simp only [_root_.Flock.Zk.PadLayout.new, _root_.Flock.Firewall.Frame.N_INNER, _root_.Flock.Firewall.Frame.K_SKIP]
  have hnm : ¬ m - 6 < 7 := by omega
  have hsz : a.size = 6 + (m - 6 - 7) + 1 + (m - 6) + 2 + (kLog - 6) + 1 := by omega
  simp only [hnm, hsz, decide_false, bne_self_eq_false, Bool.or_false, Bool.false_eq_true, ↓reduceIte]
  refine ⟨_, rfl, ?_⟩
  have a1 : 6 + (m - 6 - 7) + 1 = m - 6 := by omega
  have a2 : 6 + (m - 6 - 7) + 1 + (m - 6) + 2 = 2 * m - 10 := by omega
  have a3 : 6 + (m - 6 - 7) = 6 + (m - 13) := by omega
  rw [a2, a1, a3, eρ, et, e1]
  simp only [_root_.Flock.abcClaims, FlockComplete.MaskRank.claimPoints, List.map_toArray, List.map_cons,
    List.map_nil, toList_extract_one, Nat.sub_add_cancel (show 6 ≤ m by omega), Nat.sub_add_cancel hk]

open _root_.Flock.Firewall.Frame (Masker Cfg MaskSlot isLabel LABEL_PCS) in
/-- **A label the frame takes at the opening has full rank at the rounds**: where `Masker.label` takes the opening's
label in a masked phase of a rep with a mask slot, at a layout `PadLayout.new m k_log k0` whose coins hold the rounds as
`claimPoints_frame` reads them, the earlier reps' points and the rounds' have full rank over the slot's words. -/
theorem rank_of_label {c : Cfg} {s s' : Masker} {l out : ByteArray} (h : s.label c l = .ok (s', out))
    (hl : isLabel l LABEL_PCS = true) (hon : s.on = true) {ms : MaskSlot} (hms : c.mask = some ms) {m kLog k0 : ℕ}
    (hlay : c.lay = _root_.Flock.Zk.PadLayout.new m kLog k0) (hm : 13 ≤ m) (hk : 6 ≤ kLog)
    (hc : s.coins.size = 2 * m + kLog - 15) (coin : ℕ → ℕ → F128)
    (h1 : ∀ e < m - 13, s.coins[6 + e]! = coin 1 e) (hρ : ∀ j < m - 6, s.coins[m - 6 + j]! = coin (3 + j) 0)
    (ht : ∀ j < kLog - 6, s.coins[2 * m - 10 + j]! = coin (5 + (m - 6) + j) 0) :
    FlockComplete.MaskRank.maskRankOk (s.prior.map Array.toList ++ FlockComplete.MaskRank.claimPoints m kLog coin)
      ((List.range (ms.blocks * ms.n)).map (ms.word kLog)) = true := by
  obtain ⟨P, hP, hPl⟩ := claimPoints_frame k0 hm hk s.coins hc coin h1 hρ ht
  rw [← hlay] at hP
  have hnLc : c.lay.nLc + _root_.Flock.Firewall.Frame.K_SKIP = kLog := by
    rw [hlay]
    simp only [_root_.Flock.Zk.PadLayout.new, _root_.Flock.Firewall.Frame.K_SKIP]
    omega
  have hr : maskRank (s.prior ++ P) ms kLog = 64 * (2 * (s.prior ++ P).size) := by
    by_contra hne
    have hne' : maskRank (s.prior ++ P) ms kLog ≠ 128 * (s.prior.size + P.size) := by
      rw [Array.size_append] at hne; omega
    simp [Masker.label, hl, hon, hms, hP, hnLc] at h
    rw [show ∀ f : Array (Array F128) → _root_.Flock.V (Masker × ByteArray), (Except.ok P >>= f) = f P from
      fun _ => rfl] at h
    simp only [hne', ↓reduceIte] at h
    by_cases hp : s.pos = c.nPiop
    · rw [if_pos hp] at h
      cases h
    · rw [if_neg hp] at h
      cases h
  have := maskRankOk_of_maskRank (s.prior ++ P) ms kLog hr
  rwa [Array.map_append, hPl] at this

end Claims.ZKServer.Rank
