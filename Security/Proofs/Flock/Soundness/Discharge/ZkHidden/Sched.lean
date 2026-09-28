import Proofs.Flock.Soundness.Discharge.ZkHidden.Statements
import Proofs.Flock.Level3.ArithFacts

/-!
# `ZkSchedOK`: `Zk.schedule`'s facts at a statement the `--zk` shape check passes

* The schedule half reads only `m`, which `Zk.schedule` accepts only from 25 to 35 (`schedule_range`); at each of
  those `zkSchedFacts` is checked by the kernel (`zkSchedFacts_all`, `decide +kernel`).
* `PadOK`'s radius: `100·(2^cols + t) ≤ 98·2^logLen` puts the padded rate at most `0.98 < 0.995²`.
* The inner code: `len` is at most `400` (`m ≤ 35`, `kLog ≤ m`, `k0 ≤ 6`), so `innerN ≤ 2^13`; it is a power of two,
  so `w.lo mod innerN` is uniform (level3's `lo_balanced`).
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine ZkLink ZkExec Accounting
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

/-- `Zk.schedule` accepts only an `m` with a `fast100` schedule. -/
theorem fast100_of_schedule {m : ℕ} {p : Flock.Schedule × ℕ} (h : Flock.Zk.schedule m = .ok p) :
    ∃ s, Flock.fast100 m = some s := by
  unfold Flock.Zk.schedule at h
  cases hq : Flock.Zk.queries0 m with
  | error e => rw [hq] at h; cases h
  | ok q =>
    cases hf : Flock.fast100 m with
    | none => rw [hq] at h; simp [hf, Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    | some s => exact ⟨s, rfl⟩

/-- `Zk.schedule` accepts `m` only from 25 to 35. -/
theorem schedule_range {m : ℕ} {p : Flock.Schedule × ℕ} (h : Flock.Zk.schedule m = .ok p) : 25 ≤ m ∧ m ≤ 35 := by
  refine ⟨?_, ?_⟩
  · by_contra hc
    unfold Flock.Zk.schedule Flock.Zk.queries0 at h
    simp [show m ≤ 24 by omega, Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · obtain ⟨s, hf⟩ := fast100_of_schedule h
    by_contra hc
    unfold Flock.fast100 at hf
    simp [show 35 < m by omega] at hf

/-- The schedule facts `DrawSetupZK` records, with `PadOK`'s decidable parts and `k0 ≤ 6`, as one check per `m`. -/
def zkSchedFacts (m : ℕ) : Bool :=
  match Zk.schedule m with
  | .ok (s, t) =>
    let a := schedOf s
    let l := a.levels.headD default
    decide (2 ≤ a.levels.length) && decide (l.lanes = a.k0) && decide (a.m - 7 = a.k0 + l.logCols) &&
      levelsChained a.levels a.ext && decide (a.m = m) && decide (l.logLen ≤ 40) && decide (t ≤ 2 ^ l.logCols) &&
      decide (2 ^ l.logCols + t ≤ 2 ^ l.logLen) && decide (1 < 2 ^ l.logCols + t) &&
      decide (100 * (2 ^ l.logCols + t) ≤ 98 * 2 ^ l.logLen) && decide (s.k0 ≤ 6)
  | .error _ => true

set_option maxRecDepth 100000 in
theorem zkSchedFacts_all : ∀ m, 25 ≤ m → m ≤ 35 → zkSchedFacts m = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- **The padded radius is positive** when the padded rate is at most `0.98`. -/
theorem padRadius_pos {l : Accounting.Level} {t : ℕ} (h : 100 * (2 ^ l.logCols + t) ≤ 98 * 2 ^ l.logLen) :
    0 < l.padRadius t := by
  unfold Level.padRadius eta
  have hL : (0 : ℝ) < 2 ^ l.logLen := by positivity
  have hr : l.padRate t ≤ 98 / 100 := by
    unfold Level.padRate
    rw [div_le_div_iff₀ hL (by norm_num)]
    exact_mod_cast (by linarith [h] : (2 ^ l.logCols + t) * 100 ≤ 98 * 2 ^ l.logLen)
  have hs : Real.sqrt (l.padRate t) < 199 / 200 :=
    (Real.sqrt_lt' (by norm_num)).2 (lt_of_le_of_lt hr (by norm_num))
  linarith

/-- The pads layout's length at `m ≤ 35`, `kLog ≤ m` and `k0 ≤ 6`. -/
theorem len_le {m kLog k0 : ℕ} (hm : m ≤ 35) (hk : kLog ≤ m) (h0 : k0 ≤ 6) :
    (Zk.PadLayout.new m kLog k0).len ≤ 400 := by
  simp only [Zk.PadLayout.new, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab, Zk.PadLayout.zp,
    Zk.PadLayout.lc, Zk.PadLayout.fc, Zk.PadLayout.fa]
  omega

set_option maxRecDepth 20000 in
set_option linter.constructorNameAsVariable false in
/-- **Query positions are uniform**: at `N = 2^b ≤ 2^64`, `w.lo mod N` hits a set `D` of positions with probability
`|D| / N`. -/
theorem prCoin_innerPos_le {b : ℕ} (hb : b ≤ 64) (hN : 0 < 2 ^ b) (D : Finset (Fin (2 ^ b))) :
    prCoin (fun w : 𝔽 => innerPos (2 ^ b) hN w ∈ D) ≤ (D.card : ℝ≥0∞) / (2 ^ b : ℕ) := by
  classical
  rw [prCoin_eq_card]
  have hcard : (Finset.univ.filter fun w : 𝔽 => innerPos (2 ^ b) hN w ∈ D).card = D.card * 2 ^ (128 - b) := by
    rw [Finset.card_eq_sum_card_fiberwise (s := Finset.univ.filter fun w : 𝔽 => innerPos (2 ^ b) hN w ∈ D) (t := D)
      (f := fun w => innerPos (2 ^ b) hN w) (fun w hw => (Finset.mem_filter.1 hw).2)]
    rw [Finset.sum_const_nat (m := 2 ^ (128 - b))]
    intro t ht
    have e : ((Finset.univ.filter fun w : 𝔽 => innerPos (2 ^ b) hN w ∈ D).filter
        fun w => innerPos (2 ^ b) hN w = t) = Finset.univ.filter fun w : 𝔽 => lo w % 2 ^ b = t.val := by
      ext w
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · rintro ⟨-, rfl⟩; rfl
      · intro h
        have e' : innerPos (2 ^ b) hN w = t := Fin.ext h
        exact ⟨e' ▸ ht, e'⟩
    rw [e, lo_balanced b hb t.val t.isLt, card_F]
    exact Nat.div_eq_of_eq_mul_left hN (by rw [← pow_add, Nat.sub_add_cancel (by omega : b ≤ 128)])
  rw [hcard, card_F, show (2 : ℕ) ^ 128 = 2 ^ b * 2 ^ (128 - b) by rw [← pow_add, Nat.add_sub_cancel' (by omega : b ≤ 128)]]
  apply le_of_eq
  rw [Nat.cast_mul, Nat.cast_mul, ENNReal.mul_div_mul_right _ _ (by positivity) (ENNReal.natCast_ne_top _)]

theorem zk_schedOK' : ZkSchedOK := by
  intro st sch tPad hwf _ _ _ hz
  obtain ⟨h1, h2⟩ := schedule_range hz
  have hf := zkSchedFacts_all _ h1 h2
  simp only [zkSchedFacts, hz, Bool.and_eq_true, decide_eq_true_eq] at hf
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨a1, a2⟩, a3⟩, a4⟩, a5⟩, b1⟩, b2⟩, b3⟩, b4⟩, b5⟩, hk0⟩ := hf
  have hlen := len_le h2 hwf.hk hk0
  have hlay : layOf st sch = Zk.PadLayout.new st.m st.c.kLog sch.k0 := rfl
  obtain ⟨k, hk⟩ := Nat.isPowerOfTwo_nextPowerOfTwo ((layOf st sch).len + Zk.K_PAD)
  have hle : ((layOf st sch).len + Zk.K_PAD).nextPowerOfTwo ≤ 2 ^ 10 :=
    Nat.nextPowerOfTwo_le (by rw [hlay]; simp only [Zk.K_PAD]; omega) ⟨10, rfl⟩
  have hinn : innerN (layOf st sch) = 2 ^ (k + 3) := by
    unfold innerN; rw [hk, pow_add]; rfl
  have hN : innerN (layOf st sch) < 2 ^ 64 := by
    unfold innerN
    calc ((layOf st sch).len + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE ≤ 2 ^ 10 * 2 ^ 3 :=
          Nat.mul_le_mul_right _ hle
      _ < 2 ^ 64 := by norm_num
  have hb : k + 3 ≤ 64 := by
    have : 2 ^ (k + 3) < 2 ^ 64 := hinn ▸ hN
    exact le_of_lt ((Nat.pow_lt_pow_iff_right (by norm_num)).1 this)
  refine ⟨⟨a1, a2, a3, a4⟩, a5, ⟨b1, b2, b3, b4, padRadius_pos b5⟩, hN, ?_, fun hN' D => ?_⟩
  · unfold innerN
    calc (layOf st sch).len + Zk.K_PAD ≤ ((layOf st sch).len + Zk.K_PAD).nextPowerOfTwo := Nat.le_nextPowerOfTwo _
      _ ≤ _ := Nat.le_mul_of_pos_right _ (by simp [Zk.INNER_LOG_INV_RATE])
  · revert hN' D
    rw [hinn]
    intro hN' D
    exact prCoin_innerPos_le hb hN' D

end FlockSoundness.Discharge.ZkHidden
