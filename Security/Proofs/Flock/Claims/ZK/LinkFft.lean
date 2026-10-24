import Proofs.Flock.Claims.Complete.Zk.Level0Pad
import Proofs.Flock.Claims.Complete.Replay.LigeritoWeight
import Proofs.Flock.Claims.Complete.ZkTables.Round0
import Proofs.Flock.Soundness.Refine.Final
import Proofs.Flock.Soundness.ZK.PadNonvanishing40
import Flock.Firewall.Link

/-!
# `Flock.Firewall.Link`'s FFT is level 0's padded codeword

`Flock.Firewall.Link.rowsFft` evaluates each lane's entry, one polynomial in the novel basis over `cols + 1` bits
(`coeffs`), on all `2^d` points by the additive FFT (`fft`). Where the padding fits below `2^cols` and `cols < d ≤ 64`,
its rows are `FlockComplete.encodeRowsZk`'s (`rowsFft_eq`), so the recheck's rows are the analysis's in every case
(`rows_eq`).

* `What_omega_low`: `Ŵ_j(ω_q)` reads the bits of `q` from `k ≤ j` up: `s_j` is additive and vanishes below `2^k`.
* `fftSum_succ`: the butterfly, `T_s(n + 1, u) = T_{s+1}(n, u_even) + Ŵ_s·T_{s+1}(n, u_odd)` at one point.
* `fft_get`: entry `r` of `fft … s n u` is `T_s(n, u)` at `ω_{r·2^s}`, since `T_{s+1}` is the same at `ω_{r·2^s}` and
  `ω_{⌊r/2⌋·2^(s+1)}`.
* `fftSum_coeffs`: at `s = 0` over `cols + 1` bits, a lane's coefficients give its words' encoding plus its padding,
  `FlockComplete.encodeRowsZk_entry`'s sum.
-/

namespace Flock.Firewall.Link

open FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## `Ŵ_j(ω_q)` reads the bits of `q` from `j` up -/

theorem mul_two_pow_add_eq_xor (a b k : ℕ) (hb : b < 2 ^ k) : a * 2 ^ k + b = a * 2 ^ k ^^^ b := by
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.mul_comm, Nat.two_pow_add_eq_or_of_lt hb, Nat.testBit_or, Nat.testBit_xor]
  by_cases hj : j < k
  · have : (2 ^ k * a).testBit j = false := by
      rw [Nat.testBit_two_pow_mul]
      simp [Nat.not_le.mpr hj]
    simp [this]
  · have : b.testBit j = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hb (Nat.pow_le_pow_right (by omega) (by omega)))
    simp [this]

theorem toUInt64_xor (x y : ℕ) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64) :
    (x ^^^ y).toUInt64 = x.toUInt64 ^^^ y.toUInt64 := by
  apply UInt64.toNat.inj
  simp only [UInt64.toNat_xor, Nat.toUInt64_eq, UInt64.toNat_ofNat']
  rw [Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy, Nat.mod_eq_of_lt (Nat.xor_lt_two_pow hx hy)]

/-- `ω_{x ⊕ y} = ω_x + ω_y`: the low word's bits are the coefficients of `x^0, …, x^63`. -/
theorem omega_xor (x y : ℕ) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64) :
    FlockLevel3.omega (x ^^^ y) = FlockLevel3.omega x + FlockLevel3.omega y := by
  simp only [FlockLevel3.omega]
  rw [← FlockLevel3.of_add]
  show FlockLevel3.GF128.of ⟨(x ^^^ y).toUInt64, 0⟩ = FlockLevel3.GF128.of ⟨x.toUInt64 ^^^ y.toUInt64, 0 ^^^ 0⟩
  congr 2
  exact toUInt64_xor x y hx hy

/-- `s_j(ω_q)` reads the bits of `q` from `k ≤ j` up. -/
theorem sPoly_omega_low (j k a b : ℕ) (hk : k ≤ j) (hk64 : k ≤ 64) (hb : b < 2 ^ k) (h : a * 2 ^ k + b < 2 ^ 64) :
    (FlockLevel3.sPoly j).eval (FlockLevel3.omega (a * 2 ^ k + b)) =
      (FlockLevel3.sPoly j).eval (FlockLevel3.omega (a * 2 ^ k)) := by
  rw [mul_two_pow_add_eq_xor a b k hb,
    omega_xor _ _ (Nat.lt_of_le_of_lt (Nat.le_add_right _ _) h) (Nat.lt_of_le_of_lt (Nat.le_add_left _ _) h),
    FlockSoundness.ZK.sPoly_eval_add, FlockSoundness.ZK.sPoly_eval_omega_lt k hk64 b hb j hk, add_zero]

theorem What_omega_low (j k a b : ℕ) (hk : k ≤ j) (hk64 : k ≤ 64) (hb : b < 2 ^ k) (h : a * 2 ^ k + b < 2 ^ 64) :
    FlockLevel3.What j (FlockLevel3.omega (a * 2 ^ k + b)) = FlockLevel3.What j (FlockLevel3.omega (a * 2 ^ k)) := by
  unfold FlockLevel3.What
  rw [sPoly_omega_low j k a b hk hk64 hb h]

/-! ## The butterfly -/

/-- `T_s(n, u)` at `x`: `Σ_{c < 2^n} u(c)·Π_{i ∈ bits(c)} Ŵ_{s+i}(x)`. -/
noncomputable def fftSum (s n : ℕ) (u : ℕ → 𝔽) (x : 𝔽) : 𝔽 :=
  ∑ c ∈ Finset.range (2 ^ n), u c * ∏ i : Fin n, if c.testBit i then FlockLevel3.What (s + i) x else 1

theorem sum_range_two_mul {M : Type*} [AddCommMonoid M] (f : ℕ → M) : ∀ m : ℕ,
    ∑ c ∈ Finset.range (2 * m), f c = ∑ c ∈ Finset.range m, (f (2 * c) + f (2 * c + 1))
  | 0 => by simp
  | m + 1 => by
    rw [show 2 * (m + 1) = 2 * m + 1 + 1 by ring, Finset.sum_range_succ f (2 * m + 1), Finset.sum_range_succ f (2 * m),
      sum_range_two_mul f m, Finset.sum_range_succ (fun c => f (2 * c) + f (2 * c + 1)) m, add_assoc]

theorem testBit_two_mul_zero (c : ℕ) : (2 * c).testBit 0 = false := by
  rw [Nat.testBit_zero]
  simp

theorem testBit_two_mul_add_one_zero (c : ℕ) : (2 * c + 1).testBit 0 = true := by
  rw [Nat.testBit_zero]
  have : (2 * c + 1) % 2 = 1 := by omega
  simp [this]

theorem testBit_two_mul_succ (c i : ℕ) : (2 * c).testBit (i + 1) = c.testBit i := by
  rw [Nat.testBit_succ, show 2 * c / 2 = c by omega]

theorem testBit_two_mul_add_one_succ (c i : ℕ) : (2 * c + 1).testBit (i + 1) = c.testBit i := by
  rw [Nat.testBit_succ, show (2 * c + 1) / 2 = c by omega]

theorem sum_split {M : Type*} [CommSemiring M] (R : Finset ℕ) (a b p0 p1 q : ℕ → M) (w : M)
    (h0 : ∀ c, p0 c = q c) (h1 : ∀ c, p1 c = w * q c) :
    ∑ c ∈ R, (a c * p0 c + b c * p1 c) = ∑ c ∈ R, a c * q c + w * ∑ c ∈ R, b c * q c := by
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [h0 c, h1 c]
  ring

theorem fftSum_succ (s n : ℕ) (u : ℕ → 𝔽) (x : 𝔽) :
    fftSum s (n + 1) u x =
      fftSum (s + 1) n (fun c => u (2 * c)) x + FlockLevel3.What s x * fftSum (s + 1) n (fun c => u (2 * c + 1)) x := by
  have hi : ∀ i : ℕ, s + (i + 1) = s + 1 + i := fun i => by omega
  have hp0 : ∀ c : ℕ, (∏ i : Fin (n + 1), if (2 * c).testBit i then FlockLevel3.What (s + i) x else 1) =
      ∏ i : Fin n, if c.testBit i then FlockLevel3.What (s + 1 + i) x else 1 := by
    intro c
    rw [Fin.prod_univ_succ]
    first
      | (rw [Fin.val_zero, testBit_two_mul_zero, ite_eq_right Bool.false_ne_true, one_mul]
         exact Finset.prod_congr rfl fun i _ => by rw [Fin.val_succ, testBit_two_mul_succ, hi])
      | (simp only [Fin.val_zero, Fin.val_succ, testBit_two_mul_zero, testBit_two_mul_succ, hi, Bool.false_eq_true,
          ↓reduceIte, one_mul]; done)
  have hp1 : ∀ c : ℕ, (∏ i : Fin (n + 1), if (2 * c + 1).testBit i then FlockLevel3.What (s + i) x else 1) =
      FlockLevel3.What s x * ∏ i : Fin n, if c.testBit i then FlockLevel3.What (s + 1 + i) x else 1 := by
    intro c
    rw [Fin.prod_univ_succ]
    first
      | (rw [Fin.val_zero, testBit_two_mul_add_one_zero, ite_eq_left rfl, Nat.add_zero]
         congr 1
         exact Finset.prod_congr rfl fun i _ => by rw [Fin.val_succ, testBit_two_mul_add_one_succ, hi])
      | (simp only [Fin.val_zero, Fin.val_succ, testBit_two_mul_add_one_zero, testBit_two_mul_add_one_succ, hi,
          eq_self_iff_true, ↓reduceIte, Nat.add_zero]; done)
  simp only [fftSum]
  rw [pow_succ, mul_comm (2 ^ n) 2, sum_range_two_mul]
  first
    | exact sum_split _ (fun c => u (2 * c)) (fun c => u (2 * c + 1)) _ _ _ _ hp0 hp1
    | (rw [Finset.mul_sum, ← Finset.sum_add_distrib]
       exact Finset.sum_congr rfl fun c _ => by rw [hp0 c, hp1 c]; ring)
    | (simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
       exact Finset.sum_congr rfl fun c _ => by rw [hp0 c, hp1 c]; ring)

theorem fftSum_congr (s n : ℕ) (u : ℕ → 𝔽) (x y : 𝔽) (h : ∀ i, FlockLevel3.What (s + i) x = FlockLevel3.What (s + i) y) :
    fftSum s n u x = fftSum s n u y := by
  unfold fftSum
  refine Finset.sum_congr rfl fun c _ => ?_
  congr 1
  exact Finset.prod_congr rfl fun i _ => by rw [h i]

/-! ## The FFT -/

theorem split_two (r s : ℕ) : r * 2 ^ s = r / 2 * 2 ^ (s + 1) + r % 2 * 2 ^ s := by
  have h := Nat.div_add_mod r 2
  rw [Nat.pow_succ]
  calc r * 2 ^ s = (2 * (r / 2) + r % 2) * 2 ^ s := by rw [h]
    _ = r / 2 * (2 ^ s * 2) + r % 2 * 2 ^ s := by
      rw [Nat.add_mul, Nat.mul_comm 2 (r / 2), Nat.mul_assoc, Nat.mul_comm 2]

theorem mod_two_mul_lt (r s : ℕ) : r % 2 * 2 ^ s < 2 ^ (s + 1) := by
  rw [Nat.pow_succ]
  have : r % 2 ≤ 1 := by omega
  have hp : 0 < 2 ^ s := Nat.two_pow_pos s
  calc r % 2 * 2 ^ s ≤ 1 * 2 ^ s := Nat.mul_le_mul_right _ this
    _ < 2 ^ s * 2 := by omega

theorem half_lt (r d s : ℕ) (hs : s < d) (hr : r < 2 ^ (d - s)) : r / 2 < 2 ^ (d - (s + 1)) := by
  have : 2 ^ (d - s) = 2 ^ (d - (s + 1)) * 2 := by rw [← Nat.pow_succ]; congr 1; omega
  omega

theorem mul_two_pow_lt_word (r d s : ℕ) (hs : s ≤ d) (hd : d ≤ 64) (hr : r < 2 ^ (d - s)) : r * 2 ^ s < 2 ^ 64 := by
  have h1 : r * 2 ^ s < 2 ^ (d - s) * 2 ^ s := Nat.mul_lt_mul_of_pos_right hr (Nat.two_pow_pos s)
  rw [← Nat.pow_add, Nat.sub_add_cancel hs] at h1
  exact Nat.lt_of_lt_of_le h1 (Nat.pow_le_pow_right (by omega) hd)

/-- At `sv`'s inverses, `whatsInvGo` is `whatsGo`. -/
theorem whatsInvGo_eq (sv : Array F128) (fuel : ℕ) : ∀ (i : ℕ) (s : F128) (out : Array F128), i + fuel ≤ sv.size →
    whatsInvGo sv (sv.map F128.inv) fuel i s out = whatsGo sv fuel i s out := by
  induction fuel with
  | zero => intro i s out _; rfl
  | succ fuel ih =>
    intro i s out h
    have hi : i < sv.size := by omega
    have hm : (sv.map F128.inv)[i]! = (sv[i]!).inv := by
      rw [getElem!_pos _ i (by simpa using hi), getElem!_pos sv i hi, Array.getElem_map]
    simp only [whatsInvGo, whatsGo, hm]
    exact ih (i + 1) _ _ (by omega)

/-- **Level `s`'s twiddle** at `r` is `Ŵ_s(ω_{r·2^s})`. -/
theorem twiddles_get (N d s r : ℕ) (hs : s < N) (hr : r < 2 ^ (d - s)) :
    fOf ((twiddles (svTable N) d N)[s]!)[r]! = FlockLevel3.What s (FlockLevel3.omega (r * 2 ^ s)) := by
  unfold twiddles twiddlesWith
  rw [FlockSoundness.Refine.getElem!_range_map _ hs, FlockSoundness.Refine.getElem!_range_map _ hr,
    whatsInvGo_eq _ _ _ _ _ (by rw [FlockLevel3.svTable_size]; omega)]
  exact whats_get N (r * 2 ^ s) (s + 1) s (by omega) (by omega)

/-- **The FFT's entry `r`** at level `s` with `n` coefficient bits left is `T_s(n, u)` at `ω_{r·2^s}`. -/
theorem fft_get (N d : ℕ) (hN : N ≤ d) (hd : d ≤ 64) (n : ℕ) : ∀ (s : ℕ) (u : ℕ → F128), s + n ≤ N → ∀ r,
    r < 2 ^ (d - s) →
    fOf (fft (twiddles (svTable N) d N) d s n u)[r]! =
      fftSum s n (fun c => fOf (u c)) (FlockLevel3.omega (r * 2 ^ s)) := by
  induction n with
  | zero =>
    intro s u _ r hr
    simp only [fft]
    rw [getElem!_pos _ r (by simpa using hr), Array.getElem_replicate]
    simp [fftSum]
  | succ n ih =>
    intro s u hsn r hr
    have hs : s < d := by omega
    have hr2 := half_lt r d s hs hr
    have hw := mul_two_pow_lt_word r d s hs.le hd hr
    have hx : ∀ i, FlockLevel3.What (s + 1 + i) (FlockLevel3.omega (r / 2 * 2 ^ (s + 1))) =
        FlockLevel3.What (s + 1 + i) (FlockLevel3.omega (r * 2 ^ s)) := by
      intro i
      rw [split_two r s]
      exact (What_omega_low (s + 1 + i) (s + 1) (r / 2) (r % 2 * 2 ^ s) (by omega) (by omega) (mod_two_mul_lt r s)
        (by rw [← split_two r s]; exact hw)).symm
    simp only [fft]
    rw [FlockSoundness.Refine.getElem!_range_map _ hr, fOf_add, fOf_mul,
      ih (s + 1) _ (by omega) (r / 2) hr2, ih (s + 1) _ (by omega) (r / 2) hr2,
      twiddles_get N d s r (by omega) hr, fftSum_succ]
    first
      | (rw [fftSum_congr (s + 1) n (fun c => fOf (u (2 * c))) _ _ hx,
          fftSum_congr (s + 1) n (fun c => fOf (u (2 * c + 1))) _ _ hx]
         done)
      | exact congrArg₂ (· + ·) (fftSum_congr _ _ _ _ _ hx)
          (congrArg (FlockLevel3.What s (FlockLevel3.omega (r * 2 ^ s)) * ·) (fftSum_congr _ _ _ _ _ hx))

/-! ## Level 0's rows -/

theorem rowsFft_size (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) :
    (rowsFft lv tPad g extra lanePad).size = 2 ^ lv.d := by
  simp [rowsFft]

theorem rowsFft_get_size (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) {q : ℕ}
    (hq : q < 2 ^ lv.d) : ((rowsFft lv tPad g extra lanePad)[q]!).size = 2 ^ lv.lanes + Zk.EXTRA_LANES := by
  simp only [rowsFft]
  rw [FlockSoundness.Refine.getElem!_range_map _ hq]
  simp

theorem encodeRowsZk_size (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) :
    (FlockComplete.encodeRowsZk lv tPad g extra lanePad).size = 2 ^ lv.d := by
  simp [FlockComplete.encodeRowsZk]

theorem encodeRowsZk_get_size (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) {q : ℕ}
    (hq : q < 2 ^ lv.d) :
    ((FlockComplete.encodeRowsZk lv tPad g extra lanePad)[q]!).size = 2 ^ lv.lanes + Zk.EXTRA_LANES := by
  simp only [FlockComplete.encodeRowsZk]
  rw [FlockSoundness.Refine.getElem!_range_map _ hq]
  simp

/-- **Entry `l` of `rowsFft`'s row `q`** is lane `l`'s `T_0(cols + 1)` at `ω_q`. -/
theorem rowsFft_get (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) (hc : lv.cols < lv.d)
    (hd : lv.d ≤ 64) {q l : ℕ} (hq : q < 2 ^ lv.d) (hl : l < 2 ^ lv.lanes + Zk.EXTRA_LANES) :
    fOf ((rowsFft lv tPad g extra lanePad)[q]!)[l]! =
      fftSum 0 (lv.cols + 1) (fun c => fOf (laneCoeffs lv tPad g extra lanePad l c)) (FlockLevel3.omega q) := by
  have h := fft_get (lv.cols + 1) lv.d (by omega) hd (lv.cols + 1) 0 (laneCoeffs lv tPad g extra lanePad l) (by omega) q
    (by simpa using hq)
  rw [Nat.pow_zero, Nat.mul_one] at h
  simp only [rowsFft]
  rw [FlockSoundness.Refine.getElem!_range_map _ hq, FlockSoundness.Refine.getElem!_range_map _ hl, Array.map_map,
    FlockSoundness.Refine.getElem!_range_map _ hl]
  exact h

theorem fOf_muAt (tPad : ℕ) (lp : ℕ → F128) (i : ℕ) :
    fOf (muAt ((Array.range tPad).map lp) i) = if i < tPad then fOf (lp i) else 0 := by
  unfold muAt
  by_cases hi : i < tPad
  · rw [ite_eq_left (by simpa using hi), ite_eq_left hi, FlockSoundness.Refine.getElem!_range_map _ hi]
  · rw [ite_eq_right (by simpa using hi), ite_eq_right hi, fOf_zero]

theorem prod_What (c L q : ℕ) :
    (∏ i : Fin L, if c.testBit i then FlockLevel3.What i (FlockLevel3.omega q) else 1) =
      FlockSoundness.execArith.Xhat c L (FlockSoundness.execArith.omega q) :=
  rfl

theorem sum_range_of_zero {M : Type*} [AddCommMonoid M] (f : ℕ → M) (a b : ℕ) (hab : a ≤ b)
    (h : ∀ i, a ≤ i → i < b → f i = 0) : ∑ i ∈ Finset.range b, f i = ∑ i ∈ Finset.range a, f i := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hab
  have hz : ∑ i ∈ Finset.range k, f (a + i) = 0 :=
    Finset.sum_eq_zero fun i hi => h (a + i) (by omega) (by rw [Finset.mem_range] at hi; omega)
  rw [Finset.sum_range_add, hz, add_zero]

/-- **A lane's `T_0(cols + 1)`** on its coefficients: its words in the novel basis over `cols` bits, plus its padding
`Σ_i μ_i·(c·X̂_i + X̂_{2^cols + i})`. -/
theorem fftSum_coeffs (cols q tPad : ℕ) (v lp : ℕ → F128) (ht : tPad ≤ 2 ^ cols) :
    fftSum 0 (cols + 1) (fun c => fOf (coeffs cols v ((Array.range tPad).map lp) c)) (FlockLevel3.omega q) =
      ∑ c ∈ Finset.range (2 ^ cols),
          fOf (v c) * FlockSoundness.execArith.Xhat c (cols + 1) (FlockSoundness.execArith.omega q) +
        ∑ i ∈ Finset.range tPad, fOf (lp i) *
          (fOf Zk.PAD_C * FlockSoundness.execArith.Xhat i (cols + 1) (FlockSoundness.execArith.omega q) +
            FlockSoundness.execArith.Xhat (2 ^ cols + i) (cols + 1) (FlockSoundness.execArith.omega q)) := by
  have hW : ∀ c, (∏ i : Fin (cols + 1), if c.testBit i then FlockLevel3.What i (FlockLevel3.omega q) else 1) =
      FlockSoundness.execArith.Xhat c (cols + 1) (FlockSoundness.execArith.omega q) :=
    fun c => prod_What c (cols + 1) q
  simp only [fftSum, Nat.zero_add, hW]
  rw [pow_succ, mul_two, Finset.sum_range_add]
  have h1 : ∀ c ∈ Finset.range (2 ^ cols), fOf (coeffs cols v ((Array.range tPad).map lp) c) *
      FlockSoundness.execArith.Xhat c (cols + 1) (FlockSoundness.execArith.omega q) =
      fOf (v c) * FlockSoundness.execArith.Xhat c (cols + 1) (FlockSoundness.execArith.omega q) +
      fOf (muAt ((Array.range tPad).map lp) c) *
        (fOf Zk.PAD_C * FlockSoundness.execArith.Xhat c (cols + 1) (FlockSoundness.execArith.omega q)) := by
    intro c hc
    rw [coeffs, ite_eq_left (Finset.mem_range.1 hc), fOf_add, fOf_mul]
    ring
  have h2 : ∀ i ∈ Finset.range (2 ^ cols), fOf (coeffs cols v ((Array.range tPad).map lp) (2 ^ cols + i)) *
      FlockSoundness.execArith.Xhat (2 ^ cols + i) (cols + 1) (FlockSoundness.execArith.omega q) =
      fOf (muAt ((Array.range tPad).map lp) i) *
        FlockSoundness.execArith.Xhat (2 ^ cols + i) (cols + 1) (FlockSoundness.execArith.omega q) := by
    intro i _
    rw [coeffs, ite_eq_right (by omega), Nat.add_sub_cancel_left]
  rw [Finset.sum_congr rfl h1, Finset.sum_congr rfl h2, Finset.sum_add_distrib, add_assoc, ← Finset.sum_add_distrib]
  congr 1
  rw [sum_range_of_zero _ tPad (2 ^ cols) ht ?hz]
  · refine Finset.sum_congr rfl fun i hi => ?_
    rw [fOf_muAt, ite_eq_left (Finset.mem_range.1 hi)]
    ring
  · intro i hi _
    simp only [fOf_muAt, ite_eq_right (show ¬ i < tPad by omega), zero_mul, add_zero]

theorem arr2_ext {α : Type} [Inhabited α] {A B : Array (Array α)} (n m : ℕ) (hA : A.size = n) (hB : B.size = n)
    (hA' : ∀ q < n, (A[q]!).size = m) (hB' : ∀ q < n, (B[q]!).size = m)
    (h : ∀ q < n, ∀ l < m, (A[q]!)[l]! = (B[q]!)[l]!) : A = B := by
  apply Array.ext (by omega)
  intro q h1 h2
  have hq : q < n := by omega
  rw [← getElem!_pos A q h1, ← getElem!_pos B q h2]
  apply Array.ext (by rw [hA' q hq, hB' q hq])
  intro l l1 l2
  have hl : l < m := by rw [← hA' q hq]; exact l1
  rw [← getElem!_pos _ l l1, ← getElem!_pos _ l l2]
  exact h q hq l hl

/-- **The FFT's rows are level 0's padded codeword** where the padding fits below `2^cols` and `cols < d ≤ 64`. -/
theorem rowsFft_eq (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) (ht : tPad ≤ 2 ^ lv.cols)
    (hc : lv.cols < lv.d) (hd : lv.d ≤ 64) :
    rowsFft lv tPad g extra lanePad = FlockComplete.encodeRowsZk lv tPad g extra lanePad := by
  refine arr2_ext (2 ^ lv.d) (2 ^ lv.lanes + Zk.EXTRA_LANES) (rowsFft_size ..) (encodeRowsZk_size ..)
    (fun q hq => rowsFft_get_size lv tPad g extra lanePad hq) (fun q hq => encodeRowsZk_get_size lv tPad g extra lanePad hq)
    fun q hq l hl => ?_
  have hv := fftSum_coeffs lv.cols q tPad
    (if l < 2 ^ lv.lanes then fun c => g[c * 2 ^ lv.lanes + l]! else extra (l - 2 ^ lv.lanes)) (lanePad l) ht
  apply fOf_injective
  rw [rowsFft_get lv tPad g extra lanePad hc hd hq hl]
  refine Eq.trans hv ?_
  rw [FlockComplete.encodeRowsZk_entry lv tPad g extra lanePad hq hl ht,
    Fin.sum_univ_eq_sum_range (fun i => fOf (lanePad l i) * (fOf Zk.PAD_C *
      FlockSoundness.execArith.Xhat i (lv.cols + 1) (FlockSoundness.execArith.omega q) +
      FlockSoundness.execArith.Xhat (2 ^ lv.cols + i) (lv.cols + 1) (FlockSoundness.execArith.omega q))) tPad]

/-- **The recheck's rows are level 0's padded codeword**: by the FFT (`rowsFft_eq`) or row by row (`rowsDirect`, which
is `encodeRowsZk` as written). -/
theorem rows_eq (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) :
    rows lv tPad g extra lanePad = FlockComplete.encodeRowsZk lv tPad g extra lanePad := by
  unfold rows
  split
  · rename_i h
    exact rowsFft_eq lv tPad g extra lanePad h.1 h.2.1 h.2.2
  · rfl

end Flock.Firewall.Link
