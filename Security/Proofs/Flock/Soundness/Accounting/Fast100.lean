import Definitions.Flock.Accounting.Fast100

/-!
# The bound for `fast100` run twice on one root

For every witness size the profile supports and every statement shape the three statements of
PROTOCOL.md §16 can take (at most `2^27` bits per block, 1024 public regions and 64 link-point
coordinates, well beyond today's statements), `tableError` is at most

* `2^-205` for every `22 ≤ m ≤ 35` (`tableError_le_of_le_33`, `tableError_le_of_34_35`), at the
  analysis slack `η = 1/200`.

At `η = 1/50` these were 2^-195.5 and, with the seventh level at `m = 34, 35`, 2^-195.4. The slack
is the analysis's choice: the radius `1 − √ρ − η` and the list bound `1/(2ηρ)` follow it, and
the verifier never reads it.

**Method.** Each real term is bounded by a rational one (`…Up`): `√(2^-r)` by a rational upper
bound, and the mutual-correlated-agreement numerator of BCHKS25 Theorem 4.6 by a crude closed
form (its terms sit below 2^-180, against about 2^-102.5 for the queries). The rational bound is then
checked by exact arithmetic for each `m`.
-/

namespace FlockSoundness.Accounting

open Real

/-! ## Rational upper bounds, term by term -/

theorem sqrtUp_nonneg (r : ℕ) : 0 ≤ sqrtUp r := by
  unfold sqrtUp; split_ifs <;> positivity

theorem sqrt_rate_le (l : Level) (hr : l.logInvRate ≤ 7) :
    Real.sqrt l.rate ≤ (sqrtUp l.logInvRate : ℝ) := by
  rw [Real.sqrt_le_left (by exact_mod_cast sqrtUp_nonneg _)]
  unfold Level.rate
  generalize l.logInvRate = r at hr ⊢
  interval_cases r <;> norm_num [sqrtUp]

theorem queryError_le (l : Level) (hr : l.logInvRate ≤ 7) :
    l.queryError ≤ (queryUp l : ℝ) := by
  unfold Level.queryError queryUp eta
  push_cast
  gcongr
  exact sqrt_rate_le l hr

theorem mcaA_bounds (l : Level) (hr : l.logInvRate ≤ 7) (hc : 1 ≤ l.logCols) :
    0 ≤ l.mcaA ∧ l.mcaA ≤ (mcaUp l : ℝ) := by
  -- names for the pieces of ArkLib's expression
  set r := l.logInvRate with hr_def
  set c := l.logCols with hc_def
  have hk2 : (2 : ℕ) ≤ 2 ^ c := by
    calc (2 : ℕ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ c := Nat.pow_le_pow_right (by norm_num) hc
  set k : ℕ := 2 ^ c with hk_def
  have hkR : (2 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk2
  have hkm1 : (((k - 1 : ℕ) : ℝ)) = (k : ℝ) - 1 := by
    rw [Nat.cast_sub (by omega)]; simp
  set n : ℝ := (2 : ℝ) ^ l.logLen with hn_def
  have hn : n = (k : ℝ) * 2 ^ r := by
    rw [hn_def, Level.logLen, pow_add, hk_def]; push_cast; ring
  have h2r : (0 : ℝ) < 2 ^ r := by positivity
  set ρr : ℝ := (((k - 1 : ℕ) : ℝ)) / n with hρr_def
  set ρ : ℝ := l.rate with hρ_def
  have hρ : ρ = 1 / 2 ^ r := by
    rw [hρ_def, Level.rate, inv_pow, ← hr_def, one_div]
  have hρr_pos : 0 < ρr := by
    rw [hρr_def, hkm1, hn]; apply div_pos <;> [linarith; positivity]
  have hρr_le : ρr ≤ ρ := by
    rw [hρr_def, hkm1, hn, hρ, div_le_div_iff₀ (by positivity) (by positivity), one_mul]
    exact mul_le_mul_of_nonneg_right (by linarith) h2r.le
  have hρr_ge : 1 / 2 ^ (r + 1) ≤ ρr := by
    rw [hρr_def, hkm1, hn, div_le_div_iff₀ (by positivity) (by positivity), pow_succ, one_mul]
    have hk' : (k : ℝ) ≤ ((k : ℝ) - 1) * 2 := by linarith
    calc (k : ℝ) * 2 ^ r ≤ ((k : ℝ) - 1) * 2 * 2 ^ r := mul_le_mul_of_nonneg_right hk' h2r.le
      _ = ((k : ℝ) - 1) * (2 ^ r * 2) := by ring
  have hρ_le1 : ρ ≤ 1 := by rw [hρ]; exact div_le_one_of_le₀ (one_le_pow₀ (by norm_num)) h2r.le
  have hρr_le1 : ρr ≤ 1 := hρr_le.trans hρ_le1
  -- the square root pieces
  have hs : ρr ^ ((1 : ℝ) / 2) = Real.sqrt ρr := (Real.sqrt_eq_rpow ρr).symm
  have hs_le : Real.sqrt ρr ≤ Real.sqrt ρ := Real.sqrt_le_sqrt hρr_le
  have hs_le1 : Real.sqrt ρr ≤ 1 := Real.sqrt_le_one.2 hρr_le1
  have hsρ_le1 : Real.sqrt ρ ≤ 1 := Real.sqrt_le_one.2 hρ_le1
  have hs_ge : ρr ≤ Real.sqrt ρr := by
    rw [← hs]
    simpa using Real.rpow_le_rpow_of_exponent_ge hρr_pos hρr_le1 (by norm_num : (1 : ℝ) / 2 ≤ 1)
  have hs32 : ρr ^ 2 ≤ ρr ^ ((3 : ℝ) / 2) := by
    have := Real.rpow_le_rpow_of_exponent_ge hρr_pos hρr_le1 (by norm_num : (3 : ℝ) / 2 ≤ 2)
    simpa [Real.rpow_natCast] using this
  have h32pos : 0 < ρr ^ ((3 : ℝ) / 2) := Real.rpow_pos_of_pos hρr_pos _
  have hsq_pos : 0 < Real.sqrt ρr := Real.sqrt_pos.2 hρr_pos
  -- the radius and the multiplicity parameter
  set δ : ℝ := l.radius with hδ_def
  have hδ : δ = 1 - Real.sqrt ρ - 1 / 200 := by rw [hδ_def, Level.radius, eta]
  have hden : 1 / 200 ≤ 1 - Real.sqrt ρr - δ := by rw [hδ]; linarith
  have hx : Real.sqrt ρr / (1 - Real.sqrt ρr - δ) ≤ 200 := by
    rw [div_le_iff₀ (by linarith)]; linarith
  set mm : ℝ := max (⌈ρr ^ ((1 : ℝ) / 2) / (1 - ρr ^ ((1 : ℝ) / 2) - δ)⌉ : ℝ) 3 with hmm_def
  have hceil : (⌈ρr ^ ((1 : ℝ) / 2) / (1 - ρr ^ ((1 : ℝ) / 2) - δ)⌉ : ℝ) ≤ 200 := by
    rw [hs]
    have : ⌈Real.sqrt ρr / (1 - Real.sqrt ρr - δ)⌉ ≤ 200 := Int.ceil_le.2 (by exact_mod_cast hx)
    exact_mod_cast this
  have hmm_le : mm ≤ 200 := max_le hceil (by norm_num)
  have hmm_ge : 3 ≤ mm := le_max_right _ _
  have hδ_le1 : δ ≤ 1 := by rw [hδ]; linarith [Real.sqrt_nonneg ρ]
  have hδ_ge : -1 ≤ δ := by rw [hδ]; linarith
  -- the numerator
  have hδρ : δ * ρr ≤ 1 := by
    have := mul_le_mul_of_nonneg_right hδ_le1 hρr_pos.le
    linarith
  have hδρ' : -1 ≤ δ * ρr := by
    have := mul_le_mul_of_nonneg_right hδ_ge hρr_pos.le
    linarith
  have hnum : 2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr ≤ 2 * (401 / 2) ^ 5 + 3 * (401 / 2) := by
    have h5 : (mm + 1 / 2) ^ 5 ≤ (401 / 2 : ℝ) ^ 5 := by gcongr; linarith
    have : 3 * (mm + 1 / 2) * δ * ρr ≤ 3 * (401 / 2) := by
      have hpos : 0 ≤ 3 * (mm + 1 / 2) := by linarith
      calc 3 * (mm + 1 / 2) * δ * ρr = 3 * (mm + 1 / 2) * (δ * ρr) := by ring
        _ ≤ 3 * (mm + 1 / 2) * 1 := mul_le_mul_of_nonneg_left hδρ hpos
        _ ≤ 3 * (401 / 2) := by linarith
    linarith
  have hnum_nonneg : 0 ≤ 2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr := by
    have hm : (7 / 2 : ℝ) ≤ mm + 1 / 2 := by linarith
    have h4 : (7 / 2 : ℝ) ^ 4 ≤ (mm + 1 / 2) ^ 4 := by gcongr
    have : 2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr
        = 2 * ((mm + 1 / 2) * (mm + 1 / 2) ^ 4) + 3 * ((mm + 1 / 2) * (δ * ρr)) := by ring
    rw [this]
    have hpos : 0 ≤ mm + 1 / 2 := by linarith
    have e1 := mul_le_mul_of_nonneg_left h4 hpos
    have e2 := mul_le_mul_of_nonneg_left hδρ' hpos
    norm_num at e1 e2 ⊢
    linarith
  have hnum0 : (0 : ℝ) ≤ 2 * (401 / 2) ^ 5 + 3 * (401 / 2) := by norm_num
  -- the two terms of `a`
  have hρr2 : 1 / 2 ^ (2 * r + 2) ≤ ρr ^ 2 := by
    calc 1 / 2 ^ (2 * r + 2) = (1 / 2 ^ (r + 1) : ℝ) ^ 2 := by rw [div_pow, ← pow_mul]; ring_nf
      _ ≤ ρr ^ 2 := by gcongr
  have hT1 : (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr) / (3 * ρr ^ ((3 : ℝ) / 2)) * n
      ≤ (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) / 3 * 2 ^ (3 * r + 2 + c) := by
    have hD : 3 * (1 / 2 ^ (2 * r + 2)) ≤ 3 * ρr ^ ((3 : ℝ) / 2) := by linarith
    calc (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr) / (3 * ρr ^ ((3 : ℝ) / 2)) * n
        ≤ (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) / (3 * (1 / 2 ^ (2 * r + 2))) * n := by
          have := div_le_div₀ hnum0 hnum (by positivity) hD
          gcongr
      _ = (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) / 3 * 2 ^ (3 * r + 2 + c) := by
          rw [hn, hk_def]; push_cast
          field_simp
          ring
  have hT2 : (mm + 1 / 2) / ρr ^ ((1 : ℝ) / 2) ≤ 401 / 2 * 2 ^ (r + 1) := by
    rw [hs]
    calc (mm + 1 / 2) / Real.sqrt ρr ≤ (401 / 2) / Real.sqrt ρr := by gcongr; linarith
      _ ≤ (401 / 2) / (1 / 2 ^ (r + 1)) :=
          div_le_div_of_nonneg_left (by norm_num) (by positivity) (hρr_ge.trans hs_ge)
      _ = 401 / 2 * 2 ^ (r + 1) := by field_simp
  have hT1_nonneg : 0 ≤ (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr)
      / (3 * ρr ^ ((3 : ℝ) / 2)) * n :=
    mul_nonneg (div_nonneg hnum_nonneg (by positivity)) (by positivity)
  have hT2_nonneg : 0 ≤ (mm + 1 / 2) / ρr ^ ((1 : ℝ) / 2) := by
    have : 0 ≤ mm + 1 / 2 := by linarith
    positivity
  -- assemble
  have hA : l.mcaA = (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr) / (3 * ρr ^ ((3 : ℝ) / 2)) * n
      + (mm + 1 / 2) / ρr ^ ((1 : ℝ) / 2) := rfl
  have hup : ((mcaUp l : ℚ) : ℝ) =
      (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) / 3 * 2 ^ (3 * r + 2 + c) + 401 / 2 * 2 ^ (r + 1) := by
    simp only [mcaUp, ← hr_def, ← hc_def]; push_cast; ring
  rw [hA, hup]
  exact ⟨by linarith, by linarith⟩

theorem mcaA_le (l : Level) (hr : l.logInvRate ≤ 7) (hc : 1 ≤ l.logCols) :
    l.mcaA ≤ (mcaUp l : ℝ) := (mcaA_bounds l hr hc).2

theorem foldError_le (l : Level) (L : ℕ) (hr : l.logInvRate ≤ 7) (hc : 1 ≤ l.logCols) :
    l.foldError L ≤ (foldUp l L : ℝ) := by
  unfold Level.foldError foldUp qK
  push_cast
  have h := mcaA_le l hr hc
  have hl : (0 : ℝ) ≤ 2 ^ l.lanes - 1 := by
    have : (1 : ℝ) ≤ 2 ^ l.lanes := one_le_pow₀ (by norm_num)
    linarith
  gcongr

theorem pairs_eq (L : ℕ) : pairs L = (pairsUp L : ℝ) := by
  unfold pairs pairsUp; push_cast; ring

theorem midLevelError_le (l l' : Level) (hl : l.Ok) :
    midLevelError l l' ≤ (midUp l l' : ℝ) := by
  have hf := foldError_le l (max l.listBound l'.listBound) hl.2.1 hl.2.2
  have hq := queryError_le l hl.2.1
  unfold midLevelError midUp qF
  rw [pairs_eq]
  push_cast
  linarith

theorem lastLevelError_le (l : Level) (hl : l.Ok) :
    lastLevelError l ≤ (lastUp l : ℝ) := by
  have hf := foldError_le l l.listBound hl.2.1 hl.2.2
  have hq := queryError_le l hl.2.1
  unfold lastLevelError lastUp qF
  push_cast
  linarith

theorem ligeritoError_le : ∀ ls : List Level, (∀ l ∈ ls, l.Ok) →
    ligeritoError ls ≤ (ligUp ls : ℝ)
  | [], _ => by simp [ligeritoError, ligUp]
  | [l], h => by
      simp only [ligeritoError, ligUp]
      exact lastLevelError_le l (h l (by simp))
  | l :: l' :: rest, h => by
      simp only [ligeritoError, ligUp]
      push_cast
      have h1 := midLevelError_le l l' (h l (by simp))
      have h2 := ligeritoError_le (l' :: rest) fun x hx => h x (List.mem_cons_of_mem _ hx)
      linarith

/-! ## The statement ranges -/

theorem piopError_le (m L₀ : ℕ) (st : Stmt) (hst : st.InRange) :
    piopError m st.kLog st.nClaims L₀ ≤ ((piopNumMax m L₀ : ℚ) / 2 ^ 128 : ℚ) := by
  obtain ⟨hk, hR, _⟩ := hst
  unfold piopError qF
  push_cast
  gcongr
  have h1 : m - st.kLog ≤ m := Nat.sub_le _ _
  have h2 : st.kLog - 6 ≤ 21 := by omega
  have h3 : st.nClaims ≤ 2050 := by unfold Stmt.nClaims; omega
  unfold piopNumMax
  exact_mod_cast (by gcongr : (L₀ + 1) * (m - 13) + L₀ * 127 + L₀ * (2 * (m - 6))
      + L₀ * (m - st.kLog) + L₀ * (2 + 2 * (st.kLog - 6) + 63) + L₀ * (7 * st.nClaims + 1) + L₀
      ≤ (L₀ + 1) * (m - 13) + L₀ * 127 + L₀ * (2 * (m - 6)) + L₀ * m
      + L₀ * (2 + 2 * 21 + 63) + L₀ * (7 * 2050 + 1) + L₀)

theorem repError_le (s : Schedule) (hs : ∀ l ∈ s.levels, l.Ok) (st : Stmt) (hst : st.InRange) :
    repError s st ≤ (repUp s : ℝ) := by
  unfold repError repUp
  push_cast
  have h1 := piopError_le s.m s.listBound₀ st hst
  have h2 := ligeritoError_le s.levels hs
  push_cast at h1
  linarith

theorem pairs_nonneg (L : ℕ) : 0 ≤ pairs L := by
  unfold pairs
  rcases Nat.eq_zero_or_pos L with h | h
  · simp [h]
  · have : (1 : ℝ) ≤ L := by exact_mod_cast h
    have : 0 ≤ (L : ℝ) - 1 := by linarith
    positivity

theorem foldError_nonneg (l : Level) (L : ℕ) (hl : l.Ok) : 0 ≤ l.foldError L := by
  unfold Level.foldError qK
  have hm := (mcaA_bounds l hl.2.1 hl.2.2).1
  have hl2 : (0 : ℝ) ≤ 2 ^ l.lanes - 1 := by
    have : (1 : ℝ) ≤ 2 ^ l.lanes := one_le_pow₀ (by norm_num)
    linarith
  positivity

theorem queryError_nonneg (l : Level) : 0 ≤ l.queryError := by
  unfold Level.queryError eta; positivity

theorem ligeritoError_nonneg : ∀ ls : List Level, (∀ l ∈ ls, l.Ok) → 0 ≤ ligeritoError ls
  | [], _ => by simp [ligeritoError]
  | [l], h => by
      simp only [ligeritoError, lastLevelError, qF]
      have := foldError_nonneg l l.listBound (h l (by simp))
      have := queryError_nonneg l
      positivity
  | l :: l' :: rest, h => by
      simp only [ligeritoError, midLevelError, qF]
      have := foldError_nonneg l (max l.listBound l'.listBound) (h l (by simp))
      have := queryError_nonneg l
      have := pairs_nonneg l'.listBound
      have := ligeritoError_nonneg (l' :: rest) fun x hx => h x (List.mem_cons_of_mem _ hx)
      positivity

theorem repError_nonneg (s : Schedule) (st : Stmt) (hs : ∀ l ∈ s.levels, l.Ok) :
    0 ≤ repError s st := by
  unfold repError piopError qF
  have := ligeritoError_nonneg s.levels hs
  positivity

theorem linkError_le (s : Schedule) (st : Stmt) (hst : st.InRange) :
    linkError s st ≤ (linkUp s : ℝ) := by
  obtain ⟨_, hR, hp⟩ := hst
  unfold linkError linkUp qF
  push_cast
  have hR' : (st.nRegions : ℝ) ≤ 1024 := by exact_mod_cast hR
  have hp' : (st.mPts : ℝ) ≤ 64 := by exact_mod_cast hp
  gcongr

theorem tableError_le_up (s : Schedule) (hs : ∀ l ∈ s.levels, l.Ok) (st : Stmt)
    (hst : st.InRange) :
    tableError s st ≤ ((linkUp s + repUp s ^ 2 : ℚ) : ℝ) := by
  unfold tableError
  push_cast
  have h1 := linkError_le s st hst
  have h2 := repError_le s hs st hst
  have h0 := repError_nonneg s st hs
  nlinarith

end FlockSoundness.Accounting
