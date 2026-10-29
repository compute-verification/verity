import Proofs.Flock.Soundness.Audit.Closure

/-!
# Windows of calls

A window of calls is one population and one law. Its strata are the pairs (call, template), `call s` being stratum `s`'s
call, and `Law.stratified` draws a uniform `k_s`-subset of every stratum independently: the product of the calls' draws.

Two modelling lines sit outside the statements:
* M1: each call is drawn on its own coins, the window's key derived under a **distinct context** for every call. Two calls
  that share a context draw the same subsets, and the window's draw is then not this product law.
* M2: the window's audit is one `Analysis` over the window's session, with the window's `ε_ks` and `δ_link`. How those
  compose from the calls' is not stated here.

`audit_window_of_le` takes any law whose escapes are at most the product law's, such as an executable draw proved against
it, so the ideal law here need not be the one run. `audit_window_of_le_slack` allows those escapes an additive `η`, which
a keyed draw's pseudorandomness assumption supplies, and adds `η` to the bound; at `η = 0` it is `audit_window_of_le`.
`extraction_audit_window_of_le_slack` and `extraction_audit_window_split_of_record_of_le_slack` are the same at the
compiled layer.
`audit_window_split_of_record_of_le_slack` is its record form for the per-call split with y's floors.

Two weightings of the same draw: `w`, the work (each tile's W_ref on the tile strata, 0 elsewhere), and `v`, the y cells
(1 on the dequantization strata, 0 elsewhere).

* `Covers σ k w K`: every stratum draws at least its share `K · w_s · n_s / W`, or all its units.
  `stratified_escape_le_of_covers`, **the work bound for any law whose sizes cover the weights**:
  `escape B ≤ ((W − W_B)/W)^K`. `covers_work`: the work law covers its own work.
* `windowK`, **the per-call split**: call `c` draws by its own work law at its share
  `K_c = min(N_c, max(1, ⌈K · W_c / W⌉))` of the window's `K`. `covers_window`: that covers the window's work when each
  call's work is in one stratum (`hone`). `covers_of_floor`, **y's floors**: sizes that draw at
  least `min(n_s, f_s)`, with `K_y · v_s · n_s ≤ f_s · N`, cover `v` at `K_y`.
* `audit_window`, `extraction_audit_window`: the audit accepts while the unsound work is at least `T` or the unsound
  weight (`unsoundWork` under `v`) at least `T_y` with probability at most `max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + ε_ks + δ_link`. The larger,
  not the sum: both bound the escape of the one wrong set the prover committed before the draw (`audit_profile`).
  `audit_window_of_record`: at `K = K_y = 27,713`, a 0.1% share of either with at most `2^-40 + ε_ks + δ_link`, and
  `audit_window_split_of_record` for the per-call split with y's floors, the law `PlanDraw` runs.

**`hone` is needed, and not only by this route.** When a call's share `⌈K · W_c / W⌉` is capped at its units `N_c`, a
second stratum doing work in the call can draw well below its share, and then the window bound itself fails. With call 0
of 1,000 units at work 2 and 10 at work 1,000, call 1 of 100,000 units at work 0 and 1 at work 1, and `K = 5,000`, call
0's share caps at `K_0 = 1,010`, its work-2 stratum draws 169 of its 1,000 units, and ten wrong units there escape with
0.156 against the bound's 2.4·10⁻⁴. Uncapped it would draw 834. A call with several work strata must have a cap that
does not bind, or prove those strata whole.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m C : ℕ}

/-- **Sizes that cover the weights `w` at budget `K`**: every stratum draws at least its share `K · w_s · n_s / W` of
the weight, or all its units. -/
def Covers (σ : Fin n → Fin m) (k w : Fin m → ℕ) (K : ℕ) : Prop :=
  ∀ s, K * w s * (stratum σ s).card ≤ k s * totalWork σ w ∨ k s = (stratum σ s).card

/-- **The work bound for any stratified law whose sizes cover the weights.** -/
theorem stratified_escape_le_of_covers (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (w : Fin m → ℕ) (K : ℕ) (hW : 0 < totalWork σ w) (hc : Covers σ k w K) (B : Finset (Fin n)) :
    (stratified σ k hk).escape B ≤
      (((totalWork σ w - workOf σ w B : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  classical
  set W := totalWork σ w with hWdef
  set N : Fin m → ℕ := fun s => (stratum σ s).card with hNdef
  set M : Fin m → ℕ := fun s => (B.filter (σ · = s)).card with hMdef
  have hMN : ∀ s, M s ≤ N s := fun s => card_le_card (filter_subset_filter _ (subset_univ B))
  have hkN : ∀ s, k s ≤ N s := hk
  have hW0 : (W : ℝ≥0) ≠ 0 := by exact_mod_cast hW.ne'
  set x : Fin m → ℝ≥0 := fun s => ((N s - M s : ℕ) : ℝ≥0) / (N s : ℝ≥0) with hxdef
  set a : Fin m → ℝ≥0 := fun s => ((w s * N s : ℕ) : ℝ≥0) / (W : ℝ≥0) with hadef
  have hsumN : ∑ s, w s * N s = W := (totalWork_eq σ w).symm
  have ha1 : ∑ s, a s = 1 := by
    simp only [hadef]
    rw [← Finset.sum_div, ← Nat.cast_sum, hsumN, div_self hW0]
  have hterm : ∀ s, a s * x s = ((w s * (N s - M s) : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    intro s
    simp only [hadef, hxdef]
    rcases Nat.eq_zero_or_pos (N s) with h0 | hpos
    · have hM0 : M s = 0 := by have := hMN s; omega
      simp [h0, hM0]
    · have hN0 : (N s : ℝ≥0) ≠ 0 := by exact_mod_cast hpos.ne'
      rw [div_mul_div_comm, ← Nat.cast_mul, show w s * N s * (N s - M s) = w s * (N s - M s) * N s by ring,
        Nat.cast_mul, mul_div_mul_right _ _ hN0]
  have hax : ∑ s, a s * x s = ((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    rw [Finset.sum_congr rfl fun s _ => hterm s, ← Finset.sum_div, ← Nat.cast_sum]
    congr 2
    have hB : workOf σ w B = ∑ s, w s * M s := workOf_eq σ w B
    have hsplit : ∑ s, w s * (N s - M s) + ∑ s, w s * M s = ∑ s, w s * N s := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun s _ => by rw [← Nat.mul_add, Nat.sub_add_cancel (hMN s)]
    omega
  have hfac : ∀ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤ x s ^ ((a s : ℝ) * K) := by
    intro s
    rcases Nat.eq_zero_or_pos (N s) with h0 | hpos
    · have hk0 : k s = 0 := by have := hkN s; omega
      have hM0 : M s = 0 := by have := hMN s; omega
      simp [hadef, h0, hk0, hM0]
    have hC0 : ((N s).choose (k s) : ℝ≥0) ≠ 0 := by exact_mod_cast (Nat.choose_pos (hkN s)).ne'
    rcases Nat.eq_zero_or_pos (M s) with hM0 | hMpos
    · have hx1 : x s = 1 := by
        simp only [hxdef, hM0, Nat.sub_zero]
        exact div_self (by exact_mod_cast hpos.ne')
      rw [hx1, NNReal.one_rpow, hM0, Nat.sub_zero, div_self hC0]
    -- a stratum that draws nothing has no share to cover: its factor and the bound's are both 1
    rcases Nat.eq_zero_or_pos (k s) with hk0 | hk1
    · have hKw : K * w s * N s = 0 := by
        have h' : K * w s * N s ≤ k s * W ∨ k s = N s := hc s
        rcases h' with h | h
        · rw [hk0, zero_mul] at h; omega
        · omega
      have haK : (a s : ℝ) * K = 0 := by
        rcases Nat.mul_eq_zero.1 hKw with h | h
        · rcases Nat.mul_eq_zero.1 h with h | h
          · simp [h]
          · simp [hadef, h]
        · simp [hadef, h]
      rw [haK, NNReal.rpow_zero, hk0, Nat.choose_zero_right, Nat.choose_zero_right, Nat.cast_one, div_one]
    have hMNs := hMN s
    by_cases hzero : k s = N s ∨ M s = N s
    · have : (N s - M s).choose (k s) = 0 := Nat.choose_eq_zero_of_lt (by omega)
      rw [this, Nat.cast_zero, zero_div]
      exact zero_le
    push Not at hzero
    have hge : K * w s * N s ≤ k s * W := (hc s).resolve_right hzero.1
    have hx0 : 0 < x s := div_pos (by exact_mod_cast (show 0 < N s - M s by omega)) (by exact_mod_cast hpos)
    have hx1 : x s ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast Nat.sub_le _ _) zero_le
    have hexp : (a s : ℝ) * K ≤ (k s : ℝ) := by
      simp only [hadef, NNReal.coe_div, NNReal.coe_natCast]
      rw [div_mul_eq_mul_div, div_le_iff₀ (by exact_mod_cast hW)]
      exact_mod_cast (show w s * N s * K ≤ k s * W from
        calc w s * N s * K = K * w s * N s := by ring
          _ ≤ k s * W := hge)
    calc ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤ x s ^ (k s) :=
          choose_div_le_pow (hkN s) hpos
      _ = x s ^ ((k s : ℕ) : ℝ) := (NNReal.rpow_natCast _ _).symm
      _ ≤ x s ^ ((a s : ℝ) * K) := NNReal.rpow_le_rpow_of_exponent_ge hx0 hx1 hexp
  have hprod : ∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤
      (((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K := by
    calc _ ≤ ∏ s, x s ^ ((a s : ℝ) * K) := Finset.prod_le_prod fun s _ => hfac s
      _ = (∏ s, x s ^ (a s : ℝ)) ^ K := by
          rw [← Finset.prod_pow]
          exact Finset.prod_congr rfl fun s _ => by rw [NNReal.rpow_mul, NNReal.rpow_natCast]
      _ ≤ (∑ s, a s * x s) ^ K :=
          pow_le_pow_left₀ zero_le (NNReal.geom_mean_le_arith_mean_weighted univ a x ha1) K
      _ = _ := by rw [hax]
  have hC : ∀ s, ((N s).choose (k s) : ℝ≥0) ≠ 0 := fun s => by exact_mod_cast (Nat.choose_pos (hkN s)).ne'
  rw [stratified_escape]
  calc ∏ s, (((stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ≥0∞) /
        ((stratum σ s).card.choose (k s) : ℝ≥0∞)
      = ((∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) := by
        rw [ENNReal.ofNNReal_finsetProd]
        refine Finset.prod_congr rfl fun s _ => ?_
        rw [ENNReal.coe_div (hC s)]
        simp [hNdef, hMdef]
    _ ≤ (((((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K : ℝ≥0) : ℝ≥0∞) := ENNReal.coe_le_coe.2 hprod
    _ = _ := by rw [ENNReal.coe_pow, ENNReal.coe_div hW0, ENNReal.coe_natCast, ENNReal.coe_natCast]

/-- The work law covers its own work (`workK_ge`). -/
theorem covers_work (σ : Fin n → Fin m) (w f : Fin m → ℕ) (K : ℕ) (hW : 0 < totalWork σ w) :
    Covers σ (workK σ w f K) w K := fun s =>
  (workK_ge s hW).symm.imp (fun h => h.trans_eq (mul_comm _ _)) id

/-- Call `c`'s work `W_c`, its strata's. -/
def callWork (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), w s * (stratum σ s).card

/-- Call `c`'s units `N_c`. -/
def callUnits (σ : Fin n → Fin m) (call : Fin m → Fin C) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), (stratum σ s).card

/-- Call `c`'s budget, its share of the window's `K`: `K_c = min(N_c, max(1, ⌈K · W_c / W⌉))`. -/
def callK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (K : ℕ) (c : Fin C) : ℕ :=
  min (callUnits σ call c) (max 1 ((K * callWork σ call w c + totalWork σ w - 1) / totalWork σ w))

/-- The window's sizes: each call's work law at its own budget, `k_s = min(n_s, max(f_s, ⌈K_c · w_s · n_s / W_c⌉))`. -/
def windowK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule (callK σ call w K (call s)) (w s) (stratum σ s).card (callWork σ call w (call s)) (f s)

theorem windowK_le (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ) (s : Fin m) :
    windowK σ call w f K s ≤ (stratum σ s).card := min_le_left _ _

theorem card_le_callUnits (σ : Fin n → Fin m) (call : Fin m → Fin C) (s : Fin m) :
    (stratum σ s).card ≤ callUnits σ call (call s) :=
  Finset.single_le_sum (f := fun s => (stratum σ s).card) (fun _ _ => Nat.zero_le _)
    (mem_filter.2 ⟨mem_univ s, rfl⟩)

/-- **The per-call split covers the window's work**, when each call's work is in one stratum. -/
theorem covers_window (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ)
    (hW : 0 < totalWork σ w) (hone : ∀ s, 0 < w s → callWork σ call w (call s) = w s * (stratum σ s).card) :
    Covers σ (windowK σ call w f K) w K := by
  intro s
  rcases Nat.eq_zero_or_pos (w s) with hw | hw
  · exact Or.inl (by simp [hw])
  rcases Nat.eq_zero_or_pos (stratum σ s).card with hn | hn
  · exact Or.inr (by have := windowK_le σ call w f K s; omega)
  set Kc := callK σ call w K (call s) with hKc
  have hWc := hone s hw
  have hpos : 0 < w s * (stratum σ s).card := Nat.mul_pos hw hn
  -- one work stratum: `W_c = w_s · n_s`, so the call's ceiling is its budget
  have hceil : (Kc * w s * (stratum σ s).card + callWork σ call w (call s) - 1) / callWork σ call w (call s) = Kc := by
    rw [hWc, show Kc * w s * (stratum σ s).card + w s * (stratum σ s).card - 1 =
        (w s * (stratum σ s).card - 1) + Kc * (w s * (stratum σ s).card) by rw [mul_assoc]; omega,
      Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  have hk : windowK σ call w f K s = min (stratum σ s).card (max (f s) Kc) := by
    unfold windowK workRule
    rw [hceil]
  rcases le_or_gt (stratum σ s).card (max (f s) Kc) with h | h
  · exact Or.inr (by rw [hk, min_eq_left h])
  refine Or.inl ?_
  rw [hk, min_eq_right h.le]
  -- below the call's units, the budget is its share `max 1 ⌈K · W_c / W⌉`
  have hlt : Kc < callUnits σ call (call s) :=
    lt_of_lt_of_le (lt_of_le_of_lt (le_max_right _ _) h) (card_le_callUnits σ call s)
  have hKc' : Kc = max 1 ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w) := by
    have e : Kc = min (callUnits σ call (call s))
        (max 1 ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w)) := by rw [hKc, callK]
    rcases min_choice (callUnits σ call (call s))
        (max 1 ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w)) with h' | h'
    · rw [e, h'] at hlt; exact absurd hlt (lt_irrefl _)
    · rw [e, h']
  calc K * w s * (stratum σ s).card = K * callWork σ call w (call s) := by rw [hWc, mul_assoc]
    _ ≤ totalWork σ w * ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w) :=
        le_mul_ceil_div _ _ hW
    _ ≤ totalWork σ w * Kc := Nat.mul_le_mul_left _ (by rw [hKc']; exact le_max_right _ _)
    _ ≤ totalWork σ w * max (f s) Kc := Nat.mul_le_mul_left _ (le_max_right _ _)
    _ = max (f s) Kc * totalWork σ w := mul_comm _ _

/-- **Floors cover the weights `v` at `K_y`**, for any sizes that draw at least `min(n_s, f_s)`. -/
theorem covers_of_floor (σ : Fin n → Fin m) (k f v : Fin m → ℕ) (Ky : ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (hkf : ∀ s, min (stratum σ s).card (f s) ≤ k s)
    (hfy : ∀ s, Ky * v s * (stratum σ s).card ≤ f s * totalWork σ v) :
    Covers σ k v Ky := by
  intro s
  rcases le_total (stratum σ s).card (f s) with h | h
  · have := hkf s
    rw [min_eq_left h] at this
    exact Or.inr (le_antisymm (hk s) this)
  · have hfk := hkf s
    rw [min_eq_right h] at hfk
    exact Or.inl ((hfy s).trans (Nat.mul_le_mul_right _ hfk))

/-- One wrong set against both weightings: under the closure draw of any law whose escapes are at most the stratified
law's, its escape is at most the larger bound. -/
theorem closure_escape_le_window {L : Law n} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (hL : ∀ B, L.escape B ≤ (stratified σ k hk).escape B) (cl : Fin n → Finset (Fin n)) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < totalWork σ w) (hc : Covers σ k w K) (hN : 0 < totalWork σ v) (hcy : Covers σ k v Ky) {T Ty : ℕ}
    {B : Finset (Fin n)} (hB : T ≤ unsoundWork σ w cl B ∨ Ty ≤ unsoundWork σ v cl B) :
    (L.closure cl).escape B ≤
      max ((((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K)
        ((((totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (totalWork σ v : ℝ≥0∞)) ^ Ky) := by
  rw [closure_escape]
  refine (hL _).trans ?_
  rcases hB with hB | hB
  · refine le_max_of_le_left ((stratified_escape_le_of_covers σ k hk w K hW hc _).trans ?_)
    gcongr
  · refine le_max_of_le_right ((stratified_escape_le_of_covers σ k hk v Ky hN hcy _).trans ?_)
    gcongr

/-- `closure_escape_le_window` for a law whose escapes exceed the stratified law's by at most `η`. -/
theorem closure_escape_le_window_slack {L : Law n} (σ : Fin n → Fin m) (k : Fin m → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) {η : ℝ≥0∞} (hL : ∀ B, L.escape B ≤ (stratified σ k hk).escape B + η)
    (cl : Fin n → Finset (Fin n)) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < totalWork σ w) (hc : Covers σ k w K)
    (hN : 0 < totalWork σ v) (hcy : Covers σ k v Ky) {T Ty : ℕ} {B : Finset (Fin n)}
    (hB : T ≤ unsoundWork σ w cl B ∨ Ty ≤ unsoundWork σ v cl B) :
    (L.closure cl).escape B ≤
      max ((((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K)
        ((((totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (totalWork σ v : ℝ≥0∞)) ^ Ky) + η := by
  have h := closure_escape_le_window σ k hk (fun _ => le_rfl) cl hW hc hN hcy hB
  rw [closure_escape] at h ⊢
  exact (hL _).trans (add_le_add h le_rfl)

/-- A 0.1% share at `K = 27,713` escapes with at most `2^-40`. -/
theorem pow_record_le {W U : ℕ} (hW : 0 < W) (hU : W ≤ 1000 * U) :
    (((W - U : ℕ) : ℝ≥0∞) / (W : ℝ≥0∞)) ^ 27713 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ := by
  refine le_trans (pow_le_pow_left₀ zero_le ?_ 27713) record_sizing
  have hWt : (W : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hW0 : (W : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW.ne'
  rw [ENNReal.div_le_iff hW0 hWt, ← ENNReal.mul_div_right_comm, ENNReal.le_div_iff_mul_le (Or.inl (by norm_num))
    (Or.inl (by norm_num))]
  exact_mod_cast (show (W - U) * 1000 ≤ 999 * W by omega)

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (Law.stratum σ s).card}
  {cl : Fin n → Finset (Fin n)} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- **The window audit** (oracle layer): the audit accepts while the unsound work is at least `T` or the unsound weight at
least `T_y` with probability at most `max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + ε_ks + δ_link`. -/
theorem audit_window (A : Analysis ((Law.stratified σ k hk).closure cl) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K) (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (audit ((Law.stratified σ k hk).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit ((Law.stratified σ k hk).closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => T ≤ Law.unsoundWork σ w cl B ∨ Ty ≤ Law.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.closure_escape_le_window σ k hk (fun _ => le_rfl) cl hW hc hN hcy hB

/-- **The window audit for any law below the product law** (oracle layer): `audit_window` for a law `L` whose escapes are
at most the stratified law's, drawn with closures, such as an executable draw proved against it. -/
theorem audit_window_of_le {L : Law n} (hL : ∀ B, L.escape B ≤ (Law.stratified σ k hk).escape B)
    (A : Analysis (L.closure cl) Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K)
    (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => T ≤ Law.unsoundWork σ w cl B ∨ Ty ≤ Law.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.closure_escape_le_window σ k hk hL cl hW hc hN hcy hB

/-- **The window audit** (compiled layer, randomized extractor): at most
`max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_window (A : ExtractionAnalysis ((Law.stratified σ k hk).closure cl) Reg session)
    {εks δlink : (R : Reg) → Cont ((Law.stratified σ k hk).closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K) (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (audit ((Law.stratified σ k hk).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit ((Law.stratified σ k hk).closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ Law.unsoundWork σ w cl (A.wrong X) ∨
    Ty ≤ Law.unsoundWork σ v cl (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
      Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (Law.closure_escape_le_window σ k hk (fun _ => le_rfl) cl hW hc hN hcy hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **The window audit at the sizing of record**: with `K = K_y = 27,713`, the audit accepts while the unsound work is at
least 0.1% of the window's, or the unsound weight at least 0.1% of `N`, with probability at most `2^-40 + ε_ks + δ_link`. -/
theorem audit_window_of_record (A : Analysis ((Law.stratified σ k hk).closure cl) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) {w v : Fin m → ℕ}
    (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w 27713) (hN : 0 < Law.totalWork σ v)
    (hcy : Law.Covers σ k v 27713) (σ' : Strategy (audit ((Law.stratified σ k hk).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit ((Law.stratified σ k hk).closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl B ∨
    Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  refine iSup₂_le fun B hB => ?_
  rw [Law.closure_escape]
  rcases hB with hB | hB
  · exact (Law.stratified_escape_le_of_covers σ k hk w 27713 hW hc _).trans (Law.pow_record_le hW hB)
  · exact (Law.stratified_escape_le_of_covers σ k hk v 27713 hN hcy _).trans (Law.pow_record_le hN hB)

/-- **The per-call split with y's floors at the sizing of record**: every call drawn by
its own work law at its share of `K = 27,713`, each stratum's floor covering `v` at `K_y = 27,713`, with closures. -/
theorem audit_window_split_of_record {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ}
    (A : Analysis ((Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).closure cl)
      Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink)
    (hW : 0 < Law.totalWork σ w)
    (hone : ∀ s, 0 < w s → Law.callWork σ call w (call s) = w s * (Law.stratum σ s).card)
    (hN : 0 < Law.totalWork σ v) (hfy : ∀ s, 27713 * v s * (Law.stratum σ s).card ≤ f s * Law.totalWork σ v)
    (σ' : Strategy (audit ((Law.stratified σ (Law.windowK σ call w f 27713)
      (Law.windowK_le σ call w f 27713)).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit ((Law.stratified σ (Law.windowK σ call w f 27713)
          (Law.windowK_le σ call w f 27713)).closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + δlink :=
  audit_window_of_record A hks hlink hW (Law.covers_window σ call w f 27713 hW hone) hN
    (Law.covers_of_floor σ _ f v 27713 (Law.windowK_le σ call w f 27713)
      (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy) σ'

/-- **The window audit for a law within `η` of the product law** (oracle layer): `audit_window_of_le` for a law `L` whose
escapes are at most the stratified law's plus `η`, drawn with closures, such as a keyed draw under a pseudorandomness
assumption. The bound gains `η`; at `η = 0` it is `audit_window_of_le`. -/
theorem audit_window_of_le_slack {L : Law n} {η : ℝ≥0∞} (hL : ∀ B, L.escape B ≤ (Law.stratified σ k hk).escape B + η)
    (A : Analysis (L.closure cl) Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K)
    (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + η + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => T ≤ Law.unsoundWork σ w cl B ∨ Ty ≤ Law.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.closure_escape_le_window_slack σ k hk hL cl hW hc hN hcy hB

/-- At `η = 0`, `audit_window_of_le_slack` is `audit_window_of_le`. -/
example {L : Law n} (hL : ∀ B, L.escape B ≤ (Law.stratified σ k hk).escape B)
    (A : Analysis (L.closure cl) Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K)
    (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + εks + δlink := by
  simpa only [add_zero] using
    audit_window_of_le_slack (η := 0) (fun B => by rw [add_zero]; exact hL B) A hks hlink hW hc hN hcy T Ty σ'

/-- **The window audit for a law within `η` of the product law** (compiled layer, randomized extractor):
`extraction_audit_window` for a law `L` whose escapes are at most the stratified law's plus `η`, drawn with closures. At
most `max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + η + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_window_of_le_slack {L : Law n} {η : ℝ≥0∞}
    (hL : ∀ B, L.escape B ≤ (Law.stratified σ k hk).escape B + η) (A : ExtractionAnalysis (L.closure cl) Reg session)
    {εks δlink : (R : Reg) → Cont (L.closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K) (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + η +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ Law.unsoundWork σ w cl (A.wrong X) ∨
    Ty ≤ Law.unsoundWork σ v cl (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
      Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (Law.closure_escape_le_window_slack σ k hk hL cl hW hc hN hcy hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **The per-call split with y's floors, for a law within `η` of it, at the sizing of record** (compiled layer): at most
`2^-40 + η + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_window_split_of_record_of_le_slack {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ}
    {L : Law n} {η : ℝ≥0∞}
    (hL : ∀ B, L.escape B ≤
      (Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).escape B + η)
    (A : ExtractionAnalysis (L.closure cl) Reg session)
    {εks δlink : (R : Reg) → Cont (L.closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hW : 0 < Law.totalWork σ w)
    (hone : ∀ s, 0 < w s → Law.callWork σ call w (call s) = w s * (Law.stratum σ s).card)
    (hN : 0 < Law.totalWork σ v) (hfy : ∀ s, 27713 * v s * (Law.stratum σ s).card ≤ f s * Law.totalWork σ v)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  have hc := Law.covers_window σ call w f 27713 hW hone
  have hcy := Law.covers_of_floor σ _ f v 27713 (Law.windowK_le σ call w f 27713)
    (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy
  refine (extraction_audit_le A hks hlink (fun X _ => Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong X) ∨
    Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
      Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))
  · refine (prCoin_mono fun ω h => h.2).trans ?_
    change (L.closure cl).escape _ ≤ _
    rw [Law.closure_escape]
    refine (hL _).trans (add_le_add ?_ le_rfl)
    rcases hB with hB | hB
    · exact (Law.stratified_escape_le_of_covers σ _ _ w 27713 hW hc _).trans (Law.pow_record_le hW hB)
    · exact (Law.stratified_escape_le_of_covers σ _ _ v 27713 hN hcy _).trans (Law.pow_record_le hN hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **The per-call split with y's floors, for a law within `η` of it, at the sizing of record**: at most
`2^-40 + η + ε_ks + δ_link`. -/
theorem audit_window_split_of_record_of_le_slack {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ} {L : Law n}
    {η : ℝ≥0∞}
    (hL : ∀ B, L.escape B ≤
      (Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).escape B + η)
    (A : Analysis (L.closure cl) Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) (hW : 0 < Law.totalWork σ w)
    (hone : ∀ s, 0 < w s → Law.callWork σ call w (call s) = w s * (Law.stratum σ s).card)
    (hN : 0 < Law.totalWork σ v) (hfy : ∀ s, 27713 * v s * (Law.stratum σ s).card ≤ f s * Law.totalWork σ v)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks + δlink := by
  have hc := Law.covers_window σ call w f 27713 hW hone
  have hcy := Law.covers_of_floor σ _ f v 27713 (Law.windowK_le σ call w f 27713)
    (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy
  refine (audit_profile A hks hlink (fun B => Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl B ∨
    Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  refine iSup₂_le fun B hB => ?_
  rw [Law.closure_escape]
  refine (hL _).trans (add_le_add ?_ le_rfl)
  rcases hB with hB | hB
  · exact (Law.stratified_escape_le_of_covers σ _ _ w 27713 hW hc _).trans (Law.pow_record_le hW hB)
  · exact (Law.stratified_escape_le_of_covers σ _ _ v 27713 hN hcy _).trans (Law.pow_record_le hN hB)

end Audits

end FlockSoundness.Audit
