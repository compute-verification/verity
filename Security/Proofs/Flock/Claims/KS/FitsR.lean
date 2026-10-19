import Proofs.Flock.Claims.KS.Joint

/-!
# Fork events counted at their own round

`Escapes.Fits` charges every fork event at the busiest round's count: `mult · Σ_i c_i ≤ 2C`. The collapse needs only
each event's own round's count `m_i`: by Cauchy–Schwarz at weights `m_i`,
`Σ_i √(c_i a_i) ≤ √((k · Σ_i m_i c_i) · ((Σ_r f r)/k))` (`collapseR`), since the events at a round of `m` events share
its `f r` in `m` parts. So escapes with `Σ_i m_i c_i ≤ 2C` and `Σ_l β_l ≤ k(x)` (`Escapes.FitsR`) give `Escapes.le`'s
root, `√(16·(√(2·k(x)·C) + k(x))²·Pr[B])` (`Escapes.le_of_fitsR`), and `Fits` gives `FitsR` (`Escapes.Fits.fitsR`).

With the conflict fork of weight `c` at round `0` (`Escapes.addFork`): `Escapes.FitsForkR` counts it at round `0`,
`FitsFork` gives it (`Escapes.FitsFork.fitsForkR`), and it gives `FitsR` with the extra event
(`Escapes.fitsR_addFork`). Escapes with one fork event at every round fit `C` with the extra one when
`2·c + c_0 + Σ_r c_r ≤ 2C` (`Escapes.fitsForkR_of_round`), where `FitsFork` asks for `2·c + 2·Σ_r c_r ≤ 2C`: only
round `0`'s events are charged twice.
-/

namespace Claims

/-- **The collapse, each finder counted at its own round**: weights `c i`, advantages `a i ≤ f (j i)`, and `m_i` the
finders at `i`'s round: `Σ_i √(c_i a_i) ≤ √((k · Σ_i m_i c_i) · ((Σ_r f r)/k))`. -/
theorem collapseR {I : Type} (s : Finset I) {k : ℕ} (hk : 0 < k) (c a : I → ℝ) (hc : ∀ i ∈ s, 0 ≤ c i)
    (ha : ∀ i ∈ s, 0 ≤ a i) (j : I → Fin k) (f : Fin k → ℝ) (hf : ∀ r, 0 ≤ f r)
    (haf : ∀ i ∈ s, a i ≤ f (j i)) :
    ∑ i ∈ s, Real.sqrt (c i * a i) ≤
      Real.sqrt ((k * ∑ i ∈ s, ((s.filter fun i' => j i' = j i).card : ℝ) * c i) * ((∑ r, f r) / k)) := by
  set m : I → ℝ := fun i => ((s.filter fun i' => j i' = j i).card : ℝ) with hm
  have hm0 : ∀ i ∈ s, 0 < m i := fun i hi => by
    simp only [hm, Nat.cast_pos]
    exact Finset.card_pos.2 ⟨i, Finset.mem_filter.2 ⟨hi, rfl⟩⟩
  have hcs : (∑ i ∈ s, Real.sqrt (m i * c i) * Real.sqrt (a i / m i)) ^ 2 ≤
      (∑ i ∈ s, Real.sqrt (m i * c i) ^ 2) * ∑ i ∈ s, Real.sqrt (a i / m i) ^ 2 :=
    Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  rw [Finset.sum_congr rfl fun i hi => Real.sq_sqrt (mul_nonneg (hm0 i hi).le (hc i hi)),
    Finset.sum_congr rfl fun i hi => Real.sq_sqrt (div_nonneg (ha i hi) (hm0 i hi).le)] at hcs
  have heq : ∀ i ∈ s, Real.sqrt (m i * c i) * Real.sqrt (a i / m i) = Real.sqrt (c i * a i) := fun i hi => by
    rw [← Real.sqrt_mul (mul_nonneg (hm0 i hi).le (hc i hi))]
    congr 1
    field_simp [(hm0 i hi).ne']
  rw [Finset.sum_congr rfl heq] at hcs
  have hsa : ∑ i ∈ s, a i / m i ≤ ∑ r, f r := by
    calc ∑ i ∈ s, a i / m i ≤ ∑ i ∈ s, f (j i) / m i :=
          Finset.sum_le_sum fun i hi => div_le_div_of_nonneg_right (haf i hi) (hm0 i hi).le
      _ = ∑ r, ∑ i ∈ s.filter (fun i => j i = r), f (j i) / m i := (Finset.sum_fiberwise s j _).symm
      _ ≤ ∑ r, f r := Finset.sum_le_sum fun r _ => ?_
    rcases (s.filter fun i => j i = r).eq_empty_or_nonempty with he | hne
    · rw [he, Finset.sum_empty]
      exact hf r
    · have hfib : ∀ i ∈ s.filter (fun i => j i = r),
          f (j i) / m i = f r / ((s.filter fun i => j i = r).card : ℝ) := fun i hi => by
        show f (j i) / ((s.filter fun i' => j i' = j i).card : ℝ) = _
        rw [(Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hfib, Finset.sum_const, nsmul_eq_mul]
      have hpos : (0 : ℝ) < (s.filter fun i => j i = r).card := by exact_mod_cast hne.card_pos
      rw [mul_div_cancel₀ _ hpos.ne']
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  have hrhs : (k * ∑ i ∈ s, m i * c i) * ((∑ r, f r) / k) = (∑ i ∈ s, m i * c i) * ∑ r, f r := by
    field_simp
  rw [hrhs]
  have h0 : 0 ≤ ∑ i ∈ s, Real.sqrt (c i * a i) := Finset.sum_nonneg fun i _ => Real.sqrt_nonneg _
  rw [← Real.sqrt_sq h0]
  exact Real.sqrt_le_sqrt (hcs.trans (mul_le_mul_of_nonneg_left hsa
    (Finset.sum_nonneg fun i hi => mul_nonneg (hm0 i hi).le (hc i hi))))

section Escapes

variable {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}

/-- **The escapes fit `C`, each fork event counted at its own round**: `Σ_i m_i c_i ≤ 2C`, with `m_i` the fork events
at `i`'s round, and `Σ_l β_l ≤ k(x)`. -/
def Escapes.FitsR (e : Escapes H V P x) (C : ℝ) : Prop :=
  ∑ i, ((Finset.univ.filter fun i' => e.j i' = e.j i).card : ℝ) * e.c i ≤ 2 * C ∧ ∑ l, e.β l ≤ V.k x

/-- **The escapes fit `C` with one more fork event** of weight `c` at round `0`, each counted at its own round:
`(m_0 + 1)·c + Σ_i m'_i c_i ≤ 2C`, with `m_0` the fork events at round `0` and `m'_i` those at `i`'s round counting the
extra one, and `Σ_l β_l ≤ k(x)`. -/
def Escapes.FitsForkR (e : Escapes H V P x) (C c : ℝ) : Prop :=
  (((Finset.univ.filter fun i => (e.j i).val = 0).card : ℝ) + 1) * c +
      ∑ i, (((Finset.univ.filter fun i' => e.j i' = e.j i).card : ℝ) + if (e.j i).val = 0 then 1 else 0) * e.c i ≤
    2 * C ∧ ∑ l, e.β l ≤ V.k x

theorem Escapes.Fits.fitsR [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] {e : Escapes H V P x} {C : ℝ}
    (h : e.Fits C) : e.FitsR C := by
  obtain ⟨mult, hm, hmc, hβ⟩ := h
  refine ⟨le_trans ?_ hmc, hβ⟩
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (by exact_mod_cast hm (e.j i)) (e.c_nonneg i)

theorem Escapes.FitsFork.fitsForkR {e : Escapes H V P x} {C c : ℝ} (h : e.FitsFork C c) (hc : 0 ≤ c)
    (hk : 0 < V.k x) : e.FitsForkR C c := by
  obtain ⟨mult, hm, hmc, hβ⟩ := h
  refine ⟨le_trans ?_ hmc, hβ⟩
  have h0 : ((Finset.univ.filter fun i => (e.j i).val = 0).card : ℝ) + 1 ≤ mult := by
    have := hm ⟨0, hk⟩
    simp only [↓reduceIte] at this
    have he : (Finset.univ.filter fun i => (e.j i).val = 0) = Finset.univ.filter fun i => e.j i = ⟨0, hk⟩ :=
      Finset.filter_congr fun i _ => by rw [Fin.ext_iff]
    rw [he]
    exact_mod_cast this
  rw [mul_add, Finset.mul_sum]
  refine add_le_add (mul_le_mul_of_nonneg_right h0 hc) (Finset.sum_le_sum fun i _ =>
    mul_le_mul_of_nonneg_right ?_ (e.c_nonneg i))
  have := hm (e.j i)
  exact_mod_cast this

/-- **With the extra fork event at round `0`, the escapes fit `C` counted per round.** -/
theorem Escapes.fitsR_addFork (e : Escapes H V P x) (C c : ℝ) (hc : 0 ≤ c) (j₀ : Fin (V.k x)) (hj₀ : j₀.val = 0)
    (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop) (tied) (h : e.FitsForkR C c) :
    (e.addFork c hc j₀ ev tied).FitsR C := by
  obtain ⟨hmc, hβ⟩ := h
  refine ⟨le_trans (le_of_eq ?_) hmc, hβ⟩
  have hcard : ∀ r : Fin (V.k x), ((Finset.univ : Finset (Option e.I)).filter fun o => o.elim j₀ e.j = r).card =
      (if r.val = 0 then 1 else 0) + (Finset.univ.filter fun i => e.j i = r).card := fun r => by
    refine (Claims.Flock.card_filter_option _ (fun o : Option e.I => o.elim j₀ e.j) r).trans ?_
    have hj : j₀ = r ↔ r.val = 0 := by rw [Fin.ext_iff, hj₀, eq_comm]
    simp only [Option.elim_none, hj]
    rfl
  show ∑ o : Option e.I,
      (((Finset.univ : Finset (Option e.I)).filter fun o' => o'.elim j₀ e.j = o.elim j₀ e.j).card : ℝ) *
        o.elim c e.c = _
  rw [Claims.Flock.sum_option_any]
  show (((Finset.univ : Finset (Option e.I)).filter fun o => o.elim j₀ e.j = j₀).card : ℝ) * c +
    ∑ a, (((Finset.univ : Finset (Option e.I)).filter fun o => o.elim j₀ e.j = e.j a).card : ℝ) * e.c a = _
  rw [hcard]
  simp only [hcard]
  have h0 : (Finset.univ.filter fun i => e.j i = j₀) = Finset.univ.filter fun i => (e.j i).val = 0 :=
    Finset.filter_congr fun i _ => by rw [Fin.ext_iff, hj₀]
  rw [h0, hj₀]
  simp only [↓reduceIte]
  push_cast
  exact congrArg₂ (· + ·) (by ring) (Finset.sum_congr rfl fun i _ => by ring)

/-- **Escapes with one fork event at every round fit `C` with the extra one** at round `0` when
`2·c + c_0 + Σ_r c_r ≤ 2C` and `Σ_l β_l ≤ k(x)`. -/
theorem Escapes.fitsForkR_of_round (e : Escapes H V P x) (C c : ℝ) (hj : Function.Bijective e.j) (i₀ : e.I)
    (hi₀ : (e.j i₀).val = 0) (h : 2 * c + e.c i₀ + ∑ i, e.c i ≤ 2 * C) (hβ : ∑ l, e.β l ≤ V.k x) :
    e.FitsForkR C c := by
  classical
  refine ⟨le_trans (le_of_eq ?_) h, hβ⟩
  have hone : ∀ i, (Finset.univ.filter fun i' => e.j i' = e.j i) = {i} := fun i =>
    Finset.ext fun i' => by simp [hj.1.eq_iff]
  have h0 : (Finset.univ.filter fun i => (e.j i).val = 0) = {i₀} := Finset.ext fun i => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    refine ⟨fun h => hj.1 (Fin.ext (h.trans hi₀.symm)), fun h => h ▸ hi₀⟩
  simp only [hone, h0, Finset.card_singleton, Nat.cast_one]
  have hsplit : ∑ i, ((1 : ℝ) + if (e.j i).val = 0 then 1 else 0) * e.c i = ∑ i, e.c i + e.c i₀ := by
    have hif : ∀ i, (if (e.j i).val = 0 then (1 : ℝ) else 0) * e.c i = if i = i₀ then e.c i₀ else 0 := fun i => by
      by_cases h : i = i₀
      · subst h
        simp [hi₀]
      · have : ¬ (e.j i).val = 0 := fun h' => h (hj.1 (Fin.ext (h'.trans hi₀.symm)))
        simp [this, h]
    simp only [add_mul, one_mul, Finset.sum_add_distrib, hif, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  rw [hsplit]
  ring

variable [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin]

/-- **The escapes, into one root**, counted per round: escapes that fit `C` at their rounds average, over the tape, to
at most `√((√(2·k(x)·C) + k(x))² · Pr[B])`; four times that is `√(16·(√(2·k(x)·C) + k(x))² · Pr[B])`. -/
theorem Escapes.le_of_fitsR (e : Escapes H V P x) (C : ℝ) (hfit : e.FitsR C) :
    4 * Ex e.term ≤ Real.sqrt (16 * (Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrB H V P x) := by
  obtain ⟨hmc, hβ⟩ := hfit
  have hc0 : 0 ≤ ∑ i, ((Finset.univ.filter fun i' => e.j i' = e.j i).card : ℝ) * e.c i :=
    Finset.sum_nonneg fun i _ => mul_nonneg (Nat.cast_nonneg _) (e.c_nonneg i)
  have hC : 0 ≤ 2 * C := hc0.trans hmc
  have hp0 := PrB_nonneg H V P x
  have hp1 := PrB_le_one H V P x
  have hkx : (0 : ℝ) ≤ V.k x := Nat.cast_nonneg _
  set A : e.I → ℝ := fun i =>
    Pr fun s : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => e.ev i s.1 s.2.1 s.2.2 with hA
  set A₁ : e.L → ℝ := fun l => Pr fun s : Ω × (Fin (V.k x) → Coin) => e.ev₁ l s.1 s.2 with hA₁
  have hEx : Ex e.term ≤ (∑ i, Real.sqrt (e.c i * A i)) + ∑ l, e.β l * A₁ l := by
    unfold Escapes.term
    rw [Ex_add, Ex_sum, Ex_sum]
    refine add_le_add (Finset.sum_le_sum fun i _ => ?_) (le_of_eq (Finset.sum_congr rfl fun l _ => ?_))
    · refine (Ex_sqrt_le _ fun ω => mul_nonneg (e.c_nonneg i) (Pr_nonneg _)).trans (le_of_eq ?_)
      rw [Ex_mul_left, Ex_Pr]
    · rw [Ex_mul_left, Ex_Pr]
  have hfork : (∑ i, Real.sqrt (e.c i * A i)) ≤ Real.sqrt (2 * V.k x * C * PrB H V P x) := by
    by_cases hk : 0 < V.k x
    · have hf0 : 0 ≤ (∑ r, forkPr H V P x hk r) / V.k x := by
        rw [← PrB_eq_avg H V P x hk]
        exact hp0
      refine (collapseR Finset.univ hk e.c A (fun i _ => e.c_nonneg i) (fun i _ => Pr_nonneg _) e.j
        (forkPr H V P x hk) (fun _ => Pr_nonneg _)
        (fun i _ => Pr_fork_le H V P x hk (e.j i) (e.ev i) (e.tied i))).trans ?_
      rw [PrB_eq_avg H V P x hk]
      refine Real.sqrt_le_sqrt ?_
      have := mul_le_mul_of_nonneg_left hmc (mul_nonneg hkx hf0)
      nlinarith [this]
    · have : IsEmpty e.I := ⟨fun i => absurd (e.j i).pos hk⟩
      simp only [Finset.univ_eq_empty, Finset.sum_empty]
      exact Real.sqrt_nonneg _
  have hlin : ∑ l, e.β l * A₁ l ≤ V.k x * PrB H V P x := by
    refine (Finset.sum_le_sum fun l _ =>
      mul_le_mul_of_nonneg_left (Pr_one_le_PrB H V P x (e.ev₁ l) (e.tied₁ l)) (e.β_nonneg l)).trans ?_
    rw [← Finset.sum_mul]
    exact mul_le_mul_of_nonneg_right hβ hp0
  have hroot := sqrt_add_lin (C := 2 * V.k x * C) (β := V.k x) (by nlinarith) hkx hp0 hp1
  have h16 : Real.sqrt (16 * (Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrB H V P x) =
      4 * Real.sqrt ((Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrB H V P x) := by
    rw [mul_assoc, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 16), show (16 : ℝ) = 4 ^ 2 by norm_num,
      Real.sqrt_sq (by norm_num)]
  rw [h16]
  linarith

end Escapes

end Claims
