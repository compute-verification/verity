import Proofs.Flock.Bridge.PubFinder

/-!
# C-Flock's knowledge term from the averaged premise

`ksAvgPrZ`, C-Flock's knowledge term at its finders' collision probabilities, averages over the verifier's draw `ω`
the largest over the plan's `J = nTab I` tables of each table's bound `ksPrZ / (1 − rate)`. Each table's bound is a
sum of square roots of its finders' collision probabilities. This file bounds the term from one premise averaged over
the draw: for every table `j < J` and slot `s`, the slot-`s` finder of table `j` succeeds with probability at most `p`
on average over `ω` (`tabMix`'s, which draws `ω` itself, so one finder of a fixed budget meets it):

$$\mathrm{ksAvgPrZ}\le\frac{2^{-203}+J\big((2\cdot25638+2^{14}+2)\sqrt p+p+\sqrt{kp}\big)}{1-\rho}.$$

* `ksPrZ_le_slotsAt`: after each draw, table `j`'s bound is at most `2^-203 + ksSlots`, a sum over the slots of
  `TabIx 7` whose coefficients read no draw. A table is at C-Flock ZK's parameters (`planZHJ_M1W`), at `m1Schedule m`
  for some `25 ≤ m ≤ 35`, which has at most 7 rewinding levels, each level's `2^{log N_ℓ}·Q_ℓ` at most
  `m1Schedule 35`'s (`m1_le_35`), so level `ℓ`'s coefficient is `rootUp` of `m1Schedule 35`'s, summing to at most
  `25638` (`m1Ks_all`).
* The largest over the tables is at most their sum (the factor `J`), and the average over `ω` of each slot's square
  root is at most the square root of its average (Jensen, `ex_sqrt_le`), which is at most `√p`.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Model FlockSoundness.Accounting
  FlockSoundness.Discharge.Exec FlockSoundness.Refine FlockSoundness.Discharge.ZkSession
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

/-! ## Uniform averages -/

theorem ex_sum {α ι : Type} [Fintype α] (s : Finset ι) (f : ι → α → ℝ) :
    Claims.Ex (fun a => ∑ i ∈ s, f i a) = ∑ i ∈ s, Claims.Ex (f i) := by
  unfold Claims.Ex
  rw [Finset.sum_comm, Finset.sum_div]

theorem ex_mul_left {α : Type} [Fintype α] (c : ℝ) (f : α → ℝ) :
    Claims.Ex (fun a => c * f a) = c * Claims.Ex f := by
  unfold Claims.Ex
  rw [← Finset.mul_sum, mul_div_assoc]

/-- **Jensen for the square root**: the average of `√f` is at most `√` of the average of `f`. -/
theorem ex_sqrt_le {α : Type} [Fintype α] [Nonempty α] (f : α → ℝ) (hf : ∀ a, 0 ≤ f a) :
    Claims.Ex (fun a => Real.sqrt (f a)) ≤ Real.sqrt (Claims.Ex f) := by
  unfold Claims.Ex
  have hc : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  have h := Real.sum_sqrt_mul_sqrt_le (Finset.univ : Finset α) (f := fun _ => (1 : ℝ)) (g := f)
    (fun _ => zero_le_one) hf
  simp only [Real.sqrt_one, one_mul, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one] at h
  rw [Real.sqrt_div (Finset.sum_nonneg fun a _ => hf a), div_le_div_iff₀ hc (Real.sqrt_pos.2 hc)]
  calc (∑ a, Real.sqrt (f a)) * Real.sqrt (Fintype.card α)
      ≤ Real.sqrt (Fintype.card α) * Real.sqrt (∑ a, f a) * Real.sqrt (Fintype.card α) :=
        mul_le_mul_of_nonneg_right h (Real.sqrt_nonneg _)
    _ = Real.sqrt (∑ a, f a) * (Real.sqrt (Fintype.card α) * Real.sqrt (Fintype.card α)) := by ring
    _ = Real.sqrt (∑ a, f a) * Fintype.card α := by rw [Real.mul_self_sqrt hc.le]

theorem avg_ofReal_le_ex {α : Type} [Fintype α] [Nonempty α] (f : α → ℝ) (hf : ∀ a, 0 ≤ f a) :
    avg (fun a => ENNReal.ofReal (f a)) ≤ ENNReal.ofReal (Claims.Ex f) := by
  have hc : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  refine le_of_eq ?_
  unfold avg Claims.Ex
  rw [div_eq_mul_inv, Finset.sum_mul, ENNReal.ofReal_sum_of_nonneg fun a _ =>
    mul_nonneg (hf a) (inv_nonneg.2 hc.le)]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [ENNReal.ofReal_mul (hf a), ENNReal.ofReal_inv_of_pos hc, ENNReal.ofReal_natCast, mul_comm]

/-! ## C-Flock ZK's schedules against `m1Schedule 35` -/

/-- **`m1Schedule 35` dominates**: at `m1Schedule m`, `25 ≤ m ≤ 35`, at most 7 levels, and each level `ℓ ≥ 1`'s
`2^{log N_ℓ}·Q_ℓ` is at most `m1Schedule 35`'s. -/
theorem m1_le_35 : ∀ m, 25 ≤ m → m ≤ 35 →
    (m1Schedule m).levels.length ≤ 7 ∧
      ∀ ℓ ∈ Finset.Icc 1 (m1Schedule m).levels.length, hashArg (m1Schedule m) ℓ ≤ hashArg (m1Schedule 35) ℓ := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

theorem m1_35_length : (m1Schedule 35).levels.length = 7 := by decide +kernel

theorem hashArg_cast (s : FlockSoundness.Accounting.Schedule) (ℓ : ℕ) :
    (2 : ℝ) ^ (lvOf s ℓ).logLen * ((lvOf s ℓ).queries : ℝ) = (hashArg s ℓ : ℝ) := by
  unfold hashArg
  push_cast
  ring

/-- `m1Schedule 35`'s hash-sum coefficient over levels `1..7` is at most `25638` (`m1Ks_all`). -/
theorem hashUp35_le : ∑ ℓ ∈ Finset.Icc 1 7, (rootUp (hashArg (m1Schedule 35) ℓ) : ℝ) ≤ 25638 := by
  have h := (m1Ks_all 35 (by norm_num) le_rfl).2.1
  rw [hashUp, m1_35_length] at h
  exact_mod_cast h

/-! ## A table's bound by its finder slots -/

/-- Level `ℓ`'s rewinding slot of rep `r` (`ℓ ≥ 1`), where `TabIx 7` has it. -/
noncomputable def rsPr (Xs : TabIx 7 → ℝ) (r : ℕ) : ℕ → ℝ
  | 0 => 0
  | ℓ + 1 => if h : r < 2 ∧ ℓ < 7 then Xs (.inl (⟨r, h.1⟩, ⟨ℓ, h.2⟩)) else 0

/-- Rep `r`'s pads slot. -/
noncomputable def padsPr (Xs : TabIx 7 → ℝ) (r : ℕ) : ℝ := if h : r < 2 then Xs (.inr (.inl ⟨r, h⟩)) else 0

/-- Rep `r`'s `τ` slot. -/
noncomputable def tauPr (Xs : TabIx 7 → ℝ) (r : ℕ) : ℝ := if h : r < 2 then Xs (.inr (.inr (.inl ⟨r, h⟩))) else 0

/-- **A table's knowledge bound past `2^-203`, by slot**, at each slot's collision probability `Xs s`, with
coefficients that read no table: `m1Schedule 35`'s per level, `2^13` for the pads. -/
noncomputable def ksSlots (Xs : TabIx 7 → ℝ) (k : ℕ) : ℝ :=
  ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 7, (rootUp (hashArg (m1Schedule 35) ℓ) : ℝ) * Real.sqrt (rsPr Xs r ℓ) +
    ∑ r ∈ Finset.range 2, 2 ^ 13 * Real.sqrt (padsPr Xs r) +
    ∑ r ∈ Finset.range 2, Real.sqrt (tauPr Xs r) +
    Xs (.inr (.inr (.inr (.inr ())))) + Real.sqrt k * Real.sqrt (Xs (.inr (.inr (.inr (.inl ())))))

section Slots

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : FlockSoundness.Discharge.ZkLink.Hashes F D Sl)

/-- **`ε_c⁻` at a `M1W` table, by slot**: `epsPrZ` at most `2^-203` plus `ksSlots`' session terms, where each finder's
collision probability is its slot's. -/
theorem epsPrZ_le_slots (hF : Fintype.card F = 2 ^ 128) (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D) (hT : T.M1W)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (Xs : TabIx 7 → ℝ)
    (hRS : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
      (advRSFinderZ A Hs T pre post cap₀ r ℓ σ).ev (Game.ind (StrictCR.IsColl Hs.H)) = rsPr Xs r ℓ)
    (hP : ∀ r ∈ Finset.range 2,
      (advPadsSFinderZ A Hs T pre post cap₀ r σ).ev (Game.ind (StrictCR.IsColl Hs.H)) = padsPr Xs r)
    (hTau : ∀ r ∈ Finset.range 2,
      (advTauSFinderZ A Hs T pre post cap₀ r σ).ev (Game.ind (StrictCR.IsColl Hs.H)) = tauPr Xs r)
    (hS : (selfSFinderZ A Hs T pre post cap₀ σ).ev (Game.ind (StrictCR.IsColl Hs.H)) =
      Xs (.inr (.inr (.inr (.inr ()))))) :
    epsPrZ A Hs T pre post cap₀ σ ≤
      2 ^ (-(203 : ℝ)) +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 7,
          (rootUp (hashArg (m1Schedule 35) ℓ) : ℝ) * Real.sqrt (rsPr Xs r ℓ) +
        ∑ r ∈ Finset.range 2, 2 ^ 13 * Real.sqrt (padsPr Xs r) +
        ∑ r ∈ Finset.range 2, Real.sqrt (tauPr Xs r) +
        Xs (.inr (.inr (.inr (.inr ())))) := by
  have hb : T.bound.toReal ≤ 2 ^ (-(203 : ℝ)) :=
    ENNReal.toReal_le_of_le_ofReal (by positivity) (T.bound_le_m1w hF hT)
  obtain ⟨m, h1, h2, hsch, -, -, -, hN, -, -⟩ := hT
  obtain ⟨hlen, hdom⟩ := m1_le_35 m h1 h2
  have hr35 := (m1Ks_all 35 (by norm_num) le_rfl).1
  rw [m1_35_length] at hr35
  have hsub : Finset.Icc 1 T.sch.levels.length ⊆ Finset.Icc 1 7 :=
    Finset.Icc_subset_Icc le_rfl (by rw [hsch]; exact hlen)
  unfold epsPrZ
  refine add_le_add (add_le_add (add_le_add (add_le_add hb ?_) ?_) ?_) hS.le
  · refine Finset.sum_le_sum fun r hr => ?_
    refine le_trans (Finset.sum_le_sum fun ℓ hℓ => ?_)
      (Finset.sum_le_sum_of_subset_of_nonneg hsub fun ℓ _ _ => by positivity)
    rw [hRS r hr ℓ hℓ, hashArg_cast, Real.sqrt_mul (Nat.cast_nonneg _)]
    refine mul_le_mul_of_nonneg_right ?_ (Real.sqrt_nonneg _)
    have hℓ' : ℓ ∈ Finset.Icc 1 (m1Schedule m).levels.length := by rw [← hsch]; exact hℓ
    have hd : (hashArg T.sch ℓ : ℝ) ≤ (hashArg (m1Schedule 35) ℓ : ℝ) := by
      rw [hsch]
      exact_mod_cast hdom ℓ hℓ'
    exact (Real.sqrt_le_sqrt hd).trans (sqrt_cast_le_of_le_mul (hr35 ℓ (hsub hℓ)))
  · refine Finset.sum_le_sum fun r hr => ?_
    rw [hP r hr, Real.sqrt_mul (mul_self_nonneg _), Real.sqrt_mul_self (Nat.cast_nonneg _)]
    have hN' : (T.N : ℝ) ≤ 2 ^ 13 := by exact_mod_cast hN
    exact mul_le_mul_of_nonneg_right hN' (Real.sqrt_nonneg _)
  · exact Finset.sum_le_sum fun r hr => le_of_eq (by rw [hTau r hr])

/-- **A `M1W` table's knowledge bound, by slot**: `ksPrZ ≤ 2^-203 + ksSlots Xs k`, where each finder's collision
probability is its slot's `Xs s`. -/
theorem ksPrZ_le_slots (hF : Fintype.card F = 2 ^ 128) (ts : List (TabZK F mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionZC A Hs ts)) (hT : (ts.get j).M1W) (Xs : TabIx 7 → ℝ)
    (hRS : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (ts.get j).sch.levels.length,
      (advRSFinderZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j) r ℓ
        (stratZ A Hs ts j τ)).ev (Game.ind (StrictCR.IsColl Hs.H)) = rsPr Xs r ℓ)
    (hP : ∀ r ∈ Finset.range 2,
      (advPadsSFinderZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j) r
        (stratZ A Hs ts j τ)).ev (Game.ind (StrictCR.IsColl Hs.H)) = padsPr Xs r)
    (hTau : ∀ r ∈ Finset.range 2,
      (advTauSFinderZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j) r
        (stratZ A Hs ts j τ)).ev (Game.ind (StrictCR.IsColl Hs.H)) = tauPr Xs r)
    (hS : (selfSFinderZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
        (stratZ A Hs ts j τ)).ev (Game.ind (StrictCR.IsColl Hs.H)) = Xs (.inr (.inr (.inr (.inr ())))))
    (h0 : (adv₀SFinderZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
        (stratZ A Hs ts j τ)).ev (Game.ind (StrictCR.IsColl Hs.H)) = Xs (.inr (.inr (.inr (.inl ()))))) :
    ksPrZ A Hs ts j k τ ≤ 2 ^ (-(203 : ℝ)) + ksSlots Xs k := by
  have he := epsPrZ_le_slots A Hs hF (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1))
    (τ.1 j) hT (stratZ A Hs ts j τ) Xs hRS hP hTau hS
  unfold ksPrZ ksSlots
  rw [h0, Real.sqrt_mul (Nat.cast_nonneg k)]
  linarith

end Slots

/-- **`ksSlots`' average over the draw from the averaged premise**: where every slot's average is at most `p`, the
average of `ksSlots` is at most `(2·25638 + 2^14 + 2)·√p + p + √(kp)` (`ex_sqrt_le` on each square root). -/
theorem ex_ksSlots_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (Xs : Ω → TabIx 7 → ℝ) (k : ℕ) {p : ℝ} (hp : 0 ≤ p)
    (h0 : ∀ ω s, 0 ≤ Xs ω s) (havg : ∀ s, Claims.Ex (fun ω => Xs ω s) ≤ p) :
    Claims.Ex (fun ω => ksSlots (Xs ω) k) ≤
      (2 * 25638 + 2 ^ 14 + 2) * Real.sqrt p + p + Real.sqrt (k * p) := by
  have hsq : ∀ f : Ω → ℝ, (∀ ω, 0 ≤ f ω) → Claims.Ex f ≤ p →
      Claims.Ex (fun ω => Real.sqrt (f ω)) ≤ Real.sqrt p :=
    fun f hf h => (ex_sqrt_le f hf).trans (Real.sqrt_le_sqrt h)
  have hrs0 : ∀ ω r ℓ, 0 ≤ rsPr (Xs ω) r ℓ := by
    intro ω r ℓ
    cases ℓ with
    | zero => exact le_rfl
    | succ ℓ =>
      simp only [rsPr]
      split_ifs
      · exact h0 _ _
      · exact le_rfl
  have hrsA : ∀ r ℓ, Claims.Ex (fun ω => rsPr (Xs ω) r ℓ) ≤ p := by
    intro r ℓ
    cases ℓ with
    | zero => simp only [rsPr, Claims.Ex_const]; exact hp
    | succ ℓ =>
      by_cases h : r < 2 ∧ ℓ < 7
      · simp only [rsPr, dif_pos h]; exact havg _
      · simp only [rsPr, dif_neg h, Claims.Ex_const]; exact hp
  have hpd0 : ∀ ω r, 0 ≤ padsPr (Xs ω) r := fun ω r => by
    unfold padsPr; split_ifs
    · exact h0 _ _
    · exact le_rfl
  have hpdA : ∀ r, Claims.Ex (fun ω => padsPr (Xs ω) r) ≤ p := by
    intro r
    by_cases h : r < 2
    · simp only [padsPr, dif_pos h]; exact havg _
    · simp only [padsPr, dif_neg h, Claims.Ex_const]; exact hp
  have htu0 : ∀ ω r, 0 ≤ tauPr (Xs ω) r := fun ω r => by
    unfold tauPr; split_ifs
    · exact h0 _ _
    · exact le_rfl
  have htuA : ∀ r, Claims.Ex (fun ω => tauPr (Xs ω) r) ≤ p := by
    intro r
    by_cases h : r < 2
    · simp only [tauPr, dif_pos h]; exact havg _
    · simp only [tauPr, dif_neg h, Claims.Ex_const]; exact hp
  simp only [ksSlots, Claims.Ex_add, ex_sum, ex_mul_left]
  have hA : ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 7, (rootUp (hashArg (m1Schedule 35) ℓ) : ℝ) *
      Claims.Ex (fun ω => Real.sqrt (rsPr (Xs ω) r ℓ)) ≤ 2 * 25638 * Real.sqrt p := by
    calc _ ≤ ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 7,
            (rootUp (hashArg (m1Schedule 35) ℓ) : ℝ) * Real.sqrt p := by
          gcongr with r hr ℓ hℓ
          exact hsq _ (fun ω => hrs0 ω r ℓ) (hrsA r ℓ)
      _ = 2 * ((∑ ℓ ∈ Finset.Icc 1 7, (rootUp (hashArg (m1Schedule 35) ℓ) : ℝ)) * Real.sqrt p) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, ← Finset.sum_mul]
          push_cast
          ring
      _ ≤ 2 * 25638 * Real.sqrt p := by
          have := mul_le_mul_of_nonneg_right hashUp35_le (Real.sqrt_nonneg p)
          linarith
  have hB : ∑ r ∈ Finset.range 2, 2 ^ 13 * Claims.Ex (fun ω => Real.sqrt (padsPr (Xs ω) r)) ≤
      2 ^ 14 * Real.sqrt p := by
    calc _ ≤ ∑ _r ∈ Finset.range 2, (2 : ℝ) ^ 13 * Real.sqrt p := by
          gcongr with r hr
          exact hsq _ (fun ω => hpd0 ω r) (hpdA r)
      _ = 2 ^ 14 * Real.sqrt p := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
          push_cast
          ring
  have hC : ∑ r ∈ Finset.range 2, Claims.Ex (fun ω => Real.sqrt (tauPr (Xs ω) r)) ≤ 2 * Real.sqrt p := by
    calc _ ≤ ∑ _r ∈ Finset.range 2, Real.sqrt p :=
          Finset.sum_le_sum fun r _ => hsq _ (fun ω => htu0 ω r) (htuA r)
      _ = 2 * Real.sqrt p := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
          push_cast
          ring
  have hS := havg (.inr (.inr (.inr (.inr ()))))
  have hZ : Real.sqrt k * Claims.Ex (fun ω => Real.sqrt (Xs ω (.inr (.inr (.inr (.inl ())))))) ≤
      Real.sqrt (k * p) := by
    rw [Real.sqrt_mul (Nat.cast_nonneg k)]
    exact mul_le_mul_of_nonneg_left (hsq _ (fun ω => h0 _ _) (havg _)) (Real.sqrt_nonneg _)
  linarith

theorem ksSlots_nonneg (Xs : TabIx 7 → ℝ) (k : ℕ) (h0 : ∀ s, 0 ≤ Xs s) : 0 ≤ ksSlots Xs k := by
  unfold ksSlots
  exact add_nonneg (add_nonneg (by positivity) (h0 _)) (by positivity)

/-! ## At the hidden-output plan -/

section Family

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {L : Law n}

/-- **Table `j`'s slot-`s` collision probability after draw `ω`** (`tabAt`'s). -/
noncomputable def slotPr (qF qS : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R)
    (ω : L.Ω) (j : Fin (nTab I)) (s : TabIx 7) : ℝ :=
  (tabAt dj y₀ qF qS R τ ω j s).Pr Claims.Sampling.sha512Msg

/-- **Table `j`'s knowledge bound after draw `ω`, by slot**: at most `2^-203 + ksSlots` at `slotPr`
(`ksPrZ_le_slots`, every table at `M1W` by `planZHJ_M1W`). -/
theorem ksPrZ_le_slotsAt (qF qS : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (k : ℕ) (ω : L.Ω) (j : Fin (nTab I)) :
    ksPrZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) (tabOf dj y₀ R ω j) k
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)) ≤
      2 ^ (-(203 : ℝ)) + ksSlots (slotPr dj y₀ qF qS R τ ω j) k := by
  have hT := planZHJ_M1W dj y₀ (L.draw ω) R (tabOf dj y₀ R ω j)
  have hlen : ((planZHJ dj y₀ (L.draw ω) R).get (tabOf dj y₀ R ω j)).sch.levels.length ≤ 7 := by
    obtain ⟨m, h1, h2, hsch, -⟩ := hT
    rw [hsch]
    exact (m1_le_35 m h1 h2).1
  refine ksPrZ_le_slots execArith hs512 execArith_correct.card_F _ _ k _ hT _ ?_ ?_ ?_ ?_ ?_
  · intro r hr ℓ hℓ
    have hr2 : r < 2 := Finset.mem_range.1 hr
    obtain ⟨h1, h2⟩ := Finset.mem_Icc.1 hℓ
    obtain ⟨ℓ, rfl⟩ : ∃ ℓ', ℓ = ℓ' + 1 := ⟨ℓ - 1, by omega⟩
    have hc : r < 2 ∧ ℓ < 7 := ⟨hr2, by omega⟩
    have hl : ℓ < ((planZHJ dj y₀ (L.draw ω) R).get (tabOf dj y₀ R ω j)).sch.levels.length := by omega
    simp only [rsPr, dif_pos hc, slotPr]
    rw [tabAt_inl dj y₀ qF qS R τ ω j ⟨r, hr2⟩ ⟨ℓ, hc.2⟩ hl]
    exact (strictComp_Pr _ _).symm
  · intro r hr
    have hr2 : r < 2 := Finset.mem_range.1 hr
    simp only [padsPr, dif_pos hr2, slotPr]
    rw [tabAt_inr]
    exact (strictComp_Pr _ _).symm
  · intro r hr
    have hr2 : r < 2 := Finset.mem_range.1 hr
    simp only [tauPr, dif_pos hr2, slotPr]
    rw [tabAt_inr]
    exact (strictComp_Pr _ _).symm
  · rw [slotPr, tabAt_inr]
    exact (strictComp_Pr _ _).symm
  · rw [slotPr, tabAt_inr]
    exact (strictComp_Pr _ _).symm

/-- **C-Flock's knowledge term from the averaged premise**: where, for every table `j < J` and slot `s`, the slot's
finder succeeds with probability at most `p` on average over the draw,
$$\mathrm{ksAvgPrZ}\le\frac{2^{-203}+J\big((2\cdot25638+2^{14}+2)\sqrt p+p+\sqrt{kp}\big)}{1-\rho},$$
`J = nTab I`, at a rate cap `2^23/(ek) ≤ ρ < 1`. The largest over the tables is at most their sum, and Jensen moves
the average over the draw under each square root. -/
theorem ksAvgPrZ_le_avg (qF qS : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) {k : ℕ} {ρ : ℝ} (hρ : ρ < 1)
    (hρk : rateM1 k ≤ ρ) {p : ℝ} (hp : 0 ≤ p)
    (havg : ∀ j s, Claims.Ex (fun ω : L.Ω => slotPr dj y₀ qF qS R τ ω j s) ≤ p) :
    ksAvgPrZ execArith hs512 (planZHJ dj y₀) k R τ ≤
      ENNReal.ofReal ((2 ^ (-(203 : ℝ)) +
        nTab I * ((2 * 25638 + 2 ^ 14 + 2) * Real.sqrt p + p + Real.sqrt (k * p))) / (1 - ρ)) := by
  have h1ρ : 0 < 1 - ρ := sub_pos.2 hρ
  have hs0 : ∀ ω j, 0 ≤ ksSlots (slotPr dj y₀ qF qS R τ ω j) k := fun ω j =>
    ksSlots_nonneg _ k fun _ => Claims.Pr_nonneg _
  have hg0 : ∀ ω : L.Ω, 0 ≤ (2 ^ (-(203 : ℝ)) + ∑ j : Fin (nTab I), ksSlots (slotPr dj y₀ qF qS R τ ω j) k) /
      (1 - ρ) := fun ω =>
    div_nonneg (add_nonneg (by positivity) (Finset.sum_nonneg fun j _ => hs0 ω j)) h1ρ.le
  have hω : ∀ ω : L.Ω, (⨆ j' : Fin (planZHJ dj y₀ (L.draw ω) R).length,
      ENNReal.ofReal (ksPrZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) j' k
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)) / (1 - rateZ (planZHJ dj y₀ (L.draw ω) R) j' k))) ≤
      ENNReal.ofReal ((2 ^ (-(203 : ℝ)) + ∑ j : Fin (nTab I), ksSlots (slotPr dj y₀ qF qS R τ ω j) k) /
        (1 - ρ)) := fun ω => iSup_le fun j' => by
    obtain ⟨j, rfl⟩ : ∃ j, tabOf dj y₀ R ω j = j' :=
      ⟨Fin.cast (planZHJ_length dj y₀ (L.draw ω) R) j', Fin.ext rfl⟩
    have hks := ksPrZ_le_slotsAt dj y₀ qF qS R τ k ω j
    have hr := (rateZ_le_m1w _ _ (planZHJ_M1W dj y₀ (L.draw ω) R (tabOf dj y₀ R ω j)) k).trans hρk
    have h0 : 0 < 1 - rateZ (planZHJ dj y₀ (L.draw ω) R) (tabOf dj y₀ R ω j) k := by linarith
    have hone : ksSlots (slotPr dj y₀ qF qS R τ ω j) k ≤
        ∑ j : Fin (nTab I), ksSlots (slotPr dj y₀ qF qS R τ ω j) k :=
      Finset.single_le_sum (fun j _ => hs0 ω j) (Finset.mem_univ j)
    have hnum : 0 ≤ 2 ^ (-(203 : ℝ)) + ∑ j : Fin (nTab I), ksSlots (slotPr dj y₀ qF qS R τ ω j) k :=
      add_nonneg (by positivity) (Finset.sum_nonneg fun j _ => hs0 ω j)
    refine ENNReal.ofReal_le_ofReal ?_
    calc ksPrZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) (tabOf dj y₀ R ω j) k
            (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)) /
          (1 - rateZ (planZHJ dj y₀ (L.draw ω) R) (tabOf dj y₀ R ω j) k)
        ≤ (2 ^ (-(203 : ℝ)) + ∑ j : Fin (nTab I), ksSlots (slotPr dj y₀ qF qS R τ ω j) k) /
          (1 - rateZ (planZHJ dj y₀ (L.draw ω) R) (tabOf dj y₀ R ω j) k) :=
          div_le_div_of_nonneg_right (by linarith) h0.le
      _ ≤ _ := div_le_div_of_nonneg_left hnum h1ρ (by linarith)
  unfold ksAvgPrZ ksAvgZC
  refine le_trans (avg_mono fun ω => hω ω) ((avg_ofReal_le_ex _ hg0).trans (ENNReal.ofReal_le_ofReal ?_))
  have hJ : ∀ j : Fin (nTab I), Claims.Ex (fun ω => ksSlots (slotPr dj y₀ qF qS R τ ω j) k) ≤
      (2 * 25638 + 2 ^ 14 + 2) * Real.sqrt p + p + Real.sqrt (k * p) := fun j =>
    ex_ksSlots_le (fun ω => slotPr dj y₀ qF qS R τ ω j) k hp (fun _ _ => Claims.Pr_nonneg _) (havg j)
  rw [Claims.Ex_div, Claims.Ex_add, Claims.Ex_const, ex_sum]
  refine div_le_div_of_nonneg_right ?_ h1ρ.le
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hJ j
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
  linarith

/-- **C-Flock's knowledge term from the averaged premise**, at its knowledge bound `ksBoundAccZ`: `ksAvgZC_le_pr`,
then `ksAvgPrZ_le_avg`. It replaces a per-draw collision-resistance premise on each table's finders with one
averaged over the draw, which one finder that draws `ω` itself meets (`tabAvg_of_mix`), at the cost of the factor
`J = nTab I`. -/
theorem ksAvgZC_le_strictAvg (qF qS : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) {k : ℕ} {ρ : ℝ} (hρ : ρ < 1)
    (hρk : rateM1 k ≤ ρ) {p : ℝ} (hp : 0 ≤ p)
    (havg : ∀ j s, Claims.Ex (fun ω : L.Ω => slotPr dj y₀ qF qS R τ ω j s) ≤ p) :
    ksAvgZC execArith hs512 (planZHJ dj y₀) k
        (fun S R τ j => ksBoundAccZ execArith hs512 (planZHJ dj y₀ S R) j k τ) R τ ≤
      ENNReal.ofReal ((2 ^ (-(203 : ℝ)) +
        nTab I * ((2 * 25638 + 2 ^ 14 + 2) * Real.sqrt p + p + Real.sqrt (k * p))) / (1 - ρ)) :=
  (ksAvgZC_le_pr execArith hs512 (planZHJ dj y₀) k R τ).trans
    (ksAvgPrZ_le_avg dj y₀ qF qS R τ hρ hρk hp havg)

end Family

end Flock.Bridge
