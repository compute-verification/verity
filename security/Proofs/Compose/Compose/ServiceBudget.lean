module

public import Definitions.Compose.Guarantees.Compose
public import Proofs.Pouw.TS.Stages
public import Proofs.ResourceAccounting.Lemmas.Compose
public import Proofs.Core.Draw
public import Proofs.Core.Game.Tape
public import Proofs.Spot.Sampling.Work

/-!
# The service's window budget: proof

`ResourceAccounting.Guarantees.Compose.ServiceBudget`, step 4 of `Proofs.Compose.Compose`, in three parts.

* **The miss bound** (`miss_le`): a uniform permutation's `K`-subset (`Verity.SamplingDesign.permPrefix`) misses a set `B` of
  the `N` tiles with probability at most `(1 − |B|/N)^K`. The file reads the core: at `K ≤ N` the miss probability is
  the hypergeometric factor `C(N − |B|, K)/C(N, K)` (`miss_eq`, from `Verity.SamplingDesign.missProb_permPrefix_eq`),
  at most `(1 − |B|/N)^K` by `Spot.Sampling.SamplingDesign.choose_sub_mul_pow_le'`; at `K > N` the prefix is every tile.
* **One path of the stages' coins** (`path_le`): with `U` the work folded or whole whose checked values aren't honest,
  `ρ·U ≤ |B|/N` for `B` the tiles not honest (`ρ·N·w_v ≤ 1`), and `exp(d/Λ)·(1 − ρU)^K` is at most the product over
  the stages of `SplitRateAt`'s left side at each stage's work (`factor`), as `1 − ρ(V + U_j) ≤ (1 − ρV)(1 − ρU_j)`.
* **Backward induction over the stages** (`expect_factor_le`): given the past, `SplitRateAt` bounds the stage's factor
  by `1 + ν(L_F − p·W_F) + ν₀(L_0 − p0·W_0)`, and `SplitStages.LuckAt` with `lk ≤ p`, `lk0 ≤ p0` on every tile of
  positive weight gives `E[L_F] ≤ p·E[W_F]` and `E[L_0] ≤ p0·E[W_0]`, so its mean is at most `1`.
-/

@[expose] public section

open Claims renaming Ex → expect

namespace ResourceAccounting.SecurityProofs.Compose.Budget

open ResourceAccounting.TS ResourceAccounting.Compose Pouw.RelativeSecurity
open ResourceAccounting.SecurityProofs.TS.Stages (prob_prod sum_le_card_of_expect expect_empty)

/-! ## The miss bound -/

theorem forall_mem_permPrefix {N K : ℕ} (π : Equiv.Perm (Fin N)) (P : Fin N → Prop) :
    (∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw π, P v) ↔ ∀ i : Fin N, i.val < K → P (π i) := by
  simp [Verity.SamplingDesign.permPrefix]

/-- **A uniform permutation's `K`-prefix misses `B` with the hypergeometric factor, exactly**: `C(N − |B|, K)/C(N, K)`
for `K ≤ N`, the core's `Verity.SamplingDesign.missProb_permPrefix_eq` as a real probability. -/
theorem miss_eq (K N : ℕ) (B : Finset (Fin N)) (hK : K ≤ N) :
    prob (fun π : Equiv.Perm (Fin N) => ∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw π, v ∉ B) =
      ((N - B.card).choose K : ℝ) / (N.choose K : ℝ) := by
  have h := Verity.Game.pr_eq_toReal_prCoin
    (fun π : (Verity.SamplingDesign.permPrefix N K).Ω => Disjoint ((Verity.SamplingDesign.permPrefix N K).draw π) B)
  rw [show Verity.Game.prCoin (fun π : (Verity.SamplingDesign.permPrefix N K).Ω =>
        Disjoint ((Verity.SamplingDesign.permPrefix N K).draw π) B) =
      (Verity.SamplingDesign.permPrefix N K).missProb B from rfl,
    Verity.SamplingDesign.missProb_permPrefix_eq hK B, ENNReal.toReal_div, ENNReal.toReal_natCast,
    ENNReal.toReal_natCast,
    show (fun π : (Verity.SamplingDesign.permPrefix N K).Ω => Disjoint ((Verity.SamplingDesign.permPrefix N K).draw π) B) =
      fun π => ∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw π, v ∉ B from
      funext fun π => propext Finset.disjoint_left] at h
  first
    | exact h
    | (unfold Claims.Pr at h; convert h using 2)

/-- **The miss bound**: a uniform permutation's `K`-subset misses `B` with probability at most `(1 − |B|/N)^K`. For
`K ≤ N` it is `miss_eq`'s factor, which `C(N − |B|, K)·N^K ≤ C(N, K)·(N − |B|)^K` bounds; for `K > N` the prefix is every
tile, so a nonempty `B` is never missed. -/
theorem miss_le (K N : ℕ) (B : Finset (Fin N)) :
    prob (fun π : Equiv.Perm (Fin N) => ∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw π, v ∉ B) ≤
      (1 - (B.card : ℝ) / N) ^ K := by
  rcases B.eq_empty_or_nonempty with rfl | ⟨v, hv⟩
  · rw [Finset.card_empty, Nat.cast_zero, zero_div, sub_zero, one_pow]
    exact prob_le_one _
  have hNr : (0 : ℝ) < N := by exact_mod_cast v.pos
  have hb : B.card ≤ N := (Finset.card_le_univ B).trans_eq (Fintype.card_fin N)
  have h0 : 0 ≤ 1 - (B.card : ℝ) / N := sub_nonneg.mpr (div_le_one_of_le₀ (by exact_mod_cast hb) hNr.le)
  rcases le_or_gt K N with hK | hK
  · rw [miss_eq K N B hK]
    have hC : (0 : ℝ) < N.choose K := by exact_mod_cast Nat.choose_pos hK
    have hsub : 1 - (B.card : ℝ) / N = ((N - B.card : ℕ) : ℝ) / N := by
      rw [Nat.cast_sub hb, sub_div, div_self hNr.ne']
    have key := Spot.Sampling.SamplingDesign.choose_sub_mul_pow_le' N B.card K
    rw [hsub, div_pow, div_le_iff₀ hC, div_mul_eq_mul_div, le_div_iff₀ (pow_pos hNr K),
      mul_comm (((N - B.card : ℕ) : ℝ) ^ K)]
    exact_mod_cast key
  · refine (prob_mono (q := fun _ : Equiv.Perm (Fin N) => False) fun π hπ => ?_).trans
      ((le_of_eq (by unfold prob Claims.prUniform; simp)).trans (pow_nonneg h0 K))
    have := (forall_mem_permPrefix π _).mp hπ (π.symm v) (by have := (π.symm v).isLt; omega)
    rw [Equiv.apply_symm_apply] at this
    exact this hv

/-! ## The stages -/

section Stages

variable {C V : Type}

/-- A tile's weight where its mode is folded. -/
def inF (w : V → ℝ) (md : Mode) (_q : Prop) (v : V) : ℝ := if md = .folded then w v else 0

/-- A tile's weight where its mode is whole. -/
def in0 (w : V → ℝ) (md : Mode) (_q : Prop) (v : V) : ℝ := if md = .whole then w v else 0

open Classical in
/-- A tile's weight where its mode is folded and its checked values are honest. -/
noncomputable def luF (w : V → ℝ) (md : Mode) (q : Prop) (v : V) : ℝ := if md = .folded ∧ q then w v else 0

open Classical in
/-- A tile's weight where its mode is whole and its checked values are honest. -/
noncomputable def lu0 (w : V → ℝ) (md : Mode) (q : Prop) (v : V) : ℝ := if md = .whole ∧ q then w v else 0

open Classical in
/-- A tile's weight where it is skipped (folded or whole) and its checked values aren't honest. -/
noncomputable def un (w : V → ℝ) (md : Mode) (q : Prop) (v : V) : ℝ :=
  if (md = .folded ∨ md = .whole) ∧ ¬q then w v else 0

/-- The window's sum of `f` over the tiles at their modes and checked values along the stages' coins. -/
noncomputable def pathSum [Fintype V] (stg : V → ℕ) {m : ℕ} (T : SplitStages C V m) (x : Paths C m)
    (f : Mode → Prop → V → ℝ) : ℝ :=
  ∑ v, f (T.modeAt stg x v) (T.sameAt stg x v) v

/-- Stage `m`'s sum of `f` over its tiles at their modes and checked values at the stage's coin `c`. -/
noncomputable def stageSum [Fintype V] (stg : V → ℕ) (m : ℕ) (mode : C → V → Mode) (same : C → V → Prop)
    (f : Mode → Prop → V → ℝ) (c : C) : ℝ :=
  ∑ v, if stg v = m then f (mode c v) (same c v) v else 0

/-- `SplitRateAt`'s left side at folded and whole work `WF`, `W0` and their lucky parts `LF`, `L0`. -/
noncomputable def gain (K : ℕ) (ρ a b Λ WF W0 LF L0 : ℝ) : ℝ :=
  Real.exp ((a * WF + b * W0) / Λ) * (1 - ρ * ((WF - LF) + (W0 - L0))) ^ K

/-- The product over the stages of `gain` at each stage's work along the stages' coins. -/
noncomputable def factor [Fintype V] (stg : V → ℕ) (w : V → ℝ) (K : ℕ) (ρ a b Λ : ℝ) :
    {m : ℕ} → SplitStages C V m → Paths C m → ℝ
  | _, .done, _ => 1
  | _, .stage (m := m) mode _ same next, x =>
    gain K ρ a b Λ (stageSum stg m mode same (inF w) x.1) (stageSum stg m mode same (in0 w) x.1)
        (stageSum stg m mode same (luF w) x.1) (stageSum stg m mode same (lu0 w) x.1) *
      factor stg w K ρ a b Λ (next x.1) x.2

theorem modeAt_of_le (stg : V → ℕ) : ∀ {m : ℕ} (T : SplitStages C V m) (x : Paths C m) (v : V), m ≤ stg v →
    T.modeAt stg x v = .paid
  | _, .done, _, _, _ => rfl
  | _, .stage (m := m) _ _ _ next, x, v, h => by
    simp only [SplitStages.modeAt]
    rw [ite_eq_right (by omega)]
    exact modeAt_of_le stg (next x.1) x.2 v (by omega)

theorem pathSum_done [Fintype V] (stg : V → ℕ) (x : Paths C 0) (f : Mode → Prop → V → ℝ)
    (hf : ∀ q v, f .paid q v = 0) : pathSum stg (.done : SplitStages C V 0) x f = 0 :=
  Finset.sum_eq_zero fun v _ => hf _ v

theorem pathSum_stage [Fintype V] (stg : V → ℕ) {m : ℕ} (mode : C → V → Mode) (right same : C → V → Prop)
    (next : C → SplitStages C V m) (x : Paths C (m + 1)) (f : Mode → Prop → V → ℝ)
    (hf : ∀ q v, f .paid q v = 0) :
    pathSum stg (.stage mode right same next) x f =
      stageSum stg m mode same f x.1 + pathSum stg (next x.1) x.2 f := by
  unfold pathSum stageSum
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun v _ => ?_
  simp only [SplitStages.modeAt, SplitStages.sameAt]
  by_cases h : stg v = m
  · rw [ite_eq_left h, ite_eq_left h, ite_eq_left h, modeAt_of_le stg (next x.1) x.2 v h.ge, hf, add_zero]
  · rw [ite_eq_right h, ite_eq_right h, ite_eq_right h, zero_add]

theorem inF_paid (w : V → ℝ) (q : Prop) (v : V) : inF w .paid q v = 0 := by simp [inF]
theorem in0_paid (w : V → ℝ) (q : Prop) (v : V) : in0 w .paid q v = 0 := by simp [in0]
theorem un_paid (w : V → ℝ) (q : Prop) (v : V) : un w .paid q v = 0 := by simp [un]

theorem un_eq (w : V → ℝ) (md : Mode) (q : Prop) (v : V) :
    un w md q v = (inF w md q v - luF w md q v) + (in0 w md q v - lu0 w md q v) := by
  by_cases hq : q <;> cases md <;> simp [un, inF, in0, luF, lu0, hq]

theorem un_le (w : V → ℝ) (hw : ∀ v, 0 ≤ w v) (md : Mode) (q : Prop) (v : V) : un w md q v ≤ w v := by
  unfold un; split_ifs <;> simp [hw v]

theorem un_nonneg (w : V → ℝ) (hw : ∀ v, 0 ≤ w v) (md : Mode) (q : Prop) (v : V) : 0 ≤ un w md q v := by
  unfold un; split_ifs <;> simp [hw v]

theorem stageSum_un [Fintype V] (stg : V → ℕ) (m : ℕ) (mode : C → V → Mode) (same : C → V → Prop)
    (w : V → ℝ) (c : C) :
    stageSum stg m mode same (un w) c = (stageSum stg m mode same (inF w) c - stageSum stg m mode same (luF w) c) +
      (stageSum stg m mode same (in0 w) c - stageSum stg m mode same (lu0 w) c) := by
  unfold stageSum
  rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun v _ => ?_
  by_cases h : stg v = m
  · simp only [ite_eq_left h]; exact un_eq w _ _ v
  · simp only [ite_eq_right h]; ring

theorem stageSum_le_total [Fintype V] (stg : V → ℕ) (m : ℕ) (mode : C → V → Mode) (same : C → V → Prop)
    {w : V → ℝ} (hw : ∀ v, 0 ≤ w v) {f : Mode → Prop → V → ℝ} (hf : ∀ md q v, f md q v ≤ w v) (c : C) :
    stageSum stg m mode same f c ≤ ∑ v, w v := by
  refine Finset.sum_le_sum fun v _ => ?_
  split_ifs
  · exact hf _ _ v
  · exact hw v

theorem stageSum_nonneg [Fintype V] (stg : V → ℕ) (m : ℕ) (mode : C → V → Mode) (same : C → V → Prop)
    {f : Mode → Prop → V → ℝ} (hf : ∀ md q v, 0 ≤ f md q v) (c : C) : 0 ≤ stageSum stg m mode same f c := by
  refine Finset.sum_nonneg fun v _ => ?_
  split_ifs
  · exact hf _ _ v
  · exact le_rfl

theorem stageSum_mono [Fintype V] (stg : V → ℕ) (m : ℕ) (mode : C → V → Mode) (same : C → V → Prop)
    {f g : Mode → Prop → V → ℝ} (hfg : ∀ md q v, f md q v ≤ g md q v) (c : C) :
    stageSum stg m mode same f c ≤ stageSum stg m mode same g c := by
  refine Finset.sum_le_sum fun v _ => ?_
  split_ifs
  · exact hfg _ _ v
  · exact le_rfl

theorem gain_nonneg (K : ℕ) {ρ a b Λ WF W0 LF L0 : ℝ} (h : ρ * ((WF - LF) + (W0 - L0)) ≤ 1) :
    0 ≤ gain K ρ a b Λ WF W0 LF L0 :=
  mul_nonneg (Real.exp_pos _).le (pow_nonneg (sub_nonneg.mpr h) K)

/-- Each stage's unlucky work is at most the tiles' total weight, so each stage's `gain` is `≥ 0`. -/
theorem stage_un_le [Fintype V] (stg : V → ℕ) (m : ℕ) (mode : C → V → Mode) (same : C → V → Prop)
    {w : V → ℝ} (hw : ∀ v, 0 ≤ w v) {ρ : ℝ} (hρ : 0 ≤ ρ) (htot : ρ * ∑ v, w v ≤ 1) (c : C) :
    ρ * ((stageSum stg m mode same (inF w) c - stageSum stg m mode same (luF w) c) +
      (stageSum stg m mode same (in0 w) c - stageSum stg m mode same (lu0 w) c)) ≤ 1 := by
  rw [← stageSum_un]
  exact (mul_le_mul_of_nonneg_left (stageSum_le_total stg m mode same hw (un_le w hw) c) hρ).trans htot

theorem factor_nonneg [Fintype V] (stg : V → ℕ) {w : V → ℝ} (hw : ∀ v, 0 ≤ w v) (K : ℕ) {ρ : ℝ} (hρ : 0 ≤ ρ)
    (htot : ρ * ∑ v, w v ≤ 1) (a b Λ : ℝ) :
    ∀ {m : ℕ} (T : SplitStages C V m) (x : Paths C m), 0 ≤ factor stg w K ρ a b Λ T x
  | _, .done, _ => zero_le_one
  | _, .stage (m := m) mode _ same next, x =>
    mul_nonneg (gain_nonneg K (stage_un_le stg m mode same hw hρ htot x.1))
      (factor_nonneg stg hw K hρ htot a b Λ (next x.1) x.2)

/-- **One path of the stages' coins**: at prior unlucky work `U ≥ 0`, `exp(d/Λ)·(1 − ρ(U + U_T))^K` is at most
`(1 − ρU)^K` times the stages' `factor`, `d` the cost of the stages' skipped work and `U_T` their unlucky work. -/
theorem path_le [Fintype V] (stg : V → ℕ) {w : V → ℝ} (hw : ∀ v, 0 ≤ w v) (K : ℕ) {ρ : ℝ} (hρ : 0 ≤ ρ)
    (htot : ρ * ∑ v, w v ≤ 1) (a b Λ : ℝ) :
    ∀ {m : ℕ} (T : SplitStages C V m) (x : Paths C m) (U : ℝ), 0 ≤ U →
      ρ * (U + pathSum stg T x (un w)) ≤ 1 →
      Real.exp ((a * pathSum stg T x (inF w) + b * pathSum stg T x (in0 w)) / Λ) *
          (1 - ρ * (U + pathSum stg T x (un w))) ^ K ≤ (1 - ρ * U) ^ K * factor stg w K ρ a b Λ T x
  | _, .done, x, U, _, _ => by
    rw [pathSum_done stg x _ (inF_paid w), pathSum_done stg x _ (in0_paid w), pathSum_done stg x _ (un_paid w)]
    simp [factor]
  | _, .stage (m := m) mode right same next, x, U, hU, hρU => by
    rw [pathSum_stage stg mode right same next x _ (inF_paid w),
      pathSum_stage stg mode right same next x _ (in0_paid w),
      pathSum_stage stg mode right same next x _ (un_paid w)] at *
    set WF := stageSum stg m mode same (inF w) x.1
    set W0 := stageSum stg m mode same (in0 w) x.1
    set Us := stageSum stg m mode same (un w) x.1 with hUs
    set nF := pathSum stg (next x.1) x.2 (inF w)
    set n0 := pathSum stg (next x.1) x.2 (in0 w)
    set nU := pathSum stg (next x.1) x.2 (un w)
    have hUs0 : 0 ≤ Us := stageSum_nonneg stg m mode same (un_nonneg w hw) x.1
    have hnU0 : 0 ≤ nU := Finset.sum_nonneg fun v _ => un_nonneg w hw _ _ v
    have ih := path_le stg hw K hρ htot a b Λ (next x.1) x.2 (U + Us) (add_nonneg hU hUs0)
      (by linarith)
    have hF := factor_nonneg stg hw K hρ htot a b Λ (next x.1) x.2
    have h1 : 0 ≤ 1 - ρ * (U + Us) := by nlinarith
    have h2 : 1 - ρ * (U + Us) ≤ (1 - ρ * U) * (1 - ρ * Us) := by nlinarith [mul_nonneg (mul_nonneg hρ hρ) (mul_nonneg hU hUs0)]
    have hpow : (1 - ρ * (U + Us)) ^ K ≤ (1 - ρ * U) ^ K * (1 - ρ * Us) ^ K := by
      rw [← mul_pow]; exact pow_le_pow_left₀ h1 h2 K
    have hexp : Real.exp ((a * (WF + nF) + b * (W0 + n0)) / Λ) =
        Real.exp ((a * WF + b * W0) / Λ) * Real.exp ((a * nF + b * n0) / Λ) := by
      rw [← Real.exp_add, ← add_div]; ring_nf
    have hg : gain K ρ a b Λ WF W0 (stageSum stg m mode same (luF w) x.1) (stageSum stg m mode same (lu0 w) x.1) =
        Real.exp ((a * WF + b * W0) / Λ) * (1 - ρ * Us) ^ K := by
      rw [gain, hUs, stageSum_un]
    show _ ≤ (1 - ρ * U) ^ K * (gain K ρ a b Λ WF W0 _ _ * factor stg w K ρ a b Λ (next x.1) x.2)
    rw [hg, hexp]
    have hE := (Real.exp_pos ((a * WF + b * W0) / Λ)).le
    calc Real.exp ((a * WF + b * W0) / Λ) * Real.exp ((a * nF + b * n0) / Λ) *
          (1 - ρ * (U + (Us + nU))) ^ K
        = Real.exp ((a * WF + b * W0) / Λ) *
            (Real.exp ((a * nF + b * n0) / Λ) * (1 - ρ * (U + Us + nU)) ^ K) := by ring_nf
      _ ≤ Real.exp ((a * WF + b * W0) / Λ) *
            ((1 - ρ * (U + Us)) ^ K * factor stg w K ρ a b Λ (next x.1) x.2) :=
          mul_le_mul_of_nonneg_left ih hE
      _ ≤ Real.exp ((a * WF + b * W0) / Λ) *
            ((1 - ρ * U) ^ K * (1 - ρ * Us) ^ K * factor stg w K ρ a b Λ (next x.1) x.2) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hpow hF) hE
      _ = (1 - ρ * U) ^ K * (Real.exp ((a * WF + b * W0) / Λ) * (1 - ρ * Us) ^ K *
            factor stg w K ρ a b Λ (next x.1) x.2) := by ring

/-- The mean over the stage's coin of its tiles' weight on an event is its tiles' weighted probabilities. -/
theorem expect_stageSum [Fintype C] [Fintype V] (stg : V → ℕ) (m : ℕ) (mode : C → V → Mode)
    (same : C → V → Prop) (f : Mode → Prop → V → ℝ) (w : V → ℝ) (P : C → V → Prop) [∀ c v, Decidable (P c v)]
    (hf : ∀ c v, f (mode c v) (same c v) v = if P c v then w v else 0) :
    expect (stageSum stg m mode same f) = ∑ v, if stg v = m then w v * prob (fun c => P c v) else 0 := by
  unfold stageSum
  rw [expect_sum]
  refine Finset.sum_congr rfl fun v _ => ?_
  by_cases h : stg v = m
  · simp only [ite_eq_left h, hf]
    rw [prob_eq_expect, ← expect_mul_left]
    exact congrArg expect (funext fun c => by by_cases hP : P c v <;> simp [hP])
  · simp only [ite_eq_right h]
    unfold expect
    simp

/-- The stage's lucky work in expectation: at most `p` times its work, under the per-tile luck. -/
theorem expect_lucky_le [Fintype C] [Fintype V] (stg : V → ℕ) (m : ℕ) {w lk : V → ℝ} (hw : ∀ v, 0 ≤ w v)
    {p : ℝ} (hlk : ∀ v, 0 < w v → lk v ≤ p) (Fd S : C → V → Prop) [∀ c v, Decidable (Fd c v)]
    [∀ c v, Decidable (S c v)]
    (hL : ∀ v, stg v = m → prob (fun c => Fd c v ∧ S c v) ≤ lk v * prob (fun c => Fd c v)) :
    (∑ v, if stg v = m then w v * prob (fun c => Fd c v ∧ S c v) else 0) ≤
      p * ∑ v, if stg v = m then w v * prob (fun c => Fd c v) else 0 := by
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun v _ => ?_
  by_cases h : stg v = m
  · simp only [ite_eq_left h]
    calc w v * prob (fun c => Fd c v ∧ S c v) ≤ w v * (lk v * prob (fun c => Fd c v)) :=
          mul_le_mul_of_nonneg_left (hL v h) (hw v)
      _ ≤ p * (w v * prob (fun c => Fd c v)) := by
          rcases (hw v).eq_or_lt with h0 | h0
          · rw [← h0]; simp
          · have := mul_le_mul_of_nonneg_right (hlk v h0) (prob_nonneg (fun c => Fd c v))
            nlinarith
  · simp only [ite_eq_right h, mul_zero, le_refl]

theorem expect_affine [Fintype C] [Nonempty C] (ν ν0 p p0 : ℝ) (f g f' g' : C → ℝ) :
    expect (fun c => 1 + ν * (f c - p * g c) + ν0 * (f' c - p0 * g' c)) =
      1 + ν * (expect f - p * expect g) + ν0 * (expect f' - p0 * expect g') := by
  unfold expect
  have hC : (Fintype.card C : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul, mul_one]
  field_simp

/-- **Backward induction over the stages**: the stages' `factor` has mean at most `1`. -/
theorem expect_factor_le [Fintype C] [Fintype V] (stg : V → ℕ) {w lk lk0 : V → ℝ} (hw : ∀ v, 0 ≤ w v) (K : ℕ)
    {ρ p p0 a b wmax Λ ν ν0 : ℝ} (hρ : 0 ≤ ρ) (htot : ρ * ∑ v, w v ≤ 1)
    (hst : ∀ j, stageShare stg w j ≤ wmax) (hlk : ∀ v, 0 < w v → lk v ≤ p ∧ lk0 v ≤ p0)
    (hSR : SplitRateAt K ρ p p0 a b wmax Λ ν ν0) :
    ∀ {m : ℕ} (T : SplitStages C V m), T.LuckAt lk lk0 stg → expect (factor stg w K ρ a b Λ T) ≤ 1
  | _, .done, _ => expect_le_of_le zero_le_one fun _ => le_rfl
  | _, .stage (m := m) mode right same next, hL => by
    classical
    obtain ⟨hL, hLn⟩ := hL
    rcases isEmpty_or_nonempty C with hC | hC
    · have : IsEmpty (Paths C (m + 1)) := inferInstanceAs (IsEmpty (C × Paths C m))
      rw [expect_empty]
      exact zero_le_one
    obtain ⟨hν, hν0, hSR⟩ := hSR
    set WF := stageSum stg m mode same (inF w)
    set W0 := stageSum stg m mode same (in0 w)
    set LF := stageSum stg m mode same (luF w)
    set L0 := stageSum stg m mode same (lu0 w)
    have hg : ∀ c, 0 ≤ gain K ρ a b Λ (WF c) (W0 c) (LF c) (L0 c) := fun c =>
      gain_nonneg K (stage_un_le stg m mode same hw hρ htot c)
    rw [expect_prod (fun x : C × Paths C m => factor stg w K ρ a b Λ (.stage mode right same next) x)]
    have hstep : ∀ c, (expect fun x' : Paths C m =>
        factor stg w K ρ a b Λ (.stage mode right same next) (c, x')) ≤
          1 + ν * (LF c - p * WF c) + ν0 * (L0 c - p0 * W0 c) := by
      intro c
      show (expect fun x' => gain K ρ a b Λ (WF c) (W0 c) (LF c) (L0 c) * factor stg w K ρ a b Λ (next c) x') ≤ _
      rw [expect_mul_left]
      refine (mul_le_of_le_one_right (hg c) (expect_factor_le stg hw K hρ htot hst hlk ⟨hν, hν0, hSR⟩ (next c)
        (hLn c))).trans ?_
      have hpw : ∀ md q v, inF w md q v + in0 w md q v ≤ w v := by
        intro md q v; cases md <;> simp [inF, in0, hw v]
      refine hSR _ _ _ _ (stageSum_nonneg stg m mode same (fun md q v => ?_) c)
        (stageSum_mono stg m mode same (fun md q v => ?_) c) (stageSum_nonneg stg m mode same (fun md q v => ?_) c)
        (stageSum_mono stg m mode same (fun md q v => ?_) c) ?_
      · unfold luF; split_ifs <;> simp [hw v]
      · unfold luF inF; split_ifs <;> simp_all
      · unfold lu0; split_ifs <;> simp [hw v]
      · unfold lu0 in0; split_ifs <;> simp_all
      · refine le_trans ?_ (hst m)
        show stageSum stg m mode same (inF w) c + stageSum stg m mode same (in0 w) c ≤
          ∑ v, if stg v = m then w v else 0
        unfold stageSum
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_le_sum fun v _ => ?_
        split_ifs
        · exact hpw _ _ v
        · simp
    refine (expect_mono hstep).trans ?_
    rw [expect_affine]
    have eWF := expect_stageSum stg m mode same (inF w) w (fun c v => mode c v = .folded) fun c v => rfl
    have eW0 := expect_stageSum stg m mode same (in0 w) w (fun c v => mode c v = .whole) fun c v => rfl
    have eLF := expect_stageSum stg m mode same (luF w) w (fun c v => mode c v = .folded ∧ same c v)
      fun c v => by unfold luF; split_ifs <;> simp_all
    have eL0 := expect_stageSum stg m mode same (lu0 w) w (fun c v => mode c v = .whole ∧ same c v)
      fun c v => by unfold lu0; split_ifs <;> simp_all
    have hF := expect_lucky_le stg m hw (fun v h => (hlk v h).1) (fun c v => mode c v = .folded) same
      fun v h => (hL v h).1
    have h0 := expect_lucky_le stg m hw (fun v h => (hlk v h).2) (fun c v => mode c v = .whole) same
      fun v h => (hL v h).2
    rw [← eLF, ← eWF] at hF
    rw [← eL0, ← eW0] at h0
    nlinarith [mul_nonpos_of_nonneg_of_nonpos hν (sub_nonpos.mpr hF),
      mul_nonpos_of_nonneg_of_nonpos hν0 (sub_nonpos.mpr h0)]

end Stages

/-! ## The window -/

/-- Tiles of weight at least `ρ` times their share of the `N`: `ρ` times the weight of any set `B` of them is at most
`|B|/N`. -/
theorem weight_le_card {N : ℕ} {w : Fin N → ℝ} {ρ : ℝ} (hρN : ∀ v, ρ * N * w v ≤ 1) (B : Finset (Fin N)) :
    ρ * ∑ v ∈ B, w v ≤ (B.card : ℝ) / N := by
  rw [Finset.mul_sum, div_eq_mul_one_div, ← nsmul_eq_mul, ← Finset.sum_const]
  refine Finset.sum_le_sum fun v _ => ?_
  have hN : (0 : ℝ) < N := by exact_mod_cast v.pos
  rw [le_div_iff₀ hN]
  linarith [hρN v]

theorem serviceBudget : ResourceAccounting.Guarantees.Compose.ServiceBudget := by
  intro C Ξ _ _ N M stg W w lk lk0 K ρ p p0 a b wmax Λ hw0 hρ hρN hst hlk hL ha hb hSR
  classical
  obtain ⟨ν, ν0, hSR⟩ := hSR
  have hshare : ∀ x s, 0 ≤ (W.joint stg).share w x s := fun x s =>
    Finset.sum_nonneg fun v _ => by split_ifs <;> simp [hw0 v]
  have htot : ρ * ∑ v, w v ≤ 1 := by
    have := weight_le_card hρN Finset.univ
    rw [Finset.card_univ, Fintype.card_fin] at this
    exact this.trans (div_self_le_one _)
  refine ⟨fun ω => add_nonneg (mul_nonneg ha (hshare _ _)) (mul_nonneg hb (hshare _ _)), ?_, fun _ => trivial⟩
  apply sum_le_card_of_expect
  rcases isEmpty_or_nonempty Ξ with hΞ | hΞ
  · rw [expect_empty]
    exact zero_le_one
  rw [expect_prod]
  refine (expect_mono fun x => ?_).trans
    (expect_factor_le stg hw0 K hρ.le htot hst hlk ⟨hSR.1, hSR.2.1, hSR.2.2⟩ W hL)
  set B : Finset (Fin N) := Finset.univ.filter fun v => ¬ W.sameAt stg x v
  set e := Real.exp ((a * (W.joint stg).share w x .folded + b * (W.joint stg).share w x .whole) / Λ)
  have he0 : 0 ≤ e := (Real.exp_pos _).le
  have hE : (expect fun β : Equiv.Perm (Fin N) × Ξ =>
      (if (∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw β.1, (W.joint stg).same x v) then e else 0)) ≤
      e * prob (fun π : Equiv.Perm (Fin N) => ∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw π, v ∉ B) := by
    rw [← prob_fst (β := Ξ), prob_eq_expect, ← expect_mul_left]
    refine expect_mono fun β => ?_
    by_cases h : ∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw β.1, (W.joint stg).same x v
    · have h' : ∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw β.1, v ∉ B := fun v hv hB =>
        (Finset.mem_filter.mp hB).2 (h v hv)
      rw [ite_eq_left h, ite_eq_left h', mul_one]
    · rw [ite_eq_right h]
      split_ifs
      · rw [mul_one]; exact he0
      · rw [mul_zero]
  have hU : ρ * pathSum stg W x (un w) ≤ (B.card : ℝ) / N := by
    refine le_trans (mul_le_mul_of_nonneg_left ?_ hρ.le) (weight_le_card hρN B)
    rw [Finset.sum_filter]
    refine Finset.sum_le_sum fun v _ => ?_
    unfold un
    split_ifs <;> simp_all
  have hB1 : (B.card : ℝ) / N ≤ 1 :=
    div_le_one_of_le₀ (by exact_mod_cast (Finset.card_le_univ B).trans_eq (Fintype.card_fin N))
      (Nat.cast_nonneg _)
  have hpath := path_le stg hw0 K hρ.le htot a b Λ W x 0 le_rfl
    (by rw [zero_add]; exact hU.trans hB1)
  rw [zero_add, mul_zero, sub_zero, one_pow, one_mul] at hpath
  refine le_trans ?_ hpath
  refine le_trans (le_trans (expect_mono fun β => ?_) hE) ?_
  · dsimp only
    by_cases h : ∀ v ∈ (Verity.SamplingDesign.permPrefix N K).draw β.1, (W.joint stg).same x v
    · rw [ite_eq_left h, ite_eq_left h]
    · rw [ite_eq_right h, ite_eq_right h]
  · refine mul_le_mul_of_nonneg_left ((miss_le K N B).trans (pow_le_pow_left₀ (sub_nonneg.mpr hB1) ?_ K)) he0
    linarith

end ResourceAccounting.SecurityProofs.Compose.Budget
