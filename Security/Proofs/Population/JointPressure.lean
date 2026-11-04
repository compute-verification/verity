import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Proofs.Population.ExfiltrationBound

/-!
# One draw for several claims: the max rule and the union bound

Several claims read one deployment through one draw: PoUW's credited work, the end-to-end skipped work, exfiltration
(the egress slots a window changes), each with its own harm `harm i u` for a faulty proof block `u`. The proof service
draws once a window, after the window's commitments, and halts serving at the first failed proof; every claim reads that
one escape set and that one halt, each with its own leak.

* `le_maxRule`, `maxRule_le`: the max rule `a u = max_i κ_i harm_i(u)` meets every claim's inclusion bound
  `κ_i harm_i(u) ≤ a u`, and is the least inclusion that does.
* `pressure_maxRule`: a draw whose catch probabilities are at least the max rule gives every claim its detection
  pressure `κ_i` (`pressure_of_draw` for each claim).
* `prob_exists_le`: the union bound for `prob`.
* `joint_tail`, `joint_tail_delta`: every claim's bound through the first detection holds at once except with
  probability at most the sum of the claims' tails, `Σ_i δ_i` at `x_i = ℓ_i + ln (1/δ_i) / κ_i`. No independence
  between the claims is needed.
* `claimCum`, `joint_tail_withheld`: the same with each claim counted as it is paid out. A claim whose count waits for
  the verdict (credited work, paid at the certificate) is bounded by `tail_cumPrev`, with no per-window term; egress
  that leaves before its verdict keeps `ℓ_i`.
* `joint_of_maxRule`: the two together, from the draw's catch probabilities to every claim's bound.
* `subsetDraw_escape`, `subsetDraw_escape_le`, `pressure_subsetDraw`: a uniform `k`-subset of `N` units misses `b` given
  units with probability `C(N - b, k) / C(N, k) ≤ (1 - k/N)^b`, so a stratum drawn as a uniform `k`-subset meets
  `pressure_of_draw`'s escape premise with `a u = k / N`.
-/

namespace Population.Exfil

open Finset

/-! ## The max rule -/

/-- **The max rule**: the inclusion `max_i κ_i harm_i(u)` over the claims (at least `0`). -/
noncomputable def maxRule {I U : Type} (claims : Finset I) (κ : I → ℝ) (harm : I → U → ℝ) (u : U) : ℝ :=
  claims.fold max 0 fun i => κ i * harm i u

/-- The max rule meets every claim's inclusion bound. -/
theorem le_maxRule {I U : Type} {claims : Finset I} {i : I} (hi : i ∈ claims) (κ : I → ℝ) (harm : I → U → ℝ)
    (u : U) : κ i * harm i u ≤ maxRule claims κ harm u :=
  (Finset.le_fold_max _).2 (Or.inr ⟨i, hi, le_refl _⟩)

/-- The max rule is the least nonnegative inclusion that meets every claim's bound. -/
theorem maxRule_le {I U : Type} {claims : Finset I} {κ : I → ℝ} {harm : I → U → ℝ} {a : U → ℝ} (u : U)
    (h0 : 0 ≤ a u) (h : ∀ i ∈ claims, κ i * harm i u ≤ a u) : maxRule claims κ harm u ≤ a u :=
  (Finset.fold_max_le _).2 ⟨h0, h⟩

/-- **One draw gives every claim its pressure.** Each claim `i` reads the window's one escape set `esc h` with its own
leak `leak i h`, at most its harm over the window's wrong set `B h`. A draw whose window escapes with probability at
most `Π_{u ∈ B h} (1 - a u)`, with catch probabilities `a u` at least the max rule, meets `DetectionPressure (κ i)` for
every claim. -/
theorem pressure_maxRule {Ω U I : Type} [Fintype Ω] {μ : Ω → ℝ} {claims : Finset I} {κ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 ≤ κ i) (harm : I → U → ℝ) (leak : I → List Ω → ℝ) (esc : List Ω → Finset Ω)
    (B : List Ω → Finset U) (a : U → ℝ) (ha : ∀ u, 0 ≤ a u ∧ a u ≤ 1) (hmax : ∀ u, maxRule claims κ harm u ≤ a u)
    (hleak : ∀ i ∈ claims, ∀ h, leak i h ≤ ∑ u ∈ B h, harm i u)
    (hesc : ∀ h, ∑ c ∈ esc h, μ c ≤ ∏ u ∈ B h, (1 - a u)) :
    ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩ := fun i hi =>
  pressure_of_draw (hκ i hi) ⟨leak i, esc⟩ B (harm i) a ha
    (fun u => (le_maxRule hi κ harm u).trans (hmax u)) (hleak i hi) hesc

/-! ## The union bound -/

variable {Ω : Type} [Fintype Ω] {μ : Ω → ℝ}

open Classical in
/-- **The union bound**: some event of a family happens with probability at most the sum of their probabilities. -/
theorem prob_exists_le {I : Type} (hμ : ∀ c, 0 ≤ μ c) (n : ℕ) (s : Finset I) (E : I → (Fin n → Ω) → Prop) :
    prob μ n (fun cs => ∃ i ∈ s, E i cs) ≤ ∑ i ∈ s, prob μ n (E i) := by
  unfold prob
  calc ex μ n _ ≤ ex μ n (fun cs => ∑ i ∈ s, if E i cs then (1 : ℝ) else 0) := ex_mono hμ fun cs => by
          split_ifs with h
          · obtain ⟨i, hi, he⟩ := h
            refine le_trans (le_of_eq ?_) (single_le_sum (f := fun j => if E j cs then (1 : ℝ) else 0)
              (fun j _ => by split_ifs <;> norm_num) hi)
            simp [he]
          · exact sum_nonneg fun j _ => by split_ifs <;> norm_num
    _ = ∑ i ∈ s, ex μ n (fun cs => if E i cs then (1 : ℝ) else 0) := by
        simp only [ex, mul_sum]
        exact sum_comm

/-! ## Every claim at once -/

/-- **Every claim's bound through the first detection, at once.** Under one draw and one halt, each claim with
detection pressure `κ i ≥ 0` and per-window leak at most `ℓ i` keeps what it leaks through the first detection below
`x i`, all together, except with probability at most `Σ_i exp (-κ_i (x_i - ℓ_i))`. -/
theorem joint_tail {I : Type} (hμ : IsDist μ) [DecidableEq Ω] {claims : Finset I} {κ ℓ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 ≤ κ i) (leak : I → List Ω → ℝ) (esc : List Ω → Finset Ω)
    (hP : ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩)
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) (x : I → ℝ) :
    prob μ n (fun cs => ∃ i ∈ claims, x i ≤ Strategy.cum ⟨leak i, esc⟩ n h cs)
      ≤ ∑ i ∈ claims, Real.exp (-(κ i * (x i - ℓ i))) :=
  (prob_exists_le hμ.1 n claims _).trans
    (sum_le_sum fun i hi => tail_cum hμ (hκ i hi) (hP i hi) (hL i hi) n h (x i))

/-- **The δ form**: with `κ i > 0`, every claim leaks at most `ℓ_i + ln (1/δ_i) / κ_i` through the first detection,
all together, except with probability at most `Σ_i δ_i`. -/
theorem joint_tail_delta {I : Type} (hμ : IsDist μ) [DecidableEq Ω] {claims : Finset I} {κ ℓ δ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 < κ i) (hδ : ∀ i ∈ claims, 0 < δ i) (leak : I → List Ω → ℝ) (esc : List Ω → Finset Ω)
    (hP : ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩)
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) :
    prob μ n (fun cs => ∃ i ∈ claims, ℓ i + Real.log (1 / δ i) / κ i ≤ Strategy.cum ⟨leak i, esc⟩ n h cs)
      ≤ ∑ i ∈ claims, δ i := by
  refine (joint_tail hμ (fun i hi => (hκ i hi).le) leak esc hP hL n h _).trans (le_of_eq (sum_congr rfl ?_))
  intro i hi
  have hk := (hκ i hi).ne'
  rw [show -(κ i * (ℓ i + Real.log (1 / δ i) / κ i - ℓ i)) = Real.log (δ i) by
    rw [one_div, Real.log_inv]; field_simp; ring]
  exact Real.exp_log (hδ i hi)

/-- What claim `i` counts through the first detection: when what it counts waits for the verdict (`withheld i`, as
credited work waits for its certificate), what leaves before the first detection; otherwise (egress that leaves before
its verdict) what leaves up to and including it. -/
noncomputable def claimCum [DecidableEq Ω] {I : Type} (withheld : I → Prop) [DecidablePred withheld]
    (leak : I → List Ω → ℝ)
    (esc : List Ω → Finset Ω) (i : I) (n : ℕ) (h : List Ω) (cs : Fin n → Ω) : ℝ :=
  if withheld i then Strategy.cumPrev ⟨leak i, esc⟩ n h cs else Strategy.cum ⟨leak i, esc⟩ n h cs

/-- **Every claim at once, a withheld claim without the per-window term**: as `joint_tail`, with a claim whose count
waits for the verdict bounded by `tail_cumPrev`, so its tail is `exp (-κ_i x_i)`. -/
theorem joint_tail_withheld {I : Type} (hμ : IsDist μ) [DecidableEq Ω] {claims : Finset I} {κ ℓ : I → ℝ}
    (withheld : I → Prop) [DecidablePred withheld] (hκ : ∀ i ∈ claims, 0 ≤ κ i) (leak : I → List Ω → ℝ)
    (esc : List Ω → Finset Ω) (hP : ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩)
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) (x : I → ℝ) :
    prob μ n (fun cs => ∃ i ∈ claims, x i ≤ claimCum withheld leak esc i n h cs)
      ≤ ∑ i ∈ claims, Real.exp (-(κ i * (x i - if withheld i then 0 else ℓ i))) :=
  (prob_exists_le hμ.1 n claims _).trans (sum_le_sum fun i hi => by
    by_cases hw : withheld i
    · simp only [claimCum, hw, ↓reduceIte, sub_zero]
      exact tail_cumPrev hμ (hκ i hi) (hP i hi) n h (x i)
    · simp only [claimCum, hw, ↓reduceIte]
      exact tail_cum hμ (hκ i hi) (hP i hi) (hL i hi) n h (x i))

/-- **One draw certifies every claim**: catch probabilities at least the max rule, a window escaping with probability at
most `Π_{u ∈ B h} (1 - a u)`, and each claim's leak at most its harm over the wrong set and at most `ℓ_i` a window. Then
every claim leaks at most `ℓ_i + ln (1/δ_i) / κ_i` through the first detection, all together, except with probability
at most `Σ_i δ_i`. -/
theorem joint_of_maxRule {I U : Type} (hμ : IsDist μ) [DecidableEq Ω] {claims : Finset I} {κ ℓ δ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 < κ i) (hδ : ∀ i ∈ claims, 0 < δ i) (harm : I → U → ℝ) (leak : I → List Ω → ℝ)
    (esc : List Ω → Finset Ω) (B : List Ω → Finset U) (a : U → ℝ) (ha : ∀ u, 0 ≤ a u ∧ a u ≤ 1)
    (hmax : ∀ u, maxRule claims κ harm u ≤ a u) (hleak : ∀ i ∈ claims, ∀ h, leak i h ≤ ∑ u ∈ B h, harm i u)
    (hesc : ∀ h, ∑ c ∈ esc h, μ c ≤ ∏ u ∈ B h, (1 - a u))
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) :
    prob μ n (fun cs => ∃ i ∈ claims, ℓ i + Real.log (1 / δ i) / κ i ≤ Strategy.cum ⟨leak i, esc⟩ n h cs)
      ≤ ∑ i ∈ claims, δ i :=
  joint_tail_delta hμ hκ hδ leak esc
    (pressure_maxRule (fun i hi => (hκ i hi).le) harm leak esc B a ha hmax hleak hesc) hL n h

/-! ## A stratum drawn as a uniform k-subset -/

/-- The uniform draw of a `k`-subset of the units: each subset of `k` units with weight `1 / C(N, k)`. -/
noncomputable def subsetDraw (U : Type) [Fintype U] (k : ℕ) (S : Finset U) : ℝ :=
  if S.card = k then 1 / ((Fintype.card U).choose k : ℝ) else 0

/-- The subsets that miss the given units `B`. -/
def misses {U : Type} [Fintype U] [DecidableEq U] (B : Finset U) : Finset (Finset U) :=
  univ.filter fun S => Disjoint S B

/-- **A uniform `k`-subset misses `b` given units** with probability `C(N - b, k) / C(N, k)`. -/
theorem subsetDraw_escape {U : Type} [Fintype U] [DecidableEq U] (k : ℕ) (B : Finset U) :
    ∑ S ∈ misses B, subsetDraw U k S = ((Fintype.card U - B.card).choose k : ℝ) / (Fintype.card U).choose k := by
  have hset : (misses B).filter (fun S => S.card = k) = Bᶜ.powersetCard k := by
    ext S
    simp [misses, mem_powersetCard, subset_compl_iff_disjoint_right]
  unfold subsetDraw
  rw [← sum_filter, hset, sum_const, card_powersetCard, card_compl, nsmul_eq_mul, mul_one_div]

/-- `C(M - 1, k) / C(M, k) = (M - k) / M`, one unit fewer. -/
theorem choose_pred_ratio (M k : ℕ) (hM : 0 < M) :
    ((M - 1).choose k : ℝ) * M = (M.choose k : ℝ) * (M - k : ℕ) := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_lt hM
  simp only [zero_add, Nat.add_sub_cancel]
  exact_mod_cast Nat.choose_mul_succ_eq m k

/-- **The without-replacement bound**: `C(N - b, k) / C(N, k) ≤ (1 - k/N)^b`, so missing `b` given units is no
likelier than under `b` independent misses at rate `k / N`. -/
theorem choose_ratio_le (N k : ℕ) (hk : k ≤ N) (hN : 0 < N) :
    ∀ b : ℕ, ((N - b).choose k : ℝ) / N.choose k ≤ (1 - k / N) ^ b := by
  have hc : (0 : ℝ) < N.choose k := by exact_mod_cast Nat.choose_pos hk
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have h1 : (0 : ℝ) ≤ 1 - k / N := by rw [sub_nonneg, div_le_one hNr]; exact_mod_cast hk
  rcases Nat.eq_zero_or_pos k with rfl | hk0
  · intro b; simp
  intro b
  induction b with
  | zero => simp [hc.ne']
  | succ b ih =>
    rw [pow_succ]
    rcases Nat.lt_or_ge (N - (b + 1)) k with hlt | hge
    · rw [Nat.choose_eq_zero_of_lt hlt, Nat.cast_zero, zero_div]
      exact mul_nonneg (pow_nonneg h1 _) h1
    · have hM : 0 < N - b := by omega
      have hMr : (0 : ℝ) < ((N - b : ℕ) : ℝ) := by exact_mod_cast hM
      rw [show N - (b + 1) = N - b - 1 by omega]
      have hstep : ((N - b - 1).choose k : ℝ) = ((N - b).choose k : ℝ) * (((N - b - k : ℕ) : ℝ) / (N - b : ℕ)) := by
        rw [mul_div_assoc', eq_div_iff hMr.ne']
        exact choose_pred_ratio (N - b) k hM
      have hratio : (((N - b - k : ℕ) : ℝ) / (N - b : ℕ)) ≤ 1 - k / N := by
        rw [Nat.cast_sub (show k ≤ N - b by omega), sub_div, div_self hMr.ne']
        have hle : ((N - b : ℕ) : ℝ) ≤ N := by exact_mod_cast Nat.sub_le N b
        linarith [div_le_div_of_nonneg_left (Nat.cast_nonneg k : (0 : ℝ) ≤ k) hMr hle]
      rw [hstep, mul_div_right_comm]
      exact mul_le_mul ih hratio (div_nonneg (Nat.cast_nonneg _) hMr.le) (pow_nonneg h1 _)

/-- **A uniform `k`-subset escapes `b` given units with probability at most `(1 - k/N)^b`.** -/
theorem subsetDraw_escape_le {U : Type} [Fintype U] [DecidableEq U] (k : ℕ) (hk : k ≤ Fintype.card U)
    (hN : 0 < Fintype.card U) (B : Finset U) :
    ∑ S ∈ misses B, subsetDraw U k S ≤ ∏ _u ∈ B, (1 - (k : ℝ) / Fintype.card U) := by
  rw [subsetDraw_escape, prod_const]
  exact choose_ratio_le _ k hk hN B.card

/-- **A stratum drawn as a uniform `k`-subset gives detection pressure**: a strategy whose window escapes exactly when
the drawn subset misses its wrong set `B h`, and leaks at most `Σ_{u ∈ B h} E(u)`, meets detection pressure `κ` when
`κ E(u) ≤ k / N` for every unit. -/
theorem pressure_subsetDraw {U : Type} [Fintype U] [DecidableEq U] {κ : ℝ} (hκ : 0 ≤ κ) (k : ℕ)
    (hk : k ≤ Fintype.card U) (hN : 0 < Fintype.card U) (E : U → ℝ)
    (hκE : ∀ u, κ * E u ≤ (k : ℝ) / Fintype.card U) (A : Strategy (Finset U)) (B : List (Finset U) → Finset U)
    (hA : ∀ h, A.esc h = misses (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, E u) :
    DetectionPressure (subsetDraw U k) κ A := by
  have hNr : (0 : ℝ) < Fintype.card U := by exact_mod_cast hN
  refine pressure_of_draw hκ A B E (fun _ => (k : ℝ) / Fintype.card U)
    (fun _ => ⟨div_nonneg (Nat.cast_nonneg k) hNr.le, by rw [div_le_one hNr]; exact_mod_cast hk⟩) hκE hleak
    fun h => ?_
  rw [hA h]
  exact subsetDraw_escape_le k hk hN (B h)

end Population.Exfil
