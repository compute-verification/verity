import Proofs.Flock.Claims.ERep

/-!
# `E^rep`'s expected reruns, and `ERepCost`

At a tape `ω` a run accepts with probability `ε`. `E^rep` replays the judged run (one rerun), and only an accepted judged
run (probability `ε`) starts the `m` waits. A wait that stops at `k` accepted runs among `K` fresh draws makes
`N(K, a)` reruns in expectation from `a` accepted so far, and `ε · N(K, a) ≤ k − a` (`waitRuns_exp`, by induction on
the draws: `ε·N(K+1, a) = ε + ε·(ε·N(K, a+1)) + (1−ε)·(ε·N(K, a))`). So the waits add at most `ε · m · k/ε = m·k`, and
`E^rep` reruns the prover at most `1 + m·k` times in expectation (`ERep_runs_exp`), the second half of `ERepCost`
(`ERep_cost`).
-/

namespace Claims

/-! ## Averages over products and sequences -/

theorem Ex_prod {α β : Type} [Fintype α] [Fintype β] (g : α × β → ℝ) :
    Ex g = Ex fun a => Ex fun b => g (a, b) := by
  unfold Ex
  rw [Fintype.sum_prod_type, Fintype.card_prod, ← Finset.sum_div]
  push_cast
  rw [div_div, mul_comm]

theorem Ex_mul_left {α : Type} [Fintype α] (c : ℝ) (f : α → ℝ) : Ex (fun a => c * f a) = c * Ex f := by
  unfold Ex; rw [← Finset.mul_sum, mul_div_assoc]

theorem Ex_mul_right {α : Type} [Fintype α] (f : α → ℝ) (c : ℝ) : Ex (fun a => f a * c) = Ex f * c := by
  unfold Ex; rw [← Finset.sum_mul]; ring

theorem Ex_ind {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] :
    Ex (fun a => if p a then (1 : ℝ) else 0) = Pr p := by
  rw [Pr_eq_sum p]; rfl

theorem Ex_ite {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) [DecidablePred p] (u v : ℝ) :
    Ex (fun a => if p a then u else v) = Pr p * u + (1 - Pr p) * v := by
  have e : (fun a => if p a then u else v) = fun a => (u - v) * (if p a then (1 : ℝ) else 0) + v :=
    funext fun a => by split_ifs <;> ring
  rw [e, Ex_add, Ex_mul_left, Ex_ind, Ex_const]
  ring

/-- `K + 1` uniform coins are a uniform first coin and `K` uniform others. -/
theorem Ex_cons {D : Type} [Fintype D] (K : ℕ) (f : D → (Fin K → D) → ℝ) :
    Ex (fun ρ : Fin (K + 1) → D => f (ρ 0) fun i => ρ i.succ) = Ex fun d => Ex fun σ => f d σ := by
  refine Eq.trans ?_ (Ex_prod fun p : D × (Fin K → D) => f p.1 p.2)
  unfold Ex
  rw [Fintype.card_congr (Fin.consEquiv fun _ => D).symm]
  congr 1
  exact (Fintype.sum_equiv (Fin.consEquiv fun _ => D) _ _ fun p => by simp [Fin.consEquiv]).symm

/-- A sum over `m` independent coordinates averages to `m` times one coordinate's average. -/
theorem Ex_sum_pi {E : Type} [Fintype E] [Nonempty E] (f : E → ℝ) :
    ∀ m : ℕ, Ex (fun ρ : Fin m → E => ∑ i, f (ρ i)) = m * Ex f
  | 0 => by simp [Ex]
  | m + 1 => by
    calc Ex (fun ρ : Fin (m + 1) → E => ∑ i, f (ρ i))
        = Ex (fun ρ : Fin (m + 1) → E =>
            (fun d (σ : Fin m → E) => f d + ∑ i, f (σ i)) (ρ 0) fun i => ρ i.succ) := by
          congr 1; funext ρ; exact Fin.sum_univ_succ _
      _ = Ex fun d => Ex fun σ : Fin m → E => f d + ∑ i, f (σ i) :=
          Ex_cons m fun d (σ : Fin m → E) => f d + ∑ i, f (σ i)
      _ = Ex fun d => f d + m * Ex f := by
          congr 1; funext d; rw [Ex_add, Ex_const, Ex_sum_pi f m]
      _ = ((m + 1 : ℕ) : ℝ) * Ex f := by rw [Ex_add, Ex_const]; push_cast; ring

/-! ## One wait's reruns -/

/-- A wait's reruns over the coin draws `rs`, from `a` accepted so far. -/
def waitRuns {R : Type} (run : R → Trace) (k : ℕ) : List R → ℕ → ℕ
  | [], _ => 0
  | r :: rs, a => if k ≤ a then 0 else waitRuns run k rs (if (run r).out = .acc then a + 1 else a) + 1

theorem waitAcc_runs_eq {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (acc : List Seen), ((waitAcc k rs acc).eval H run).2.2 = waitRuns run k rs acc.length := by
  intro rs
  induction rs with
  | nil => intro acc; simp [waitAcc, waitRuns, RP.eval]
  | cons r rs ih =>
    intro acc
    simp only [waitAcc, waitRuns]
    split_ifs with h hacc
    · simp [RP.eval]
    all_goals simp [RP.eval, ih, Trace.seen, hacc]

/-- The waits rerun the prover at most as often as the waits do, each on its own. -/
theorem firstWait_runs_le_sum {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) :
    ∀ rss : List (List R), ((firstWait k opens assemble check rss).eval H run).2.2 ≤
      (rss.map fun rs => waitRuns run k rs 0).sum := by
  intro rss
  induction rss with
  | nil => simp [firstWait, RP.eval]
  | cons rs rss ih =>
    simp only [firstWait, List.map_cons, List.sum_cons]
    rw [(RP.eval_bind H run _ _).2, waitAcc_runs_eq, List.length_nil]
    refine Nat.add_le_add_left ?_ _
    split
    · split
      · simp [RP.eval]
      · exact ih
    · exact ih

/-- `N(K, a)`: one wait's expected reruns over `K` uniform draws, from `a` accepted so far. -/
noncomputable def waitExp {D : Type} [Fintype D] (run : D → Trace) (k K a : ℕ) : ℝ :=
  Ex fun σ : Fin K → D => (waitRuns run k ((List.finRange K).map σ) a : ℝ)

/-- **One wait's expected reruns**: `ε · N(K, a) ≤ k − a`, so a wait makes at most `k/ε` reruns in expectation. -/
theorem waitRuns_exp {D : Type} [Fintype D] [Nonempty D] (run : D → Trace) (k K : ℕ) :
    ∀ a, a ≤ k → Pr (fun d => (run d).out = .acc) * waitExp run k K a ≤ k - a := by
  induction K with
  | zero => intro a ha; simp [waitExp, waitRuns, Ex, ha]
  | succ K ih =>
    intro a ha
    obtain hlt | rfl := ha.lt_or_eq
    · have hstep : ∀ ρ : Fin (K + 1) → D, (waitRuns run k ((List.finRange (K + 1)).map ρ) a : ℝ) =
          (waitRuns run k ((List.finRange K).map fun i => ρ i.succ)
            (if (run (ρ 0)).out = .acc then a + 1 else a) : ℝ) + 1 := by
        intro ρ
        rw [List.finRange_succ, List.map_cons, List.map_map]
        simp only [waitRuns, not_le.2 hlt, ↓reduceIte, Nat.cast_add, Nat.cast_one, Function.comp_def]
      have h1 : waitExp run k (K + 1) a = Ex fun d => Ex fun σ : Fin K → D =>
          (waitRuns run k ((List.finRange K).map σ) (if (run d).out = .acc then a + 1 else a) : ℝ) + 1 :=
        (congrArg Ex (funext hstep)).trans (Ex_cons K fun d σ =>
          (waitRuns run k ((List.finRange K).map σ) (if (run d).out = .acc then a + 1 else a) : ℝ) + 1)
      have h2 : ∀ d : D, (Ex fun σ : Fin K → D =>
          (waitRuns run k ((List.finRange K).map σ) (if (run d).out = .acc then a + 1 else a) : ℝ) + 1) =
          if (run d).out = .acc then waitExp run k K (a + 1) + 1 else waitExp run k K a + 1 := by
        intro d
        rw [Ex_add, Ex_const]
        by_cases hc : (run d).out = .acc <;> simp only [hc, ↓reduceIte, waitExp]
      have hE : waitExp run k (K + 1) a = Pr (fun d => (run d).out = .acc) * (waitExp run k K (a + 1) + 1) +
          (1 - Pr fun d => (run d).out = .acc) * (waitExp run k K a + 1) := by
        rw [h1, ← Ex_ite]
        congr 1; funext d; exact h2 d
      have hε0 := Pr_nonneg fun d => (run d).out = .acc
      have hε1 := Pr_le_one fun d => (run d).out = .acc
      have ih1 := mul_le_mul_of_nonneg_left (ih (a + 1) (Nat.succ_le_of_lt hlt)) hε0
      have ih0 := mul_le_mul_of_nonneg_left (ih a ha) (sub_nonneg.2 hε1)
      push_cast at ih1
      rw [hE]
      linarith
    · simp [waitExp, waitRuns, List.finRange_succ, Ex]

/-- The waits' bound, over the judged run's coins and `m` waits of `K` draws: `𝔼[1[acc] · Σᵢ wait_i + 1] ≤ 1 + m·k`. -/
theorem waits_exp {D : Type} [Fintype D] [Nonempty D] (run : D → Trace) (k m K : ℕ) :
    Ex (fun s : D × (Fin m → Fin K → D) => (if (run s.1).out = .acc then (1 : ℝ) else 0) *
      ∑ i, (waitRuns run k ((List.finRange K).map (s.2 i)) 0 : ℝ) + 1) ≤ (1 : ℝ) + m * k := by
  have hin : ∀ d : D, (Ex fun ρ : Fin m → Fin K → D => (if (run d).out = .acc then (1 : ℝ) else 0) *
      ∑ i, (waitRuns run k ((List.finRange K).map (ρ i)) 0 : ℝ)) =
      (if (run d).out = .acc then (1 : ℝ) else 0) * (m * waitExp run k K 0) := by
    intro d
    rw [Ex_mul_left, Ex_sum_pi (fun σ : Fin K → D => (waitRuns run k ((List.finRange K).map σ) 0 : ℝ)) m]
    rfl
  calc Ex (fun s : D × (Fin m → Fin K → D) => (if (run s.1).out = .acc then (1 : ℝ) else 0) *
        ∑ i, (waitRuns run k ((List.finRange K).map (s.2 i)) 0 : ℝ) + 1)
      = Ex (fun s : D × (Fin m → Fin K → D) => (if (run s.1).out = .acc then (1 : ℝ) else 0) *
          ∑ i, (waitRuns run k ((List.finRange K).map (s.2 i)) 0 : ℝ)) + 1 := by rw [Ex_add, Ex_const]
    _ = Ex (fun d : D => (if (run d).out = .acc then (1 : ℝ) else 0) * (m * waitExp run k K 0)) + 1 := by
        rw [Ex_prod]; congr 1; congr 1; funext d; exact hin d
    _ = Pr (fun d => (run d).out = .acc) * (m * waitExp run k K 0) + 1 := by rw [Ex_mul_right, Ex_ind]
    _ ≤ 1 + m * k := by
        have h := mul_le_mul_of_nonneg_left (waitRuns_exp run k K 0 (Nat.zero_le _)) (Nat.cast_nonneg (α := ℝ) m)
        simp only [Nat.cast_zero, sub_zero] at h
        linarith [show Pr (fun d => (run d).out = .acc) * (m * waitExp run k K 0) =
          m * (Pr (fun d => (run d).out = .acc) * waitExp run k K 0) by ring]

/-! ## `E^rep`'s expected reruns -/

section ERep

variable {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`E^rep` reruns the prover at most `1 + m·k` times in expectation**, at every prover and tape. -/
theorem ERep_runs_exp (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) :
    ExtractRunsExpLe H (ERep V m k K opens assemble check) (fun x => 1 + m x * k x) := by
  intro x Ω P ω
  refine (Ex_mono fun s => ?_).trans (waits_exp (run H P V x ω) (k x) (m x) (K x))
  obtain ⟨r₀, ρ⟩ := s
  simp only [Extractor.eval, ERep, RP.eval]
  by_cases h : (run H P V x ω r₀).out = .acc
  · have hs : (run H P V x ω r₀).seen.out = .acc := h
    have hw := firstWait_runs_le_sum H (run H P V x ω) (k x) opens (assemble x) (check x)
      ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (ρ i))
    simp only [List.map_map, Function.comp_def] at hw
    rw [← Fin.sum_univ_def] at hw
    simp only [hs, h, ↓reduceIte, one_mul]
    exact_mod_cast Nat.add_le_add_right hw 1
  · have hs : ¬ (run H P V x ω r₀).seen.out = .acc := h
    simp [hs, h, RP.eval]

/-- **`ERepCost`**: `E^rep` reruns the prover at most `1 + m·K` times, and `1 + m·k` in expectation. -/
theorem ERep_cost (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) :
    ERepCost H V m k K opens assemble check :=
  ⟨ERep_runs_le H V m k K opens assemble check, ERep_runs_exp H V m k K opens assemble check⟩

end ERep

/-- **Flock: efficiency rests on the decoder alone**: `ERepCost` holds at every deployment, so Flock's `Efficient` is a
decoder that meets the list-decoding spec, with its stated complexity. -/
theorem Flock.Efficient_of_decoder {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (counted : List (ℕ × Col) → List T × ℕ) (size : List (ℕ × Col) → ℕ) (p : Polynomial ℕ)
    (hl : Flock.ListComplete D.enc D.agree D.nMin decode) (hd : DecoderComplexity decode counted size p) :
    Flock.Efficient H D decode counted size p :=
  ⟨ERep_cost H D.V D.waits D.k D.cap D.cols _ _, hl, hd⟩

end Claims
