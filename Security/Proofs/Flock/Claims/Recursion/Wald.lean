import Proofs.Flock.Claims.Recursion.Serve

/-!
# `E^rep`'s weighted reruns: Wald's identity

When an outer stage serves `E^rep`'s reruns, each rerun costs what serving it costs, so the composed extractor's cost is
`E^rep`'s reruns weighted by a cost `w` of the draw each is at. A wait's draws are uniform and independent, and whether
it reruns its `j`-th draw reads only the draws before it, so its expected weighted reruns are the expected weight times
its expected reruns (`waitCost_exp`, Wald's identity, by induction on the draws as `waitRuns_exp` is). With one wait's
`ε · N(K, 0) ≤ k`, `E^rep` over any draw type `D` (`erepProg`, of which `ERep`'s program is the instance at the run's
coins) has expected weighted reruns at most `(1 + m·k) · 𝔼[w]` (`erepProg_cost_exp`), and at most `1 + m·K` reruns on
every draw (`erepProg_runs_le`), against every run.
-/

namespace Claims

/-- **`E^rep`'s program over draws `D`**: rerun the judged draw `d₀`; if it accepted, make the `m` waits over the
draws `ρ`, each stopping at `k` accepted reruns, and return the first assembled candidate that `check` accepts. -/
def erepProg {D W Op : Type} (m k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W)
    (check : W → Bool) (d₀ : D) (ρ : Fin m → Fin K → D) : RP D (Option W) :=
  .rerun d₀ fun v₀ =>
    if v₀.out = .acc then
      firstWait k opens assemble check ((List.finRange m).map fun i => (List.finRange K).map (ρ i))
    else .ret none

/-- `E^rep`'s program is `erepProg` at the judged run's coins. -/
theorem ERep_ext_eq {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] (V : Verifier Coin) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg)
    (r₀ : Fin (V.k x) → Coin) (ρ : (ERep V m k K opens assemble check).C x) :
    (ERep V m k K opens assemble check).ext x r₀ ρ = erepProg (m x) (k x) (K x) opens (assemble x) (check x) r₀ ρ :=
  rfl

/-! ## One wait -/

/-- **A wait's weighted reruns**: `Σ w` over the draws `rs` it reruns at, from `a` accepted so far. -/
def waitCost {R : Type} (run : R → Trace) (w : R → ℝ) (k : ℕ) : List R → ℕ → ℝ
  | [], _ => 0
  | r :: rs, a => if k ≤ a then 0 else w r + waitCost run w k rs (if (run r).out = .acc then a + 1 else a)

theorem waitCost_nonneg {R : Type} (run : R → Trace) {w : R → ℝ} (hw : ∀ r, 0 ≤ w r) (k : ℕ) :
    ∀ (rs : List R) (a : ℕ), 0 ≤ waitCost run w k rs a := by
  intro rs
  induction rs with
  | nil => intro a; simp [waitCost]
  | cons r rs ih =>
    intro a
    simp only [waitCost]
    split_ifs
    · exact le_rfl
    all_goals exact add_nonneg (hw r) (ih _)

theorem waitAcc_cost_eq {R : Type} (H : Msg → Msg) (run : R → Trace) (w : R → ℝ) (k : ℕ) :
    ∀ (rs : List R) (acc : List Seen),
      (((waitAcc k rs acc).rerunsAt H run).map w).sum = waitCost run w k rs acc.length := by
  intro rs
  induction rs with
  | nil => intro acc; simp [waitAcc, waitCost, RP.rerunsAt]
  | cons r rs ih =>
    intro acc
    simp only [waitAcc, waitCost]
    split_ifs with h hacc
    · simp [RP.rerunsAt]
    · simp only [RP.rerunsAt, List.map_cons, List.sum_cons, Trace.seen, hacc, ↓reduceIte, ih,
        List.length_append, List.length_singleton]
    · simp only [RP.rerunsAt, List.map_cons, List.sum_cons, Trace.seen, hacc, ↓reduceIte, ih]

/-- **Wald's identity for one wait**: over `K` uniform draws, its expected weighted reruns from `a` accepted are the
expected weight times its expected reruns `N(K, a)`. -/
theorem waitCost_exp {D : Type} [Fintype D] [Nonempty D] (run : D → Trace) (w : D → ℝ) (k K : ℕ) :
    ∀ a, Ex (fun σ : Fin K → D => waitCost run w k ((List.finRange K).map σ) a) = Ex w * waitExp run k K a := by
  induction K with
  | zero => intro a; simp [waitCost, waitExp, waitRuns, Ex]
  | succ K ih =>
    intro a
    by_cases hka : k ≤ a
    · have h0 : (fun σ : Fin (K + 1) → D => waitCost run w k ((List.finRange (K + 1)).map σ) a) = fun _ => 0 := by
        funext σ; rw [List.finRange_succ, List.map_cons]; simp [waitCost, hka]
      have h1 : waitExp run k (K + 1) a = 0 := by
        unfold waitExp
        rw [show (fun σ : Fin (K + 1) → D => (waitRuns run k ((List.finRange (K + 1)).map σ) a : ℝ)) =
          fun _ => 0 from funext fun σ => by rw [List.finRange_succ, List.map_cons]; simp [waitRuns, hka]]
        exact Ex_const 0
      rw [h0, h1, Ex_const, mul_zero]
    · have hstep : ∀ σ : Fin (K + 1) → D, waitCost run w k ((List.finRange (K + 1)).map σ) a =
          (fun d (τ : Fin K → D) => w d +
            waitCost run w k ((List.finRange K).map τ) (if (run d).out = .acc then a + 1 else a)) (σ 0)
            fun i => σ i.succ := by
        intro σ
        rw [List.finRange_succ, List.map_cons, List.map_map]
        simp only [waitCost, hka, ↓reduceIte, Function.comp_def]
      have hrun : ∀ σ : Fin (K + 1) → D, (waitRuns run k ((List.finRange (K + 1)).map σ) a : ℝ) =
          (fun d (τ : Fin K → D) => (waitRuns run k ((List.finRange K).map τ)
            (if (run d).out = .acc then a + 1 else a) : ℝ) + 1) (σ 0) fun i => σ i.succ := by
        intro σ
        rw [List.finRange_succ, List.map_cons, List.map_map]
        simp only [waitRuns, hka, ↓reduceIte, Nat.cast_add, Nat.cast_one, Function.comp_def]
      have hE : waitExp run k (K + 1) a =
          Ex fun d => waitExp run k K (if (run d).out = .acc then a + 1 else a) + 1 := by
        unfold waitExp
        rw [congrArg Ex (funext hrun), Ex_cons K fun d (τ : Fin K → D) => (waitRuns run k ((List.finRange K).map τ)
          (if (run d).out = .acc then a + 1 else a) : ℝ) + 1]
        congr 1; funext d; rw [Ex_add, Ex_const]
      rw [congrArg Ex (funext hstep), Ex_cons K fun d (τ : Fin K → D) => w d +
        waitCost run w k ((List.finRange K).map τ) (if (run d).out = .acc then a + 1 else a), hE]
      have hin : ∀ d, (Ex fun τ : Fin K → D => w d +
          waitCost run w k ((List.finRange K).map τ) (if (run d).out = .acc then a + 1 else a)) =
          w d + Ex w * waitExp run k K (if (run d).out = .acc then a + 1 else a) := by
        intro d; rw [Ex_add, Ex_const, ih]
      rw [congrArg Ex (funext hin), Ex_add, Ex_mul_left, Ex_add, Ex_const]
      ring

/-! ## The waits and `E^rep` -/

/-- The waits' weighted reruns are at most the waits', each on its own, for a nonnegative weight. -/
theorem firstWait_cost_le_sum {R W Op : Type} (H : Msg → Msg) (run : R → Trace) {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (k : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) :
    ∀ rss : List (List R), (((firstWait k opens assemble check rss).rerunsAt H run).map w).sum ≤
      (rss.map fun rs => waitCost run w k rs 0).sum := by
  intro rss
  induction rss with
  | nil => simp [firstWait, RP.rerunsAt]
  | cons rs rss ih =>
    simp only [firstWait, List.map_cons, List.sum_cons]
    rw [RP.rerunsAt_bind, List.map_append, List.sum_append, waitAcc_cost_eq, List.length_nil]
    refine add_le_add le_rfl ?_
    have hrest : 0 ≤ (rss.map fun rs => waitCost run w k rs 0).sum :=
      List.sum_nonneg fun c hc => by
        obtain ⟨rs', _, rfl⟩ := List.mem_map.1 hc
        exact waitCost_nonneg run hw k rs' 0
    split
    · split
      · simpa [RP.rerunsAt] using hrest
      · exact ih
    · exact ih

/-- **`E^rep` reruns at most `1 + m·K` times** over any draws, against every run. -/
theorem erepProg_runs_le {D W Op : Type} (H : Msg → Msg) (run : D → Trace) (m k K : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (d₀ : D) (ρ : Fin m → Fin K → D) :
    ((erepProg m k K opens assemble check d₀ ρ).eval H run).2.2 ≤ 1 + m * K := by
  simp only [erepProg, RP.eval]
  rw [Nat.add_comm 1]
  refine Nat.add_le_add_right ?_ 1
  split
  · refine (firstWait_runs_le H _ _ _ _ _ _).trans (le_of_eq ?_)
    simp [List.map_map, Function.comp_def, List.sum_replicate]
  · simp [RP.eval]

/-- **`E^rep`'s expected weighted reruns**: over a uniform judged draw and uniform waits' draws, against every run and
for every nonnegative weight `w` of a draw, `𝔼[Σ w] ≤ (1 + m·k) · 𝔼[w]`. -/
theorem erepProg_cost_exp {D W Op : Type} [Fintype D] [Nonempty D] (H : Msg → Msg) (run : D → Trace) {w : D → ℝ}
    (hw : ∀ d, 0 ≤ w d) (m k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) :
    Ex (fun s : D × (Fin m → Fin K → D) =>
      (((erepProg m k K opens assemble check s.1 s.2).rerunsAt H run).map w).sum) ≤ (1 + m * k) * Ex w := by
  have hpt : ∀ s : D × (Fin m → Fin K → D),
      (((erepProg m k K opens assemble check s.1 s.2).rerunsAt H run).map w).sum ≤
        w s.1 + (if (run s.1).out = .acc then (1 : ℝ) else 0) *
          ∑ i, waitCost run w k ((List.finRange K).map (s.2 i)) 0 := by
    intro s
    obtain ⟨d₀, ρ⟩ := s
    simp only [erepProg, RP.rerunsAt, Trace.seen, List.map_cons, List.sum_cons]
    refine add_le_add le_rfl ?_
    by_cases h : (run d₀).out = .acc
    · simp only [h, ↓reduceIte, one_mul]
      have hw' := firstWait_cost_le_sum H run hw k opens assemble check
        ((List.finRange m).map fun i => (List.finRange K).map (ρ i))
      simp only [List.map_map, Function.comp_def] at hw'
      rwa [← Fin.sum_univ_def] at hw'
    · simp [h, RP.rerunsAt]
  have hin : ∀ d : D, (Ex fun ρ : Fin m → Fin K → D => w d + (if (run d).out = .acc then (1 : ℝ) else 0) *
      ∑ i, waitCost run w k ((List.finRange K).map (ρ i)) 0) =
      w d + (if (run d).out = .acc then (1 : ℝ) else 0) * (m * (Ex w * waitExp run k K 0)) := by
    intro d
    rw [Ex_add, Ex_const, Ex_mul_left,
      Ex_sum_pi (fun σ : Fin K → D => waitCost run w k ((List.finRange K).map σ) 0) m, waitCost_exp]
  have hEw : 0 ≤ Ex w := Ex_nonneg hw
  have hwait := waitRuns_exp run k K 0 (Nat.zero_le _)
  simp only [Nat.cast_zero, sub_zero] at hwait
  calc Ex (fun s : D × (Fin m → Fin K → D) =>
        (((erepProg m k K opens assemble check s.1 s.2).rerunsAt H run).map w).sum)
      ≤ Ex (fun s : D × (Fin m → Fin K → D) => w s.1 + (if (run s.1).out = .acc then (1 : ℝ) else 0) *
          ∑ i, waitCost run w k ((List.finRange K).map (s.2 i)) 0) := Ex_mono hpt
    _ = Ex (fun d : D => w d + (if (run d).out = .acc then (1 : ℝ) else 0) * (m * (Ex w * waitExp run k K 0))) := by
        rw [Ex_prod]; exact congrArg Ex (funext hin)
    _ = Ex w + Pr (fun d => (run d).out = .acc) * waitExp run k K 0 * (m * Ex w) := by
        rw [Ex_add, Ex_mul_right, Ex_ind]; ring
    _ ≤ Ex w + k * (m * Ex w) := by
        exact add_le_add le_rfl (mul_le_mul_of_nonneg_right hwait (mul_nonneg (Nat.cast_nonneg _) hEw))
    _ = (1 + m * k) * Ex w := by ring

end Claims
