import Proofs.Flock.Claims.ERepCost

/-!
# The sampled extractor: one extraction per drawn region, and its reruns

The sampled verifier draws regions from its coins, and each drawn region's Flock proof runs on rounds of its own,
`I x i ⊆ Fin (V.k x)`. The sampled extractor `SExt` replays the judged run and, if it accepted, runs for each drawn
region `i` (`sel x r₀`, read from the judged run's coins) region `i`'s extractor `p x i` on reruns that keep the judged
run's coins off `I x i`, the draw among them, and draw `I x i` fresh (`forkOn`): every such rerun samples region `i`.
`finish` assembles the regions' results.

* `forkOn`, `Ex_forkOn`: coins fresh on `I` and the judged run's elsewhere are uniform when both are.
  `Ex_acc_forkOn`, `Pr_acc_forkOn`: a judged run's acceptance against a weight that ignores `I` averages as the
  acceptance probability of the runs fresh on `I`, the probability region `i`'s extractor sees.
* `comap`, `eval_comap`: a program on a sub-run's coins, run on the whole run's; `forEach`, `eval_forEach`: programs
  in sequence, their reruns added.
* `SExt_eval`: the sampled extractor's result and reruns, region by region.
* `SExt_runs_exp`: if each region's extractor reruns at most `b(x)/ε` times in expectation against any run that accepts
  with probability `ε`, the sampled extractor reruns at most `1 + b(x) · 𝔼|sel|` times in expectation
  (`ExtractRunsExpLe`). `SExt_runs_le`: at most `1 + n_max · K_b` in the worst case (`ExtractRunsLe`).
* `waitsProg`, `waits_amortized`: Flock's `E^rep` waits, `m` of them over `K` draws each stopping at `k` accepted runs,
  satisfy that with `b = m·k` (`waitRuns_exp`). `SERep`, `SERep_cost`: the sampled `E^rep`, at most `1 + n_max·m·K`
  reruns and `1 + m·k·𝔼|sel|` in expectation.
* `NoHash`, `SExt_calls_le`: the sampled extractor makes no SHA-512 call of its own, so on a prover whose runs make at
  most `t` calls each, its calls are at most `t` times its reruns.
-/

namespace Claims.Sampling

open Claims

/-! ## Coins fresh on a set of rounds -/

/-- The coins `d` on the rounds `I`, and the judged run's `r₀` elsewhere. -/
def forkOn {k : ℕ} {Coin : Type} (I : Finset (Fin k)) (r₀ d : Fin k → Coin) : Fin k → Coin :=
  fun t => if t ∈ I then d t else r₀ t

theorem forkOn_forkOn {k : ℕ} {Coin : Type} (I : Finset (Fin k)) (r d d' : Fin k → Coin) :
    forkOn I (forkOn I r d) d' = forkOn I r d' :=
  funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h]

theorem forkOn_forkOn' {k : ℕ} {Coin : Type} (I : Finset (Fin k)) (r d : Fin k → Coin) :
    forkOn I (forkOn I r d) = forkOn I r :=
  funext fun d' => forkOn_forkOn I r d d'

/-- Swapping two coin vectors on the rounds `I`: an involution. -/
def swapOn {k : ℕ} {Coin : Type} (I : Finset (Fin k)) :
    (Fin k → Coin) × (Fin k → Coin) ≃ (Fin k → Coin) × (Fin k → Coin) where
  toFun p := (forkOn I p.1 p.2, forkOn I p.2 p.1)
  invFun p := (forkOn I p.1 p.2, forkOn I p.2 p.1)
  left_inv p := Prod.ext (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])
    (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])
  right_inv p := Prod.ext (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])
    (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])

/-- **Coins fresh on `I` are uniform**: `forkOn I r₀ d` at uniform `r₀, d` is uniform. -/
theorem Ex_forkOn {k : ℕ} {Coin : Type} [Fintype Coin] [Nonempty Coin] (I : Finset (Fin k))
    (F : (Fin k → Coin) → ℝ) :
    Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn I p.1 p.2)) = Ex F := by
  have h1 : Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn I p.1 p.2)) =
      Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F p.1) := by
    unfold Ex
    congr 1
    exact Fintype.sum_equiv (swapOn I) _ _ fun p => rfl
  rw [h1, Ex_prod]
  congr 1
  funext a
  exact Ex_const (α := Fin k → Coin) (F a)

/-- **Acceptance given the coins off `I`**: a judged run's acceptance, weighted by `g` that ignores the rounds `I`,
averages as the acceptance probability of the runs fresh on `I`. -/
theorem Ex_acc_forkOn {k : ℕ} {Coin : Type} [Fintype Coin] [Nonempty Coin] (I : Finset (Fin k))
    (acc : (Fin k → Coin) → Prop) [DecidablePred acc] (g : (Fin k → Coin) → ℝ) (hg : ∀ r d, g (forkOn I r d) = g r) :
    Ex (fun r => (if acc r then (1 : ℝ) else 0) * g r) =
      Ex (fun r => Pr (fun d => acc (forkOn I r d)) * g r) := by
  rw [← Ex_forkOn I (fun r => (if acc r then (1 : ℝ) else 0) * g r), Ex_prod]
  congr 1
  funext r
  simp only [hg]
  rw [Ex_mul_right, Ex_ind]

/-- **The judged run's acceptance and an event that ignores `I`**: their joint probability averages the acceptance
probability of the runs fresh on `I` times the event's. -/
theorem Pr_acc_forkOn {k : ℕ} {Coin C : Type} [Fintype Coin] [Nonempty Coin] [Fintype C] (I : Finset (Fin k))
    (acc : (Fin k → Coin) → Prop) [DecidablePred acc] (Q : (Fin k → Coin) → C → Prop)
    (hQ : ∀ r d c, Q (forkOn I r d) c ↔ Q r c) :
    Pr (fun s : (Fin k → Coin) × C => acc s.1 ∧ Q s.1 s.2) =
      Ex (fun r => Pr (fun d => acc (forkOn I r d)) * Pr (fun c => Q r c)) := by
  classical
  have hs : Pr (fun s : (Fin k → Coin) × C => acc s.1 ∧ Q s.1 s.2) =
      Ex (fun r => (if acc r then (1 : ℝ) else 0) * Pr (fun c => Q r c)) := by
    rw [← Ex_ind, Ex_prod]
    congr 1
    funext r
    rw [← Ex_ind (fun c => Q r c), ← Ex_mul_left]
    congr 1
    funext c
    by_cases h1 : acc r <;> by_cases h2 : Q r c <;> simp [h1, h2]
  rw [hs, Ex_acc_forkOn I acc _ fun r d => Pr_congr fun c => hQ r d c]

/-! ## Averages over regions -/

theorem Ex_finsum {α ι : Type} [Fintype α] (s : Finset ι) (f : ι → α → ℝ) :
    Ex (fun a => ∑ i ∈ s, f i a) = ∑ i ∈ s, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

/-- One coordinate of independent uniform coordinates is uniform. -/
theorem Ex_pi_eval {ι E : Type} [Fintype ι] [DecidableEq ι] [Fintype E] [Nonempty E] (i : ι) (f : E → ℝ) :
    Ex (fun c : ι → E => f (c i)) = Ex f := by
  have h : Ex (fun c : ι → E => f (c i)) = Ex (fun p : E × ({j // j ≠ i} → E) => f p.1) := by
    unfold Ex
    rw [Fintype.card_congr (Equiv.funSplitAt i E)]
    congr 1
    exact Fintype.sum_equiv (Equiv.funSplitAt i E) _ _ fun c => rfl
  rw [h, Ex_prod]
  congr 1
  funext a
  exact Ex_const (α := {j // j ≠ i} → E) (f a)

/-- A sum over a list of regions, region by region with its multiplicity. -/
theorem sum_map_eq_count {ι : Type} [Fintype ι] [DecidableEq ι] (l : List ι) (f : ι → ℝ) :
    (l.map f).sum = ∑ i, (l.count i : ℝ) * f i := by
  rw [Finset.sum_list_map_count, ← Finset.sum_subset (Finset.subset_univ l.toFinset)]
  · simp only [nsmul_eq_mul]
  · intro i _ hi
    rw [List.count_eq_zero.2 (fun h => hi (List.mem_toFinset.2 h)), Nat.cast_zero, zero_mul]

theorem sum_count_eq_length {ι : Type} [Fintype ι] [DecidableEq ι] (l : List ι) :
    ∑ i, (l.count i : ℝ) = l.length := by
  have := sum_map_eq_count l fun _ => 1
  simp only [mul_one, List.map_const', List.sum_replicate, nsmul_eq_mul, mul_one] at this
  exact this.symm

/-! ## Rerun programs on sub-runs, and in sequence -/

/-- A program on a sub-run's coins `R'`, run on the whole run's: each rerun at `f r'`. -/
def comap {R R' α : Type} (f : R' → R) : RP R' α → RP R α
  | .ret a => .ret a
  | .hash q k => .hash q fun a => comap f (k a)
  | .rerun r k => .rerun (f r) fun v => comap f (k v)

theorem eval_comap {R R' α : Type} (H : Msg → Msg) (run : R → Trace) (f : R' → R) (p : RP R' α) :
    (comap f p).eval H run = p.eval H (run ∘ f) := by
  induction p with
  | ret a => rfl
  | hash q k ih => simp only [comap, RP.eval, ih]
  | rerun r k ih => simp only [comap, RP.eval, ih, Function.comp]

/-- The programs `p i` for the regions `l` in order, collecting each region's result. -/
def forEach {R ι α : Type} (p : ι → RP R α) : List ι → RP R (List (ι × α))
  | [] => .ret []
  | i :: l => (p i).bind fun a => (forEach p l).bind fun rest => .ret ((i, a) :: rest)

theorem eval_forEach {R ι α : Type} (H : Msg → Msg) (run : R → Trace) (p : ι → RP R α) :
    ∀ l : List ι, ((forEach p l).eval H run).1 = l.map (fun i => (i, ((p i).eval H run).1)) ∧
      ((forEach p l).eval H run).2.2 = (l.map fun i => ((p i).eval H run).2.2).sum := by
  intro l
  induction l with
  | nil => exact ⟨rfl, rfl⟩
  | cons i l ih =>
    simp only [forEach, List.map_cons, List.sum_cons]
    obtain ⟨h1, h2⟩ := RP.eval_bind H run (p i) fun a => (forEach p l).bind fun rest => .ret ((i, a) :: rest)
    obtain ⟨h3, h4⟩ := RP.eval_bind H run (forEach p l) fun rest => .ret ((i, ((p i).eval H run).1) :: rest)
    refine ⟨?_, ?_⟩
    · rw [h1, h3, ih.1]; rfl
    · rw [h2, h4, ih.2]; simp [RP.eval]

/-! ## The sampled extractor -/

/-- **The sampled extractor**: replay the judged run; if it accepted, run region `i`'s extractor `p x i (c i)` for each
drawn region `i ∈ sel x r₀`, on reruns fresh on its rounds `I x i` and the judged run's elsewhere, and assemble the
results with `finish`. -/
def SExt {Coin W T : Type} (V : Verifier Coin) (n : Msg → ℕ) (I : (x : Msg) → Fin (n x) → Finset (Fin (V.k x)))
    (sel : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x))) (Cr : Msg → Type) [∀ x, Fintype (Cr x)]
    [∀ x, Nonempty (Cr x)] (p : (x : Msg) → Fin (n x) → Cr x → RP (Fin (V.k x) → Coin) (Option T))
    (finish : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x) × Option T) → Option W) : Extractor Coin W V where
  C x := Fin (n x) → Cr x
  ext x r₀ c := .rerun r₀ fun v₀ =>
    if v₀.out = .acc then
      (forEach (fun i => comap (forkOn (I x i) r₀) (p x i (c i))) (sel x r₀)).bind fun res => .ret (finish x r₀ res)
    else .ret none

/-- **The sampled extractor, region by region**: on an accepted judged run it returns `finish` of each drawn region's
result against the runs fresh on that region's rounds, and reruns once plus each region's reruns; on a rejected one,
`none` after one rerun. -/
theorem SExt_eval {Ω Coin W T : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (n : Msg → ℕ)
    (I : (x : Msg) → Fin (n x) → Finset (Fin (V.k x))) (sel : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x)))
    (Cr : Msg → Type) [∀ x, Fintype (Cr x)] [∀ x, Nonempty (Cr x)]
    (p : (x : Msg) → Fin (n x) → Cr x → RP (Fin (V.k x) → Coin) (Option T))
    (finish : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x) × Option T) → Option W) (P : Prover Ω) (x : Msg)
    (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : Fin (n x) → Cr x) :
    let sub := fun i => (p x i (c i)).eval H (run H P V x ω ∘ forkOn (I x i) r₀)
    ((SExt V n I sel Cr p finish).eval H P x ω r₀ c).1 =
        (if (run H P V x ω r₀).out = .acc then finish x r₀ ((sel x r₀).map fun i => (i, (sub i).1)) else none) ∧
      ((SExt V n I sel Cr p finish).eval H P x ω r₀ c).2.2 =
        (if (run H P V x ω r₀).out = .acc then ((sel x r₀).map fun i => (sub i).2.2).sum else 0) + 1 := by
  intro sub
  simp only [Extractor.eval, SExt, RP.eval]
  by_cases h : (run H P V x ω r₀).out = .acc
  · have hs : (run H P V x ω r₀).seen.out = .acc := h
    simp only [hs, h, ↓reduceIte]
    obtain ⟨h1, h2⟩ := RP.eval_bind H (run H P V x ω)
      (forEach (fun i => comap (forkOn (I x i) r₀) (p x i (c i))) (sel x r₀)) fun res => .ret (finish x r₀ res)
    obtain ⟨h3, h4⟩ := eval_forEach H (run H P V x ω) (fun i => comap (forkOn (I x i) r₀) (p x i (c i))) (sel x r₀)
    simp only [eval_comap] at h3 h4
    refine ⟨?_, ?_⟩
    · rw [h1]; simp only [RP.eval]; rw [h3]
    · rw [h2, h4]; simp [RP.eval, sub]
  · have hs : ¬ (run H P V x ω r₀).seen.out = .acc := h
    simp [hs, h, RP.eval]

/-- **The sampled extractor's expected reruns**: if region `i`'s extractor, against any run that accepts with
probability `ε`, reruns at most `b(x)/ε` times in expectation (`ε · 𝔼[reruns] ≤ b(x)`), and the draw `sel` ignores
every region's rounds, the sampled extractor reruns at most `1 + b(x) · 𝔼|sel|` times in expectation. -/
theorem SExt_runs_exp {Coin W T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (n : Msg → ℕ) (I : (x : Msg) → Fin (n x) → Finset (Fin (V.k x)))
    (sel : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x))) (Cr : Msg → Type) [∀ x, Fintype (Cr x)]
    [∀ x, Nonempty (Cr x)] (p : (x : Msg) → Fin (n x) → Cr x → RP (Fin (V.k x) → Coin) (Option T))
    (finish : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x) × Option T) → Option W) (b : Msg → ℝ)
    (hsel : ∀ x i r d, sel x (forkOn (I x i) r d) = sel x r)
    (hp : ∀ x i (run' : (Fin (V.k x) → Coin) → Trace),
      Pr (fun d => (run' d).out = .acc) * Ex (fun c : Cr x => (((p x i c).eval H run').2.2 : ℝ)) ≤ b x) :
    ExtractRunsExpLe H (SExt V n I sel Cr p finish)
      (fun x => 1 + b x * Ex (fun r : Fin (V.k x) → Coin => ((sel x r).length : ℝ))) := by
  classical
  intro x Ω P ω
  let N : Fin (n x) → (Fin (V.k x) → Coin) → Cr x → ℝ := fun i r ci =>
    (((p x i ci).eval H (run H P V x ω ∘ forkOn (I x i) r)).2.2 : ℝ)
  let EN : Fin (n x) → (Fin (V.k x) → Coin) → ℝ := fun i r => Ex (N i r)
  let A : (Fin (V.k x) → Coin) → ℝ := fun r => if (run H P V x ω r).out = .acc then (1 : ℝ) else 0
  have hpt : ∀ s : (Fin (V.k x) → Coin) × (Fin (n x) → Cr x),
      (((SExt V n I sel Cr p finish).eval H P x ω s.1 s.2).2.2 : ℝ) =
        (∑ i, A s.1 * (((sel x s.1).count i : ℝ) * N i s.1 (s.2 i))) + 1 := by
    intro s
    rw [(SExt_eval H V n I sel Cr p finish P x ω s.1 s.2).2]
    push_cast
    congr 1
    simp only [A, ← Finset.mul_sum]
    split_ifs
    · rw [one_mul, List.map_map]
      exact sum_map_eq_count (sel x s.1) fun i => N i s.1 (s.2 i)
    · simp [*]
  have hinv : ∀ i r d, EN i (forkOn (I x i) r d) = EN i r := by
    intro i r d
    simp only [EN, N, forkOn_forkOn']
  have hreg : ∀ i, Ex (fun s : (Fin (V.k x) → Coin) × (Fin (n x) → Cr x) =>
      A s.1 * (((sel x s.1).count i : ℝ) * N i s.1 (s.2 i))) ≤
        b x * Ex (fun r : Fin (V.k x) → Coin => ((sel x r).count i : ℝ)) := by
    intro i
    calc Ex (fun s : (Fin (V.k x) → Coin) × (Fin (n x) → Cr x) =>
          A s.1 * (((sel x s.1).count i : ℝ) * N i s.1 (s.2 i)))
        = Ex (fun r => A r * (((sel x r).count i : ℝ) * EN i r)) := by
          rw [Ex_prod]
          congr 1
          funext r
          dsimp only
          rw [Ex_mul_left, Ex_mul_left, Ex_pi_eval i (N i r)]
      _ = Ex (fun r => Pr (fun d => (run H P V x ω (forkOn (I x i) r d)).out = .acc) * (((sel x r).count i : ℝ) * EN i r)) :=
          Ex_acc_forkOn (I x i) (fun r => (run H P V x ω r).out = .acc)
            (fun r => ((sel x r).count i : ℝ) * EN i r) fun r d => by simp only [hsel, hinv]
      _ ≤ Ex (fun r => b x * ((sel x r).count i : ℝ)) := by
          refine Ex_mono fun r => ?_
          have h := hp x i (run H P V x ω ∘ forkOn (I x i) r)
          have hc : (0 : ℝ) ≤ (sel x r).count i := Nat.cast_nonneg _
          calc Pr (fun d => (run H P V x ω (forkOn (I x i) r d)).out = .acc) * (((sel x r).count i : ℝ) * EN i r)
              = ((sel x r).count i : ℝ) * (Pr (fun d => ((run H P V x ω ∘ forkOn (I x i) r) d).out = .acc) *
                  Ex (fun c : Cr x => (((p x i c).eval H (run H P V x ω ∘ forkOn (I x i) r)).2.2 : ℝ))) := by
                simp only [EN, N, Function.comp]; ring
            _ ≤ ((sel x r).count i : ℝ) * b x := mul_le_mul_of_nonneg_left h hc
            _ = b x * ((sel x r).count i : ℝ) := mul_comm _ _
      _ = b x * Ex (fun r : Fin (V.k x) → Coin => ((sel x r).count i : ℝ)) := Ex_mul_left _ _
  calc Ex (fun s : (Fin (V.k x) → Coin) × (Fin (n x) → Cr x) =>
        (((SExt V n I sel Cr p finish).eval H P x ω s.1 s.2).2.2 : ℝ))
      = (∑ i, Ex (fun s : (Fin (V.k x) → Coin) × (Fin (n x) → Cr x) =>
          A s.1 * (((sel x s.1).count i : ℝ) * N i s.1 (s.2 i)))) + 1 := by
        rw [show (fun s : (Fin (V.k x) → Coin) × (Fin (n x) → Cr x) =>
            (((SExt V n I sel Cr p finish).eval H P x ω s.1 s.2).2.2 : ℝ)) = _ from funext hpt, Ex_add, Ex_const,
          Ex_finsum]
    _ ≤ (∑ i, b x * Ex (fun r : Fin (V.k x) → Coin => ((sel x r).count i : ℝ))) + 1 := by
        gcongr with i
        exact hreg i
    _ = 1 + b x * Ex (fun r : Fin (V.k x) → Coin => ((sel x r).length : ℝ)) := by
        rw [← Finset.mul_sum, ← Ex_finsum]
        simp only [sum_count_eq_length]
        ring

/-- **The sampled extractor's worst-case reruns**: at most `n_max` drawn regions, each extractor rerunning at most
`K_b` times, make at most `1 + n_max · K_b` reruns. -/
theorem SExt_runs_le {Coin W T : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (n : Msg → ℕ)
    (I : (x : Msg) → Fin (n x) → Finset (Fin (V.k x))) (sel : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x)))
    (Cr : Msg → Type) [∀ x, Fintype (Cr x)] [∀ x, Nonempty (Cr x)]
    (p : (x : Msg) → Fin (n x) → Cr x → RP (Fin (V.k x) → Coin) (Option T))
    (finish : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x) × Option T) → Option W) (nmax Kb : Msg → ℕ)
    (hsel : ∀ x r, (sel x r).length ≤ nmax x)
    (hp : ∀ x i c (run' : (Fin (V.k x) → Coin) → Trace), ((p x i c).eval H run').2.2 ≤ Kb x) :
    ExtractRunsLe H (SExt V n I sel Cr p finish) (fun x => 1 + nmax x * Kb x) := by
  intro x Ω P ω r₀ c
  rw [(SExt_eval H V n I sel Cr p finish P x ω r₀ c).2, add_comm]
  refine Nat.add_le_add_left ?_ 1
  split_ifs
  · refine (List.sum_le_length_nsmul _ (Kb x) fun a ha => ?_).trans ?_
    · obtain ⟨i, _, rfl⟩ := List.mem_map.1 ha
      exact hp x i (c i) _
    · rw [List.length_map, smul_eq_mul]
      exact Nat.mul_le_mul_right _ (hsel x r₀)
  · exact Nat.zero_le _

/-! ## Flock's waits, region by region -/

/-- `E^rep`'s waits: `m` waits, each over `K` draws `ρ a` and stopping at `k` accepted runs, returning the first
assembled candidate that `check` accepts. -/
def waitsProg {D W Op : Type} (m k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W)
    (check : W → Bool) (ρ : Fin m → Fin K → D) : RP D (Option W) :=
  firstWait k opens assemble check ((List.finRange m).map fun a => (List.finRange K).map (ρ a))

/-- **The waits' reruns, amortized**: against a run accepting with probability `ε`, `m` waits stopping at `k` accepted
runs rerun at most `m·k/ε` times in expectation. -/
theorem waits_amortized {D W Op : Type} [Fintype D] [Nonempty D] (H : Msg → Msg) (run' : D → Trace) (m k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) :
    Pr (fun d => (run' d).out = .acc) *
        Ex (fun ρ : Fin m → Fin K → D => (((waitsProg m k K opens assemble check ρ).eval H run').2.2 : ℝ)) ≤
      m * k := by
  have hle : ∀ ρ : Fin m → Fin K → D, (((waitsProg m k K opens assemble check ρ).eval H run').2.2 : ℝ) ≤
      ∑ a, (waitRuns run' k ((List.finRange K).map (ρ a)) 0 : ℝ) := by
    intro ρ
    have hw := firstWait_runs_le_sum H run' k opens assemble check
      ((List.finRange m).map fun a => (List.finRange K).map (ρ a))
    simp only [List.map_map, Function.comp_def] at hw
    rw [← Fin.sum_univ_def] at hw
    exact_mod_cast hw
  have hE : Ex (fun ρ : Fin m → Fin K → D => ∑ a, (waitRuns run' k ((List.finRange K).map (ρ a)) 0 : ℝ)) =
      m * waitExp run' k K 0 :=
    Ex_sum_pi (fun σ : Fin K → D => (waitRuns run' k ((List.finRange K).map σ) 0 : ℝ)) m
  have h0 := waitRuns_exp run' k K 0 (Nat.zero_le _)
  have hε := Pr_nonneg fun d => (run' d).out = .acc
  calc Pr (fun d => (run' d).out = .acc) *
        Ex (fun ρ : Fin m → Fin K → D => (((waitsProg m k K opens assemble check ρ).eval H run').2.2 : ℝ))
      ≤ Pr (fun d => (run' d).out = .acc) * (m * waitExp run' k K 0) :=
        mul_le_mul_of_nonneg_left (hE ▸ Ex_mono hle) hε
    _ = m * (Pr (fun d => (run' d).out = .acc) * waitExp run' k K 0) := by ring
    _ ≤ m * k := by
        simp only [Nat.cast_zero, sub_zero] at h0
        exact mul_le_mul_of_nonneg_left h0 (Nat.cast_nonneg _)

/-- The waits rerun at most once per draw they hold. -/
theorem waits_runs_le {D W Op : Type} (H : Msg → Msg) (run' : D → Trace) (m k K : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (ρ : Fin m → Fin K → D) :
    ((waitsProg m k K opens assemble check ρ).eval H run').2.2 ≤ m * K := by
  refine (firstWait_runs_le H run' k opens assemble check _).trans_eq ?_
  simp only [List.map_map, Function.comp_def, List.length_map, List.length_finRange, List.map_const',
    List.sum_replicate, smul_eq_mul]

/-- **The sampled `E^rep`**: `SExt` with each drawn region's extractor `E^rep`'s waits over coin vectors, fresh on the
region's rounds. -/
def SERep {Coin W T Op : Type} (V : Verifier Coin) [Fintype Coin] [Nonempty Coin] (n : Msg → ℕ)
    (I : (x : Msg) → Fin (n x) → Finset (Fin (V.k x))) (sel : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x)))
    (m k K : Msg → ℕ) (opens : (x : Msg) → Fin (n x) → Seen → List Op)
    (assemble : (x : Msg) → Fin (n x) → List Op → Option T) (check : (x : Msg) → Fin (n x) → T → Bool)
    (finish : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x) × Option T) → Option W) : Extractor Coin W V :=
  SExt V n I sel (fun x => Fin (m x) → Fin (K x) → Fin (V.k x) → Coin)
    (fun x i ρ => waitsProg (m x) (k x) (K x) (opens x i) (assemble x i) (check x i) ρ) finish

/-- **The sampled `E^rep`'s cost**: at most `1 + n_max·m·K` reruns, and `1 + m·k·𝔼|sel|` in expectation, when the draw
ignores every region's rounds. -/
theorem SERep_cost {Coin W T Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (n : Msg → ℕ) (I : (x : Msg) → Fin (n x) → Finset (Fin (V.k x)))
    (sel : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x))) (m k K : Msg → ℕ)
    (opens : (x : Msg) → Fin (n x) → Seen → List Op) (assemble : (x : Msg) → Fin (n x) → List Op → Option T)
    (check : (x : Msg) → Fin (n x) → T → Bool)
    (finish : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x) × Option T) → Option W) (nmax : Msg → ℕ)
    (hlen : ∀ x r, (sel x r).length ≤ nmax x) (hsel : ∀ x i r d, sel x (forkOn (I x i) r d) = sel x r) :
    ExtractRunsLe H (SERep V n I sel m k K opens assemble check finish) (fun x => 1 + nmax x * (m x * K x)) ∧
      ExtractRunsExpLe H (SERep V n I sel m k K opens assemble check finish)
        (fun x => 1 + (m x * k x : ℝ) * Ex (fun r : Fin (V.k x) → Coin => ((sel x r).length : ℝ))) :=
  ⟨SExt_runs_le H V n I sel _ _ finish nmax (fun x => m x * K x) hlen
      fun _ _ c run' => waits_runs_le H run' _ _ _ _ _ _ c,
    SExt_runs_exp H V n I sel _ _ finish (fun x => (m x * k x : ℝ)) hsel
      fun _ _ run' => waits_amortized H run' _ _ _ _ _ _⟩

/-! ## SHA-512 calls -/

/-- A rerun program that makes no SHA-512 call of its own: its calls are its reruns'. -/
inductive NoHash {R α : Type} : RP R α → Prop
  | ret (a : α) : NoHash (.ret a)
  | rerun (r : R) (k : Seen → RP R α) : (∀ v, NoHash (k v)) → NoHash (.rerun r k)

theorem NoHash.bind {R α β : Type} {p : RP R α} {f : α → RP R β} (hp : NoHash p) (hf : ∀ a, NoHash (f a)) :
    NoHash (p.bind f) := by
  induction hp with
  | ret a => exact hf a
  | rerun r k _ ih => exact NoHash.rerun r _ fun v => ih v

theorem NoHash.comap {R R' α : Type} {f : R' → R} {p : RP R' α} (hp : NoHash p) : NoHash (comap f p) := by
  induction hp with
  | ret a => exact NoHash.ret a
  | rerun r k _ ih => exact NoHash.rerun (f r) _ fun v => ih v

theorem NoHash.forEach {R ι α : Type} {p : ι → RP R α} (hp : ∀ i, NoHash (p i)) :
    ∀ l : List ι, NoHash (Sampling.forEach p l) := by
  intro l
  induction l with
  | nil => exact NoHash.ret _
  | cons i l ih => exact (hp i).bind fun a => ih.bind fun rest => NoHash.ret _

theorem NoHash.waitAcc {R : Type} (k : ℕ) : ∀ (rs : List R) (acc : List Seen), NoHash (Claims.waitAcc k rs acc) := by
  intro rs
  induction rs with
  | nil => intro acc; exact NoHash.ret _
  | cons r rs ih =>
    intro acc
    simp only [Claims.waitAcc]
    split
    · exact NoHash.ret _
    · exact NoHash.rerun r _ fun v => ih _

theorem NoHash.firstWait {R W Op : Type} (k : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W)
    (check : W → Bool) : ∀ rss : List (List R), NoHash (Claims.firstWait k opens assemble check rss) := by
  intro rss
  induction rss with
  | nil => exact NoHash.ret _
  | cons rs rss ih =>
    refine (NoHash.waitAcc k rs []).bind fun vs => ?_
    split
    · split
      · exact NoHash.ret _
      · exact ih
    · exact ih

/-- **A program without SHA-512 calls of its own** calls SHA-512 at most `t` times per rerun, when every run makes at
most `t` calls. -/
theorem calls_le_of_noHash {R α : Type} (H : Msg → Msg) (run : R → Trace) (t : ℕ)
    (ht : ∀ r, (run r).pLog.length + (run r).vLog.length ≤ t) {p : RP R α} (hp : NoHash p) :
    (p.eval H run).2.1 ≤ t * (p.eval H run).2.2 := by
  induction hp with
  | ret a => simp [RP.eval]
  | rerun r k _ ih =>
    simp only [RP.eval]
    have := ih (run r).seen
    have := ht r
    rw [Nat.mul_succ]
    omega

/-- **The sampled extractor's SHA-512 calls**: with region extractors that make none of their own, at most `t` per
rerun on a prover whose runs make at most `t` each. -/
theorem SExt_calls_le {Ω Coin W T : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (n : Msg → ℕ)
    (I : (x : Msg) → Fin (n x) → Finset (Fin (V.k x))) (sel : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x)))
    (Cr : Msg → Type) [∀ x, Fintype (Cr x)] [∀ x, Nonempty (Cr x)]
    (p : (x : Msg) → Fin (n x) → Cr x → RP (Fin (V.k x) → Coin) (Option T))
    (finish : (x : Msg) → (Fin (V.k x) → Coin) → List (Fin (n x) × Option T) → Option W)
    (hp : ∀ x i c, NoHash (p x i c)) (P : Prover Ω) (x : Msg) (ω : Ω) (t : ℕ)
    (ht : ∀ r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) (r₀ : Fin (V.k x) → Coin)
    (c : Fin (n x) → Cr x) :
    ((SExt V n I sel Cr p finish).eval H P x ω r₀ c).2.1 ≤ t * ((SExt V n I sel Cr p finish).eval H P x ω r₀ c).2.2 := by
  refine calls_le_of_noHash H _ t ht (NoHash.rerun r₀ _ fun v₀ => ?_)
  split
  · exact (NoHash.forEach (fun i => (hp x i (c i)).comap) _).bind fun res => NoHash.ret _
  · exact NoHash.ret _

end Claims.Sampling
