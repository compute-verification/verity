import Proofs.Flock.Claims.Recursion.Wald

/-!
# The composed extractor: the outer extractor, then the inner one

A recursion's prover proves, in an outer session, that the inner verifier accepts its hidden inner transcript. Its
extractor composes two: the inner stage is `E^rep` (`m` waits of at most `K` draws, each stopping at `k` accepted
reruns), and each rerun it makes is served by the outer extractor `E₂` at that rerun's coins, whose output (the outer
witness: the hidden inner transcript) `dec` reads as what the inner stage sees. Its draws are the prover's run coins
with `E₂`'s coins; its judged draw is the judged run's coins `r₀` with `E₂`'s coins of its own (`servedERep`).

* **Cost** (`servedERep_runs_le`, `servedERep_runs_exp`): `E₂`'s reruns times the inner stage's, `(1 + m·K)·N₂` in the
  worst case and `(1 + m·k)·N₂` in expectation, for `E₂` within `N₂` (`ExtractRunsLe`, `ExtractRunsExpLe`); with
  `E₂ = E^rep` at `(m₂, k₂, K₂)`, `N₂ = 1 + m₂·K₂` and `1 + m₂·k₂`. The SHA-512 calls likewise
  (`servedERep_hashes_le`, `servedERep_hashes_exp`): the inner stage makes none of its own, so they are `E₂`'s at each
  of its reruns.
* **Knowledge soundness** (`servedERep_bad_le`): the composed extractor fails on an accepted run only if `E₂` fails at
  the judged run, or `E₂` succeeds there and the inner stage fails (`PrBadServed`, the inner stage's failure at the
  outer's output). So the outer's `KS`/`KSLin` and a bound on the inner's term give the composed extractor's `KS`
  (`KSLin_of_servedERep`, `KS_servedERep`, `KSLin_servedERep`), with an extraction-based finder's linear term folded in
  through `Finder.either`, which runs two finders and outputs a collision either finds.
-/

namespace Claims

/-! ## Averages over regrouped coins -/

theorem Ex_assoc {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ] (g : α × (β × γ) → ℝ) :
    Ex g = Ex fun s : (α × β) × γ => g (s.1.1, (s.1.2, s.2)) := by
  unfold Ex
  rw [← Fintype.card_congr (Equiv.prodAssoc α β γ)]
  congr 1
  exact ((Equiv.prodAssoc α β γ).sum_comp g).symm

/-- An event of the first three coordinates of `α × β × (γ × δ)` has its probability over `α × β × γ`. -/
theorem Pr_drop_last {α β γ δ : Type} [Fintype α] [Fintype β] [Fintype γ] [Fintype δ] [Nonempty δ]
    (p : α → β → γ → Prop) :
    Pr (fun s : α × β × (γ × δ) => p s.1 s.2.1 s.2.2.1) = Pr (fun s : α × β × γ => p s.1 s.2.1 s.2.2) := by
  classical
  rw [Pr_eq_sum (fun s : α × β × (γ × δ) => p s.1 s.2.1 s.2.2.1), Pr_eq_sum (fun s : α × β × γ => p s.1 s.2.1 s.2.2)]
  simp only [Fintype.sum_prod_type, Fintype.card_prod]
  have : (0 : ℝ) < Fintype.card δ := by exact_mod_cast Fintype.card_pos
  have e : ∀ a b c, (∑ _d : δ, if p a b c then (1 : ℝ) else 0) = Fintype.card δ * (if p a b c then 1 else 0) := by
    intro a b c; split_ifs <;> simp
  simp only [e, ← Finset.mul_sum]
  push_cast
  field_simp

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  rw [← Ex_ind, ← Ex_ind, ← Ex_ind, ← Ex_add]
  refine Ex_mono fun a => ?_
  by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq]

theorem Pr_snd {α β : Type} [Fintype α] [Fintype β] [Nonempty α] (p : β → Prop) :
    Pr (fun s : α × β => p s.2) = Pr p := by
  classical
  rw [← Ex_ind, ← Ex_ind, Ex_prod]
  exact Ex_const (α := α) (Ex fun b : β => if p b then (1 : ℝ) else 0)

/-! ## Both finders -/

/-- **Both finders**: each on its own coins, the first's pair when it is a collision of `H`, else the second's; its
cost is both finders'. -/
def Finder.either (H : Msg → Msg) (F₁ F₂ : Finder) : Finder where
  S := F₁.S × F₂.S
  out s := match F₁.out s.1 with
    | some (u, v) => if u ≠ v ∧ H u = H v then some (u, v) else F₂.out s.2
    | none => F₂.out s.2
  cost s := F₁.cost s.1 + F₂.cost s.2

theorem Finder.Pr_le_either_left (H : Msg → Msg) (F₁ F₂ : Finder) : F₁.Pr H ≤ (Finder.either H F₁ F₂).Pr H := by
  unfold Finder.Pr
  rw [← Pr_fst (β := F₂.S)]
  refine Pr_mono fun s ⟨u, v, h, hc⟩ => ⟨u, v, ?_, hc⟩
  simp only [Finder.either, h]
  rw [ite_eq_left_iff]
  exact fun hn => absurd hc hn

theorem Finder.Pr_le_either_right (H : Msg → Msg) (F₁ F₂ : Finder) : F₂.Pr H ≤ (Finder.either H F₁ F₂).Pr H := by
  unfold Finder.Pr
  rw [← Pr_snd (α := F₁.S)]
  refine Pr_mono fun s ⟨u, v, h, hc⟩ => ?_
  show ∃ u v, (Finder.either H F₁ F₂).out s = some (u, v) ∧ Collision H u v
  simp only [Finder.either]
  split
  · rename_i u' v' _
    split_ifs with hc'
    · exact ⟨u', v', rfl, hc'⟩
    · exact ⟨u, v, h, hc⟩
  · exact ⟨u, v, h, hc⟩

/-- Two linear terms into one: `β₁·Pr[F₁] + β₂·Pr[F₂] ≤ (β₁ + β₂)·Pr[either]`. -/
theorem Finder.lin_le_either (H : Msg → Msg) (F₁ F₂ : Finder) {β₁ β₂ : ℝ} (h₁ : 0 ≤ β₁) (h₂ : 0 ≤ β₂) :
    β₁ * F₁.Pr H + β₂ * F₂.Pr H ≤ (β₁ + β₂) * (Finder.either H F₁ F₂).Pr H := by
  have := mul_le_mul_of_nonneg_left (Finder.Pr_le_either_left H F₁ F₂) h₁
  have := mul_le_mul_of_nonneg_left (Finder.Pr_le_either_right H F₁ F₂) h₂
  linarith

/-- Both finder families, one prover and statement at a time. -/
def FinderFamily.either (H : Msg → Msg) (Fe₁ Fe₂ : FinderFamily) : FinderFamily :=
  fun Ω _ _ P x => Finder.either H (Fe₁ Ω P x) (Fe₂ Ω P x)

/-! ## The served `E^rep` -/

section Served

variable {Coin W W₂ Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The outer stage at a draw**: `E₂` at the draw's run coins and its own coins, its output read by `dec` as what the
inner stage sees. -/
def outerStage {V : Verifier Coin} (E₂ : Extractor Coin W₂ V) (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen)
    (x : Msg) (d : (Fin (V.k x) → Coin) × E₂.C x) : RP (Fin (V.k x) → Coin) Seen :=
  (E₂.ext x d.1 d.2).bind fun w₂ => .ret (dec x d.1 w₂)

/-- **The served `E^rep`, the composed extractor**: `E^rep` over draws of the run's coins and `E₂`'s coins, its judged
draw the judged run's coins `r₀` with coins `c.1` of its own, each draw's rerun served by `E₂` there (`outerStage`). -/
def servedERep (V : Verifier Coin) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) : Extractor Coin W V where
  C x := E₂.C x × (Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x)
  ext x r₀ c := (erepProg (m x) (k x) (K x) opens (assemble x) (check x) (r₀, c.1) c.2).serve (outerStage E₂ dec x)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem outerStage_eval {V : Verifier Coin} (H : Msg → Msg) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (x : Msg) (run : (Fin (V.k x) → Coin) → Trace)
    (d : (Fin (V.k x) → Coin) × E₂.C x) :
    (outerStage E₂ dec x d).eval H run =
      (dec x d.1 ((E₂.ext x d.1 d.2).eval H run).1, ((E₂.ext x d.1 d.2).eval H run).2.1,
        ((E₂.ext x d.1 d.2).eval H run).2.2) :=
  RP.eval_bind_ret H run _ _

/-- **The composed extractor's worst case**: `E₂`'s `N₂` reruns at each of the inner stage's at most `1 + m·K`. -/
theorem servedERep_runs_le (H : Msg → Msg) (V : Verifier Coin) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) {N₂ : Msg → ℕ} (hE₂ : ExtractRunsLe H E₂ N₂) :
    ExtractRunsLe H (servedERep V E₂ dec m k K opens assemble check) fun x => (1 + m x * K x) * N₂ x := by
  intro x Ω P ω r₀ c
  refine RP.serve_runs_le H _ _ _ (erepProg_runs_le H _ _ _ _ _ _ _ _ _) fun d => ?_
  rw [outerStage_eval]
  exact hE₂ x Ω P ω d.1 d.2

/-- **The composed extractor's expected reruns**: `E₂`'s `N₂` in expectation at each of the inner stage's `1 + m·k`. -/
theorem servedERep_runs_exp (H : Msg → Msg) (V : Verifier Coin) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) {N₂ : Msg → ℝ}
    (hE₂ : ExtractRunsExpLe H E₂ N₂) :
    ExtractRunsExpLe H (servedERep V E₂ dec m k K opens assemble check) fun x => (1 + m x * k x) * N₂ x := by
  intro x Ω P ω
  set run₂ := run H P V x ω
  set w : (Fin (V.k x) → Coin) × E₂.C x → ℝ := fun d => (((outerStage E₂ dec x d).eval H run₂).2.2 : ℝ)
  have hw : ∀ d, 0 ≤ w d := fun d => Nat.cast_nonneg _
  have hEw : Ex w ≤ N₂ x := by
    refine le_of_eq_of_le (congrArg Ex (funext fun d => ?_)) (hE₂ x Ω P ω)
    simp only [w, outerStage_eval]
    rfl
  have hk : (0 : ℝ) ≤ 1 + m x * k x := by positivity
  calc Ex (fun s : (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
        (((servedERep V E₂ dec m k K opens assemble check).eval H P x ω s.1 s.2).2.2 : ℝ))
      = Ex (fun s : ((Fin (V.k x) → Coin) × E₂.C x) ×
            (Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x) =>
          (((erepProg (m x) (k x) (K x) opens (assemble x) (check x) s.1 s.2).rerunsAt H
            (servedRun H (outerStage E₂ dec x) run₂)).map w).sum) := by
        refine (Ex_assoc (α := Fin (V.k x) → Coin) (β := E₂.C x)
          (γ := Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x) _).trans ?_
        refine congrArg Ex (funext fun s => ?_)
        exact RP.serve_runs_cast H (outerStage E₂ dec x) run₂ _
    _ ≤ (1 + m x * k x) * Ex w := erepProg_cost_exp H _ hw _ _ _ _ _ _
    _ ≤ (1 + m x * k x) * N₂ x := mul_le_mul_of_nonneg_left hEw hk

/-- A wait makes no SHA-512 call of its own: against runs that log none, it makes none. -/
theorem waitAcc_hashes {R : Type} (H : Msg → Msg) (run : R → Trace) (hrun : ∀ r, (run r).pLog = [] ∧ (run r).vLog = [])
    (k : ℕ) : ∀ (rs : List R) (acc : List Seen), ((waitAcc k rs acc).eval H run).2.1 = 0 := by
  intro rs
  induction rs with
  | nil => intro acc; rfl
  | cons r rs ih =>
    intro acc
    simp only [waitAcc]
    split
    · rfl
    · simp only [RP.eval, ih, (hrun r).1, (hrun r).2, List.length_nil]

theorem firstWait_hashes {R W' Op' : Type} (H : Msg → Msg) (run : R → Trace)
    (hrun : ∀ r, (run r).pLog = [] ∧ (run r).vLog = []) (k : ℕ) (opens : Seen → List Op')
    (assemble : List Op' → Option W') (check : W' → Bool) :
    ∀ rss : List (List R), ((firstWait k opens assemble check rss).eval H run).2.1 = 0 := by
  intro rss
  induction rss with
  | nil => rfl
  | cons rs rss ih =>
    simp only [firstWait]
    rw [RP.eval_bind_hashes, waitAcc_hashes H run hrun, zero_add]
    split
    · split
      · rfl
      · exact ih
    · exact ih

/-- **`E^rep` makes no SHA-512 call of its own**: against runs that log none, it makes none. -/
theorem erepProg_hashes {D W' Op' : Type} (H : Msg → Msg) (run : D → Trace)
    (hrun : ∀ d, (run d).pLog = [] ∧ (run d).vLog = []) (m k K : ℕ) (opens : Seen → List Op')
    (assemble : List Op' → Option W') (check : W' → Bool) (d₀ : D) (ρ : Fin m → Fin K → D) :
    ((erepProg m k K opens assemble check d₀ ρ).eval H run).2.1 = 0 := by
  simp only [erepProg, RP.eval, (hrun d₀).1, (hrun d₀).2, List.length_nil, add_zero]
  split
  · exact firstWait_hashes H run hrun k opens assemble check _
  · rfl

theorem servedRun_logs {R₁ R₂ : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Trace) (r : R₁) :
    (servedRun H q run r).pLog = [] ∧ (servedRun H q run r).vLog = [] := ⟨rfl, rfl⟩

/-- **The composed extractor's SHA-512 calls, worst case**: the inner stage makes none of its own, so they are at most
`N_h` for `E₂` at each of the inner stage's at most `1 + m·K` reruns. -/
theorem servedERep_hashes_le (H : Msg → Msg) (V : Verifier Coin) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) {Nh : Msg → ℕ}
    (hE₂ : ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : E₂.C x),
      (E₂.eval H P x ω r₀ c).2.1 ≤ Nh x)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (V.k x) → Coin)
    (c : (servedERep V E₂ dec m k K opens assemble check).C x) :
    ((servedERep V E₂ dec m k K opens assemble check).eval H P x ω r₀ c).2.1 ≤ (1 + m x * K x) * Nh x := by
  have h := RP.serve_hashes_le H (outerStage E₂ dec x) (run H P V x ω)
    (erepProg (m x) (k x) (K x) opens (assemble x) (check x) (r₀, c.1) c.2)
    (erepProg_runs_le H _ _ _ _ _ _ _ _ _) fun d => by rw [outerStage_eval]; exact hE₂ x Ω P ω d.1 d.2
  rwa [erepProg_hashes H _ (servedRun_logs H _ _), zero_add] at h

/-- **The composed extractor's expected SHA-512 calls**: at most `(1 + m·k)` times `E₂`'s expected ones, `N_h`. -/
theorem servedERep_hashes_exp (H : Msg → Msg) (V : Verifier Coin) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) {Nh : Msg → ℝ}
    (hE₂ : ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω),
      Ex (fun s : (Fin (V.k x) → Coin) × E₂.C x => ((E₂.eval H P x ω s.1 s.2).2.1 : ℝ)) ≤ Nh x)
    (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) :
    Ex (fun s : (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
      (((servedERep V E₂ dec m k K opens assemble check).eval H P x ω s.1 s.2).2.1 : ℝ)) ≤
      (1 + m x * k x) * Nh x := by
  set run₂ := run H P V x ω
  set w : (Fin (V.k x) → Coin) × E₂.C x → ℝ := fun d => (((outerStage E₂ dec x d).eval H run₂).2.1 : ℝ)
  have hw : ∀ d, 0 ≤ w d := fun d => Nat.cast_nonneg _
  have hEw : Ex w ≤ Nh x := by
    refine le_of_eq_of_le (congrArg Ex (funext fun d => ?_)) (hE₂ x Ω P ω)
    simp only [w, outerStage_eval]
    rfl
  have hk : (0 : ℝ) ≤ 1 + m x * k x := by positivity
  calc Ex (fun s : (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
        (((servedERep V E₂ dec m k K opens assemble check).eval H P x ω s.1 s.2).2.1 : ℝ))
      = Ex (fun s : ((Fin (V.k x) → Coin) × E₂.C x) ×
            (Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x) =>
          (((erepProg (m x) (k x) (K x) opens (assemble x) (check x) s.1 s.2).rerunsAt H
            (servedRun H (outerStage E₂ dec x) run₂)).map w).sum) := by
        refine (Ex_assoc (α := Fin (V.k x) → Coin) (β := E₂.C x)
          (γ := Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x) _).trans ?_
        refine congrArg Ex (funext fun s => ?_)
        rw [show ((servedERep V E₂ dec m k K opens assemble check).eval H P x ω s.1.1 (s.1.2, s.2)).2.1 =
          (((erepProg (m x) (k x) (K x) opens (assemble x) (check x) s.1 s.2).serve (outerStage E₂ dec x)).eval H
            run₂).2.1 from rfl, RP.serve_hashes_cast, erepProg_hashes H _ (servedRun_logs H _ _), Nat.cast_zero,
          zero_add]
    _ ≤ (1 + m x * k x) * Ex w := erepProg_cost_exp H _ hw _ _ _ _ _ _
    _ ≤ (1 + m x * k x) * Nh x := mul_le_mul_of_nonneg_left hEw hk

/-! ## Knowledge soundness, composed -/

/-- **The inner stage's failure at the outer stage's output**: `Pr[run = acc ∧ R₂(x, E₂(x)) ∧ ¬R(x, E(x))]`, `E₂` at the
judged run and its own coins in the composed extractor's, `E` the composed extractor. -/
noncomputable def PrBadServed (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (R₂ : Rel W₂)
    (E₂ : Extractor Coin W₂ V) (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type}
    [Fintype Ω] (P : Prover Ω) : ℝ :=
  Pr fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
    (run H P V x s.1 s.2.1).out = .acc ∧ R₂.ext x (E₂.eval H P x s.1 s.2.1 s.2.2.1).1 ∧
      ¬ R.ext x ((servedERep V E₂ dec m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1

/-- **The composition's failure**: the composed extractor fails on an accepted run at most as often as `E₂` does, plus
the inner stage's failure at `E₂`'s output. -/
theorem servedERep_bad_le (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (R₂ : Rel W₂) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} [Fintype Ω]
    (P : Prover Ω) :
    PrBad H V R (servedERep V E₂ dec m k K opens assemble check) x P ≤
      PrBad H V R₂ E₂ x P + PrBadServed H V R R₂ E₂ dec m k K opens assemble check x P := by
  unfold PrBad PrBadServed
  calc Pr (fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
        (run H P V x s.1 s.2.1).out = .acc ∧
          ¬ R.ext x ((servedERep V E₂ dec m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
          ((run H P V x s.1 s.2.1).out = .acc ∧ ¬ R₂.ext x (E₂.eval H P x s.1 s.2.1 s.2.2.1).1) ∨
          ((run H P V x s.1 s.2.1).out = .acc ∧ R₂.ext x (E₂.eval H P x s.1 s.2.1 s.2.2.1).1 ∧
            ¬ R.ext x ((servedERep V E₂ dec m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1)) := by
        refine Pr_mono fun s h => ?_
        by_cases h₂ : R₂.ext x (E₂.eval H P x s.1 s.2.1 s.2.2.1).1
        · exact Or.inr ⟨h.1, h₂, h.2⟩
        · exact Or.inl ⟨h.1, h₂⟩
    _ ≤ _ := Pr_or_le _ _
    _ = _ := by
        congr 1
        exact Pr_drop_last (δ := Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x)
          fun ω r c => (run H P V x ω r).out = .acc ∧ ¬ R₂.ext x (E₂.eval H P x ω r c).1

/-- **Knowledge soundness, composed, with the inner stage's linear term**: `E₂` meets `KS` for the outer relation `R₂`,
and the inner stage's failure at `E₂`'s output is at most `κ₁ + β·Pr[Fe]`; then the composed extractor meets `KSLin`
for `R` at `κ₂ + κ₁`, `E₂`'s `γ`, and `β·Pr[Fe]`. -/
theorem KSLin_of_servedERep (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (R₂ : Rel W₂)
    (E₂ : Extractor Coin W₂ V) (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) {κ₂ γ κ₁ β : Msg → ℝ}
    {Fe : FinderFamily} (hKS₂ : KS H V R₂ E₂ κ₂ γ)
    (hin : ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBadServed H V R R₂ E₂ dec m k K opens assemble check x P ≤ κ₁ x + β x * (Fe Ω P x).Pr H) :
    KSLin H V R (servedERep V E₂ dec m k K opens assemble check) (fun x => κ₂ x + κ₁ x) γ β Fe := by
  intro x Ω _ _ P
  have h₂ := hKS₂ x Ω P
  have h₁ := hin x Ω P
  have := servedERep_bad_le H V R R₂ E₂ dec m k K opens assemble check x P
  linarith

/-- **Knowledge soundness, composed, both stages to `B`**: `E₂` meets `KS` at `γ₂`, and the inner stage's failure at
`E₂`'s output is at most `κ₁ + √(γ₁·Pr[B])` at the composed verifier's `B`; then the composed extractor meets `KS` at
`κ₂ + κ₁` and `(√γ₂ + √γ₁)²`. -/
theorem KS_servedERep (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (R₂ : Rel W₂) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) {κ₂ γ₂ κ₁ γ₁ : Msg → ℝ}
    (hγ₂ : ∀ x, 0 ≤ γ₂ x) (hγ₁ : ∀ x, 0 ≤ γ₁ x) (hKS₂ : KS H V R₂ E₂ κ₂ γ₂)
    (hin : ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBadServed H V R R₂ E₂ dec m k K opens assemble check x P ≤ κ₁ x + Real.sqrt (γ₁ x * PrB H V P x)) :
    KS H V R (servedERep V E₂ dec m k K opens assemble check) (fun x => κ₂ x + κ₁ x)
      (fun x => (Real.sqrt (γ₂ x) + Real.sqrt (γ₁ x)) ^ 2) := by
  intro x Ω _ _ P
  have h₂ := hKS₂ x Ω P
  have h₁ := hin x Ω P
  have hbad := servedERep_bad_le H V R R₂ E₂ dec m k K opens assemble check x P
  have hp : 0 ≤ PrB H V P x := by
    unfold PrB; split
    · exact Pr_nonneg _
    · exact le_rfl
  have hsq : Real.sqrt (γ₂ x * PrB H V P x) + Real.sqrt (γ₁ x * PrB H V P x) =
      Real.sqrt ((Real.sqrt (γ₂ x) + Real.sqrt (γ₁ x)) ^ 2 * PrB H V P x) := by
    rw [Real.sqrt_mul (hγ₂ x), Real.sqrt_mul (hγ₁ x), Real.sqrt_mul (sq_nonneg _),
      Real.sqrt_sq (add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))]
    ring
  linarith

/-- **Knowledge soundness, composed, both stages with linear terms**: `E₂` meets `KSLin` at `β₂·Pr[Fe₂]`, and the inner
stage's failure at `E₂`'s output is at most `κ₁ + β₁·Pr[Fe₁]`; then the composed extractor meets `KSLin` at `κ₂ + κ₁`,
`E₂`'s `γ`, and `(β₂ + β₁)·Pr[Fe₂ or Fe₁]` (`FinderFamily.either`). -/
theorem KSLin_servedERep (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (R₂ : Rel W₂) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) {κ₂ γ β₂ κ₁ β₁ : Msg → ℝ}
    {Fe₂ Fe₁ : FinderFamily} (hβ₂ : ∀ x, 0 ≤ β₂ x) (hβ₁ : ∀ x, 0 ≤ β₁ x) (hKS₂ : KSLin H V R₂ E₂ κ₂ γ β₂ Fe₂)
    (hin : ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBadServed H V R R₂ E₂ dec m k K opens assemble check x P ≤ κ₁ x + β₁ x * (Fe₁ Ω P x).Pr H) :
    KSLin H V R (servedERep V E₂ dec m k K opens assemble check) (fun x => κ₂ x + κ₁ x) γ
      (fun x => β₂ x + β₁ x) (FinderFamily.either H Fe₂ Fe₁) := by
  intro x Ω _ _ P
  have h₂ := hKS₂ x Ω P
  have h₁ := hin x Ω P
  have hbad := servedERep_bad_le H V R R₂ E₂ dec m k K opens assemble check x P
  have hlin := Finder.lin_le_either H (Fe₂ Ω P x) (Fe₁ Ω P x) (hβ₂ x) (hβ₁ x)
  show _ ≤ _ + _ + (β₂ x + β₁ x) * (Finder.either H (Fe₂ Ω P x) (Fe₁ Ω P x)).Pr H
  linarith

end Served

end Claims
