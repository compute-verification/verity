module

public import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob

/-!
# `E^rep`'s waits, for a property of the candidate returned

`ERep_bad_le` factors the event that every wait fails. This file factors the event that the waits return a candidate
with a property `G`, and bounds it by the same event at one wait.

* **The waits, one at a time** (`waitsHit_cons`): they return a candidate with `G` exactly when the first wait yields
  one, or the first wait fails and the others return one.
* **Independent and identically distributed** (`waitsHit_eq`): over `m` waits of `K` fresh coin samples each, the
  probability is exactly `S · (1 + f + ⋯ + f^{m−1})`, with `S` the probability that one wait yields a candidate that
  `check` accepts and that has `G`, and `f` the probability that one wait fails. `G` is any property of the candidate,
  so it may read whatever is fixed before the waits: the oracle, the statement, the tape and the judged run's coins.
* **One tape** (`tape_hit_bound`): `tape_bound`'s cases, for that event. On a tape where `tape_bound` pays the
  acceptance probability itself (`ε ≤ 2k/K`, or `ε < 4a`) so does this; on any other tape one wait fails with
  probability at most `1/2 + t`, `4t ≤ e^{−k/8}`, so `1 + f + ⋯ + f^{m−1} ≤ 2 + 8t`.
* **`E^rep` against `E^rep` with one wait** (`ERep_hit_le`): under `ERep_trunc`'s hypotheses, the probability that the
  judged run is accepted and `m` waits return a candidate with `G` is at most twice the probability that it is
  accepted and one wait does, plus `4·𝔼[a] + 2k/K + 2e^{−k/8}`.
-/

@[expose] public section

namespace Claims

/-! ## One wait, and the waits, for a property of the candidate -/

/-- One wait over the coin samples `rs` yields a candidate that `check` accepts and that has `G`. -/
def WaitHit {R W Op : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (G : W → Prop) (rs : List R) : Prop :=
  ∃ w, assemble (((waitAcc k rs []).eval H run).1.flatMap opens) = some w ∧ check w = true ∧ G w

/-- The waits over the coin samples `rss` return a candidate that has `G`. -/
def WaitsHit {R W Op : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (G : W → Prop) (rss : List (List R)) : Prop :=
  ∃ w, ((firstWait k opens assemble check rss).eval H run).1 = some w ∧ G w

/-- A wait that yields a candidate with `G` did not fail. -/
theorem WaitHit.ok {R W Op : Type} {H : Msg → Msg} {run : R → Transcript} {k : ℕ} {opens : Seen → List Op}
    {assemble : List Op → Option W} {check : W → Bool} {G : W → Prop} {rs : List R}
    (h : WaitHit H run k opens assemble check G rs) : WaitOk H run k opens assemble check rs := by
  obtain ⟨w, hw, hc, _⟩ := h
  exact ⟨w, hw, hc⟩

/-- No wait returns nothing. -/
theorem waitsHit_nil {R W Op : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (G : W → Prop) :
    ¬ WaitsHit H run k opens assemble check G ([] : List (List R)) := by
  rintro ⟨w, hw, -⟩
  simp [firstWait, RP.eval] at hw

/-- **The waits, one at a time**: they return a candidate with `G` exactly when the first wait yields one, or the
first wait fails and the others return one. -/
theorem waitsHit_cons {R W Op : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (G : W → Prop) (rs : List R) (rss : List (List R)) :
    WaitsHit H run k opens assemble check G (rs :: rss) ↔
      WaitHit H run k opens assemble check G rs ∨
        (¬ WaitOk H run k opens assemble check rs ∧ WaitsHit H run k opens assemble check G rss) := by
  unfold WaitsHit WaitHit WaitOk
  simp only [firstWait]
  rw [(RP.eval_bind H run _ _).1]
  cases hw : assemble (((waitAcc k rs []).eval H run).1.flatMap opens) with
  | none => simp
  | some w₀ =>
    by_cases hc : check w₀ = true
    · simp [hc, RP.eval]
    · simp [hc]

/-! ## Probability -/

/-- An event that holds outright (`A`), or else only on `C` and only when `B` fails, where `A` implies `B`. -/
theorem Pr_or_not_and {α : Type} [Fintype α] [Nonempty α] {A B : Prop} [Decidable A] [Decidable B] (hAB : A → B)
    (C : α → Prop) :
    Pr (fun a => A ∨ (¬ B ∧ C a)) = (if A then (1 : ℝ) else 0) + (if B then (0 : ℝ) else 1) * Pr C := by
  by_cases hA : A
  · have hB : B := hAB hA
    have e : Pr (fun a : α => A ∨ (¬ B ∧ C a)) = Pr (fun _ : α => True) :=
      Pr_congr fun _ => ⟨fun _ => trivial, fun _ => Or.inl hA⟩
    rw [e, Pr_const_of trivial]
    simp only [hA, hB, ↓reduceIte, zero_mul, add_zero]
  · by_cases hB : B
    · have e : Pr (fun a : α => A ∨ (¬ B ∧ C a)) = Pr (fun _ : α => False) :=
        Pr_congr fun _ => ⟨fun h => h.elim hA fun h => h.1 hB, False.elim⟩
      rw [e, Pr_false]
      simp only [hA, hB, ↓reduceIte, zero_mul, add_zero]
    · have e : Pr (fun a : α => A ∨ (¬ B ∧ C a)) = Pr C :=
        Pr_congr fun _ => ⟨fun h => h.elim (fun h => absurd h hA) fun h => h.2, fun h => Or.inr ⟨hB, h⟩⟩
      rw [e]
      simp only [hA, hB, ↓reduceIte, zero_add, one_mul]

/-- The complement's probability, as an average. -/
theorem Ex_ind_not {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] :
    Ex (fun a => if p a then (0 : ℝ) else 1) = Pr fun a => ¬ p a := by
  rw [← Ex_ind fun a => ¬ p a]
  congr 1; funext a
  by_cases h : p a <;> simp [h]

/-- **The waits are independent and identically distributed, for a property of the candidate**: over `m` waits of `K`
fresh coin samples each, the waits return a candidate with `G` with probability exactly `S · (1 + f + ⋯ + f^{m−1})`,
`S` the probability that one wait yields a candidate that `check` accepts and that has `G`, and `f` the probability
that one wait fails. `G` is any property of the candidate, so it may read whatever is fixed before the waits. -/
theorem waitsHit_eq {D W Op : Type} [Fintype D] [Nonempty D] (H : Msg → Msg) (run : D → Transcript) (k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) (G : W → Prop) :
    ∀ m : ℕ,
      Pr (fun ρ : Fin m → Fin K → D => WaitsHit H run k opens assemble check G
          ((List.finRange m).map fun i => (List.finRange K).map (ρ i))) =
        Pr (fun σ : Fin K → D => WaitHit H run k opens assemble check G ((List.finRange K).map σ)) *
          ∑ i ∈ Finset.range m,
            Pr (fun σ : Fin K → D => ¬ WaitOk H run k opens assemble check ((List.finRange K).map σ)) ^ i
  | 0 => by
    have h0 : ∀ ρ : Fin 0 → Fin K → D, WaitsHit H run k opens assemble check G
        ((List.finRange 0).map fun i => (List.finRange K).map (ρ i)) ↔ False := fun ρ =>
      iff_false_intro (by
        rw [List.finRange_zero, List.map_nil]
        exact waitsHit_nil H run k opens assemble check G)
    rw [Pr_congr h0, Pr_false, Finset.range_zero, Finset.sum_empty, mul_zero]
  | m + 1 => by
    classical
    have e1 : Pr (fun ρ : Fin (m + 1) → Fin K → D => WaitsHit H run k opens assemble check G
          ((List.finRange (m + 1)).map fun i => (List.finRange K).map (ρ i))) =
        Pr (fun ρ : Fin (m + 1) → Fin K → D =>
          (fun (d : Fin K → D) (τ : Fin m → Fin K → D) =>
            WaitHit H run k opens assemble check G ((List.finRange K).map d) ∨
              (¬ WaitOk H run k opens assemble check ((List.finRange K).map d) ∧
                WaitsHit H run k opens assemble check G
                  ((List.finRange m).map fun i => (List.finRange K).map (τ i))))
            (ρ 0) fun i => ρ i.succ) := by
      refine Pr_congr fun ρ => ?_
      have hcons : ((List.finRange (m + 1)).map fun i => (List.finRange K).map (ρ i)) =
          (List.finRange K).map (ρ 0) :: (List.finRange m).map fun i => (List.finRange K).map (ρ i.succ) := by
        rw [List.finRange_succ, List.map_cons, List.map_map]; rfl
      rw [hcons]
      exact waitsHit_cons H run k opens assemble check G _ _
    rw [e1]
    refine (Pr_cons m fun (d : Fin K → D) (τ : Fin m → Fin K → D) =>
      WaitHit H run k opens assemble check G ((List.finRange K).map d) ∨
        (¬ WaitOk H run k opens assemble check ((List.finRange K).map d) ∧
          WaitsHit H run k opens assemble check G
            ((List.finRange m).map fun i => (List.finRange K).map (τ i)))).trans ?_
    refine Eq.trans (congrArg Ex (funext fun d => Pr_or_not_and WaitHit.ok _)) ?_
    rw [waitsHit_eq H run k K opens assemble check G m, Ex_add, Ex_mul_right, Ex_ind, Ex_ind_not, geom_sum_succ]
    ring

/-! ## One tape -/

/-- **One tape, for a property of the candidate**: acceptance `ε`, one wait's failure `f ≤ t + Q` as in
`tape_bound`, and `s` the probability that the judged run is accepted and one wait yields a candidate with the
property, so that `m` waits return one with probability `s · (1 + f + ⋯ + f^{m−1})`, itself at most `ε`. Then `m`
waits return one with probability at most `2s + 4a + 2k/K + 2e^{−k/8}`: on a tape where `tape_bound` pays the
acceptance itself (`ε ≤ 2k/K` or `ε < 4a`) so does this, and on any other tape `f ≤ 1/2 + t` with `4t ≤ e^{−k/8}`,
so the sum is at most `2 + 8t`. -/
theorem tape_hit_bound {ε f a t Q s : ℝ} {m k K : ℕ} (hf0 : 0 ≤ f) (ha : 0 ≤ a) (ht0 : 0 ≤ t) (hs0 : 0 ≤ s)
    (hs1 : s ≤ 1) (hsm : s * ∑ i ∈ Finset.range m, f ^ i ≤ ε) (hK : 0 < K) (hk : 8 ≤ k)
    (hf : 0 < ε → f ≤ t + Q) (hQ : 0 < ε → ε * Q ≤ a + ε / 4)
    (ht : 2 * (k : ℝ) ≤ ε * K → t ≤ 2 ^ k * Real.exp (-(k : ℝ))) :
    s * ∑ i ∈ Finset.range m, f ^ i ≤ 2 * s + 4 * a + 2 * k / K + 2 * Real.exp (-(k : ℝ) / 8) := by
  have hK' : (0 : ℝ) < K := by exact_mod_cast hK
  have hr2 : 0 ≤ 2 * (k : ℝ) / K := by positivity
  have hr3 : 0 ≤ Real.exp (-(k : ℝ) / 8) := (Real.exp_pos _).le
  by_cases h1 : ε ≤ 2 * k / K
  · linarith
  push Not at h1
  have hεpos : 0 < ε := lt_of_le_of_lt hr2 h1
  have hεK : 2 * (k : ℝ) ≤ ε * K := by rw [div_lt_iff₀ hK'] at h1; linarith
  by_cases h2 : ε / 4 < a
  · linarith
  push Not at h2
  have hQ' : Q ≤ 1 / 2 := by
    have := hQ hεpos
    by_contra hc; push Not at hc
    nlinarith
  have h4t : 4 * t ≤ Real.exp (-(k : ℝ) / 8) :=
    (mul_le_mul_of_nonneg_left (ht hεK) (by norm_num)).trans (four_two_pow_exp_le k hk)
  have hexp1 : Real.exp (-(k : ℝ) / 8) ≤ 1 := Real.exp_le_one_iff.2 (by
    have : (0 : ℝ) ≤ k := Nat.cast_nonneg k
    linarith)
  have hfc : f ≤ 1 / 2 + t := by linarith [hf hεpos]
  have hg0 : 0 ≤ ∑ i ∈ Finset.range m, f ^ i := Finset.sum_nonneg fun i _ => pow_nonneg hf0 i
  have hgeo : (1 - f) * ∑ i ∈ Finset.range m, f ^ i = 1 - f ^ m := mul_neg_geom_sum f m
  have hfm0 : 0 ≤ f ^ m := pow_nonneg hf0 m
  have hpos : 0 < 1 / 2 - t := by linarith
  have hg : ∑ i ∈ Finset.range m, f ^ i ≤ 2 + 8 * t := by
    by_contra hc; push Not at hc
    nlinarith [mul_lt_mul_of_pos_left hc hpos,
      mul_le_mul_of_nonneg_right (show 1 / 2 - t ≤ 1 - f by linarith) hg0,
      mul_nonneg ht0 (show 0 ≤ 1 - 4 * t by linarith)]
  nlinarith [mul_le_mul_of_nonneg_left hg hs0, mul_le_of_le_one_left (show 0 ≤ 8 * t by linarith) hs1]

/-- An event of the first two coordinates and one of all three, over a triple: the average over the first two of the
third's conditional probability, where the first event holds. -/
theorem Pr_and_cond {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ] [Nonempty γ] (A : α → β → Prop)
    [∀ a b, Decidable (A a b)] (B : α → β → γ → Prop) :
    Pr (fun s : α × β × γ => A s.1 s.2.1 ∧ B s.1 s.2.1 s.2.2) =
      Ex fun a => Ex fun b => if A a b then Pr (fun c => B a b c) else 0 := by
  classical
  rw [← Ex_ind (fun s : α × β × γ => A s.1 s.2.1 ∧ B s.1 s.2.1 s.2.2), Ex_prod]
  congr 1; funext a
  rw [Ex_prod]
  congr 1; funext b
  by_cases h : A a b
  · simp only [h, true_and, ↓reduceIte]
    exact Ex_ind _
  · simp only [h, false_and, ↓reduceIte]
    exact Ex_const 0

section ERep

variable {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `s_ω`: at tape `ω`, the probability that the judged run is accepted and one wait yields a candidate that `check`
accepts and that has `G` at the judged run's coins. -/
noncomputable def hitPr (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool)
    (G : (Fin (V.k x) → Coin) → W → Prop) : ℝ :=
  Ex fun r₀ : Fin (V.k x) → Coin => if (run H P V x ω r₀).out = .acc then
    Pr (fun σ : Fin K → Fin (V.k x) → Coin =>
      WaitHit H (run H P V x ω) k opens assemble check (G r₀) ((List.finRange K).map σ)) else 0

omit [Nonempty Coin] in
theorem hitPr_nonneg (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool)
    (G : (Fin (V.k x) → Coin) → W → Prop) : 0 ≤ hitPr H V P x ω k K opens assemble check G := by
  unfold hitPr
  refine Ex_nonneg fun r₀ => ?_
  split_ifs
  exacts [Pr_nonneg _, le_rfl]

theorem hitPr_le_one (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool)
    (G : (Fin (V.k x) → Coin) → W → Prop) : hitPr H V P x ω k K opens assemble check G ≤ 1 := by
  unfold hitPr
  refine (Ex_mono (g := fun _ => (1 : ℝ)) fun r₀ => ?_).trans (Ex_const 1).le
  split_ifs
  exacts [Pr_le_one _, zero_le_one]

/-- At one tape, `m` waits return a candidate with `G` on an accepted judged run no more often than the judged run is
accepted. -/
theorem hitPr_mul_le (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool)
    (G : (Fin (V.k x) → Coin) → W → Prop) (m : ℕ) :
    hitPr H V P x ω k K opens assemble check G *
        ∑ i ∈ Finset.range m, waitFail H V P x ω k K opens assemble check ^ i ≤ accPr H V P x ω := by
  unfold hitPr waitFail accPr
  rw [← Ex_mul_right, ← Ex_ind fun r : Fin (V.k x) → Coin => (run H P V x ω r).out = .acc]
  refine Ex_mono fun r₀ => ?_
  split_ifs with h
  · exact ((waitsHit_eq H (run H P V x ω) k K opens assemble check (G r₀) m).symm.le).trans (Pr_le_one _)
  · rw [zero_mul]

/-- **`E^rep` returns a candidate with `G`** on an accepted judged run with probability
`𝔼_ω[s_ω · (1 + f_ω + ⋯ + f_ω^{m−1})]`: the waits rerun the prover at the judged run's tape on fresh coins, so given
the tape and the judged run's coins they are independent and identically distributed. `G` may read the tape and the
judged run's coins. -/
theorem ERep_hit_eq (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω)
    (G : Ω → (Fin (V.k x) → Coin) → W → Prop) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × (ERep V m k K opens assemble check).C x =>
        (run H P V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((ERep V m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.1 s.2.1 w) =
      Ex fun ω => hitPr H V P x ω (k x) (K x) opens (assemble x) (check x) (G ω) *
        ∑ i ∈ Finset.range (m x), waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ i := by
  have hev : ∀ s : Ω × (Fin (V.k x) → Coin) × (ERep V m k K opens assemble check).C x,
      ((run H P V x s.1 s.2.1).out = .acc ∧
        ∃ w, ((ERep V m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.1 s.2.1 w) ↔
      ((run H P V x s.1 s.2.1).out = .acc ∧
        WaitsHit H (run H P V x s.1) (k x) opens (assemble x) (check x) (G s.1 s.2.1)
          ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (s.2.2 i))) := by
    intro s
    refine and_congr_right fun hacc => ?_
    unfold WaitsHit
    simp only [Extractor.eval, ERep, RP.eval, Transcript.seen, hacc, ↓reduceIte]
  refine (Pr_congr hev).trans ?_
  refine (Pr_and_cond (fun ω (r₀ : Fin (V.k x) → Coin) => (run H P V x ω r₀).out = .acc) fun ω r₀
    (ρ : Fin (m x) → Fin (K x) → Fin (V.k x) → Coin) =>
      WaitsHit H (run H P V x ω) (k x) opens (assemble x) (check x) (G ω r₀)
        ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (ρ i))).trans ?_
  congr 1; funext ω
  unfold hitPr waitFail
  rw [← Ex_mul_right]
  congr 1; funext r₀
  rw [waitsHit_eq]
  split_ifs
  · rfl
  · rw [zero_mul]

/-- **`E^rep` against `E^rep` with one wait, for a property of the candidate**: under `ERep_trunc`'s hypotheses
(`K > 0`, `k ≥ 8`, and each tape with acceptance `ε > 0` has one wait's knowledge bound on `k` accepted runs,
`ε · Q/ε^k ≤ a + rate·ε` with `rate ≤ 1/4`), the probability that the judged run is accepted and `m` waits return a
candidate with `G` is at most twice the probability that it is accepted and one wait does, plus
`4·𝔼[a] + 2k/K + 2e^{−k/8}`. `G` may read the tape and the judged run's coins. -/
theorem ERep_hit_le (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (G : Ω → (Fin (V.k x) → Coin) → W → Prop) (hK : 0 < K x) (hk : 8 ≤ k x)
    (a : Ω → ℝ) (ha : ∀ ω, 0 ≤ a ω) {rate : ℝ} (hrate : rate ≤ 1 / 4)
    (hwait : ∀ ω, 0 < accPr H V P x ω →
      accPr H V P x ω * (accFail H V P x ω (k x) (Assembles opens (assemble x) (check x)) /
        accPr H V P x ω ^ k x) ≤ a ω + rate * accPr H V P x ω) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × (ERep V m k K opens assemble check).C x =>
        (run H P V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((ERep V m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.1 s.2.1 w) ≤
      2 * Pr (fun s : Ω × (Fin (V.k x) → Coin) × (ERep V (fun _ => 1) k K opens assemble check).C x =>
          (run H P V x s.1 s.2.1).out = .acc ∧
            ∃ w, ((ERep V (fun _ => 1) k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧
              G s.1 s.2.1 w) +
        4 * Ex a + 2 * k x / K x + 2 * Real.exp (-(k x : ℝ) / 8) := by
  rw [ERep_hit_eq, ERep_hit_eq]
  simp only [Finset.sum_range_one, pow_zero, mul_one]
  have hpt : ∀ ω, hitPr H V P x ω (k x) (K x) opens (assemble x) (check x) (G ω) *
        ∑ i ∈ Finset.range (m x), waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ i ≤
      2 * hitPr H V P x ω (k x) (K x) opens (assemble x) (check x) (G ω) + 4 * a ω +
        (2 * k x / K x + 2 * Real.exp (-(k x : ℝ) / 8)) := by
    intro ω
    have hε0 : 0 ≤ accPr H V P x ω := Pr_nonneg _
    have hrε : rate * accPr H V P x ω ≤ accPr H V P x ω / 4 := by
      have := mul_le_mul_of_nonneg_right hrate hε0; linarith
    have h := tape_hit_bound (ε := accPr H V P x ω)
      (f := waitFail H V P x ω (k x) (K x) opens (assemble x) (check x)) (a := a ω) (m := m x)
      (Pr_nonneg _) (ha ω) (Pr_nonneg _) (hitPr_nonneg H V P x ω (k x) (K x) opens (assemble x) (check x) (G ω))
      (hitPr_le_one H V P x ω (k x) (K x) opens (assemble x) (check x) (G ω))
      (hitPr_mul_le H V P x ω (k x) (K x) opens (assemble x) (check x) (G ω) (m x)) hK hk
      (waitFail_le H V P x ω (k x) (K x) opens (assemble x) (check x))
      (fun hε => (hwait ω hε).trans (by linarith))
      (fun hεK => (wait_short_le (run H P V x ω) (k x) (K x)).trans (short_le_exp (Pr_le_one _) _ _ hεK))
    linarith
  refine (Ex_mono hpt).trans (le_of_eq ?_)
  rw [Ex_add, Ex_add, Ex_mul_left, Ex_mul_left, Ex_const]
  ring

end ERep

end Claims
