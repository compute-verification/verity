module

public import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob

/-!
# What `E^rep`'s waits find, one wait finds at half the rate

`E^rep` returns the candidate of the first of its `m` waits that assembles one `check` accepts (`firstWait_eval`,
`ERep_eval_fst`). The waits are independent given the tape. So at a tape where one wait fails with probability `f`,
the waits return a candidate with a property `G` with probability `g·(1 + f + … + f^{m−1})`, `g` the probability that
one wait returns a candidate with `G` (`Pr_findSome`, `ERep_hit`).

At a tape whose acceptance `ε` is above `2k/K` and above four times its escapes `a`, one wait fails with probability
at most `1/2 + 2^k·e^{−k}` (the cases of `tape_bound`), so the sum is at most `2 + 2·e^{−k/8}` (`geom_le_two_add`); the
other tapes accept with probability at most `2k/K + 4·a` (`tape_finds`). Averaged over the tape, under `ERep_trunc`'s
hypotheses, for every property `G` of the judged run's coins and the candidate (`ERep_finds`):

  `Pr[acc ∧ E^rep returns w ∧ G(r₀, w)] ≤ 2·Pr[acc ∧ E^rep at one wait returns w ∧ G(r₀, w)]
    + 4·𝔼[a] + 2k/K + 2·e^{−k/8}`.

`Finds` names that bound at an extractor, its one-wait form and the errors of a knowledge-soundness claim,
`2·κ + √(γ·Pr[B])` in place of the last three terms. A bound on a finder that reads `E^rep`'s candidates only through
such a property can then be stated at the one-wait extractor, which reruns the prover `1 + k` times in expectation in
place of `1 + m·k`.
-/

@[expose] public section

namespace Claims

/-! ## One wait's candidate, and the first of the waits' -/

/-- What one wait over the coin samples `rs` returns: the candidate it assembles, where `check` accepts it. -/
def waitOut {R W Op : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (rs : List R) : Option W :=
  match assemble (((waitAcc k rs []).eval H run).1.flatMap opens) with
  | some w => if check w then some w else none
  | none => none

/-- One wait returns nothing exactly when it fails (`WaitOk`). -/
theorem waitOut_eq_none {R W Op : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (rs : List R) :
    waitOut H run k opens assemble check rs = none ↔ ¬ WaitOk H run k opens assemble check rs := by
  unfold waitOut WaitOk
  cases assemble (((waitAcc k rs []).eval H run).1.flatMap opens) with
  | none => simp
  | some w => by_cases hc : check w = true <;> simp [hc]

/-- **The waits return the first wait's candidate that `check` accepts.** -/
theorem firstWait_eval {R W Op : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) :
    ∀ rss : List (List R), ((firstWait k opens assemble check rss).eval H run).1 =
      rss.findSome? (waitOut H run k opens assemble check)
  | [] => by simp [firstWait, RP.eval]
  | rs :: rss => by
    have ih := firstWait_eval H run k opens assemble check rss
    have hw : waitOut H run k opens assemble check rs =
        match assemble (((waitAcc k rs []).eval H run).1.flatMap opens) with
        | some w => if check w then some w else none
        | none => none := rfl
    simp only [firstWait, List.findSome?_cons]
    rw [(RP.eval_bind H run _ _).1, hw]
    cases assemble (((waitAcc k rs []).eval H run).1.flatMap opens) with
    | none => simpa using ih
    | some w => by_cases hc : check w = true <;> simp [hc, RP.eval, ih]

/-! ## The first of `m` independent tries -/

section FindSome

variable {δ W : Type} [Fintype δ] [Nonempty δ]

/-- **The first of `m` independent tries that returns something** has a property `G` with the probability that one
try returns something with `G`, times `1 + f + … + f^{m−1}`, `f` the probability that one try returns nothing. -/
theorem Pr_findSome (o : δ → Option W) (G : W → Prop) : ∀ m : ℕ,
    Pr (fun ρ : Fin m → δ => ∃ w, ((List.finRange m).map ρ).findSome? o = some w ∧ G w) =
      Pr (fun d : δ => ∃ w, o d = some w ∧ G w) * ∑ i ∈ Finset.range m, Pr (fun d : δ => o d = none) ^ i
  | 0 => by
    rw [Finset.range_zero, Finset.sum_empty, mul_zero]
    exact (Pr_congr fun ρ => ⟨fun ⟨w, hw, _⟩ => by simp at hw, False.elim⟩).trans Pr_false
  | m + 1 => by
    classical
    have ih := Pr_findSome o G m
    have hcons : ∀ ρ : Fin (m + 1) → δ,
        (List.finRange (m + 1)).map ρ = ρ 0 :: (List.finRange m).map fun i => ρ i.succ := by
      intro ρ; rw [List.finRange_succ, List.map_cons, List.map_map]; rfl
    have hpt : ∀ d : δ, Pr (fun τ : Fin m → δ =>
          ∃ w, (d :: (List.finRange m).map τ).findSome? o = some w ∧ G w) =
        (if ∃ w, o d = some w ∧ G w then (1 : ℝ) else 0) + (if o d = none then (1 : ℝ) else 0) *
          Pr (fun τ : Fin m → δ => ∃ w, ((List.finRange m).map τ).findSome? o = some w ∧ G w) := by
      intro d
      simp only [List.findSome?_cons]
      cases o d with
      | none => simp
      | some w₀ =>
        show Pr (fun _ : Fin m → δ => ∃ w, some w₀ = some w ∧ G w) = _
        by_cases hG : G w₀
        · have hyes : ∃ w, some w₀ = some w ∧ G w := ⟨w₀, rfl, hG⟩
          rw [Pr_const_of hyes, ite_eq_left hyes, ite_eq_right (Option.some_ne_none w₀), zero_mul, add_zero]
        · have hno : ¬ ∃ w, some w₀ = some w ∧ G w := fun ⟨w, hw, hGw⟩ => hG (Option.some.inj hw ▸ hGw)
          rw [Pr_const_of_not hno, ite_eq_right hno, ite_eq_right (Option.some_ne_none w₀), zero_mul, add_zero]
    calc Pr (fun ρ : Fin (m + 1) → δ => ∃ w, ((List.finRange (m + 1)).map ρ).findSome? o = some w ∧ G w)
        = Pr (fun ρ : Fin (m + 1) → δ => (fun (d : δ) (τ : Fin m → δ) =>
            ∃ w, (d :: (List.finRange m).map τ).findSome? o = some w ∧ G w) (ρ 0) fun i => ρ i.succ) := by
          refine Pr_congr fun ρ => ?_
          rw [hcons]
      _ = Ex fun d => Pr fun τ : Fin m → δ => ∃ w, (d :: (List.finRange m).map τ).findSome? o = some w ∧ G w :=
          Pr_cons m fun (d : δ) (τ : Fin m → δ) => ∃ w, (d :: (List.finRange m).map τ).findSome? o = some w ∧ G w
      _ = Ex fun d => (if ∃ w, o d = some w ∧ G w then (1 : ℝ) else 0) + (if o d = none then (1 : ℝ) else 0) *
            Pr (fun τ : Fin m → δ => ∃ w, ((List.finRange m).map τ).findSome? o = some w ∧ G w) :=
          congrArg Ex (funext hpt)
      _ = Pr (fun d : δ => ∃ w, o d = some w ∧ G w) + Pr (fun d : δ => o d = none) *
            Pr (fun τ : Fin m → δ => ∃ w, ((List.finRange m).map τ).findSome? o = some w ∧ G w) := by
          rw [Ex_add, Ex_mul_right, Ex_ind, Ex_ind]
      _ = _ := by
          rw [ih, Finset.sum_range_succ', pow_zero]
          simp only [pow_succ, ← Finset.sum_mul]
          ring

end FindSome

/-! ## One tape -/

/-- `1 + f + … + f^{m−1} ≤ 2 + 8·t` when `f ≤ 1/2 + t` and `t ≤ 1/4`. -/
theorem geom_le_two_add {f t : ℝ} (hf0 : 0 ≤ f) (ht0 : 0 ≤ t) (ht : t ≤ 1 / 4) (hf : f ≤ 1 / 2 + t) (m : ℕ) :
    ∑ i ∈ Finset.range m, f ^ i ≤ 2 + 8 * t := by
  have hle : ∑ i ∈ Finset.range m, f ^ i ≤ ∑ i ∈ Finset.range m, (1 / 2 + t) ^ i :=
    Finset.sum_le_sum fun i _ => pow_le_pow_left₀ hf0 hf i
  have hgeo := mul_neg_geom_sum (1 / 2 + t) m
  have hpow : 0 ≤ (1 / 2 + t) ^ m := pow_nonneg (by linarith) m
  have hS : 0 ≤ ∑ i ∈ Finset.range m, (1 / 2 + t) ^ i := Finset.sum_nonneg fun i _ => pow_nonneg (by linarith) i
  nlinarith [mul_nonneg ht0 (sub_nonneg.2 ht), mul_nonneg hS (mul_nonneg ht0 (sub_nonneg.2 ht))]

/-- **One tape**: acceptance `ε`, one wait's failure `f ≤ t + Q` with `t` the short-wait probability and `Q` the
failure on `k` accepted runs, `ε·Q ≤ a + ε/4`, as `tape_bound` takes them; `X ≤ ε` the probability that the run accepts
and one wait returns a candidate with the property, and `X·(1 + f + … + f^{m−1}) ≤ ε` that of `m` waits. Then
`X·(1 + f + … + f^{m−1}) ≤ 2·X + 4a + 2k/K + 2·e^{−k/8}`. -/
theorem tape_finds {ε f a t Q X : ℝ} {m k K : ℕ} (hε1 : ε ≤ 1) (hf0 : 0 ≤ f) (ha : 0 ≤ a) (ht0 : 0 ≤ t)
    (hK : 0 < K) (hk : 8 ≤ k) (hf : 0 < ε → f ≤ t + Q) (hQ : 0 < ε → ε * Q ≤ a + ε / 4)
    (ht : 2 * (k : ℝ) ≤ ε * K → t ≤ 2 ^ k * Real.exp (-(k : ℝ))) (hX0 : 0 ≤ X) (hXε : X ≤ ε)
    (hXS : X * ∑ i ∈ Finset.range m, f ^ i ≤ ε) :
    X * ∑ i ∈ Finset.range m, f ^ i ≤ 2 * X + (4 * a + 2 * k / K + 2 * Real.exp (-(k : ℝ) / 8)) := by
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
  have hS := geom_le_two_add hf0 ht0 (by linarith) (by linarith [hf hεpos]) m
  have h1' := mul_le_mul_of_nonneg_left hS hX0
  have h2' := mul_le_mul_of_nonneg_right (hXε.trans hε1) ht0
  linarith

/-! ## `E^rep` -/

section ERep

variable {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **What `E^rep` returns**: on an accepted judged run, the first of its waits' candidates; else nothing. -/
theorem ERep_eval_fst (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r₀ : Fin (V.k x) → Coin) (ρ : Fin (m x) → Fin (K x) → Fin (V.k x) → Coin) :
    ((ERep V m k K opens assemble check).eval H P x ω r₀ ρ).1 =
      if (run H P V x ω r₀).out = .acc then
        ((List.finRange (m x)).map ρ).findSome? fun σ =>
          waitOut H (run H P V x ω) (k x) opens (assemble x) (check x) ((List.finRange (K x)).map σ)
      else none := by
  by_cases hacc : (run H P V x ω r₀).out = .acc
  · simp only [Extractor.eval, ERep, RP.eval, Transcript.seen, hacc, ↓reduceIte]
    rw [firstWait_eval, List.findSome?_map, List.findSome?_map]
    rfl
  · simp only [Extractor.eval, ERep, RP.eval, Transcript.seen, hacc, ↓reduceIte]

/-- `g`: the probability that one of `E^rep`'s waits at tape `ω`, over `K` fresh coin samples, returns a candidate
with `G`. -/
noncomputable def waitHit (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) (G : W → Prop) : ℝ :=
  Claims.Pr fun σ : Fin K → Fin (V.k x) → Coin =>
    ∃ w, waitOut H (run H P V x ω) k opens assemble check ((List.finRange K).map σ) = some w ∧ G w

/-- **`E^rep` at one judged run**: over its coins, the run accepts and it returns a candidate with `G` with the
probability `g` that one wait returns one, times `1 + f + … + f^{m−1}`, `f` one wait's failure probability. -/
theorem ERep_hit (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r₀ : Fin (V.k x) → Coin) (G : W → Prop) [Decidable ((run H P V x ω r₀).out = .acc)] :
    Pr (fun ρ : Fin (m x) → Fin (K x) → Fin (V.k x) → Coin => (run H P V x ω r₀).out = .acc ∧
        ∃ w, ((ERep V m k K opens assemble check).eval H P x ω r₀ ρ).1 = some w ∧ G w) =
      (if (run H P V x ω r₀).out = .acc then waitHit H V P x ω (k x) (K x) opens (assemble x) (check x) G else 0) *
        ∑ i ∈ Finset.range (m x), waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ i := by
  by_cases hacc : (run H P V x ω r₀).out = .acc
  · rw [ite_eq_left hacc]
    have hfail : Pr (fun σ : Fin (K x) → Fin (V.k x) → Coin =>
        waitOut H (run H P V x ω) (k x) opens (assemble x) (check x) ((List.finRange (K x)).map σ) = none) =
        waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) :=
      Pr_congr fun σ => waitOut_eq_none _ _ _ _ _ _ _
    rw [← hfail]
    refine (Pr_congr fun ρ => ?_).trans (Pr_findSome (fun σ : Fin (K x) → Fin (V.k x) → Coin =>
      waitOut H (run H P V x ω) (k x) opens (assemble x) (check x) ((List.finRange (K x)).map σ)) G (m x))
    rw [ERep_eval_fst, ite_eq_left hacc]
    exact and_iff_right hacc
  · rw [ite_eq_right hacc, zero_mul]
    exact (Pr_congr fun ρ => ⟨fun h => hacc h.1, False.elim⟩).trans Pr_false

/-- **`E^rep`: one wait finds what the waits find, at half the rate.** Under `ERep_trunc`'s hypotheses (`K > 0`,
`k ≥ 8`, and at each tape with acceptance `ε > 0` one wait's bound on `k` accepted runs, `ε·Q/ε^k ≤ a + rate·ε` with
`rate ≤ 1/4`), for every property `G` of the judged run's coins and the candidate: the run accepts and `E^rep` returns
a candidate with `G` with probability at most twice that at `E^rep` with one wait, plus
`4·𝔼[a] + 2k/K + 2·e^{−k/8}`. -/
theorem ERep_finds (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (hK : 0 < K x) (hk : 8 ≤ k x) (a : Ω → ℝ) (ha : ∀ ω, 0 ≤ a ω) {rate : ℝ} (hrate : rate ≤ 1 / 4)
    (hwait : ∀ ω, 0 < accPr H V P x ω →
      accPr H V P x ω * (accFail H V P x ω (k x) (Assembles opens (assemble x) (check x)) /
        accPr H V P x ω ^ k x) ≤ a ω + rate * accPr H V P x ω)
    (G : (Fin (V.k x) → Coin) → W → Prop) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × (Fin (m x) → Fin (K x) → Fin (V.k x) → Coin) =>
        (run H P V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((ERep V m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) ≤
      2 * Pr (fun s : Ω × (Fin (V.k x) → Coin) × (Fin 1 → Fin (K x) → Fin (V.k x) → Coin) =>
        (run H P V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((ERep V (fun _ => 1) k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧
            G s.2.1 w) +
        (4 * Ex a + 2 * k x / K x + 2 * Real.exp (-(k x : ℝ) / 8)) := by
  classical
  have hsplit : ∀ (n : Msg → ℕ),
      Pr (fun s : Ω × (Fin (V.k x) → Coin) × (Fin (n x) → Fin (K x) → Fin (V.k x) → Coin) =>
        (run H P V x s.1 s.2.1).out = .acc ∧
          ∃ w, ((ERep V n k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) =
      Ex fun ω => (Ex fun r₀ : Fin (V.k x) → Coin =>
        if (run H P V x ω r₀).out = .acc then
          waitHit H V P x ω (k x) (K x) opens (assemble x) (check x) (G r₀) else 0) *
        ∑ i ∈ Finset.range (n x), waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ i := by
    intro n
    rw [← Ex_ind, Ex_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Ex_prod, ← Ex_mul_right]
    refine congrArg Ex (funext fun r₀ => ?_)
    rw [Ex_ind]
    exact ERep_hit H V n k K opens assemble check x P ω r₀ (G r₀)
  rw [hsplit m, hsplit fun _ => 1]
  simp only [Finset.sum_range_one, pow_zero, mul_one]
  have hpt : ∀ ω, (Ex fun r₀ : Fin (V.k x) → Coin =>
        if (run H P V x ω r₀).out = .acc then
          waitHit H V P x ω (k x) (K x) opens (assemble x) (check x) (G r₀) else 0) *
        ∑ i ∈ Finset.range (m x), waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ i ≤
      2 * (Ex fun r₀ : Fin (V.k x) → Coin =>
        if (run H P V x ω r₀).out = .acc then
          waitHit H V P x ω (k x) (K x) opens (assemble x) (check x) (G r₀) else 0) +
        (4 * a ω + 2 * k x / K x + 2 * Real.exp (-(k x : ℝ) / 8)) := by
    intro ω
    have hε0 : 0 ≤ accPr H V P x ω := Pr_nonneg _
    have hrε : rate * accPr H V P x ω ≤ accPr H V P x ω / 4 := by
      have := mul_le_mul_of_nonneg_right hrate hε0; linarith
    have hacc : (Ex fun r₀ : Fin (V.k x) → Coin => if (run H P V x ω r₀).out = .acc then (1 : ℝ) else 0) =
        accPr H V P x ω := Ex_ind _
    refine tape_finds (ε := accPr H V P x ω) (a := a ω) (Pr_le_one _) (Pr_nonneg _) (ha ω) (Pr_nonneg _) hK hk
      (waitFail_le H V P x ω (k x) (K x) opens (assemble x) (check x))
      (fun hε => (hwait ω hε).trans (by linarith))
      (fun hεK => (wait_short_le (run H P V x ω) (k x) (K x)).trans (short_le_exp (Pr_le_one _) _ _ hεK))
      (Ex_nonneg fun r₀ => by split_ifs; exacts [Pr_nonneg _, le_rfl]) ?_ ?_
    · rw [← hacc]
      exact Ex_mono fun r₀ => by split_ifs; exacts [Pr_le_one _, le_rfl]
    · rw [← hacc, ← Ex_mul_right]
      refine Ex_mono fun r₀ => ?_
      rw [← ERep_hit H V m k K opens assemble check x P ω r₀ (G r₀)]
      by_cases h : (run H P V x ω r₀).out = .acc
      · rw [ite_eq_left h]; exact Pr_le_one _
      · rw [ite_eq_right h]
        exact (Pr_congr fun ρ => ⟨fun h' => h h'.1, False.elim⟩).trans_le Pr_false.le
  calc _ ≤ Ex fun ω => 2 * (Ex fun r₀ : Fin (V.k x) → Coin =>
          if (run H P V x ω r₀).out = .acc then
            waitHit H V P x ω (k x) (K x) opens (assemble x) (check x) (G r₀) else 0) +
          (4 * a ω + 2 * k x / K x + 2 * Real.exp (-(k x : ℝ) / 8)) := Ex_mono hpt
    _ = _ := by
        rw [Ex_add, Ex_mul_left, Ex_add, Ex_add, Ex_mul_left, Ex_const, Ex_const]

end ERep

/-! ## The claim -/

section Finds

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`Finds(V, E, E₁, κ, γ)`: what `E` finds, `E₁` finds at half the rate.** For every prover and every property `G`
of the judged run's coins and a witness, the run accepts and `E` returns a witness with `G` with probability at most
twice that at `E₁`, plus `2·κ(x) + √(γ(x)·Pr[B^{P*}(x) outputs a collision])`. Read at `E^rep` and `E^rep` with one
wait (`ERep_finds`), at the errors of the proof system's knowledge soundness (`KS`). -/
def Finds (H : Msg → Msg) (V : Verifier Coin) (E E₁ : Extractor Coin W V) (κ γ : Msg → ℝ) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (G : (Fin (V.k x) → Coin) → W → Prop),
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x => (run H P V x s.1 s.2.1).out = .acc ∧
        ∃ w, (E.eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) ≤
      2 * Pr (fun s : Ω × (Fin (V.k x) → Coin) × E₁.C x => (run H P V x s.1 s.2.1).out = .acc ∧
        ∃ w, (E₁.eval H P x s.1 s.2.1 s.2.2).1 = some w ∧ G s.2.1 w) +
        2 * κ x + Real.sqrt (γ x * PrB H V P x)

end Finds

end Claims
