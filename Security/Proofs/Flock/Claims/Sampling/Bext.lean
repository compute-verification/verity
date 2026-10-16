import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Cost

/-!
# `B_ext` at the claims block's finder, and the block's bits from its KS

The claims block (`Definitions.Claims.Sampling`) states `KS` with the linear term `1 · Pr[B_ext]`, where `Bext` forks
like `B`, runs the extractor in each branch, and outputs the first collision among the `H` calls the two extractions
make themselves: the openings' checks, `H(packBits(vals) ‖ salt)` and the path check. Commit strings opened inside the
sessions' Flock proofs bind the boundary values because two accepted openings of one row to different values make
such a collision. This file states that reduction at `Bext` and derives the block's `Bits` from its `KS`.

* `OwnColl`: one extraction's own calls hold two different inputs with one output.
* `Pr_ownColl_le_Bext`: **the reduction's probability step**. `B_ext`'s first branch is the extraction on the judged
  run's tape and coins and uniform extractor coins, so `Pr[OwnColl] ≤ Pr[B_ext finds a collision]`.
* `KSLin_of_ownColl`: **the reduction**. If an accepted run whose extraction fails either falls in an event that
  `κ + √(γ · Pr[B])` bounds or has an own-call collision, `KSLin` holds at `β = 1` and `Bext`. A verifier with no rounds
  at `x` rejects every run, so `V.k x = 0` needs no case of its own.
* `Bext_cost_le`: `B_ext`'s branches are one extraction each, the second's coins uniform too (`bextSwap`), so its
  expected calls are at most `2·(own + t·runs)` for an extractor with `own` own calls, `runs` expected reruns, and
  runs of at most `t` calls.
* `lamLin_bound`, `Bits_of_KSLin`: `κ + t·√γ·2^{−255.5} + t·2βc/2^256 ≤ t·2^{−lamLin κ γ β c}`, so `KSLin` with
  `StrictBirthday` for `B` and `ExpectedCR` for `Bext` gives `BitsLin` at `lamLin`.
* `Bits_of_KS`: **the block's bits from its KS**, given `OpenCallsLe` and a nonnegative table error.
-/

namespace Claims.Sampling

open Claims

/-! ## Own calls and `B_ext`'s coins -/

/-- Every call a rerun program logs as its own is an evaluation of `H`. -/
theorem ownLog_valid {R α : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) :
    ∀ q ∈ ownLog H run p, H q.1 = q.2 := by
  induction p with
  | ret a => simp [ownLog]
  | hash q k ih =>
    intro q' hq'
    simp only [ownLog, List.mem_cons] at hq'
    rcases hq' with rfl | hq'
    · rfl
    · exact ih _ q' hq'
  | rerun r k ih => exact ih _

/-- A rerun program's SHA-512 calls are its own and its reruns', each rerun making at most `t`. -/
theorem calls_le_own {R α : Type} (H : Msg → Msg) (run : R → Trace) (t : ℕ)
    (ht : ∀ r, (run r).pLog.length + (run r).vLog.length ≤ t) (p : RP R α) :
    (p.eval H run).2.1 ≤ (ownLog H run p).length + t * (p.eval H run).2.2 := by
  induction p with
  | ret a => simp [RP.eval]
  | hash q k ih =>
    simp only [RP.eval, ownLog, List.length_cons]
    have := ih (H q)
    omega
  | rerun r k ih =>
    simp only [RP.eval, ownLog]
    have h1 := ih (run r).seen
    have h2 := ht r
    rw [Nat.mul_succ]
    omega

theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Pr (fun a => p (e a)) = Pr p := by
  classical
  rw [Pr_eq_sum (fun a => p (e a)), Pr_eq_sum p, Fintype.card_congr e]
  congr 1
  exact Fintype.sum_equiv e _ _ fun a => rfl

theorem Ex_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (f : β → ℝ) :
    Ex (fun a => f (e a)) = Ex f := by
  unfold Ex
  rw [Fintype.card_congr e]
  congr 1
  exact Fintype.sum_equiv e _ _ fun a => rfl

theorem Ex_fst {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (f : α → ℝ) :
    Ex (fun s : α × β => f s.1) = Ex f := by
  rw [Ex_prod]
  congr 1
  funext a
  exact Ex_const (α := β) (f a)

/-- `B_ext`'s coins, split into its first branch's (tape, judged coins, extractor coins) and the rest. -/
def bextSplit (K Ω D C : Type) : (K × Ω × D × D) × C × C ≃ (Ω × D × C) × (K × D × C) where
  toFun s := ((s.1.2.1, s.1.2.2.1, s.2.1), (s.1.1, s.1.2.2.2, s.2.2))
  invFun p := ((p.2.1, p.1.1, p.1.2.1, p.2.2.1), p.1.2.2, p.2.2.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- `B_ext`'s branches exchanged: at the round guess `j`, the coins `(r, r′)` go to the second branch's
`forkCoins j r r′` and its complement `forkCoins j r′ r`, and the extractor's two coins swap. An involution. -/
def bextSwap (Ω Coin C : Type) (k : ℕ) : BCoins Ω Coin k × C × C ≃ BCoins Ω Coin k × C × C where
  toFun s := ((s.1.1, s.1.2.1, forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, forkCoins s.1.1 s.1.2.2.2 s.1.2.2.1), s.2.2, s.2.1)
  invFun s := ((s.1.1, s.1.2.1, forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, forkCoins s.1.1 s.1.2.2.2 s.1.2.2.1), s.2.2, s.2.1)
  left_inv := by
    rintro ⟨⟨j, ω, r, r'⟩, c₁, c₂⟩
    simp only [Prod.mk.injEq, true_and, and_true]
    constructor <;> funext i <;> by_cases h : i.val < j.val <;> simp [forkCoins, h]
  right_inv := by
    rintro ⟨⟨j, ω, r, r'⟩, c₁, c₂⟩
    simp only [Prod.mk.injEq, true_and, and_true]
    constructor <;> funext i <;> by_cases h : i.val < j.val <;> simp [forkCoins, h]

/-- A verifier with no rounds at `x` rejects every run. -/
theorem run_out_of_k_eq_zero {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin)
    (x : Msg) (hk : V.k x = 0) (ω : Ω) (r : Fin (V.k x) → Coin) : (run H P V x ω r).out = .rej := by
  have h : ∀ n vnext, n = 0 → (stepWith H P x ω vnext n [] [] ⟨[], [], [], .rej⟩).out = .rej := by
    rintro n vnext rfl; rfl
  exact h _ _ hk

section Bext

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **A collision among one extraction's own calls**: two different inputs with one output, in the `H` calls the
extractor makes itself on the judged run `(ω, r₀)` at its coins `c`. -/
def OwnColl (H : Msg → Msg) {V : Verifier Coin} (E : Extractor Coin W V) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (V.k x) → Coin) (c : E.C x) : Prop :=
  ∃ p ∈ ownLog H (run H P V x ω) (E.ext x r₀ c), ∃ p' ∈ ownLog H (run H P V x ω) (E.ext x r₀ c),
    p.1 ≠ p'.1 ∧ p.2 = p'.2

omit [Fintype Coin] [Nonempty Coin] in
/-- `B_ext`'s first branch's collisions are `B_ext`'s. -/
theorem Bext_out_of_ownColl (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) {Ω : Type} (P : Prover Ω)
    (x : Msg) (s : BCoins Ω Coin (V.k x) × E.C x × E.C x) (h : OwnColl H E P x s.1.2.1 s.1.2.2.1 s.2.1) :
    ∃ u v, findColl (ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1) ++
      ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2)) = some (u, v) ∧
        Collision H u v := by
  obtain ⟨p, hp, p', hp', hne, heq⟩ := h
  set l₂ := ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2)
  have hs := findColl_isSome (ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1) ++ l₂)
    ⟨p, List.mem_append_left l₂ hp, p', List.mem_append_left l₂ hp', hne, heq⟩
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hs
  refine ⟨u, v, huv, findColl_spec H _ (fun q hq => ?_) u v huv⟩
  rcases List.mem_append.1 hq with hq | hq <;> exact ownLog_valid H _ _ q hq

/-- **The reduction's probability step**: an extraction's own calls collide with probability at most `B_ext`'s, over
the tape, the judged run's coins and the extractor's. -/
theorem Pr_ownColl_le_Bext (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x => OwnColl H E P x s.1 s.2.1 s.2.2) ≤ (Bext H V E Ω P x).Pr H := by
  have : Nonempty (Fin (V.k x)) := ⟨⟨0, hk⟩⟩
  have e : Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x => OwnColl H E P x s.1 s.2.1 s.2.2) =
      Pr (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x => OwnColl H E P x s.1.2.1 s.1.2.2.1 s.2.1) := by
    rw [← Pr_fst (β := Fin (V.k x) × (Fin (V.k x) → Coin) × E.C x),
      ← Pr_equiv (bextSplit (Fin (V.k x)) Ω (Fin (V.k x) → Coin) (E.C x))]
    rfl
  rw [e]
  unfold Bext Finder.Pr
  rw [dite_eq_left_of_eq_true (eq_true hk)]
  exact Pr_mono fun s h => Bext_out_of_ownColl H V E P x s h

/-- **`B_ext` is `KSLin`'s linear term at `β = 1`.** If, on every prover, an accepted run whose extraction fails
either falls in an event `Q` that `κ + √(γ · Pr[B])` bounds or has a collision among the extraction's own calls, then
`KSLin` holds at `β = 1` and the block's `Bext`. -/
theorem KSLin_of_ownColl (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (κ γ : Msg → ℝ)
    (Q : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω × (Fin (V.k x) → Coin) × E.C x → Prop)
    (hsplit : ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (V.k x) → Coin) × E.C x),
      (run H P V x s.1 s.2.1).out = .acc → ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1 →
        Q x Ω P s ∨ OwnColl H E P x s.1 s.2.1 s.2.2)
    (hQ : ∀ x (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      Pr (Q x Ω P) ≤ κ x + Real.sqrt (γ x * PrB H V P x)) :
    KSLin H V R E κ γ (fun _ => 1) (Bext H V E) := by
  intro x Ω _ _ P
  unfold PrBad
  rw [one_mul]
  by_cases hk : 0 < V.k x
  · calc Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x =>
          (run H P V x s.1 s.2.1).out = .acc ∧ ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1)
        ≤ Pr (fun s => Q x Ω P s ∨ OwnColl H E P x s.1 s.2.1 s.2.2) :=
          Pr_mono fun s ⟨h1, h2⟩ => hsplit x Ω P s h1 h2
      _ ≤ Pr (Q x Ω P) + Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x => OwnColl H E P x s.1 s.2.1 s.2.2) :=
          Pr_union_le _ _
      _ ≤ _ := add_le_add (hQ x Ω P) (Pr_ownColl_le_Bext H V E P x hk)
  · have h0 : V.k x = 0 := Nat.eq_zero_of_not_pos hk
    calc Pr (fun s : Ω × (Fin (V.k x) → Coin) × E.C x =>
          (run H P V x s.1 s.2.1).out = .acc ∧ ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1)
        = 0 := by
          rw [Pr_congr (q := fun _ => False) fun s =>
            ⟨fun h => by simp [run_out_of_k_eq_zero H P V x h0] at h, False.elim⟩]
          exact Pr_false
      _ ≤ Pr (Q x Ω P) + (Bext H V E Ω P x).Pr H := add_nonneg (Pr_nonneg _) (by unfold Finder.Pr; exact Pr_nonneg _)
      _ ≤ _ := add_le_add (hQ x Ω P) le_rfl

/-! ## `B_ext`'s expected calls -/

omit [Fintype Coin] [Nonempty Coin] in
/-- An extraction's SHA-512 calls: its own, at most `own(x)`, and its reruns', at most `t` each. -/
theorem ext_calls_le (H : Msg → Msg) {V : Verifier Coin} (E : Extractor Coin W V) {Ω : Type} (P : Prover Ω) (x : Msg)
    (t : ℕ) (ht : ∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) {own : Msg → ℕ}
    (hown : ExtractOwnLe H E own) (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : E.C x) :
    ((E.eval H P x ω r₀ c).2.1 : ℝ) ≤ own x + t * (E.eval H P x ω r₀ c).2.2 := by
  have h := calls_le_own H (run H P V x ω) t (ht ω) (E.ext x r₀ c)
  have h' := hown x Ω P ω r₀ c
  unfold Extractor.eval
  exact_mod_cast (h.trans (Nat.add_le_add_right h' _))

/-- **`B_ext`'s expected calls**: with runs of at most `t` calls, an extractor with at most `own(x)` own calls and
`runs(x)` reruns in expectation makes `B_ext` spend at most `2·(own(x) + t·runs(x))` calls in expectation. -/
theorem Bext_cost_le (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) (t : ℕ)
    (ht : ∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) {runs : Msg → ℝ}
    {own : Msg → ℕ} (hruns : ExtractRunsExpLe H E runs) (hown : ExtractOwnLe H E own) :
    Ex (fun s => ((Bext H V E Ω P x).cost s : ℝ)) ≤ 2 * (own x + t * runs x) := by
  have : Nonempty (Fin (V.k x)) := ⟨⟨0, hk⟩⟩
  set f : Ω × (Fin (V.k x) → Coin) × E.C x → ℝ := fun q => ((E.eval H P x q.1 q.2.1 q.2.2).2.1 : ℝ) with hf
  have hb1 : Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x => f (s.1.2.1, s.1.2.2.1, s.2.1)) = Ex f := by
    rw [← Ex_fst (β := Fin (V.k x) × (Fin (V.k x) → Coin) × E.C x) f,
      ← Ex_equiv (bextSplit (Fin (V.k x)) Ω (Fin (V.k x) → Coin) (E.C x))]
    rfl
  have hb2 : Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x =>
      f (s.1.2.1, forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)) = Ex f := by
    rw [← hb1]
    exact Ex_equiv (bextSwap Ω Coin (E.C x) (V.k x)) fun s => f (s.1.2.1, s.1.2.2.1, s.2.1)
  have hf1 : Ex f ≤ own x + t * runs x := by
    calc Ex f ≤ Ex (fun q : Ω × (Fin (V.k x) → Coin) × E.C x =>
          (own x : ℝ) + t * ((E.eval H P x q.1 q.2.1 q.2.2).2.2 : ℝ)) :=
          Ex_mono fun q => ext_calls_le H E P x t ht hown q.1 q.2.1 q.2.2
      _ = own x + t * Ex (fun ω : Ω => Ex (fun q : (Fin (V.k x) → Coin) × E.C x =>
          ((E.eval H P x ω q.1 q.2).2.2 : ℝ))) := by
          rw [Ex_add, Ex_const, Ex_mul_left, Ex_prod]
      _ ≤ own x + t * Ex (fun _ : Ω => runs x) :=
          add_le_add le_rfl (mul_le_mul_of_nonneg_left (Ex_mono fun ω => hruns x Ω P ω) (Nat.cast_nonneg _))
      _ = own x + t * runs x := by rw [Ex_const]
  unfold Bext
  rw [dite_eq_left_of_eq_true (eq_true hk)]
  show Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x => (((E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.1 +
    (E.eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).2.1 : ℕ) : ℝ)) ≤ _
  simp only [Nat.cast_add]
  show Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x =>
    f (s.1.2.1, s.1.2.2.1, s.2.1) + f (s.1.2.1, forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)) ≤ _
  rw [Ex_add, hb1, hb2]
  linarith

end Bext

/-! ## The bits -/

theorem sqrt_birthday {γ t : ℝ} (hγ : 0 ≤ γ) (ht : 0 ≤ t) :
    Real.sqrt (γ * ((2 * t) ^ 2 / 2 ^ (513 : ℝ))) = t * (Real.sqrt γ * 2 ^ (-(255.5 : ℝ))) := by
  have h513 : Real.sqrt ((2 : ℝ) ^ (513 : ℝ)) = 2 ^ (256.5 : ℝ) := by
    rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by positivity), ← Real.rpow_add (by norm_num)]
    norm_num
  rw [Real.sqrt_mul hγ, Real.sqrt_div (by positivity), Real.sqrt_sq (by linarith), h513]
  have : (2 : ℝ) ^ (-(255.5 : ℝ)) = 2 / 2 ^ (256.5 : ℝ) := by
    rw [Real.rpow_neg (by norm_num), show (256.5 : ℝ) = 255.5 + 1 by norm_num,
      Real.rpow_add (by norm_num), Real.rpow_one]
    field_simp
  rw [this]; ring

/-- `κ + √γ·2^{−255.5} + 2βc/2^256 ≤ 2^{−λ}` at `λ = lamLin κ γ β c`, for `κ > 0`, `γ ≥ 0` and `βc > 0`. -/
theorem lamLin_bound {κ γ β c : ℝ} (hκ : 0 < κ) (hγ : 0 ≤ γ) (hβc : 0 < 2 * β * c) :
    κ + Real.sqrt γ * 2 ^ (-(255.5 : ℝ)) + 2 * β * c / 2 ^ (256 : ℝ) ≤ 2 ^ (-lamLin κ γ β c) := by
  have h2 : Real.sqrt γ * 2 ^ (-(255.5 : ℝ)) ≤ 2 ^ (-(255.5 - Real.logb 2 γ / 2)) := by
    rcases hγ.lt_or_eq with hγ | rfl
    · refine le_of_eq ?_
      have : (2 : ℝ) ^ (-(255.5 - Real.logb 2 γ / 2)) = 2 ^ (Real.logb 2 γ * (1 / 2)) * 2 ^ (-(255.5 : ℝ)) := by
        rw [← Real.rpow_add (by norm_num)]; congr 1; ring
      rw [this, Real.rpow_mul (by norm_num), Real.rpow_logb (by norm_num) (by norm_num) hγ, Real.sqrt_eq_rpow]
    · simp only [Real.sqrt_zero, zero_mul]; positivity
  have h1 : κ = 2 ^ (-Real.logb 2 (1 / κ)) := by
    rw [one_div, Real.logb_inv, neg_neg, Real.rpow_logb (by norm_num) (by norm_num) hκ]
  have h3 : 2 * β * c / 2 ^ (256 : ℝ) = 2 ^ (-(256 - Real.logb 2 (2 * β * c))) := by
    rw [show -(256 - Real.logb 2 (2 * β * c)) = Real.logb 2 (2 * β * c) + -256 by ring, Real.rpow_add (by norm_num),
      Real.rpow_logb (by norm_num) (by norm_num) hβc, Real.rpow_neg (by norm_num), div_eq_mul_inv]
  unfold lamLin
  set m := min (min (Real.logb 2 (1 / κ)) (255.5 - Real.logb 2 γ / 2)) (256 - Real.logb 2 (2 * β * c)) with hm
  have h3' : (2 : ℝ) ^ (-(m - Real.logb 2 3)) = 3 * 2 ^ (-m) := by
    rw [show -(m - Real.logb 2 3) = Real.logb 2 3 + -m by ring, Real.rpow_add (by norm_num),
      Real.rpow_logb (by norm_num) (by norm_num) (by norm_num)]
  have hM : ∀ L, m ≤ L → (2 : ℝ) ^ (-L) ≤ 2 ^ (-m) := fun L h =>
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  have e1 := hM _ ((min_le_left _ _).trans (min_le_left _ _))
  have e2 := hM _ ((min_le_left _ _).trans (min_le_right _ _))
  have e3 := hM _ (min_le_right _ _)
  rw [h3']
  linarith

section Bits

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Bits from `KSLin` at `B_ext`**: `KSLin` at `β = 1` and `Bext`, an extractor with at most `own(x)` own calls and
`runs(x)` reruns in expectation, `κ > 0`, `γ ≥ 0`, and `runs + own ≤ c` with `c > 0`, give `λ = lamLin κ γ 1 c` bits,
where `B` obeys the strict birthday bound and `B_ext` the expected-time one. -/
theorem Bits_of_KSLin {H : Msg → Msg} {V : Verifier Coin} {R : Rel W} {E : Extractor Coin W V} {κ γ runs c : Msg → ℝ}
    {own : Msg → ℕ} (h : KSLin H V R E κ γ (fun _ => 1) (Bext H V E)) (hruns : ExtractRunsExpLe H E runs)
    (hown : ExtractOwnLe H E own) (hκ : ∀ x, 0 < κ x) (hγ : ∀ x, 0 ≤ γ x) (hc0 : ∀ x, 0 < c x)
    (hc : ∀ x, runs x + own x ≤ c x) :
    BitsLin H V R E (Bext H V E) fun x => lamLin (κ x) (γ x) 1 (c x) := by
  intro x Ω _ _ P t ht hrun hSB hECR
  have hks := h x Ω P
  have ht1 : (1 : ℝ) ≤ t := by exact_mod_cast ht
  have hB : PrB H V P x ≤ (2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ) := by
    unfold PrB
    split
    · rename_i hk
      refine hSB hk (2 * t) fun s => ?_
      exact_mod_cast B_cost_le H V P x hk t hrun s
    · positivity
  have hE : (Bext H V E Ω P x).Pr H ≤ t * (2 * 1 * c x / 2 ^ (256 : ℝ)) := by
    by_cases hk : 0 < V.k x
    · refine hECR.trans ?_
      rw [mul_div_assoc', div_le_div_iff_of_pos_right (by positivity)]
      refine (Bext_cost_le H V E P x hk t hrun hruns hown).trans ?_
      have hown0 : (0 : ℝ) ≤ own x := Nat.cast_nonneg _
      have : (own x : ℝ) ≤ t * own x := le_mul_of_one_le_left hown0 ht1
      nlinarith [hc x]
    · have h0 : (Bext H V E Ω P x).Pr H = 0 := by
        unfold Bext Finder.Pr
        rw [dite_eq_right_of_eq_false (eq_false hk)]
        exact (Pr_congr fun s => ⟨fun ⟨_, _, h, _⟩ => by simp at h, False.elim⟩).trans Pr_false
      rw [h0]
      have := hc0 x
      positivity
  have hp : PrBad H V R E x P ≤ κ x + t * (Real.sqrt (γ x) * 2 ^ (-(255.5 : ℝ))) + t * (2 * 1 * c x / 2 ^ (256 : ℝ)) := by
    rw [← sqrt_birthday (hγ x) (by linarith)]
    refine hks.trans ?_
    rw [one_mul]
    gcongr
    exact hγ x
  have hl := lamLin_bound (hκ x) (hγ x) (show 0 < 2 * 1 * c x by linarith [hc0 x])
  have hκt : κ x ≤ t * κ x := le_mul_of_one_le_left (hκ x).le ht1
  have ht0 : (0 : ℝ) ≤ t := by linarith
  nlinarith [mul_le_mul_of_nonneg_left hl ht0]

end Bits

/-! ## The block's bits -/

section Block

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem escape_nonneg' (L : Law) (N : ℕ) (B : Finset ℕ) : 0 ≤ L.escape N B := by
  cases L with
  | subset k => exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  | bernoulli num den => exact pow_nonneg (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) _
  | stratified ss =>
    refine List.prod_nonneg fun a ha => ?_
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
    exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem missAt_nonneg (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : 0 ≤ missAt D t x := by
  unfold missAt
  split
  · exact Real.iSup_nonneg fun B => escape_nonneg' _ _ _
  · exact le_rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem κ_pos (D : Deploy Coin Col T) (t : ℕ) (x : Msg) (hε : 0 ≤ D.εtable x) : 0 < κ D t x := by
  unfold κ
  have hκF : 0 ≤ κF D x := by unfold κF; positivity
  have hcov : 0 ≤ cover D t x := by unfold cover; positivity
  have := missAt_nonneg D t x
  have h1 : 0 ≤ cover D t x * (missAt D t x + κF D x) := by positivity
  have h2 : (0 : ℝ) ≤ 2 * D.k x / D.cap x := by positivity
  have h3 : 0 < Real.exp (-(D.k x : ℝ) / 8) := Real.exp_pos _
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem γ_nonneg (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : 0 ≤ γ D t x := by
  unfold γ; positivity

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem work_pos (D : Deploy Coin Col T) (x : Msg) : 0 < work D x := by
  unfold work; positivity

/-- **Sampling: bits from KS**: the block's `KS`, with one rerun's openings checks bounded (`OpenCallsLe`) and the
table error nonnegative, gives the block's `Bits`. -/
theorem Bits_of_KS {H : Msg → Msg} {D : Deploy Coin Col T} (h : KS H D) (hopen : OpenCallsLe H D)
    (hε : ∀ x, 0 ≤ D.εtable x) : Bits H D := fun t decode hdec =>
  Bits_of_KSLin (h t decode hdec) (E_runs_exp H D decode) (E_own_le H D decode hopen) (fun x => κ_pos D t x (hε x))
    (γ_nonneg D t) (work_pos D) fun x => le_of_eq (by unfold work; push_cast; ring)

end Block

end Claims.Sampling
