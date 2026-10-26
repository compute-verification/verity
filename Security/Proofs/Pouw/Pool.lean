import Definitions.Pouw.Assumptions.Pool
import Proofs.Pouw.Dither

/-!
# Pooling the fleet's checks (CMP-152)

The fleet's blocks are not one sequence (RT-PROTO-2 P16): a block's work can read the dither of a block that roots
after it, and two concurrent blocks can each read the other's. `SecurityProofs.Dither.expect_prod_adapted` needs each
factor to read only the coins up to its own, so `WorkSound` holds per replica and `FleetSound` adds the replicas by a
union bound, at `ln(G/δ)` a replica.

* **The pooled supermartingale** (`expect_prod_pool`, `prob_pool_le`): reveal every dither first, all at once, and run
  the martingale over the check coins alone, in the order the challenger releases them (one after each root). A
  block's factor may read every dither; it needs a moment over its own check coin at every dither (`ζ` a block), not
  an average over its dither.
* **In the model** (`pool_luck`): under `Assumptions.SAKCheck`, a fleet whose blocks' advice reads any dither but its
  own, and only the check coins released before its root, passes every check with rated shortfalls past
  `t + Σ ζ` with probability at most `e^{-t} + m κ_deg(B)`. At `ζ = 0` the premise implies `SAKMatmul`
  (`sakMatmul_of_sakCheck`).
* **Average moments don't pool** (`not_pool_of_average_moments`, `parity_all_pass`): blocks with 1-bit dithers that
  guess their own as the sum of the others' each pass with probability 1/2 at every value of the others', the form
  `SAKMatmul` gives, yet all pass together with probability 1/2, and two of them have a product averaging 2. The
  pointwise premise refuses them (`parity_pointwise`).
* **Pooling replicas by convexity** (`prob_convex_pool`, `fleet_convex_luck`; RT-POOL's RP1): without a new premise,
  each replica's own moment (`expect_replica_rounds`: `WorkSound`'s rounds, average at most 1 under `SAKMatmul`,
  against advice that reads every other replica's coins) bounds the fleet's total shortfall at `ln(1/δ)` rather than
  `ln(G/δ)`: the weights `1/η` put the total's exponential below the replicas' mixture, which needs no independence
  between them. The bound is on the total, and its rate is a replica's: the total's share stays `ln(1/δ)/(η q)`, not
  `ln(1/δ)/(η G q)`. That is the best per-replica moments give (`convex_pool_tight`): replicas that cheat on one
  shared coin attain it.
* **On the accepted set** (`prob_convex_pool_accepted`, `fleet_convex_luck_accepted`; RT-POOL's RP24): a failed
  replica voids only its own period, and one Markov bound covers every set `S` of accepted replicas at once, at
  `Λ_S (ln (Λ / Λ_S) + t)` for `Λ_S = ∑_{u ∈ S} 1 / η u`: at equal rates, `ln(G/(s δ))/(η q)` of the accepted credit
  at `s` accepted replicas.
-/

namespace Pouw.SecurityProofs.Pool

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.SecurityProofs.Dither Finset

/-! ## The pooled supermartingale -/

/-- `expect_prod_adapted` at a rate: factors adapted to the coins' order whose averages over their own coin are at
most `e^{ζ i}` have a product averaging at most `e^{Σ ζ}`. -/
theorem expect_prod_adapted_exp {Ω : Type} [Fintype Ω] (n : ℕ) (Y : Fin n → (Fin n → Ω) → ℝ) (ζ : Fin n → ℝ)
    (h0 : ∀ i x, 0 ≤ Y i x) (hY : ∀ i x x', (∀ j, j ≤ i → x j = x' j) → Y i x = Y i x')
    (h1 : ∀ i x, expect (fun ω => Y i (Function.update x i ω)) ≤ Real.exp (ζ i)) :
    expect (fun x => ∏ i, Y i x) ≤ Real.exp (∑ i, ζ i) := by
  have key := expect_prod_adapted n (fun i x => Real.exp (-ζ i) * Y i x)
    (fun i x => mul_nonneg (Real.exp_pos _).le (h0 i x))
    (fun i x x' hx => by rw [hY i x x' hx])
    (fun i x => by
      rw [expect_mul_left]
      calc Real.exp (-ζ i) * expect (fun ω => Y i (Function.update x i ω))
          ≤ Real.exp (-ζ i) * Real.exp (ζ i) := mul_le_mul_of_nonneg_left (h1 i x) (Real.exp_pos _).le
        _ = 1 := by rw [← Real.exp_add, neg_add_cancel, Real.exp_zero])
  have hprod : (fun x => ∏ i, (Real.exp (-ζ i) * Y i x)) = fun x => Real.exp (-∑ i, ζ i) * ∏ i, Y i x := by
    funext x
    rw [Finset.prod_mul_distrib, ← Real.exp_sum, Finset.sum_neg_distrib]
  rw [hprod, expect_mul_left] at key
  have hpos := Real.exp_pos (∑ i, ζ i)
  calc expect (fun x => ∏ i, Y i x)
      = Real.exp (∑ i, ζ i) * (Real.exp (-∑ i, ζ i) * expect (fun x => ∏ i, Y i x)) := by
        rw [← mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, one_mul]
    _ ≤ Real.exp (∑ i, ζ i) * 1 := mul_le_mul_of_nonneg_left key hpos.le
    _ = Real.exp (∑ i, ζ i) := mul_one _

/-- **The pooled supermartingale.** All the blocks' dithers `d` are revealed first, at once; the check coins `c` follow
in their release order (the challenger's log). Block `i`'s factor reads every dither, its own check coin and the check
coins released before it, and averages at most `e^{ζ i}` over its own check coin at every dither and every other
check coin. Then the product averages at most `e^{Σ ζ}`: no factor needs to be adapted in the dithers. -/
theorem expect_prod_pool {Δ Γ : Type} [Fintype Δ] [Fintype Γ] (n : ℕ) (Y : Fin n → Δ × (Fin n → Γ) → ℝ)
    (ζ : Fin n → ℝ) (h0 : ∀ i z, 0 ≤ Y i z)
    (hY : ∀ i d c c', (∀ j, j ≤ i → c j = c' j) → Y i (d, c) = Y i (d, c'))
    (h1 : ∀ i d c, expect (fun γ => Y i (d, Function.update c i γ)) ≤ Real.exp (ζ i)) :
    expect (fun z => ∏ i, Y i z) ≤ Real.exp (∑ i, ζ i) := by
  rw [expect_prod]
  exact expect_le_of_le (Real.exp_pos _).le fun d =>
    expect_prod_adapted_exp n (fun i c => Y i (d, c)) ζ (fun i c => h0 i _) (fun i c c' h => hY i d c c' h)
      (fun i c => h1 i d c)

/-- **The pooled luck.** `expect_prod_pool` with each block's moment holding outside a degenerate event `Deg i`
(which reads the dithers and the check coins released before block `i`'s): the product reaches `e^t` with
probability at most `e^{Σ ζ − t} + P[∃ i, Deg i]`. -/
theorem prob_pool_le_exists {Δ Γ : Type} [Fintype Δ] [Fintype Γ] (n : ℕ) (Y : Fin n → Δ × (Fin n → Γ) → ℝ)
    (ζ : Fin n → ℝ) (Deg : Fin n → Δ × (Fin n → Γ) → Prop) (h0 : ∀ i z, 0 ≤ Y i z)
    (hY : ∀ i d c c', (∀ j, j ≤ i → c j = c' j) → Y i (d, c) = Y i (d, c'))
    (hD : ∀ i d c c', (∀ j, j < i → c j = c' j) → (Deg i (d, c) ↔ Deg i (d, c')))
    (h1 : ∀ i d c, ¬ Deg i (d, c) → expect (fun γ => Y i (d, Function.update c i γ)) ≤ Real.exp (ζ i))
    (t : ℝ) :
    prob (fun z => Real.exp t ≤ ∏ i, Y i z) ≤ Real.exp (∑ i, ζ i - t) + prob (fun z => ∃ i, Deg i z) := by
  classical
  let Y' : Fin n → Δ × (Fin n → Γ) → ℝ := fun i z => if Deg i z then Real.exp (ζ i) else Y i z
  have hY'0 : ∀ i z, 0 ≤ Y' i z := fun i z => by
    simp only [Y']
    split_ifs
    exacts [(Real.exp_pos _).le, h0 i z]
  have hupd : ∀ i d c γ, Deg i (d, Function.update c i γ) ↔ Deg i (d, c) := fun i d c γ =>
    hD i d _ _ fun j hj => Function.update_of_ne (ne_of_lt hj) _ _
  have hE : expect (fun z => ∏ i, Y' i z) ≤ Real.exp (∑ i, ζ i) := by
    refine expect_prod_pool n Y' ζ hY'0 (fun i d c c' h => ?_) (fun i d c => ?_)
    · have hDi := hD i d c c' (fun j hj => h j hj.le)
      simp only [Y', hY i d c c' h]
      by_cases hd : Deg i (d, c)
      · rw [ite_eq_left hd, ite_eq_left (hDi.mp hd)]
      · rw [ite_eq_right hd, ite_eq_right (fun h' => hd (hDi.mpr h'))]
    · by_cases hd : Deg i (d, c)
      · simp only [Y', ite_eq_left ((hupd i d c _).mpr hd)]
        exact expect_le_of_le (Real.exp_pos _).le fun _ => le_rfl
      · simp only [Y', ite_eq_right (fun h' => hd ((hupd i d c _).mp h'))]
        exact h1 i d c hd
  have hM : prob (fun z => Real.exp t ≤ ∏ i, Y' i z) ≤ Real.exp (∑ i, ζ i - t) := by
    have h := (markov (fun z => Finset.prod_nonneg fun i _ => hY'0 i z) (Real.exp t)).trans hE
    rw [Real.exp_sub, le_div_iff₀ (Real.exp_pos t), mul_comm]
    exact h
  calc prob (fun z => Real.exp t ≤ ∏ i, Y i z)
      ≤ prob (fun z => (∃ i, Deg i z) ∨ Real.exp t ≤ ∏ i, Y' i z) :=
        prob_mono fun z hz => by
          by_cases hd : ∃ i, Deg i z
          · exact Or.inl hd
          · refine Or.inr ?_
            have : ∀ i, Y' i z = Y i z := fun i => ite_eq_right fun h => hd ⟨i, h⟩
            simp only [this]
            exact hz
    _ ≤ prob (fun z => ∃ i, Deg i z) + prob (fun z => Real.exp t ≤ ∏ i, Y' i z) := prob_or_le _ _
    _ ≤ prob (fun z => ∃ i, Deg i z) + Real.exp (∑ i, ζ i - t) := add_le_add le_rfl hM
    _ = Real.exp (∑ i, ζ i - t) + prob (fun z => ∃ i, Deg i z) := add_comm _ _

/-- `prob_pool_le_exists` with the union bound over the blocks' degenerate events. -/
theorem prob_pool_le {Δ Γ : Type} [Fintype Δ] [Fintype Γ] (n : ℕ) (Y : Fin n → Δ × (Fin n → Γ) → ℝ)
    (ζ : Fin n → ℝ) (Deg : Fin n → Δ × (Fin n → Γ) → Prop) (h0 : ∀ i z, 0 ≤ Y i z)
    (hY : ∀ i d c c', (∀ j, j ≤ i → c j = c' j) → Y i (d, c) = Y i (d, c'))
    (hD : ∀ i d c c', (∀ j, j < i → c j = c' j) → (Deg i (d, c) ↔ Deg i (d, c')))
    (h1 : ∀ i d c, ¬ Deg i (d, c) → expect (fun γ => Y i (d, Function.update c i γ)) ≤ Real.exp (ζ i))
    (t : ℝ) :
    prob (fun z => Real.exp t ≤ ∏ i, Y i z) ≤ Real.exp (∑ i, ζ i - t) + ∑ i, prob (Deg i) := by
  refine (prob_pool_le_exists n Y ζ Deg h0 hY hD h1 t).trans (add_le_add le_rfl ?_)
  exact (prob_mono fun z ⟨i, hi⟩ => ⟨i, mem_univ i, hi⟩).trans (prob_exists_le _ _)

/-! ## Average moments don't pool: the parity counterexample -/

/-- Block `i`'s factor when every block's work reads every other block's (1-bit) dither: it guesses its own dither as
the sum of the others' mod 2, passes when the guess is right (the dithers' sum is 0), and then skips work worth
`ln 2 / η`, so its factor `1_pass · e^{η D}` is 2. -/
noncomputable def parityY (n : ℕ) (_i : Fin n) (d : Fin n → ZMod 2) : ℝ := if ∑ j, d j = 0 then 2 else 0

theorem parityY_nonneg (n : ℕ) (i : Fin n) (d : Fin n → ZMod 2) : 0 ≤ parityY n i d := by
  unfold parityY
  split_ifs <;> norm_num

theorem expect_zmod2 (a : ℝ) (s : ZMod 2) : expect (fun ω : ZMod 2 => if ω + s = 0 then a else 0) = a / 2 := by
  unfold Pouw.RelativeSecurity.expect
  have hsum : ∀ f : ZMod 2 → ℝ, ∑ ω : ZMod 2, f ω = f 0 + f 1 := fun f => Fin.sum_univ_two f
  have h00 : (0 : ZMod 2) + 0 = 0 := by decide
  have h10 : ¬ (1 : ZMod 2) + 0 = 0 := by decide
  have h01 : ¬ (0 : ZMod 2) + 1 = 0 := by decide
  have h11 : (1 : ZMod 2) + 1 = 0 := by decide
  have hs : ∀ s : ZMod 2, s = 0 ∨ s = 1 := by decide
  rw [hsum, ZMod.card]
  rcases hs s with rfl | rfl
  · rw [ite_eq_left h00, ite_eq_right h10]
    push_cast
    ring
  · rw [ite_eq_right h01, ite_eq_left h11]
    push_cast
    ring

theorem sum_update (n : ℕ) (i : Fin n) (d : Fin n → ZMod 2) (ω : ZMod 2) :
    ∑ j, Function.update d i ω j = ω + ∑ j ∈ univ.erase i, d j := by
  rw [← Finset.add_sum_erase _ _ (mem_univ i), Function.update_self]
  congr 1
  exact Finset.sum_congr rfl fun j hj => Function.update_of_ne (ne_of_mem_erase hj) _ _

/-- **Each block's average moment is 1** over its own dither, at every value of every other block's dither: the
form `SAKMatmul` gives a block whose work reads the other blocks' dithers. -/
theorem parityY_moment (n : ℕ) (i : Fin n) (d : Fin n → ZMod 2) :
    expect (fun ω => parityY n i (Function.update d i ω)) = 1 := by
  simp only [parityY, sum_update]
  rw [expect_zmod2]
  norm_num

theorem prob_parity (m : ℕ) : prob (fun d : Fin (m + 1) → ZMod 2 => ∑ j, d j = 0) = 1 / 2 := by
  classical
  rw [prob_eq_expect, ← expect_equiv (Fin.consEquiv fun _ => ZMod 2), expect_prod_swap]
  have hs : ∀ (ω : ZMod 2) (d' : Fin m → ZMod 2),
      ∑ j, (Fin.consEquiv (fun _ => ZMod 2) (ω, d') : Fin (m + 1) → ZMod 2) j = ω + ∑ j, d' j := fun ω d' => by
    simp [Fin.consEquiv, Fin.sum_univ_succ]
  simp only [hs, expect_zmod2]
  exact expect_le_of_le (by norm_num) (fun _ => le_rfl) |>.antisymm (by
    unfold Pouw.RelativeSecurity.expect
    rcases Nat.eq_zero_or_pos (Fintype.card (Fin m → ZMod 2)) with h0 | h0
    · exact absurd h0 Fintype.card_ne_zero
    · rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_div_cancel_left₀ _ (by exact_mod_cast h0.ne')])

theorem parity_prod (n : ℕ) (d : Fin n → ZMod 2) :
    ∏ i, parityY n i d = if ∑ j, d j = 0 then (2 : ℝ) ^ n else 0 := by
  unfold parityY
  by_cases h : ∑ j, d j = 0
  · simp [h]
  · rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp at h
    · simp [h, Finset.prod_eq_zero (mem_univ (⟨0, hn⟩ : Fin n))]

/-- The product of the parity factors averages `2^m` over `m + 1` blocks. -/
theorem expect_parity_prod (m : ℕ) :
    expect (fun d : Fin (m + 1) → ZMod 2 => ∏ i, parityY (m + 1) i d) = 2 ^ m := by
  classical
  simp only [parity_prod]
  have : (fun d : Fin (m + 1) → ZMod 2 => if ∑ j, d j = 0 then (2 : ℝ) ^ (m + 1) else 0) =
      fun d => 2 ^ (m + 1) * (if ∑ j, d j = 0 then 1 else 0) := by
    funext d
    split_ifs <;> simp
  rw [this, expect_mul_left, ← prob_eq_expect, prob_parity, pow_succ]
  ring

/-- **Average moments don't pool** (P16's gap, at its smallest): two blocks whose factors are nonnegative and average
at most 1 over their own coin at every value of the other's, but whose product averages 2. So
`SecurityProofs.Dither.expect_prod_adapted` fails without its adaptedness hypothesis, and a moment averaged over the
block's dither (`SAKMatmul`'s form) does not bound blocks whose work reads each other's dithers. -/
theorem not_pool_of_average_moments :
    ¬ ∀ (n : ℕ) (Y : Fin n → (Fin n → ZMod 2) → ℝ), (∀ i x, 0 ≤ Y i x) →
      (∀ i x, expect (fun ω => Y i (Function.update x i ω)) ≤ 1) → expect (fun x => ∏ i, Y i x) ≤ 1 := by
  intro h
  have h2 := h 2 (parityY 2) (parityY_nonneg 2) (fun i x => (parityY_moment 2 i x).le)
  rw [expect_parity_prod 1] at h2
  norm_num at h2

/-- **All `m + 1` blocks pass together with probability 1/2**, though each passes with probability 1/2 at every
value of the other blocks' dithers: under adaptedness all would pass with probability `2^{-(m+1)}`. -/
theorem parity_all_pass (m : ℕ) :
    prob (fun d : Fin (m + 1) → ZMod 2 => ∀ i, parityY (m + 1) i d = 2) = 1 / 2 := by
  have : (fun d : Fin (m + 1) → ZMod 2 => ∀ i, parityY (m + 1) i d = 2) = fun d => ∑ j, d j = 0 := by
    funext d
    apply propext
    unfold parityY
    constructor
    · intro h
      by_contra hne
      have := h 0
      rw [ite_eq_right hne] at this
      norm_num at this
    · intro h i
      rw [ite_eq_left h]
  rw [this, prob_parity]

/-- **The pointwise premise refuses the counterexample.** With the check coin trivial (`Γ = Unit`), the parity factor's
moment over its check coin at an even dither is 2, so `prob_pool_le`'s premise holds there only at `ζ i ≥ ln 2`, or with
the even dithers, of probability 1/2, in `Deg i`. -/
theorem parity_pointwise (n : ℕ) (i : Fin n) (d : Fin n → ZMod 2) (hd : ∑ j, d j = 0) :
    expect (fun _ : Unit => parityY n i d) = 2 := by
  unfold Pouw.RelativeSecurity.expect parityY
  simp [hd]

/-! ## The pointwise premise in the model -/

/-- `Assumptions.SAKCheck` at `ζ = 0` implies `Assumptions.SAKMatmul`: the words moment is the check moment's average
over the dithers outside `Deg`. -/
theorem sakMatmul_of_sakCheck (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (Reg : Set P.Inst)
    (pub η : P.Inst → ℝ) (κdeg : ℕ → ℝ) (h : SAKCheck G P μ Reg pub η (fun _ => 0) κdeg) :
    SAKMatmul G P μ Reg pub η κdeg := by
  intro h1 h2 h3 h4 a ha B St hB s Mw hext
  obtain ⟨Deg, hDeg, hm⟩ := h h1 h2 h3 h4 a ha B St hB s Mw hext
  refine ⟨Deg, hDeg, ?_⟩
  classical
  have hw : wordsMoment G P μ (pub a) (η a) a s Mw Deg =
      expect fun d => if Deg d then 0 else checkMoment G P μ (pub a) (η a) a s Mw d := by
    unfold wordsMoment checkMoment
    change Pouw.RelativeSecurity.expect _ = Pouw.RelativeSecurity.expect _
    rw [expect_prod]
    congr 1
    funext d
    by_cases hd : Deg d
    · simp only [hd, not_true_eq_false, false_and, ite_false, ite_true]
      unfold Pouw.RelativeSecurity.expect
      simp
    · simp only [hd, not_false_eq_true, true_and, ite_false]
      rfl
  rw [hw]
  refine expect_le_of_le zero_le_one fun d => ?_
  by_cases hd : Deg d
  · simp [hd]
  · simpa [hd] using hm d hd

/-! ## The pooled luck in the model -/

open Classical in
/-- Slot `i`'s factor: the block `blk i z` whose root is the `i`-th in (whose check coin is the `i`-th the challenger
releases), its check at slot `i`'s check coin passing on the words its program computes from its advice and its
dither, times `e^{η D}` at that program's shortfall. -/
noncomputable def poolY (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (pub η : P.Inst → ℝ) {m : ℕ} {Adv : Type}
    (inst : Fin m → P.Inst) (blk : Fin m → (Fin m → P.Draw) × (Fin m → P.Chk) → Fin m)
    (adv : Fin m → (Fin m → P.Draw) × (Fin m → P.Chk) → Adv) (Mw : Fin m → G.Mc.Alg (Adv × P.Draw) P.Tr)
    (i : Fin m) (z : (Fin m → P.Draw) × (Fin m → P.Chk)) : ℝ :=
  if P.ok (inst (blk i z)) (z.1 (blk i z)) (z.2 i) (G.Mc.eval (Mw (blk i z)) (adv (blk i z) z, z.1 (blk i z))) then
    Real.exp (η (inst (blk i z)) *
      roundShortfall (pub (inst (blk i z))) (G.cost μ (Mw (blk i z)) (adv (blk i z) z, z.1 (blk i z))))
  else 0

/-- A block's degenerate event has probability at most `κdeg B` when its advice reads nothing of its own dither. -/
theorem prob_deg_le {D C Adv : Type} [Fintype D] [Fintype C] {m : ℕ} (b : Fin m)
    (adv : (Fin m → D) × C → Adv) (Dg : Adv → D → Prop) {κ : ℝ} (hκ : 0 ≤ κ) (hDg : ∀ s, prob (Dg s) ≤ κ)
    (hown : ∀ d d' c, (∀ j, j ≠ b → d j = d' j) → adv (d, c) = adv (d', c)) :
    prob (fun z : (Fin m → D) × C => Dg (adv z) (z.1 b)) ≤ κ := by
  classical
  let e : (Fin m → D) × C ≃ D × (({j : Fin m // j ≠ b} → D) × C) :=
    ((Equiv.funSplitAt b D).prodCongr (Equiv.refl C)).trans (Equiv.prodAssoc _ _ _)
  let F : D × (({j : Fin m // j ≠ b} → D) × C) → Prop := fun w => Dg (adv (e.symm w)) w.1
  have hF : ∀ y, prob (fun x => F (x, y)) ≤ κ := fun y => by
    rcases isEmpty_or_nonempty D with hD | ⟨⟨x₀⟩⟩
    · unfold prob
      simp [hκ]
    · have hconst : ∀ x, adv (e.symm (x, y)) = adv (e.symm (x₀, y)) := fun x =>
        hown _ _ _ fun j hj => by
          simp [Equiv.funSplitAt, Equiv.piSplitAt, hj]
      have : (fun x => F (x, y)) = Dg (adv (e.symm (x₀, y))) := by
        funext x
        simp only [F, hconst x]
      rw [this]
      exact hDg _
  have he1 : ∀ w, (e.symm w).1 b = w.1 := fun w => by
    simp [e, Equiv.funSplitAt, Equiv.piSplitAt]
  calc prob (fun z : (Fin m → D) × C => Dg (adv z) (z.1 b))
      = prob (fun w => Dg (adv (e.symm w)) ((e.symm w).1 b)) := prob_equiv e _
    _ = prob F := by simp only [F, he1]
    _ ≤ κ := prob_le_of_slices F hκ hF

/-- **The pooled luck in the model** (CMP-152). A fleet of `m` blocks on registered instances `inst`, every block's
dither drawn by the challenger (`d`) and the blocks' check coins released one at a time, the `i`-th after the `i`-th
root (`c i`, slot `i`). The block in slot `i` (`blk i`, a bijection of slots and blocks) is fixed by the check coins
released before it, and so is its advice (`hblk`, `hadv`): a block's words are fixed at its root, before its check
coin. A block's advice may read every other block's dither, including those of blocks that root later, but nothing
of its own (`hown`). Under `SAKCheck` at `ζ`, the probability that every block's check passes and the blocks' rated
shortfalls sum past `t + Σ_b ζ_b` is at most `e^{-t} + m κdeg(B)`: `ln(1/δ)` once for the whole fleet, with no `ln m`
and no union bound over replicas. -/
theorem pool_luck (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (Reg : Set P.Inst) (pub η ζ : P.Inst → ℝ)
    (κdeg : ℕ → ℝ) (hG1 : G.G1) (hG2 : G.G2) (hP1 : G.P1) (hP2 : G.P2) (hS : SAKCheck G P μ Reg pub η ζ κdeg)
    (m : ℕ) (inst : Fin m → P.Inst) (hreg : ∀ b, inst b ∈ Reg) (B : ℕ) (Adv : Type)
    (hB : ∃ e : Adv → Fin (2 ^ B), Function.Injective e) (hκ : 0 ≤ κdeg B)
    (blk : Fin m → (Fin m → P.Draw) × (Fin m → P.Chk) → Fin m)
    (adv : Fin m → (Fin m → P.Draw) × (Fin m → P.Chk) → Adv) (Mw : Fin m → G.Mc.Alg (Adv × P.Draw) P.Tr)
    (hext : ∀ b x, G.ext (Mw b) x = 0) (hbij : ∀ z, Function.Bijective fun i => blk i z)
    (hblk : ∀ i d c c', (∀ j, j < i → c j = c' j) → blk i (d, c) = blk i (d, c'))
    (hadv : ∀ i d c c', (∀ j, j < i → c j = c' j) → adv (blk i (d, c)) (d, c) = adv (blk i (d, c)) (d, c'))
    (hown : ∀ b d d' c, (∀ j, j ≠ b → d j = d' j) → adv b (d, c) = adv b (d', c)) (t : ℝ) :
    prob (fun z : (Fin m → P.Draw) × (Fin m → P.Chk) =>
        (∀ i, P.ok (inst (blk i z)) (z.1 (blk i z)) (z.2 i)
          (G.Mc.eval (Mw (blk i z)) (adv (blk i z) z, z.1 (blk i z)))) ∧
        t + ∑ b, ζ (inst b) ≤
          ∑ b, η (inst b) * roundShortfall (pub (inst b)) (G.cost μ (Mw b) (adv b z, z.1 b))) ≤
      Real.exp (-t) + m * κdeg B := by
  classical
  have hS' := fun b (s : Adv) => hS hG1 hG2 hP1 hP2 (inst b) (hreg b) B Adv hB s (Mw b) (hext b)
  choose Dg hDg hDm using hS'
  let Y : Fin m → (Fin m → P.Draw) × (Fin m → P.Chk) → ℝ := fun i z =>
    Real.exp (-ζ (inst (blk i z))) * poolY G P μ pub η inst blk adv Mw i z
  let Deg : Fin m → (Fin m → P.Draw) × (Fin m → P.Chk) → Prop := fun i z =>
    Dg (blk i z) (adv (blk i z) z) (z.1 (blk i z))
  have hpY : ∀ i z, 0 ≤ poolY G P μ pub η inst blk adv Mw i z := fun i z => by
    unfold poolY
    split_ifs
    exacts [(Real.exp_pos _).le, le_rfl]
  have h0 : ∀ i z, 0 ≤ Y i z := fun i z => mul_nonneg (Real.exp_pos _).le (hpY i z)
  have hY : ∀ i d c c', (∀ j, j ≤ i → c j = c' j) → Y i (d, c) = Y i (d, c') := fun i d c c' h => by
    have hb := hblk i d c c' fun j hj => h j hj.le
    have ha := hadv i d c c' fun j hj => h j hj.le
    have hc := h i le_rfl
    simp only [Y, poolY, hb, hc]
    rw [hb] at ha
    rw [ha]
  have hD : ∀ i d c c', (∀ j, j < i → c j = c' j) → (Deg i (d, c) ↔ Deg i (d, c')) := fun i d c c' h => by
    have hb := hblk i d c c' h
    have ha := hadv i d c c' h
    simp only [Deg]
    rw [ha, hb]
  have h1 : ∀ i d c, ¬ Deg i (d, c) →
      expect (fun γ => Y i (d, Function.update c i γ)) ≤ Real.exp ((fun _ : Fin m => (0 : ℝ)) i) := by
    intro i d c hnd
    have hu : ∀ γ, ∀ j, j < i → Function.update c i γ j = c j := fun γ j hj =>
      Function.update_of_ne (ne_of_lt hj) _ _
    have hb : ∀ γ, blk i (d, Function.update c i γ) = blk i (d, c) := fun γ => hblk i d _ _ (hu γ)
    have ha : ∀ γ, adv (blk i (d, c)) (d, Function.update c i γ) = adv (blk i (d, c)) (d, c) := fun γ => by
      have := hadv i d _ _ (hu γ)
      rw [hb γ] at this
      exact this
    have hm := hDm (blk i (d, c)) (adv (blk i (d, c)) (d, c)) (d (blk i (d, c))) hnd
    have heq : (fun γ => Y i (d, Function.update c i γ)) = fun γ =>
        Real.exp (-ζ (inst (blk i (d, c)))) *
          (if P.ok (inst (blk i (d, c))) (d (blk i (d, c))) γ
              (G.Mc.eval (Mw (blk i (d, c))) (adv (blk i (d, c)) (d, c), d (blk i (d, c)))) then
            Real.exp (η (inst (blk i (d, c))) * roundShortfall (pub (inst (blk i (d, c))))
              (G.cost μ (Mw (blk i (d, c))) (adv (blk i (d, c)) (d, c), d (blk i (d, c))))) else 0) := by
      funext γ
      simp only [Y, poolY, hb γ, ha γ, Function.update_self]
    rw [heq, expect_mul_left]
    have hcm : expect (fun γ => if P.ok (inst (blk i (d, c))) (d (blk i (d, c))) γ
              (G.Mc.eval (Mw (blk i (d, c))) (adv (blk i (d, c)) (d, c), d (blk i (d, c)))) then
            Real.exp (η (inst (blk i (d, c))) * roundShortfall (pub (inst (blk i (d, c))))
              (G.cost μ (Mw (blk i (d, c))) (adv (blk i (d, c)) (d, c), d (blk i (d, c))))) else 0) =
        checkMoment G P μ (pub (inst (blk i (d, c)))) (η (inst (blk i (d, c)))) (inst (blk i (d, c)))
          (adv (blk i (d, c)) (d, c)) (Mw (blk i (d, c))) (d (blk i (d, c))) := rfl
    rw [hcm]
    calc Real.exp (-ζ (inst (blk i (d, c)))) * checkMoment G P μ (pub (inst (blk i (d, c))))
            (η (inst (blk i (d, c)))) (inst (blk i (d, c))) (adv (blk i (d, c)) (d, c)) (Mw (blk i (d, c)))
            (d (blk i (d, c)))
        ≤ Real.exp (-ζ (inst (blk i (d, c)))) * Real.exp (ζ (inst (blk i (d, c)))) :=
          mul_le_mul_of_nonneg_left hm (Real.exp_pos _).le
      _ = Real.exp ((fun _ : Fin m => (0 : ℝ)) i) := by
          rw [← Real.exp_add, neg_add_cancel]
  have hmain := prob_pool_le_exists m Y (fun _ => 0) Deg h0 hY hD h1 t
  simp only [Finset.sum_const_zero, zero_sub] at hmain
  have hsub : ∀ z : (Fin m → P.Draw) × (Fin m → P.Chk),
      ((∀ i, P.ok (inst (blk i z)) (z.1 (blk i z)) (z.2 i)
          (G.Mc.eval (Mw (blk i z)) (adv (blk i z) z, z.1 (blk i z)))) ∧
        t + ∑ b, ζ (inst b) ≤
          ∑ b, η (inst b) * roundShortfall (pub (inst b)) (G.cost μ (Mw b) (adv b z, z.1 b))) →
      Real.exp t ≤ ∏ i, Y i z := by
    rintro z ⟨hok, hsum⟩
    have hYi : ∀ i, Y i z = Real.exp (η (inst (blk i z)) * roundShortfall (pub (inst (blk i z)))
        (G.cost μ (Mw (blk i z)) (adv (blk i z) z, z.1 (blk i z))) - ζ (inst (blk i z))) := fun i => by
      simp only [Y, poolY, ite_eq_left (hok i)]
      rw [← Real.exp_add]
      ring_nf
    simp only [hYi]
    rw [← Real.exp_sum]
    apply Real.exp_le_exp.mpr
    have hre := (hbij z).sum_comp (fun b => η (inst b) * roundShortfall (pub (inst b))
        (G.cost μ (Mw b) (adv b z, z.1 b)) - ζ (inst b))
    rw [hre, Finset.sum_sub_distrib]
    linarith
  have hdeg : prob (fun z => ∃ i, Deg i z) ≤ m * κdeg B := by
    calc prob (fun z => ∃ i, Deg i z)
        ≤ prob (fun z : (Fin m → P.Draw) × (Fin m → P.Chk) =>
            ∃ b ∈ (univ : Finset (Fin m)), Dg b (adv b z) (z.1 b)) :=
          prob_mono fun z ⟨i, hi⟩ => ⟨blk i z, mem_univ _, hi⟩
      _ ≤ ∑ b ∈ (univ : Finset (Fin m)), prob (fun z : (Fin m → P.Draw) × (Fin m → P.Chk) =>
            Dg b (adv b z) (z.1 b)) := prob_exists_le _ _
      _ ≤ ∑ _b ∈ (univ : Finset (Fin m)), κdeg B :=
          Finset.sum_le_sum fun b _ => prob_deg_le b (adv b) (Dg b) hκ (hDg b)
            fun d d' c h => hown b d d' c h
      _ = m * κdeg B := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc _ ≤ prob (fun z => Real.exp t ≤ ∏ i, Y i z) := prob_mono hsub
    _ ≤ Real.exp (-t) + prob (fun z => ∃ i, Deg i z) := hmain
    _ ≤ Real.exp (-t) + m * κdeg B := add_le_add le_rfl hdeg

/-! ## Pooling replicas by convexity -/

/-- **Pooling replicas by convexity**: if replica `u`'s shortfall `D u`, scaled by its rate `η u`, has its exponential
below a nonnegative `M u` of average at most 1 wherever `A` holds, the replicas' total shortfall exceeds
`t · ∑ u, 1 / η u` on `A` with probability at most `e^{-t}`. The replicas' moments need no independence: the weights
`(1 / η u) / ∑ v, 1 / η v` put the total's exponential below their mixture of the `M u` (Jensen), whose average is at
most 1. -/
theorem prob_convex_pool {Ω ι : Type*} [Fintype Ω] [Fintype ι] [Nonempty ι] (M D : ι → Ω → ℝ) (η : ι → ℝ)
    (hη : ∀ u, 0 < η u) (hM0 : ∀ u ω, 0 ≤ M u ω) (hM : ∀ u, expect (M u) ≤ 1) (A : Ω → Prop)
    (hZ : ∀ u ω, A ω → Real.exp (η u * D u ω) ≤ M u ω) (t : ℝ) :
    prob (fun ω => A ω ∧ t * ∑ u, 1 / η u ≤ ∑ u, D u ω) ≤ Real.exp (-t) := by
  set S := ∑ u, 1 / η u with hS
  have hSpos : 0 < S := Finset.sum_pos (fun u _ => one_div_pos.mpr (hη u)) Finset.univ_nonempty
  set w : ι → ℝ := fun u => 1 / η u / S with hw
  have hw0 : ∀ u ∈ (univ : Finset ι), 0 ≤ w u := fun u _ => div_nonneg (one_div_pos.mpr (hη u)).le hSpos.le
  have hw1 : ∑ u, w u = 1 := by rw [hw, ← Finset.sum_div, div_self hSpos.ne']
  refine prob_le_of_expect_le_one (f := fun ω => ∑ u, w u * M u ω)
    (fun ω => Finset.sum_nonneg fun u hu => mul_nonneg (hw0 u hu) (hM0 u ω)) ?_ t ?_
  · rw [Pouw.RelativeSecurity.expect_sum]
    calc ∑ u, Pouw.RelativeSecurity.expect (fun ω => w u * M u ω)
          = ∑ u, w u * Pouw.RelativeSecurity.expect (M u) := by
          simp only [Pouw.RelativeSecurity.expect_mul_left]
      _ ≤ ∑ u, w u * 1 := Finset.sum_le_sum fun u hu => mul_le_mul_of_nonneg_left (hM u) (hw0 u hu)
      _ = 1 := by simp only [mul_one, hw1]
  · rintro ω ⟨hA, ht⟩
    have hmix : ∑ u, w u • (η u * D u ω) = (∑ u, D u ω) / S := by
      rw [Finset.sum_div]
      refine Finset.sum_congr rfl fun u _ => ?_
      rw [hw, smul_eq_mul]
      field_simp [(hη u).ne']
    have htS : t ≤ (∑ u, D u ω) / S := by rw [le_div_iff₀ hSpos]; linarith
    calc Real.exp t ≤ Real.exp (∑ u, w u • (η u * D u ω)) := Real.exp_le_exp.mpr (hmix ▸ htS)
      _ ≤ ∑ u, w u • Real.exp (η u * D u ω) :=
          convexOn_exp.map_sum_le hw0 hw1 fun _ _ => Set.mem_univ _
      _ ≤ ∑ u, w u * M u ω := Finset.sum_le_sum fun u hu => by
          rw [smul_eq_mul]; exact mul_le_mul_of_nonneg_left (hZ u ω hA) (hw0 u hu)

/-- **A replica's moment at `FleetSound`'s quantifier**: replica `u`'s round `w` has a nonnegative factor of its
advice and its own coin, averaging at most 1 over that coin at every advice (`WorkSound`'s round), and its advice
reads its own coins before `w` and every other replica's coins. Then the replica's rounds' product averages at most
1 over every replica's coins. -/
theorem expect_replica_rounds {Ω Adv : Type} [Fintype Ω] {N k : ℕ}
    (adv : Fin N → (Fin N → Fin k → Ω) → Fin k → Adv)
    (hadv : ∀ u x x' w, (∀ v : Fin k, v < w → x u v = x' u v) → (∀ u', u' ≠ u → x u' = x' u') →
      adv u x w = adv u x' w)
    (F : Fin N → Fin k → Adv → Ω → ℝ) (h0 : ∀ u w s ω, 0 ≤ F u w s ω) (h1 : ∀ u w s, expect (F u w s) ≤ 1)
    (u : Fin N) : expect (fun x : Fin N → Fin k → Ω => ∏ w, F u w (adv u x w) (x u w)) ≤ 1 := by
  let e := Equiv.funSplitAt u (Fin k → Ω)
  have hu : ∀ a y, e.symm (a, y) u = a := fun a y => by simp [e, Equiv.funSplitAt, Equiv.piSplitAt]
  have hne : ∀ a a' y u', u' ≠ u → e.symm (a, y) u' = e.symm (a', y) u' := fun a a' y u' h => by
    simp [e, Equiv.funSplitAt, Equiv.piSplitAt, h]
  rw [← Pouw.SecurityProofs.Dither.expect_equiv e.symm, Pouw.RelativeSecurity.expect_prod_swap]
  refine Pouw.RelativeSecurity.expect_le_of_le zero_le_one fun y => ?_
  have hr := expect_prod_rounds (fun a w => adv u (e.symm (a, y)) w)
    (fun a a' w h => hadv u _ _ w (fun v hv => by rw [hu, hu]; exact h v hv) fun u' h' => hne a a' y u' h')
    (F u) (h0 u) (h1 u)
  simp only [hu] at hr ⊢
  exact hr

/-- **The fleet's luck by convexity**, at `FleetSound`'s quantifier (`expect_replica_rounds`): where `A` holds and
each replica's rated shortfall has its exponential below its rounds' product, the replicas' total shortfall exceeds
`t · ∑ u, 1 / η u` with probability at most `e^{-t}`. -/
theorem fleet_convex_luck {Ω Adv : Type} [Fintype Ω] {N k : ℕ} [NeZero N]
    (adv : Fin N → (Fin N → Fin k → Ω) → Fin k → Adv)
    (hadv : ∀ u x x' w, (∀ v : Fin k, v < w → x u v = x' u v) → (∀ u', u' ≠ u → x u' = x' u') →
      adv u x w = adv u x' w)
    (F : Fin N → Fin k → Adv → Ω → ℝ) (h0 : ∀ u w s ω, 0 ≤ F u w s ω) (h1 : ∀ u w s, expect (F u w s) ≤ 1)
    (η : Fin N → ℝ) (hη : ∀ u, 0 < η u) (D : Fin N → (Fin N → Fin k → Ω) → ℝ) (A : (Fin N → Fin k → Ω) → Prop)
    (hZ : ∀ u x, A x → Real.exp (η u * D u x) ≤ ∏ w, F u w (adv u x w) (x u w)) (t : ℝ) :
    prob (fun x => A x ∧ t * ∑ u, 1 / η u ≤ ∑ u, D u x) ≤ Real.exp (-t) :=
  prob_convex_pool (fun u x => ∏ w, F u w (adv u x w) (x u w)) D η hη
    (fun _ _ => Finset.prod_nonneg fun _ _ => h0 _ _ _ _) (expect_replica_rounds adv hadv F h0 h1) A hZ t

/-- **Pooling the accepted replicas by convexity** (RT-POOL's RP24): ACCEL's verdict is per replica, so a failed
replica voids only its own period and the others are credited. One Markov bound on `prob_convex_pool`'s mixture covers
every set `S` of accepted replicas at once, with no union over sets: on `S`, Jensen within `S` gives
`∑_{u ∈ S} D u ≤ Λ_S (ln (Λ / Λ_S) + t)`, where `Λ_S = ∑_{u ∈ S} 1 / η u` and `Λ = ∑ u, 1 / η u`. At `S` every replica
it is `prob_convex_pool`. -/
theorem prob_convex_pool_accepted {Ω ι : Type*} [Fintype Ω] [Fintype ι] [Nonempty ι] (M D : ι → Ω → ℝ)
    (η : ι → ℝ) (hη : ∀ u, 0 < η u) (hM0 : ∀ u ω, 0 ≤ M u ω) (hM : ∀ u, expect (M u) ≤ 1)
    (acc : ι → Ω → Prop) (hZ : ∀ u ω, acc u ω → Real.exp (η u * D u ω) ≤ M u ω) (t : ℝ) :
    prob (fun ω => ∃ S : Finset ι, (∀ u ∈ S, acc u ω) ∧ S.Nonempty ∧
      (∑ u ∈ S, 1 / η u) * (Real.log ((∑ u, 1 / η u) / ∑ u ∈ S, 1 / η u) + t) ≤ ∑ u ∈ S, D u ω) ≤
      Real.exp (-t) := by
  classical
  set L := ∑ u, 1 / η u
  have hLpos : 0 < L := Finset.sum_pos (fun u _ => one_div_pos.mpr (hη u)) Finset.univ_nonempty
  set w : ι → ℝ := fun u => 1 / η u / L with hw
  have hw0 : ∀ u, 0 ≤ w u := fun u => div_nonneg (one_div_pos.mpr (hη u)).le hLpos.le
  have hw1 : ∑ u, w u = 1 := by rw [hw, ← Finset.sum_div, div_self hLpos.ne']
  refine prob_le_of_expect_le_one (f := fun ω => ∑ u, w u * M u ω)
    (fun ω => Finset.sum_nonneg fun u _ => mul_nonneg (hw0 u) (hM0 u ω)) ?_ t ?_
  · rw [Pouw.RelativeSecurity.expect_sum]
    calc ∑ u, Pouw.RelativeSecurity.expect (fun ω => w u * M u ω)
          = ∑ u, w u * Pouw.RelativeSecurity.expect (M u) := by
          simp only [Pouw.RelativeSecurity.expect_mul_left]
      _ ≤ ∑ u, w u * 1 := Finset.sum_le_sum fun u _ => mul_le_mul_of_nonneg_left (hM u) (hw0 u)
      _ = 1 := by simp only [mul_one, hw1]
  · rintro ω ⟨S, hacc, hne, hD⟩
    set LS := ∑ u ∈ S, 1 / η u
    have hLSpos : 0 < LS := Finset.sum_pos (fun u _ => one_div_pos.mpr (hη u)) hne
    set v : ι → ℝ := fun u => 1 / η u / LS with hv
    have hv0 : ∀ u ∈ S, 0 ≤ v u := fun u _ => div_nonneg (one_div_pos.mpr (hη u)).le hLSpos.le
    have hv1 : ∑ u ∈ S, v u = 1 := by rw [hv, ← Finset.sum_div, div_self hLSpos.ne']
    have hmix : ∑ u ∈ S, v u • (η u * D u ω) = (∑ u ∈ S, D u ω) / LS := by
      rw [Finset.sum_div]
      refine Finset.sum_congr rfl fun u _ => ?_
      rw [hv, smul_eq_mul]
      field_simp [(hη u).ne']
    have hjensen : Real.exp ((∑ u ∈ S, D u ω) / LS) ≤ ∑ u ∈ S, v u * Real.exp (η u * D u ω) := by
      calc Real.exp ((∑ u ∈ S, D u ω) / LS) = Real.exp (∑ u ∈ S, v u • (η u * D u ω)) := by rw [hmix]
        _ ≤ ∑ u ∈ S, v u • Real.exp (η u * D u ω) :=
            convexOn_exp.map_sum_le hv0 hv1 fun _ _ => Set.mem_univ _
        _ = ∑ u ∈ S, v u * Real.exp (η u * D u ω) := by simp only [smul_eq_mul]
    have hratio : 0 < L / LS := div_pos hLpos hLSpos
    have ht : Real.log (L / LS) + t ≤ (∑ u ∈ S, D u ω) / LS := by
      rw [le_div_iff₀ hLSpos, mul_comm]
      exact hD
    have hc : 0 ≤ LS / L := div_nonneg hLSpos.le hLpos.le
    have key : Real.exp t = LS / L * Real.exp (Real.log (L / LS) + t) := by
      rw [Real.exp_add, Real.exp_log hratio, ← mul_assoc, div_mul_div_comm, mul_comm LS L,
        div_self (mul_pos hLpos hLSpos).ne', one_mul]
    have hwv : ∀ u, LS / L * (v u * Real.exp (η u * D u ω)) = w u * Real.exp (η u * D u ω) := fun u => by
      show LS / L * (1 / η u / LS * Real.exp (η u * D u ω)) = 1 / η u / L * Real.exp (η u * D u ω)
      rw [← mul_assoc, div_mul_div_comm, mul_comm LS (1 / η u), mul_div_mul_right _ _ hLSpos.ne']
    calc Real.exp t = LS / L * Real.exp (Real.log (L / LS) + t) := key
      _ ≤ LS / L * Real.exp ((∑ u ∈ S, D u ω) / LS) :=
          mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ht) hc
      _ ≤ LS / L * ∑ u ∈ S, v u * Real.exp (η u * D u ω) := mul_le_mul_of_nonneg_left hjensen hc
      _ = ∑ u ∈ S, w u * Real.exp (η u * D u ω) := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun u _ => hwv u
      _ ≤ ∑ u ∈ S, w u * M u ω :=
          Finset.sum_le_sum fun u hu => mul_le_mul_of_nonneg_left (hZ u ω (hacc u hu)) (hw0 u)
      _ ≤ ∑ u, w u * M u ω :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ S)
            fun u _ _ => mul_nonneg (hw0 u) (hM0 u ω)

/-- `fleet_convex_luck` on the accepted set (RT-POOL's RP24): the same rounds and advice, with each replica's moment
bound read only where it is accepted. -/
theorem fleet_convex_luck_accepted {Ω Adv : Type} [Fintype Ω] {N k : ℕ} [NeZero N]
    (adv : Fin N → (Fin N → Fin k → Ω) → Fin k → Adv)
    (hadv : ∀ u x x' w, (∀ v : Fin k, v < w → x u v = x' u v) → (∀ u', u' ≠ u → x u' = x' u') →
      adv u x w = adv u x' w)
    (F : Fin N → Fin k → Adv → Ω → ℝ) (h0 : ∀ u w s ω, 0 ≤ F u w s ω) (h1 : ∀ u w s, expect (F u w s) ≤ 1)
    (η : Fin N → ℝ) (hη : ∀ u, 0 < η u) (D : Fin N → (Fin N → Fin k → Ω) → ℝ)
    (acc : Fin N → (Fin N → Fin k → Ω) → Prop)
    (hZ : ∀ u x, acc u x → Real.exp (η u * D u x) ≤ ∏ w, F u w (adv u x w) (x u w)) (t : ℝ) :
    prob (fun x => ∃ S : Finset (Fin N), (∀ u ∈ S, acc u x) ∧ S.Nonempty ∧
      (∑ u ∈ S, 1 / η u) * (Real.log ((∑ u, 1 / η u) / ∑ u ∈ S, 1 / η u) + t) ≤ ∑ u ∈ S, D u x) ≤
      Real.exp (-t) :=
  prob_convex_pool_accepted (fun u x => ∏ w, F u w (adv u x w) (x u w)) D η hη
    (fun _ _ => Finset.prod_nonneg fun _ _ => h0 _ _ _ _) (expect_replica_rounds adv hadv F h0 h1) acc hZ t

/-- **Convexity pooling is tight against per-replica premises**: `N` replicas that all cheat on one shared coin of
`n` values each meet `prob_convex_pool`'s premises at `η = 1`, and their total shortfall reaches `N · ln n` with
probability exactly `e^{-ln n}`, whatever `N` is. So no bound from per-replica moments alone falls with the number of
replicas. -/
theorem convex_pool_tight (N n : ℕ) [NeZero n] :
    (∀ (_ : Fin N) (ω : Fin n), 0 ≤ (if ω = 0 then (n : ℝ) else 0)) ∧
    (∀ (_ : Fin N), Pouw.RelativeSecurity.expect (fun ω : Fin n => if ω = 0 then (n : ℝ) else 0) ≤ 1) ∧
    (∀ (_ : Fin N) (ω : Fin n), ω = 0 → Real.exp (1 * Real.log n) ≤ (if ω = 0 then (n : ℝ) else 0)) ∧
    prob (fun ω : Fin n => ω = 0 ∧ Real.log n * ∑ _u : Fin N, 1 / (1 : ℝ) ≤ ∑ _u : Fin N, Real.log n) =
      Real.exp (-Real.log n) := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  refine ⟨fun _ ω => by split_ifs <;> positivity, fun _ => ?_, fun _ ω h => ?_, ?_⟩
  · unfold Pouw.RelativeSecurity.expect
    rw [Finset.sum_ite_eq' Finset.univ (0 : Fin n) (fun _ => (n : ℝ))]
    simp [Fintype.card_fin]
  · rw [one_mul, Real.exp_log hn]
    simp [h]
  · unfold prob
    have key : ∀ ω : Fin n,
        (ω = 0 ∧ Real.log n * ∑ _u : Fin N, 1 / (1 : ℝ) ≤ ∑ _u : Fin N, Real.log n) ↔ ω = 0 :=
      fun ω => by simp [Finset.sum_const, mul_comm]
    simp_rw [key]
    simp [Finset.filter_eq', Fintype.card_fin, Real.exp_neg, Real.exp_log hn]

end Pouw.SecurityProofs.Pool
