import Proofs.Pouw.Dither

/-!
# Pooling a fleet's replicas by convexity (tier A)

A fleet's sampling blocks are not one sequence: a block's work can read the dither of a block on
another replica that the challenger issues after it, so `WorkMoment` holds per replica. The union bound over replicas
paid `ln(N/δ_coin)` of luck a replica; convexity pays `ln(N/(s δ_coin))` on `s` accepted replicas' credit.

* **Pooling the accepted replicas** (`prob_convex_pool_accepted`): each replica's moment, of
  average at most 1, with no independence between replicas. The weights `1/η` put the shortfalls' exponential below
  the replicas' mixture of moments (Jensen), and one Markov bound on the mixture covers every set `S` of accepted
  replicas at once, at `Λ_S (ln (Λ / Λ_S) + t)`, `Λ_S = Σ_{u ∈ S} 1/η_u`. A failed replica voids only its own period.
* **At the sampling blocks** (`expect_replica_rounds`, `fleet_convex_luck_accepted`): a replica's blocks' product,
  `WorkMoment`'s moment, averages at most 1 over every replica's coins when its advice reads every other replica's.
* **At `FleetSound`'s quantifier** (`expect_replica_le`, `prob_replica_le`): under `OwnDither` and
  `CoinsConfidential`, a replica's moment and bad event, bounded at every prover, stay bounded over the challenger's
  randomness, each sliced at the other replicas' coins.
* `FleetSound`: the guarantee `Definitions.Pouw.Guarantees.Dither.FleetSound`.
* **Its rate is one replica's** (`convex_pool_tight`): replicas that cheat on one shared coin attain the bound, so no
  bound from per-replica moments falls with the number of replicas.
* **Average moments don't pool** (`not_pool_of_average_moments`, `parity_all_pass`): sampling blocks whose 1-bit
  dithers each guess their own as the sum of the others' pass with probability 1/2 at every value of the others', the
  form `SAKMatmul` gives, yet all pass together with probability 1/2. A product over replicas would need moments
  pointwise in the other replicas' dithers.
-/

namespace Pouw.SecurityProofs.Pool

open Pouw.RelativeSecurity Pouw.Dither Pouw.SecurityProofs.Dither Finset

/-! ## Pooling the accepted replicas by convexity -/

/-- **Pooling the accepted replicas by convexity**: if replica `u`'s shortfall `D u`, scaled by
its rate `η u`, has its exponential below a nonnegative moment `M u` of average at most 1 wherever `u` is accepted
(`acc u`), then with probability at least `1 − e^{-t}` every nonempty set `S` of accepted replicas has
`Σ_{u ∈ S} D u < Λ_S (ln (Λ / Λ_S) + t)`, where `Λ_S = Σ_{u ∈ S} 1 / η u` and `Λ = Σ_u 1 / η u`. The replicas'
moments need no independence: the weights `(1 / η u) / Λ` put a mixture of the `M u`, of average at most 1, above
`e^t` on that event (Jensen within `S`), and one Markov bound covers every `S` at once, with no union over sets. -/
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

/-! ## At the sampling blocks -/

/-- **A replica's moment over every replica's coins**: replica `u`'s sampling block `w` has a nonnegative factor of
its advice and its own coin, averaging at most 1 over that coin at every advice (`WorkMoment`'s block), and its
advice reads its own coins before `w` and every other replica's coins. Then the replica's blocks' product averages at
most 1 over every replica's coins. -/
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

/-- **The fleet's luck on its accepted replicas, at the sampling blocks**:
`prob_convex_pool_accepted` at each replica's blocks' product (`expect_replica_rounds`), with each replica's moment
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

/-! ## At `FleetSound`'s quantifier -/

/-- **A replica's moment over the challenger's randomness**: the challenger's randomness `Ω` gives the replicas'
coins `coin` the uniform distribution (`OwnDither`'s form), and the strategy `A` run on replica `u` reads only the
other replicas' coins (`CoinsConfidential`'s form). A function `g` of the prover and of the oracle and replica `u`'s
coins, averaging at most `c ≥ 0` at every prover, then averages at most `c` at `A`'s: sliced at the other replicas'
coins, `A` is one prover and `u`'s coins are uniform. -/
theorem expect_replica_le {Orc Y Ω : Type} {Prv : Type*} [Fintype Orc] [Fintype Y] [Fintype Ω] {N : ℕ}
    (coin : Ω → Fin N → Y) (hown : ∀ f : (Fin N → Y) → ℝ, avg (fun ω => f (coin ω)) = avg f) (A : Ω → Prv)
    (u : Fin N) (hconf : ∀ ω ω', (∀ v, v ≠ u → coin ω v = coin ω' v) → A ω = A ω') (g : Prv → Orc × Y → ℝ)
    {c : ℝ} (hc : 0 ≤ c) (hg : ∀ ω, expect (g (A ω)) ≤ c) :
    expect (fun oω : Orc × Ω => g (A oω.2) (oω.1, coin oω.2 u)) ≤ c := by
  classical
  rcases isEmpty_or_nonempty Ω with hΩ | ⟨⟨ω₀⟩⟩
  · rw [expect_prod_swap]
    unfold Pouw.RelativeSecurity.expect
    simp [hc]
  have hsurj : ∀ y : Fin N → Y, ∃ ω, coin ω = y := by
    intro y
    by_contra hy
    push Not at hy
    have h := hown (fun y' => if y' = y then (1 : ℝ) else 0)
    have hl : avg (fun ω => (fun y' : Fin N → Y => if y' = y then (1 : ℝ) else 0) (coin ω)) = 0 := by
      unfold avg
      simp [hy]
    have hr : 0 < avg (fun y' : Fin N → Y => if y' = y then (1 : ℝ) else 0) := by
      unfold avg
      rw [Finset.sum_ite_eq' Finset.univ y (fun _ => (1 : ℝ)), ite_eq_left (Finset.mem_univ y)]
      exact div_pos one_pos (by exact_mod_cast Fintype.card_pos_iff.mpr ⟨y⟩)
    linarith
  choose s hs using hsurj
  have hA : ∀ ω, A ω = A (s (coin ω)) := fun ω => hconf _ _ fun v _ => by rw [hs]
  let e := Equiv.piSplitAt u (fun _ : Fin N => Y)
  have hu : ∀ a r, e.symm (a, r) u = a := fun a r => by simp [e, Equiv.piSplitAt_symm_apply]
  have hPv : ∀ a r, A (s (e.symm (a, r))) = A (s (e.symm (coin ω₀ u, r))) := fun a r =>
    hconf _ _ fun v hv => by
      rw [hs, hs]
      simp [e, Equiv.piSplitAt_symm_apply, hv]
  let H : (Fin N → Y) → ℝ := fun y => expect fun o : Orc => g (A (s y)) (o, y u)
  calc expect (fun oω : Orc × Ω => g (A oω.2) (oω.1, coin oω.2 u))
      = expect (fun ω => H (coin ω)) := by
        rw [expect_prod_swap]
        congr 1
        funext ω
        simp only [H]
        rw [← hA ω]
    _ = expect H := hown H
    _ = expect (fun ar => H (e.symm ar)) := (expect_equiv e.symm H).symm
    _ = expect (fun r => expect fun a => H (e.symm (a, r))) := expect_prod_swap _
    _ ≤ c := expect_le_of_le hc fun r => by
        have h : (fun a => H (e.symm (a, r))) = fun a => expect fun o => g (A (s (e.symm (coin ω₀ u, r)))) (o, a) := by
          funext a
          simp only [H, hPv, hu]
        rw [h, ← expect_prod_swap]
        exact hg _

/-- `expect_replica_le` for an event: one of probability at most `c ≥ 0` at every prover has probability at most `c`
at the strategy's. -/
theorem prob_replica_le {Orc Y Ω : Type} {Prv : Type*} [Fintype Orc] [Fintype Y] [Fintype Ω] {N : ℕ}
    (coin : Ω → Fin N → Y) (hown : ∀ f : (Fin N → Y) → ℝ, avg (fun ω => f (coin ω)) = avg f) (A : Ω → Prv)
    (u : Fin N) (hconf : ∀ ω ω', (∀ v, v ≠ u → coin ω v = coin ω' v) → A ω = A ω') (E : Prv → Orc × Y → Prop)
    {c : ℝ} (hc : 0 ≤ c) (hE : ∀ ω, prob (E (A ω)) ≤ c) :
    prob (fun oω : Orc × Ω => E (A oω.2) (oω.1, coin oω.2 u)) ≤ c := by
  classical
  rw [prob_eq_expect]
  exact expect_replica_le coin hown A u hconf (fun p z => if E p z then 1 else 0) hc
    fun ω => (prob_eq_expect (E (A ω))).symm.trans_le (hE ω)

/-! ## Convexity's rate is one replica's -/

/-- **Convexity pooling is tight against per-replica premises**: `N` replicas that all cheat on one shared coin of
`n` values each meet `prob_convex_pool_accepted`'s premises at `η = 1`, every replica accepted, and their total
shortfall reaches `N · ln n` (the bound `t Λ` at `S` every replica, where `ln (Λ / Λ_S) = 0`, at `t = ln n`) with
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

/-! ## Average moments don't pool: the parity counterexample -/

/-- Sampling block `i`'s factor when every block's work reads every other block's (1-bit) dither: it guesses its own
dither as the sum of the others' mod 2, passes when the guess is right (the dithers' sum is 0), and then skips work
worth `ln 2 / η`, so its factor `1_pass · e^{η D}` is 2. -/
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

/-- **Average moments don't pool** (a fleet's blocks aren't one sequence, at its smallest): two sampling blocks whose
factors are nonnegative and average at most 1 over their own coin at every value of the other's, but whose product
averages 2. So `SecurityProofs.Dither.expect_prod_adapted` fails without its adaptedness hypothesis, and a moment
averaged over the block's dither (`SAKMatmul`'s form) does not bound blocks whose work reads each other's dithers:
replicas pool by convexity (`prob_convex_pool_accepted`), not by a product. -/
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

end Pouw.SecurityProofs.Pool

namespace Pouw.SecurityProofs.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.SecurityProofs.Pool Finset

/-- `Definitions.Pouw.Guarantees.Dither.FleetSound`: `prob_convex_pool_accepted` at the replicas' moments, each over
the challenger's randomness sliced at the other replicas' coins (`expect_replica_le`), with the luck at
`t = ln(1/δ_coin)`; the replicas' bad events by the union bound (`prob_replica_le`). -/
theorem FleetSound : Pouw.Guarantees.Dither.FleetSound := by
  intro G P k N Obs verdict dev Pd ε κ C η κmax δ Ω _ coin A hη hδ hκ hmom hown hconf hA
  classical
  rcases isEmpty_or_nonempty (Fin N) with hN | hN
  · refine le_trans (prob_mono (q := fun _ => False) fun oω ⟨S, hS, _⟩ => ?_) ?_
    · obtain ⟨u, _⟩ := hS
      exact isEmptyElim u
    · rw [prob_false]
      exact add_nonneg hδ.le (Finset.sum_nonneg fun u _ => hκ u)
  unfold MomentSound at hmom
  choose D M Bad hD0 hM0 hM hBad hZ using hmom
  obtain ⟨acc, hacc⟩ : ∃ acc : Fin N → P.Orc × Ω → Prop, ∀ u oω, acc u oω ↔
      ((A u oω.2).Passes (verdict u) (dev u) oω.1 (coin oω.2 u) ∧ ¬ Bad u (A u oω.2) (oω.1, coin oω.2 u) ∧
        ∃ z, (Pd u).Fits ((A u oω.2).timedUse oω.1 (coin oω.2 u)) z) :=
    ⟨_, fun _ _ => Iff.rfl⟩
  have hδe : Real.exp (-Real.log (1 / δ)) = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  have hMu : ∀ u, Pouw.RelativeSecurity.expect
      (fun oω : P.Orc × Ω => M u (A u oω.2) (oω.1, coin oω.2 u)) ≤ 1 := by
    intro u
    refine expect_replica_le (Orc := P.Orc) (Y := Fin k → P.Draw × P.Chk) (Prv := Prover G P k) coin hown (A u) u
      (hconf u) (M u) zero_le_one fun ω => ?_
    exact hM u (A u ω)
  have hluck : prob (fun oω : P.Orc × Ω => ∃ S : Finset (Fin N), (∀ u ∈ S, acc u oω) ∧ S.Nonempty ∧
      (∑ u ∈ S, 1 / η u) * (Real.log ((∑ u, 1 / η u) / ∑ u ∈ S, 1 / η u) + Real.log (1 / δ)) ≤
        ∑ u ∈ S, D u (A u oω.2) (oω.1, coin oω.2 u)) ≤ δ :=
    (prob_convex_pool_accepted (fun u oω => M u (A u oω.2) (oω.1, coin oω.2 u))
      (fun u oω => D u (A u oω.2) (oω.1, coin oω.2 u)) η hη (fun u oω => hM0 u _ _)
      hMu acc
      (fun u oω h => by
        obtain ⟨hv, hb, z, hfit⟩ := (hacc u oω).mp h
        exact (hZ u _ _ z hv hb hfit).1)
      (Real.log (1 / δ))).trans hδe.le
  have hbad : prob (fun oω : P.Orc × Ω =>
      ∃ u ∈ (univ : Finset (Fin N)), Bad u (A u oω.2) (oω.1, coin oω.2 u)) ≤ ∑ u, κmax u :=
    (prob_exists_le _ _).trans (Finset.sum_le_sum fun u _ =>
      prob_replica_le coin hown (A u) u (hconf u) (Bad u) (hκ u) fun ω => (hBad u (A u ω)).trans (hA u ω))
  refine le_trans (prob_mono (q := fun oω : P.Orc × Ω =>
      (∃ u ∈ (univ : Finset (Fin N)), Bad u (A u oω.2) (oω.1, coin oω.2 u)) ∨
        ∃ S : Finset (Fin N), (∀ u ∈ S, acc u oω) ∧ S.Nonempty ∧
          (∑ u ∈ S, 1 / η u) * (Real.log ((∑ u, 1 / η u) / ∑ u ∈ S, 1 / η u) + Real.log (1 / δ)) ≤
            ∑ u ∈ S, D u (A u oω.2) (oω.1, coin oω.2 u)) fun oω hev => ?_)
    ((prob_or_le _ _).trans ((add_le_add hbad hluck).trans (le_of_eq (add_comm _ _))))
  obtain ⟨S, hne, z, hpf, hnot⟩ := hev
  by_cases hb : ∃ u ∈ S, Bad u (A u oω.2) (oω.1, coin oω.2 u)
  · obtain ⟨u, _, hu⟩ := hb
    exact Or.inl ⟨u, mem_univ u, hu⟩
  · push Not at hb
    refine Or.inr ⟨S, fun u hu => (hacc u oω).mpr ⟨(hpf u hu).1, hb u hu, z u, (hpf u hu).2⟩, hne, ?_⟩
    exact not_lt.mp fun hlt => hnot ⟨fun u => D u (A u oω.2) (oω.1, coin oω.2 u),
      fun u hu => ⟨hD0 u _ _, (hZ u _ _ (z u) (hpf u hu).1 (hb u hu) (hpf u hu).2).2⟩, hlt.le⟩

end Pouw.SecurityProofs.Dither
