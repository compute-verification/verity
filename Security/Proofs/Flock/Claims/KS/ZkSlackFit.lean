import Proofs.Flock.Claims.KS.ZkSlackW
import Proofs.Flock.Claims.KS.PlainSlackFit

/-!
# The slack fits the masked weight's room at the coupling's chances

`ZkSimWJ` asks that the slack's escapes fit the room `zk_fork_room` leaves in the masked weight
(`Claims.Flock.SlackFitW`). At chances at most `ε ≤ 1/2` before the last round and at most `1` there, a fork's slack
weights before the last round total at most `8ε` (`Claims.Flock.sum_slackW_le`), so:

* `sum_slackCW_le`, `sum_slackTW_le`: by Cauchy–Schwarz each table's slack weights total at most its weight times
  `|IZW|·8ε`, and every table's at most the joined weights times `c·8ε`, `c` a bound on `|IZW|`;
* `slackFitW_of_le`: the slack fits where its weights are at most `ε'` times the joined ones, the joined at most `A`
  times the room, and `4·√ε'·A + 2·ε'·A ≤ 1`;
* `m1_slackNums`, `zk_slackFitW`: at C-Flock ZK's schedules a table has at most `7` levels, level 1 a query and level
  0's weight with its `2` under `2^40`, so the joined weights are at most `2^41` times the room and the slack fits at
  chances at most `2^{-128}`, whatever the rounds;
* `zk_slackFitW_δN`: at the chances `FnFork.δN GF128 65536 n` of the coupling at `fnAt`, where every revealed length
  before the last round is positive.
-/

namespace Claims.Flock

section Fit

variable {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
  (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K)

theorem card_IZW (L : ℕ) : Fintype.card (IZW L) = 2 * L + 5 := by
  show Fintype.card (((Fin 2 × Fin L) ⊕ (Fin 2 ⊕ Fin 2)) ⊕ Unit) = _
  simp only [Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, Fintype.card_unit]

/-- **Table `j`'s slack weights total at most its weight times `|IZW|·8ε`**, its forks before the last round, at
chances at most `ε ≤ 1/2` before the last round and at most `1` there (Cauchy–Schwarz, `sum_slackW_le`). -/
theorem sum_slackCW_le {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ε)
    (hδ1 : ∀ t, δ t ≤ 1) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2) (j : ℕ) (hj : ∀ i, (rd j i).val + 1 < K) :
    ∑ t, slackCW sch Np rd δ j t ≤
      (∑ i, cZW (sch j) (Np j) i) * (Fintype.card (IZW (sch j).levels.length) * (8 * ε)) := by
  have hw : ∀ i (t : Fin K), 0 ≤ slackW K δ (rd j i).val t.val := fun i t => slackW_nonneg hδ0 K _ _
  have hc0 : 0 ≤ ∑ i, cZW (sch j) (Np j) i := Finset.sum_nonneg fun i _ => cZW_nonneg _ _ i
  have hC : ∀ t : Fin K, slackCW sch Np rd δ j t ≤
      (∑ i, cZW (sch j) (Np j) i) * ∑ i, slackW K δ (rd j i).val t.val := fun t => by
    unfold slackCW
    rw [Finset.sum_congr rfl fun i _ => Real.sqrt_mul (cZW_nonneg (sch j) (Np j) i) (slackW K δ (rd j i).val t.val)]
    refine (Finset.sum_mul_sq_le_sq_mul_sq _ _ _).trans (le_of_eq ?_)
    congr 1
    · exact Finset.sum_congr rfl fun i _ => Real.sq_sqrt (cZW_nonneg _ _ i)
    · exact Finset.sum_congr rfl fun i _ => Real.sq_sqrt (hw i t)
  calc ∑ t, slackCW sch Np rd δ j t
      ≤ ∑ t : Fin K, (∑ i, cZW (sch j) (Np j) i) * ∑ i, slackW K δ (rd j i).val t.val :=
        Finset.sum_le_sum fun t _ => hC t
    _ = (∑ i, cZW (sch j) (Np j) i) * ∑ i, ∑ t : Fin K, slackW K δ (rd j i).val t.val := by
        rw [← Finset.mul_sum, Finset.sum_comm]
    _ ≤ (∑ i, cZW (sch j) (Np j) i) * ∑ _i : IZW (sch j).levels.length, 8 * ε :=
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => sum_slackW_le hδ0 hδ hδ1 hε0 hε (hj i)) hc0
    _ = (∑ i, cZW (sch j) (Np j) i) * (Fintype.card (IZW (sch j).levels.length) * (8 * ε)) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- **Every table's slack weights total at most the joined weights times `c·8ε`**, `c` a bound on each table's
`|IZW|`. -/
theorem sum_slackTW_le {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ε)
    (hδ1 : ∀ t, δ t ≤ 1) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2) (hj : ∀ j < n, ∀ i, (rd j i).val + 1 < K) {c : ℝ}
    (hc : ∀ j < n, (Fintype.card (IZW (sch j).levels.length) : ℝ) ≤ c) :
    ∑ t, slackTW n sch Np rd δ t ≤ c * (8 * ε) * ∑ t, joinCW n sch Np rd t := by
  unfold slackTW
  rw [Finset.sum_comm, joinCW_sum, Finset.mul_sum]
  refine Finset.sum_le_sum fun j hj' => ?_
  have hjn := Finset.mem_range.1 hj'
  have hc0 : 0 ≤ ∑ i, cZW (sch j) (Np j) i := Finset.sum_nonneg fun i _ => cZW_nonneg _ _ i
  refine (sum_slackCW_le sch Np rd hδ0 hδ hδ1 hε0 hε j (hj j hjn)).trans ?_
  rw [mul_comm (c * (8 * ε))]
  exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right (hc j hjn) (by linarith)) hc0

/-- **The slack fits the room** where its weights are at most `ε'` times the joined ones, the joined weights at most
`A` times the room, and `4·√ε'·A + 2·ε'·A ≤ 1`. -/
theorem slackFitW_of_le (δ : ℕ → ℝ) {ε' A : ℝ} (hε' : 0 ≤ ε')
    (hS : ∑ t, slackTW n sch Np rd δ t ≤ ε' * ∑ t, joinCW n sch Np rd t)
    (hJ : ∑ t, joinCW n sch Np rd t ≤ A * roomW n sch Np) (h : 4 * Real.sqrt ε' * A + 2 * ε' * A ≤ 1) :
    SlackFitW n sch Np rd δ := by
  unfold SlackFitW
  set J := ∑ t, joinCW n sch Np rd t
  set S := ∑ t, slackTW n sch Np rd δ t
  set R := roomW n sch Np
  have hJ0 : 0 ≤ J := Finset.sum_nonneg fun t _ => joinCW_nonneg n sch Np rd t
  have hR0 : 0 ≤ R := roomW_nonneg n sch Np
  have hs0 := Real.sqrt_nonneg ε'
  have hsq : Real.sqrt (J * S) ≤ Real.sqrt ε' * J := by
    calc Real.sqrt (J * S) ≤ Real.sqrt (ε' * J ^ 2) :=
          Real.sqrt_le_sqrt ((mul_le_mul_of_nonneg_left hS hJ0).trans_eq (by ring))
      _ = Real.sqrt ε' * J := by rw [Real.sqrt_mul hε', Real.sqrt_sq hJ0]
  calc 4 * Real.sqrt (J * S) + 2 * S ≤ (4 * Real.sqrt ε' + 2 * ε') * J := by nlinarith
    _ ≤ (4 * Real.sqrt ε' + 2 * ε') * (A * R) := mul_le_mul_of_nonneg_left hJ (by linarith)
    _ = (4 * Real.sqrt ε' * A + 2 * ε' * A) * R := by ring
    _ ≤ 1 * R := mul_le_mul_of_nonneg_right h hR0
    _ = R := one_mul R

/-- **The joined weights are at most `A` times the room** where each table's level 0 with its `2` weighs at most
`A - 1` times a level 1's, and has a level. -/
theorem joinCW_le_room {A : ℝ}
    (h : ∀ j < n, 1 ≤ (sch j).levels.length ∧
      2 + 2 ^ (FlockSoundness.lvOf (sch j) 0).logLen * (2 * (FlockSoundness.lvOf (sch j) 0).queries) ≤
        (A - 1) * cOf (sch j) 1)
    (h1 : 1 ≤ A) : ∑ t, joinCW n sch Np rd t ≤ A * roomW n sch Np := by
  rw [joinCW_sum]
  unfold roomW
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun j hj' => ?_
  obtain ⟨hL, hc⟩ := h j (Finset.mem_range.1 hj')
  have h1c : cOf (sch j) 1 ≤
      (∑ a : Fin 2 × Fin (sch j).levels.length, cOf (sch j) (a.2.val + 1)) + 2 * ((Np j : ℝ) * Np j) := by
    have := Finset.single_le_sum (f := fun a : Fin 2 × Fin (sch j).levels.length => cOf (sch j) (a.2.val + 1))
      (fun _ _ => cOf_nonneg _ _) (Finset.mem_univ ((0 : Fin 2), (⟨0, hL⟩ : Fin (sch j).levels.length)))
    have hN : 0 ≤ 2 * ((Np j : ℝ) * Np j) := by positivity
    simp only [zero_add] at this
    linarith
  have hroom : 0 ≤ (∑ a : Fin 2 × Fin (sch j).levels.length, cOf (sch j) (a.2.val + 1)) +
      2 * ((Np j : ℝ) * Np j) := le_trans (cOf_nonneg _ _) h1c
  rw [cZW_sum]
  nlinarith [mul_le_mul_of_nonneg_left h1c (show 0 ≤ A - 1 by linarith)]

end Fit

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness

/-- **The numbers C-Flock ZK's schedules give the slack's fit**: at `m ∈ [25, 35]` at most `7` levels and at least
one, level 1 a query, and level 0's weight `2^{d₀}·2Q₀` with `2` under `2^40`. -/
theorem m1_slackNums : ∀ m : Fin 11, (Accounting.m1Schedule (m.val + 25)).levels.length ≤ 7 ∧
    1 ≤ (Accounting.m1Schedule (m.val + 25)).levels.length ∧
    1 ≤ 2 ^ (lvOf (Accounting.m1Schedule (m.val + 25)) 1).logLen *
      (lvOf (Accounting.m1Schedule (m.val + 25)) 1).queries ∧
    2 + 2 ^ (lvOf (Accounting.m1Schedule (m.val + 25)) 0).logLen *
      (2 * (lvOf (Accounting.m1Schedule (m.val + 25)) 0).queries) ≤ 2 ^ 40 := by
  decide +kernel

/-- **The slack fits the masked room at C-Flock ZK's schedules at chances at most `2^{-128}`**, every fork before the
last round, whatever the rounds `K`: `|IZW| ≤ 19`, the joined weights at most `2^41` times the room, and
`4·√(19·8·2^{-128})·2^41 + 2·19·8·2^{-128}·2^41 ≤ 1`. -/
theorem zk_slackFitW {K n : ℕ} (st : ℕ → Stmt) (hm : ∀ j < n, 25 ≤ (st j).m ∧ (st j).m ≤ 35)
    (sch : ℕ → FlockSoundness.Accounting.Schedule) (hsch : ∀ j < n, sch j = Accounting.m1Schedule (st j).m) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin K) (hj : ∀ j < n, ∀ i, (rd j i).val + 1 < K)
    {δ : ℕ → ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ((2 : ℝ) ^ 128)⁻¹) (hδ1 : ∀ t, δ t ≤ 1) :
    Claims.Flock.SlackFitW n sch Np rd δ := by
  have hnum : ∀ j < n, (sch j).levels.length ≤ 7 ∧ 1 ≤ (sch j).levels.length ∧
      1 ≤ 2 ^ (lvOf (sch j) 1).logLen * (lvOf (sch j) 1).queries ∧
      2 + 2 ^ (lvOf (sch j) 0).logLen * (2 * (lvOf (sch j) 0).queries) ≤ 2 ^ 40 := fun j hjn => by
    obtain ⟨h25, h35⟩ := hm j hjn
    have he := m1_slackNums ⟨(st j).m - 25, by omega⟩
    simp only [Nat.sub_add_cancel h25] at he
    rw [hsch j hjn]
    exact he
  have hε0 : (0 : ℝ) ≤ ((2 : ℝ) ^ 128)⁻¹ := by positivity
  have hε : ((2 : ℝ) ^ 128)⁻¹ ≤ 1 / 2 := by norm_num
  have hc : ∀ j < n, (Fintype.card (Claims.Flock.IZW (sch j).levels.length) : ℝ) ≤ 19 := fun j hjn => by
    rw [Claims.Flock.card_IZW]
    have := (hnum j hjn).1
    exact_mod_cast (show 2 * (sch j).levels.length + 5 ≤ 19 by omega)
  have hS := Claims.Flock.sum_slackTW_le n sch Np rd hδ0 hδ hδ1 hε0 hε hj hc
  have hJ : ∑ t, Claims.Flock.joinCW n sch Np rd t ≤ 2 ^ 41 * Claims.Flock.roomW n sch Np := by
    refine Claims.Flock.joinCW_le_room n sch Np rd (fun j hjn => ⟨(hnum j hjn).2.1, ?_⟩) (by norm_num)
    obtain ⟨-, -, h1, h0⟩ := hnum j hjn
    have h1' : (1 : ℝ) ≤ Claims.Flock.cOf (sch j) 1 := by
      unfold Claims.Flock.cOf
      exact_mod_cast h1
    have h0' : (2 : ℝ) + 2 ^ (lvOf (sch j) 0).logLen * (2 * (lvOf (sch j) 0).queries) ≤ 2 ^ 40 := by
      exact_mod_cast h0
    nlinarith
  have hsq : Real.sqrt (19 * (8 * ((2 : ℝ) ^ 128)⁻¹)) ≤ ((2 : ℝ) ^ 60)⁻¹ :=
    (Real.sqrt_le_sqrt (by norm_num : 19 * (8 * ((2 : ℝ) ^ 128)⁻¹) ≤ (((2 : ℝ) ^ 60)⁻¹) ^ 2)).trans_eq
      (Real.sqrt_sq (by positivity))
  refine Claims.Flock.slackFitW_of_le n sch Np rd δ (by positivity) hS hJ ?_
  have h2 : 2 * (19 * (8 * ((2 : ℝ) ^ 128)⁻¹)) * 2 ^ 41 ≤ 1 / 2 := by norm_num
  have h4 : 4 * ((2 : ℝ) ^ 60)⁻¹ * 2 ^ 41 ≤ 1 / 2 := by norm_num
  nlinarith [Real.sqrt_nonneg (19 * (8 * ((2 : ℝ) ^ 128)⁻¹))]

/-- **The coupling's chance at GF128's coins at a round whose revealed length is positive** is at most
`2^{-128}`. -/
theorem δN_GF128_le {K : ℕ} {n : Fin K → ℕ} {t : ℕ} (hn : ∀ ht : t < K, 1 ≤ n ⟨t, ht⟩) :
    Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n t ≤ ((2 : ℝ) ^ 128)⁻¹ := by
  have := Claims.Flock.δN_le_inv (A := FlockLevel3.GF128) (M := 65536) (by norm_num) hn
  rwa [FlockLevel3.card_F, Nat.cast_pow, Nat.cast_ofNat] at this

/-- **The slack fits the masked room at the coupling's chances** `FnFork.δN GF128 65536 n`, at C-Flock ZK's schedules,
every fork before the last round, where every revealed length before the last round is positive. -/
theorem zk_slackFitW_δN {K n : ℕ} (st : ℕ → Stmt) (hm : ∀ j < n, 25 ≤ (st j).m ∧ (st j).m ≤ 35)
    (sch : ℕ → FlockSoundness.Accounting.Schedule) (hsch : ∀ j < n, sch j = Accounting.m1Schedule (st j).m) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin K) (hj : ∀ j < n, ∀ i, (rd j i).val + 1 < K)
    (c : Fin K → ℕ) (hc : ∀ t : Fin K, t.val + 1 < K → 1 ≤ c t) :
    Claims.Flock.SlackFitW n sch Np rd (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 c) :=
  zk_slackFitW st hm sch hsch Np rd hj (Claims.Flock.FnFork.δN_nonneg c)
    (fun t ht => δN_GF128_le fun h => hc ⟨t, h⟩ ht) (Claims.Flock.δN_le_one c)

end FlockDeploy.KS
