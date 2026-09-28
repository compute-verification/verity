import Proofs.Flock.Level3

/-!
# The clear zerocheck is complete at every coin with `r_j ≠ 1`

`Zk.replay` replays the zerocheck on the masked values; at the honest pads every form it builds evaluates to the clear
value, so the zerocheck's two constraints (`zerocheck c_eval` and `zerocheck final a·b`) hold for the honest prover
exactly when the clear zerocheck's checks hold on its clear messages. This file proves the clear checks, in the
executable's own arithmetic (`F128`, `dot`, `lagrange nodesL den6Inv`, `combWeights`, `inv`):

* `claim`: the running claim after `j` residual rounds, as `Zk.replay` forms it on clear values (`claim0`, `step`).
* `honest`: the honest prover's clear zerocheck messages for tables `a`, `b`, `c` (the A-side, B-side and witness bits at
  address `s + 64·x`), the zerocheck vector `r`, the residual coins `rho` and the skip coin `z`. Round `j` reads only
  `rho[0…j−1]`.
* `c_eval`: `Pc(z) = final_c`, the `zerocheck c_eval` constraint's two sides, at every coin.
* `claim_honest`: for valid tables (`a·b = c`), after `j ≤ n` rounds whose `r_k ≠ 1` the replayed claim is the honest
  sum `Σ_x eq(r_{j…}, x)·A_j(x)·B_j(x)` (`sumAt`), `A_j`, `B_j` the skip rows at `z` folded at `rho[0…j−1]`.
* `final_ab`: the final claim is `final_a · final_b`, the `zerocheck final a·b` constraint's two sides.

`r_j = 1` is excluded: there `G(0) = (claim + r_j·g₁)·(1 + r_j)⁻¹` reads `inv 0 = 0`, and the replayed claim leaves the
honest one. The seven pinned coordinates are never 1 (`pinned_append_ne_one`).
-/

namespace FlockSoundness.Discharge.Composed.Zerocheck

open Flock FlockLevel3 Polynomial

local notation "𝔽" => FlockLevel3.GF128

/-! ## The clear replay -/

/-- The starting claim `Pcomb(z) + Pc(z)` (§9.2), as `Zk.replay` forms it on the clear values. -/
def claim0 (zc : ZerocheckProof) (z : F128) : F128 :=
  dot (combWeights z) ((Array.range 64).map fun i => zc.round1ab[i]! + zc.round1c[i]!) +
    dot (lagrange nodesL den6Inv z) zc.round1c

/-- A residual round's update, as `Zk.replay` forms it: `G(0) = (claim + g₁·r_eq)·(1 + r_eq)⁻¹`, then
`G(0)·(1 + ρ) + g₁·ρ + g_∞·(ρ·(1 + ρ))`. -/
def step (req rr cl g1 ginf : F128) : F128 :=
  let g0 := (cl + g1 * req) * (1 + req).inv
  g0 * (1 + rr) + g1 * rr + ginf * (rr * (1 + rr))

/-- The running claim after `j` residual rounds: round `j` reads `r_eq = r[j]`, the coin `rho[j]` and the message
`rounds[j]`. -/
def claim (zc : ZerocheckProof) (r rho : Array F128) (z : F128) : ℕ → F128
  | 0 => claim0 zc z
  | j + 1 => step r[j]! rho[j]! (claim zc r rho z j) zc.rounds[j]!.1 zc.rounds[j]!.2

/-! ## The honest prover -/

/-- `eq(r_{k…k+len−1}, x)` over the low `len` bits of `x`. -/
noncomputable def eqR (r : ℕ → 𝔽) (k len x : ℕ) : 𝔽 :=
  ∏ j ∈ Finset.range len, if x.testBit j then r (k + j) else 1 + r (k + j)

/-- A table's skip row at `Y = ζ`: `t̂(ζ, x) = Σ_s L_s(ζ)·t[s + 64·x]`. -/
noncomputable def skipAt (t : ℕ → 𝔽) (ζ : 𝔽) (x : ℕ) : 𝔽 :=
  ∑ s : Fin 64, lagS ζ s * t (s.val + 64 * x)

/-- The output-coset node `φ8(64 + i)`. -/
def nodeL (i : ℕ) : 𝔽 := GF128.of (phi8 (64 + i))

/-- Bind the lowest variable at `t`. -/
noncomputable def foldLo (t : 𝔽) (u : ℕ → 𝔽) : ℕ → 𝔽 := fun y => u (2 * y) + t * (u (2 * y + 1) + u (2 * y))

/-- Bind the lowest `j` variables at `ρ_0 … ρ_{j−1}`, lowest first. -/
noncomputable def foldN (ρ : ℕ → 𝔽) : ℕ → (ℕ → 𝔽) → ℕ → 𝔽
  | 0, u => u
  | j + 1, u => foldLo (ρ j) (foldN ρ j u)

/-- The honest claim after `j` of `n` rounds: `Σ_{x < 2^{n−j}} eq(r_{j…n−1}, x)·U_j(x)·V_j(x)`. -/
noncomputable def sumAt (n : ℕ) (r ρ u v : ℕ → 𝔽) (j : ℕ) : 𝔽 :=
  ∑ x ∈ Finset.range (2 ^ (n - j)), eqR r j (n - j) x * foldN ρ j u x * foldN ρ j v x

/-- Round `j`'s honest message `(G(1), G(∞))` of `G(t) = Σ_y eq(r_{j+1…}, y)·U_j(2y + t)·V_j(2y + t)` (multilinear in
`t`): `G(1)` and the coefficient of `t²`. -/
noncomputable def roundMsg (n : ℕ) (r ρ u v : ℕ → 𝔽) (j : ℕ) : 𝔽 × 𝔽 :=
  (∑ y ∈ Finset.range (2 ^ (n - j - 1)), eqR r (j + 1) (n - j - 1) y * foldN ρ j u (2 * y + 1) * foldN ρ j v (2 * y + 1),
   ∑ y ∈ Finset.range (2 ^ (n - j - 1)), eqR r (j + 1) (n - j - 1) y *
     (foldN ρ j u (2 * y + 1) + foldN ρ j u (2 * y)) * (foldN ρ j v (2 * y + 1) + foldN ρ j v (2 * y)))

/-- A coin array as field elements. -/
def coin (r : Array F128) (k : ℕ) : 𝔽 := GF128.of r[k]!

/-- **The honest prover's clear zerocheck messages** over `n = m − 6` residual variables, for the tables `a`, `b`, `c`
(value at address `s + 64·x`), the zerocheck vector `r`, the residual coins `rho` and the skip coin `z`:
`round1_ab[i] = Σ_x eq(r, x)·â(λ_i, x)·b̂(λ_i, x)`, `round1_c[i] = Σ_x eq(r, x)·ĉ(λ_i, x)` at `λ_i = φ8(64 + i)`, the
residual rounds of `Σ_x eq(r, x)·â(z, x)·b̂(z, x)`, `final_a`, `final_b` the skip rows at `z` folded at `rho`, and
`final_c = Σ_x eq(r, x)·ĉ(z, x)`. -/
noncomputable def honest (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128) : ZerocheckProof where
  round1ab := (Array.range 64).map fun i =>
    (∑ x ∈ Finset.range (2 ^ n), eqR (coin r) 0 n x * skipAt a (nodeL i) x * skipAt b (nodeL i) x).val
  round1c := (Array.range 64).map fun i =>
    (∑ x ∈ Finset.range (2 ^ n), eqR (coin r) 0 n x * skipAt c (nodeL i) x).val
  rounds := (Array.range n).map fun j =>
    let g := roundMsg n (coin r) (coin rho) (skipAt a (GF128.of z)) (skipAt b (GF128.of z)) j
    (g.1.val, g.2.val)
  finalA := (foldN (coin rho) n (skipAt a (GF128.of z)) 0).val
  finalB := (foldN (coin rho) n (skipAt b (GF128.of z)) 0).val
  finalC := (∑ x ∈ Finset.range (2 ^ n), eqR (coin r) 0 n x * skipAt c (GF128.of z) x).val

/-! ## The executable's arithmetic as the field's -/

theorem of_val (x : 𝔽) : GF128.of x.val = x := rfl

theorem of_one' : GF128.of 1 = 1 := GF128.one_eq.symm

theorem eq_of_of {x y : F128} (h : GF128.of x = GF128.of y) : x = y := h

theorem ne_one_of {x : F128} (h : x ≠ 1) : GF128.of x ≠ 1 := fun h' => h (eq_of_of (h'.trans of_one'.symm))

theorem two_eq_zero' : (2 : 𝔽) = 0 := CharTwo.two_eq_zero

theorem mul_sum' (R : 𝔽) (s : Finset ℕ) (f : ℕ → 𝔽) : R * ∑ y ∈ s, f y = ∑ y ∈ s, R * f y := Finset.mul_sum _ _ _

theorem sum_mul' (R : 𝔽) (s : Finset ℕ) (f : ℕ → 𝔽) : (∑ y ∈ s, f y) * R = ∑ y ∈ s, f y * R := Finset.sum_mul _ _ _

theorem dot_eq_foldl (a b : Array F128) :
    dot a b = (List.range (min a.size b.size)).foldl (fun acc i => acc + a[i]! * b[i]!) 0 := by
  unfold dot
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  simp [Std.Legacy.Range.size, List.range_eq_range']
  rfl

theorem of_foldl_dot (a b : Array F128) :
    ∀ (k : ℕ) (acc : F128), GF128.of ((List.range k).foldl (fun acc i => acc + a[i]! * b[i]!) acc) =
      GF128.of acc + ∑ i ∈ Finset.range k, GF128.of a[i]! * GF128.of b[i]! := by
  intro k
  induction k with
  | zero => intro acc; simp
  | succ k ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, of_add, ih, of_mul,
        Finset.sum_range_succ, add_assoc]

/-- `dot` is a sum over the first array's entries. -/
theorem of_dot (a b : Array F128) (n : ℕ) (ha : a.size = n) :
    GF128.of (dot a b) = ∑ i : Fin n, GF128.of a[i.val]! * GF128.of b[i.val]! := by
  rw [dot_eq_foldl, of_foldl_dot, ← GF128.zero_eq, zero_add,
    Fin.sum_univ_eq_sum_range (fun i => GF128.of a[i]! * GF128.of b[i]!) n, ha]
  by_cases hb : n ≤ b.size
  · rw [min_eq_left hb]
  · rw [min_eq_right (by omega)]
    refine Finset.sum_subset (Finset.range_mono (by omega)) fun i hi hni => ?_
    simp only [Finset.mem_range, not_lt] at hi hni
    rw [getElem!_neg b i (by omega)]
    show GF128.of a[i]! * GF128.of 0 = 0
    rw [← GF128.zero_eq, mul_zero]

theorem lagrange_size (nodes : Array F128) (d z : F128) : (lagrange nodes d z).size = nodes.size := by
  unfold lagrange
  split <;> simp

theorem nodesL_size : nodesL.size = 64 := by simp [nodesL]

theorem combWeights_size (z : F128) : (combWeights z).size = 64 := by
  unfold combWeights
  split <;> simp

theorem get_map_range {α : Type} [Inhabited α] (f : ℕ → α) {n i : ℕ} (hi : i < n) :
    ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using hi)]
  simp

/-! ## Rows as polynomials -/

/-- Row `x` of a table as the polynomial of degree below 64 through `t[s + 64·x]` on the skip nodes. -/
noncomputable def rowPoly (t : ℕ → 𝔽) (x : ℕ) : 𝔽[X] :=
  Lagrange.interpolate Finset.univ nodeS fun s => t (s.val + 64 * x)

theorem rowPoly_natDegree (t : ℕ → 𝔽) (x : ℕ) : (rowPoly t x).natDegree < 64 := by
  by_cases h : rowPoly t x = 0
  · rw [h, natDegree_zero]; norm_num
  · have hdeg := Lagrange.degree_interpolate_lt (s := Finset.univ) (r := fun s : Fin 64 => t (s.val + 64 * x))
      nodeS_injective.injOn
    rw [Finset.card_univ, Fintype.card_fin] at hdeg
    exact (natDegree_lt_iff_degree_lt h).2 hdeg

theorem rowPoly_node (t : ℕ → 𝔽) (x : ℕ) (s : Fin 64) : (rowPoly t x).eval (nodeS s) = t (s.val + 64 * x) :=
  Lagrange.eval_interpolate_at_node _ nodeS_injective.injOn (Finset.mem_univ s)

theorem skipAt_eq (t : ℕ → 𝔽) (ζ : 𝔽) (x : ℕ) : skipAt t ζ x = (rowPoly t x).eval ζ := by
  rw [← lagS_interp _ (rowPoly_natDegree t x) ζ]
  simp only [skipAt, rowPoly_node]

/-- `P_c(Y) = Σ_x eq(r, x)·ĉ_x(Y)`. -/
noncomputable def pc (n : ℕ) (r c : ℕ → 𝔽) : 𝔽[X] :=
  ∑ x ∈ Finset.range (2 ^ n), C (eqR r 0 n x) * rowPoly c x

/-- `P_ab(Y) = Σ_x eq(r, x)·â_x(Y)·b̂_x(Y)`. -/
noncomputable def pab (n : ℕ) (r a b : ℕ → 𝔽) : 𝔽[X] :=
  ∑ x ∈ Finset.range (2 ^ n), C (eqR r 0 n x) * (rowPoly a x * rowPoly b x)

theorem pc_eval (n : ℕ) (r c : ℕ → 𝔽) (ζ : 𝔽) :
    (pc n r c).eval ζ = ∑ x ∈ Finset.range (2 ^ n), eqR r 0 n x * skipAt c ζ x := by
  simp only [pc, eval_finsetSum, eval_mul, eval_C, skipAt_eq]

theorem pab_eval (n : ℕ) (r a b : ℕ → 𝔽) (ζ : 𝔽) :
    (pab n r a b).eval ζ = ∑ x ∈ Finset.range (2 ^ n), eqR r 0 n x * skipAt a ζ x * skipAt b ζ x := by
  simp only [pab, eval_finsetSum, eval_mul, eval_C, skipAt_eq, mul_assoc]

theorem pc_natDegree (n : ℕ) (r c : ℕ → 𝔽) : (pc n r c).natDegree < 64 := by
  have : (pc n r c).natDegree ≤ 63 := by
    unfold pc
    refine natDegree_sum_le_of_forall_le _ _ fun x _ => (natDegree_C_mul_le _ _).trans ?_
    have := rowPoly_natDegree c x
    omega
  omega

theorem pab_add_pc_natDegree (n : ℕ) (r a b c : ℕ → 𝔽) : (pab n r a b + pc n r c).natDegree < 128 := by
  have h1 : (pab n r a b).natDegree ≤ 126 := by
    unfold pab
    refine natDegree_sum_le_of_forall_le _ _ fun x _ => (natDegree_C_mul_le _ _).trans (natDegree_mul_le.trans ?_)
    have := rowPoly_natDegree a x
    have := rowPoly_natDegree b x
    omega
  have := natDegree_add_le (pab n r a b) (pc n r c)
  have := pc_natDegree n r c
  omega

/-- On valid tables `P_ab + P_c` vanishes on the skip nodes (characteristic 2). -/
theorem pab_add_pc_nodeS (n : ℕ) (r a b c : ℕ → 𝔽) (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) (s : Fin 64) :
    (pab n r a b + pc n r c).eval (nodeS s) = 0 := by
  simp only [pab, pc, eval_add, eval_finsetSum, eval_mul, eval_C, rowPoly_node, ← Finset.sum_add_distrib]
  refine Finset.sum_eq_zero fun x hx => ?_
  have hx' : x < 2 ^ n := Finset.mem_range.1 hx
  have hpow : 2 ^ (n + 6) = 2 ^ n * 64 := by rw [pow_add]; norm_num
  have hi : s.val + 64 * x < 2 ^ (n + 6) := by
    rw [hpow]; generalize 2 ^ n = N at hx'; have := s.isLt; omega
  rw [hvalid _ hi, ← mul_add, CharTwo.add_self_eq_zero, mul_zero]

/-! ## The residual rounds -/

theorem eqR_even (r : ℕ → 𝔽) (k L y : ℕ) : eqR r k (L + 1) (2 * y) = (1 + r k) * eqR r (k + 1) L y := by
  unfold eqR
  rw [Finset.prod_range_succ', mul_comm]
  congr 1
  · simp [Nat.testBit_zero]
  · refine Finset.prod_congr rfl fun j _ => ?_
    have htb : (2 * y).testBit (j + 1) = y.testBit j := by rw [Nat.testBit_succ]; congr 1; omega
    rw [htb, show k + (j + 1) = k + 1 + j by omega]

theorem eqR_odd (r : ℕ → 𝔽) (k L y : ℕ) : eqR r k (L + 1) (2 * y + 1) = r k * eqR r (k + 1) L y := by
  unfold eqR
  rw [Finset.prod_range_succ', mul_comm]
  congr 1
  · simp [Nat.testBit_zero, Nat.add_mod]
  · refine Finset.prod_congr rfl fun j _ => ?_
    have htb : (2 * y + 1).testBit (j + 1) = y.testBit j := by rw [Nat.testBit_succ]; congr 1; omega
    rw [htb, show k + (j + 1) = k + 1 + j by omega]

/-- Sums over `2^(n+1)` indices by pairs `(2y, 2y+1)`. -/
theorem sum_range_two_mul {M : Type} [AddCommMonoid M] (f : ℕ → M) :
    ∀ N : ℕ, ∑ x ∈ Finset.range (2 * N), f x = ∑ y ∈ Finset.range N, (f (2 * y) + f (2 * y + 1))
  | 0 => by simp
  | N + 1 => by
      rw [show 2 * (N + 1) = 2 * N + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
        sum_range_two_mul f N, Finset.sum_range_succ, add_assoc]

theorem sum_range_pairs {M : Type} [AddCommMonoid M] (n : ℕ) (f : ℕ → M) :
    ∑ x ∈ Finset.range (2 ^ (n + 1)), f x = ∑ y ∈ Finset.range (2 ^ n), (f (2 * y) + f (2 * y + 1)) := by
  rw [pow_succ, mul_comm]
  exact sum_range_two_mul f _

/-- One residual round of the honest prover: from the claim `sumAt … j`, with `r_j ≠ 1`, the replay's update on the
round's honest message is `sumAt … (j + 1)`. -/
theorem step_honest (n j : ℕ) (hj : j < n) (r rho : Array F128) (u v : ℕ → 𝔽) (hr : r[j]! ≠ 1) (cl : F128)
    (hcl : GF128.of cl = sumAt n (coin r) (coin rho) u v j) :
    GF128.of (step r[j]! rho[j]! cl (roundMsg n (coin r) (coin rho) u v j).1.val
      (roundMsg n (coin r) (coin rho) u v j).2.val) = sumAt n (coin r) (coin rho) u v (j + 1) := by
  obtain ⟨L, hL⟩ : ∃ L, n - j = L + 1 := ⟨n - j - 1, by omega⟩
  have hL' : n - j - 1 = L := by omega
  have hL'' : n - (j + 1) = L := by omega
  set R := coin r j
  set P := coin rho j
  have hR1 : R ≠ 1 := ne_one_of hr
  have h1R : 1 + R ≠ 0 := fun h => hR1 (by linear_combination h - two_eq_zero')
  set U := foldN (coin rho) j u
  set V := foldN (coin rho) j v
  set e := fun y => eqR (coin r) (j + 1) L y
  have hsplit : sumAt n (coin r) (coin rho) u v j =
      (1 + R) * ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y) * V (2 * y) +
        R * ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y + 1) * V (2 * y + 1) := by
    simp only [sumAt]
    rw [hL, sum_range_pairs, mul_sum', mul_sum', ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [eqR_even, eqR_odd]
    ring
  have hnext : sumAt n (coin r) (coin rho) u v (j + 1) =
      ∑ y ∈ Finset.range (2 ^ L), e y * (U (2 * y) + P * (U (2 * y + 1) + U (2 * y))) *
        (V (2 * y) + P * (V (2 * y + 1) + V (2 * y))) := by
    simp only [sumAt, hL'', foldN, foldLo]
    rfl
  have hmsg : roundMsg n (coin r) (coin rho) u v j =
      (∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y + 1) * V (2 * y + 1),
       ∑ y ∈ Finset.range (2 ^ L), e y * (U (2 * y + 1) + U (2 * y)) * (V (2 * y + 1) + V (2 * y))) := by
    simp only [roundMsg, hL']
    rfl
  rw [hmsg]
  simp only [step, of_add, of_mul, of_inv, of_one', of_val, hcl, hsplit, hnext]
  rw [show GF128.of r[j]! = R from rfl, show GF128.of rho[j]! = P from rfl]
  have hg0 : ((1 + R) * ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y) * V (2 * y) +
        R * ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y + 1) * V (2 * y + 1) +
        (∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y + 1) * V (2 * y + 1)) * R) * (1 + R)⁻¹ =
      ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y) * V (2 * y) := by
    rw [show (1 + R) * ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y) * V (2 * y) +
        R * ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y + 1) * V (2 * y + 1) +
        (∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y + 1) * V (2 * y + 1)) * R =
        (∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y) * V (2 * y)) * (1 + R) by
      linear_combination (R * ∑ y ∈ Finset.range (2 ^ L), e y * U (2 * y + 1) * V (2 * y + 1)) * two_eq_zero']
    rw [mul_assoc, mul_inv_cancel₀ h1R, mul_one]
  rw [hg0, sum_mul', sum_mul', sum_mul', ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  linear_combination (e y * P * U (2 * y + 1) * V (2 * y + 1)) * two_eq_zero'

/-- The replay's starting claim on the honest messages is `P_ab(z)`, for valid tables. -/
theorem claim0_honest (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128)
    (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) :
    GF128.of (claim0 (honest n a b c r rho z) z) = (pab n (coin r) a b).eval (GF128.of z) := by
  have hc : GF128.of (dot (lagrange nodesL den6Inv z) (honest n a b c r rho z).round1c) =
      (pc n (coin r) c).eval (GF128.of z) := by
    rw [of_dot _ _ 64 (by rw [lagrange_size, nodesL_size])]
    rw [← lagΛ_interp _ (pc_natDegree n _ c) (GF128.of z)]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [honest, get_map_range _ i.isLt, of_val, pc_eval]
    rfl
  have hab : GF128.of (dot (combWeights z) ((Array.range 64).map fun i =>
      (honest n a b c r rho z).round1ab[i]! + (honest n a b c r rho z).round1c[i]!)) =
      (pab n (coin r) a b + pc n (coin r) c).eval (GF128.of z) := by
    rw [of_dot _ _ 64 (combWeights_size z)]
    rw [← combW_interp _ (pab_add_pc_natDegree n _ a b c) (pab_add_pc_nodeS n _ a b c hvalid) (GF128.of z)]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [honest, get_map_range _ i.isLt, of_add, of_val, eval_add, pab_eval, pc_eval]
    rfl
  rw [claim0, of_add, hab, hc, eval_add, add_assoc, CharTwo.add_self_eq_zero, add_zero]

/-! ## Completeness -/

/-- **`zerocheck c_eval`'s two sides agree** for the honest prover, at every coin: `Pc(z) = final_c`. -/
theorem c_eval (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128) :
    dot (lagrange nodesL den6Inv z) (honest n a b c r rho z).round1c = (honest n a b c r rho z).finalC := by
  apply eq_of_of
  rw [of_dot _ _ 64 (by rw [lagrange_size, nodesL_size])]
  have h := lagΛ_interp _ (pc_natDegree n (coin r) c) (GF128.of z)
  rw [pc_eval] at h
  rw [← of_val (∑ x ∈ Finset.range (2 ^ n), eqR (coin r) 0 n x * skipAt c (GF128.of z) x)] at h
  refine Eq.trans ?_ h
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [honest, get_map_range _ i.isLt, of_val, pc_eval]
  rfl

/-- **Each replayed claim is the honest sum.** For valid tables, after `j ≤ n` residual rounds with `r_k ≠ 1`, the
claim `Zk.replay` replays on the honest prover's clear messages is `Σ_x eq(r_{j…}, x)·A_j(x)·B_j(x)`. -/
theorem claim_honest (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128)
    (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) (j : ℕ) (hj : j ≤ n) (hr : ∀ k < j, r[k]! ≠ 1) :
    claim (honest n a b c r rho z) r rho z j =
      (sumAt n (coin r) (coin rho) (skipAt a (GF128.of z)) (skipAt b (GF128.of z)) j).val := by
  apply eq_of_of
  rw [of_val]
  induction j with
  | zero =>
      rw [claim, claim0_honest n a b c r rho z hvalid, pab_eval]
      simp only [sumAt, Nat.sub_zero, foldN]
  | succ j ih =>
      have hjn : j < n := by omega
      have hround : (honest n a b c r rho z).rounds[j]! =
          ((roundMsg n (coin r) (coin rho) (skipAt a (GF128.of z)) (skipAt b (GF128.of z)) j).1.val,
           (roundMsg n (coin r) (coin rho) (skipAt a (GF128.of z)) (skipAt b (GF128.of z)) j).2.val) := by
        simp only [honest]
        rw [get_map_range _ hjn]
      rw [claim, hround]
      exact step_honest n j hjn r rho _ _ (hr j (by omega)) _ (ih (by omega) fun k hk => hr k (by omega))

/-- **`zerocheck final a·b`'s two sides agree** for the honest prover: for valid tables, at every coin with
`r_k ≠ 1` for `k < n`, the final replayed claim is `final_a · final_b`. -/
theorem final_ab (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128)
    (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) (hr : ∀ k < n, r[k]! ≠ 1) :
    claim (honest n a b c r rho z) r rho z n = (honest n a b c r rho z).finalA * (honest n a b c r rho z).finalB := by
  rw [claim_honest n a b c r rho z hvalid n le_rfl hr]
  apply eq_of_of
  rw [of_val, of_mul]
  simp only [sumAt, Nat.sub_self, pow_zero, Finset.sum_range_one, eqR, Finset.prod_range_zero, one_mul]
  rfl

/-- The zerocheck vector `pinned ++ r_outer` is never 1 where `r_outer` isn't: the pinned coordinates never are, and
an out-of-range read is 0. -/
theorem pinned_append_ne_one (rOuter : Array F128) (h : ∀ k < rOuter.size, rOuter[k]! ≠ 1) :
    ∀ k : ℕ, (pinned ++ rOuter)[k]! ≠ 1 := by
  intro k hk
  have hp : pinned.size = 7 := pinned_size
  by_cases hs : k < (pinned ++ rOuter).size
  · rw [getElem!_pos _ k hs, Array.getElem_append] at hk
    by_cases h7 : k < pinned.size
    · simp only [h7, dite_true] at hk
      apply pinned_ne_one ⟨k, by omega⟩
      unfold pinnedF
      rw [getElem!_pos pinned k h7, hk]
      exact of_one'
    · simp only [h7, dite_false] at hk
      rw [Array.size_append] at hs
      apply h (k - pinned.size) (by omega)
      rw [getElem!_pos rOuter _ (by omega)]
      exact hk
  · rw [getElem!_neg _ k hs] at hk
    exact absurd hk (by decide)

end FlockSoundness.Discharge.Composed.Zerocheck
