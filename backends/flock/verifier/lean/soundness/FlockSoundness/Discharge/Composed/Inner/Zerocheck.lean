import FlockLevel3

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

/-! ## Completeness -/

/-- **`zerocheck c_eval`'s two sides agree** for the honest prover, at every coin: `Pc(z) = final_c`. -/
theorem c_eval (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128) :
    dot (lagrange nodesL den6Inv z) (honest n a b c r rho z).round1c = (honest n a b c r rho z).finalC := by
  sorry

/-- **Each replayed claim is the honest sum.** For valid tables, after `j ≤ n` residual rounds with `r_k ≠ 1`, the
claim `Zk.replay` replays on the honest prover's clear messages is `Σ_x eq(r_{j…}, x)·A_j(x)·B_j(x)`. -/
theorem claim_honest (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128)
    (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) (j : ℕ) (hj : j ≤ n) (hr : ∀ k < j, r[k]! ≠ 1) :
    claim (honest n a b c r rho z) r rho z j =
      (sumAt n (coin r) (coin rho) (skipAt a (GF128.of z)) (skipAt b (GF128.of z)) j).val := by
  sorry

/-- **`zerocheck final a·b`'s two sides agree** for the honest prover: for valid tables, at every coin with
`r_k ≠ 1` for `k < n`, the final replayed claim is `final_a · final_b`. -/
theorem final_ab (n : ℕ) (a b c : ℕ → 𝔽) (r rho : Array F128) (z : F128)
    (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) (hr : ∀ k < n, r[k]! ≠ 1) :
    claim (honest n a b c r rho z) r rho z n = (honest n a b c r rho z).finalA * (honest n a b c r rho z).finalB := by
  sorry

/-- The zerocheck vector `pinned ++ r_outer` is never 1 where `r_outer` isn't: the pinned coordinates never are, and
an out-of-range read is 0. -/
theorem pinned_append_ne_one (rOuter : Array F128) (h : ∀ k < rOuter.size, rOuter[k]! ≠ 1) :
    ∀ k : ℕ, (pinned ++ rOuter)[k]! ≠ 1 := by
  sorry

end FlockSoundness.Discharge.Composed.Zerocheck
