import Proofs.Flock.Soundness.ZK.SHVZK
import Proofs.Flock.Soundness.ZK.Masking
import Proofs.Flock.Soundness.Assumptions
import Definitions.Flock.LigeritoPad

/-!
# Zero knowledge: the reference prover is complete, and C-Flock ZK's level-0 padding is onto

What masking adds to a table keeps the verifier accepting:

* **`padColumn_honest`, against the model verifier.** `Model.ligeritoPad` checks rep `r`'s level-0 rows through
  `Model.padColumn`: the witness lanes folded by the lane-round coins, the extra lanes weighted by `ρ`, and the padding
  correction for `ȳ`. On the honest padded table, that column is the codeword of the restarted table `y₁'`, at every
  position: the padding cancels (characteristic 2), so level 0's check sees what the unpadded Ligerito would on `y₁'`.
  This is the identity (★) of §4.3 at every position, in the model's own terms.
* **`inner_complete`, the masked verifier's inner check.** On the reference prover's output, `⟨λ, Y₁⟩ = θ + β·τ`, and at
  every opened pads position `Code(Y)_k = C_h[k] + β·C_μ[k]`; every opened leaf is the leaf of its opened content.
* **`padOnto_M1`.** Lemma B's row (ii) for C-Flock ZK's code (`zk::lane_codeword`): at distinct opened positions, no
  more than `t_pad` of them, a padding `(X_L + κ)·μ` over the first `t_pad` novel-basis polynomials reaches every vector
  of values, given `X_L + κ ≠ 0` there (`Assumptions.PadNonvanishing`, §1.2).
* **`padsOnto_monomial`.** Row (viii) for the pads code: at distinct nonzero points, no more than the padding's length,
  the padding `x^s·Σ_j pad_j·x^j` reaches every vector of values.

What the clear protocol needs for completeness (the zerocheck, lincheck and ring-switch rounds, Ligerito from `y₁'` on) is
the unmasked protocol's own completeness, which masking doesn't change and this file doesn't prove.
-/

namespace FlockSoundness.ZK

open Finset Model Polynomial

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Level 0 against `Model.padColumn` -/

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- Folding columns and then encoding is encoding the folded messages. -/
theorem sum_embed_swap {α : Type} (s : Finset α) {n : ℕ} (A : Arith F K) (a : α → K) (M : α → Fin n → F)
    (b : Fin n → F) :
    ∑ j, (∑ ℓ ∈ s, a ℓ * A.embed (M ℓ j)) * A.embed (b j) = ∑ ℓ ∈ s, a ℓ * A.embed (∑ j, M ℓ j * b j) := by
  simp only [Finset.sum_mul, map_sum, map_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun ℓ _ => Finset.sum_congr rfl fun j _ => by ring

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- **Level 0's padded column is the codeword of the restarted table** (§4.3 (★), completeness of C-Flock ZK's level-0
row check against `Model.padColumn`). Lane `ℓ < 2^|ch|` of the committed table holds `Enc(P_ℓ) + Pad(μ_ℓ)` at position
`p`, and extra lane `xl i` holds `Enc(R_i) + Pad(μR_i)`, where `Pad(μ)(p) = Σ_j μ_j·(c₀·X̂_j + X̂_{2^cols+j})(ω_p)` is
C-Flock ZK's padding and the code has `d = cols + 1` bits. With `ȳ = Σ_ℓ eq(ch, ℓ)·μ_ℓ + Σ_i ρ_i·μR_i`, the verifier's
column is `Σ_c y₁'[c]·X̂_c(ω_p)`, `y₁' = Σ_ℓ eq(ch, ℓ)·P_ℓ + Σ_i ρ_i·R_i`. -/
theorem padColumn_honest [CharP K 2] (A : Arith F K) (cols t : ℕ) (c₀ : F) (ch : List K) {e : ℕ} (ρ : Fin e → K)
    (xl : Fin e → ℕ) (C : Oracle F) (P : ℕ → Fin (2 ^ cols) → F) (μ : ℕ → Fin t → F) (R : Fin e → Fin (2 ^ cols) → F)
    (μR : Fin e → Fin t → F) (p : ℕ)
    (hC : ∀ ℓ < 2 ^ ch.length, C ℓ p = ∑ c, P ℓ c * A.Xhat c.val (cols + 1) (A.omega p)
      + ∑ j, μ ℓ j * (c₀ * A.Xhat j.val (cols + 1) (A.omega p) + A.Xhat (2 ^ cols + j.val) (cols + 1) (A.omega p)))
    (hX : ∀ i, C (xl i) p = ∑ c, R i c * A.Xhat c.val (cols + 1) (A.omega p)
      + ∑ j, μR i j * (c₀ * A.Xhat j.val (cols + 1) (A.omega p) + A.Xhat (2 ^ cols + j.val) (cols + 1) (A.omega p))) :
    padColumn A (cols + 1) cols t c₀ ch ρ xl
        (fun j => ∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (μ ℓ j) + ∑ i, ρ i * A.embed (μR i j)) C p
      = ∑ c, (∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (P ℓ c) + ∑ i, ρ i * A.embed (R i c))
          * A.embed (A.Xhat c.val (cols + 1) (A.omega p)) := by
  set Xc : Fin (2 ^ cols) → F := fun c => A.Xhat c.val (cols + 1) (A.omega p)
  set Gc : Fin t → F := fun j => c₀ * A.Xhat j.val (cols + 1) (A.omega p) + A.Xhat (2 ^ cols + j.val) (cols + 1) (A.omega p)
  have hfold : foldColumn A ch (fun ℓ => C ℓ p)
      = ∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (∑ c, P ℓ c * Xc c)
        + ∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (∑ j, μ ℓ j * Gc j) := by
    rw [foldColumn, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun ℓ hℓ => ?_
    rw [hC ℓ (Finset.mem_range.1 hℓ), map_add, mul_add]
  have hext : ∑ i, ρ i * A.embed (C (xl i) p)
      = ∑ i, ρ i * A.embed (∑ c, R i c * Xc c) + ∑ i, ρ i * A.embed (∑ j, μR i j * Gc j) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [hX i, map_add, mul_add]
  have hpad : padCorr A (cols + 1) cols t c₀
        (fun j => ∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (μ ℓ j) + ∑ i, ρ i * A.embed (μR i j)) p
      = ∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (∑ j, μ ℓ j * Gc j)
        + ∑ i, ρ i * A.embed (∑ j, μR i j * Gc j) := by
    rw [padCorr]
    simp only [add_mul, Finset.sum_add_distrib]
    rw [sum_embed_swap, sum_embed_swap]
  have hrhs : ∑ c, (∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (P ℓ c) + ∑ i, ρ i * A.embed (R i c))
        * A.embed (Xc c)
      = ∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (∑ c, P ℓ c * Xc c)
        + ∑ i, ρ i * A.embed (∑ c, R i c * Xc c) := by
    simp only [add_mul, Finset.sum_add_distrib]
    rw [sum_embed_swap, sum_embed_swap]
  rw [padColumn, hfold, hext, hpad, hrhs]
  have two : ∀ x : K, x + x = 0 := fun x => by rw [← two_mul, CharTwo.two_eq_zero, zero_mul]
  linear_combination two (∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (∑ j, μ ℓ j * Gc j))
    + two (∑ i, ρ i * A.embed (∑ j, μR i j * Gc j))

/-! ## C-Flock ZK's level-0 padding is onto the opened positions -/

/-- **Lemma B's row (ii) for C-Flock ZK's code**: when the table's level-0 code at its opened positions `pos` is C-Flock ZK's (message in
`[0, 2^cols)`, padding `c₀·X̂_j + X̂_{2^cols+j}` at `cols + 1` bits), the positions are distinct, there are at most
`t_pad ≤ 2^cols` of them, and `X_L + κ` vanishes at none (`PadNonvanishing`), the padding reaches every vector of values
there. -/
theorem padOnto_M1 {W Salt Seed D Hi VL : Type} (T : Table F K W Salt Seed D Hi VL) (A : Arith F K) (hA : A.Correct)
    (cols : ℕ) (hL : cols ≤ 40) (ht : T.tpad ≤ 2 ^ cols) (hS : T.nS ≤ T.tpad) (κ : F) (pos : Fin T.nS → ℕ)
    (hpos : Function.Injective pos) (hlt : ∀ q, pos q < 2 ^ (cols + 1))
    (hGp : ∀ q j, T.Gp q j = κ * A.Xhat j.val (cols + 1) (A.omega (pos q))
      + A.Xhat (2 ^ cols + j.val) (cols + 1) (A.omega (pos q)))
    (hg : Assumptions.PadNonvanishing A cols κ) : T.PadOnto := by
  let b : Fin T.tpad → F[X] := fun j => xP A hA cols hL (Fin.castLE ht j)
  let x : Fin T.nS → F := fun q => A.omega (pos q)
  let g : Fin T.nS → F := fun q => A.What cols (A.omega (pos q)) + κ
  have hx : Function.Injective x := fun q q' h =>
    hpos (hA.omega_injective (cols + 1) (by omega) (hlt q) (hlt q') h)
  have hspan : ∀ P : F[X], P.degree < Fintype.card (Fin T.nS) → ∃ m : Fin T.tpad → F, ∑ k, C (m k) * b k = P := by
    intro P hP
    by_cases hP0 : P = 0
    · exact ⟨0, by simp [hP0]⟩
    · have hnat : P.natDegree < T.tpad :=
        lt_of_lt_of_le ((Polynomial.natDegree_lt_iff_degree_lt hP0).2 hP) (by simpa using hS)
      exact xP_span_lt A hA cols hL T.tpad ht P hnat
  have hsurj := padding_surjective b x hx g (fun q => hg (pos q) (hlt q)) hspan
  intro y
  obtain ⟨μ, hμ⟩ := hsurj y
  refine ⟨μ, funext fun q => ?_⟩
  have e : ∀ j : Fin T.tpad, T.Gp q j = g q * (b j).eval (x q) := fun j => by
    have hj : j.val < 2 ^ cols := lt_of_lt_of_le j.isLt ht
    rw [hGp, Xhat_succ_low A cols j.val hj, Xhat_succ_high A cols j.val hj]
    simp only [g, b, x]
    rw [← xP_eval A hA cols hL (Fin.castLE ht j)]
    simp only [Fin.val_castLE]
    ring
  have : g q * (∑ k, C (μ k) * b k).eval (x q) = y q := congrFun hμ q
  simp only [eval_finsetSum, eval_mul, eval_C] at this
  rw [← this, Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by rw [e]; ring

omit [Fintype F] [Fintype K] [DecidableEq K] in
/-- **Lemma B's row (viii) for the pads code** (§2.4 line 0, `Code(m) = Σ_i m_i·X^i` at the points of the pads tree): when
rep `r`'s opened pads positions are distinct nonzero points `x r k`, at most `nPad` of them, and the padding's
coefficient at position `k` is `x^(s + j)` (the padding follows the `s` pads), the padding reaches every vector of values
there. This is T4 with the monomials and `g(x) = x^s`. -/
theorem padsOnto_monomial {W Salt Seed D Hi VL : Type} (T : Table F K W Salt Seed D Hi VL) (s : ℕ) (x : Fin 2 → Fin T.nQ → F)
    (hx : ∀ r, Function.Injective (x r)) (h0 : ∀ r k, x r k ≠ 0) (hQ : T.nQ ≤ T.nPad)
    (hCp : ∀ r k j, T.Cp r k j = x r k ^ (s + j.val)) : T.PadsOnto := by
  intro r y
  let b : Fin T.nPad → F[X] := fun j => (Polynomial.X : F[X]) ^ j.val
  have hspan : ∀ P : F[X], P.degree < Fintype.card (Fin T.nQ) → ∃ m : Fin T.nPad → F, ∑ k, C (m k) * b k = P := by
    intro P hP
    by_cases hP0 : P = 0
    · exact ⟨0, by simp [hP0]⟩
    have hdeg : P.natDegree < T.nPad :=
      lt_of_lt_of_le ((Polynomial.natDegree_lt_iff_degree_lt hP0).2 (by simpa using hP)) hQ
    refine ⟨fun j => P.coeff j.val, ?_⟩
    conv_rhs => rw [P.as_sum_range' T.nPad hdeg]
    rw [← Fin.sum_univ_eq_sum_range (fun i => monomial i (P.coeff i))]
    simp only [b, C_mul_X_pow_eq_monomial]
  obtain ⟨p, hp⟩ := padding_surjective b (x r) (hx r) (fun k => x r k ^ s) (fun k => pow_ne_zero _ (h0 r k)) hspan y
  refine ⟨p, funext fun k => ?_⟩
  have : x r k ^ s * (∑ j, C (p j) * b j).eval (x r k) = y k := congrFun hp k
  simp only [b, eval_finsetSum, eval_mul, eval_C, eval_pow, Polynomial.eval_X] at this
  rw [← this, Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by rw [hCp, pow_add]; ring

/-! ## The masked verifier's inner check -/

namespace Table

variable {W Salt Seed D Hi VL : Type} (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- **The inner proof is complete** (§2.4 line 16): on the reference prover's output, for a witness whose pads satisfy
the constraint, the masked verifier's checks hold: `⟨λ, Y₁⟩ = θ + β·τ`, `Code(Y)_k = C_h[k] + β·C_μ[k]` at every opened
pads position, and every opened leaf (level 0, the pads tree, `τ`'s commitment) is the leaf of its opened content. -/
theorem inner_complete (w : W) (hinner : T.InnerHolds w) (ω : T.Rand) (r : Fin 2) :
    let v := T.view w ω
    ∑ i, T.lam r v.pre i * v.Y1 r i = T.theta r v.pre + T.β r * v.tau r
      ∧ (∀ k, ∑ i, T.Ch r k i * v.Y1 r i + ∑ j, T.Cp r k j * v.Y2 r j = v.ch r k + T.β r * v.cm r k)
      ∧ (∀ k, (v.leavesP r).1 k = T.leafP (v.ch r k, v.cm r k) (v.saltsP r k))
      ∧ v.tauCom r = T.leafT (v.tau r) (v.ytau r)
      ∧ ∀ q, v.leaves0.1 q = T.leaf0 (v.rows q, v.xrows q) (v.salts0 q) := by
  intro v
  refine ⟨?_, fun k => ?_, fun k => rfl, rfl, fun q => rfl⟩
  · have hI : ∑ i, T.lam r v.pre i * T.hFull (ω.2.2.1 r) i = T.theta r v.pre := hinner ω r
    show ∑ i, T.lam r v.pre i * (T.hFull (ω.2.2.1 r) i + T.β r * ω.2.2.2.2.2.1 r i)
      = T.theta r v.pre + T.β r * ∑ i, T.lam r v.pre i * ω.2.2.2.2.2.1 r i
    rw [← hI, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  · show ∑ i, T.Ch r k i * (T.hFull (ω.2.2.1 r) i + T.β r * ω.2.2.2.2.2.1 r i)
        + ∑ j, T.Cp r k j * (ω.2.2.2.1 r j + T.β r * ω.2.2.2.2.1 r j)
      = (∑ i, T.Ch r k i * T.hFull (ω.2.2.1 r) i + ∑ j, T.Cp r k j * ω.2.2.2.1 r j)
        + T.β r * (∑ i, T.Ch r k i * ω.2.2.2.2.2.1 r i + ∑ j, T.Cp r k j * ω.2.2.2.2.1 r j)
    simp only [mul_add, Finset.sum_add_distrib, Finset.mul_sum]
    have e1 : ∀ i, T.Ch r k i * (T.β r * ω.2.2.2.2.2.1 r i) = T.β r * (T.Ch r k i * ω.2.2.2.2.2.1 r i) :=
      fun i => by ring
    have e2 : ∀ j, T.Cp r k j * (T.β r * ω.2.2.2.2.1 r j) = T.β r * (T.Cp r k j * ω.2.2.2.2.1 r j) :=
      fun j => by ring
    simp only [e1, e2]
    ring

end Table

end FlockSoundness.ZK
