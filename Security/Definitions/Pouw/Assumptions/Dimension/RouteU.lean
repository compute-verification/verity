import Definitions.Pouw.Assumptions.Dimension.Assumptions
import Definitions.Pouw.Assumptions.Dimension.Mixing
import Definitions.Pouw.Dimension.Causal
import Definitions.Pouw.Assumptions.NCP.RouteUWord

/-!
# Theorem D on route U's checked words (trusted statements)

Route U's checked value at bias `β` and position `p` of unit `u` is `checkedUPos β` (`Pouw.NCP.RouteUWord`). As a
function of the noise, with `A = g(F₁)`, it is `wordU β`; `wordCU β` reads a causal `A` at the noise itself, as
`wordC` does. Every statement holds for every `β`: the shift lies in `V₀` whatever its scale.

Route U's word minus the unbiased one is `β·(Σ_{l < 16t} Y[l][j] − Σ_{l < 3k} Y[l][j])`, a function of `F₁` alone
(the fixed `B`, `R` and the shared `F₁`; no `E₁`, no `A`), so it lies in `V₀` (`WordUSubV0`, `WordCUSubV0`). Everything
that reads the target words only modulo `V₀` transfers:

* a combination of route-U words lies in `V₀` iff the same combination of unbiased words does (`CombUV0Iff`), so
  the four-point evaluation lemma holds as it is (`SecondDifferenceU`);
* a family's independence modulo a subspace `V` passes to every family that differs from it by elements of `V`
  (`IndepModShift`, generic); with `count_le`, whose bound reads only the targets' independence modulo `V(inputs)`,
  Theorem D (`AlgBoundU`), its priced count, its sharpness (`AlgSharpU`), `TTNCPAlgU`, the causal bound
  (`AlgBoundCausalU`) and the mixing lemma (`MixingLemmaU`) hold for route-U words with the same bounds, and so does
  the wrapped form for `β ∈ {127, 128, 129}` when `R` is a signed permutation and `k ≤ 2^15` (`AlgBoundWrappedU`) or
  a permutation matrix and `k ≤ 2^16` (`AlgBoundWrappedUUnsigned`);
* A1 is stated on the unbiased span and is unchanged: it gives the same inequality on the route-U span (`A1U`), so
  `AlgBound4090U` holds with the same A1.

No reduction between programs is used, so no free instruction is added.

The final checkpoint is excluded, as in Theorem D: `C^U_T = A·B` (`RouteU.CheckedUFinal`).
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

/-- Unit `u`'s route-U word at position `p`, as a function of the noise, with the inputs `A = g(F₁)`. -/
def wordU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checkedUPos β k m n N R (A ω.2) B ω u p : ℚ)

/-- Unit `u`'s route-U word at position `p` for inputs `A` read at the noise itself. -/
def wordCU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checkedUPos β k m n N R (A ω) B ω u p : ℚ)

/-- `P` on inputs `L` outputs every non-final route-U word of the units in `U`. -/
def OutputsU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (P : List (Instr (Noise N m n k))) (L : List (Noise N m n k → ℚ)) : Prop :=
  ∀ u ∈ U, ∀ p < nonFinal k m n, wordU β k m n N R A B u p ∈ slots P L

/-- The span of the non-final route-U words of the units in `U`. -/
def targetSpanU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) :
    Submodule ℚ (Noise N m n k → ℚ) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) => wordU β k m n N R A B x.1.1 x.1.2)

/-! ## Pinned statements -/

/-- **The shift is free**: at every position, route U's word minus the unbiased word lies in `V₀`. -/
def WordUSubV0 : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ),
    wordU β k m n N R A B u p - word k m n N R A B u p ∈ V0 N m n k

/-- **The shift is free, causal inputs**: the same for every `A` read at the noise. -/
def WordCUSubV0 : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ),
    wordCU β k m n N R A B u p - wordC k m n N R A B u p ∈ V0 N m n k

/-- **Combinations agree modulo `V₀`**: a combination of non-final route-U words lies in `V₀` iff the same
combination of unbiased words does. -/
def CombUV0Iff : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ),
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordU β k m n N R A B u p) ∈ V0 N m n k ↔
      (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • word k m n N R A B u p) ∈ V0 N m n k

/-- **Independence modulo a subspace is invariant under shifts in it** (generic): if `t' i − t i ∈ V` for every
`i`, and no nonzero combination of the `t i` lies in `V`, then no nonzero combination of the `t' i` does. -/
def IndepModShift : Prop :=
  ∀ (G ι : Type) [Fintype ι] (V : Submodule ℚ (G → ℚ)) (t t' : ι → G → ℚ), (∀ i, t' i - t i ∈ V) →
    (∀ c : ι → ℚ, ∑ i, c i • t i ∈ V → ∀ i, c i = 0) → ∀ c : ι → ℚ, ∑ i, c i • t' i ∈ V → ∀ i, c i = 0

/-- **Route U's words are independent modulo `V₀`** under checkpoint independence. -/
def IndepModU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ), CheckpointIndependent k 16 R → U ⊆ range N →
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordU β k m n N R A B u p) ∈ V0 N m n k →
    ∀ u ∈ U, ∀ p < nonFinal k m n, c u p = 0

/-- **The four-point evaluation lemma on route U** (`SecondDifference` for route-U words): if a combination of
non-final route-U words lies in `V₀`, its coefficients on each entry's checkpoints annihilate the checkpoint
matrices. -/
def SecondDifferenceU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ), U ⊆ range N →
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordU β k m n N R A B u p) ∈ V0 N m n k →
    ∀ u ∈ U, ∀ e < m * n, ∀ a < k, ∀ b < k,
      ∑ t ∈ range (3 * k / 16 - 1), c u (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) a b : ℚ) = 0

/-- **The priced count on route U** (`PricedCount` for route-U words). -/
def PricedCountU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → OutputsU β k m n N R A B U P L →
    U.card * (m * n) * (3 * k / 16 - 1) ≤ nPriced P

/-- **Theorem D on route U** (`AlgBound` for route-U words, the same bound): under checkpoint independence and
H_16, every program on inputs in `V₀` that outputs the non-final route-U words of the units in `U` (sharing `F₁`,
with `A = g(F₁)`) costs at least `16·|U|·m·n·(3k/16 − 1)`. -/
def AlgBoundU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P → OutputsU β k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **Theorem D on route U is sharp** (`AlgSharp` for route-U words): below `γ₀ = 16/(3k)` some H_16 program on the
noise coordinates outputs the non-final route-U words and TT_NCP's event holds for it. -/
def AlgSharpU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) (γ₀ : ℝ),
    16 ∣ k → 0 < k → U ⊆ range N → U.Nonempty → 0 < m * n → γ₀ < 16 / (3 * k : ℝ) →
    ∃ P : List (Instr (Noise N m n k)), H16 P ∧ OutputsU β k m n N R A B U P (coords N m n k) ∧
      (cost P : ℝ) / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ)

/-- **TT_NCP in AlgM_16 on route U** (`TTNCPAlg` for route-U words). -/
def TTNCPAlgU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (γ₀ : ℝ),
    16 ∣ k → 0 < k → CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    OutputsU β k m n N R A B U P L → 16 / (3 * k : ℝ) ≤ γ₀ → γ₀ < 1 →
    ¬ ((cost P : ℝ) / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **Theorem D with causal inputs on route U** (`AlgBoundCausal` for route-U words). -/
def AlgBoundCausalU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → Causal A → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n, wordCU β k m n N R A B u p ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **The mixing lemma on route U** (`MixingLemma` for route-U words, `T` their span). -/
def MixingLemmaU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H4090 cs P →
    OutputsU β k m n N R A B U P L →
    Module.finrank ℚ (targetSpanU β k m n N R A B U) = U.card * (m * n) * (3 * k / 16 - 1) ∧
    16 * (bigOuts P).length + cs * (prodOuts P).length ≤ cost P ∧
    U.card * (m * n) * (3 * k / 16 - 1) ≤ (bigOuts P).length +
      Module.finrank ℚ ↥(targetSpanU β k m n N R A B U ⊓
        (Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k)) ∧
    Lmul (targetSpanU β k m n N R A B U ⊓ (Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k)) ≤
      (prodOuts P).length

/-- **Theorem D for route U's protocol words** (mod `2^32`, which are `Params.YU`'s by `RouteU.YUChecked`), as
`AlgBoundWrapped`: for `β ∈ {127, 128, 129}`, a signed-permutation `R`, `k ≤ 2^15`, `A = g(F₁)` int7 at every `F₁`
and `B` int7 on the units in `U`, under checkpoint independence and H_16, a program on inputs in `V₀` that outputs
the wrapped non-final route-U words costs at least `16·|U|·m·n·(3k/16 − 1)`. There the wrap is the identity
(`RouteU.Int7Int32U`). -/
def AlgBoundWrappedU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), 127 ≤ β → β ≤ 129 →
    CheckpointIndependent k 16 R → SignedPerm k R → k ≤ 2 ^ 15 → U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n,
      (fun ω => (wrap32 (checkedUPos β k m n N R (A ω.2) B ω u p) : ℚ)) ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **Theorem D for route U's protocol words, permutation-matrix `R`** (the production `P`): `AlgBoundWrappedU` with
a permutation matrix `R` and `k ≤ 2^16`, where the wrap is the identity by `RouteU.Int7Int32UUnsigned`. -/
def AlgBoundWrappedUUnsigned : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), 127 ≤ β → β ≤ 129 →
    CheckpointIndependent k 16 R → Unsigned k R → k ≤ 2 ^ 16 → U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n,
      (fun ω => (wrap32 (checkedUPos β k m n N R (A ω.2) B ω u p) : ℚ)) ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **A1 on route U's span** (A1 is unchanged): under checkpoint independence, `A1(c, γ₁)`, which is stated on the
span of the unbiased words, gives the same inequality for every subspace of the span of the route-U words. Both
spans are complements of `V₀` in the same space (`T ⊔ V₀ = T^U ⊔ V₀`), and `L(V)` reads `V` only modulo `V₀`. -/
def A1U : Prop :=
  ∀ (β : ℤ) (c γ₁ : ℝ) (k : ℕ) (R : Mat), CheckpointIndependent k 16 R → A1_multiplicative c γ₁ k R →
    ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ V : Submodule ℚ (Noise N m n k → ℚ), V ≤ targetSpanU β k m n N R A B U →
    ∀ (r : ℕ) (p : Fin r → Noise N m n k → ℚ), (∀ i, ProdFree (p i)) →
      V ≤ Submodule.span ℚ (Set.range p) ⊔ V0 N m n k →
      c * (1 - γ₁) * (Module.finrank ℚ V : ℝ) ≤ r

/-- **Theorem D in AlgM_4090 on route U** (`AlgBound4090` for route-U words, the same A1 and the same bound). -/
def AlgBound4090U : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ₁ : ℝ),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H4090 cs P →
    OutputsU β k m n N R A B U P L → A1_multiplicative c γ₁ k R → 16 ≤ c * cs → 0 ≤ γ₁ → γ₁ ≤ 1 →
    16 * (1 - γ₁) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **Theorem D on route U's hypotheses hold jointly, and its bound is attained** (`AlgWitness` for route-U words):
for every `k ≥ 1024` with `16 ∣ k`, some admissible, checkpoint-independent `R` and some H_16 program on the noise
coordinates output the non-final route-U words at cost exactly `16·|U|·m·n·(3k/16 − 1)`. -/
def AlgWitnessU : Prop :=
  ∀ (β : ℤ) (k : ℕ), 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ), U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      (∀ f ∈ coords N m n k, f ∈ V0 N m n k) ∧ H16 P ∧ OutputsU β k m n N R A B U P (coords N m n k) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

/-- **AlgM_4090's hypotheses on route U hold jointly** (`Alg4090Witness` for route-U words): for every `k ≥ 1024` with
`16 ∣ k`, some admissible, checkpoint-independent `R` and some H_4090 program (at `cs = 8`, one product) on the
noise coordinates output the non-final route-U words at cost `16·|U|·m·n·(3k/16 − 1) + 8`. -/
def Alg4090WitnessU : Prop :=
  ∀ (β : ℤ) (k : ℕ), 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ), U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      H4090 8 P ∧ OutputsU β k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 1 ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

/-- **`AlgBoundWrappedU`'s hypotheses hold jointly, and its bound is attained**: for every `β ∈ {127, 128, 129}`,
every `k` with `1024 ≤ k ≤ 2^16` and `16 ∣ k`, and every `A = g(F₁)` and `B` int7 on the units in `U`, some
admissible, permutation-matrix (so signed-permutation), checkpoint-independent `R` and some H_16 program on the noise
coordinates output the wrapped non-final route-U words at cost exactly `16·|U|·m·n·(3k/16 − 1)`. -/
def AlgWrappedWitnessU : Prop :=
  ∀ (β : ℤ) (k : ℕ), 127 ≤ β → β ≤ 129 → 1024 ≤ k → 16 ∣ k → k ≤ 2 ^ 16 →
    ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ Unsigned k R ∧ SignedPerm k R ∧
      CheckpointIndependent k 16 R ∧ (∀ f ∈ coords N m n k, f ∈ V0 N m n k) ∧ H16 P ∧
      (∀ u ∈ U, ∀ p < nonFinal k m n,
        (fun ω => (wrap32 (checkedUPos β k m n N R (A ω.2) B ω u p) : ℚ)) ∈ slots P (coords N m n k)) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

end Pouw.Dimension
