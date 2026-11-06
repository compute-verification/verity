import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Fintype.Sum
import Mathlib.Data.List.Chain
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Pebbling.FischClaims1314
import Proofs.Pous.B5.ExPostFacto

/-!
# The one-way bound for Fisch's stacked graph: `PousBridge.b2ToB5` and `PousBridge.b5_stacked`

Three parts, composed at the end:

1. **Claim 13**, `PousFisch.claim13` (`Pous.SecurityProofs.Pebbling.FischClaims1314`). Here it is
   `PousBridge.fischClaim13`.
2. **B5**, `PousB5` (`Pous.SecurityProofs.B5.ExPostFacto`).
3. **The bridge**, `PousBridge.B2ToB5` (`b2ToB5_of_claim13`), in this file.

`b2ToB5` is the bridge with Claim 13 discharged. `b5_stacked` chains it with `Pous.Guarantees.B5ExPostFactoBounded` at
`s = ⌊(1 − ε)n⌋`, `D = ⌊βn⌋ − 1`, `k = n − ⌈εn/2⌉ + 1`, with the `n` last-layer nodes as challenges.
-/

/-!
# Bridge B2 → B5 (draft statement `B2ToB5`, proved from `Pous.Guarantees.FischClaim13`)

B5 (`Pous.Guarantees.B5ExPostFacto(Bounded)`) needs a `TopoDAG` and greedy black pebbling hardness `PebblingHard`.
Claims 12–14 (`Pous.Guarantees.B2Stacking`, `FischClaim13`, `FischClaim14`) speak about Fisch's stacked graph `V` with
edges `StackedEdge`. `B2ToB5` is the bridge. A topological numbering `e : V ≃ Fin N` of the stacked graph, with
`G.par` its edges, makes the last layer `V_t` an `(s, D, k)`-hard challenge set whenever `s ≤ (1 − ε)n`,
`D + 1 ≤ βn` and `k > n − εn/2`.

Modelling choices (points for review):

* **One-way labels, `δ = 0`.** B5's labels cannot be decoded, so there are no red pebbles and no reverse (green)
  moves. The game is B5's greedy black pebbling over all stacked edges, and Claim 13 enters with `δ = 0`,
  `red = ∅`. Claim 14's white/green game is not needed: an unpebbled path of `L` nodes delays black pebbling by
  `L` rounds (`not_mem_reach_of_chain`).
* **Numbering.** `e` and `G` are given, with `G.par (e v) ∋ e u ↔ StackedEdge u v`. A `TopoDAG` exists exactly when
  `e` is topological, and it gives Claim 13's `hdag` (rank `e`).
* **Challenges.** `VC = e '' V_t`, the `n` last-layer nodes. B5's bound then reads, at `s = ⌊(1 − ε)n⌋`,
  `D = ⌊βn⌋ − 1` and `k = n − ⌈εn/2⌉ + 1`: `N · 2^m · ((n(Q + 1) + N)²/2^w)^(⌊(1 − ε)n⌋ + 1)`.

`b2ToB5_of_claim13` proves `FischClaim13 → B2ToB5`, which checks that the draft is true given Claim 13. With the
pebbling prover's proof of `FischClaim13` it is unconditional.
-/

namespace PousBridge

open Pous Pous.DRG Pous.Labelling

/-- Greedy black pebbling needs `j` rounds to reach the `j`-th node of a path that starts with no pebble. -/
theorem not_mem_reach_of_chain {N : ℕ} (G : TopoDAG N) (P₀ : Finset (Fin N)) (p : List (Fin N))
    (hp : p.IsChain fun u v => u ∈ G.par v) (hfree : ∀ v ∈ p, v ∉ P₀) :
    ∀ (r j : ℕ) (hj : j < p.length), r ≤ j → p[j] ∉ reach G P₀ r := by
  intro r
  induction r with
  | zero => intro j hj _; exact hfree _ (List.getElem_mem _)
  | succ r ih =>
    intro j hj hrj h
    simp only [reach, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at h
    rcases h with h | h
    · exact ih j hj (by omega) h
    · obtain ⟨j', rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
      exact ih j' (by omega) (by omega) (h ((List.isChain_iff_getElem.1 hp) j' hj))

set_option linter.unusedVariables false in
/-- **Bridge B2 → B5** (draft for pinning). In `ℓ ≥ 1/(1 − α)` stacked `(n, α, β)`-depth-robust layers joined by
superconcentrators and numbered topologically by `e`, the last layer is an `(s, D, k)`-hard challenge set for greedy
black pebbling when `s ≤ (1 − ε)n`, `D + 1 ≤ βn` and `k > n − εn/2`. -/
def B2ToB5 : Prop :=
  ∀ {ℓ n N : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β ε : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hℓ : 1 ≤ (1 - α) * ℓ)
    (e : V ≃ Fin N) (G : TopoDAG N) (hG : ∀ u v, e u ∈ G.par (e v) ↔ StackedEdge node drg green u v)
    (t : Fin ℓ) (ht : (t : ℕ) + 1 = ℓ) (s D k : ℕ)
    (hs : (s : ℝ) ≤ (1 - ε) * n) (hD : ((D + 1 : ℕ) : ℝ) ≤ β * n) (hk : (n : ℝ) - ε / 2 * n < k),
    PebblingHard G (Finset.univ.image fun b => e (node t b)) s D k

/-- The bridge holds given Claim 13 (at `δ = 0`, no red pebbles). -/
theorem b2ToB5_of_claim13 (h13 : Pous.Guarantees.FischClaim13) : B2ToB5 := by
  intro ℓ n N V _ _ α β ε node level drg green hnode hlevel hdr hsc hn hε hℓ e G hG t ht s D k hs hD hk P₀ hP₀
  classical
  have hdag : ∃ r : V → ℕ, ∀ u v, StackedEdge node drg green u v → r u < r v :=
    ⟨fun v => (e v : ℕ), fun u v huv => G.par_lt (e v) (e u) ((hG u v).2 huv)⟩
  have hblack : ((P₀.image e.symm).card : ℝ) ≤ (1 - ε) * n :=
    le_trans (by exact_mod_cast Finset.card_image_le.trans hP₀) hs
  obtain ⟨E, hEcard, hE⟩ := h13 α β ε 0 node level drg green hnode hlevel hdag hdr hsc hn hε
    (by linarith) (by simpa using hℓ) (P₀.image e.symm) ∅ hblack (fun i => by simp) t ht
  -- every endpoint in `E` ends an unpebbled path of at least `D + 1` nodes, so it is not pebbled in `D` rounds
  have hnot : ∀ b ∈ E, e (node t b) ∉ reach G P₀ D := by
    intro b hb
    obtain ⟨P, hchain, -, hlen, hunp, hlast, -⟩ := hE b hb
    have hp : (P.map e).IsChain fun u v => u ∈ G.par v :=
      (List.isChain_map e).2 (hchain.imp fun u v h => (hG u v).2 h)
    have hfree : ∀ v ∈ P.map e, v ∉ P₀ := by
      intro v hv h
      obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hv
      exact (hunp u hu).1 (Finset.mem_image.2 ⟨e u, h, e.symm_apply_apply u⟩)
    have hlenp : D + 1 ≤ (P.map e).length := by
      rw [List.length_map]
      exact_mod_cast hD.trans hlen
    have hlastp : (P.map e)[(P.map e).length - 1]'(by omega) = e (node t b) := by
      have := List.getLast?_eq_getElem? (l := P.map e)
      rw [List.getLast?_map, hlast, Option.map_some, List.getElem?_eq_getElem (by omega)] at this
      exact (Option.some.inj this).symm
    rw [← hlastp]
    exact not_mem_reach_of_chain G P₀ _ hp hfree D _ (by omega) (by omega)
  -- so at most `n − |E| ≤ n − εn/2 < k` challenges are pebbled
  have hinj : Function.Injective fun b => e (node t b) := fun a b h => hnode t (e.injective h)
  have hsub : (Finset.univ.image fun b => e (node t b)).filter (fun c => c ∈ reach G P₀ D) ⊆
      (Finset.univ.image fun b => e (node t b)) \ E.image fun b => e (node t b) := by
    intro c hc
    rw [Finset.mem_filter] at hc
    rw [Finset.mem_sdiff]
    refine ⟨hc.1, fun hcE => ?_⟩
    obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 hcE
    exact hnot b hb hc.2
  have hEn : E.card ≤ n := by simpa using E.card_le_univ
  have hcard : ((Finset.univ.image fun b => e (node t b)) \ E.image fun b => e (node t b)).card = n - E.card := by
    rw [Finset.card_sdiff, Finset.inter_eq_left.2 (Finset.image_subset_image (Finset.subset_univ _)),
      Finset.card_image_of_injective _ hinj, Finset.card_image_of_injective _ hinj, Finset.card_univ,
      Fintype.card_fin]
  refine lt_of_le_of_lt ((Finset.card_le_card hsub).trans hcard.le) ?_
  have : ((n - E.card : ℕ) : ℝ) < k := by
    rw [Nat.cast_sub hEn]
    linarith
  exact_mod_cast this

end PousBridge

/-! ## Composition: the bridge unconditionally, and B5 on the stacked graph -/

namespace PousBridge

open Pous Pous.DRG Pous.Labelling
open scoped ENNReal

/-- Claim 13 (`Pous.Guarantees.FischClaim13`), by the pebbling prover's proof copied verbatim above. -/
theorem fischClaim13 : Pous.Guarantees.FischClaim13 := @PousFisch.claim13

/-- **The bridge B2 → B5, unconditionally.** -/
theorem b2ToB5 : B2ToB5 := b2ToB5_of_claim13 fischClaim13

set_option linter.unusedVariables false in
/-- **The one-way bound for the stacked graph** (B2's graph under B5, with the bridge's hardness at
`s = ⌊(1 − ε)n⌋`, `D = ⌊βn⌋ − 1`, `k = n − ⌈εn/2⌉ + 1`). In `ℓ ≥ 1/(1 − α)` stacked `(n, α, β)`-depth-robust layers
joined by superconcentrators, numbered topologically by `e`, with one-way data-carrying labels, an `m`-bit
preprocessor and `(⌊βn⌋ − 1, Q)`-bounded online programs answer at least `n − ⌈εn/2⌉ + 1` of the `n` last-layer
challenges with probability at most `N · 2^m · ((n(Q + 1) + N)² / 2^w)^(⌊(1 − ε)n⌋ + 1)`. -/
theorem b5_stacked {ℓ n N : ℕ} {V : Type} [Fintype V] [DecidableEq V] (α β ε : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hε1 : ε ≤ 1) (hℓ : 1 ≤ (1 - α) * ℓ) (hβ : 1 ≤ β * n)
    (e : V ≃ Fin N) (G : TopoDAG N) (hG : ∀ u v, e u ∈ G.par (e v) ↔ StackedEdge node drg green u v)
    (t : Fin ℓ) (ht : (t : ℕ) + 1 = ℓ)
    (w m Q : ℕ) (d : Fin N → Bits w)
    (A₁ : (LabelQry N w → Bits w) → Bits m)
    (A₂ : Bits m → Fin N → Prog (IdealModel.randomOracle (LabelQry N w) (Bits w)) (Bits w))
    (hA₂ : ∀ σ c, (A₂ σ c).Bounded (⌊β * n⌋₊ - 1) Q) :
    pr (fun H : LabelQry N w → Bits w =>
        n - ⌈ε / 2 * n⌉₊ + 1 ≤ ((Finset.univ.image fun b => e (node t b)).filter fun c =>
          (A₂ (A₁ H) c).run H = label G d H c).card) ≤
      N * 2 ^ m * ((((n * (Q + 1) + N) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w) ^ (⌊(1 - ε) * n⌋₊ + 1) := by
  have hVC : (Finset.univ.image fun b => e (node t b)).card = n := by
    rw [Finset.card_image_of_injective _ fun a b h => hnode t (e.injective h), Finset.card_univ, Fintype.card_fin]
  have hβn : (0 : ℝ) ≤ β * n := by linarith
  have hfl : 1 ≤ ⌊β * n⌋₊ := Nat.le_floor (by exact_mod_cast hβ)
  have hard : PebblingHard G (Finset.univ.image fun b => e (node t b)) ⌊(1 - ε) * n⌋₊ (⌊β * n⌋₊ - 1)
      (n - ⌈ε / 2 * n⌉₊ + 1) := by
    refine b2ToB5 α β ε node level drg green hnode hlevel hdr hsc hn hε hℓ e G hG t ht _ _ _
      (Nat.floor_le (mul_nonneg (by linarith) (Nat.cast_nonneg n))) ?_ ?_
    · rw [show ⌊β * n⌋₊ - 1 + 1 = ⌊β * n⌋₊ by omega]
      exact Nat.floor_le hβn
    · have hx : (0 : ℝ) ≤ ε / 2 * n := by positivity
      have hc := Nat.ceil_lt_add_one hx
      by_cases hle : ⌈ε / 2 * n⌉₊ ≤ n
      · rw [Nat.cast_add, Nat.cast_sub hle, Nat.cast_one]
        linarith
      · have h0 : n - ⌈ε / 2 * n⌉₊ = 0 := by omega
        have hlt : (n : ℝ) < ⌈ε / 2 * n⌉₊ := by exact_mod_cast not_le.1 hle
        rw [h0, zero_add, Nat.cast_one]
        linarith
  have := PousB5.b5_ex_post_facto_bounded N w m ⌊(1 - ε) * n⌋₊ (⌊β * n⌋₊ - 1) Q (n - ⌈ε / 2 * n⌉₊ + 1) G _ d
    hard A₁ A₂ hA₂
  rwa [hVC] at this

end PousBridge
