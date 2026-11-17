module

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.Data.Fintype.Option
public import Mathlib.Data.Fintype.Sum
public import Mathlib.Data.List.Chain
public import Mathlib.Tactic.GCongr
public import Mathlib.Tactic.Linarith
public import Definitions.Pous
public import Definitions.Pous.Assumptions
public import Proofs.Pous.Statements
public import Proofs.Pous.Sanity
public import Proofs.Pous.Accounting.Numbers
public import Proofs.Pous.Pebbling.FischClaims1314
public import Proofs.Pous.B5.ExPostFacto

/-! The one-way bound for Fisch's stacked graph, from Claim 13, B5 and the bridge `PousBridge.B2ToB5`. -/

@[expose] public section

/-! The bridge B2 → B5: the draft statement `B2ToB5`, proved from `Pous.Guarantees.FischClaim13`. -/

namespace PousBridge

open Pous Pous.DRG Pous.Labelling

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
/-- **Bridge B2 → B5** (draft for pinning). -/
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

theorem b2ToB5_of_claim13 (h13 : Pous.Guarantees.FischClaim13) : B2ToB5 := by
  intro ℓ n N V _ _ α β ε node level drg green hnode hlevel hdr hsc hn hε hℓ e G hG t ht s D k hs hD hk P₀ hP₀
  classical
  have hdag : ∃ r : V → ℕ, ∀ u v, StackedEdge node drg green u v → r u < r v :=
    ⟨fun v => (e v : ℕ), fun u v huv => G.par_lt (e v) (e u) ((hG u v).2 huv)⟩
  have hblack : ((P₀.image e.symm).card : ℝ) ≤ (1 - ε) * n :=
    le_trans (by exact_mod_cast Finset.card_image_le.trans hP₀) hs
  obtain ⟨E, hEcard, hE⟩ := h13 α β ε 0 node level drg green hnode hlevel hdag hdr hsc hn hε
    (by linarith) (by simpa using hℓ) (P₀.image e.symm) ∅ hblack (fun i => by simp) t ht
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

namespace PousBridge

open Pous Pous.DRG Pous.Labelling
open scoped ENNReal

theorem fischClaim13 : Pous.Guarantees.FischClaim13 := @PousFisch.claim13

theorem b2ToB5 : B2ToB5 := b2ToB5_of_claim13 fischClaim13

set_option linter.unusedVariables false in
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
    (A₂ : Bits m → Fin N → OracleAlgorithm (IdealModel.randomOracle (LabelQry N w) (Bits w)) (Bits w))
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
