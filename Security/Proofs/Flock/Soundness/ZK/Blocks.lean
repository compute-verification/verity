import Proofs.Flock.Soundness.ZK.Adaptivity
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.FinCases

/-!
# Zero knowledge: `Adaptivity`'s two lemmas for four blocks of randomness

A masked table's randomness falls into four blocks (`Table`): the mask slot `u`; the level-0 paddings with the extra
lanes `R`; the pads `h`; the inner proof's masks with the salts. `Adaptivity`'s lemmas are stated for a family
`G : Fin K → Type`; here they are restated for a plain product `G₀ × G₁ × G₂ × G₃`, so that the prover's randomness is an
ordinary product and each block an ordinary type.

* `card_seq_masked4`: `card_seq_masked` for four blocks. If component `k` of a view is anything computed from blocks
  `j < k` plus a surjective image of block `k`, every value is taken as often as the masks' kernels allow.
* `card_fiber_eq_of_triShift4`: `card_fiber_eq_of_triShift` for four blocks. A translation of block `k` by an amount that
  reads only blocks `j < k` maps one view to another; so they take every value equally often.

Both are proved by instantiating the general lemma at `G = ![G₀, G₁, G₂, G₃]`, through the equivalence `blocksEquiv`.
-/

namespace FlockSoundness.ZK

open Finset

section Four

variable {G₀ G₁ G₂ G₃ : Type}

/-- The four blocks as a family over `Fin 4`. -/
abbrev Blocks (G₀ G₁ G₂ G₃ : Type) : Fin 4 → Type := ![G₀, G₁, G₂, G₃]

/-- A family of instances on the four blocks, from one on each. -/
abbrev blocksInst (P : Type → Type) (i₀ : P G₀) (i₁ : P G₁) (i₂ : P G₂) (i₃ : P G₃) :
    ∀ k, P (Blocks G₀ G₁ G₂ G₃ k) :=
  Fin.cases i₀ (Fin.cases i₁ (Fin.cases i₂ (Fin.cases i₃ finZeroElim)))

/-- The product of the four blocks is the dependent function type over `Fin 4`. -/
def blocksEquiv : G₀ × G₁ × G₂ × G₃ ≃ ∀ k, Blocks G₀ G₁ G₂ G₃ k where
  toFun ω := Fin.cases ω.1 (Fin.cases ω.2.1 (Fin.cases ω.2.2.1 (Fin.cases ω.2.2.2 finZeroElim)))
  invFun g := (g 0, g 1, g 2, g 3)
  left_inv ω := rfl
  right_inv g := by
    funext k
    fin_cases k <;> rfl

@[simp] theorem blocksEquiv_apply_zero (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 0 = ω.1 := rfl
@[simp] theorem blocksEquiv_apply_one (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 1 = ω.2.1 := rfl
@[simp] theorem blocksEquiv_apply_two (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 2 = ω.2.2.1 := rfl
@[simp] theorem blocksEquiv_apply_three (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 3 = ω.2.2.2 := rfl

end Four

/-- **`card_seq_masked` for four blocks.** Component `k` of the view is `f_k` of the blocks before `k` plus `φ_k` of block
`k`, each `φ_k` a surjective additive map. Then every value of the view is taken by exactly as many `ω` as the four
kernels' product has elements, whatever the `f_k` are. -/
theorem card_seq_masked4 {G₀ G₁ G₂ G₃ B₀ B₁ B₂ B₃ : Type}
    [AddCommGroup G₀] [AddCommGroup G₁] [AddCommGroup G₂] [AddCommGroup G₃]
    [Fintype G₀] [Fintype G₁] [Fintype G₂] [Fintype G₃]
    [DecidableEq G₀] [DecidableEq G₁] [DecidableEq G₂] [DecidableEq G₃]
    [AddCommGroup B₀] [AddCommGroup B₁] [AddCommGroup B₂] [AddCommGroup B₃]
    [DecidableEq B₀] [DecidableEq B₁] [DecidableEq B₂] [DecidableEq B₃]
    (φ₀ : G₀ →+ B₀) (φ₁ : G₁ →+ B₁) (φ₂ : G₂ →+ B₂) (φ₃ : G₃ →+ B₃)
    (h₀ : Function.Surjective φ₀) (h₁ : Function.Surjective φ₁) (h₂ : Function.Surjective φ₂)
    (h₃ : Function.Surjective φ₃)
    (f₀ : B₀) (f₁ : G₀ → B₁) (f₂ : G₀ → G₁ → B₂) (f₃ : G₀ → G₁ → G₂ → B₃) (v : B₀ × B₁ × B₂ × B₃) :
    #{ω : G₀ × G₁ × G₂ × G₃ | (f₀ + φ₀ ω.1, f₁ ω.1 + φ₁ ω.2.1, f₂ ω.1 ω.2.1 + φ₂ ω.2.2.1,
        f₃ ω.1 ω.2.1 ω.2.2.1 + φ₃ ω.2.2.2) = v}
      = #{κ : G₀ × G₁ × G₂ × G₃ | φ₀ κ.1 = 0 ∧ φ₁ κ.2.1 = 0 ∧ φ₂ κ.2.2.1 = 0 ∧ φ₃ κ.2.2.2 = 0} := by
  let G := Blocks G₀ G₁ G₂ G₃
  let B := Blocks B₀ B₁ B₂ B₃
  let iG : ∀ k, AddCommGroup (G k) := blocksInst AddCommGroup ‹_› ‹_› ‹_› ‹_›
  let iGf : ∀ k, Fintype (G k) := blocksInst Fintype ‹_› ‹_› ‹_› ‹_›
  let iGd : ∀ k, DecidableEq (G k) := blocksInst DecidableEq ‹_› ‹_› ‹_› ‹_›
  let iB : ∀ k, AddCommGroup (B k) := blocksInst AddCommGroup ‹_› ‹_› ‹_› ‹_›
  let iBd : ∀ k, DecidableEq (B k) := blocksInst DecidableEq ‹_› ‹_› ‹_› ‹_›
  let φ : ∀ k, G k →+ B k := Fin.cases φ₀ (Fin.cases φ₁ (Fin.cases φ₂ (Fin.cases φ₃ finZeroElim)))
  have hφ : ∀ k, Function.Surjective (φ k) := by
    intro k; fin_cases k
    exacts [h₀, h₁, h₂, h₃]
  let f : ∀ k, (∀ j, G j) → B k :=
    Fin.cases (fun _ => f₀) (Fin.cases (fun g => f₁ (g 0)) (Fin.cases (fun g => f₂ (g 0) (g 1))
      (Fin.cases (fun g => f₃ (g 0) (g 1) (g 2)) finZeroElim)))
  have hf : ∀ k (ω ω' : ∀ j, G j), (∀ j, j < k → ω j = ω' j) → f k ω = f k ω' := by
    intro k ω ω' h
    fin_cases k
    · rfl
    · show f₁ (ω 0) = f₁ (ω' 0)
      rw [h 0 (by decide)]
    · show f₂ (ω 0) (ω 1) = f₂ (ω' 0) (ω' 1)
      rw [h 0 (by decide), h 1 (by decide)]
    · show f₃ (ω 0) (ω 1) (ω 2) = f₃ (ω' 0) (ω' 1) (ω' 2)
      rw [h 0 (by decide), h 1 (by decide), h 2 (by decide)]
  let vv : ∀ k, B k := blocksEquiv v
  have key := card_seq_masked φ hφ f hf vv
  have e1 : #{ω : G₀ × G₁ × G₂ × G₃ | (f₀ + φ₀ ω.1, f₁ ω.1 + φ₁ ω.2.1, f₂ ω.1 ω.2.1 + φ₂ ω.2.2.1,
        f₃ ω.1 ω.2.1 ω.2.2.1 + φ₃ ω.2.2.2) = v} = #{ω : ∀ k, G k | ∀ k, f k ω + φ k (ω k) = vv k} := by
    refine Finset.card_equiv blocksEquiv (fun ω => ?_)
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · rintro rfl k
      fin_cases k <;> rfl
    · intro h
      have h0 := h 0
      have h1 := h 1
      have h2 := h 2
      have h3 := h 3
      exact Prod.ext h0 (Prod.ext h1 (Prod.ext h2 h3))
  have e2 : #{κ : G₀ × G₁ × G₂ × G₃ | φ₀ κ.1 = 0 ∧ φ₁ κ.2.1 = 0 ∧ φ₂ κ.2.2.1 = 0 ∧ φ₃ κ.2.2.2 = 0}
      = #{κ : ∀ k, G k | ∀ k, φ k (κ k) = 0} := by
    refine Finset.card_equiv blocksEquiv (fun ω => ?_)
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨a, b, c, d⟩ k
      fin_cases k
      exacts [a, b, c, d]
    · intro h
      exact ⟨h 0, h 1, h 2, h 3⟩
  rw [e1, e2]
  convert key

/-- **`card_fiber_eq_of_triShift` for four blocks.** If `view ω = view' ω'`, where `ω'` translates block `k` of `ω` by an
amount that reads only the blocks before `k`, the two views take every value equally often. -/
theorem card_fiber_eq_of_triShift4 {G₀ G₁ G₂ G₃ V : Type} [AddGroup G₀] [AddGroup G₁] [AddGroup G₂] [AddGroup G₃]
    [Fintype G₀] [Fintype G₁] [Fintype G₂] [Fintype G₃]
    [DecidableEq G₀] [DecidableEq G₁] [DecidableEq G₂] [DecidableEq G₃] [DecidableEq V]
    (view view' : G₀ × G₁ × G₂ × G₃ → V) (d₀ : G₀) (d₁ : G₀ → G₁) (d₂ : G₀ → G₁ → G₂) (d₃ : G₀ → G₁ → G₂ → G₃)
    (h : ∀ ω, view ω = view' (ω.1 + d₀, ω.2.1 + d₁ ω.1, ω.2.2.1 + d₂ ω.1 ω.2.1,
      ω.2.2.2 + d₃ ω.1 ω.2.1 ω.2.2.1)) (v : V) :
    #{ω | view ω = v} = #{ω | view' ω = v} := by
  let G := Blocks G₀ G₁ G₂ G₃
  let iG : ∀ k, AddGroup (G k) := blocksInst AddGroup ‹_› ‹_› ‹_› ‹_›
  let iGf : ∀ k, Fintype (G k) := blocksInst Fintype ‹_› ‹_› ‹_› ‹_›
  let iGd : ∀ k, DecidableEq (G k) := blocksInst DecidableEq ‹_› ‹_› ‹_› ‹_›
  let δ : ∀ k, (∀ j, G j) → G k :=
    Fin.cases (fun _ => d₀) (Fin.cases (fun g => d₁ (g 0)) (Fin.cases (fun g => d₂ (g 0) (g 1))
      (Fin.cases (fun g => d₃ (g 0) (g 1) (g 2)) finZeroElim)))
  have hδ : Triangular δ := by
    intro k ω ω' h
    fin_cases k
    · rfl
    · show d₁ (ω 0) = d₁ (ω' 0)
      rw [h 0 (by decide)]
    · show d₂ (ω 0) (ω 1) = d₂ (ω' 0) (ω' 1)
      rw [h 0 (by decide), h 1 (by decide)]
    · show d₃ (ω 0) (ω 1) (ω 2) = d₃ (ω' 0) (ω' 1) (ω' 2)
      rw [h 0 (by decide), h 1 (by decide), h 2 (by decide)]
  let e := (blocksEquiv (G₀ := G₀) (G₁ := G₁) (G₂ := G₂) (G₃ := G₃))
  have key := card_fiber_eq_of_triShift (fun g => view (e.symm g)) (fun g => view' (e.symm g)) hδ
    (fun g => by
      rw [h (e.symm g)]
      rfl) v
  have e1 : ∀ w : G₀ × G₁ × G₂ × G₃ → V, #{ω | w ω = v} = #{g : ∀ k, G k | w (e.symm g) = v} := fun w =>
    Finset.card_equiv e (fun ω => by simp)
  rw [e1 view, e1 view']
  convert key

end FlockSoundness.ZK
