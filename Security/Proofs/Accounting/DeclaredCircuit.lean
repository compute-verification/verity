import Mathlib
import Models.Accounting.Guarantees.DeclaredCircuit
import Proofs.Accounting.Stacking

/-! The pieces of the proof of `Accounting.Guarantees.DeclaredCircuitRequirement`, whose theorem waits on the red team
of the statement (until then the lock lists the statement as `unstated`): a carrier of a declared
circuit's is within its census over its life (`Provenance.OfCircuit.budget`), which the stacking property bounds by its
caps a certified byte and its run-input bytes a window, a uniform function restricted along an
injection is uniform (`pr_comp_embedding`), and key binding moves a store whose layout is steered after the coins to
uniform blocks at `2^a` times their event, `δ` once (`KeyBinding.pr_steered_le`). -/

namespace Accounting.RateSketch.Provenance

/-- **A carrier of a declared circuit's is within its census over its life**: over at most `life` of the audit's
windows, its inputs are at most `life` times the most run-input bits the circuit's census has in a window, and its
encodes at most `life` times the most budgeted units. -/
theorem OfCircuit.budget {N m : ℕ} {α ι : Type} {K : Provenance N m α} {cost : ℕ → ℕ∞} {W : Gates.Dag ι}
    {A : Gates.Audit} {I : Finset ℕ} {life : ℕ} (hK : K.OfCircuit cost W A I) (hI : I ⊆ Finset.range A.windows)
    (hcard : I.card ≤ life) (E : SketchStore N m) :
    K.inputs E ≤ life * (Finset.range A.windows).sup (fun v => (Gates.census W fun t => A.win t = v).runBits) ∧
      K.gates cost E ≤
        (life * (Finset.range A.windows).sup (fun v => (Gates.census W fun t => A.win t = v).budgeted) : ℕ) := by
  have hsum : ∀ f : ℕ → ℕ, ∑ v ∈ I, f v ≤ life * (Finset.range A.windows).sup f := fun f =>
    calc ∑ v ∈ I, f v ≤ ∑ _v ∈ I, (Finset.range A.windows).sup f := Finset.sum_le_sum fun v hv => Finset.le_sup (hI hv)
      _ = I.card * (Finset.range A.windows).sup f := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ life * (Finset.range A.windows).sup f := Nat.mul_le_mul_right _ hcard
  refine ⟨(hK E).1.trans (hsum _), (hK E).2.trans ?_⟩
  exact_mod_cast hsum fun v => (Gates.census W fun t => A.win t = v).budgeted

end Accounting.RateSketch.Provenance

namespace Accounting

open Pous (pr)
open scoped ENNReal

/-- **A uniform function restricted along an injection is uniform**: an event of `F ∘ π`, for a uniform `F` on `π`'s
codomain, is as likely as the event of a uniform function on its domain. -/
theorem pr_comp_embedding {α β B : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β] [Fintype B] [Nonempty B]
    (π : α ↪ β) (Q : (α → B) → Prop) : pr (fun F : β → B => Q (F ∘ π)) = pr Q := by
  classical
  let p : β → Prop := fun x => x ∈ Set.range π
  let e : (β → B) ≃ (α → B) × ({x // ¬ p x} → B) :=
    (Equiv.piEquivPiSubtypeProd p fun _ => B).trans
      (Equiv.prodCongr (Equiv.arrowCongr (Equiv.ofInjective π π.injective).symm (Equiv.refl B)) (Equiv.refl _))
  have he : ∀ F, (e F).1 = F ∘ π := fun F => by
    funext x
    rfl
  have hs : ∀ y, e.symm y ∘ π = y.1 := fun y => by rw [← he, Equiv.apply_symm_apply]
  rw [Rate.pr_equiv e (fun F => Q (F ∘ π))]
  simp only [hs]
  rw [Assumptions.pr_prod_eq_sum_div (fun y : (α → B) × ({x // ¬ p x} → B) => Q y.1)]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [mul_comm, ENNReal.mul_div_cancel_right (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)]

/-- `pr_comp_embedding` beside independent coins. -/
theorem pr_comp_embedding_prod {α β B C : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β] [Fintype B]
    [Nonempty B] [Fintype C] (π : α ↪ β) (Q : (α → B) → C → Prop) :
    pr (fun y : (β → B) × C => Q (y.1 ∘ π) y.2) = pr (fun y : (α → B) × C => Q y.1 y.2) := by
  rw [Assumptions.pr_prod_eq_sum_div (fun y : (β → B) × C => Q (y.1 ∘ π) y.2),
    Assumptions.pr_prod_eq_sum_div (fun y : (α → B) × C => Q y.1 y.2)]
  congr 1
  exact Finset.sum_congr rfl fun c _ => pr_comp_embedding π (fun E => Q E c)

/-- **Key binding with a layout steered after the coins**: when the layout is one of `2^a`, chosen by any function
`steer` of the codec and the coins, an event of the blocks and of independent coins is at most `2^a` times as likely
as on uniform blocks, plus `δ`. Key binding applies once, at the whole layout, and the union over the `2^a` layouts is
taken inside the uniform store, so `δ` is not multiplied. -/
theorem Assumptions.KeyBinding.pr_steered_le {Ω Ca Ch B : Type} [Fintype Ω] [Fintype Ca] [Fintype B] [Nonempty B]
    {T M N a : ℕ} {δ : ℝ≥0∞} {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B}
    (h : Assumptions.KeyBinding T M δ enc) (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (slot : Pous.Bits a → Fin N ↪ Fin T × Fin M) (steer : Ω × (Fin T → Ca) → Pous.Bits a) {C : Type} [Fintype C]
    (Q : (Fin N → B) → C → Prop) :
    pr (fun x : (Ω × (Fin T → Ca)) × C => Q (appendCoinBlocks enc choose (slot (steer x.1)) x.1) x.2)
      ≤ 2 ^ a * pr (fun y : (Fin N → B) × C => Q y.1 y.2) + δ := by
  classical
  let full : Fin (T * M) ↪ Fin T × Fin M := finProdFinEquiv.symm.toEmbedding
  let π : Pous.Bits a → Fin N ↪ Fin (T * M) := fun s => (slot s).trans finProdFinEquiv.toEmbedding
  have hfull : ∀ p, full (finProdFinEquiv p) = p := fun p => finProdFinEquiv.symm_apply_apply p
  have hblk : ∀ s y, appendCoinBlocks enc choose (slot s) y = appendCoinBlocks enc choose full y ∘ π s :=
    fun s y => by
      funext u
      show _ = appendCoinBlocks enc choose full y (finProdFinEquiv (slot s u))
      unfold appendCoinBlocks
      rw [hfull]
  have h1 := (h.uniformBut_blocks choose full).pr_le () (fun F c => ∃ s, Q (F ∘ π s) c)
  calc pr (fun x : (Ω × (Fin T → Ca)) × C => Q (appendCoinBlocks enc choose (slot (steer x.1)) x.1) x.2)
      ≤ pr (fun x : (Ω × (Fin T → Ca)) × C => ∃ s, Q (appendCoinBlocks enc choose full x.1 ∘ π s) x.2) :=
        Bandwidth.pr_mono fun x hx => ⟨steer x.1, by rw [← hblk]; exact hx⟩
    _ ≤ pr (fun y : (Fin (T * M) → B) × C => ∃ s, Q (y.1 ∘ π s) y.2) + δ := h1
    _ ≤ (∑ s, pr (fun y : (Fin (T * M) → B) × C => Q (y.1 ∘ π s) y.2)) + δ :=
        add_le_add (Bandwidth.pr_exists_le _) le_rfl
    _ = (∑ _s : Pous.Bits a, pr (fun y : (Fin N → B) × C => Q y.1 y.2)) + δ := by
        congr 1
        exact Finset.sum_congr rfl fun s _ => pr_comp_embedding_prod (π s) Q
    _ = 2 ^ a * pr (fun y : (Fin N → B) × C => Q y.1 y.2) + δ := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        simp [Pous.Bits]

end Accounting
