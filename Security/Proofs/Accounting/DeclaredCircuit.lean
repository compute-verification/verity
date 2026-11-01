import Mathlib
import Definitions.Accounting.Guarantees.DeclaredCircuit
import Proofs.Accounting.Stacking

/-! The first pieces of the proof of `Accounting.Guarantees.DeclaredCircuitRequirement`, whose theorem is in
`DeclaredCircuit/Requirement.lean`: a carrier of a declared circuit's is within its census over its life, its inputs
within the run-input bytes the stacking property's caps allow its windows (`Provenance.OfCircuit.budget`), a uniform
function restricted along an injection is uniform (`pr_comp_embedding`), and key binding moves a store whose layout is
steered after the coins to
uniform blocks at `2^a` times their likeliest summary's event, `δ` once, against an event that reads the summary as a
device told it does (`KeyBinding.pr_steered_le`). -/

namespace Accounting.RateSketch.Provenance

/-- **A carrier of a declared circuit's is within its census over its life**: over at most `life` of the audit's
windows, its inputs are at most the run-input bits the stacking property's caps allow in any `life` of them
(`8·R_life`), and its encodes at most `life` times the most nonlinear units (budgeted units and data products) the
circuit has in a window. -/
theorem OfCircuit.budget {N m : ℕ} {α ι : Type} {K : Provenance N m α} {cost : ℕ → ℕ∞} {W : Gates.IntDag ι}
    {caps : Gates.Caps} {A : Gates.Audit} {I : Finset ℕ} {life : ℕ} (hK : K.OfCircuit cost W A I)
    (hP : Gates.Property caps A W) (hI : I ⊆ Finset.range A.windows) (hcard : I.card ≤ life)
    (E : SketchStore N m) :
    K.inputs E ≤ 8 * ((Finset.range A.windows).powerset.filter fun I => I.card ≤ life).sup
        (fun I => ∑ v ∈ I, caps.runBytes v) ∧
      K.gates cost E ≤
        (life * (Finset.range A.windows).sup (fun v =>
          (Gates.census W fun t => A.win t = v).budgeted + Gates.productsAt W fun t => A.win t = v) : ℕ) := by
  have hsum : ∀ f : ℕ → ℕ, ∑ v ∈ I, f v ≤ life * (Finset.range A.windows).sup f := fun f =>
    calc ∑ v ∈ I, f v ≤ ∑ _v ∈ I, (Finset.range A.windows).sup f := Finset.sum_le_sum fun v hv => Finset.le_sup (hI hv)
      _ = I.card * (Finset.range A.windows).sup f := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ life * (Finset.range A.windows).sup f := Nat.mul_le_mul_right _ hcard
  refine ⟨(hK E).1.trans ?_, (hK E).2.trans ?_⟩
  · calc ∑ v ∈ I, (Gates.census W fun t => A.win t = v).runBits ≤ ∑ v ∈ I, 8 * caps.runBytes v :=
          Finset.sum_le_sum fun v hv => (hP.2.1 v (Finset.mem_range.mp (hI hv))).2.2.2.2.2.1
      _ = 8 * ∑ v ∈ I, caps.runBytes v := (Finset.mul_sum _ _ _).symm
      _ ≤ _ := Nat.mul_le_mul_left 8 (Finset.le_sup (f := fun I => ∑ v ∈ I, caps.runBytes v)
          (Finset.mem_filter.mpr ⟨Finset.mem_powerset.mpr hI, hcard⟩))
  · exact_mod_cast hsum fun v =>
      (Gates.census W fun t => A.win t = v).budgeted + Gates.productsAt W fun t => A.win t = v

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

/-- **Key binding with a layout steered after the coins, against devices told the summary**: when the layout is one
of `2^a`, `slot` of a summary `steer` that is any function of the codec and the coins, an event of the summary, the
blocks and independent coins is at most `2^a` times as likely as its likeliest summary's event on uniform blocks, plus
`δ`. The event may read the summary, as a device told it does. Key binding applies once, at the whole layout, and the
union over the `2^a` summaries is taken inside the uniform store, so `δ` is not multiplied. -/
theorem Assumptions.KeyBinding.pr_steered_le {Ω Ca Ch B : Type} [Fintype Ω] [Fintype Ca] [Fintype B] [Nonempty B]
    {T M N a : ℕ} {δ : ℝ≥0∞} {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B}
    (h : Assumptions.KeyBinding T M δ enc) (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (slot : Pous.Bits a → Fin N ↪ Fin T × Fin M) (steer : Ω × (Fin T → Ca) → Pous.Bits a) {C : Type} [Fintype C]
    (Q : Pous.Bits a → (Fin N → B) → C → Prop) {p : ℝ≥0∞}
    (hQ : ∀ s, pr (fun y : (Fin N → B) × C => Q s y.1 y.2) ≤ p) :
    pr (fun x : (Ω × (Fin T → Ca)) × C =>
        Q (steer x.1) (appendCoinBlocks enc choose (slot (steer x.1)) x.1) x.2)
      ≤ 2 ^ a * p + δ := by
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
  have h1 := (h.uniformBut_blocks choose full).pr_le () (fun F c => ∃ s, Q s (F ∘ π s) c)
  calc pr (fun x : (Ω × (Fin T → Ca)) × C =>
        Q (steer x.1) (appendCoinBlocks enc choose (slot (steer x.1)) x.1) x.2)
      ≤ pr (fun x : (Ω × (Fin T → Ca)) × C => ∃ s, Q s (appendCoinBlocks enc choose full x.1 ∘ π s) x.2) :=
        Bandwidth.pr_mono fun x hx => ⟨steer x.1, by rw [← hblk]; exact hx⟩
    _ ≤ pr (fun y : (Fin (T * M) → B) × C => ∃ s, Q s (y.1 ∘ π s) y.2) + δ := h1
    _ ≤ (∑ s, pr (fun y : (Fin (T * M) → B) × C => Q s (y.1 ∘ π s) y.2)) + δ :=
        add_le_add (Bandwidth.pr_exists_le _) le_rfl
    _ = (∑ s, pr (fun y : (Fin N → B) × C => Q s y.1 y.2)) + δ := by
        congr 1
        exact Finset.sum_congr rfl fun s _ => pr_comp_embedding_prod (π s) (Q s)
    _ ≤ (∑ _s : Pous.Bits a, p) + δ := add_le_add (Finset.sum_le_sum fun s _ => hQ s) le_rfl
    _ = 2 ^ a * p + δ := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        simp [Pous.Bits]

end Accounting
