module

public import Proofs.Slowdown.CutCharged
public import Proofs.Core.Hardware

/-!
# Training work on a machine whose work is mostly used by inference blocks

A machine runs the registered circuit (`Hardware.Computation`), and the blocks `G` of a partition, each reading at
most `k` gates outside `θ`, use at least `x` of its resources `R`. Every computed gate uses at least one of `R`, so the
gates of the other blocks number at most what the machine has left, `r = (∑ cap) T - x` (`Computation.gates_outside`,
`Computation.work_outside`), and they hold whole at most `r / γ` MACs of at least `γ` gates each. With `HECharged`'s
count for the blocks of `G`: `γ N ≤ r + γ (T' + ½ √(S O / q) + ½ √(S T' / (q b)))` (`on_machine`).
-/

@[expose] public section

namespace Slowdown.Cut.SecurityProofs

open Verity Finset InferenceOnly
open scoped NNRat

/-- The gates of the blocks outside `G` are computed gates outside the blocks of `G`. -/
theorem card_rest_le {C : Circuit} {m : ℕ} (P : Partition C m) (G : Finset (Fin m)) :
    ∑ u ∈ Gᶜ, (P.gates u).card ≤ C.size (C.computed \ G.biUnion P.gates) := by
  have hd : ∀ u ∈ Gᶜ, ∀ v ∈ Gᶜ, u ≠ v → Disjoint (P.gates u) (P.gates v) := fun u _ v _ huv =>
    disjoint_left.mpr fun g hu hv => huv (Option.some.inj ((mem_filter.mp hu).2.symm.trans (mem_filter.mp hv).2))
  rw [← card_biUnion hd]
  refine card_le_card fun g hg => ?_
  obtain ⟨u, hu, hgu⟩ := mem_biUnion.mp hg
  have hunit := (mem_filter.mp hgu).2
  refine mem_sdiff.mpr ⟨mem_filter.mpr ⟨mem_univ g, ?_⟩, fun hS => ?_⟩
  · by_contra hin
    rw [(P.input_iff g).mpr (by simpa using hin)] at hunit
    exact absurd hunit (by simp)
  · obtain ⟨v, hv, hgv⟩ := mem_biUnion.mp hS
    have : u = v := Option.some.inj (hunit.symm.trans (mem_filter.mp hgv).2)
    exact (mem_compl.mp hu) (this ▸ hv)

/-- **Training work on a machine.** A machine runs the registered circuit within its limits, every computed gate using
at least one of its resources `R`, and the blocks of `G` use at least `x` of them. Under `HECharged`'s premises, with
MACs of at least `γ` gates, `γ N` is at most what the machine has left of `R` over the `T` cycles and `γ` times the
count for the blocks of `G`. -/
theorem on_machine {ι V κ : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit)
    (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) (k m γ T' S : ℕ) (P : Partition d.D m) (G : Finset (Fin m))
    (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k)
    (hE : Assumptions.Exports W b) (hR : ReadsOn d P G k T' S) (hγ : ∀ s, γ ≤ (W.mac s).card)
    {M : Hardware.Machine κ} {T : ℕ} (X : Hardware.Computation M d.D T) (hX : X.Within) (R : Finset M.Res)
    (h1 : ∀ g ∈ d.D.computed, 1 ≤ ∑ e ∈ R, X.cost e g) {x : ℚ≥0} (hx' : x ≤ X.used R (G.biUnion P.gates)) :
    (γ : ℝ) * W.N ≤
      (((∑ e ∈ R, (M.node e.1).cap e.2) * T - x : ℚ≥0) : ℝ) + γ * workHEOn T' S (βw * βx) b W.O := by
  have hN := (HECharged ι V d W val k m 0 γ T' S P G βw βx b hw hx hb hval hshare hH hE hR).1
  have hin : ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) * γ ≤
      (((∑ e ∈ R, (M.node e.1).cap e.2) * T - x : ℚ≥0) : ℝ) := by
    have h1' : (∑ u ∈ Gᶜ, (W.inside (P.gates u)).card) * γ ≤ d.D.size (d.D.computed \ G.biUnion P.gates) := by
      rw [sum_mul]
      exact (sum_le_sum fun u _ => card_inside_mul_le W (P.gates u) hγ).trans (card_rest_le P G)
    have h2 : ((d.D.size (d.D.computed \ G.biUnion P.gates) : ℚ≥0)) ≤
        (∑ e ∈ R, (M.node e.1).cap e.2) * T - x :=
      (X.gates_outside R _ h1).trans (X.work_outside hX R hx')
    have h3 : ((d.D.size (d.D.computed \ G.biUnion P.gates) : ℕ) : ℝ) ≤
        (((∑ e ∈ R, (M.node e.1).cap e.2) * T - x : ℚ≥0) : ℝ) := by
      have := NNRat.cast_le (K := ℝ) |>.mpr h2
      simpa using this
    exact (by exact_mod_cast h1' : ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) * γ ≤ _).trans h3
  have hγ0 : (0 : ℝ) ≤ γ := Nat.cast_nonneg γ
  nlinarith [mul_le_mul_of_nonneg_left hN hγ0]

end Slowdown.Cut.SecurityProofs
