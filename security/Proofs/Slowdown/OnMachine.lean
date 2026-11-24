module

public import Proofs.Slowdown.CutCharged
public import Proofs.ResourceAccounting.Actual.Outside

/-!
# Training work on a machine whose work is mostly used by inference blocks

The registered circuit runs at a location `X` over `[0, t)` (an actual circuit, `ResourceAccounting.Placed`), and the
blocks `G` of a partition, each reading at most `k` gates outside `θ`, do at least `x` of its work. At a pricing of at
least one a gate, the gates of the other blocks number at most what the location has left, `r = Φ_X t − x`
(`ResourceAccounting.SecurityProofs.Actual.gates_outside`), and they hold whole at most `r / γ` MACs of at least `γ`
gates each. With `HECharged`'s count for the blocks of `G`: `γ N ≤ r + γ (T' + ½ √(S O / q) + ½ √(S T' / (q b)))`
(`on_machine`).
-/

@[expose] public section

namespace Slowdown.Cut.SecurityProofs

open Verity Finset InferenceOnly ResourceAccounting

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

/-- **Training work on a machine.** The registered circuit, placed by `px`, `pt` and `pcell`, runs at the location `X`
over `[0, t)` at compute rate `Φ_X`, at a pricing of at least one a gate, and the blocks of `G` do at least `x` of the work. Under `HECharged`'s
premises, with MACs of at least `γ` gates, `γ N` is at most what the location has left, `Φ_X t − x`, and `γ` times the
count for the blocks of `G`. -/
theorem on_machine {ι V : Type} {L : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit)
    (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) (k m γ T' S : ℕ) (P : Partition d.D m) (G : Finset (Fin m))
    (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k)
    (hE : Assumptions.Exports W b) (hR : ReadsOn d P G k T' S) (hγ : ∀ s, γ ≤ (W.mac s).card)
    (px : Fin d.D.N → L) (pt pcell : Fin d.D.N → ℕ) (pinj : Function.Injective fun g => (px g, pcell g, pt g))
    {𝒳 : Set (Finset L)} {π : Fin d.D.N → ℕ} {compute : Finset L → ℕ}
    (hrate : (⟨d.D, px, pt, pcell, pinj⟩ : Placed L).ComputeRate 𝒳 π compute)
    (hπ : ∀ g, d.D.isInput g = false → 1 ≤ π g) {X : Finset L} (hX : X ∈ 𝒳) (t : ℕ)
    (hall : ∀ g, d.D.isInput g = false → px g ∈ X ∧ pt g < t) {x : ℕ}
    (hx' : x ≤ ∑ g ∈ d.D.computed ∩ G.biUnion P.gates, π g) :
    (γ : ℝ) * W.N ≤ ((compute X * t - x : ℕ) : ℝ) + γ * workHEOn T' S (βw * βx) b W.O := by
  set A : Placed L := ⟨d.D, px, pt, pcell, pinj⟩
  have hreg : A.region X (Set.Ico 0 t) = d.D.computed := by
    ext g
    rw [ResourceAccounting.SecurityProofs.Actual.mem_region A]
    simp only [Circuit.computed, mem_filter, mem_univ, true_and, Set.mem_Ico, Nat.zero_le]
    exact ⟨fun h => h.1, fun h => ⟨h, hall g h⟩⟩
  have hx'' : x ≤ A.work π (A.region X (Set.Ico 0 t) ∩ G.biUnion P.gates) := by
    rw [hreg]
    exact hx'
  have hout := ResourceAccounting.SecurityProofs.Actual.gates_outside A hrate hπ hX 0 t (G.biUnion P.gates) hx''
  rw [hreg, Nat.sub_zero] at hout
  have hN := (HECharged ι V d W val k m 0 γ T' S P G βw βx b hw hx hb hval hshare hH hE hR).1
  have hin : (∑ u ∈ Gᶜ, (W.inside (P.gates u)).card) * γ ≤ compute X * t - x := by
    rw [sum_mul]
    exact ((sum_le_sum fun u _ => card_inside_mul_le W (P.gates u) hγ).trans (card_rest_le P G)).trans hout
  have hin' : ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) * γ ≤ ((compute X * t - x : ℕ) : ℝ) := by
    exact_mod_cast hin
  have hγ0 : (0 : ℝ) ≤ γ := Nat.cast_nonneg γ
  nlinarith [mul_le_mul_of_nonneg_left hN hγ0]

end Slowdown.Cut.SecurityProofs
