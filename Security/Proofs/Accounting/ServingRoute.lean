import Definitions.Accounting.Guarantees.ServingRoute
import Proofs.Accounting.ServingBlocks
import Proofs.Accounting.ServingBlocks.Witnesses
import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Overhead
import Proofs.Nci.Ldl

/-!
# The serving route: proofs

`ServingPerBlockCap` is `FanOutBound`'s per-block step at one block: a block `I` of the serving budget is block `0` of
the registration that puts every other operation gate in a block of its own (`single`), every block of which fits the
budget, so Lemma 1 (`BlockDescription`) and the activation premise bound its activations, and the fan-out its counted
products.

`ServingEndToEnd` is `Overhead.EndToEnd` with the serving class as the class, the counted weight as the count and the
server-windows as the cost: on the class, `ServingBlockBound` at the uniform budget `(c_S, k_S)` with no inference
blocks gives `Σ_v ω_v ≤ windows · q_S + n_dark` (`total_le_windows`), the budget's reads from the split and the links.
-/

namespace Accounting.SecurityProofs.ServingRoute

open Verity Finset Accounting.ServingBlocks Accounting.ServingRoute
open scoped ENNReal

/-! ## The per-block cap at serving scale -/

/-- **The registration of one block**: the operation gates of `I` are block `0`, every other operation gate `g` a block
`g + 1` of its own. -/
def single (C : Circuit) (I : Finset (Fin C.N)) : Partition C (C.N + 1) where
  unit g := if C.isInput g then none else if g ∈ I then some 0 else some g.succ
  input_iff g := by by_cases h : C.isInput g <;> by_cases hg : g ∈ I <;> simp [h, hg]

theorem mem_single_gates {C : Circuit} {I : Finset (Fin C.N)} {g : Fin C.N} {u : Fin (C.N + 1)} :
    g ∈ (single C I).gates u ↔ C.isInput g = false ∧ (if g ∈ I then 0 else g.succ) = u := by
  rw [Partition.mem_gates]
  unfold single
  by_cases h : C.isInput g <;> by_cases hg : g ∈ I <;> simp [h, hg]

theorem single_gates_zero {C : Circuit} {I : Finset (Fin C.N)} (hI : I ⊆ C.computed) : (single C I).gates 0 = I := by
  ext g
  rw [mem_single_gates]
  by_cases hg : g ∈ I
  · have := hI hg
    simp only [Circuit.computed, mem_filter, mem_univ, true_and] at this
    simp [hg, this]
  · simp [hg, Fin.succ_ne_zero]

/-- Every other block is empty or one operation gate. -/
theorem single_gates_ne {C : Circuit} {I : Finset (Fin C.N)} {u : Fin (C.N + 1)} (hu : u ≠ 0) :
    (single C I).gates u = ∅ ∨ ∃ g, C.isInput g = false ∧ (single C I).gates u = {g} := by
  rcases ((single C I).gates u).eq_empty_or_nonempty with h | ⟨g, hg⟩
  · exact Or.inl h
  · refine Or.inr ⟨g, (mem_single_gates.1 hg).1, ?_⟩
    have key : ∀ g' ∈ (single C I).gates u, g' = g := fun g' hg' => by
      obtain ⟨-, h1⟩ := mem_single_gates.1 hg
      obtain ⟨-, h2⟩ := mem_single_gates.1 hg'
      by_cases a : g ∈ I <;> by_cases b : g' ∈ I <;> simp only [a, b, ite_true, ite_false] at h1 h2
      · exact absurd h1.symm hu
      · exact absurd h1.symm hu
      · exact absurd h2.symm hu
      · exact Fin.succ_injective _ (h2.trans h1.symm)
    exact eq_singleton_iff_unique_mem.2 ⟨hg, key⟩

theorem novelReads_empty (d : Declaration) : Nci.novelReads d ∅ = 0 := by
  unfold Nci.novelReads Circuit.width Circuit.into
  simp

theorem ServingPerBlockCap : Guarantees.ServingRoute.ServingPerBlockCap := by
  intro d J ι α _ M e D t aBlk ℓ y β Λ inp cS kS q hbody hω hJM hfan hcS hgate hN hβ hev hrate hq I hIc hc hk
  classical
  set R : Nci.Ldl.Registration d (d.D.N + 1) := ⟨d.D.eval inp, single d.D I⟩ with hRdef
  have hcons : R.Consistent (d.D.eval inp) := fun _ _ => rfl
  have h0 : R.P.gates 0 = I := single_gates_zero hIc
  have hsize : ∀ u, d.D.size (R.P.gates u) ≤ cS := fun u => by
    by_cases hu : u = 0
    · rw [hu, h0]; exact hc
    · rcases single_gates_ne (I := I) hu with h | ⟨g, -, h⟩
      · change d.D.size ((single d.D I).gates u) ≤ cS
        rw [h]; exact Nat.zero_le _
      · change d.D.size ((single d.D I).gates u) ≤ cS
        rw [h]; exact hcS
  have hread : Nci.Ldl.ReadCap d R.P kS := fun u => by
    by_cases hu : u = 0
    · rw [hu, h0]; exact hk
    · rcases single_gates_ne (I := I) hu with h | ⟨g, hg, h⟩
      · change Nci.novelReads d ((single d.D I).gates u) ≤ kS
        rw [h, novelReads_empty]; exact Nat.zero_le _
      · change Nci.novelReads d ((single d.D I).gates u) ≤ kS
        rw [h]
        exact hgate g (by simp [Circuit.computed, hg])
  have h1 := Nci.Ldl.SecurityProofs.BlockDescription d (d.D.N + 1) kS R (D _ R) t (fun u => y (R.P.gates u)) aBlk
    inp hread hcons (hev _ R hcons hsize hread) 0
  have h2 := hrate _ R hcons hsize hread 0
  have h3 : ENNReal.ofReal (β * (M.acts R.P 0).card - Λ) ≤
      ((kS + Nat.clog 2 (d.D.N + 1) + aBlk : ℕ) : ℝ≥0∞) :=
    h2.trans (by exact_mod_cast ENat.toENNReal_le.2 h1)
  rw [ENNReal.ofReal_le_natCast] at h3
  push_cast at h3
  have hclog : Nat.clog 2 (d.D.N + 1) ≤ ℓ := Nat.clog_le_of_le_pow hN
  have hclog' : (Nat.clog 2 (d.D.N + 1) : ℝ) ≤ ℓ := by exact_mod_cast hclog
  have hV : β * (M.acts R.P 0).card ≤ kS + ℓ + aBlk + Λ := by linarith
  have hJ : (M.inBlock R.P 0).card ≤ e * (M.acts R.P 0).card :=
    Finset.card_le_mul_card_image _ e fun j _ =>
      (Finset.card_le_card (Finset.filter_subset_filter _ (Finset.filter_subset _ _))).trans (hfan j)
  have hin : J.inside I ≤ (M.inBlock R.P 0).card := by
    unfold Counted.inside
    refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
    rw [nsmul_one, Nat.cast_le]
    refine (Finset.card_le_card fun v hv => ?_).trans (Finset.card_image_le (s := M.inBlock R.P 0) (f := M.rep))
    obtain ⟨hvJ, hvI⟩ := Finset.mem_filter.1 hv
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 (hJM hvJ)
    refine Finset.mem_image.2 ⟨p, Finset.mem_filter.2 ⟨hp, ?_⟩, rfl⟩
    rw [← Partition.mem_gates, h0]
    exact hvI (hbody _ hvJ)
  have he : (0 : ℝ) ≤ e := Nat.cast_nonneg e
  have hJ' : ((M.inBlock R.P 0).card : ℝ) ≤ e * (M.acts R.P 0).card := by exact_mod_cast hJ
  refine hin.trans (hJ'.trans ((le_div_iff₀ hβ).2 ?_ |>.trans hq))
  calc (e : ℝ) * (M.acts R.P 0).card * β = e * (β * (M.acts R.P 0).card) := by ring
    _ ≤ e * (kS + ℓ + aBlk + Λ) := mul_le_mul_of_nonneg_left hV he

/-! ## The serving route, end to end -/

/-- **The serving bound on the class**: on the serving class, the counted weight is at most the server-windows times
`q_S` plus the dark inventory. -/
theorem total_le_windows {L σ E : Type} [Fintype σ] [Fintype E] (P : Params L σ E) (p : Play L σ)
    (hcS : ∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) (h : P.Class p) :
    p.J.total ≤ p.windows * P.qS + P.nDark := by
  classical
  obtain ⟨hC, hΦ, ⟨link, hL⟩, hF, hω, hR, hcov, hdark, hq, hB, hμ, hw, hω1⟩ := h
  have hblk : ∀ S : Block p.κ, p.A.size (block p.A P.X p.κ S) ≤ P.cS ∧
      p.A.width (p.A.into (block p.A P.X p.κ S) \ p.θ) ≤ P.kS := by
    rintro ⟨s, j⟩
    have hlen : (p.κ s).τ j.succ - (p.κ s).τ j.castSucc ≤ P.ω := by have := hω s j; omega
    constructor
    · have h1 := hΦ (P.X s) ⟨s, rfl⟩ ((p.κ s).τ j.castSucc) ((p.κ s).τ j.succ)
      have hwork : p.A.work (fun _ => 1) (block p.A P.X p.κ ⟨s, j⟩) = p.A.size (block p.A P.X p.κ ⟨s, j⟩) := by
        simp [Placed.work, Circuit.size]
      rw [← hwork]
      exact h1.trans ((Nat.mul_le_mul_left _ hlen).trans (hcS s))
    · refine (SecurityProofs.Actual.width_mono p.A Finset.sdiff_subset).trans ?_
      refine (SecurityProofs.Actual.width_into_region_le p.A (P.X s) _ _
        (SecurityProofs.Actual.forward_of_causal p.A hC _)).trans ?_
      refine Nat.add_le_add (hR s j) ((SecurityProofs.Actual.ingress_le p.A hL _ _ _).trans ?_)
      exact (Nat.mul_le_mul (hF s) hlen)
  let inf : (S : Block p.κ) → Fin 0 → Finset (Fin p.A.N) := fun _ i => i.elim0
  have hD : UncertifiedCap p.J (p.A.region P.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin p.A.N)) P.nDark :=
    (SecurityProofs.ServingBlocks.Counted.outside_le_card p.J hω1 _ _).trans (by exact_mod_cast hdark)
  have hbound := SecurityProofs.ServingBlocks.ServingBlockBound p.A p.θ p.J P.q P.cS P.kS P.dark P.X p.t p.κ
    (fun _ => P.cS) (fun _ => P.kS) (fun _ => 0) inf 1 P.μ (fun S => p.J.outside (block p.A P.X p.κ S) (inf S))
    P.nDark ⟨hcov, hblk⟩ (fun _ => hq) hq (fun _ i => i.elim0) hB hμ hw zero_le_one hω1 (fun _ => le_rfl) hD
  refine hbound.trans ?_
  have hsum : ∑ S : Block p.κ, min (P.q P.cS P.kS + 1 * P.μ * P.kS)
      (((fun _ => 0 : Block p.κ → ℕ) S : ℝ) * P.q P.cS P.kS + p.J.outside (block p.A P.X p.κ S) (inf S)) ≤
      p.windows * P.qS := by
    refine (Finset.sum_le_sum fun S _ => min_le_left _ _).trans (le_of_eq ?_)
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Play.windows, Params.qS]
    ring
  linarith

theorem ServingEndToEnd : Guarantees.ServingRoute.ServingEndToEnd := by
  intro O L σ E _ _ G P play Accept Claims Success N δ δb η hcS hqS hL2 hbr hcov s
  refine Overhead.EndToEnd O G Accept Claims (fun s o => P.Class (play s o)) Success
    (fun s o => (play s o).J.total) (fun s o => ((play s o).windows : ℝ)) N (P.windowsMin N) δ δb η hL2 hbr
    (fun s o hc hN => ?_) hcov s
  have := total_le_windows P (play s o) hcS hc
  rw [Params.windowsMin, div_le_iff₀ hqS]
  linarith

end Accounting.SecurityProofs.ServingRoute
