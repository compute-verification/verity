import Definitions.Accounting.Guarantees.ServingRoute
import Proofs.Accounting.ServingBlocks
import Proofs.Accounting.ServingBlocks.Witnesses
import Proofs.Nci.CompliesCK
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
server-windows as the cost: on the class, the serving-block bound's serving cap (`assigned_le_serving`) at the uniform
budget `(c_S, k_S)` at each serving block, and the dark inventory at the rest, summed by `ServingBlockSum`, give
`Σ_v ω_v ≤ windows · q_S + n_dark` (`total_le_windows`), the budget's reads from the split and the links.
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

/-- On a Boolean declaration, a block's novel reads are its unregistered input gates. -/
theorem novelReads_eq_card {d : Declaration} (hd : d.D.Boolean) (S : Finset (Fin d.D.N)) :
    Nci.novelReads d S = (d.unregistered S).card := by
  unfold Nci.novelReads Declaration.unregistered
  exact Circuit.width_eq_card hd _

theorem ServingPerBlockCap : Guarantees.ServingRoute.ServingPerBlockCap := by
  intro d J ι α _ M e D t aBlk ℓ y β Λ inp cS kS q hbool hω hJM hfan hcS hgate hN hβ hev hrate hq I hIc hc hk
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
    rw [novelReads_eq_card hbool]
    by_cases hu : u = 0
    · rw [hu, h0]; exact hk
    · rcases single_gates_ne (I := I) hu with h | ⟨g, hg, h⟩
      · change (d.unregistered ((single d.D I).gates u)).card ≤ kS
        rw [h, ← novelReads_eq_card hbool, novelReads_empty]; exact Nat.zero_le _
      · change (d.unregistered ((single d.D I).gates u)).card ≤ kS
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
    exact hvI (J.rep_mem _ hvJ)
  have he : (0 : ℝ) ≤ e := Nat.cast_nonneg e
  have hJ' : ((M.inBlock R.P 0).card : ℝ) ≤ e * (M.acts R.P 0).card := by exact_mod_cast hJ
  refine hin.trans (hJ'.trans ((le_div_iff₀ hβ).2 ?_ |>.trans hq))
  calc (e : ℝ) * (M.acts R.P 0).card * β = e * (β * (M.acts R.P 0).card) := by ring
    _ ≤ e * (kS + ℓ + aBlk + Λ) := mul_le_mul_of_nonneg_left hV he

/-! ## The serving route, end to end -/

/-- With every weight at most 1, what blocks leave of a set of gates is at most its number of gates. -/
theorem outside_le_card {C : Circuit} (J : Counted C) (hω : ∀ v ∈ J.V, J.ω v ≤ 1) (R : Finset (Fin C.N)) {ι : Type*}
    [Fintype ι] (B : ι → Finset (Fin C.N)) : J.outside R B ≤ R.card := by
  unfold Counted.outside
  refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
  rw [nsmul_one, Nat.cast_le]
  exact Finset.card_le_card fun v h => (Finset.mem_filter.1 h).2.1

/-- **The serving bound on the class**: on the serving class, the counted weight is at most the server-windows times
`q_S` plus the dark inventory. -/
theorem total_le_windows {L σ E : Type} [Fintype E] (P : Params L σ E) (p : Play L)
    (hcS : ∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) (h : P.Class p) :
    p.J.total ≤ p.windows * P.qS + P.nDark := by
  classical
  obtain ⟨hC, hΦ, ⟨link, hL⟩, hF, hsrv, hdk, hdark, hq, hB, hμ, hw, hω1⟩ := h
  have hS : ∀ u ∈ p.S, ∑ v ∈ p.J.V ∩ p.P.gates u, p.J.ω v ≤ P.q P.cS P.kS + 1 * P.μ * P.kS := fun u hu => by
    obtain ⟨s, a, hst, hgu⟩ := hsrv u hu
    rw [SecurityProofs.ServingBlocks.Counted.sum_inter]
    refine SecurityProofs.ServingBlocks.assigned_le_serving p.decl p.J P.q P.cS P.kS (p.P.gates u) 1 P.μ hq
      (p.P.gates_subset_computed u) ?_ ?_ hB hμ zero_le_one hω1
    · have h1 := hΦ (P.X s) ⟨s, rfl⟩ a (a + P.ω)
      have hwork : p.A.work (fun _ => 1) (p.A.region (P.X s) (Set.Ico a (a + P.ω))) =
          (p.A.region (P.X s) (Set.Ico a (a + P.ω))).card := by simp [Placed.work]
      rw [hwork, Nat.add_sub_cancel_left] at h1
      rw [hgu]
      exact h1.trans (hcS s)
    · rw [hgu]
      calc (p.decl.unregistered (p.A.region (P.X s) (Set.Ico a (a + P.ω)))).card
          ≤ p.A.width (p.decl.unregistered (p.A.region (P.X s) (Set.Ico a (a + P.ω)))) :=
            Circuit.card_le_width _ _ fun g _ => hw g
        _ ≤ p.A.width (p.A.into (p.A.region (P.X s) (Set.Ico a (a + P.ω)))) :=
            SecurityProofs.Actual.width_mono p.A Finset.sdiff_subset
        _ ≤ p.A.state (P.X s) a + p.A.ingress (P.X s) (Set.Ico a (a + P.ω)) :=
            SecurityProofs.Actual.width_into_region_le p.A (P.X s) a (a + P.ω)
              (SecurityProofs.Actual.forward_of_causal p.A hC _)
        _ ≤ P.R + P.F * P.ω :=
            Nat.add_le_add hst ((SecurityProofs.Actual.ingress_le p.A hL _ _ _).trans
              (Nat.mul_le_mul (hF s) (by omega)))
  have hD : ∑ u ∈ univ \ p.S, ∑ v ∈ p.J.V ∩ p.P.gates u, p.J.ω v ≤ P.nDark := by
    refine SecurityProofs.ServingBlocks.dark_le p.J p.P _ ((outside_le_card p.J hω1 _ _).trans ?_)
    have hsub : (univ \ p.S).biUnion p.P.gates ⊆ p.A.region P.dark Set.univ :=
      Finset.biUnion_subset.2 fun u hu => hdk u (Finset.mem_sdiff.1 hu).2
    exact_mod_cast (Finset.card_le_card hsub).trans hdark
  have hsum := Verity.SecurityProofs.ServingBlockSum p.A.toCircuit p.B p.P p.S p.J.V p.J.ω
    (fun _ => P.q P.cS P.kS + 1 * P.μ * P.kS) (fun _ => P.q P.cS P.kS + 1 * P.μ * P.kS) P.nDark
    (SecurityProofs.ServingBlocks.Counted.V_subset p.J) hS hS hD
  refine hsum.trans (le_of_eq ?_)
  simp only [min_self, Finset.sum_const, nsmul_eq_mul, Play.windows, Params.qS]
  ring

theorem ServingEndToEnd : Guarantees.ServingRoute.ServingEndToEnd := by
  intro O L σ E _ G P play Accept Claims Success N δ δb η hcS hqS hL2 hbr hcov s
  refine Overhead.EndToEnd O G Accept Claims (fun s o => P.Class (play s o)) Success
    (fun s o => (play s o).J.total) (fun s o => ((play s o).windows : ℝ)) N (P.windowsMin N) δ δb η hL2 hbr
    (fun s o hc hN => ?_) hcov s
  have := total_le_windows P (play s o) hcS hc
  rw [Params.windowsMin, div_le_iff₀ hqS]
  linarith

end Accounting.SecurityProofs.ServingRoute
