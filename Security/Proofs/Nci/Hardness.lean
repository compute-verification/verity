import Definitions.Nci.Guarantees.Hardness
import Proofs.Nci.Ldl
import Proofs.Nci.Cover.Gates

/-!
# Priced reads: the proofs

* `bound_of_blocks`: Corollary 4b's count from `β |Ops(I)| ≤ h` at every block, fan-out and `|C| = c B`. Both
  `FanOutBoundPriced` (from `Priced`) and `fanOutBound_of_activationRate` (the LDL route, through
  `PricedOfActivationRate`) are it.
* `updateHardness_of_priced`, `priced_of_updateHardness`: on a Boolean declaration's gate view, the cover theorem's
  `UpdateHardness` is `Priced`'s cover instance with `in(S)` counted the cover theorem's way. `Priced` at block size `s`
  gives `UpdateHardness` at `s`, and `UpdateHardness` at `s` gives `Priced`'s inequality on every set `I` of gates that
  each read something, with `|I ∪ in(I)| ≤ s`: the cover theorem hands a set its boundary gates, `in(I)` here being the
  gates outside `I` that it reads.
-/

namespace Nci.Hardness

open Verity Finset Nci.Ldl
open scoped ENNReal

section OfActs

variable {C : Circuit} {ι V : Type*} (J : Finset ι) (gates : ι → Finset (Fin C.N)) (act : ι → V)

@[simp] theorem ofActs_J : (Operations.ofActs C J gates act).J = J := rfl

@[simp] theorem ofActs_gates : (Operations.ofActs C J gates act).gates = gates := rfl

@[simp] theorem ofActs_operands (p : ι) : (Operations.ofActs C J gates act).operands p = {act p} := rfl

theorem ofActs_whole (S : Finset (Fin C.N)) :
    (Operations.ofActs C J gates act).whole S = J.filter fun p => gates p ⊆ S := rfl

theorem ops_ofActs [DecidableEq V] (S : Finset (Fin C.N)) :
    (Operations.ofActs C J gates act).ops S = ((Operations.ofActs C J gates act).whole S).image act := by
  ext v
  simp [Operations.ops, eq_comm]

theorem fanOut_ofActs [DecidableEq V] {e : ℕ} (h : (Operations.ofActs C J gates act).FanOut e) (v : V) :
    (J.filter fun p => act p = v).card ≤ e := by
  have hf : (J.filter fun p => act p = v) =
      J.filter fun p => v ∈ (Operations.ofActs C J gates act).operands p :=
    Finset.filter_congr fun p _ => by rw [ofActs_operands, Finset.mem_singleton, eq_comm]
  rw [hf]
  exact h v

end OfActs

/-- **Corollary 4b's count**: from `β |Ops(I)| ≤ h` at every block of an exact partition, with every product whole in a
block and fan-out at most `e`, `ρ ≥ c β M / (T₀ e h)`. -/
theorem bound_of_blocks {d : Declaration} {n c : ℕ} {P : Partition d.D n} {ι V : Type*} [DecidableEq V]
    {J : Finset ι} {gates : ι → Finset (Fin d.D.N)} {act : ι → V} {e : ℕ} {β h T₀ : ℝ} (hc : Exact P c)
    (hwhole : (Operations.ofActs d.D J gates act).WholeIn P) (hfan : (Operations.ofActs d.D J gates act).FanOut e)
    (hβ : 0 < β) (hT : 0 < T₀) (hblk : ∀ u, β * ((Operations.ofActs d.D J gates act).ops (P.gates u)).card ≤ h) :
    c * β * J.card / (T₀ * e * h) ≤ d.D.size d.D.computed / T₀ := by
  classical
  set O := Operations.ofActs d.D J gates act with hO
  have hW : ∀ u, ((O.whole (P.gates u)).card : ℝ) ≤ e * (O.ops (P.gates u)).card := fun u => by
    rw [hO, ops_ofActs, ofActs_whole]
    exact_mod_cast Finset.card_le_mul_card_image _ e fun v _ =>
      (Finset.card_le_card (Finset.filter_subset_filter _ (Finset.filter_subset _ _))).trans
        (fanOut_ofActs J gates act hfan v)
  have hM : (J.card : ℝ) ≤ ∑ u, ((O.whole (P.gates u)).card : ℝ) := by
    have hsub : J ⊆ Finset.univ.biUnion fun u => O.whole (P.gates u) := fun p hp => by
      obtain ⟨u, hu⟩ := hwhole p hp
      refine Finset.mem_biUnion.2 ⟨u, Finset.mem_univ _, ?_⟩
      rw [hO, ofActs_whole]
      exact Finset.mem_filter.2 ⟨hp, hu⟩
    exact_mod_cast (Finset.card_le_card hsub).trans Finset.card_biUnion_le
  have hsum : β * J.card ≤ n * (e * h) := by
    calc β * J.card ≤ β * ∑ u, ((O.whole (P.gates u)).card : ℝ) := mul_le_mul_of_nonneg_left hM hβ.le
      _ = ∑ u, β * ((O.whole (P.gates u)).card : ℝ) := Finset.mul_sum _ _ _
      _ ≤ ∑ _u : Fin n, e * h := Finset.sum_le_sum fun u _ => by
          have := hW u
          have := hblk u
          nlinarith [Nat.cast_nonneg (α := ℝ) e]
      _ = n * (e * h) := by simp
  have hsize : (d.D.size d.D.computed : ℝ) = c * n := by exact_mod_cast size_eq P hc
  have hcM : c * β * J.card ≤ d.D.size d.D.computed * (e * h) := by
    rw [hsize]
    have := mul_le_mul_of_nonneg_left hsum (Nat.cast_nonneg (α := ℝ) c)
    linarith
  have hrhs : 0 ≤ (d.D.size d.D.computed : ℝ) / T₀ := div_nonneg (Nat.cast_nonneg _) hT.le
  by_cases hpos : 0 < (e : ℝ) * h
  · have hden : 0 < T₀ * e * h := by rw [mul_assoc]; exact mul_pos hT hpos
    rw [div_le_div_iff₀ hden hT]
    have : (c * β * J.card : ℝ) * T₀ ≤ d.D.size d.D.computed * (e * h) * T₀ :=
      mul_le_mul_of_nonneg_right hcM hT.le
    linarith
  · refine le_trans ?_ hrhs
    apply div_nonpos_of_nonneg_of_nonpos
    · have := Nat.cast_nonneg (α := ℝ) c
      have := Nat.cast_nonneg (α := ℝ) J.card
      positivity
    · have : T₀ * e * h = T₀ * (e * h) := by ring
      rw [this]
      exact mul_nonpos_of_nonneg_of_nonpos hT.le (not_lt.1 hpos)

namespace SecurityProofs

theorem FanOutBoundPriced : Guarantees.FanOutBoundPriced := by
  intro d n c k P ι V _ J gates act e β Λ T₀ hck hwhole hfan hβ hT hpr
  obtain ⟨hc, hk⟩ := hck
  refine bound_of_blocks hc hwhole hfan hβ hT fun u => ?_
  have h1 := hpr (P.gates u) (P.gates_subset_computed u) (hc u).le
  have h2 : (novelReads d (P.gates u) : ℝ) ≤ k := by exact_mod_cast hk u
  simp only [Finset.sum_const, nsmul_eq_mul, Finset.union_empty] at h1
  unfold novelReads at h2
  linarith

theorem PricedOfActivationRate : Guarantees.PricedOfActivationRate := by
  intro d n R ι A _ M gates D t aBlk a β Λ inp hrep hβ hX hev hrate u
  obtain ⟨enc, henc⟩ := hev
  obtain ⟨-, hlen, hdec⟩ := henc u _ hX
  rw [localEval_eval] at hdec
  have h1 : LDL D t (a u (d.D.eval inp)) ≤ (Nat.clog 2 n + novelReads d (R.P.gates u) + aBlk : ℕ) :=
    (LDL_le hdec).trans (by exact_mod_cast hlen)
  have h2 : ENNReal.ofReal (β * (M.acts R.P u).card - Λ) ≤
      ((Nat.clog 2 n + novelReads d (R.P.gates u) + aBlk : ℕ) : ℝ≥0∞) :=
    (hrate u).trans (by exact_mod_cast ENat.toENNReal_le.2 h1)
  rw [ENNReal.ofReal_le_natCast] at h2
  push_cast at h2
  have hsub : (Operations.ofActs d.D M.J gates M.act).ops (R.P.gates u) ⊆ M.acts R.P u := by
    rw [ops_ofActs]
    refine Finset.image_subset_image fun p hp => ?_
    obtain ⟨hpJ, hpg⟩ := Finset.mem_filter.1 hp
    exact Finset.mem_filter.2 ⟨hpJ, (R.P.mem_gates).1 (hpg (hrep p hpJ))⟩
  have hcard : (((Operations.ofActs d.D M.J gates M.act).ops (R.P.gates u)).card : ℝ) ≤ (M.acts R.P u).card := by
    exact_mod_cast Finset.card_le_card hsub
  simp only [Finset.sum_const, nsmul_eq_mul]
  nlinarith

end SecurityProofs

/-- **The LDL route is a corollary**: Corollary 4b from `ActivationRate` and `EvaluatesBlocks` (with every product
whole in a block) is `bound_of_blocks` on `PricedOfActivationRate`'s inequality, at `h = k + Λ + ⌈log₂ B⌉ + aBlk`. -/
theorem fanOutBound_of_activationRate {d : Declaration} {n c k : ℕ} {R : Registration d n} {ι A : Type}
    [DecidableEq A] {M : Products d.D ι A} {gates : ι → Finset (Fin d.D.N)} {e : ℕ} {D : Decoder} {t aBlk : ℕ}
    {a : Fin n → d.D.Transcript → List Bool} {β Λ T₀ : ℝ} {inp : d.D.Transcript} (hck : CKPartition d R.P c k)
    (hrep : ∀ p ∈ M.J, M.rep p ∈ gates p) (hwhole : (Operations.ofActs d.D M.J gates M.act).WholeIn R.P)
    (hfan : M.FanOut e) (hβ : 0 < β) (hT : 0 < T₀) (hX : R.Consistent (d.D.eval inp))
    (hev : Ldl.Assumptions.EvaluatesBlocks D t R a aBlk)
    (hrate : Ldl.Assumptions.ActivationRate β Λ t D R M a (d.D.eval inp)) :
    c * β * M.J.card / (T₀ * e * (k + (Λ + Nat.clog 2 n + aBlk))) ≤ d.D.size d.D.computed / T₀ := by
  obtain ⟨hc, hk⟩ := hck
  have hfan' : (Operations.ofActs d.D M.J gates M.act).FanOut e := fun v => by
    have hf : (M.J.filter fun p => v ∈ (Operations.ofActs d.D M.J gates M.act).operands p) =
        M.J.filter fun p => M.act p = v :=
      Finset.filter_congr fun p _ => by rw [ofActs_operands, Finset.mem_singleton, eq_comm]
    show (M.J.filter fun p => v ∈ (Operations.ofActs d.D M.J gates M.act).operands p).card ≤ e
    rw [hf]
    exact hfan v
  refine bound_of_blocks hc hwhole hfan' hβ hT fun u => ?_
  have h1 := SecurityProofs.PricedOfActivationRate d n R M gates D t aBlk a β Λ inp hrep hβ.le hX hev hrate u
  have h2 : (novelReads d (R.P.gates u) : ℝ) ≤ k := by exact_mod_cast hk u
  simp only [Finset.sum_const, nsmul_eq_mul] at h1
  linarith

section Cover

open Nci.Cover.SecurityProofs

/-- The cover theorem's products on a declaration's gate view as counted operations: all products, each pricing its
updated parameter. -/
def coverOps {d : Declaration} {J L : Type*} (M : Nci.Cover.MulBlocks (ofGates d) J L) : Operations d.D (J × L) L :=
  ⟨M.Pall, M.m, fun p => {p.2}⟩

theorem coverOps_ops {d : Declaration} {J L : Type*} [DecidableEq L] (M : Nci.Cover.MulBlocks (ofGates d) J L)
    (S : Finset (Fin d.D.N)) : (coverOps M).ops S = (M.Pall.filter fun p => M.m p ⊆ S).image Prod.snd := by
  ext l
  simp only [Operations.ops, Operations.whole, coverOps, Finset.mem_biUnion, Finset.mem_singleton, Finset.mem_image]
  constructor
  · rintro ⟨p, hp, rfl⟩
    exact ⟨p, hp, rfl⟩
  · rintro ⟨p, hp, rfl⟩
    exact ⟨p, hp, rfl⟩

theorem width_eq_card {d : Declaration} (hw : ∀ g, d.D.w g = 1) (X : Finset (Fin d.D.N)) : d.D.width X = X.card := by
  simp [Circuit.width, hw]

/-- **`Priced` gives `UpdateHardness`** on a Boolean declaration's gate view, at the same block size and price: a set
`S` computes what its operation gates `S \ in_cover(S)` compute, and those read only gates of `in_cover(S)` outside
themselves. -/
theorem updateHardness_of_priced {d : Declaration} (hw : ∀ g, d.D.w g = 1) {J L : Type*} [DecidableEq L]
    (M : Nci.Cover.MulBlocks (ofGates d) J L) (s : ℕ) (b : ℝ)
    (h : Assumptions.Priced d (coverOps M) (fun _ => b) 0 s (fun _ => ∅)) :
    Nci.Cover.Assumptions.UpdateHardness M s b := by
  intro S hS B hB hBS
  set I := (ofGates d).computed S with hI
  have hIc : I ⊆ d.D.computed := fun g hg => by
    obtain ⟨-, hgi⟩ := mem_computed.1 hg
    have hr : d.D.reads g ≠ ∅ := fun h0 => hgi (mem_inputs.2 ⟨(mem_computed.1 hg).1, Or.inl h0⟩)
    simp only [Circuit.computed, Finset.mem_filter, Finset.mem_univ, true_and]
    cases hgin : d.D.isInput g
    · rfl
    · exact absurd (d.D.input_reads g hgin) hr
  have hIs : d.D.size I ≤ s := (Finset.card_le_card (Finset.sdiff_subset)).trans hS
  have hinto : d.D.into I ⊆ (ofGates d).inputs S := fun x hx => by
    obtain ⟨hxI, g, hgI, hxg⟩ := (Finset.mem_filter.1 hx).2
    obtain ⟨hgS, hgi⟩ := mem_computed.1 hgI
    have hpS : d.D.reads g ⊆ S := by
      by_contra hn
      exact hgi (mem_inputs.2 ⟨hgS, Or.inr hn⟩)
    have hxS := hpS hxg
    by_contra hxin
    exact hxI (mem_computed.2 ⟨hxS, hxin⟩)
  have h1 := h I hIc hIs
  simp only [Finset.sum_const, nsmul_eq_mul, Finset.union_empty, sub_zero, width_eq_card hw] at h1
  have hBops : B.image Prod.snd ⊆ (coverOps M).ops I := by
    rw [coverOps_ops]
    exact Finset.image_subset_image fun p hp => Finset.mem_filter.2 ⟨hB hp, hBS p hp⟩
  have h3 : ((d.D.into I \ d.θ).card : ℝ) ≤ ((ofGates d).inputs S \ (ofGates d).θ).card := by
    exact_mod_cast Finset.card_le_card (Finset.sdiff_subset_sdiff hinto subset_rfl)
  rcases le_or_gt 0 b with hb | hb
  · have h4 : b * (B.image Prod.snd).card ≤ b * ((coverOps M).ops I).card :=
      mul_le_mul_of_nonneg_left (by exact_mod_cast Finset.card_le_card hBops) hb
    linarith
  · exact (mul_nonpos_of_nonpos_of_nonneg hb.le (Nat.cast_nonneg _)).trans (Nat.cast_nonneg _)

/-- **`UpdateHardness` gives `Priced`'s inequality** on a Boolean declaration's gate view, on every set `I` of
operation gates that each read something, with `|I ∪ in(I)| ≤ s`: the cover theorem's set `I ∪ in(I)` computes `I`
and is handed only gates of `in(I)`. -/
theorem priced_of_updateHardness {d : Declaration} (hw : ∀ g, d.D.w g = 1) {J L : Type*} [DecidableEq L]
    (M : Nci.Cover.MulBlocks (ofGates d) J L) (s : ℕ) (b : ℝ) (h : Nci.Cover.Assumptions.UpdateHardness M s b)
    (I : Finset (Fin d.D.N)) (hread : ∀ g ∈ I, (d.D.reads g).Nonempty) (hs : (I ∪ d.D.into I).card ≤ s) :
    ∑ _v ∈ (coverOps M).ops I, b - 0 ≤ d.D.width (d.D.into I \ (d.θ ∪ ∅)) := by
  have hcomp : I ⊆ (ofGates d).computed (I ∪ d.D.into I) := fun g hg => by
    refine mem_computed.2 ⟨Finset.mem_union_left _ hg, fun hin => ?_⟩
    rcases (mem_inputs.1 hin).2 with h0 | hn
    · exact (hread g hg).ne_empty h0
    · refine hn fun x hx => ?_
      by_cases hxI : x ∈ I
      · exact Finset.mem_union_left _ hxI
      · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hxI, g, hg, hx⟩)
  have hins : (ofGates d).inputs (I ∪ d.D.into I) ⊆ d.D.into I := fun x hx => by
    obtain ⟨hxU, hx'⟩ := mem_inputs.1 hx
    rcases Finset.mem_union.1 hxU with hxI | hxi
    · exact absurd hx (mem_computed.1 (hcomp hxI)).2
    · exact hxi
  have h1 := h (I ∪ d.D.into I) hs (M.Pall.filter fun p => M.m p ⊆ I) (Finset.filter_subset _ _)
    fun p hp => (Finset.mem_filter.1 hp).2.trans hcomp
  have h2 : (((ofGates d).inputs (I ∪ d.D.into I) \ (ofGates d).θ).card : ℝ) ≤ (d.D.into I \ d.θ).card := by
    exact_mod_cast Finset.card_le_card (Finset.sdiff_subset_sdiff hins subset_rfl)
  simp only [Finset.sum_const, nsmul_eq_mul, Finset.union_empty, sub_zero, width_eq_card hw, coverOps_ops]
  linarith

end Cover

end Nci.Hardness
