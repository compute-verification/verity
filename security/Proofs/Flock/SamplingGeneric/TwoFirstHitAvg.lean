module

public import Proofs.Flock.SamplingGeneric.TwoFirstHitAt
public import Proofs.Flock.SamplingGeneric.RerunCalls
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.Claims.Sampling.Extract

/-! The first-hit bound's expected calls from the prover's average calls, not a budget on every run. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling

theorem Ex_tape_coins {Ω R C : Type} [Fintype Ω] [Fintype R] [Fintype C] [Nonempty C] (f : Ω → R → ℝ) :
    Ex (fun q : Ω × (R × C) => f q.1 q.2.1) = Ex fun q : Ω × R => f q.1 q.2 := by
  rw [Verity.FirstHit.ex_tape fun (ω : Ω) (rc : R × C) => f ω rc.1, Verity.FirstHit.ex_tape f]
  exact congrArg Ex (funext fun ω => Ex_fst (f ω))

theorem Ex_add_four {α : Type} [Fintype α] (f₁ f₂ f₃ f₄ : α → ℝ) :
    Ex (fun a => f₁ a + f₂ a + f₃ a + f₄ a) = Ex f₁ + Ex f₂ + Ex f₃ + Ex f₄ := by
  rw [Ex_add (fun a => f₁ a + f₂ a + f₃ a) f₄, Ex_add (fun a => f₁ a + f₂ a) f₃, Ex_add f₁ f₂]

theorem Ex_mul_le_of_le {Q : Type} [Fintype Q] [Nonempty Q] {cnt pc : Q → ℝ} {N T : ℝ} (hc0 : ∀ q, 0 ≤ cnt q)
    (hc : ∀ q, cnt q ≤ N) (hp : ∀ q, 0 ≤ pc q) (hT : Ex pc ≤ T) : Ex (fun q => cnt q * pc q) ≤ N * T := by
  have hN : 0 ≤ N := (hc0 (Classical.arbitrary Q)).trans (hc _)
  exact ((Ex_mono fun q => mul_le_mul_of_nonneg_right (hc q) (hp q)).trans_eq (Ex_mul_left N pc)).trans
    (mul_le_mul_of_nonneg_left hT hN)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- Its extractions' calls are not counted. -/
def partiesCalls (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) : ℕ :=
  (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length

noncomputable def runCostAvg (L : Compiled Coin A) (x : Msg) (T As : ℝ) : ℝ :=
  T + L.es x * As + (L.nPU x : ℝ) * L.openCalls x

noncomputable def firstHitCostAvg (L : Compiled Coin A) (x : Msg) (T As Ak Am Ks : ℝ) : ℝ :=
  T + L.es x * As + Ak + L.es x * Am + (1 + Ks) * ((L.nPU x : ℝ) * L.openCalls x)

theorem callsAt_eq (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    L.callsAt H P x s ω r c = L.partiesCalls H P x ω r +
        (ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length +
      (((L.runExt x s r (run H P L.V x ω r).seen c).rerunsAt H (run H P L.V x ω)).map
        (L.partiesCalls H P x ω)).sum := by
  have h := calls_eq_own H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)
  unfold callsAt
  rw [h, ← Nat.add_assoc]
  rfl

theorem callsBy_eq (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (needs : ℕ → Finset ℕ) (g : ℕ) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    L.callsBy H P x s needs g ω r c = L.partiesCalls H P x ω r +
        (ownLog H (run H P L.V x ω) (L.runExtAt x s needs g r (run H P L.V x ω r).seen c)).length +
      (((L.runExtAt x s needs g r (run H P L.V x ω r).seen c).rerunsAt H (run H P L.V x ω)).map
        (L.partiesCalls H P x ω)).sum := by
  have h := calls_eq_own H (run H P L.V x ω) (L.runExtAt x s needs g r (run H P L.V x ω r).seen c)
  unfold callsBy
  rw [h, ← Nat.add_assoc]
  rfl

noncomputable def slotCost (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (W : (Fin (L.V.k x) → Coin) → ℕ) (i : Fin (L.nPU x))
    (r : Fin (L.V.k x) → Coin) (ρ : L.BlockEC x) : ℝ :=
  (((L.checked x r)[i.val]?).map fun u =>
    if hu : u < L.nPU x ∧ (L.sessSlots x (L.checked x r) i).head? = some i.val then
      ((((L.sessExt x s (L.sessBlocks x (L.checked x r) i) u r (run' r).seen (ρ ⟨u, hu.1⟩)).rerunsAt H
        run').map W).sum : ℝ)
    else 0).getD 0

theorem runExt_cost_eq (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (W : (Fin (L.V.k x) → Coin) → ℕ) (r : Fin (L.V.k x) → Coin)
    (c : L.SlotEC x) :
    ((((L.runExt x s r (run' r).seen c).rerunsAt H run').map W).sum : ℝ) =
      ∑ i, L.slotCost H x s run' W i r (c i) := by
  unfold runExt
  rw [flatMapRP_rerunsAt_sum, Nat.cast_list_sum, List.map_map]
  refine Eq.trans ?_ (sum_zip_finRange (L.nPU x) (fun iu =>
    if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
      ((((L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen
        (c iu.1 ⟨iu.2, hu.1⟩)).rerunsAt H run').map W).sum : ℝ)
    else 0) (L.checked x r))
  refine congrArg List.sum (List.map_congr_left fun iu _ => ?_)
  simp only [Function.comp_apply]
  split
  · rfl
  · exact Nat.cast_zero

theorem slotCost_none (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (W : (Fin (L.V.k x) → Coin) → ℕ) (i : Fin (L.nPU x))
    {r : Fin (L.V.k x) → Coin} (h : L.selSlot x (L.firstSel x i) r = none) (ρ : L.BlockEC x) :
    L.slotCost H x s run' W i r ρ = 0 := by
  unfold slotCost
  cases hr : (L.checked x r)[i.val]? with
  | none => rfl
  | some u =>
    rw [Option.map_some, Option.getD_some]
    by_cases hu : u < L.nPU x ∧ (L.sessSlots x (L.checked x r) i).head? = some i.val
    · have hs := selSlot_eq_some.2 (show L.firstSel x i (L.checked x r) = some i ∧
        (L.checked x r)[i.val]? = some (⟨u, hu.1⟩ : Fin (L.nPU x)).val from ⟨ite_eq_left hu.2, hr⟩)
      rw [h] at hs
      exact (Option.some_ne_none _ hs.symm).elim
    · exact dite_eq_right hu

theorem slotCost_some (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (W : (Fin (L.V.k x) → Coin) → ℕ) {i u : Fin (L.nPU x)}
    {r : Fin (L.V.k x) → Coin} (h : (L.checked x r)[i.val]? = some u.val)
    (hf : (L.sessSlots x (L.checked x r) i).head? = some i.val) (ρ : L.BlockEC x) :
    L.slotCost H x s run' W i r ρ =
      ((((L.sessExt x s (L.sessBlocks x (L.checked x r) i) u r (run' r).seen (ρ u)).rerunsAt H run').map
        W).sum : ℝ) := by
  unfold slotCost
  rw [h, Option.map_some, Option.getD_some]
  exact dite_eq_left ⟨u.isLt, hf⟩

theorem sum_runExtAt_cost_le_in (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (W : (Fin (L.V.k x) → Coin) → ℕ) (needs : ℕ → Finset ℕ)
    (S : Finset (Fin (L.nG x))) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    ((∑ g ∈ S, (((L.runExtAt x s needs g.val r (run' r).seen c).rerunsAt H run').map W).sum : ℕ) : ℝ) ≤
      ∑ i : Fin (L.nPU x),
        (((S.filter fun g : Fin (L.nG x) =>
            decide (∃ u ∈ L.sessBlocks x (L.checked x r) i.val, g.val ∈ needs u)).card : ℕ) : ℝ) *
          L.slotCost H x s run' W i r (c i) := by
  have hmid := sum_filter_map_card S
    (fun (g : Fin (L.nG x)) (iu : Fin (L.nPU x) × ℕ) =>
      decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u))
    (fun iu : Fin (L.nPU x) × ℕ =>
      (((if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
          L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)
        else .ret []).rerunsAt H run').map W).sum)
    ((List.finRange (L.nPU x)).zip (L.checked x r))
  have hsum : ∑ g ∈ S, (((L.runExtAt x s needs g.val r (run' r).seen c).rerunsAt H run').map W).sum =
      (((List.finRange (L.nPU x)).zip (L.checked x r)).map fun iu : Fin (L.nPU x) × ℕ =>
        (S.filter fun g : Fin (L.nG x) =>
            decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u)).card *
          (((if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
              L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)
            else .ret []).rerunsAt H run').map W).sum).sum := by
    refine Eq.trans (Finset.sum_congr rfl fun g _ => ?_) hmid
    unfold runExtAt
    exact flatMapRP_rerunsAt_sum H run' W _ _
  rw [hsum, Nat.cast_list_sum, List.map_map]
  refine le_of_eq_of_le (Eq.trans ?_ (sum_zip_finRange (L.nPU x) (fun iu : Fin (L.nPU x) × ℕ =>
    (((S.filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u)).card : ℕ) : ℝ) *
      if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
        ((((L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen
          (c iu.1 ⟨iu.2, hu.1⟩)).rerunsAt H run').map W).sum : ℝ)
      else 0) (L.checked x r))) (Finset.sum_le_sum fun i _ => ?_)
  · refine congrArg List.sum (List.map_congr_left fun iu _ => ?_)
    simp only [Function.comp_apply, Nat.cast_mul]
    congr 1
    split
    · rfl
    · exact Nat.cast_zero
  · unfold slotCost
    cases hl : (L.checked x r)[i.val]? with
    | none => simp
    | some u =>
      simp only [Option.map_some, Option.getD_some]
      refine mul_le_mul_of_nonneg_right le_rfl ?_
      split
      · exact Nat.cast_nonneg _
      · exact le_rfl

variable {H : Msg → Msg} {L : Compiled Coin A}

theorem slot_cost_le_weighted (hS : L.CostContract H) {x : Msg} (s : AuditStatement) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (W : (Fin (L.V.k x) → Coin) → ℕ) (i : Fin (L.nPU x)) (w : List ℕ → ℝ) (hw : ∀ l, 0 ≤ w l)
    (r : Fin (L.V.k x) → Coin) :
    Ex (fun d : Fin (L.V.k x) → Coin => w (L.checked x (forkOn (L.secondStage x) r d)) *
        Ex fun ρ : L.BlockEC x => L.slotCost H x s (run H P L.V x ω) W i (forkOn (L.secondStage x) r d) ρ) ≤
      Ex fun d : Fin (L.V.k x) → Coin => w (L.checked x (forkOn (L.secondStage x) r d)) *
        if (L.sessSlots x (L.checked x (forkOn (L.secondStage x) r d)) i).head? = some i.val then
          L.es x * (W (forkOn (L.secondStage x) r d) : ℝ) else 0 := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hresL := Ex_resample (fun d => L.selRounds x (L.firstSel x i) (forkOn (L.secondStage x) r d))
    (selRounds_resample hS.toRunsFixed (L.firstSel x i) r) fun d =>
      w (L.checked x (forkOn (L.secondStage x) r d)) *
        Ex fun ρ : L.BlockEC x => L.slotCost H x s (run H P L.V x ω) W i (forkOn (L.secondStage x) r d) ρ
  have hresR := Ex_resample (fun d => L.selRounds x (L.firstSel x i) (forkOn (L.secondStage x) r d))
    (selRounds_resample hS.toRunsFixed (L.firstSel x i) r) fun d =>
      w (L.checked x (forkOn (L.secondStage x) r d)) *
        if (L.sessSlots x (L.checked x (forkOn (L.secondStage x) r d)) i).head? = some i.val then
          L.es x * (W (forkOn (L.secondStage x) r d) : ℝ) else 0
  beta_reduce at hresL hresR
  rw [← hresL, ← hresR]
  refine Ex_mono fun d => ?_
  cases h : L.selSlot x (L.firstSel x i) (forkOn (L.secondStage x) r d) with
  | none =>
    rw [selRounds_none h]
    simp only [forkOn_empty, slotCost_none H L x s (run H P L.V x ω) W i h, Ex_const, mul_zero]
    refine mul_nonneg (hw _) ?_
    split_ifs
    · exact mul_nonneg he0 (Nat.cast_nonneg _)
    · exact le_rfl
  | some iu =>
    obtain ⟨i', u⟩ := iu
    obtain ⟨hsel, hm⟩ := selSlot_eq_some.1 h
    have hfi : (L.sessSlots x (L.checked x (forkOn (L.secondStage x) r d)) i).head? = some i.val ∧ i' = i := by
      unfold firstSel at hsel
      by_cases hf : (L.sessSlots x (L.checked x (forkOn (L.secondStage x) r d)) i).head? = some i.val
      · rw [ite_eq_left hf] at hsel
        exact ⟨hf, (Option.some.inj hsel).symm⟩
      · rw [ite_eq_right hf] at hsel
        cases hsel
    rw [hfi.2] at h hm
    have hmem : u.val ∈ L.checked x (forkOn (L.secondStage x) r d) := List.mem_of_getElem? hm
    have hsub : ∀ σ : Fin (L.V.k x) → Coin, forkOn (L.secondStage x) r (forkOn (L.rounds x u) d σ) =
        forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ := forkOn_sub (hS.fixed x _ u hmem).1 r d
    have hcσ : ∀ σ : Fin (L.V.k x) → Coin,
        L.checked x (forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ) = L.checked x (forkOn (L.secondStage x) r d) :=
      (hS.fixed x _ u hmem).2
    rw [selRounds_some h]
    simp only [hsub, hcσ, ite_eq_left hfi.1]
    rw [Ex_mul_left, Ex_mul_left, Ex_mul_left]
    refine mul_le_mul_of_nonneg_left ?_ (hw _)
    obtain ⟨xS, P', hshape, -, -, hcard, -, hview⟩ := hS.runs x Ω P ω (forkOn (L.secondStage x) r d) u hmem
    have hag : ∀ σ : Fin (L.V.k x) → Coin, ∀ t ∉ L.rounds x u,
        forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ t = forkOn (L.secondStage x) r d t :=
      fun σ t ht => by simp [forkOn, ht]
    have hunit : ∀ σ (ρ : L.BlockEC x),
        L.slotCost H x s (run H P L.V x ω) W i (forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ) ρ =
          (((A.ext xS (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) (ρ u))).rerunsAt H
            (run H P' A.V xS ())).map fun d' => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).sum := by
      intro σ ρ
      have hmσ : (L.checked x (forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ))[i.val]? = some u.val := by
        rw [hcσ σ]
        exact hm
      have hfσ : (L.sessSlots x (L.checked x (forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ)) i).head? =
          some i.val := by
        rw [hcσ σ]
        exact hfi.1
      rw [slotCost_some H L x s _ W hmσ hfσ ρ, L.sessExt_rerunsAt H P ω P' (forkOn (L.secondStage x) r d) hshape hcard
        (fun r' h => (hview r' h).1) (hag σ) s _ (ρ u), L.coinsOf_forkOn hcard, List.map_map, Nat.cast_list_sum,
        List.map_map]
      refine congrArg List.sum (List.map_congr_left fun d' _ => ?_)
      simp only [Function.comp_apply]
      rw [L.place_congr hcard (hag σ) d']
    calc Ex (fun σ : Fin (L.V.k x) → Coin => Ex fun ρ : L.BlockEC x =>
            L.slotCost H x s (run H P L.V x ω) W i (forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ) ρ)
        = Ex (fun σ : Fin (L.V.k x) → Coin => Ex fun c : A.EC (L.sh x u) =>
            (((A.ext xS (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) c)).rerunsAt H
              (run H P' A.V xS ())).map
                fun d' => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).sum) := by
          refine congrArg Ex (funext fun σ => ?_)
          simp only [hunit σ]
          exact Ex_pi_eval_dep (E := fun v : Fin (L.nPU x) => A.EC (L.sh x v)) u fun c : A.EC (L.sh x u) =>
            (((A.ext xS (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) c)).rerunsAt H
              (run H P' A.V xS ())).map
                fun d' => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).sum
      _ = Ex (fun p : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
            (((A.ext xS (fun t => p.1 (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) p.2)).rerunsAt H
              (run H P' A.V xS ())).map
                fun d' => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).sum) :=
          (Ex_prod fun p : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
            (((A.ext xS (fun t => p.1 (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) p.2)).rerunsAt H
              (run H P' A.V xS ())).map
                fun d' => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).sum).symm
      _ = Ex (fun p : (Fin (A.V.k xS) → Coin) × A.EC (A.shape xS) =>
            (((A.ext xS p.1 p.2).rerunsAt H (run H P' A.V xS ())).map
              fun d' => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).sum) :=
          Ex_restrict_shape (L.emb x u xS hcard) (L.emb x u xS hcard).injective hshape
            fun ρ' c' => (((A.ext xS ρ' c').rerunsAt H (run H P' A.V xS ())).map
              fun d' => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).sum
      _ ≤ (1 + A.waits (A.shape xS) * A.k (A.shape xS)) *
            Ex (fun d' : Fin (A.V.k xS) → Coin => (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)) :=
          A.rerunsCostExp_le H xS (run H P' A.V xS ()) fun _ => Nat.cast_nonneg _
      _ = (1 + A.waits (L.sh x u) * A.k (L.sh x u)) *
            Ex (fun σ : Fin (L.V.k x) → Coin =>
              (W (forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ) : ℝ)) := by
          rw [hshape]
          refine congrArg (fun z => (1 + (A.waits (L.sh x u) : ℝ) * A.k (L.sh x u)) * z) ?_
          refine Eq.trans (Ex_comp_inj (L.emb x u xS hcard) (L.emb x u xS hcard).injective
            fun d' : Fin (A.V.k xS) → Coin =>
              (W (L.place x u xS (forkOn (L.secondStage x) r d) d') : ℝ)).symm ?_
          exact congrArg Ex (funext fun σ =>
            congrArg (fun z => (W z : ℝ)) (L.place_emb hcard (forkOn (L.secondStage x) r d) σ))
      _ ≤ L.es x * Ex (fun σ : Fin (L.V.k x) → Coin =>
            (W (forkOn (L.rounds x u) (forkOn (L.secondStage x) r d) σ) : ℝ)) :=
          mul_le_mul_of_nonneg_right
            (le_maxBelow (fun v => 1 + (A.waits (L.sh x v) : ℝ) * A.k (L.sh x v)) u.isLt)
            (Ex_nonneg fun _ => Nat.cast_nonneg _)

theorem ex_weighted_cost_le (hS : L.CostContract H) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) (s : AuditStatement)
    (W : (Fin (L.V.k x) → Coin) → ℕ) (w : Fin (L.nPU x) → List ℕ → ℝ) (hw : ∀ i l, 0 ≤ w i l) :
    Ex (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        ∑ i, w i (L.checked x p.1) * L.slotCost H x s (run H P L.V x ω) W i p.1 (p.2 i)) ≤
      L.es x * Ex (fun r : Fin (L.V.k x) → Coin =>
        (∑ i ∈ Finset.univ.filter (fun i : Fin (L.nPU x) =>
            (L.sessSlots x (L.checked x r) i.val).head? = some i.val), w i (L.checked x r)) * (W r : ℝ)) := by
  have hslot : ∀ i : Fin (L.nPU x), Ex (fun r : Fin (L.V.k x) → Coin =>
      w i (L.checked x r) * Ex fun ρ : L.BlockEC x => L.slotCost H x s (run H P L.V x ω) W i r ρ) ≤
        Ex fun r : Fin (L.V.k x) → Coin =>
          w i (L.checked x r) *
            if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x * (W r : ℝ) else 0 := by
    intro i
    rw [← Ex_forkOn (L.secondStage x) (fun r => w i (L.checked x r) *
        Ex fun ρ : L.BlockEC x => L.slotCost H x s (run H P L.V x ω) W i r ρ),
      ← Ex_forkOn (L.secondStage x) (fun r => w i (L.checked x r) *
        if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x * (W r : ℝ) else 0), Ex_prod, Ex_prod]
    exact Ex_mono fun r => slot_cost_le_weighted hS s P ω W i (w i) (hw i) r
  calc Ex (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          ∑ i, w i (L.checked x p.1) * L.slotCost H x s (run H P L.V x ω) W i p.1 (p.2 i))
      = Ex (fun r : Fin (L.V.k x) → Coin =>
          ∑ i, w i (L.checked x r) * Ex fun ρ : L.BlockEC x => L.slotCost H x s (run H P L.V x ω) W i r ρ) := by
        rw [Ex_prod]
        refine congrArg Ex (funext fun r => ?_)
        show Ex (fun c : L.SlotEC x =>
          ∑ i, w i (L.checked x r) * L.slotCost H x s (run H P L.V x ω) W i r (c i)) = _
        rw [Ex_finsum]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [Ex_mul_left]
        exact congrArg (fun z => w i (L.checked x r) * z)
          (Ex_pi_eval i fun ρ => L.slotCost H x s (run H P L.V x ω) W i r ρ)
    _ = ∑ i, Ex (fun r : Fin (L.V.k x) → Coin =>
          w i (L.checked x r) * Ex fun ρ : L.BlockEC x => L.slotCost H x s (run H P L.V x ω) W i r ρ) := by
        rw [Ex_finsum]
    _ ≤ ∑ i : Fin (L.nPU x), Ex (fun r : Fin (L.V.k x) → Coin =>
          w i (L.checked x r) *
            if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x * (W r : ℝ) else 0) :=
        Finset.sum_le_sum fun i _ => hslot i
    _ = Ex (fun r : Fin (L.V.k x) → Coin => ∑ i : Fin (L.nPU x),
          w i (L.checked x r) *
            if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x * (W r : ℝ) else 0) := by
        rw [Ex_finsum]
    _ = Ex (fun r : Fin (L.V.k x) → Coin => L.es x *
          ((∑ i ∈ Finset.univ.filter (fun i : Fin (L.nPU x) =>
            (L.sessSlots x (L.checked x r) i.val).head? = some i.val), w i (L.checked x r)) * (W r : ℝ))) := by
        refine congrArg Ex (funext fun r => ?_)
        rw [Finset.sum_mul, Finset.mul_sum, Finset.sum_filter]
        refine Finset.sum_congr rfl fun i _ => ?_
        split_ifs
        · ring
        · exact mul_zero _
    _ = _ := Ex_mul_left _ _

theorem callsAt_le_cost (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    (L.callsAt H P x s ω r c : ℝ) ≤ ((L.partiesCalls H P x ω r : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
      ∑ i, L.slotCost H x s (run H P L.V x ω) (L.partiesCalls H P x ω) i r (c i) := by
  have hown : ((ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length : ℝ) ≤
      (L.nPU x : ℝ) * L.openCalls x := by exact_mod_cast runExt_own_le hS hopen hs P ω r c
  rw [L.callsAt_eq H P x s ω r c, Nat.cast_add, Nat.cast_add,
    runExt_cost_eq H L x s (run H P L.V x ω) (L.partiesCalls H P x ω) r c]
  linarith

theorem callsBy_le_cost (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω) (needs : ℕ → Finset ℕ) (g : ℕ) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    L.callsBy H P x s needs g ω r c ≤ (L.partiesCalls H P x ω r + L.nPU x * L.openCalls x) +
      (((L.runExtAt x s needs g r (run H P L.V x ω r).seen c).rerunsAt H (run H P L.V x ω)).map
        (L.partiesCalls H P x ω)).sum := by
  rw [L.callsBy_eq H P x s needs g ω r c]
  exact Nat.add_le_add_right (Nat.add_le_add_left
    ((L.runExtAt_own_le H x (run H P L.V x ω) s needs g r (run H P L.V x ω r).seen c).trans
      (runExt_own_le hS hopen hs P ω r c)) _) _

theorem ex_runExt_cost_le (hS : L.CostContract H) {x : Msg} (s : AuditStatement) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {As : ℝ}
    (hAs : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ As) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ∑ i, L.slotCost H x s (run H P L.V x q.1) (L.partiesCalls H P x q.1) i q.2.1 (q.2.2 i)) ≤
      L.es x * As := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  refine (Verity.FirstHit.ex_tape fun (ω : Ω) (p : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
    ∑ i, L.slotCost H x s (run H P L.V x ω) (L.partiesCalls H P x ω) i p.1 (p.2 i)).trans_le ?_
  refine (Ex_mono fun ω => le_of_eq_of_le ?_ (ex_weighted_cost_le hS P ω s (L.partiesCalls H P x ω)
    (fun _ _ => (1 : ℝ)) fun _ _ => zero_le_one)).trans ?_
  · exact congrArg Ex (funext fun p => Finset.sum_congr rfl fun i _ => (one_mul _).symm)
  · rw [Ex_mul_left]
    refine mul_le_mul_of_nonneg_left (le_of_eq_of_le ?_ hAs) he0
    rw [Verity.FirstHit.ex_tape fun (ω : Ω) (r : Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x r)).card : ℝ) * (L.partiesCalls H P x ω r : ℝ)]
    refine congrArg Ex (funext fun ω => congrArg Ex (funext fun r => ?_))
    unfold firstSlots
    rw [Finset.sum_const, nsmul_eq_mul, mul_one]

/-- No bound on any one run's calls is read. -/
theorem ex_callsAt_le_avg (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) {T As : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hAs : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ As) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ)) ≤
      L.runCostAvg x T As := by
  have hpt : ∀ q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x), (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ) ≤
      ((L.partiesCalls H P x q.1 q.2.1 : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
        ∑ i, L.slotCost H x s (run H P L.V x q.1) (L.partiesCalls H P x q.1) i q.2.1 (q.2.2 i) :=
    fun q => callsAt_le_cost hS hopen hs P q.1 q.2.1 q.2.2
  have h1 : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (L.partiesCalls H P x q.1 q.2.1 : ℝ) + (L.nPU x : ℝ) * L.openCalls x) ≤
      T + (L.nPU x : ℝ) * L.openCalls x := by
    rw [Ex_add, Ex_const]
    exact add_le_add ((Ex_tape_coins (C := L.SlotEC x) fun (ω : Ω) (r : Fin (L.V.k x) → Coin) =>
      (L.partiesCalls H P x ω r : ℝ)).trans_le hT) le_rfl
  refine ((Ex_mono hpt).trans_eq (Ex_add _ _)).trans ?_
  refine (add_le_add h1 (ex_runExt_cost_le hS s P hAs)).trans (le_of_eq ?_)
  unfold runCostAvg
  ring

theorem ex_bconfX_cost_le_avg (hS : L.CostContract H) (hopen : L.OpenCallsLe H) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {T As : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hAs : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ As) :
    Ex (fun q => ((BconfX H L Ω P x).cost q : ℝ)) ≤ 2 * L.runCostAvg x T As := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hT0 : 0 ≤ T := le_trans (Ex_nonneg fun _ => Nat.cast_nonneg _) hT
  have hAs0 : 0 ≤ As :=
    le_trans (Ex_nonneg fun _ => mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) hAs
  have hrc0 : 0 ≤ L.runCostAvg x T As := by
    unfold runCostAvg
    exact add_nonneg (add_nonneg hT0 (mul_nonneg he0 hAs0)) (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  delta BconfX
  rcases hs : L.parse x with _ | s
  · show Ex (fun _ : Unit => ((0 : ℕ) : ℝ)) ≤ _
    rw [Nat.cast_zero, Ex_const]
    exact mul_nonneg zero_le_two hrc0
  · refine (Ex_mono (g := fun q : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
          (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        (L.callsAt H P x s q.1.1 q.1.2.1 q.1.2.2 : ℝ) + (L.callsAt H P x s q.2.1 q.2.2.1 q.2.2.2 : ℝ))
      fun q => le_of_eq ?_).trans ?_
    · simp only [callsAt]
      push_cast
      ring
    · exact (Ex_two fun q' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (L.callsAt H P x s q'.1 q'.2.1 q'.2.2 : ℝ)).trans_le
          (mul_le_mul_of_nonneg_left (ex_callsAt_le_avg hS hopen hs P hT hAs) zero_le_two)

theorem bconfX_le_avg (hS : L.CostContract H) (hopen : L.OpenCallsLe H) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {T As : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hAs : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ As)
    (hB : ExpectedCR H (BconfX H L Ω P x)) :
    (BconfX H L Ω P x).Pr H ≤ 2 * L.runCostAvg x T As / 2 ^ (256 : ℝ) := by
  unfold ExpectedCR at hB
  exact hB.trans (div_le_div_of_nonneg_right (ex_bconfX_cost_le_avg hS hopen x P hT hAs) (by positivity))

theorem ex_calls_le_avg_in (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (G : List ℕ → Finset (Fin (L.nG x))) {T As Ak Am Ks : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hAs : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ As)
    (hAk : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((G (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Ak)
    (hAm : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      (L.sessCountIn x needs G (L.checked x q.2) : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Am)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin => ((G (L.checked x r)).card : ℝ)) ≤ Ks) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ g ∈ G (L.checked x q.2.1),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤ L.firstHitCostAvg x T As Ak Am Ks := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hN0 : (0 : ℝ) ≤ (L.nPU x : ℝ) * L.openCalls x := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hpt : ∀ q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x),
      ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
        ∑ g ∈ G (L.checked x q.2.1),
          L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ) ≤
        ((L.partiesCalls H P x q.1 q.2.1 : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
          (∑ i, L.slotCost H x s (run H P L.V x q.1) (L.partiesCalls H P x q.1) i q.2.1 (q.2.2 i)) +
          (((G (L.checked x q.2.1)).card : ℝ) * (L.partiesCalls H P x q.1 q.2.1 : ℝ) +
            ((G (L.checked x q.2.1)).card : ℝ) * ((L.nPU x : ℝ) * L.openCalls x)) +
          ∑ i : Fin (L.nPU x), (((G (L.checked x q.2.1)).filter fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.sessBlocks x (L.checked x q.2.1) i.val, g.val ∈ needs u)).card : ℝ) *
            L.slotCost H x s (run H P L.V x q.1) (L.partiesCalls H P x q.1) i q.2.1 (q.2.2 i) := by
    rintro ⟨ω, r, c⟩
    have h1 := callsAt_le_cost hS hopen hs P ω r c
    have h4 : ∑ g ∈ G (L.checked x r), L.callsBy H P x s needs g.val ω r c ≤
        (G (L.checked x r)).card * (L.partiesCalls H P x ω r + L.nPU x * L.openCalls x) +
          ∑ g ∈ G (L.checked x r), (((L.runExtAt x s needs g.val r (run H P L.V x ω r).seen c).rerunsAt H
            (run H P L.V x ω)).map (L.partiesCalls H P x ω)).sum := by
      refine (Finset.sum_le_sum fun g _ => callsBy_le_cost hS hopen hs P needs g.val ω r c).trans_eq ?_
      rw [Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul]
    have h4r : ((∑ g ∈ G (L.checked x r), L.callsBy H P x s needs g.val ω r c : ℕ) : ℝ) ≤
        ((G (L.checked x r)).card : ℝ) * ((L.partiesCalls H P x ω r : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
          ((∑ g ∈ G (L.checked x r), (((L.runExtAt x s needs g.val r (run H P L.V x ω r).seen c).rerunsAt H
            (run H P L.V x ω)).map (L.partiesCalls H P x ω)).sum : ℕ) : ℝ) := by
      exact_mod_cast h4
    have h3 := L.sum_runExtAt_cost_le_in H x s (run H P L.V x ω) (L.partiesCalls H P x ω) needs
      (G (L.checked x r)) r c
    show ((L.callsAt H P x s ω r c +
        ∑ g ∈ G (L.checked x r), L.callsBy H P x s needs g.val ω r c : ℕ) : ℝ) ≤
      ((L.partiesCalls H P x ω r : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
        (∑ i, L.slotCost H x s (run H P L.V x ω) (L.partiesCalls H P x ω) i r (c i)) +
        (((G (L.checked x r)).card : ℝ) * (L.partiesCalls H P x ω r : ℝ) +
          ((G (L.checked x r)).card : ℝ) * ((L.nPU x : ℝ) * L.openCalls x)) +
        ∑ i : Fin (L.nPU x), (((G (L.checked x r)).filter fun g : Fin (L.nG x) =>
            decide (∃ u ∈ L.sessBlocks x (L.checked x r) i.val, g.val ∈ needs u)).card : ℝ) *
          L.slotCost H x s (run H P L.V x ω) (L.partiesCalls H P x ω) i r (c i)
    rw [Nat.cast_add]
    linarith
  have h1 : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (L.partiesCalls H P x q.1 q.2.1 : ℝ) + (L.nPU x : ℝ) * L.openCalls x) ≤
      T + (L.nPU x : ℝ) * L.openCalls x := by
    rw [Ex_add, Ex_const]
    exact add_le_add ((Ex_tape_coins (C := L.SlotEC x) fun (ω : Ω) (r : Fin (L.V.k x) → Coin) =>
      (L.partiesCalls H P x ω r : ℝ)).trans_le hT) le_rfl
  have h2 := ex_runExt_cost_le hS s P hAs
  have h3 : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ((G (L.checked x q.2.1)).card : ℝ) * (L.partiesCalls H P x q.1 q.2.1 : ℝ) +
        ((G (L.checked x q.2.1)).card : ℝ) * ((L.nPU x : ℝ) * L.openCalls x)) ≤
      Ak + Ks * ((L.nPU x : ℝ) * L.openCalls x) := by
    rw [Ex_add, Ex_mul_right]
    exact add_le_add ((Ex_tape_coins (C := L.SlotEC x) fun (ω : Ω) (r : Fin (L.V.k x) → Coin) =>
        ((G (L.checked x r)).card : ℝ) * (L.partiesCalls H P x ω r : ℝ)).trans_le hAk)
      (mul_le_mul_of_nonneg_right
        ((Ex_coins fun r : Fin (L.V.k x) → Coin => ((G (L.checked x r)).card : ℝ)).trans_le hKs) hN0)
  have h4 : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ∑ i : Fin (L.nPU x), (((G (L.checked x q.2.1)).filter fun g : Fin (L.nG x) =>
          decide (∃ u ∈ L.sessBlocks x (L.checked x q.2.1) i.val, g.val ∈ needs u)).card : ℝ) *
        L.slotCost H x s (run H P L.V x q.1) (L.partiesCalls H P x q.1) i q.2.1 (q.2.2 i)) ≤ L.es x * Am := by
    refine (Verity.FirstHit.ex_tape fun (ω : Ω) (p : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ∑ i : Fin (L.nPU x), (((G (L.checked x p.1)).filter fun g : Fin (L.nG x) =>
          decide (∃ u ∈ L.sessBlocks x (L.checked x p.1) i.val, g.val ∈ needs u)).card : ℝ) *
        L.slotCost H x s (run H P L.V x ω) (L.partiesCalls H P x ω) i p.1 (p.2 i)).trans_le ?_
    refine (Ex_mono fun ω => ex_weighted_cost_le hS P ω s (L.partiesCalls H P x ω)
      (fun i l => (((G l).filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.sessBlocks x l i.val, g.val ∈ needs u)).card : ℝ)) fun _ _ => Nat.cast_nonneg _).trans ?_
    rw [Ex_mul_left]
    refine mul_le_mul_of_nonneg_left (le_of_eq_of_le ?_ hAm) he0
    rw [Verity.FirstHit.ex_tape fun (ω : Ω) (r : Fin (L.V.k x) → Coin) =>
      (L.sessCountIn x needs G (L.checked x r) : ℝ) * (L.partiesCalls H P x ω r : ℝ)]
    refine congrArg Ex (funext fun ω => congrArg Ex (funext fun r => ?_))
    unfold sessCountIn firstSlots
    rw [Nat.cast_sum]
  refine ((Ex_mono hpt).trans_eq (Ex_add_four _ _ _ _)).trans ?_
  refine (add_le_add (add_le_add (add_le_add h1 h2) h3) h4).trans (le_of_eq ?_)
  unfold firstHitCostAvg
  ring

theorem ex_calls_le_avg_cls (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) {T As Ak Am Ks : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hAs : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ As)
    (hAk : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      (((L.sampleNeeds x needs (L.checked x q.2)).image cls).card : ℝ) *
        (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Ak)
    (hAm : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x q.2) : ℝ) *
        (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Am)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ c ∈ (Finset.univ.filter (fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
            L.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤ L.firstHitCostAvg x T As Ak Am Ks :=
  ex_calls_le_avg_in hS hopen hs P needs (fun l => (L.sampleNeeds x needs l).image cls) hT hAs hAk hAm hKs

theorem ex_calls_le_exp_cls (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) {T Ns Ks Km : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hNs : ∀ r : Fin (L.V.k x) → Coin, ((L.firstSlots x (L.checked x r)).card : ℝ) ≤ Ns)
    (hKs : ∀ r : Fin (L.V.k x) → Coin, (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ) ≤ Ks)
    (hKm : ∀ r : Fin (L.V.k x) → Coin,
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ) ≤ Km) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ c ∈ (Finset.univ.filter (fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
            L.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤
      T * (1 + L.es x * Ns + Ks + L.es x * Km) + (1 + Ks) * ((L.nPU x : ℝ) * L.openCalls x) := by
  have hAs : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Ns * T :=
    Ex_mul_le_of_le (fun _ => Nat.cast_nonneg _) (fun q => hNs q.2) (fun _ => Nat.cast_nonneg _) hT
  have hAk : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      (((L.sampleNeeds x needs (L.checked x q.2)).image cls).card : ℝ) *
        (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Ks * T :=
    Ex_mul_le_of_le (fun _ => Nat.cast_nonneg _) (fun q => hKs q.2) (fun _ => Nat.cast_nonneg _) hT
  have hAm : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x q.2) : ℝ) *
        (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Km * T :=
    Ex_mul_le_of_le (fun _ => Nat.cast_nonneg _) (fun q => hKm q.2) (fun _ => Nat.cast_nonneg _) hT
  refine (ex_calls_le_avg_cls hS hopen hs P needs cls hT hAs hAk hAm
    ((Ex_mono hKs).trans_eq (Ex_const Ks))).trans (le_of_eq ?_)
  unfold firstHitCostAvg
  ring

end Compiled

end Claims.SamplingGeneric
