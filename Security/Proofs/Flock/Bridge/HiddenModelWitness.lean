import Proofs.Flock.Bridge.Assumptions.ToySetup

/-!
# `EndToEndHiddenModel` is nontrivial

`endToEndHiddenModel_nontrivial`: under `toy-setup` (`Flock.Assumptions.ToySetupHJ`), `EndToEndHiddenModel` fails at the
verifier that accepts everything (`EndToEndHiddenModelAll`). At the toy, with every table refused, its hypotheses hold
(`toy_lowering_refused`, `toy_readsOwn`, `sampledKSLiveV2_honest`), its one unit is wrong at every registration and
transcript (`toy_mem_wrongRegZ`), so the event at that verifier holds with probability 1, while the bound is below 1
(`boundPr_le_pub`, `deltaPub_lt_one`).
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Refine FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

/-- **`EndToEndHiddenModel` is nontrivial**: under `toy-setup`, its statement at the verifier that accepts everything
is false. At the toy's reference setup and prover, its bound at the deployed parameters (`k = ⌈2^24/e⌉`,
`R_w = M = 2^192`, `ρ = 1/2`, `K₀ = 1`) is below 1, and its event holds with probability 1. -/
theorem endToEndHiddenModel_nontrivial (h : Flock.Assumptions.ToySetupHJ) :
    ¬ Flock.Guarantees.EndToEndHiddenModelAll := by
  intro hAll
  obtain ⟨I, mPts, S₀, d₀, y₀, σ, t', v, hB⟩ := h
  obtain ⟨hk, hRw, hM, hρ, hr⟩ := Flock.Guarantees.sampledKSLiveV2_honest (Reg := Unit) toyDj y₀
  have hle := hAll 1 Unit I mPts toyDj S₀ d₀ y₀ toyCircuit toyPartition
    (toyDecode (planZHJ toyDj y₀) (tabZHJ toyDj y₀)) _ H512
    (toyBinding execArith (planZHJ toyDj y₀) (tabZHJ toyDj y₀) H512) (toy_lowering_refused y₀) toy_readsOwn
    (fun _ _ => True) (fun _ _ _ _ _ _ _ _ => trivial) (fun _ => False) (fun _ => default) toyLaw
    kDep (2 ^ 192) (2 ^ 192) hk hRw hM (1 / 2) hρ hr 1 σ
  rw [rowsPrZC_noReads, add_zero] at hle
  have hpub := boundPr_le_pub toyDj y₀ toyPartition (toyBinding execArith (planZHJ toyDj y₀) (tabZHJ toyDj y₀) H512)
    (kd := 1) (K₀ := 1) le_rfl (k := kDep) (Rw := 2 ^ 192) (M := 2 ^ 192) (ρ := 1 / 2) (by norm_num) rateM1_kDep
    (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
  have hlt := deltaPub_lt_one (n := 1) (kd := 1) (K₀ := 1) (W := ioSum toyPartition) one_pos le_rfl (by norm_num)
    toy_ioSum_le (mul_nonneg (mul_nonneg (sq_nonneg _) (Nat.cast_nonneg _)) (Claims.Pr_nonneg _)) hB
  refine absurd hle (not_le.2 (lt_of_le_of_lt ?_ (exceeds_of σ (Finset.one_le_card.2
    ⟨0, Finset.mem_union_left _ (wrongRegZ_subset_wrongRegZc execArith toyPartition _ _ _ _ _ _
      (toy_mem_wrongRegZ execArith (planZHJ toyDj y₀) (tabZHJ toyDj y₀) H512 (reg σ) _))⟩) hlt)))
  rw [toyLaw.miss_eq_zero_of_draw_univ (fun _ => rfl) le_rfl]
  exact le_trans (add_le_add (add_le_add zero_le le_rfl) le_rfl) hpub

end Flock.Bridge
