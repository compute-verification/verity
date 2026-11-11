module

public import Proofs.Flock.Claims.Witnesses.SamplingTwoReg
public import Proofs.Flock.Claims.Witnesses.SamplingGenericKSRegAt
public import Proofs.Flock.SamplingGeneric.TwoRegAt
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ZK.NonVacuous

/-!
# Refutation witnesses: the two-stage committed form read at a time budget

`CompiledTwo.KSRegTwoAt` fails at `sksLTwo` (`Witnesses.SamplingTwo`), its silent prover and any hash, at the budget
`0`. There `B_conf` and `B_conf,2` declared at `0` find nothing, as `B_conf` and `B_conf,2` do (`Compiled.BconfAt_pr`,
`CompiledTwo.BconfTwoAt_pr`, `sksLTwo_bconfTwo`), so `ExpectedCR` holds at both (`sksL_expectedCR_bconfAt`,
`sksLTwo_expectedCR_bconfTwoAt`), and their expected cost bound is `0`, `sksL` opening nothing, so the error read at the
budget `0` is `0` (`sksLTwo_εregTwoAt`). It accepts always, so at `δ = 0` the threshold form bounds `1` by `0`
(`compiledTwoKSRegAt_fails`), and `CompilerKSRegTwoAt`'s conclusion fails there (`compilerKSRegTwoAt_nontrivial`); its
premises are `CompilerKSAtTwo`'s (`compilerKSTwo_honest`, `compilerBits_honest`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (footprintAt)
open Claims.Witness.Sampling (toyStmt toyPart)

section KSRegTwoAt

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- At `sksLTwo`'s silent prover, `B_conf,2` declared at the budget `0` finds nothing, so `ExpectedCR` holds there. -/
theorem sksLTwo_expectedCR_bconfTwoAt (H : Msg → Msg) (x : Msg) :
    ExpectedCR H (CompiledTwo.BconfTwoAt H (sksLTwo Coin) 0 Unit silentP x) := by
  unfold ExpectedCR
  rw [CompiledTwo.BconfTwoAt_pr, sksLTwo_bconfTwo H Unit silentP x]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- `sksLTwo`'s committed form's error at two stages read at the budget `0` is `0`: both finders' expected cost bound
there is `0`, `sksL` opening nothing. -/
theorem sksLTwo_εregTwoAt (H : Msg → Msg) (x : Msg) : (sksLTwo Coin).εregTwoAt H x silentP 0 = 0 := by
  have h0 : (sksLTwo Coin).εregAt H x silentP 0 = 0 := sksL_εregAt H x
  have hc : (sksLTwo Coin).bconfCost x 0 = 0 := by
    simp [Compiled.bconfCost, sksLTwo, sksL]
  unfold CompiledTwo.εregTwoAt
  rw [h0, hc]
  simp

/-- **`CompiledTwo.KSRegTwoAt` fails** at `sksLTwo`'s silent prover, at `δ = 0` and the budget `0`: it accepts always,
with no interiors putting its committed boundary trace's footprint in `M̄_0`, at an error of `0`. -/
theorem compiledTwoKSRegAt_fails (H : Msg → Msg) : ¬ (sksLTwo Coin).KSRegTwoAt H := by
  intro h
  have h1 := (h [] (sksLTwo_scope []) Unit silentP 0 (sksL_expectedCR_bconfAt H [])
    (sksLTwo_expectedCR_bconfTwoAt H [])).2 toyStmt toyPart toyTwo rfl rfl rfl 0 le_rfl
  have hp : Pr (fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      (run H silentP (sksLTwo Coin).V [] p.1 p.2).out = .acc ∧
        ∀ ι : ℕ → ℕ → Bool,
          footprintAt toyStmt toyPart toyTwo.part ((sksLTwo Coin).regVal H silentP [] p.1) ι ∉ toyTwo.M toyPart 0) =
      1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      iff_true_intro ⟨run_accV_out H silentP [] p.1 p.2, sksLTwo_notM H silentP [] p.1⟩) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksLTwo Coin).εregTwoAt H [] silentP 0 := hp.symm.trans_le h1
  rw [sksLTwo_εregTwoAt] at h2
  norm_num at h2

end KSRegTwoAt

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

/-- **`CompiledTwo.KSRegTwoAt` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.KSRegTwoAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) : ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.KSRegTwoAt H :=
  ⟨_, _, Witness.SamplingGeneric.compiledTwoKSRegAt_fails H⟩

/-- **`CompilerKSRegTwoAt`'s conclusion fails** at `sksLTwo`, which accepts with its committed boundary trace's
footprint outside `M̄_0` at every interior. -/
theorem compilerKSRegTwoAt_nontrivial : ¬ (Witness.SamplingGeneric.sksLTwo Bool).KSRegTwoAt FlockDeploy.sha512 :=
  Witness.SamplingGeneric.compiledTwoKSRegAt_fails _

end Claims.SamplingGeneric
