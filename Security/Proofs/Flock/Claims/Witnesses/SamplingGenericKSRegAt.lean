module

public import Proofs.Flock.Claims.Witnesses.SamplingGenericCompiler
public import Proofs.Flock.SamplingGeneric.RegAt

/-!
# Refutation witnesses: the committed form read at a time budget

`Compiled.KSRegAt` fails at `sksL` (`Witnesses.SamplingGeneric`), whose silent prover is accepted with its committed
trace's incorrect set outside `M_0` (`sksL_notM`). At the budget `0`, `B_conf` declared there finds nothing, as `B_conf`
does (`bconf_pr`, `Compiled.BconfAt_pr`), so `ExpectedCR` holds there (`sksL_expectedCR_bconfAt`), and its expected
cost bound is `0`, `sksL` opening nothing, so the error read at the budget `0` is `0` (`sksL_εregAt`), below the silent
prover's probability `1` (`compiledKSRegAt_fails`). So `CompilerKSRegAt`'s conclusion fails there
(`compilerKSRegAt_nontrivial`); its premises are `CompilerKSAt`'s, which hold together at `twoL` (`compilerKS_honest`,
`compilerBits_honest`). Read at the strict birthday bound the same: `StrictBirthday` holds at `B_conf` declared at the
budget `0` (`sksL_strictBirthday_bconfAt`), whose cost bound on every coin is `0`, so the error is `0`
(`sksL_εregSB`, `compiledKSRegAtSB_fails`, `compilerKSRegAtSB_nontrivial`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Witness.Sampling (toyStmt)

section KSRegAt

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- At `sksL`'s silent prover, `B_conf` declared at the budget `0` finds nothing, so `ExpectedCR` holds there. -/
theorem sksL_expectedCR_bconfAt (H : Msg → Msg) (x : Msg) :
    ExpectedCR H ((sksL Coin).BconfAt H 0 Unit silentP x) := by
  unfold ExpectedCR
  rw [Compiled.BconfAt_pr, bconf_pr H (sksL Coin) (fun _ _ _ => rfl) Unit silentP x]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- `sksL`'s committed form's error read at the budget `0` is `0`: `B_conf`'s expected cost bound there is `0`, `sksL`
opening nothing. -/
theorem sksL_εregAt (H : Msg → Msg) (x : Msg) : (sksL Coin).εregAt H x silentP 0 = 0 := by
  have hb : PrB H (sksL Coin).V silentP x = 0 := prB_accV H silentP x
  have hc : (sksL Coin).bconfCost x 0 = 0 := by
    simp [Compiled.bconfCost, sksL]
  unfold Compiled.εregAt
  rw [sksL_κs, hb, hc]
  simp

/-- **`Compiled.KSRegAt` fails** at `sksL`'s silent prover, at `δ = 0` and the budget `0`: it accepts always, with its
committed trace's incorrect set outside `M_0`, at an error of `0`. -/
theorem compiledKSRegAt_fails (H : Msg → Msg) : ¬ (sksL Coin).KSRegAt H := by
  intro h
  have h1 := (h [] (sksL_scope []) Unit silentP 0 (sksL_expectedCR_bconfAt H [])).2 toyStmt rfl 0 le_rfl
  have hp : Pr (fun p : Unit × (Fin ((sksL Coin).V.k []) → Coin) =>
      (run H silentP (sksL Coin).V [] p.1 p.2).out = .acc ∧
        (sksL Coin).Freg H silentP [] p.1 ∉ toyStmt.design.M ((sksL Coin).part toyStmt) 0) = 1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksL Coin).V.k []) → Coin) =>
      iff_true_intro ⟨run_accV_out H silentP [] p.1 p.2, sksL_notM H silentP [] p.1⟩) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksL Coin).εregAt H [] silentP 0 := hp.symm.trans_le h1
  rw [sksL_εregAt] at h2
  norm_num at h2

/-- At `sksL`'s silent prover, `B_conf` declared at the budget `0` finds nothing, so `StrictBirthday` holds there. -/
theorem sksL_strictBirthday_bconfAt (H : Msg → Msg) (x : Msg) :
    StrictBirthday H ((sksL Coin).BconfAt H 0 Unit silentP x) := by
  intro q _
  rw [Compiled.BconfAt_pr, bconf_pr H (sksL Coin) (fun _ _ _ => rfl) Unit silentP x]
  positivity

/-- `sksL`'s committed form's error at the strict birthday bound and the budget `0` is `0`: `B_conf`'s cost bound on
every coin there is `0`, `sksL` opening nothing. -/
theorem sksL_εregSB (H : Msg → Msg) (x : Msg) : (sksL Coin).εregSB H x silentP 0 = 0 := by
  have hb : PrB H (sksL Coin).V silentP x = 0 := prB_accV H silentP x
  have hc : (sksL Coin).bconfCostStrict x 0 = 0 := by
    simp [Compiled.bconfCostStrict, sksL]
  unfold Compiled.εregSB
  rw [sksL_κs, hb, hc]
  simp

/-- **`Compiled.KSRegAtSB` fails** at `sksL`'s silent prover, at `δ = 0` and the budget `0`: it accepts always, with
its committed trace's incorrect set outside `M_0`, at an error of `0`. -/
theorem compiledKSRegAtSB_fails (H : Msg → Msg) : ¬ (sksL Coin).KSRegAtSB H := by
  intro h
  have h1 := (h [] (sksL_scope []) Unit silentP 0 (sksL_strictBirthday_bconfAt H [])).2 toyStmt rfl 0 le_rfl
  have hp : Pr (fun p : Unit × (Fin ((sksL Coin).V.k []) → Coin) =>
      (run H silentP (sksL Coin).V [] p.1 p.2).out = .acc ∧
        (sksL Coin).Freg H silentP [] p.1 ∉ toyStmt.design.M ((sksL Coin).part toyStmt) 0) = 1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksL Coin).V.k []) → Coin) =>
      iff_true_intro ⟨run_accV_out H silentP [] p.1 p.2, sksL_notM H silentP [] p.1⟩) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksL Coin).εregSB H [] silentP 0 := hp.symm.trans_le h1
  rw [sksL_εregSB] at h2
  norm_num at h2

end KSRegAt

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

/-- **`Compiled.KSRegAt` fails** at some compiled verifier. -/
theorem Compiled.KSRegAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.KSRegAt H :=
  ⟨_, _, Witness.SamplingGeneric.compiledKSRegAt_fails H⟩

/-- **`CompilerKSRegAt`'s conclusion fails** at `sksL`, which accepts with an empty relation. -/
theorem compilerKSRegAt_nontrivial : ¬ (Witness.SamplingGeneric.sksL Bool).KSRegAt FlockDeploy.sha512 :=
  Witness.SamplingGeneric.compiledKSRegAt_fails _

/-- **`Compiled.KSRegAtSB` fails** at some compiled verifier. -/
theorem Compiled.KSRegAtSB_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.KSRegAtSB H :=
  ⟨_, _, Witness.SamplingGeneric.compiledKSRegAtSB_fails H⟩

/-- **`CompilerKSRegAtSB`'s conclusion fails** at `sksL`, which accepts with an empty relation. -/
theorem compilerKSRegAtSB_nontrivial : ¬ (Witness.SamplingGeneric.sksL Bool).KSRegAtSB FlockDeploy.sha512 :=
  Witness.SamplingGeneric.compiledKSRegAtSB_fails _

end Claims.SamplingGeneric
