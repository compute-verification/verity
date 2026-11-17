module

public import Definitions.Claims.SamplingTwoRegX
public import Definitions.Claims.SamplingTwoServedX
public import Proofs.Flock.Claims.Witnesses.SamplingTwoRegMiss
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ZK.NonVacuous

/-! The two-stage committed form at the boundary trace read at the statement says something: it fails at `sksLTwo`. -/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (footprintAt incorrectAtBoundaryAt)
open Claims.Witness.Sampling (toyStmt toyPart)

theorem sksLTwo_notMX {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type}
    [Fintype Ω] (P : Prover Ω) (x : Msg) (ι : ℕ → ℕ → Bool) :
    footprintAt toyStmt toyPart toyPart ((sksLTwo Coin).regValX H P x) ι ∉ toyTwo.M toyPart 0 := by
  intro hM
  have hlt : (0 : ℝ) < toyTwo.Pa toyPart (footprintAt toyStmt toyPart toyPart ((sksLTwo Coin).regValX H P x) ι) :=
    hM
  rw [toyTwo_Pa (toy_footprintAt (val := (sksLTwo Coin).regValX H P x) (sksL_regValX H P x 0) ι)] at hlt
  exact lt_irrefl _ hlt

theorem sksLTwo_εregTwoX {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (x : Msg) :
    (sksLTwo Coin).εregTwoX H x silentP = 0 := by
  unfold CompiledTwo.εregTwoX
  rw [sksLTwo_bconfTwo, Real.sqrt_zero, mul_zero, add_zero]
  exact sksL_εregX H x

/-- **`CompiledTwo.KSRegTwoX` fails** at `sksLTwo`'s silent prover, at `δ = 0`: it accepts always, with no interiors
putting the footprint of its boundary trace read at the statement in `M̄_0`, at an error of `0`. -/
theorem compiledTwoKSRegX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).KSRegTwoX H := by
  intro h
  have h1 := (h [] (sksLTwo_scope []) Unit silentP).2 toyStmt toyPart toyTwo rfl rfl rfl 0 le_rfl
  have hp : Pr (fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      (run H silentP (sksLTwo Coin).V [] p.1 p.2).out = .acc ∧
        ∀ ι : ℕ → ℕ → Bool,
          footprintAt toyStmt toyPart toyTwo.part ((sksLTwo Coin).regValX H silentP []) ι ∉ toyTwo.M toyPart 0) =
      1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      iff_true_intro ⟨run_accV_out H silentP [] p.1 p.2, sksLTwo_notMX H silentP []⟩) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksLTwo Coin).εregTwoX H [] silentP := hp.symm.trans_le h1
  rw [sksLTwo_εregTwoX] at h2
  norm_num at h2

/-- **`CompiledTwo.HeadlineTwoX` fails** at `sksLTwo`'s silent prover: it accepts always, while `P̄_a` at the footprint
of its boundary trace read at the statement is `0` at every interior, at an error of `0`. -/
theorem compiledTwoHeadlineX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).HeadlineTwoX H := by
  intro h
  have h1 := h [] toyStmt toyPart toyTwo (sksLTwo_scope []) rfl rfl rfl Unit silentP 0 fun ι =>
    (toyTwo_Pa (toy_footprintAt (val := (sksLTwo Coin).regValX H silentP []) (sksL_regValX H silentP [] 0) ι)).le
  have hp : Pr (fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      (run H silentP (sksLTwo Coin).V [] p.1 p.2).out = .acc) = 1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      iff_true_intro (run_accV_out H silentP [] p.1 p.2)) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksLTwo Coin).εregTwoX H [] silentP := hp.symm.trans_le h1
  rw [sksLTwo_εregTwoX] at h2
  norm_num at h2

/-- **`CompiledTwo.MissTwoX` fails** at `sksLTwo`'s silent prover, at both rates `1`: its sample misses the incorrect
set always, while its replay block is wrong at the boundary trace read at the statement, so the bound is `0`. -/
theorem compiledTwoMissX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).MissTwoX H 1 1 := by
  intro h
  have h1 := h [] toyStmt toyPart toyTwo (sksLTwo_scope []) rfl rfl rfl toyTwo_firstRate toyTwo_secondRate Unit
    silentP ()
  have hW : 0 < (incorrectAtBoundaryAt toyStmt toyPart toyTwo.part ((sksLTwo Coin).regValX H silentP [])).card :=
    Finset.card_pos.2 ⟨0, toy_incorrectAtBoundaryAt (val := (sksLTwo Coin).regValX H silentP [])
      (sksL_regValX H silentP [] 0)⟩
  have hp : Pr (fun r : Fin ((sksLTwo Coin).V.k []) → Coin =>
      ∀ u ∈ (sksLTwo Coin).checked [] r, u ∉ (sksLTwo Coin).FregTwoX H silentP [] () r) = 1 :=
    Eq.trans (Pr_congr fun _ => iff_true_intro fun u hu => absurd (show u ∈ ([] : List ℕ) from hu) (by simp))
      Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ (1 - 1 * 1) ^
      (incorrectAtBoundaryAt toyStmt toyPart toyTwo.part ((sksLTwo Coin).regValX H silentP [])).card :=
    hp.symm.trans_le h1
  rw [show (1 : ℝ) - 1 * 1 = 0 by norm_num, zero_pow hW.ne'] at h2
  norm_num at h2

end Claims.Witness.SamplingGeneric

namespace Claims.SamplingGeneric

open Claims.Witness.SamplingGeneric

/-- **`CompiledTwo.KSRegTwoX` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.KSRegTwoX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.KSRegTwoX H :=
  ⟨_, _, compiledTwoKSRegX_fails H⟩

/-- **`CompiledTwo.ExfilRegTwoX` fails** at some compiled two-stage verifier: its first part is the committed form
(`KSRegTwoX`), which fails at `sksLTwo`. -/
theorem CompiledTwo.ExfilRegTwoX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) : ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.ExfilRegTwoX H :=
  ⟨_, sksLTwo Coin, fun h => compiledTwoKSRegX_fails H h.1⟩

/-- **`CompiledTwo.HeadlineTwoX` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.HeadlineTwoX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) : ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.HeadlineTwoX H :=
  ⟨_, _, compiledTwoHeadlineX_fails H⟩

/-- **`CompiledTwo.MissTwoX` fails** at some compiled two-stage verifier and rates. -/
theorem CompiledTwo.MissTwoX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : CompiledTwo Coin A) (r₁ r₂ : ℝ), ¬ L.MissTwoX H r₁ r₂ :=
  ⟨_, _, 1, 1, compiledTwoMissX_fails H⟩

/-- **`CompilerKSRegTwoX`'s conclusion fails** at `sksLTwo`, which accepts with the footprint of its boundary trace
read at the statement outside `M̄_0` at every interior. -/
theorem compilerKSRegTwoX_nontrivial : ¬ (sksLTwo Bool).KSRegTwoX FlockDeploy.sha512 := compiledTwoKSRegX_fails _

/-- **`CompilerHeadlineTwoX`'s conclusion fails** at `sksLTwo`, whose `P̄_a` is `0` at the footprint of its boundary
trace read at the statement at every interior. -/
theorem compilerHeadlineTwoX_nontrivial : ¬ (sksLTwo Bool).HeadlineTwoX FlockDeploy.sha512 :=
  compiledTwoHeadlineX_fails _

/-- **`CompilerMissTwoX`'s conclusion fails** at `sksLTwo`, at both rates `1`, which its design meets: it samples
nothing, while its replay block is wrong at the boundary trace read at the statement. -/
theorem compilerMissTwoX_nontrivial : ¬ (sksLTwo Bool).MissTwoX FlockDeploy.sha512 1 1 := compiledTwoMissX_fails _

end Claims.SamplingGeneric
