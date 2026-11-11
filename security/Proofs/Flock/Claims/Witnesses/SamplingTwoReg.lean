module

public import Definitions.Claims.SamplingTwoReg
public import Proofs.Flock.Claims.Witnesses.SamplingTwo
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ZK.NonVacuous

/-!
# The two-stage committed form's statements say something

`CompiledTwo.KSRegTwo` and `CompiledTwo.HeadlineTwo` fail at `sksLTwo` (`Witnesses.SamplingTwo`: `sksL`, which accepts
after one round, samples nothing and opens nothing, with the two-stage design `toyTwo`), its silent prover and any hash:
`CompiledTwo.KSRegTwo_fails` and `CompiledTwo.HeadlineTwo_fails` over the spec, and at coins `Bool` and SHA-512
(`FlockDeploy.sha512`) the nontrivial lemmas of `CompilerKSRegTwo` and `CompilerHeadlineTwo`.

* `sksLTwo`'s committed boundary trace opens nothing (`sksL_regVal`), so its one proof block is incorrect there at any
  interior (`toy_wrongInAt`), the footprint is at least `1` in the one cell, and the design's `P̄_a` is `0` at every
  interior: the footprint is outside `M̄_0` (`sksLTwo_notM`).
* Its sessions make no call, so `B_conf,2` finds nothing (`sksLTwo_bconfTwo`), and with `ε_reg = 0` (`sksL_εreg`) the
  committed form's error at two stages is `0` (`sksLTwo_εregTwo`).
* It accepts always. At `δ = 0` the threshold form then bounds `1` by `0` (`compiledTwoKSReg_fails`), and so does the
  headline at the bound `0` on `P̄_a` (`compiledTwoHeadline_fails`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (wrongInAt footprintAt)
open Claims.Witness.Sampling (ownLog_noHash toyStmt toyPart toyPart_blocks)

/-- At boundary values that commit nothing at the input, the toy's proof block is incorrect at any interior values. -/
theorem toy_wrongInAt {val : ℕ → Option Bool} (hv : val 0 = none) (ι : ℕ → Bool) :
    0 ∈ wrongInAt toyStmt toyPart toyPart val 0 0 ι := by
  classical
  unfold wrongInAt
  refine Finset.mem_filter.2 ⟨by decide, ?_⟩
  rintro ⟨u, r, hu, hr, hc⟩
  rw [toyPart_blocks] at hu hr
  simp only [List.getElem?_cons_zero, Option.some.injEq] at hu hr
  subst hu hr
  obtain ⟨b, hb⟩ := toy_correct_val _ ι hc
  rw [hv] at hb
  cases hb

theorem toy_footprintAt {val : ℕ → Option Bool} (hv : val 0 = none) (ι : ℕ → ℕ → Bool) :
    1 ≤ footprintAt toyStmt toyPart toyPart val ι 0 0 :=
  Finset.card_pos.2 ⟨0, toy_wrongInAt hv (ι 0)⟩

/-- `sksLTwo`'s committed boundary trace's footprint is outside `M̄_0` at every interior. -/
theorem sksLTwo_notM {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (ι : ℕ → ℕ → Bool) :
    footprintAt toyStmt toyPart toyPart ((sksLTwo Coin).regVal H P x ω) ι ∉ toyTwo.M toyPart 0 := by
  intro hM
  have hlt : (0 : ℝ) < toyTwo.Pa toyPart (footprintAt toyStmt toyPart toyPart ((sksLTwo Coin).regVal H P x ω) ι) :=
    hM
  rw [toyTwo_Pa (toy_footprintAt (val := (sksLTwo Coin).regVal H P x ω) (sksL_regVal H P x ω 0) ι)] at hlt
  exact lt_irrefl _ hlt

/-- `sksLTwo`'s sessions make no call, so `B_conf,2` finds nothing. -/
theorem sksLTwo_bconfTwo {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (Ω : Type)
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : (CompiledTwo.BconfTwo H (sksLTwo Coin) Ω P x).Pr H = 0 := by
  simp only [CompiledTwo.BconfTwo]
  rcases (sksLTwo Coin).parse x with _ | s
  · exact finder_pr_none H _ fun _ => rfl
  · exact finder_pr_none H _ fun q => by
      simp [ownLog_noHash _ _ _ (noHash_runExt (sksLTwo Coin).toCompiled (fun _ _ _ => rfl) _ _ _ _ _), findColl]

/-- `sksLTwo`'s committed form's error at two stages is `0`. -/
theorem sksLTwo_εregTwo {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (x : Msg) :
    (sksLTwo Coin).εregTwo H x silentP = 0 := by
  unfold CompiledTwo.εregTwo
  rw [sksLTwo_bconfTwo, Real.sqrt_zero, mul_zero, add_zero]
  exact sksL_εreg H x

/-- **`CompiledTwo.KSRegTwo` fails** at `sksLTwo`'s silent prover, at `δ = 0`: it accepts always, with no interiors
putting its committed boundary trace's footprint in `M̄_0`, at an error of `0`. -/
theorem compiledTwoKSReg_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).KSRegTwo H := by
  intro h
  have h1 := (h [] (sksLTwo_scope []) Unit silentP).2 toyStmt toyPart toyTwo rfl rfl rfl 0 le_rfl
  have hp : Pr (fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      (run H silentP (sksLTwo Coin).V [] p.1 p.2).out = .acc ∧
        ∀ ι : ℕ → ℕ → Bool,
          footprintAt toyStmt toyPart toyTwo.part ((sksLTwo Coin).regVal H silentP [] p.1) ι ∉ toyTwo.M toyPart 0) =
      1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      iff_true_intro ⟨run_accV_out H silentP [] p.1 p.2, sksLTwo_notM H silentP [] p.1⟩) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksLTwo Coin).εregTwo H [] silentP := hp.symm.trans_le h1
  rw [sksLTwo_εregTwo] at h2
  norm_num at h2

/-- **`CompiledTwo.HeadlineTwo` fails** at `sksLTwo`'s silent prover: it accepts always, while `P̄_a` at the committed
boundary trace's footprint is `0` at every interior, at an error of `0`. -/
theorem compiledTwoHeadline_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).HeadlineTwo H := by
  intro h
  have h1 := h [] toyStmt toyPart toyTwo (sksLTwo_scope []) rfl rfl rfl Unit silentP (fun _ => 0) fun ω ι =>
    (toyTwo_Pa (toy_footprintAt (val := (sksLTwo Coin).regVal H silentP [] ω) (sksL_regVal H silentP [] ω 0) ι)).le
  have hp : Pr (fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      (run H silentP (sksLTwo Coin).V [] p.1 p.2).out = .acc) = 1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      iff_true_intro (run_accV_out H silentP [] p.1 p.2)) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ Ex (fun _ : Unit => (0 : ℝ)) + (sksLTwo Coin).εregTwo H [] silentP := hp.symm.trans_le h1
  rw [sksLTwo_εregTwo, Ex_const] at h2
  norm_num at h2

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

open Claims.Witness.SamplingGeneric

/-- **`CompiledTwo.KSRegTwo` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.KSRegTwo_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.KSRegTwo H :=
  ⟨_, _, compiledTwoKSReg_fails H⟩

/-- **`CompiledTwo.HeadlineTwo` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.HeadlineTwo_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) : ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.HeadlineTwo H :=
  ⟨_, _, compiledTwoHeadline_fails H⟩

/-- **`CompilerKSRegTwo`'s conclusion fails** at `sksLTwo`, which accepts with its committed boundary trace's footprint
outside `M̄_0` at every interior. -/
theorem compilerKSRegTwo_nontrivial : ¬ (sksLTwo Bool).KSRegTwo FlockDeploy.sha512 := compiledTwoKSReg_fails _

/-- **`CompilerHeadlineTwo`'s conclusion fails** at `sksLTwo`, whose `P̄_a` is `0` at the committed boundary trace's
footprint at every interior. -/
theorem compilerHeadlineTwo_nontrivial : ¬ (sksLTwo Bool).HeadlineTwo FlockDeploy.sha512 :=
  compiledTwoHeadline_fails _

end Claims.SamplingGeneric
