module

public import Proofs.Flock.Claims.Witnesses.SamplingGenericCompiler
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Witnesses.Sampling
import Proofs.Flock.Claims.Witnesses.Shared

/-!
# Refutation witnesses: KS read at a time budget

`Sampling.KSAt` and `Compiled.KSAt` fail at `sksL` (`Witnesses.SamplingGeneric`), which accepts every run while no
witness is in its relation at `δ = 0`: its `γ` is `0`, so the birthday term vanishes, and at the budget `0` its `B_ext`
finds nothing at no cost, so the binding term is `0` (`sksL_ecrBound`). The bound is then `κ_0 < 1/3`, below the silent
prover's `Pr[bad] = 1` (`sksL_ksAt_fails`), at any scope holding `[]` and any relation with no witness there
(`sksL_ksAt_fails_at`, which `SamplingTwoKSAt` reads). So `CompilerKSAt`'s conclusion fails there
(`compilerKSAt_nontrivial`); its premises are `CompilerBits`', which hold together at `twoL` (`compilerKS_honest`,
`compilerBits_honest`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (ownLog)
open Claims.Witness.Sampling (sksD_noR ownLog_noHash)

section KSAt

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- At `sksL`'s silent prover, `B_ext` at the budget `0` finds nothing at no cost, so its binding term at the budget `0`
is `0`. -/
theorem sksL_ecrBound (H : Msg → Msg) (x : Msg) :
    ecrBound H (Claims.Sampling.BextAt H (sksL Coin).V (sksL Coin).E 0 Unit silentP x) 0 = 0 := by
  have hn : ∀ (ω : Unit) (r₀ : Fin ((sksL Coin).V.k x) → Coin) (c : (sksL Coin).E.C x),
      ownLog H (run H silentP (sksL Coin).V x ω) ((sksL Coin).E.ext x r₀ c) = [] := fun _ r₀ c =>
    ownLog_noHash _ _ _ (noHash_compiledE (sksL Coin) (fun _ _ _ => rfl) x r₀ c)
  have hk : 0 < (sksL Coin).V.k x := Nat.one_pos
  have hpr : (Claims.Sampling.BextAt H (sksL Coin).V (sksL Coin).E 0 Unit silentP x).Pr H = 0 := by
    unfold Claims.Sampling.BextAt
    rw [dite_eq_left hk]
    exact finder_pr_none H _ fun s => by simp [hn, findColl]
  have hc : Ex (fun s => ((Claims.Sampling.BextAt H (sksL Coin).V (sksL Coin).E 0 Unit silentP x).cost s : ℝ)) =
      0 := by
    unfold Claims.Sampling.BextAt
    rw [dite_eq_left hk]
    simp only [hn, List.length_nil, zero_mul, add_zero, Nat.cast_zero, Ex, Finset.sum_const_zero, zero_div]
  have hE : ExpectedCR H (Claims.Sampling.BextAt H (sksL Coin).V (sksL Coin).E 0 Unit silentP x) := by
    unfold ExpectedCR
    rw [hpr, hc, zero_div]
  unfold ecrBound
  rw [ite_eq_left (And.intro hE hc.le), zero_div, ENNReal.ofReal_zero]

/-- **`Sampling.KSAt` fails** at `sksL`'s verifier, extractor and errors, at the budget `0`, on any scope holding `[]`
and any relation with no witness at `[]`: the silent prover is accepted while no witness is in the relation, `γ = 0`,
and `B_ext` at the budget `0` finds nothing at no cost, so the bound is `κ_0 < 1`. -/
theorem sksL_ksAt_fails_at (H : Msg → Msg) {Scope : Msg → Prop} {R : Rel Claims.Sampling.Wit} (hs : Scope [])
    (hR : ∀ w, ¬ R [] w) :
    ¬ Claims.Sampling.KSAt H (sksL Coin).V Scope R (sksL Coin).E ((sksL Coin).κ 0) (sksL Coin).γ (sksL Coin).work := by
  intro h
  have h1 := h [] hs Unit silentP 0
  have hp : PrBad H (sksL Coin).V R (sksL Coin).E [] silentP = 1 := prBad_accV H _ _ [] silentP hR
  rw [hp, sksL_γ, Real.sqrt_zero, ENNReal.ofReal_zero, zero_mul, add_zero, Nat.cast_zero, mul_zero, zero_mul,
    sksL_ecrBound, add_zero, ENNReal.ofReal_le_ofReal_iff'] at h1
  have := sksL_κ (Coin := Coin) []
  rcases h1 with h1 | h1 <;> linarith

/-- **`Sampling.KSAt` fails** at `sksL`, at `δ = 0` and the budget `0` (`sksL_ksAt_fails_at`). -/
theorem sksL_ksAt_fails (H : Msg → Msg) :
    ¬ Claims.Sampling.KSAt H (sksL Coin).V (sksL Coin).Scope
      (Claims.Sampling.R H (sksL Coin).commit (sksL Coin).parse (sksL Coin).partition 0) (sksL Coin).E
      ((sksL Coin).κ 0) (sksL Coin).γ (sksL Coin).work := by
  refine sksL_ksAt_fails_at H ?_ ?_
  · exact sksL_scope []
  · exact sksD_noR (Coin := Coin) H 0 le_rfl []

/-- **`Compiled.KSAt` fails** at `sksL`: its first half, at `δ = 0`. -/
theorem compiledKSAt_fails (H : Msg → Msg) : ¬ (sksL Coin).KSAt H := fun h => sksL_ksAt_fails H (h.1 0 le_rfl)

end KSAt

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.Sampling

/-- **`KSAt` fails** somewhere: at `sksL`'s verifier, extractor and errors. -/
theorem KSAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (V : Verifier Coin) (Scope : Msg → Prop) (R : Rel Wit) (E : Extractor Coin Wit V) (κ γ work : Msg → ℝ),
      ¬ KSAt H V Scope R E κ γ work :=
  ⟨_, _, _, _, _, _, _, Witness.SamplingGeneric.sksL_ksAt_fails H⟩

end Claims.Sampling

namespace Claims.SamplingGeneric

/-- **`Compiled.KSAt` fails** at some compiled verifier. -/
theorem Compiled.KSAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.KSAt H :=
  ⟨_, _, Witness.SamplingGeneric.compiledKSAt_fails H⟩

/-- **`CompilerKSAt`'s conclusion fails** at `sksL`, which accepts with an empty relation. -/
theorem compilerKSAt_nontrivial : ¬ (Witness.SamplingGeneric.sksL Bool).KSAt FlockDeploy.sha512 :=
  Witness.SamplingGeneric.compiledKSAt_fails _

end Claims.SamplingGeneric
