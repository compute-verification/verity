module

public import Proofs.Flock.Claims.Witnesses.SamplingGenericKSAt
public import Proofs.Flock.Claims.Witnesses.SamplingTwo

/-!
# Refutation witnesses: the two-stage extractor's cost and the two-stage KS read at a time budget

`CompiledTwo.KSAt` fails at `sksLTwo` (`Witnesses.SamplingTwo`), `sksL` with the two-stage design `toyTwo`: the empty
statement is in its scope and no witness is in its relation at `δ = 0` (`sksLTwo_scope`, `sksLTwo_noR`), so
`Sampling.KSAt` fails there as at `sksL` (`sksL_ksAt_fails_at`), and with it `CompilerKSAtTwo`'s conclusion
(`compilerKSAtTwo_nontrivial`). `CompilerEfficientTwo`'s conclusion is `Compiled.Efficient` at the two-stage verifier's
own `Compiled`, so it fails at `effLTwo`, `effL` (`Witnesses.SamplingGenericCost`) with that design
(`compilerEfficientTwo_nontrivial`). Both guarantees' premises hold together at `twoLTwo`, with two sampled sessions
(`compilerKSTwo_honest`, and `compilerBits_honest` at its `Compiled`, `twoL`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric

/-- `effL` with the design `toyTwo`, its first stage sampling no replay block. -/
noncomputable def effLTwo (Coin : Type) : CompiledTwo Coin (findA Coin) where
  toCompiled := effL Coin
  two _ := some toyTwo
  replaySample _ _ := []

/-- **`CompiledTwo.KSAt` fails** at `sksLTwo`: its first half, at `δ = 0`. -/
theorem compiledTwoKSAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).KSAt H := by
  intro h
  refine sksL_ksAt_fails_at H ?_ ?_ (h.1 0 le_rfl)
  · exact sksLTwo_scope []
  · exact sksLTwo_noR H []

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

open Claims.Witness.SamplingGeneric

/-- **`CompiledTwo.KSAt` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.KSAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.KSAt H :=
  ⟨_, _, compiledTwoKSAt_fails H⟩

/-- **`CompilerKSAtTwo`'s conclusion fails** at `sksLTwo`, which accepts with an empty relation. -/
theorem compilerKSAtTwo_nontrivial : ¬ (sksLTwo Bool).KSAt FlockDeploy.sha512 := compiledTwoKSAt_fails _

/-- **`CompilerEfficientTwo`'s conclusion fails** at `effLTwo`, whose extractor is `effL`'s. -/
theorem compilerEfficientTwo_nontrivial : ¬ (effLTwo Bool).Efficient FlockDeploy.sha512 := compiledEfficient_fails _

end Claims.SamplingGeneric
