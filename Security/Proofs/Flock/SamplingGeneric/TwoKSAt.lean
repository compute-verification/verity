module

public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.KSAt
public import Proofs.Flock.SamplingGeneric.Two

/-!
# The two-stage extractor's costs and the two-stage KS read at a time budget

The two-stage extractor is `Compiled.E` at the two-stage verifier's own `Compiled`, so at its own waits' counts `k_out`
and `k_in`, and the two-stage contract gives the facts its cost reads (`CompiledTwo.Sessions.toCostContract`: the
sessions' runs on their fixed rounds, and the sample's slots, at most `n_PU` holding at most `n_sess` keys). So
`Compiled.efficient_of` bounds its reruns and own calls (`compilerEfficientTwo`), which come to `Compiled.work`, and
`KS` at the footprint gives `KSAt` there, each hash term at a budget on one run's whole SHA-512 work
(`CompiledTwo.ksAt_of_ks`): `CompilerKSTwo` and `CompilerEfficientTwo` give `CompilerKSAtTwo` (`compilerKSAtTwo_of`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Nonempty Coin] in
/-- **The two-stage contract gives the facts the extractor's cost reads**: its `fixed`, `runs`, `slots` and `keys`. -/
theorem Sessions.toCostContract {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) :
    L.toCompiled.CostContract H :=
  { toRunsFixed := ⟨hS.fixed, hS.runs⟩, slots := hS.slots, keys := hS.keys }

/-- **`KSAt` from `KS`**, at an extractor whose expected reruns `runs` and own SHA-512 calls `own` come to at most
`work`. -/
theorem ksAt_of_ks {H : Msg → Msg} {L : CompiledTwo Coin A} (hks : L.KS H) {runs : Msg → ℝ} {own : Msg → ℕ}
    (hruns : ExtractRunsExpLe H L.E runs) (hown : Sampling.ExtractOwnLe H L.E own)
    (hw : ∀ x, runs x + own x ≤ L.work x) : L.KSAt H :=
  ⟨fun δ hδ => Sampling.KSAt_of_KSLinOn L.work (hks δ hδ), Sampling.BextAtCostLe_of hruns hown hw⟩

end CompiledTwo

/-- **`CompilerEfficientTwo`**: `Compiled.efficient_of` at the two-stage contract's cost facts. -/
theorem compilerEfficientTwo : CompilerEfficientTwo := by
  intro Coin _ _ _ A L H hS hopen
  exact Compiled.efficient_of hS.toCostContract hopen

/-- **The two-stage KS read at a time budget**, from `CompilerKSTwo` and `CompilerEfficientTwo`'s expected reruns and
own calls, which come to `work`. -/
theorem compilerKSAtTwo_of (hks : CompilerKSTwo) (heff : CompilerEfficientTwo) : CompilerKSAtTwo := by
  intro Coin _ _ _ A L H hA hB hS hopen
  obtain ⟨-, hr, ho⟩ := heff Coin A L H hS hopen
  exact CompiledTwo.ksAt_of_ks (hks Coin A L H hA hB hS) hr ho
    fun x => le_of_eq (by unfold Compiled.work; push_cast; ring)

/-- **KS read at a time budget at two stages**: `CompilerKSTwo` and `CompilerEfficientTwo` (`compilerKSAtTwo_of`). -/
theorem compilerKSAtTwo : CompilerKSAtTwo := compilerKSAtTwo_of compilerKSTwo compilerEfficientTwo

end Claims.SamplingGeneric
