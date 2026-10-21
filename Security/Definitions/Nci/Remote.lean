import Definitions.Nci.Dampener

/-!
# The compute dampener, remote protocol: the vocabulary

The remote protocol's dampener (NCI, PoUW and network accounting, with no memory layer: `benchmarks/nci/tiers.py` at
tiers T1 and T2, with no wipes and no sampling; "The dampener ρ, remote protocol", Notion) is
`1/ρ = s + (r̂ + ε' + JΔ/H + ν)/MFU₀` with `ρ ≥ 1`. Two of its terms come from reading the monitored compute residual
`r̂ = 1 − W_cert/C` (`residualShare`) against what the developer can do outside it:

* the certificate slack `ε'`: PoUW certifies each window, and a window's certificate may be wrong, so the work certified
  correctly falls short of `W_cert` by at most the capacity of the windows whose certificates are wrong
  (`wrongCapacity`), which is at most `ε' C` outside an event of probability (`prob`) at most `p/ε'`;
* the verdict lag `JΔ/H` (`lagShare`): from an identity's first faulty proof block until the halt `Δ` later, its
  developer skips declared work, once for each of its `J` identities, over the horizon `H`.

`remoteDampener` is the hardware multiple the bound gives, as `tiers.evaluate` computes `dampener` and `fit.dampener`
computes it at `s = 0`; `hardwareMultiple` is the one a developer needs.
-/

namespace Nci.Dampener

/-- **The probability of an event** `E` under a distribution `μ` on a finite outcome space, as a real number:
`∑_{ω ∈ E} μ(ω)`. -/
noncomputable def prob {Ω : Type} [Fintype Ω] (μ : PMF Ω) (E : Ω → Prop) : ℝ :=
  open scoped Classical in ∑ ω, if E ω then (μ ω).toReal else 0

/-- **The capacity of the windows whose certificates are wrong**: `∑_{i : wrong i} c_i`, window `i` of `N` having
capacity `c_i`. -/
noncomputable def wrongCapacity {N : ℕ} (c : Fin N → ℝ) (wrong : Fin N → Prop) : ℝ :=
  open scoped Classical in ∑ i, if wrong i then c i else 0

/-- **The verdict lag's share** `JΔ/H`: `J` identities, each skipping declared work for at most `Δ` of the horizon
`H`. -/
noncomputable def lagShare (J Δ H : ℝ) : ℝ := J * Δ / H

/-- **The hardware multiple a developer needs**, `max(1, MFU₀/η)`: the hardware-time it spends, `C + U` FLOP at peak,
over the hardware-time an unconstrained developer at model-FLOP utilization `MFU₀` needs for the same `M` counted
multiplications, `(2/κ) M/MFU₀`; at least 1, since the unconstrained developer has the same recipe and hardware. -/
noncomputable def hardwareMultiple (mfu κ M C U : ℝ) : ℝ := max 1 (mfu / efficiency κ M C U)

/-- **The dampener** `ρ = max(1, 1/(s + x/MFU₀))` (`tiers.evaluate`'s `dampener`; `fit.dampener` at `s = 0`, where it
is `max(1, MFU₀/x)`): unmonitored compute, the share `s`, trains like the unconstrained developer, at `MFU₀`, and the
other terms `x` are counted at peak. -/
noncomputable def remoteDampener (mfu s x : ℝ) : ℝ := max 1 (1 / (s + x / mfu))

end Nci.Dampener
