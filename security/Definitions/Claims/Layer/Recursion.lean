module

public import Definitions.Claims.RecursionGeneric

/-!
# The recursion layer: its interface and its theorem's shape

The recursion layer proves an inner argument Π_in's acceptance with an outer argument Π_out, run on Π_in's opened view
behind a server. Its theorem is `Claims.RecGeneric.CompositionKS` (L1, generic in the outer extraction `E_O`) and, for
any Π_out, `Claims.RecGeneric.CompilerKS` (L2), both in `Definitions/Claims/RecursionGeneric.lean`; this module restates
neither.

**What it takes.** Π_in knowledge sound against aborting provers in run-finder form (`Proto.KSAbortLin`, the argument
layer's `ProofSystem.KSAbort`), well formed and reading no last coin; a binding server (`Server.Binds`); and Π_out's
extraction, knowledge sound at `κ_O(x)`, `γ_O(x)` (L1, `Comp.OuterKSAt`) or a Π_out whose extractor is knowledge sound
(L2, `Arg.KS`), with the compiler's contract.

**What it gives.** On every prover and loop bound, `Pr[accept ∧ ¬R(x, E*(x))] ≤ Comp.err`, with `E*` Π_in's
extractor served by `E_O`.

**Its error, in the layer below's.** `Comp.err = κ + √(γ · Pr_B) + Σ_i w_i·(F_i + tail_i) + reads·Pr_out + Pr[B_I] +
β·(Pr[F_dev] + 1/(M+1) + Pr_out)`, with `κ = κ_Π + N_I·κ_O`, `γ = N_I²·γ_O` and `N_I = 1 + m_Π·k_Π`: Π_in's knowledge
error and terms, Π_out's at each of Π_in's `N_I` reruns. Under L2, `κ_O = Σ_s κb_s` and `γ_O = (Σ_s √(γb_s·k(x)))²` over
Π_out's sessions.

**Instances.**
* Generic: `Claims.RecGeneric.compositionKS` and `compilerKS` prove L1 and L2; proved.
* Flock in Flock: `Flock.Guarantees.RecursiveKSEvery` (L1 at Flock) and `RecursiveKSCompiledEvery` (L2 at Flock), proved;
  `RecursiveKSDeployed`, deployed and proved.
* Over the sampling layer: `Flock.Guarantees.RecursiveSampledKS`, from L1; proved.
* Zero knowledge kept: the ZK layer (`Definitions.Core.Layer.Zk`).

**Duplicates, for next week.** `Claims.RecGeneric.Proto` and `Arg` state the argument system again beside
`Verity.ProofSystem` (`KSAbortLin` beside `ProofSystem.KSAbort`).
-/
