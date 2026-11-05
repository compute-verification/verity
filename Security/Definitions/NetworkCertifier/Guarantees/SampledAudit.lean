import Definitions.NetworkCertifier.SampledAudit

/-!
# The sampled audit's guarantee: it implies `Accepted`, at `ε` at `δ` over the selection

`SampledAuditAccepted` is the statement, proved in `Proofs.NetworkCertifier.SampledAuditAccepted`. Wherever sampling's
knowledge soundness holds at the incorrect set (`KSWired`, at the layout's wiring), the sampled audit of a program with
a public boundary layout implies `Accepted` by the circuit's Program (`KSAccepted`,
`Definitions.NetworkCertifier.SampledAudit`). The free advice is held at the incorrect proof blocks' committed gates
only, and the incorrect set is in `M_δ` and meets no class sampled `all`. Its one hypothesis is `KSWired` at the same
deployment, the sampling claim it rests on, which is `Claims.Sampling.KS` at a layout with no wires; every condition on
the run is either a verifier refusal (the scope, `ScopeAcc`) or a condition, inside the conclusion, that picks out
which observations the conclusion is about (`Carries`, `Covered`).
-/

namespace NetTiming.Guarantees

open Claims Claims.Sampling

/-- **The sampled audit implies `Accepted`, at `ε` at `δ` over the selection.** At every deployment `D` of the sampled
verifier, and every public layout `lay` and horizon `k` of its statements, at which sampling at the incorrect set holds
(`KSWired`, at the layout's wiring), `KSAccepted` holds. On the statements in scope (the sampling scope, and the layout
fits the program, its wiring included), at every `δ ≥ 0`: `Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind`, at `KSWired`'s
errors. The relation `RAcc δ` says three things:

* the incorrect set (the faulty and the wire-wrong proof blocks) is in `M_δ`;
* the incorrect set holds no proof block of a class sampled `all` (the transfer-gate proof blocks and N2's head checks:
  error `0`);
* the advice whose inputs are the trace's committed values, holding values exactly at the incorrect proof blocks'
  committed gates, makes `Accepted` by the circuit's Program, each wired gate reading its writer, every observation
  sequence and operator that the trace's transfer gates carry with every status complete.

With no proof block incorrect, the advice holds no gate, and `Accepted` is exact. -/
def SampledAuditAccepted : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ),
    KSWired H D (fun s => (lay s).wires) → KSAccepted H D L K S lay k

end NetTiming.Guarantees
