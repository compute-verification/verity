import Definitions.NetworkCertifier.SampledAudit

/-!
# The sampled audit's guarantee: it implies `Accepted`, at `ε` at `δ` over the selection

`SampledAuditAccepted` is the statement, proved in `Proofs.NetworkCertifier.SampledAuditAccepted`. Wherever sampling's
knowledge soundness holds (`Claims.Sampling.KS`), the sampled audit of a program with a public boundary layout implies
`Accepted` by the circuit's Program (`KSAccepted`, `Definitions.NetworkCertifier.SampledAudit`). The free advice is held
at the faulty units' committed gates only, and the faulty set is in `M_δ` and meets no class drawn `all`. Its one
hypothesis is `KS` at the same deployment, the sampling claim it rests on; every condition on the run is either a
verifier refusal (the scope, `ScopeAcc`) or a condition, inside the conclusion, that picks out which observations the
conclusion is about (`Carries`, `Covered`).
-/

namespace NetTiming.Guarantees

open Claims Claims.Sampling

/-- **The sampled audit implies `Accepted`, at `ε` at `δ` over the selection.** At every deployment `D` of the sampled
verifier at which `Claims.Sampling.KS` holds, and every public layout `lay` and horizon `k` of its statements,
`KSAccepted` holds. On the statements in scope (the sampling scope, and the layout fits the program), at every `δ ≥ 0`:
`Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind`, at `KS`'s errors. The relation `RAcc δ` says three things:

* the faulty set is in `M_δ`;
* the faulty set holds no unit of a class drawn `all` (the transfer-gate units and N2's head checks: error `0`);
* the advice whose inputs are the trace's committed values, holding values exactly at the faulty units' committed
  gates, makes `Accepted` by the circuit's Program every observation sequence and operator that the trace's transfer gates
  carry with every status complete.

With no unit faulty, the advice holds no gate, and `Accepted` is exact. -/
def SampledAuditAccepted : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : AuditStatement → Layout L S) (k : AuditStatement → ℕ),
    KS H D → KSAccepted H D L K S lay k

end NetTiming.Guarantees
