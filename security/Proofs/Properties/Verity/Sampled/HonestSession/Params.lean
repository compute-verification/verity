module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Inputs
public import Proofs.Properties.Verity.Sampled.HonestSession.JsonCanon

/-!
# v2's parameters in `θ₀`

`θ₀` is canonical and decodes to one component whose parameters `paramsOfJson` reads at the one-stage form (`params_spec`):
the verifier's partition object and program are `partitionObj` and `programFile`, its program, query, law and work table
the session's, and its challenger the deployed key at `label₀`. The digests it names are the circuit and public files'
(`digests`). Every step rewrites by `Flock.canon`'s equation lemmas and the JSON facts (`JsonCanon.JsonFacts`), taken as
a hypothesis here; nothing is decided.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Params

open Flock

theorem params_spec (hJ : JsonCanon.JsonFacts) : ∃ p, ParamsAt θ₀ p ∧ p.own.partition = partitionObj ∧
    p.own.program = programFile ∧ p.base.verifier.program = program ∧ p.base.verifier.query = query ∧
    p.base.verifier.law = law ∧ p.base.verifier.workTable = none ∧ p.base.challenger = some (key₀, label₀) := by
  sorry -- piece P7 (parameters)

theorem digests (hJ : JsonCanon.JsonFacts) :
    programDigestOf θ₀ = some (Verity.sha512 circuitFile.data.toList) ∧
      publicDigestOf θ₀ = some (Verity.sha512 publicFile.data.toList) := by
  sorry -- piece P7 (parameters)

end Verity.Sampled.Instance.HonestSession.Params
