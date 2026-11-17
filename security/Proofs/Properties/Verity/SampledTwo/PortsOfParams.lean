module

public import Proofs.Properties.Verity.SampledTwo.AcceptsTop
public import Proofs.Flock.SpotFlock.ParamsWf

/-!
# A play's registrations list each port once, where its parameters' tree is well formed

`PortsDistinctAt` is a premise of the form's accept joined to the run's (`AcceptsTop`): at each accepted play, the
run of record's input lists no port twice, at its boundary and at its interior. The input's parameters are the form's
boundary's and its interior's the form's interior's (`params_of_inputOfCase`), and each lists no port
twice where the tree it was read from is well formed (`SpotFlock.ParamsWf.ports_nodup_form`). So it holds at every
play whose case has an input, accepted or not.

## What this does not show

That the tree `Core.canonicalJson` returns is well formed: the hypothesis `hcj`, as in `SpotFlock.ParamsWf`.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims.Json (Wf)

/-- **A case's input lists each port once**, at its boundary and at its interior, where `Core.canonicalJson`
returns well-formed trees. -/
theorem ports_of_inputOfCase (hcj : ∀ b j, Core.canonicalJson b = .ok j → Wf j) {k : Case} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} (hi : inputOfCase k = some i) (hit : i.interior = some it) :
    (i.params.verifier.ports.map (·.1)).toList.Nodup ∧ (it.params.verifier.ports.map (·.1)).toList.Nodup := by
  obtain ⟨-, j, p, -, hj, hb, hn⟩ := params_of_inputOfCase hi hit
  rw [hb, hn]
  exact SpotFlock.ParamsWf.ports_nodup_form hcj hj

/-- **`PortsDistinctAt` holds** where `Core.canonicalJson` returns well-formed trees. -/
theorem portsDistinctAt_of (hcj : ∀ b j, Core.canonicalJson b = .ok j → Wf j) (C : CaseOf) : PortsDistinctAt C := by
  intro o i it _ hi hit
  unfold inputOn at hi
  cases hk : C o with
  | none =>
    rw [hk] at hi
    cases hi
  | some k =>
    rw [hk, Option.bind_some] at hi
    exact ports_of_inputOfCase hcj hi hit

end Verity.SampledTwo
