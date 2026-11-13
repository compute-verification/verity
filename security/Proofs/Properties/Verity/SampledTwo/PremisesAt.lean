module

public import Proofs.Properties.Verity.SampledTwo.AcceptsTop
public import Proofs.Flock.SpotFlock.Honest.AtHeldOf

/-!
# The premises `SpotKSTwo` takes, at the session of record, from one accepted play

`Flock.Guarantees.SpotKSTwo` takes two premises of the held copies: the frame is SPOT's own at the wired program and
the addresses are connected (`SpotFlock.Honest.Premises`). Where each accepted play's held copies are the ones
`heldOf` builds (`HeldOfAt`), one accepted play gives both at the session's held copies:

* `premises_of_formAccepts`: the premises, from the play's run having passed its population check
  (`RunsAs.population_of_accepted`) and its first stage being of a count `K` (`firstK_of_formAccepts`), with the
  frame and the always-checked blocks by `heldOf`'s construction (`SpotFlock.Honest.premises_of_heldOf`);
* `inScope_of_formAccepts`: the two-stage scope at the connected program holds every statement that decodes.

## What this does not show

`HeldOfAt`; that the held copies' reference reads are well formed (`SpotFlock.RefReadsWf`); that the registered
wiring's check passes (`SpotFlock.AddressRow.wiresChecked`). Each is a hypothesis.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **The premises `SpotKSTwo` takes hold of the session's held copies**, wherever a play is accepted and the
accepted plays' held copies are the ones `heldOf` builds. -/
theorem premises_of_formAccepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) (hWf : SpotFlock.RefReadsWf A.h)
    (hwc : SpotFlock.AddressRow.wiresChecked A.h = true) : SpotFlock.Honest.Premises A.h := by
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨j, hj⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨n, -, hp⟩ := SpotFlock.RunsAs.population_of_accepted hj
  obtain ⟨K, hK⟩ := firstK_of_formAccepts hof hA hi hit
  exact SpotFlock.Honest.premises_of_heldOf hu hWf hp hK hwc

/-- **The two-stage scope at the connected program is inhabited at the session's held copies**, by every statement
that decodes, wherever a play is accepted and the accepted plays' held copies are the ones `heldOf` builds. -/
theorem inScope_of_formAccepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) {x : Msg} {st : SpotFlock.Roots} (hx : SpotFlock.decode x = some st) :
    SpotFlock.Honest.InScope A.h := by
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨j, hj⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨n, -, hp⟩ := SpotFlock.RunsAs.population_of_accepted hj
  exact SpotFlock.Honest.inScope_of_heldOf hu hp hx

end Verity.SampledTwo
