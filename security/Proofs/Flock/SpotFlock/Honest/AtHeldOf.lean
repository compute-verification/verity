module

public import Proofs.Flock.SpotFlock.Honest.Any
public import Proofs.Flock.SpotFlock.HeldOfRun

/-!
# The two-stage premises at the copies `heldOf` builds

`Honest/Any.lean` states the premises `SpotKSTwo` takes, and the two-stage scope, at any auditor's copies. The copies
the two-stage form's check runs at are the ones `HeldOf.heldOf` builds from a run's input, its interior and what the
run does not read. Two of `premises_of_checks`' hypotheses hold of those by construction, and one more reads the run's
input rather than the copies:

* `builtWired_withFrame`, `builtWired_heldOf`: the frame is SPOT's own at the wired program (`BuiltWired`), since
  `heldOf` sets it with `HeldOf.withFrame`;
* `always_heldOf`: no proof block is always checked;
* `premises_of_heldOf`: the premises, from the population check at the run's own program and query
  (`Slots.popOk`, which `Run.run` refuses without), a first stage of a count `K` (`firstK`), the registered wiring's
  check (`AddressRow.wiresChecked`) and the reference reads being well formed (`RefReadsWf`);
* `inScope_of_heldOf`: the two-stage scope holds every statement that decodes, from the population check alone.

## What this does not show

That `heldOf` builds a copy at an accepted run (`heldOf i it u = some h`), that its first stage is of a count `K`, or
that the registered wiring's check passes: each is a hypothesis here. An accepted run: `Any.lean`'s `accepts_of` takes
a witness in Flock's relation, which is the prover's.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open SpotFlock.HeldOf (Unread heldOf heldOf_spec withFrame twoHolds_of firstOf_program)

/-- **SPOT's own frame, set on any copies, is built at the wired program**: neither the wire map nor the gates the
boundary record holds read the frame. -/
theorem builtWired_withFrame (h : Held) (c₀ : Code) :
    BuiltWired (withFrame h c₀) (wireMap (withFrame h c₀)) (returned (withFrame h c₀)) := by
  have e₁ : wireMap (withFrame h c₀) = wireMap h := wireMap_frame _ _
  have e₂ : returned (withFrame h c₀) = returned h := returned_frame _ _
  rw [e₁, e₂]
  exact builtWired_spot h (wireMap h) (returned h) c₀

/-- **The frame of the copies `heldOf` builds is SPOT's own at the wired program.** -/
theorem builtWired_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) : BuiltWired h (wireMap h) (returned h) := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  exact builtWired_withFrame _ _

/-- **No proof block is always checked at the copies `heldOf` builds.** -/
theorem always_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (v : ℕ) : h.reg.always v = false := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  rfl

/-- **The premises `SpotKSTwo` takes, at the copies `heldOf` builds**: `premises_of_checks` with the frame and the
always-checked blocks by construction, and the population check read at the run's own program and query, which are
the boundary registration's (`HeldOf.firstOf_program`). -/
theorem premises_of_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hWf : RefReadsWf h) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) : Premises h := by
  have hp' : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true := by
    obtain ⟨first, ns, ks, l, hz, hf, -, -, rfl, -⟩ := heldOf_spec hh
    obtain ⟨hpr, hq⟩ := firstOf_program hf
    show _root_.Spot.Slots.popOk first.program first.query n = true
    rw [hpr, hq]
    exact hp
  exact premises_of_checks (builtWired_heldOf hh) hWf (always_heldOf hh) hp' hK hwc

/-- **The two-stage scope is inhabited at the copies `heldOf` builds**, by every statement that decodes, where the
run's population check passes: the design evaluates there (`HeldOf.twoHolds_of`). -/
theorem inScope_of_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) {n : ℕ}
    (hp : _root_.Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true) {x : Msg}
    {st : Roots} (hx : decode x = some st) : InScope h := by
  obtain ⟨s, -⟩ : ∃ s, parse h x = some s := by
    unfold parse
    rw [hx]
    exact ⟨_, rfl⟩
  obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 (twoHolds_of hh hp s)
  exact inScope_of_two hx hd

end SpotFlock.Honest
