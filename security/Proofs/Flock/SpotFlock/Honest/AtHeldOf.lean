module

public import Proofs.Flock.SpotFlock.Honest.Any
public import Proofs.Flock.SpotFlock.HeldOfRun

/-! The two-stage premises at the copies `heldOf` builds (`honest_of_heldOf`); that it builds a copy at an accepted
run, its first stage's count `K`, the wiring's check and a play the verifier accepts are hypotheses. -/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open SpotFlock.HeldOf (Unread heldOf heldOf_spec withFrame twoHolds_of_pop firstOf_program)

theorem builtWired_withFrame (h : Held) (c₀ : Code) :
    BuiltWired (withFrame h c₀) (wireMap (withFrame h c₀)) (returned (withFrame h c₀)) := by
  have e₁ : wireMap (withFrame h c₀) = wireMap h := wireMap_frame _ _
  have e₂ : returned (withFrame h c₀) = returned h := returned_frame _ _
  rw [e₁, e₂]
  exact builtWired_spot h (wireMap h) (returned h) c₀

theorem builtWired_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) : BuiltWired h (wireMap h) (returned h) := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  exact builtWired_withFrame _ _

theorem always_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (v : ℕ) : h.reg.always v = false := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  rfl

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

theorem sampleSlots_of_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true)
    (hK : firstK h.reg.first = some K) : AddressRow.SampleSlots h := by
  have hp' : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true := by
    obtain ⟨first, ns, ks, l, hz, hf, -, -, rfl, -⟩ := heldOf_spec hh
    obtain ⟨hpr, hq⟩ := firstOf_program hf
    show _root_.Spot.Slots.popOk first.program first.query n = true
    rw [hpr, hq]
    exact hp
  exact AddressRow.sampleSlots_of_firstK hp' hK

theorem inScope_of_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) {n : ℕ}
    (hp : _root_.Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true) {x : Msg}
    {st : Roots} (hx : decode x = some st) : InScope h := by
  obtain ⟨s, -⟩ : ∃ s, parse h x = some s := by
    unfold parse
    rw [hx]
    exact ⟨_, rfl⟩
  obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 (twoHolds_of_pop hh hp s)
  exact inScope_of_two hx hd

theorem honest_of_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hWf : RefReadsWf h) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) (hA : AcceptsPlay h) :
    HonestExists :=
  ⟨h, premises_of_heldOf hh hWf hp hK hwc, inScope_of_play hA, hA⟩

end SpotFlock.Honest
