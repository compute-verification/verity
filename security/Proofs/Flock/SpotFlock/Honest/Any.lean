module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressFacts.RowFirst

/-!
# The two-stage premises at any registration, from its decided facts and one accepted run

`Honest.Instance` and `Honest.Sits` state what `SpotKSTwo` and `SpotExfilTwo` take at one registration built for
them (`held R B desc q c₀`: one template, no wires). Neither conclusion reads that shape. At any auditor's copies
`h`:

* `inScope_of_run`: an accepted run's statement is in the two-stage scope at the compiled code. The run's design
  evaluated (`CheckedRun.Ok`), so each wire's writer is first (`wiresFirst_of_two`), the code's wiring map rewires
  the members' programs (`rewires_of_two`), and a rewiring keeps the two-stage scope at the wired program, the
  partition and the design's replay partition cut (`scope_rewired`);
* `premises_of_checks`: the premises, where the frame is SPOT's own at the wired program (`BuiltWired`), from five
  facts of the auditor's copies (`addressesConnected_of_firstK`): the reference reads are well formed, no proof
  block is always checked, the boundary registration passes the run's population check, its first stage samples a
  uniform `K`-subset, and the condition decided on the registered wires holds;
* `framed`, `builtWired_framed`: any copies, given SPOT's own frame at the wired program, are `BuiltWired`;
* `honest_of_run`: `HonestExists` from those five facts and one accepted run.

So a registration with wires, several templates and several replay blocks is an instance of `HonestExists` where a
run of it is accepted: the design's own decided conditions (`scopeOk`, `filesOk`, `templateOk`, `membersScoped`,
`wiringOk`) are not asked again, since the accepted run's design evaluated.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot

/-- **An accepted run's statement is in the two-stage scope at the compiled code**, at any auditor's copies: the
run's statement at the members' programs wired by the code's wiring map, its partition and its design's replay
partition cut (`scope_rewired`), the map rewiring the members' programs where the design evaluates
(`rewires_of_two`). -/
theorem inScope_of_run {h : Held} (hA : Accepts h) : InScope h := by
  obtain ⟨r, Pt, d, spec, hp, hP, h2, hsc, -⟩ := hA
  obtain rfl := partition_eq h hP
  have hW := rewires_of_two h2 (wiresFirst_of_two h h2)
  have hs := scope_rewired h hp hP hW hsc
  rw [parse_prog h hp] at hs
  refine ⟨r.x, _, _, _, ?_, ?_, ?_, hs⟩
  · rw [parse_compiled]
    unfold parseWired
    rw [hp]
    rfl
  · rw [partition_compiled]
    rfl
  · rw [two_compiled]
    show (two h r.s).map (fun d => ({ d with part := d.part.cut (wireMap h) } : Design.Two)) = _
    rw [h2]
    rfl

/-- **The premises `SpotKSTwo` takes, at any auditor's copies whose frame is SPOT's own at the wired program**, from
five facts of the copies: the reference reads are well formed, no proof block is always checked, the boundary
registration passes the run's population check at `n` units, its first stage samples a uniform `K`-subset, and the
condition decided on the registered wires holds (`addressesConnected_of_firstK`). -/
theorem premises_of_checks {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) : Premises h :=
  ⟨hB, addressesConnected_of_firstK hWf hna hp hK hwc⟩

/-- **The auditor's copies `h₀` with SPOT's own frame at the wired program**, from the code `c₀`. -/
noncomputable def framed (h₀ : Held) (c₀ : Code) : Held :=
  { h₀ with
    frame := frameAt (slotsOf (addressMapWired h₀ (wireMap h₀) (returned h₀)))
      (revealCodeWired h₀ (wireMap h₀) (returned h₀) (addressMapWired h₀ (wireMap h₀) (returned h₀)) c₀) }

/-- **Any copies given SPOT's own frame are `BuiltWired`**: neither the wiring map nor the gates the boundary
record holds reads the frame. -/
theorem builtWired_framed (h₀ : Held) (c₀ : Code) :
    BuiltWired (framed h₀ c₀) (wireMap (framed h₀ c₀)) (returned (framed h₀ c₀)) := by
  have e₁ : wireMap (framed h₀ c₀) = wireMap h₀ := wireMap_frame _ _
  have e₂ : returned (framed h₀ c₀) = returned h₀ := returned_frame _ _
  rw [e₁, e₂]
  exact builtWired_spot h₀ (wireMap h₀) (returned h₀) c₀

/-- **The instance is its copies given SPOT's own frame.** -/
theorem held_eq_framed (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code) :
    held R B desc q c₀ = framed (held₀ R B desc q c₀) c₀ := rfl

/-- **`HonestExists`, from five facts of the auditor's copies and one accepted run**: the copies meet the premises
(`premises_of_checks`), and the accepted run's statement is in scope (`inScope_of_run`). -/
theorem honest_of_run {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) (hA : Accepts h) :
    HonestExists :=
  ⟨h, premises_of_checks hB hWf hna hp hK hwc, inScope_of_run hA, hA⟩

end SpotFlock.Honest
