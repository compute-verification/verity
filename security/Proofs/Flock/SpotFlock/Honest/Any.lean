module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressFacts.RowFirst

/-!
# The two-stage premises at any registration, from its decided facts

`Honest.Instance` and `Honest.Sits` state what `SpotKSTwo` and `SpotExfilTwo` take at one registration built for
them (`held R B desc q c₀`: one template, no wires). Neither conclusion reads that shape. At any auditor's copies
`h`:

* `scope_of_two`: a statement that parses is in the two-stage scope at the compiled code where the design evaluates
  (`two`). Each wire's writer is then first (`wiresFirst_of_two`), so the code's wiring map rewires the members'
  programs (`rewires_of_two`), and a rewiring keeps the two-stage scope at the wired program, the partition and the
  design's replay partition cut (`scope_rewired`);
* `inScope_of_two`, `inScope_of_run`: so some statement is in scope where the design evaluates and a statement
  decodes, and where a run is accepted, whose design evaluated (`CheckedRun.Ok`);
* `premises_of_checks`: the premises, where the frame is SPOT's own at the wired program (`BuiltWired`), from five
  facts of the auditor's copies (`addressesConnected_of_firstK`): the reference reads are well formed, no proof
  block is always checked, the boundary registration passes the run's population check, its first stage samples a
  uniform `K`-subset, and the condition decided on the registered wires holds;
* `honest_of_run`: `HonestExists` from those five facts and one accepted run.

So a registration with wires, several templates and several replay blocks is an instance of `HonestExists` where a
run of it is accepted: the design's own decided conditions (`scopeOk`, `filesOk`, `templateOk`, `membersScoped`,
`wiringOk`) are not asked again, since the accepted run's design evaluated.

## What this does not show

* An accepted run (`Accepts`). `CheckedRun.Ok` holds a witness that Flock's relation holds of a checked unit's
  session statement, which no check of the verifier's gives: it is the prover's.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot

/-- **A statement that parses is in the two-stage scope at the compiled code, where the design evaluates**, at any
auditor's copies: the statement at the members' programs wired by the code's wiring map, its partition and its
design's replay partition cut (`scope_rewired`), the map rewiring the members' programs (`rewires_of_two`). -/
theorem scope_of_two {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) : (compiled h).Scope x := by
  obtain ⟨l, -, -, -, hsc, -⟩ := two_eq h h2
  have hP : partition h s = some (membersPart h.ms) := rfl
  have hW := rewires_of_two h2 (wiresFirst_of_two h h2)
  have hs := scope_rewired h hp hP hW (scopeOk_sound hsc s (parse_prog h hp))
  rw [parse_prog h hp] at hs
  refine ⟨_, _, _, ?_, ?_, ?_, hs⟩
  · rw [parse_compiled]
    unfold parseWired
    rw [hp]
    rfl
  · rw [partition_compiled]
    rfl
  · rw [two_compiled]
    show (two h s).map (fun d => ({ d with part := d.part.cut (wireMap h) } : Design.Two)) = _
    rw [h2]
    rfl

/-- **Some statement is in the two-stage scope where the design evaluates**: any statement that decodes. -/
theorem inScope_of_two {h : Held} {x : Msg} {st : Roots} (hx : decode x = some st) {s : AuditStatement}
    {d : Design.Two} (h2 : two h s = some d) : InScope h := by
  obtain ⟨s₁, hs₁⟩ : ∃ s₁, parse h x = some s₁ := by simp [parse, hx]
  have h2' : two h s₁ = some d := h2
  exact ⟨x, scope_of_two hs₁ h2'⟩

/-- **An accepted run's statement is in the two-stage scope at the compiled code**, at any auditor's copies: the
run's design evaluated. -/
theorem inScope_of_run {h : Held} (hA : Accepts h) : InScope h := by
  obtain ⟨r, Pt, d, spec, hp, -, h2, -⟩ := hA
  exact ⟨r.x, scope_of_two hp h2⟩

/-- **The premises `SpotKSTwo` takes, at any auditor's copies whose frame is SPOT's own at the wired program**, from
five facts of the copies: the reference reads are well formed, no proof block is always checked, the boundary
registration passes the run's population check at `n` units, its first stage samples a uniform `K`-subset, and the
condition decided on the registered wires holds (`addressesConnected_of_firstK`). -/
theorem premises_of_checks {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) : Premises h :=
  ⟨hB, addressesConnected_of_firstK hWf hna hp hK hwc⟩

/-- **`HonestExists`, from five facts of the auditor's copies and one accepted run**: the copies meet the premises
(`premises_of_checks`), and the accepted run's statement is in scope (`inScope_of_run`). -/
theorem honest_of_run {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) (hA : Accepts h) :
    HonestExists :=
  ⟨h, premises_of_checks hB hWf hna hp hK hwc, inScope_of_run hA, hA⟩

end SpotFlock.Honest
