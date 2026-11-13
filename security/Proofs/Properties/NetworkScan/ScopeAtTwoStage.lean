module

public import Proofs.Flock.SpotFlock.KS
public import Proofs.Flock.Claims.Sampling.EndToEndDeploy

/-!
# The one-stage scope at SPOT[Flock]'s two-stage parser

A scratch module. The one-stage sampling scope (`Claims.SamplingGeneric.Compiled.Scope`) asks a design with as many
rules as the partition has classes (`Design.WellFormed`). A statement of the two-stage parser (`SpotFlock.parse`)
holds the first stage's design, the two rules of `slotDesign` or none, and the two-stage partition
(`SpotFlock.partition`) is `membersPart`, of one class. So the one-stage scope holds at no statement there.

* `replayDesign_rules`: the statement's design has two rules or none.
* `not_inScope_parse`: no statement of the two-stage parser is in the one-stage scope at the two-stage partition.
* `not_scope`: `Compiled.Scope` holds at no statement of a compiled verifier with that parser and partition.
* `not_scope_spot`: so at none of SPOT[Flock]'s deployed two-stage verifier.
-/

@[expose] public section

namespace ScanScopeAtTwoStage

open Claims Claims.Sampling Properties.Spot

/-- **The two-stage statement's design has two rules or none**, never one. -/
theorem replayDesign_rules (h : SpotFlock.Held) : ((SpotFlock.replayDesign h).getD ⟨[]⟩).rules.length ≠ 1 := by
  cases hr : SpotFlock.replayDesign h with
  | none => simp
  | some d =>
    obtain ⟨first, hd, -⟩ := Claims.Sampling.EndToEndDraw.slotDesign_rules (SpotFlock.ok_of_toOption hr)
    simp [hd]

/-- **No statement of the two-stage parser is in the one-stage scope** at the two-stage partition. -/
theorem not_inScope_parse (h : SpotFlock.Held) {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hs : SpotFlock.parse h x = some s) (hP : SpotFlock.partition h s = some Pt) : ¬ InScope Pt s := by
  rintro ⟨-, -, hlen, -⟩
  obtain rfl := SpotFlock.partition_eq h hP
  unfold SpotFlock.parse at hs
  obtain ⟨st, -, rfl⟩ := Option.map_eq_some_iff.1 hs
  have h1 : (membersPart h.ms).classes.length = 1 := rfl
  exact replayDesign_rules h (hlen.trans h1)

/-- **`Compiled.Scope` holds at no statement** of a compiled verifier whose parser and partition are SPOT[Flock]'s
two-stage ones. -/
theorem not_scope {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (D : Claims.SamplingGeneric.Compiled Coin A)
    (h : SpotFlock.Held) (hD : D.parse = SpotFlock.parse h) (hDp : D.partition = SpotFlock.partition h) (x : Msg) :
    ¬ D.Scope x := by
  rintro ⟨s, P, hs, hP, hin⟩
  rw [hD] at hs
  rw [hDp] at hP
  exact not_inScope_parse h hs hP hin

/-- **At SPOT[Flock]'s deployed two-stage verifier `Compiled.Scope` holds at no statement.** -/
theorem not_scope_spot (h : SpotFlock.Held) (x : Msg) :
    ¬ (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
      SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h) (SpotFlock.cap h)).toCompiled.Scope x :=
  not_scope _ h rfl rfl x

end ScanScopeAtTwoStage
