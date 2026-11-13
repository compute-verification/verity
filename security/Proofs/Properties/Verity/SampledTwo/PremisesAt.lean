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
* `inScope_of_formAccepts`: the two-stage scope at the connected program holds every statement that decodes;
* `acceptsPlay_of_playAccepts`: SPOT[Flock]'s verifier at the held copies accepts a play
  (`SpotFlock.Honest.AcceptsPlay`), where the form's accept is a play SPOT[Flock] accepts on the coins the check read
  (`PlayAccepts`) and some coins are ones the check read (`SeqCoinsRead`);
* `honest_of_formAccepts`: so the two-stage guarantees are not vacuous (`SpotFlock.Honest.HonestExists`) where the
  form accepts one play;
* `ks_of_formAccepts`: the knowledge soundness the session's leg takes as a hypothesis (`Live₂.KS`), from the
  premises (`Live₂.ks_of_connected`);
* `scope_of_accept`: the statement of a play the session's verifier accepts is in the scope the leg's error is
  bounded at (`Live₂.L`'s).

## What this does not show

`HeldOfAt` and `PlayAccepts`; that the held copies' reference reads are well formed (`SpotFlock.RefReadsWf`); that
the registered wiring's check passes (`SpotFlock.AddressRow.wiresChecked`); that some coins start with the bytes the
accepted play's run read and agree with its sessions' (`SeqCoinsRead`); that the form accepts a play. Each is a
hypothesis.
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

/-- **SPOT[Flock]'s verifier at the held copies accepts a play** (`SpotFlock.Honest.AcceptsPlay`): at one play the
form's check accepts, whose records' bytes are the statement `x` and round 1's message `m₁`, the run of record is a
play SPOT[Flock] accepts on every coins the check read (`PlayAccepts`), and some coins are such. -/
theorem acceptsPlay_of_playAccepts {A : Live₂} {C : CaseOf} {S : SessionReading}
    {P' : Verity.Exchange → ℕ → Claims.Prover Unit} (hplay : PlayAccepts A C S P') {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} {x m₁ : Msg} (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) (hx : recordMsg i.record = some x) (hm : recordMsg it.record = some m₁)
    (hc : ∀ m₂, S.stmts o = some m₂ → ∃ c₀ c₁ r', SeqCoinsRead A S o i it m₂ c₀ c₁ r') :
    SpotFlock.Honest.AcceptsPlay A.h := by
  obtain ⟨m₂, hm₂, -, hall⟩ := hplay o i it x m₁ hA hi hit hx hm
  obtain ⟨c₀, c₁, r', hr⟩ := hc m₂ hm₂
  have hacc := hall c₀ c₁ r' hr
  unfold seqRun at hacc
  exact ⟨_, _, _, _, _, hacc⟩

/-- **The two-stage guarantees are not vacuous where the form accepts one play** (`SpotFlock.Honest.HonestExists`,
at the session's held copies): they meet the premises (`premises_of_formAccepts`), their verifier accepts the play
(`acceptsPlay_of_playAccepts`), and the play's statement is in scope (`SpotFlock.Honest.inScope_of_play`). -/
theorem honest_of_formAccepts {A : Live₂} {C : CaseOf} {S : SessionReading}
    {P' : Verity.Exchange → ℕ → Claims.Prover Unit} (hof : HeldOfAt A C) (hplay : PlayAccepts A C S P')
    {o : Verity.Exchange} {i : Spot.Run.Input} {it : Spot.Run.Interior} {x m₁ : Msg} (hA : FormAccepts C o)
    (hi : inputOn C o = some i) (hit : i.interior = some it) (hx : recordMsg i.record = some x)
    (hm : recordMsg it.record = some m₁)
    (hc : ∀ m₂, S.stmts o = some m₂ → ∃ c₀ c₁ r', SeqCoinsRead A S o i it m₂ c₀ c₁ r')
    (hWf : SpotFlock.RefReadsWf A.h) (hwc : SpotFlock.AddressRow.wiresChecked A.h = true) :
    SpotFlock.Honest.HonestExists := by
  have hP := acceptsPlay_of_playAccepts hplay hA hi hit hx hm hc
  exact ⟨A.h, premises_of_formAccepts hof hA hi hit hWf hwc, SpotFlock.Honest.inScope_of_play hP, hP⟩

/-- **The session's knowledge soundness, from one accepted play** (`Live₂.KS`, the hypothesis the leg takes): the
session's held copies meet the two premises `Flock.Guarantees.SpotKSTwo` takes (`premises_of_formAccepts`), which give
it at the connected program (`Live₂.ks_of_connected`). -/
theorem ks_of_formAccepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) (hWf : SpotFlock.RefReadsWf A.h)
    (hwc : SpotFlock.AddressRow.wiresChecked A.h = true) : A.KS := by
  obtain ⟨hB, hAd⟩ := premises_of_formAccepts hof hA hi hit hWf hwc
  exact A.ks_of_connected hB hAd

/-- **The statement of a play the session's verifier accepts is in the session's scope** (`Live₂.L`'s, which the
session's error is bounded at: `Live₂.prBad_le`), at any hash. -/
theorem scope_of_accept {A : Live₂} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((SpotFlock.V A.h).k x) → FlockDeploy.Coin}
    (hacc : (Claims.run H P (SpotFlock.V A.h) x ω r).out = .acc) : A.L.Scope x :=
  SpotFlock.Honest.scope_of_play H hacc

end Verity.SampledTwo
