module

public import Proofs.Properties.Verity.SampledTwo.PremisesAt
public import Proofs.Core.CanonicalJson
public import Proofs.Flock.Claims.Complete.Json.RoundTrip

/-!
# The wired statement's audit statement, partition and design

The session's verifier (`Live₂.L`) reads a statement at the connected program of the held copies: it parses the
statement (`SpotFlock.parseWired`), cuts the members' partition (`SpotFlock.partitionWired`) and evaluates the
two-stage design (`SpotFlock.twoConnected`). At the statement a commitment carries (`stmtOf`), each evaluates:

* `partition_wired`: the partition, at every audit statement: the members' partition does not read it;
* `two_wired`: the design, at every audit statement, where the held copies' design evaluates (`TwoHolds`), which one
  accepted play gives under `HeldOfAt` (`twoHolds_of_formAccepts`);
* `recordGood`, `decode_of_stmtOf`, `parse_wired`: the statement, at every commitment. The commitment's record is what
  `Core.canonicalJson` returned of the bytes, which is printable (`Claims.Json.Good`,
  `Verity.CanonicalJson.canonicalJson_good`); `Flock.canon` printed the statement of it, and the proved verifier's
  reader reads back what `canon` prints of a printable value (`FlockComplete.JsonRT.parse_canon`).

## What this does not show

That the held copies' design evaluates where no play is accepted.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **The wired partition evaluates at every audit statement**: the members' partition, cut by the code's wiring map. -/
theorem partition_wired (A : Live₂) (s : Claims.Sampling.AuditStatement) : ∃ Pt, A.L.partition s = some Pt :=
  ⟨_, rfl⟩

/-- **The held copies' design evaluates wherever a play is accepted** under `HeldOfAt`: the accepted play's run passed
its population check (`RunsAs.population_of_accepted`), at which the copies `heldOf` builds evaluate their design
(`HeldOf.twoHolds_of_pop`). -/
theorem twoHolds_of_formAccepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) : SpotFlock.RunsAs.Round2.TwoHolds A.h := by
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨j, hj⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨n, -, hp⟩ := SpotFlock.RunsAs.population_of_accepted hj
  exact SpotFlock.HeldOf.twoHolds_of_pop hu hp

/-- **The wired design evaluates at every audit statement** where the held copies' design does. -/
theorem two_wired {A : Live₂} (hT : SpotFlock.RunsAs.Round2.TwoHolds A.h) (s : Claims.Sampling.AuditStatement) :
    ∃ dd, A.L.two s = some dd := by
  obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 (hT s)
  have h2 : (A.L.two s).isSome = true := by
    change (SpotFlock.twoWired A.h (SpotFlock.wireMap A.h) s).isSome = true
    unfold SpotFlock.twoWired
    rw [hd]
    rfl
  exact Option.isSome_iff_exists.1 h2

/-- **The record a commitment carries is printable**: every object in it a well-formed tree and every number an
integer. `Spot.OneStage.commitmentOf` keeps what `Core.canonicalJson` returned of the bytes. -/
theorem recordGood {c : ByteArray} {cm : Spot.OneStage.Commitment} (h : Spot.OneStage.commitmentOf c = .ok cm) :
    Claims.Json.Good cm.record := by
  unfold Spot.OneStage.commitmentOf at h
  rcases hj : Core.canonicalJson c with e | j
  · simp [hj, Bind.bind, Except.bind] at h
  rcases hd : Spot.Registration.digest j with e | d
  · cases j <;> simp [hj, hd, Bind.bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  cases j <;> simp [hj, hd, Bind.bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  subst h
  exact Verity.CanonicalJson.canonicalJson_good hj

/-- **The statement a commitment carries decodes**: the statement is what `Flock.canon` printed of the commitment's
record, which is printable (`recordGood`), so the proved verifier's reader reads it back. -/
theorem decode_of_stmtOf {c : ByteArray} {x : Msg} (hx : stmtOf c = some x) : ∃ st, SpotFlock.decode x = some st := by
  unfold stmtOf at hx
  split at hx
  next cm hcm =>
    unfold recordMsg at hx
    cases hc : Flock.canon cm.record with
    | error _ =>
      rw [hc] at hx
      cases hx
    | ok s =>
      rw [hc] at hx
      obtain rfl : s.toUTF8.data.toList = x := Option.some.inj hx
      obtain ⟨j', hp, -⟩ := FlockComplete.JsonRT.parse_canon cm.record s (recordGood hcm) hc
      refine ⟨⟨j', SpotFlock.portRoots j'⟩, ?_⟩
      unfold SpotFlock.decode SpotFlock.jsonMsg FlockDeploy.jsonOf
      have hb : (⟨s.toUTF8.data.toList.toArray⟩ : ByteArray) = s.toUTF8 := rfl
      rw [hb, FlockComplete.fromUTF8?_toUTF8]
      show (((Flock.JsonParse.parse s).mapError _).toOption).map _ = _
      rw [hp]
      rfl
  next => cases hx

/-- **A statement that decodes parses at the wired program.** -/
theorem parse_wired (A : Live₂) {x : Msg} {st : SpotFlock.Roots} (hx : SpotFlock.decode x = some st) :
    ∃ s, A.L.parse x = some s := by
  have hp : (A.L.parse x).isSome = true := by
    change (SpotFlock.parseWired A.h (SpotFlock.wireMap A.h) (SpotFlock.returned A.h) x).isSome = true
    unfold SpotFlock.parseWired SpotFlock.parse
    rw [hx]
    rfl
  exact Option.isSome_iff_exists.1 hp

end Verity.SampledTwo
