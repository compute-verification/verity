module

public import Proofs.Properties.Verity.Sampled.Record
public import Proofs.Flock.Bridge.SlotsTemplate
public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Claims.Sampling.Select.Take

/-!
# The statement is in scope at v2's slots

`scopeSlotsOk`: the session v2 sets up (`SetsUp`) on a circuit whose template check passes is in scope at v2's slots
(`ScopeSlotsOk`).

* The partition: `FlockSoundness.Discharge.Layout.partitions_of_templateOk` at `A.P`, `A.κ` and the registration's
  strata, which hold the population (`Flock.Bridge.nsOf_sum`).
* The law: `law_wellFormed`, `Flock.Bridge.wellFormed_of_lawOn` at the law v2 derives over the population
  (`SetsUp.params`' `lawAt`) on the strata `A.hlaw` gives.
* The template check v2 runs reads the circuit `ProvedScope.circuitOf` parses under the session's tags, which is
  `A.D.unroll` (`templateOk_unroll`): v2's statement has hidden outputs (`hidden_of_setsUp`), where `circuitOf` reads
  the circuit `HmOut.parse` reads, the descriptor's (`Flock.Desc.parse_flat`).
-/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling FlockSoundness.Discharge.ZkExec

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

theorem hidden_of_find {t : Flock.Tags} (h : Flock.Tags.find? _root_.SampledProofs.OneStageZk.STATEMENT = some t) :
    t.hiddenOutputs = true := by
  have hall : (Flock.Tags.find? _root_.SampledProofs.OneStageZk.STATEMENT).map (·.hiddenOutputs) = some true := by
    rfl
  rw [h] at hall
  exact Option.some.inj hall

/-- **v2's session tags have hidden outputs.** -/
theorem hidden_of_setsUp {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (h : SetsUp V θ F A) :
    (zkTags A.I).hiddenOutputs = true :=
  hidden_of_find h.tags

/-- **The circuit v2's template check reads is the descriptor's.** -/
theorem templateOk_unroll {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (h : SetsUp V θ F A) :
    (Flock.ProvedScope.circuitOf (zkTags A.I) A.I.circuitFile A.I.tables >>=
        _root_.SampledProofs.OneStageZk.templateOk) = _root_.SampledProofs.OneStageZk.templateOk A.D.unroll := by
  rw [Flock.ProvedScope.circuitOf_of_parse (hidden_of_setsUp h) (Flock.Desc.parse_flat A.hD)]
  rfl

open Properties.SampledProofs.SecurityProofs in
/-- **The law v2 derives over the population is well formed on the registration's strata.** -/
theorem law_wellFormed {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (h : SetsUp V θ F A) :
    (_root_.Properties.SampledProofs.oneStageOf A.law).WellFormed (Flock.Bridge.nsOf A.desc A.qr) := by
  obtain ⟨p, -, -, -, hl, -, -⟩ := h.params
  unfold lawAt at hl
  obtain ⟨lawJ, hJ, hl⟩ := bind_eq_ok.1 hl
  obtain ⟨derived, hd, hl⟩ := bind_eq_ok.1 hl
  obtain ⟨parsed, hp, hl⟩ := bind_eq_ok.1 hl
  have he : derived.getD parsed = A.law := Except.ok.inj hl
  rw [← he]
  exact Flock.Bridge.wellFormed_of_lawOn hJ hd hp (he ▸ A.hlaw)

/-- **The statement is in scope at v2's slots** where v2 sets the session up and its template check passes on the
descriptor's circuit. -/
theorem scopeSlotsOk {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (h : SetsUp V θ F A)
    (ht : _root_.SampledProofs.OneStageZk.templateOk A.D.unroll = .ok ()) : ScopeSlotsOk A :=
  ⟨FlockSoundness.Discharge.Layout.partitions_of_templateOk {} A.hU A.n A.κ (Flock.Bridge.nsOf_sum A.hpop).ge ht,
    law_wellFormed h⟩

end Verity.Sampled.Record
