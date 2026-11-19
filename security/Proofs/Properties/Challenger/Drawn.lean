module

public import Proofs.Properties.Challenger
public import Proofs.Properties.Challenger.Coins
public import Proofs.Properties.Challenger.Window

/-!
# The coins a sample reads are the ones its challenger drew

`sampleCoins` reads its coins under the session the ledger opens on the window (`ChallengePhase.drawCoins_ok`), and they are the
coins any run of the device released whose commitment and log are that session's (`Coins.sampled_released`), or a break.
The session is one a run of the device under the key closed with that commitment and that log, its log that run's
lines (`ChallengePhase.challengePhaseOf_run`), and the device's own ledger opens it withheld on the window (`Window.run_opens`), or a
break; so that run released the coins, its log the session's, and the stream `challengerStream` accepted is read under
that run's commitment and log (`challengePhase_run`, with `sampled_log` from it), and `SampledByDevice` holds
(`drawn`, `sampledByDevice`), with `CoinsFromLog` from it (`coinsFromLog_holds`). `challengerStream` accepts that log
only when every root of the family in it is the sample's `after` (`Coins.stream_ok`), so the run's own log shows no other
root in the family (`mem_family_roots`, `challengePhase_family`, `family`), and `FamilyOneCommitment` holds (`familyOneCommitment`). `coinsReason` reads what
`sampleCoins` reads from the same evidence, under the key it is handed (`Coins.coinsReason_none`), so the same holds of
the bytes a stream's run read (`familyOneCommitmentRead`).
-/

@[expose] public section

namespace Properties.Challenger.Drawn

open Lean (Json)
open Spot Spot.Challenger

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-- **A stream read under the session the ledger opens is a run of the device's**: when the ledger opens a session
withheld on `w` under `key`, the segment is of family `label`, and `challengerStream` accepts its openings under that
session's commitment and release log, a run the device's own ledger opens withheld on `w` released the stream as its
coins of `label` after `after`, the commitment and the log that run's own (`releasedLogOf` over its lines, which are
the session's); or the finder outputs a break. What `sampleCoins` checks (`Coins.drawCoins_ok`), and
`coinsReason` through it. -/
theorem challengePhase_run {w : String} {after ξ : ByteArray} {c : LiveCommitment} {log : Log}
    (ht : trustedChallengePhaseOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
      (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, log))
    (hw : withheldOn (getD coins "ledger") w = true)
    (hl : getD (getD coins "segment") "label" = Json.str label)
    (hst : challengerStream c (getD coins "segment") after (some log) = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      s.commitment = some c ∧ releasedLogOf c s.log = .ok log ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨r, hrj, hrc⟩ := Coins.trustedChallengePhase_record ht
  obtain ⟨a, hp, hn⟩ := ChallengePhase.trustedChallengePhaseOf_ok ht
  obtain ⟨⟨c', log', sid⟩, ho, rfl, rfl, rfl⟩ := Window.challengePhaseNamed_sid hn
  rcases Window.run_opens (dev := dev) (label := label) hp hw with hopen | b
  · rcases ChallengePhase.challengePhaseOf_run (dev := dev) (label := label) ho with ⟨s, hs, hc, hid, -, hr, hlog⟩ | b
    · rcases Coins.stream_released hs hc hr hrj hrc hl hst with e | b
      · exact .inl ⟨s, hs, hid ▸ hopen, e, hc, hr, hlog⟩
      · exact .inr b
    · exact .inr b
  · exact .inr b

/-- A root `Run.roots` shows in a family is an entry's of that family in the log it reads. -/
theorem mem_family_roots {c : LiveCommitment} {log : Log} {r : ByteArray}
    (h : r ∈ (((byFamily c log).find? (·.1 == label)).map (·.2)).getD []) :
    ∃ e ∈ log, e.label = label ∧ e.root = r := by
  cases hf : (byFamily c log).find? (·.1 == label) with
  | none => simp [hf] at h
  | some p =>
    rw [hf] at h
    have hp := List.mem_of_find?_eq_some hf
    have hpl := List.find?_some hf
    simp only [byFamily, List.mem_map] at hp
    obtain ⟨⟨l, n⟩, -, rfl⟩ := hp
    simp only [beq_iff_eq] at hpl
    simp only [Option.map_some, Option.getD_some, List.mem_map, List.mem_filter, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨e, ⟨he, -, hel⟩, rfl⟩ := h
    exact ⟨e, he, hel.trans hpl, rfl⟩

/-- The run `challengePhase_run` gives shows no root in the family but `after`: `challengerStream` accepted its log
(`Coins.stream_ok`). -/
theorem challengePhase_family {w : String} {after ξ : ByteArray} {c : LiveCommitment} {log : Log}
    (ht : trustedChallengePhaseOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
      (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, log))
    (hw : withheldOn (getD coins "ledger") w = true)
    (hl : getD (getD coins "segment") "label" = Json.str label)
    (hst : challengerStream c (getD coins "segment") after (some log) = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      ∀ r ∈ s.roots label, r = after) ∨
      Break dev key label coins :=
  (challengePhase_run ht hw hl hst).imp (fun ⟨s, hs, ho, e, hc, hr, _⟩ => ⟨s, hs, ho, e, by
    obtain ⟨_, _, _, _, hl', -, -, -, -, -, hall⟩ := Coins.stream_ok hst
    rw [hl] at hl'
    cases hl'
    intro x hx
    simp only [Run.roots, hc, hr, Functor.map, Except.map] at hx
    obtain ⟨e, he, hel, rfl⟩ := mem_family_roots hx
    exact hall e he hel⟩) id

/-- **A sample `sampleCoins` accepts reads the coins a run of the device released, and that run's log**, a run the device's
own ledger opens withheld on the sample's window, whose release log reads as its coin commitment's (`releasedLogOf`),
or the finder outputs a break. -/
theorem sampled_log {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      (∃ c log, s.commitment = some c ∧ releasedLogOf c s.log = .ok log) ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨c, log, ht, hl, hst⟩ := Coins.drawCoins_ok h
  exact (challengePhase_run ht (Window.sampleCoins_withheld h) hl hst).imp
    (fun ⟨s, hs, ho, e, hc, hr, hlog⟩ => ⟨s, hs, ho, e, ⟨c, log, hc, hr⟩, hlog⟩) id

/-- **A sample `sampleCoins` accepts reads a family that holds no other root**: the run that released its coins shows, in
its own log, every coin of the family released after the sample's `after`; or the finder outputs a break.
`FamilyOneCommitment` at one sample. -/
theorem family {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      ∀ r ∈ s.roots label, r = after) ∨
      Break dev key label coins := by
  obtain ⟨c, log, ht, hl, hst⟩ := Coins.drawCoins_ok h
  exact challengePhase_family ht (Window.sampleCoins_withheld h) hl hst

/-- **A sample `sampleCoins` accepts reads the coins a run of the device released**, a run the device's own ledger opens
withheld on the sample's window, or the finder outputs a break: `SampledByDevice` at one sample. -/
theorem drawn {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after) ∨
      Break dev key label coins :=
  (sampled_log h).imp (fun ⟨s, hs, ho, e, _⟩ => ⟨s, hs, ho, e⟩) id

/-- **`SampledByDevice`.** -/
theorem sampledByDevice : SampledByDevice :=
  fun _ _ _ _ _ _ _ h => drawn h

/-- **`FamilyOneCommitment`.** -/
theorem familyOneCommitment : FamilyOneCommitment :=
  fun _ _ _ _ _ _ _ h => family h

/-- **`FamilyOneCommitmentRead`**: `coinsReason` reads `sampleCoins`' stream, under the key it is handed. -/
theorem familyOneCommitmentRead : FamilyOneCommitmentRead := by
  intro dev key main w label after read coins h
  obtain ⟨ξ, hd, -, hpre, hall⟩ := Coins.coinsReason_none h
  refine (family (dev := dev) hd).imp ?_ id
  rintro ⟨s, hs, ho, rfl, hroots⟩
  exact ⟨s, hs, ho, hpre, hall, hroots⟩

/-- **`CoinsFromLog`**, from `SampledByDevice`. -/
theorem coinsFromLog_holds : CoinsFromLog :=
  coinsFromLog sampledByDevice

end Properties.Challenger.Drawn
