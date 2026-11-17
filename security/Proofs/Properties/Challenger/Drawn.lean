module

public import Proofs.Properties.Challenger
public import Proofs.Properties.Challenger.Coins
public import Proofs.Properties.Challenger.Window

/-! The coins a sample reads are the ones its challenger drew, or a break (`SampledByDevice`). -/

@[expose] public section

namespace Properties.Challenger.Drawn

open Lean (Json)
open Spot Spot.Challenger

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

theorem challengePhase_run {w : String} {after ξ : ByteArray} {c : LiveCommitment} {roots : Roots}
    (ht : trustedChallengePhaseOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
      (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots))
    (hw : withheldOn (getD coins "ledger") w = true)
    (hl : getD (getD coins "segment") "label" = Json.str label)
    (hst : challengerStream c (getD coins "segment") after
      (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      s.commitment = some c ∧ releasedRootsOf c s.log = .ok roots ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨r, hrj, hrc⟩ := Coins.trustedChallengePhase_record ht
  obtain ⟨a, hp, hn⟩ := ChallengePhase.trustedChallengePhaseOf_ok ht
  obtain ⟨⟨c', roots', sid⟩, ho, rfl, rfl, rfl⟩ := Window.challengePhaseNamed_sid hn
  rcases Window.run_opens (dev := dev) (label := label) hp hw with hopen | b
  · rcases ChallengePhase.challengePhaseOf_run (dev := dev) (label := label) ho with ⟨s, hs, hc, hid, -, hr, hlog⟩ | b
    · rcases Coins.stream_released hs hc hr hrj hrc hl hst with e | b
      · exact .inl ⟨s, hs, hid ▸ hopen, e, hc, hr, hlog⟩
      · exact .inr b
    · exact .inr b
  · exact .inr b

theorem challengePhase_family {w : String} {after ξ : ByteArray} {c : LiveCommitment} {roots : Roots}
    (ht : trustedChallengePhaseOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
      (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots))
    (hw : withheldOn (getD coins "ledger") w = true)
    (hl : getD (getD coins "segment") "label" = Json.str label)
    (hst : challengerStream c (getD coins "segment") after
      (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      ∀ r ∈ s.roots label, r = after) ∨
      Break dev key label coins :=
  (challengePhase_run ht hw hl hst).imp (fun ⟨s, hs, ho, e, hc, hr, _⟩ => ⟨s, hs, ho, e, by
    obtain ⟨_, _, _, _, -, -, -, -, -, -, hall⟩ := Coins.stream_ok hst
    simpa only [Run.roots, hc, hr] using hall⟩) id

theorem sampled_log {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      (∃ c roots, s.commitment = some c ∧ releasedRootsOf c s.log = .ok roots) ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨c, roots, ht, hl, hst⟩ := Coins.drawCoins_ok h
  exact (challengePhase_run ht (Window.sampleCoins_withheld h) hl hst).imp
    (fun ⟨s, hs, ho, e, hc, hr, hlog⟩ => ⟨s, hs, ho, e, ⟨c, roots, hc, hr⟩, hlog⟩) id

theorem family {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      ∀ r ∈ s.roots label, r = after) ∨
      Break dev key label coins := by
  obtain ⟨c, roots, ht, hl, hst⟩ := Coins.drawCoins_ok h
  exact challengePhase_family ht (Window.sampleCoins_withheld h) hl hst

theorem drawn {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after) ∨
      Break dev key label coins :=
  (sampled_log h).imp (fun ⟨s, hs, ho, e, _⟩ => ⟨s, hs, ho, e⟩) id

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

theorem coinsFromLog_holds : CoinsFromLog :=
  coinsFromLog sampledByDevice

end Properties.Challenger.Drawn
