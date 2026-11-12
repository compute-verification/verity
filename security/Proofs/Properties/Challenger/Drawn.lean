module

public import Proofs.Properties.Challenger
public import Proofs.Properties.Challenger.Coins
public import Proofs.Properties.Challenger.Window

/-!
# The coins a sample reads are the ones its challenger drew

`drawCoins` reads its coins under the session the ledger opens on the window (`Session.drawCoins_ok`), and they are the
coins any run of the device released whose commitment and roots are that session's (`Coins.drawn_released`), or a break.
The session is one a run of the device under the key closed with that commitment and those roots, its log that run's
lines (`Session.sessionOf_run`), and the device's own ledger opens it withheld on the window (`Window.window_opens`), or a
break; so that run released the coins, its log the session's, and the stream `challengerStream` accepted is read under
that run's commitment and its family's roots (`session_run`, with `drawn_log` from it), and `DrawnByDevice` holds
(`drawn`, `drawnByDevice`), with `CoinsFromLog` from it (`coinsFromLog_holds`). `challengerStream` accepts those roots
only when every one is the sample's `after` (`Coins.stream_ok`), so the run's own log shows no other root in the family
(`session_family`, `family`), and `FamilyOneCommitment` holds (`familyOneCommitment`). `session_run` takes the four
checks themselves, which `coinsReason` runs too under the key its evidence names (`Coins.coinsReason_none`), so the
same holds of the bytes a stream's run read (`familyOneCommitmentRead`).
-/

@[expose] public section

namespace Properties.Challenger.Drawn

open Lean (Json)
open Spot Spot.Challenger

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-- **A stream read under the session the ledger opens is a run of the device's**: when the ledger opens a session
withheld on `w` under `key`, the segment is of family `label`, and `challengerStream` accepts its openings under that
session's commitment and its roots of the family, a run the device's own ledger opens withheld on `w` released the
stream as its coins of `label` after `after`, the commitment and the roots that run's own (`releasedRootsOf` over its
log, which is the session's); or the finder outputs a break. What `drawCoins` and `coinsReason` both check
(`Coins.drawCoins_ok`, `Coins.coinsReason_none`). -/
theorem session_run {w : String} {after ξ : ByteArray} {c : LiveCommitment} {roots : Roots}
    (ht : trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
      (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots))
    (hw : withheldOn (getD coins "ledger") w = true)
    (hl : getD (getD coins "segment") "label" = Json.str label)
    (hst : challengerStream c (getD coins "segment") after
      (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      s.commitment = some c ∧ releasedRootsOf c s.log = .ok roots ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨r, hrj, hrc⟩ := Coins.trustedSession_record ht
  obtain ⟨a, hp, hn⟩ := Session.trustedSessionOf_ok ht
  obtain ⟨⟨c', roots', sid⟩, ho, rfl, rfl, rfl⟩ := Window.sessionNamed_sid hn
  rcases Window.window_opens (dev := dev) (label := label) hp hw with hopen | b
  · rcases Session.sessionOf_run (dev := dev) (label := label) ho with ⟨s, hs, hc, hid, -, hr, hlog⟩ | b
    · rcases Coins.stream_released hs hc hr hrj hrc hl hst with e | b
      · exact .inl ⟨s, hs, hid ▸ hopen, e, hc, hr, hlog⟩
      · exact .inr b
    · exact .inr b
  · exact .inr b

/-- The run `session_run` gives shows no root in the family but `after`: `challengerStream` accepted its roots
(`Coins.stream_ok`). -/
theorem session_family {w : String} {after ξ : ByteArray} {c : LiveCommitment} {roots : Roots}
    (ht : trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key
      (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots))
    (hw : withheldOn (getD coins "ledger") w = true)
    (hl : getD (getD coins "segment") "label" = Json.str label)
    (hst : challengerStream c (getD coins "segment") after
      (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      ∀ r ∈ s.roots label, r = after) ∨
      Break dev key label coins :=
  (session_run ht hw hl hst).imp (fun ⟨s, hs, ho, e, hc, hr, _⟩ => ⟨s, hs, ho, e, by
    obtain ⟨_, _, _, _, -, -, -, -, -, -, hall⟩ := Coins.stream_ok hst
    simpa only [Run.roots, hc, hr] using hall⟩) id

/-- **A sample `drawCoins` accepts reads the coins a run of the device released, and that run's log**, a run the device's
own ledger opens withheld on the sample's window, whose release log reads as its coin commitment's (`releasedRootsOf`),
or the finder outputs a break. -/
theorem drawn_log {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      (∃ c roots, s.commitment = some c ∧ releasedRootsOf c s.log = .ok roots) ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨c, roots, ht, hl, hst⟩ := Coins.drawCoins_ok h
  exact (session_run ht (Window.drawCoins_withheld h) hl hst).imp
    (fun ⟨s, hs, ho, e, hc, hr, hlog⟩ => ⟨s, hs, ho, e, ⟨c, roots, hc, hr⟩, hlog⟩) id

/-- **A sample `drawCoins` accepts reads a family that holds no other root**: the run that released its coins shows, in
its own log, every coin of the family released after the sample's `after`; or the finder outputs a break.
`FamilyOneCommitment` at one sample. -/
theorem family {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      ∀ r ∈ s.roots label, r = after) ∨
      Break dev key label coins := by
  obtain ⟨c, roots, ht, hl, hst⟩ := Coins.drawCoins_ok h
  exact session_family ht (Window.drawCoins_withheld h) hl hst

/-- **A sample `drawCoins` accepts reads the coins a run of the device released**, a run the device's own ledger opens
withheld on the sample's window, or the finder outputs a break: `DrawnByDevice` at one sample. -/
theorem drawn {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after) ∨
      Break dev key label coins :=
  (drawn_log h).imp (fun ⟨s, hs, ho, e, _⟩ => ⟨s, hs, ho, e⟩) id

/-- **`DrawnByDevice`.** -/
theorem drawnByDevice : DrawnByDevice :=
  fun _ _ _ _ _ _ _ h => drawn h

/-- **`FamilyOneCommitment`.** -/
theorem familyOneCommitment : FamilyOneCommitment :=
  fun _ _ _ _ _ _ _ h => family h

/-- **`FamilyOneCommitmentRead`**: `coinsReason`'s four checks are `drawCoins`', under the key its evidence names. -/
theorem familyOneCommitmentRead : FamilyOneCommitmentRead := by
  intro dev w label after read coins h
  obtain ⟨k, c, roots, ξ, hk, ht, hw, hl, hst, hpre, hall⟩ := Coins.coinsReason_none h
  refine ⟨k, hk, (session_family (dev := dev) ht hw hl hst).imp ?_ id⟩
  rintro ⟨s, hs, ho, rfl, hroots⟩
  exact ⟨s, hs, ho, hpre, hall, hroots⟩

/-- **`CoinsFromLog`**, from `DrawnByDevice`. -/
theorem coinsFromLog_holds : CoinsFromLog :=
  coinsFromLog drawnByDevice

end Properties.Challenger.Drawn
