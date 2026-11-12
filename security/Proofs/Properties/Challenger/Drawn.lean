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
that run's commitment and its family's roots (`drawn_stream`, with `drawn_log` from it), and `DrawnByDevice` holds
(`drawn`, `drawnByDevice`), with `CoinsFromLog` from it (`coinsFromLog_holds`). `challengerStream` accepts those roots
only when every one is the sample's `after` (`Coins.stream_ok`), so the run's own log shows no other root in the family
(`family`), and `FamilyOneCommitment` holds (`familyOneCommitment`).
-/

@[expose] public section

namespace Properties.Challenger.Drawn

open Lean (Json)
open Spot Spot.Challenger

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-- **A sample `drawCoins` accepts reads its stream under a run of the device's commitment and roots**: a run the
device's own ledger opens withheld on the sample's window, which released the sample's coins, whose release log reads
as its coin commitment's (`releasedRootsOf`) and is the session's, the stream `challengerStream` accepted read under
that commitment and that log's roots of the family; or the finder outputs a break. -/
theorem drawn_stream {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      (∃ c roots, s.commitment = some c ∧ releasedRootsOf c s.log = .ok roots ∧
        challengerStream c (getD coins "segment") after
          (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨c, roots, ht, hl, hst⟩ := Coins.drawCoins_ok h
  obtain ⟨r, hrj, hrc⟩ := Coins.trustedSession_record ht
  obtain ⟨a, hp, hn⟩ := Session.trustedSessionOf_ok ht
  obtain ⟨⟨c', roots', sid⟩, ho, rfl, rfl, rfl⟩ := Window.sessionNamed_sid hn
  rcases Window.window_opens (dev := dev) (label := label) hp (Window.drawCoins_withheld h) with hopen | b
  · rcases Session.sessionOf_run (dev := dev) (label := label) ho with ⟨s, hs, hc, hid, -, hr, hlog⟩ | b
    · rcases Coins.stream_released hs hc hr hrj hrc hl hst with e | b
      · exact .inl ⟨s, hs, hid ▸ hopen, e, ⟨_, _, hc, hr, hst⟩, hlog⟩
      · exact .inr b
    · exact .inr b
  · exact .inr b

/-- **A sample `drawCoins` accepts reads the coins a run of the device released, and that run's log**, a run the device's
own ledger opens withheld on the sample's window, whose release log reads as its coin commitment's (`releasedRootsOf`),
or the finder outputs a break. -/
theorem drawn_log {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      (∃ c roots, s.commitment = some c ∧ releasedRootsOf c s.log = .ok roots) ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins :=
  (drawn_stream h).imp
    (fun ⟨s, hs, ho, e, ⟨c, roots, hc, hr, _⟩, hlog⟩ => ⟨s, hs, ho, e, ⟨c, roots, hc, hr⟩, hlog⟩) id

/-- **A sample `drawCoins` accepts reads a family that holds no other root**: the run that released its coins shows, in
its own log, every coin of the family released after the sample's `after`; or the finder outputs a break.
`FamilyOneCommitment` at one sample. -/
theorem family {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      ∀ r ∈ s.roots label, r = after) ∨
      Break dev key label coins :=
  (drawn_stream h).imp (fun ⟨s, hs, ho, e, ⟨c, roots, hc, hr, hst⟩, _⟩ => ⟨s, hs, ho, e, by
    obtain ⟨_, _, _, _, -, -, -, -, -, -, hall⟩ := Coins.stream_ok hst
    simpa only [Run.roots, hc, hr] using hall⟩) id

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

/-- **`CoinsFromLog`**, from `DrawnByDevice`. -/
theorem coinsFromLog_holds : CoinsFromLog :=
  coinsFromLog drawnByDevice

end Properties.Challenger.Drawn
