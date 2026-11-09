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
break; so that run released the coins, its log the session's (`drawn_log`), and `DrawnByDevice` holds (`drawn`,
`drawnByDevice`), with `CoinsFromLog` from it (`coinsFromLog_holds`).
-/

@[expose] public section

namespace Properties.Challenger.Drawn

open Lean (Json)
open Spot Spot.Challenger

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-- **A sample `drawCoins` accepts reads the coins a run of the device released, and that run's log**, a run the device's
own ledger opens withheld on the sample's window, whose release log reads as its coin commitment's (`releasedRootsOf`),
or the finder outputs a break. -/
theorem drawn_log {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after ∧
      (∃ c roots, s.commitment = some c ∧ releasedRootsOf c s.log = .ok roots) ∧
      ∃ ls : Array Json, getD (getD coins "session") "log" = .arr ls ∧ ls.toList = s.log.map Json.str) ∨
      Break dev key label coins := by
  obtain ⟨c, roots, ht, hall⟩ := Coins.drawn_released (dev := dev) h
  obtain ⟨a, hp, hn⟩ := Session.trustedSessionOf_ok ht
  obtain ⟨⟨c', roots', sid⟩, ho, rfl, rfl, rfl⟩ := Window.sessionNamed_sid hn
  rcases Window.window_opens (dev := dev) (label := label) hp (Window.drawCoins_withheld h) with hopen | b
  · rcases Session.sessionOf_run (dev := dev) (label := label) ho with ⟨s, hs, hc, hid, -, hr, hlog⟩ | b
    · rcases hall s hs hc hr with e | b
      · exact .inl ⟨s, hs, hid ▸ hopen, e, ⟨_, _, hc, hr⟩, hlog⟩
      · exact .inr b
    · exact .inr b
  · exact .inr b

/-- **A sample `drawCoins` accepts reads the coins a run of the device released**, a run the device's own ledger opens
withheld on the sample's window, or the finder outputs a break: `DrawnByDevice` at one sample. -/
theorem drawn {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after) ∨
      Break dev key label coins :=
  (drawn_log h).imp (fun ⟨s, hs, ho, e, _⟩ => ⟨s, hs, ho, e⟩) id

/-- **`DrawnByDevice`.** -/
theorem drawnByDevice : DrawnByDevice :=
  fun _ _ _ _ _ _ _ h => drawn h

/-- **`CoinsFromLog`**, from `DrawnByDevice`. -/
theorem coinsFromLog_holds : CoinsFromLog :=
  coinsFromLog drawnByDevice

end Properties.Challenger.Drawn
