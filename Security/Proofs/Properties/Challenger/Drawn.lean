import Proofs.Properties.Challenger
import Proofs.Properties.Challenger.Coins
import Proofs.Properties.Challenger.Window

/-!
# The coins a draw reads are the ones its challenger drew

`drawCoins` reads its coins under the session the ledger opens on the window (`Session.drawCoins_ok`), and they are the
coins any run of the device released whose commitment and roots are that session's (`Coins.drawn_released`), or a break.
The session is one a run of the device under the key closed with that commitment and those roots
(`Session.sessionOf_bound`), and the device's own ledger opens it withheld on the window (`Window.window_opens`), or a
break; so that run released the coins (`drawn`), and `DrawnByDevice` holds (`drawnByDevice`), with `CoinsFromLog` from
it (`coinsFromLog_of_label`).

Each takes the segment's label as a hypothesis, as `drawn_released` does: `drawCoins` compares it with `==` on `Json`, a
`partial def` the kernel can't unfold.
-/

namespace Properties.Challenger.Drawn

open Lean (Json)
open SampledProofs SampledProofs.Challenger

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-- **A draw `drawCoins` accepts reads the coins a run of the device released**, a run the device's own ledger opens
withheld on the draw's window, or the finder outputs a break: `DrawnByDevice` at one draw whose segment is labelled
`label`. -/
theorem drawn {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ)
    (hl : getD (getD coins "segment") "label" = Json.str label) :
    (∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ ξ = s.released label after) ∨
      Break dev key label coins := by
  obtain ⟨c, roots, ht, hall⟩ := Coins.drawn_released (dev := dev) h hl
  obtain ⟨a, hp, hn⟩ := Session.trustedSessionOf_ok ht
  obtain ⟨⟨c', roots', sid⟩, ho, rfl, rfl, rfl⟩ := Window.sessionNamed_sid hn
  rcases Window.window_opens (dev := dev) (label := label) hp (Window.drawCoins_withheld h) with hopen | b
  · rcases Session.sessionOf_bound (dev := dev) (label := label) ho with ⟨s, hs, hc, hid, -, hr⟩ | b
    · rcases hall s hs hc hr with e | b
      · exact .inl ⟨s, hs, hid ▸ hopen, e⟩
      · exact .inr b
    · exact .inr b
  · exact .inr b

/-- **`DrawnByDevice`**, where every draw `drawCoins` accepts reads a segment labelled with its family. -/
theorem drawnByDevice (hl : ∀ (key : ChallengerKey) (w label : String) (after : ByteArray) (coins : Json)
      (ξ : ByteArray), drawCoins key w label after coins = .ok ξ →
        getD (getD coins "segment") "label" = Json.str label) :
    DrawnByDevice :=
  fun _ _ _ _ _ _ _ h => drawn h (hl _ _ _ _ _ _ h)

/-- **`CoinsFromLog`**, where every draw `drawCoins` accepts reads a segment labelled with its family. -/
theorem coinsFromLog_of_label (hl : ∀ (key : ChallengerKey) (w label : String) (after : ByteArray) (coins : Json)
      (ξ : ByteArray), drawCoins key w label after coins = .ok ξ →
        getD (getD coins "segment") "label" = Json.str label) :
    CoinsFromLog :=
  coinsFromLog (drawnByDevice hl)

end Properties.Challenger.Drawn
