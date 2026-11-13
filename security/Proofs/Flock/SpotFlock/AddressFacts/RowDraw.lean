module

public import Proofs.Flock.SpotFlock.AddressFacts.Row
public import Proofs.Flock.Claims.Complete.Json.Value

/-!
# A subset draw's JSON, as Flock parses it

The second stage holds a session statement's draw to its own (`SecondStage.ownDraw`, `second-sample`) by `jeq`, and
Flock's verifier finds the session's instances by parsing that draw (`Draw.UnitDraw.ofJson`). Both read a subset
draw: the four keys `law` (`subset`), `population`, `k` (the units' count) and `units` (`subsetJson`). `SubsetParses`
states that a printable draw `jeq` to `subsetJson pop us` that Flock parses proves exactly `us`. The second stage's
own draw is one (`ownDraw_eq`), and so is a draw of every unit of a population of `m` (units `0..m-1`, no closure).
-/

@[expose] public section

namespace SpotFlock

open Lean (Json)

namespace AddressRow

/-- **A subset draw's JSON**: law `subset`, population `pop`, `k` the units' count, and the units `us`. -/
def subsetJson (pop : ℕ) (us : Array ℕ) : Json :=
  Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson pop), ("k", Lean.toJson us.size),
    ("units", Spot.natsJson us)]

/-- **The second stage's own draw for part `p`'s session is a subset draw**: of its `q · n` slots, the second draw's
units among them, counted from the part's first. -/
theorem ownDraw_eq (q : ℕ) (p : Spot.SecondStage.Part) (drawn : Array ℕ) :
    Spot.SecondStage.ownDraw q p drawn = subsetJson (q * p.n) ((drawn.filter (p.holds q)).map (· - p.lo)) := rfl

/-- **A draw held to a subset draw parses to its units**: a printable draw that `jeq` holds to `subsetJson pop us`
and that Flock's verifier parses proves `us`, with no closure. -/
def SubsetParses : Prop :=
  ∀ {j : Json} {pop : ℕ} {us : Array ℕ} {D : _root_.Flock.Draw.UnitDraw}, Claims.Json.Good j →
    Spot.jeq j (subsetJson pop us) = true → _root_.Flock.Draw.UnitDraw.ofJson j = .ok D → D.proved = us

end AddressRow

end SpotFlock
