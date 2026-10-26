import Proofs.Flock.Claims.Bridge.Coins
import Proofs.Flock.Claims.Bridge.OfGame
import Proofs.Flock.Claims.Bridge.Transport
import Proofs.Flock.Claims.Bridge.Live
import Proofs.Flock.Claims.Bridge.LiveZk
import Proofs.Flock.Claims.Bridge.LivePlain
import Proofs.Flock.Claims.Bridge.Committed
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Claims.Bridge.CustodyPlain
import Proofs.Flock.Claims.Bridge.DecodePlain
import Proofs.Flock.Claims.Bridge.Dead
import Proofs.Flock.Claims.Bridge.Returns
import Proofs.Flock.Claims.Bridge.Hm96Tree
import Proofs.Flock.Claims.Bridge.ErrorTerms
import Proofs.Flock.Claims.Bridge.WrongSet
import Proofs.Flock.Claims.Bridge.BindsColl
import Proofs.Flock.Claims.Bridge.Transports

/-! The run-to-live-game bridge: the coin law (`Coins`), the public-coin verifier of a game and its run path by path
(`OfGame`), the transport of `Claims.run`'s probabilities to the live game's (`Transport`), and Flock's live session
as that verifier, with the transport at it (`Live`), and that verifier deciding by flock-verify `--zk`, chained to the
old transfer (`LiveZk`), and by plain flock-verify, chained to the old salted transfer (`LivePlain`). `Committed`:
`--zk`'s coins committed at `Hello` and opened round by round are within the coin tree's hiding distance `δ_hide` of
coins drawn after each message, and `Hm96Tree` bounds it by `2 k δ₁` from `Hm96Hiding`. `Custody`: `Deploy.V --zk`'s
custody checks make the last message's record a record of the run's live outcome (`RecordsLiveZKJ`), so its accepting
runs are the old theorems' event at a writer that reads the record out of the last message; `CustodyPlain`, the same
for `Deploy.V false` (`RecordsLiveJ`). `DecodePlain`: at one table, the old one-table decoder discharge makes the
deployed plain decoders right at that writer. `WrongSet`: the claims' faulty set is no larger than the game's
units wrong on the registration, at a slot map, where each unit lowers and the witness's openings bind. `BindsColl`: those openings bind unless two passing openings'
checks collide, given that the registered openings read as claims openings and the witness covers the unit.
`Transports`: the deployed check opens shaped rows only: a misshaped value's leaf is a deployed opening's leaf only by
a SHA-512 collision (`leaf_misshaped`). `ErrorTerms`: `EndToEnd`'s error `miss + ε_ks + δ_link` under `KS`'s
`κ D δ x + √(γ·Pr[B]) + Pr[B_ext]` at a deployment whose `δ` covers the miss and whose `4·ε_table` the rest
(`e2e_error_le_ks`). -/
