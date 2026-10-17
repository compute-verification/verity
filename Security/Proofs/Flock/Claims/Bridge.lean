import Proofs.Flock.Claims.Bridge.Coins
import Proofs.Flock.Claims.Bridge.OfGame
import Proofs.Flock.Claims.Bridge.Transport
import Proofs.Flock.Claims.Bridge.Live
import Proofs.Flock.Claims.Bridge.LiveZk
import Proofs.Flock.Claims.Bridge.Committed
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Claims.Bridge.Hm96Tree

/-! The run-to-live-game bridge: the coin law (`Coins`), the public-coin verifier of a game and its run path by path
(`OfGame`), the transport of `Claims.run`'s probabilities to the live game's (`Transport`), and Flock's live session
as that verifier, with the transport at it (`Live`), and that verifier deciding by flock-verify `--zk`, chained to the
old transfer (`LiveZk`). `Committed`: `--zk`'s coins committed at `Hello` and opened round by round are within the coin
tree's hiding distance `δ_hide` of coins drawn after each message, and `Hm96Tree` bounds it by `2 k δ₁` from
`Hm96Hiding`. `Custody`: `Deploy.V --zk`'s custody checks make the
last message's record a record of the run's live outcome (`RecordsLiveZKJ`), so its accepting runs are the old
theorems' event at a writer that reads the record out of the last message. -/
