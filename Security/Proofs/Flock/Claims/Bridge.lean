module

public import Proofs.Flock.Claims.Bridge.Coins
public import Proofs.Flock.Claims.Bridge.OfGame
public import Proofs.Flock.Claims.Bridge.Transport
public import Proofs.Flock.Claims.Bridge.Live
public import Proofs.Flock.Claims.Bridge.LiveZk
public import Proofs.Flock.Claims.Bridge.LivePlain
public import Proofs.Flock.Claims.Bridge.Committed
public import Proofs.Flock.Claims.Bridge.Custody
public import Proofs.Flock.Claims.Bridge.CustodyPlain
public import Proofs.Flock.Claims.Bridge.DecodePlain
public import Proofs.Flock.Claims.Bridge.Dead
public import Proofs.Flock.Claims.Bridge.Returns
public import Proofs.Flock.Claims.Bridge.Hm96Tree
public import Definitions.Claims.Sampling
public import Proofs.Flock.Claims.Bridge.WrongSet
public import Proofs.Flock.Claims.Bridge.Transports

/-! The run-to-live-game bridge: the coin distribution (`Coins`), the public-coin verifier of a game and its run path by path
(`OfGame`), the transport of `Claims.run`'s probabilities to the live game's (`Transport`), and Flock's live session as
that verifier, with the transport at it (`Live`), and that verifier deciding by flock-verify `--zk`, chained to the old
transfer (`LiveZk`), and by plain flock-verify, chained to the old salted transfer (`LivePlain`). `Committed`: `--zk`'s
coins committed at `Hello` and opened round by round are within the coin tree's hiding distance `δ_hide` of coins flipped
after each message, and `Hm96Tree` bounds it by `2 k δ₁` from `Hm96Hiding`. `Custody`: `Deploy.V --zk`'s custody checks
make the last message's record a record of the run's live outcome (`RecordsLiveZKJ`), so its accepting runs are the old
theorems' event at a writer that reads the record out of the last message; `CustodyPlain`, the same for `Deploy.V false`
(`RecordsLiveJ`). `DecodePlain`: at one table, the old one-table decoder discharge makes the deployed plain decoders
right at that writer. `WrongSet`: the claims' incorrect set is no larger than the game's units wrong on the registration,
at a slot map, where each unit lowers and the witness's openings bind. `Transports`: the deployed check opens shaped
rows only: a misshaped value's leaf is a deployed opening's leaf only by a SHA-512 collision (`leaf_misshaped`). -/

@[expose] public section
