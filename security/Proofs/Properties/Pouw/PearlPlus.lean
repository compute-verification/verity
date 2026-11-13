module

public import Proofs.Properties.Pouw.PearlPlus.Open
public import Proofs.Properties.Pouw.PearlPlus.Verdict
public import Proofs.Properties.Pouw.PearlPlus.Draw

/-!
# Proofs of Pearl+'s run-verdict properties (`Properties.Pouw.PearlPlus`)

One module per group of properties: `OpeningBinds` in `Proofs.Properties.Pouw.PearlPlus.Open`, and `PlanMeans`,
`AcceptMeans`, `RefusedBeforeTheCoin`, `LoadedGivesUp`, `TileAcceptMeans`, `TileKeepsTheCap`, `YIsTheUnscale` and
`WorkIsTheCredit` in `Proofs.Properties.Pouw.PearlPlus.Verdict`, on Lean core; the run draw's `DrawIsOnItsStreams` and
`CoinBytesGiveUniformTickets` (`Properties.Pouw.PearlPlusDraw`) in `Proofs.Properties.Pouw.PearlPlus.Draw`, on Mathlib
as their statements are.
-/

@[expose] public section
