import Proofs.Flock.EndToEnd.Served.Bound

/-! The served sampling block's draw is a `StratifiedBlock` play:
`Flock.SecurityProofs.ServedDraw : Flock.Guarantees.ServedDraw` and
`Flock.SecurityProofs.JointServed : Flock.Guarantees.JointServed`, with their honest and nontrivial lemmas
(`Served/Bound.lean`), stated in `Served/Statement.lean`. The served tapes are defined in `Served/Draw.lean`; the draw on
them is the served draw (`Served/Tie.lean`), uniform under A3 at the coins (`Served/Uniform.lean`), and ends at the
honest instance (`Served/Honest.lean`). -/
