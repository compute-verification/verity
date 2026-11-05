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
public import Proofs.Flock.Claims.Bridge.ErrorTerms
public import Proofs.Flock.Claims.Bridge.WrongSet
public import Proofs.Flock.Claims.Bridge.BindsColl
public import Proofs.Flock.Claims.Bridge.Transports
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

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

@[expose] public section
