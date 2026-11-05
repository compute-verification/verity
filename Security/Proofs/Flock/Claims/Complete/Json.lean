module

public import Proofs.Flock.Claims.Complete.Json.Spec
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Hex
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Json.Decode
public import Proofs.Flock.Claims.Complete.Json.Session
public import Proofs.Flock.Claims.Complete.Json.Live
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Claims.Complete.Json.Goals
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

/-! The session record's JSON round trip: the lemmas on alike values it reads (`Spec`'s lemmas on `Same`), the hex round trips (`Hex`: `unhex_hexOf`, `f128OfHex_toHex`), and the honest record as a JSON value with
the decoder's reads through `Same` (`Record`: `recordObj`, `good_mkObj`, `str_same`, …), and the decoder's checks
S1–S14 on it (`Decode`: `decode_recordObj`), what the one-table session returns and sends (`Session`: `recordJson_eq`,
`sessionP_facts`, with a rep's rounds its sends, `repP_rec`), the record's live session and draw (`Live`:
`live_recordObj`, `recordDraw_recordObj`), what the record needs of the statement, from its setup (`Stmt`: `stmtOk`),
and B's record checks at the honest run under named hypotheses (`Goals`: `Run.json`, `decodes_goal`, …,
`custodyRounds_goal`). -/

@[expose] public section
