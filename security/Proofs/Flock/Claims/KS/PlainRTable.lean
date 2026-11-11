module

public import Proofs.Flock.Claims.KS.PlainKSLinR
public import Proofs.Flock.Claims.KS.PlainR
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
import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Claims.KS.PlainRConflict

/-!
# The table bound and `KSFlockLin` at the plain record at the deployed reader `VR rd`

`tableBoundAbortLin_VR` at `deployPlainR rel rd`, with `VR rd`'s `AccLast` and `DecideFree` (`accLast_VR`,
`decideFree_VR`), leaves the reader's three facts at the level columns `lc` (`RdAllAt`, `LevelTopAt`, `CapHitAt`) as
the table bound's premises (`plainR_tableBoundAbortLin`); with conflicts tied to collisions (`plainR_conflictTied`)
they give `KSFlockLin` at the trimmed record (`plainR_KSFlockLin_of_reader`).
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **The table bound at the plain record at `VR rd`**, at SHA-512, linearly at `CpDep`, from the reader's bound, top
and cap facts at `lc` (`tableBoundAbortLin_VR`, `accLast_VR`, `decideFree_VR`). -/
theorem plainR_tableBoundAbortLin (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) (hall : RdAllAt (deployPlainR rel rd) lc)
    (htopL : LevelTopAt (VR rd) lc) (hcap : CapHitAt (VR rd) lc) :
    Claims.Recursion.TableBoundAbortLin sha512 (deployPlainR rel rd) (wordLen false) (wide false) CpDep :=
  tableBoundAbortLin_VR rd (deployPlainR rel rd) rfl (fun x => εtable_nonneg false x) lc (accLast_VR rd)
    (decideFree_VR rd) hall htopL hcap

/-- **`KSFlockLin` at the trimmed plain record at `VR rd`**, at SHA-512 and `CpDep`, from the reader's three facts at
`lc` (`plainR_tableBoundAbortLin`, `plainR_KSFlockLin_of_table`). -/
theorem plainR_KSFlockLin_of_reader (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) (hall : RdAllAt (deployPlainR rel rd) lc)
    (htopL : LevelTopAt (VR rd) lc) (hcap : CapHitAt (VR rd) lc) :
    Claims.Recursion.KSFlockLin sha512 (deployPlainTrimR rel rd) CpDep :=
  plainR_KSFlockLin_of_table rel rd (plainR_tableBoundAbortLin rel rd lc hall htopL hcap)

end FlockDeploy.KS
