module

public import Definitions.Pouw.TS.Machine
public import Mathlib.Algebra.Group.End
public import Mathlib.Data.Fintype.Sum
public import Mathlib.Tactic.DeriveFintype
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

/-! The data-sheet machine, the TS machine a device's vendor documents give on their own, every facet SM-clocked and each row in SMSP-cycles a warp instruction. -/

@[expose] public section

namespace ResourceAccounting.TS

/-- **A device's data sheet** over classes `K` and pipes `P`: its maximum boost clock `fmax`; per class, the dense
tensor MACs `macs k` one warp instruction does (`0` off the tensor pipe; a sparse form at its dense equivalent; a
tensor form whose operand type the sheet doesn't rate has `macs = 0`, so it costs issue only), the sheet's dense MACs
an SM clock `peak k` for its operand type, the Nsight Compute pipe `pipe k` it issues to (`none`: issue only) and the
shared-memory bank bytes `smem k` it holds at least (`0` off shared memory); per pipe, the Programming Guide's results
an SM clock `width p`. -/
structure Sheet (K P : Type) where
  /-- The top of the declared vBIOS's clock table, which no SM clock on the node exceeds; the H200's and B200's tables are
  unmeasured. -/
  fmax : ℝ
  macs : K → ℝ
  peak : K → ℝ
  pipe : K → Option P
  width : P → ℝ
  smem : K → ℝ

/-- The data-sheet machine's facets: issue, the tensor pipe, one per pipe of `P`, shared memory. -/
inductive SheetRow (P : Type)
  | issue
  | tensor
  | pipe (p : P)
  | smem
  deriving DecidableEq, Fintype

instance instNonemptySheetRow (P : Type) : Nonempty (SheetRow P) := ⟨.issue⟩

variable {K P : Type} [DecidableEq P]

/-- **The data-sheet machine** `Π_DS` of sheet `S`: issue, tensor, pipe and shared-memory rows as above, every facet
SM-clocked, `f_max` the sheet's. -/
noncomputable def Sheet.machine (S : Sheet K P) : Machine K (SheetRow P) where
  a r k := match r with
    | .issue => 1
    | .tensor => S.macs k / (S.peak k / 4)
    | .pipe p => if S.pipe k = some p then 128 / S.width p else 0
    | .smem => S.smem k / 32
  smClocked _ := true
  fmax := S.fmax

end ResourceAccounting.TS
