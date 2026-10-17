import Definitions.Accounting.TS.Machine

/-!
# The data-sheet machine (trusted)

The TS machine a device's vendor documents give on their own (note:20261017T1818Z-draft-axioms §2; decision 96:
data-sheet constants, no GPU self-attestation), over instruction classes `K` and Nsight Compute pipes `P`. Every facet
is SM-clocked, and each row is in SMSP-cycles a warp instruction (four SMSPs an SM):

* issue: `1` for every class, uniform-datapath classes included (A2, A2u);
* the tensor pipe: `macs_k / (peak_k / 4)`, a class's dense MACs over a quarter of the SM's dense MACs a clock for its
  operand type (A3);
* one facet per pipe `p`: `128 / width_p` for the classes on `p`, `width_p` the Programming Guide's results an SM
  clock (A4);
* shared memory: `smem_k / 32`, a class's bytes over a quarter of the SM's 128 B a clock (A5).

The machine model's `DS-NN` vectors are these rows at each device's vendor values (status `datasheet`).
-/

namespace Accounting.TS

/-- **A device's data sheet** over classes `K` and pipes `P`: its maximum boost clock `fmax`; per class, the dense
tensor MACs `macs k` one warp instruction does (`0` off the tensor pipe; a sparse form at its dense equivalent), the
sheet's dense MACs an SM clock `peak k` for its operand type, the Nsight Compute pipe `pipe k` it issues to (`none`:
issue only) and the shared-memory bytes `smem k` it moves; per pipe, the Programming Guide's results an SM clock
`width p`. -/
structure Sheet (K P : Type) where
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

instance (P : Type) : Nonempty (SheetRow P) := ⟨.issue⟩

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

end Accounting.TS
