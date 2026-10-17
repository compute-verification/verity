import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Finset.Card

/-!
# The circuit model: a declared circuit's gates, their classes, its window and its census (trusted)

One model of what a declared circuit `W` computes, for the lower bounds every resource lane proves over it: the stacking
property's carry and encode bounds, the requirement `A1` reads, and compute accounting's work statement and fan-in floor.
It follows the IR (`verity/ml/circuits/PROTOCOL.md`) gate for gate, so that census/v1 and property/v1
(`verity/ml/circuits/census.py`, `stacking.py`) can be held to it by difftest.

* **Values.** Every value is an integer (`ℤ`): a `Value<w>` leaf read as the integer its Definition gives it, two's
  complement for a signed word, the bit pattern for a float. A gate's declared interval `[lo, hi]` is its type's range,
  or the narrower one an interval analysis derives.
* **Gates.** A circuit is a list of gates in the IR's canonical, topological order (§6): gate `i` reads the values of
  the gates `args` names, each earlier than `i` (`Topological`). A gate is one primitive: one `Value` out.
* **Inputs.** An input gate reads slot `i` of the input assignment `x : ι → ℤ`. A `registered` input (the weights, the
  store's payload, a codec's public parameters) is bound before the audit's coins, a `run` input (a request's tokens,
  anything the prover sends a running circuit) after them, so a run input may be any function of the coins.
* **Classes.** Free (no unit in any cap): an input, a constant (an operation's own bits), an integer affine map with
  constant coefficients (`I32Add`, `I32Sub`, a product with a constant), and an s8-family step (`mma`: the exact int8
  dot of `verity/ml/circuits/silicon/integer.py`, `D = C + Σ a_k·b_k` with s8 or u8 operands and s32 accumulation,
  clamped by `.satfinite` or wrapped mod 2^32 outside the s32 range). Budgeted, one unit per output value: a rounding
  (`ρ_b(x) = ⌊x/2^b⌋`, and every cast, quantizer, requantizer, FP elementwise op, integer floor, mod, division, bit op,
  limb split and word reinterpretation), a comparison or selection, a gather, and a product of two runtime values. A
  budgeted gate is any function of its operands (`budgeted`), the most a cheater could ask of a primitive, except a
  gather, which selects a word of its column by its index (`gather`). The registry's primitives map to exactly one class
  each (`Gates/Registry.lean`); an unlisted primitive is refused, so it has no `Op`.
* **The window.** On an input, every gate's exact value lies in its declared interval, and every s8-family step's
  operands lie in their formats and its exact sum in s32, so no accumulator clamps or wraps (`InWindow`). The property
  proves it for the free gates by interval analysis over the whole DAG from the declared intervals of inputs and
  budgeted gates (`Proven`), never from a free gate's declaration.
* **Cost.** The census (`census`) counts the gates of a set of steps by class, the s8 family's MACs, and the bits of
  the run inputs. The counting bounds charge each budgeted gate and each run input the size of its declared interval
  (`volume`, `chargedBits`).
* **The property.** `Property caps A W`: the stacking property `P(c, R_in, L)` of property/v1 as a predicate on `W`'s
  census and the audit's certified reads: per window, each budgeted class within its cap per certified byte read and
  the run inputs within `R_in`, every gate in a window, and no value read more than `L` steps after it is produced.
* **Fan-in.** MULT-LB's lane programs (`ofLane`: inputs, then instructions that each apply an arbitrary function to at
  most `f` earlier values) are this model's circuits of budgeted gates of fan-in at most `f` (`FanIn`).
-/

namespace Accounting.Gates

/-- When an input is bound: at registration, before the audit's coins, or at run time, after them. -/
inductive Binding
  | registered
  | run
  deriving DecidableEq, Repr

/-- An s8-family operand's format: `s8` (`[−128, 127]`) or `u8` (`[0, 255]`). -/
inductive Int8Fmt
  | s8
  | u8
  deriving DecidableEq, Repr

/-- The least value of an operand format. -/
def Int8Fmt.lo : Int8Fmt → ℤ
  | .s8 => -128
  | .u8 => 0

/-- The greatest value of an operand format. -/
def Int8Fmt.hi : Int8Fmt → ℤ
  | .s8 => 127
  | .u8 => 255

/-- The least s32 word, `−2^31`. -/
def s32Min : ℤ := -2 ^ 31

/-- The greatest s32 word, `2^31 − 1`. -/
def s32Max : ℤ := 2 ^ 31 - 1

/-- `.satfinite`: clamped to `[−2^31, 2^31 − 1]`. -/
def clampS32 (z : ℤ) : ℤ := max s32Min (min s32Max z)

/-- Without `.satfinite`: wrapped modulo `2^32` into `[−2^31, 2^31 − 1]`. -/
def wrapS32 (z : ℤ) : ℤ := (z - s32Min) % 2 ^ 32 + s32Min

/-- A budgeted class: what the property caps, one unit per output value. -/
inductive Budget
  /-- `ρ_b(x) = ⌊x/2^b⌋`, and every cast, quantizer, requantizer, FP elementwise op, integer floor, mod or division,
  bit op, limb split and word reinterpretation -/
  | rounding
  /-- a comparison, a selection by a condition, a min or max -/
  | compareSelect
  /-- a selection of a word of a column by a runtime index -/
  | gather
  /-- a product of two runtime values outside the s8 family -/
  | product
  deriving DecidableEq, Repr

/-- A gate's operation, over input slots `ι`. -/
inductive Op (ι : Type)
  /-- the input in slot `i`, bound at `b` -/
  | input (b : Binding) (i : ι)
  /-- a constant: an operation's own bits -/
  | const (c : ℤ)
  /-- `c₀ + Σ coeff_k · v_k`: an integer affine map with constant coefficients -/
  | affine (c₀ : ℤ) (coeff : List ℤ)
  /-- an s8-family step's coordinate: operands `C :: a_1 … a_k ++ b_1 … b_k`, `C + Σ a_i·b_i`, then clamped
  (`satfinite`) or wrapped to s32 -/
  | mma (a b : Int8Fmt) (k : ℕ) (satfinite : Bool)
  /-- a gather: operands `idx :: column`, the column's word at `idx` (0 outside it) -/
  | gather
  /-- a budgeted primitive of class `cls`: any function of its operands -/
  | budgeted (cls : Budget) (f : List ℤ → ℤ)

/-- The dot of the first `k` operands with the next `k`. -/
def dot (k : ℕ) (v : List ℤ) : ℤ := (List.zipWith (· * ·) (v.take k) (v.drop k)).sum

namespace Op

variable {ι : Type}

/-- `ρ_b(x) = ⌊x/2^b⌋`, the rounding. -/
def round (b : ℕ) : Op ι := budgeted .rounding fun v => v.headD 0 / 2 ^ b

/-- `[a ≤ b]`, a comparison. -/
def le : Op ι := budgeted .compareSelect fun v => if v.headD 0 ≤ v.tail.headD 0 then 1 else 0

/-- `c ≠ 0 ? a : b`, a selection. -/
def select : Op ι := budgeted .compareSelect fun v => if v.headD 0 ≠ 0 then v.tail.headD 0 else v.tail.tail.headD 0

/-- `a · b`, a product of two runtime values. -/
def mul : Op ι := budgeted .product fun v => v.headD 0 * v.tail.headD 0

/-- The exact value of an operation on operands `v`, before an s8-family step's clamp or wrap. -/
def raw : Op ι → (ι → ℤ) → List ℤ → ℤ
  | input _ i, x, _ => x i
  | const c, _, _ => c
  | affine c₀ cs, _, v => c₀ + (List.zipWith (· * ·) cs v).sum
  | mma _ _ k _, _, v => v.headD 0 + dot k v.tail
  | gather, _, v => v.tail.getD (v.headD 0).toNat 0
  | budgeted _ f, _, v => f v

/-- The value an operation produces from its exact value: an s8-family step clamps or wraps it, every other passes it. -/
def out : Op ι → ℤ → ℤ
  | mma _ _ _ true, z => clampS32 z
  | mma _ _ _ false, z => wrapS32 z
  | _, z => z

/-- An operation whose value the counting bounds charge: a run input or a budgeted gate. -/
def charged : Op ι → Bool
  | input .run _ => true
  | gather => true
  | budgeted _ _ => true
  | _ => false

/-- Its budgeted class, if any. -/
def budget : Op ι → Option Budget
  | gather => some .gather
  | budgeted cls _ => some cls
  | _ => none

/-- An s8-family step's operands are `C` and `2k` words in their formats; other operations ask nothing. -/
def OperandsOk : Op ι → List ℤ → Prop
  | mma a b k _, v =>
      v.length = 2 * k + 1 ∧ (∀ z ∈ v.tail.take k, a.lo ≤ z ∧ z ≤ a.hi) ∧
        ∀ z ∈ v.tail.drop k, b.lo ≤ z ∧ z ≤ b.hi
  | _, _ => True

/-- An s8-family step. -/
def IsMma : Op ι → Prop
  | mma .. => True
  | _ => False

/-- A free operation whose interval the property derives: a constant, an affine map, an s8-family step. -/
def IsDerived : Op ι → Prop
  | const _ | affine .. | mma .. => True
  | _ => False

end Op

/-- A gate: its operation, its operands (earlier gates), its declared interval, and the step that runs it. -/
structure Gate (ι : Type) where
  op : Op ι
  args : List ℕ
  lo : ℤ
  hi : ℤ
  step : ℕ

/-- A circuit: its gates in canonical order. -/
abbrev Circuit (ι : Type) := List (Gate ι)

namespace Gate

variable {ι : Type} (g : Gate ι)

/-- The gate's operand values, read from the values `vs` of the gates before it. -/
def operands (vs : List ℤ) : List ℤ := g.args.map fun j => vs.getD j 0

/-- The gate's exact value. -/
def raw (x : ι → ℤ) (vs : List ℤ) : ℤ := g.op.raw x (g.operands vs)

/-- The gate's value. -/
def value (x : ι → ℤ) (vs : List ℤ) : ℤ := g.op.out (g.raw x vs)

/-- The gate is in its window: its exact value in its declared interval, an s8-family step's operands in their
formats and its declared interval within s32 (so it neither clamps nor wraps). -/
def Ok (x : ι → ℤ) (vs : List ℤ) : Prop :=
  g.lo ≤ g.raw x vs ∧ g.raw x vs ≤ g.hi ∧ g.op.OperandsOk (g.operands vs) ∧
    (g.op.IsMma → s32Min ≤ g.lo ∧ g.hi ≤ s32Max)

/-- The number of values its declared interval holds. -/
def size : ℕ := (g.hi + 1 - g.lo).toNat

/-- Its declared bits: `⌈log₂ size⌉`. -/
def bits : ℕ := Nat.clog 2 g.size

/-- What the counting bounds charge it: its declared interval's size if it is charged, else 1. -/
def charge : ℕ := if g.op.charged then g.size else 1

/-- Its charged bits: its declared bits if it is charged, else 0. -/
def chargedBits : ℕ := if g.op.charged then g.bits else 0

end Gate

variable {ι : Type}

/-- The values of a circuit's gates on input `x`, in order. -/
def run (C : Circuit ι) (x : ι → ℤ) : List ℤ := C.foldl (fun vs g => vs ++ [g.value x vs]) []

/-- Gate `i`'s value on input `x`. -/
def val (C : Circuit ι) (x : ι → ℤ) (i : ℕ) : ℤ := (run C x).getD i 0

/-- Canonical order: every gate's operands are earlier gates. -/
def Topological (C : Circuit ι) : Prop := ∀ i (h : i < C.length), ∀ j ∈ C[i].args, j < i

/-- **The bounded window** on input `x`: every gate's exact value lies in its declared interval, and no s8-family step's
accumulator clamps or wraps. -/
def InWindow (C : Circuit ι) (x : ι → ℤ) : Prop := ∀ i (h : i < C.length), C[i].Ok x (run (C.take i) x)

/-- The product of the charges: the number of joint values the charged gates' declared intervals allow. -/
def volume (C : Circuit ι) : ℕ := (C.map Gate.charge).prod

/-- The charged bits: the declared bits of the run inputs and budgeted gates. -/
def chargedBits (C : Circuit ι) : ℕ := (C.map Gate.chargedBits).sum

/-! ## Interval analysis: the window the property proves -/

/-- An interval `[lo, hi]`. -/
abbrev Iv := ℤ × ℤ

/-- `[a] + [b]`. -/
def Iv.add (a b : Iv) : Iv := (a.1 + b.1, a.2 + b.2)

/-- `c · [a]`. -/
def Iv.scale (c : ℤ) (a : Iv) : Iv := if 0 ≤ c then (c * a.1, c * a.2) else (c * a.2, c * a.1)

/-- `[a] · [b]`. -/
def Iv.mul (a b : Iv) : Iv :=
  (min (min (a.1 * b.1) (a.1 * b.2)) (min (a.2 * b.1) (a.2 * b.2)),
    max (max (a.1 * b.1) (a.1 * b.2)) (max (a.2 * b.1) (a.2 * b.2)))

/-- `[a] ⊆ [lo, hi]`. -/
def Iv.Within (a : Iv) (lo hi : ℤ) : Prop := lo ≤ a.1 ∧ a.2 ≤ hi

/-- The interval a gate's value is derived to lie in, from its operands' derived intervals `ivs`: a constant's own
value, an affine map's or an s8-family step's image, and every other gate's declared interval. -/
def Gate.derive (g : Gate ι) (ivs : List Iv) : Iv :=
  let a := g.args.map fun j => ivs.getD j (0, 0)
  match g.op with
  | .const c => (c, c)
  | .affine c₀ cs => (List.zipWith Iv.scale cs a).foldl Iv.add (c₀, c₀)
  | .mma _ _ k _ =>
      (List.zipWith Iv.mul (a.tail.take k) (a.tail.drop k)).foldl Iv.add (a.headD (0, 0))
  | _ => (g.lo, g.hi)

/-- The derived intervals of a circuit's gates, in order. -/
def intervals (C : Circuit ι) : List Iv := C.foldl (fun ivs g => ivs ++ [g.derive ivs]) []

/-- **The window, proven** (property/v1's interval analysis over the whole DAG): every constant, affine map and
s8-family step has its derived interval within its declared one, and every s8-family step its operands' derived
intervals within their formats and its declared interval within s32. -/
def Proven (C : Circuit ι) : Prop :=
  ∀ i (h : i < C.length),
    let g := C[i]
    let ivs := intervals (C.take i)
    (g.op.IsDerived → (g.derive ivs).Within g.lo g.hi) ∧
      ∀ fa fb k sat, g.op = .mma fa fb k sat →
        g.args.length = 2 * k + 1 ∧ s32Min ≤ g.lo ∧ g.hi ≤ s32Max ∧
          (∀ j ∈ g.args.tail.take k, (ivs.getD j (0, 0)).Within fa.lo fa.hi) ∧
          ∀ j ∈ g.args.tail.drop k, (ivs.getD j (0, 0)).Within fb.lo fb.hi

/-- Every input and budgeted gate's value lies in its declared interval on input `x`: what a primitive's output type
guarantees, and a run input's. -/
def DeclaredHold (C : Circuit ι) (x : ι → ℤ) : Prop :=
  ∀ i (h : i < C.length), ¬ C[i].op.IsDerived →
    C[i].lo ≤ C[i].raw x (run (C.take i) x) ∧ C[i].raw x (run (C.take i) x) ≤ C[i].hi

/-! ## Fan-in: MULT-LB's lane programs -/

/-- Every budgeted gate but a gather reads at most `f` operands (a gather reads its index and the one word it
selects). -/
def FanIn (f : ℕ) (C : Circuit ι) : Prop :=
  ∀ g ∈ C, g.op.charged → g.op.budget ≠ some .gather → g.args.length ≤ f

/-- Every value's declared interval lies within `w` bits: `[−2^w, 2^w)`. -/
def Width (w : ℕ) (C : Circuit ι) : Prop := ∀ g ∈ C, -2 ^ w ≤ g.lo ∧ g.hi < 2 ^ w

/-- MULT-LB's lane program as a circuit: run inputs `0 … N−1`, then each instruction an arbitrary function of the
earlier nodes it names, a product of runtime values in the count. -/
def ofLane (N : ℕ) (instrs : List (List ℕ × (List ℤ → ℤ))) (lo hi : ℤ) : Circuit ℕ :=
  (List.range N).map (fun i => ⟨.input .run i, [], lo, hi, 0⟩) ++
    instrs.map fun ins => ⟨.budgeted .product ins.2, ins.1, lo, hi, 0⟩

/-! ## The census and the property -/

/-- A circuit's census over a set of steps: its gates by class, the s8 family's MACs, and its inputs' bits. -/
structure Census where
  /-- s8-family coordinates -/
  family : ℕ
  /-- their MACs: `k` each -/
  macs : ℕ
  /-- affine maps and constants -/
  affine : ℕ
  /-- roundings -/
  rounding : ℕ
  /-- comparisons and selections -/
  compareSelect : ℕ
  /-- gathers -/
  gather : ℕ
  /-- products of two runtime values -/
  product : ℕ
  /-- the run inputs' declared bits -/
  runBits : ℕ
  /-- the registered inputs' declared bits -/
  registeredBits : ℕ

/-- The number of gates of `C` at steps in `s` that `p` picks. -/
def countAt (C : Circuit ι) (s : ℕ → Prop) [DecidablePred s] (p : Gate ι → ℕ) : ℕ :=
  ((C.filter fun g => decide (s g.step)).map p).sum

/-- One for a gate of budgeted class `b`. -/
def Gate.isBudget (b : Budget) (g : Gate ι) : ℕ := if g.op.budget = some b then 1 else 0

/-- **The census** of `C`'s gates at the steps `s` picks. -/
def census (C : Circuit ι) (s : ℕ → Prop) [DecidablePred s] : Census where
  family := countAt C s fun g => match g.op with | .mma .. => 1 | _ => 0
  macs := countAt C s fun g => match g.op with | .mma _ _ k _ => k | _ => 0
  affine := countAt C s fun g => match g.op with | .affine .. | .const _ => 1 | _ => 0
  rounding := countAt C s (Gate.isBudget .rounding)
  compareSelect := countAt C s (Gate.isBudget .compareSelect)
  gather := countAt C s (Gate.isBudget .gather)
  product := countAt C s (Gate.isBudget .product)
  runBits := countAt C s fun g => match g.op with | .input .run _ => g.bits | _ => 0
  registeredBits := countAt C s fun g => match g.op with | .input .registered _ => g.bits | _ => 0

/-- The budgeted units of a census: every budgeted class, one unit a gate. -/
def Census.budgeted (c : Census) : ℕ := c.rounding + c.compareSelect + c.gather + c.product

/-- The budgeted gates of a circuit, one unit each. -/
def budgeted (C : Circuit ι) : ℕ := (C.map fun g => if g.op.budget.isSome then 1 else 0).sum

/-- The run inputs' declared bits. -/
def runBits (C : Circuit ι) : ℕ := (C.map fun g => match g.op with | .input .run _ => g.bits | _ => 0).sum

/-- The property's parameters: each budgeted class's cap per certified byte read, `R_in` and `L`. -/
structure Caps where
  rounding : ℕ
  compareSelect : ℕ
  gather : ℕ
  product : ℕ
  /-- `R_in`: run-input bytes a window -/
  runBytes : ℕ
  /-- `L`: the steps a value may be read after the step that produced it -/
  life : ℕ

/-- The budgeted units the caps allow per certified byte read. -/
def Caps.total (k : Caps) : ℕ := k.rounding + k.compareSelect + k.gather + k.product

/-- The audit's windows: `windows` of them, the window `win t` of each step `t`, and the bytes the audit certified
`W` read in each (`reads w`). -/
structure Audit where
  windows : ℕ
  win : ℕ → ℕ
  reads : ℕ → ℕ

/-- **The stacking property `P(c, R_in, L)`** of a declared circuit `W`, on its census and the audit's certified reads:
every gate is in a window; in each window each budgeted class is within its cap per certified byte read and the run
inputs within `R_in` bytes; and no gate reads a value produced more than `L` steps before it. -/
def Property (k : Caps) (A : Audit) (W : Circuit ι) : Prop :=
  (∀ g ∈ W, A.win g.step < A.windows) ∧
    (∀ w < A.windows,
      let c := census W (fun t => A.win t = w)
      c.rounding ≤ k.rounding * A.reads w ∧ c.compareSelect ≤ k.compareSelect * A.reads w ∧
        c.gather ≤ k.gather * A.reads w ∧ c.product ≤ k.product * A.reads w ∧ c.runBits ≤ 8 * k.runBytes) ∧
    ∀ i (h : i < W.length), ∀ j ∈ W[i].args, ∀ g ∈ W[j]?, W[i].step ≤ g.step + k.life

/-- `good` holds more than the fraction `ε = εnum/εden` of `S`: `ε·|S| < |good|`. -/
def MoreThan {κ : Type} (S good : Finset κ) (εnum εden : ℕ) : Prop :=
  εnum * S.card < εden * good.card

end Accounting.Gates
