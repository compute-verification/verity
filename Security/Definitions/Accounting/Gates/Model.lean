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
* **Gates.** A circuit (`Dag`) is a list of gates in the IR's canonical, topological order (§6): gate `i` reads the
  values of the gates `args` names, each earlier than `i` (`Topological`). A gate is one primitive: one `Value` out.
  Its values on an input are `values` (`val` for one gate's).
* **Provenance.** Each gate carries its primitive's id (`prim`), which names the device form the bridge to the actual
  run prices (`BlackwellS8ImmaDot32_v1` is sm_120's m16n8k32 IMMA, so `Op.mma` has no m and n), and its role in the
  Definition that made it (`Role`: a served product's step toward committed word `T_t`, a tail group's step, a limb
  split's gate, a receipt fold's). A gate is committed (`commit`) when its value is one of the circuit's committed
  outputs (`committed`): S8-PORT's COMMIT layout, the final `T_1` or the prefix `T_1 … T_{K/L}`.
* **Inputs.** An input gate reads slot `i` of the input assignment `x : ι → ℤ`, by its binding time (circuits' input
  origins, `verity/ml/circuits/ports.py`). A `registered` input (the weights, the store's payload, a codec's public
  parameters) is bound before the audit's coins; a `coin` input is a live verifier coin (a noise or fold key), passed
  through untouched; a `run` input (a request's tokens, anything the prover sends a running circuit) is bound after
  the coins, so it may be any function of them. Only run inputs are charged, and only they count against `R_in`.
* **Classes.** Free (no unit in any cap): an input, a constant (an operation's own bits), an integer affine map with
  constant coefficients (`I32Add`, `I32Sub`, a product with a constant), and an s8-family step (`mma`: the exact int8
  dot of `verity/ml/circuits/silicon/integer.py`, `D = C + Σ a_k·b_k` with s8 or u8 operands and s32 accumulation,
  clamped by `.satfinite` or wrapped mod 2^32 outside the s32 range). Budgeted, one unit per output value: a rounding
  (`ρ_b(x) = ⌊x/2^b⌋`, and every cast, quantizer, requantizer, FP elementwise op, integer floor, mod, division, bit op,
  limb split and word reinterpretation), a comparison or selection, a gather, a product of two runtime values, a
  receipt fold's step, and a neutral gate (a lane instruction whose class is its own, in no cap). A budgeted gate is
  any function of its operands (`budgeted`), the most a cheater could ask of a primitive, except a gather, which selects
  a word of its column by its index (`gather`). The registry's primitives map to exactly one class each
  (`Gates/Registry.lean`); an unlisted primitive is refused, so it has no `Op`.
* **The window.** On an input, every gate's exact value lies in its declared interval, and every s8-family step's
  operands lie in their formats and its exact sum in s32, so no accumulator clamps or wraps (`InWindow`). The property
  proves it for the free gates by interval analysis over the whole DAG from the declared intervals of inputs and
  budgeted gates (`Proven`), never from a free gate's declaration; so on an input on which the inputs and budgeted
  gates hold their declared intervals (`DeclaredHold`), the circuit is in its window, where an s8-family step's clamp
  and wrap agree (`Gate.Reclamped` relates the two arms).
* **Cost.** The census (`census`) counts the gates of a set of steps by class, the s8 family's MACs (served and tail
  apart), the limb splits' carries, the committed outputs and the inputs' bits. The counting bounds charge each
  budgeted gate and each run input the size of its declared interval (`volume`, `chargedBits`).
* **The property.** `Property caps A W`: the stacking property `P(c, R_in, L)` of property/v1 as a predicate on `W`'s
  census and the audit's certified reads: per window, each budgeted class within its cap per certified byte read, no
  neutral gate and the run inputs within `R_in`, every gate in a window, and no value read more than `L` engine steps
  after it is produced.
* **Fan-in.** MULT-LB's lane programs (`ofLane`: inputs, then instructions that each apply an arbitrary function to at
  most `f` earlier values, and `c` outputs) are this model's circuits of neutral gates of fan-in at most `f` (`FanIn`).
-/

namespace Accounting.Gates

/-- When an input is bound, as circuits' input origins: at registration, before the audit's coins; a live verifier
coin; or at run time, after the coins. -/
inductive Binding
  | registered
  | coin
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
  /-- a receipt fold's step: a fold's state after absorbing words (`PouwDuoAbsorb_v1`) -/
  | fold
  /-- a lane instruction whose class is its own (MULT-LB's `ofLane`): one unit, in no cap -/
  | neutral
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

/-- A gate's role in the Definition that made it (S8-PORT's `ops/s8.py`). -/
inductive Role
  /-- a step of a served product's main chain, toward its committed word `T_t` -/
  | served (t : ℕ)
  /-- a step of tail group `g` -/
  | tail (g : ℕ)
  /-- a limb split's gate: `carry` for its `N − 1` carries (`S32ByteCarry`, `S32ClampRest`), not for a limb read
  (`S32LowByte`, `S32ClampS8`) -/
  | limbSplit (carry : Bool)
  /-- a receipt fold's gate -/
  | fold
  /-- any other gate -/
  | other
  deriving DecidableEq, Repr

/-- A served product's step. -/
def Role.isServed : Role → Bool
  | .served _ => true
  | _ => false

/-- A tail group's step. -/
def Role.isTail : Role → Bool
  | .tail _ => true
  | _ => false

/-- A gate: its operation, its operands (earlier gates), its declared interval, the engine step that runs it, its
primitive's id, its role, and whether it is committed. -/
structure Gate (ι : Type) where
  op : Op ι
  args : List ℕ
  lo : ℤ
  hi : ℤ
  /-- the engine step (the serving engine's scheduler step, A1's `step`): not a device tick, nor an IMMA k-step -/
  step : ℕ
  /-- its primitive's id (`classOf`'s key), which gives its device form -/
  prim : String
  role : Role
  /-- its value is one of the circuit's committed outputs -/
  commit : Bool

/-- A circuit of the model: its gates in canonical order. -/
abbrev Dag (ι : Type) := List (Gate ι)

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

/-- `g'` computes as `g` does but for an s8-family step's clamp or wrap: the same operands, and the same operation
but for an s8-family step's `satfinite` (`BlackwellS8ImmaDot32_v1`'s clamp against `BlackwellU8S8ImmaDot32_v1`'s
wrap). -/
def Reclamped (g' : Gate ι) : Prop :=
  g'.args = g.args ∧ (g'.op = g.op ∨ ∃ a b k s s', g.op = .mma a b k s ∧ g'.op = .mma a b k s')

end Gate

variable {ι : Type}

/-- The values of a circuit's gates on input `x`, in order. -/
def values (C : Dag ι) (x : ι → ℤ) : List ℤ := C.foldl (fun vs g => vs ++ [g.value x vs]) []

/-- Gate `i`'s value on input `x`. -/
def val (C : Dag ι) (x : ι → ℤ) (i : ℕ) : ℤ := (values C x).getD i 0

/-- The committed outputs: the committed gates, in canonical order. -/
def committed (C : Dag ι) : List ℕ := (List.range C.length).filter fun i => C[i]?.any Gate.commit

/-- Canonical order: every gate's operands are earlier gates. -/
def Topological (C : Dag ι) : Prop := ∀ i (h : i < C.length), ∀ j ∈ C[i].args, j < i

/-- Slot `i` is read by an uncharged input of `C`: a registered or coin input. -/
def FreeInput (C : Dag ι) (i : ι) : Prop := ∃ g ∈ C, ∃ b, b ≠ .run ∧ g.op = .input b i

/-- **The bounded window** on input `x`: every gate's exact value lies in its declared interval, and no s8-family step's
accumulator clamps or wraps. -/
def InWindow (C : Dag ι) (x : ι → ℤ) : Prop := ∀ i (h : i < C.length), C[i].Ok x (values (C.take i) x)

/-- The product of the charges: the number of joint values the charged gates' declared intervals allow. -/
def volume (C : Dag ι) : ℕ := (C.map Gate.charge).prod

/-- The charged bits: the declared bits of the run inputs and budgeted gates. -/
def chargedBits (C : Dag ι) : ℕ := (C.map Gate.chargedBits).sum

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
def intervals (C : Dag ι) : List Iv := C.foldl (fun ivs g => ivs ++ [g.derive ivs]) []

/-- **The window, proven** (property/v1's interval analysis over the whole DAG): every constant, affine map and
s8-family step has its derived interval within its declared one, and every s8-family step its operands' derived
intervals within their formats and its declared interval within s32. -/
def Proven (C : Dag ι) : Prop :=
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
def DeclaredHold (C : Dag ι) (x : ι → ℤ) : Prop :=
  ∀ i (h : i < C.length), ¬ C[i].op.IsDerived →
    C[i].lo ≤ C[i].raw x (values (C.take i) x) ∧ C[i].raw x (values (C.take i) x) ≤ C[i].hi

/-! ## Fan-in: MULT-LB's lane programs -/

/-- Every budgeted gate but a gather reads at most `f` operands (a gather reads its index and the one word it
selects). -/
def FanIn (f : ℕ) (C : Dag ι) : Prop :=
  ∀ g ∈ C, g.op.charged → g.op.budget ≠ some .gather → g.args.length ≤ f

/-- Every value's declared interval lies within `[−2^w, 2^w)`: `w` bits and a sign, so s32 is `Width 31`. -/
def Width (w : ℕ) (C : Dag ι) : Prop := ∀ g ∈ C, -2 ^ w ≤ g.lo ∧ g.hi < 2 ^ w

/-- MULT-LB's lane program (`Accounting.Program.Prog`) as a circuit: run inputs `0 … N−1`, then each instruction a
neutral gate (`Budget.neutral`) that applies an arbitrary function to the earlier nodes it names, with the nodes `outs`
(its `c` outputs) committed. A lane program is abstract, so its gates name no primitive (`prim = ""`). -/
def ofLane (N : ℕ) (instrs : List (List ℕ × (List ℤ → ℤ))) (outs : List ℕ) (lo hi : ℤ) : Dag ℕ :=
  (List.range N).map (fun i => ⟨.input .run i, [], lo, hi, 0, "", .other, decide (i ∈ outs)⟩) ++
    instrs.zipIdx.map fun e => ⟨.budgeted .neutral e.1.2, e.1.1, lo, hi, 0, "", .other, decide (N + e.2 ∈ outs)⟩

/-! ## The census and the property -/

/-- A circuit's census over a set of steps: its gates by class, the s8 family's MACs, the limb splits' carries, its
committed outputs, and its inputs' bits. -/
structure Census where
  /-- s8-family coordinates -/
  family : ℕ
  /-- their MACs: `k` each -/
  macs : ℕ
  /-- the MACs of served products' main chains (S8-PORT's `served_macs`) -/
  servedMacs : ℕ
  /-- the MACs of tail groups (S8-PORT's `tail_atoms`: 32 an atom) -/
  tailMacs : ℕ
  /-- affine maps and constants -/
  affine : ℕ
  /-- roundings -/
  rounding : ℕ
  /-- the limb splits' carries (S8-PORT's `carries`: `N − 1` a split) -/
  carries : ℕ
  /-- comparisons and selections -/
  compareSelect : ℕ
  /-- gathers -/
  gather : ℕ
  /-- products of two runtime values -/
  product : ℕ
  /-- receipt folds' steps -/
  fold : ℕ
  /-- neutral gates -/
  neutral : ℕ
  /-- committed outputs -/
  committed : ℕ
  /-- the run inputs' declared bits -/
  runBits : ℕ
  /-- the coin inputs' declared bits -/
  coinBits : ℕ
  /-- the registered inputs' declared bits -/
  registeredBits : ℕ

/-- The number of gates of `C` at steps in `s` that `p` picks. -/
def countAt (C : Dag ι) (s : ℕ → Prop) [DecidablePred s] (p : Gate ι → ℕ) : ℕ :=
  ((C.filter fun g => decide (s g.step)).map p).sum

/-- One for a gate of budgeted class `b`. -/
def Gate.isBudget (b : Budget) (g : Gate ι) : ℕ := if g.op.budget = some b then 1 else 0

/-- An s8-family step's MACs, `k`; none for another gate. -/
def Gate.macs (g : Gate ι) : ℕ := match g.op with | .mma _ _ k _ => k | _ => 0

/-- The input bits of binding `b`. -/
def Gate.inputBits (b : Binding) (g : Gate ι) : ℕ :=
  match g.op with
  | .input b' _ => if b' = b then g.bits else 0
  | _ => 0

/-- **The census** of `C`'s gates at the steps `s` picks. -/
def census (C : Dag ι) (s : ℕ → Prop) [DecidablePred s] : Census where
  family := countAt C s fun g => match g.op with | .mma .. => 1 | _ => 0
  macs := countAt C s Gate.macs
  servedMacs := countAt C s fun g => if g.role.isServed then g.macs else 0
  tailMacs := countAt C s fun g => if g.role.isTail then g.macs else 0
  affine := countAt C s fun g => match g.op with | .affine .. | .const _ => 1 | _ => 0
  rounding := countAt C s (Gate.isBudget .rounding)
  carries := countAt C s fun g => if g.role = .limbSplit true then 1 else 0
  compareSelect := countAt C s (Gate.isBudget .compareSelect)
  gather := countAt C s (Gate.isBudget .gather)
  product := countAt C s (Gate.isBudget .product)
  fold := countAt C s (Gate.isBudget .fold)
  neutral := countAt C s (Gate.isBudget .neutral)
  committed := countAt C s fun g => if g.commit then 1 else 0
  runBits := countAt C s (Gate.inputBits .run)
  coinBits := countAt C s (Gate.inputBits .coin)
  registeredBits := countAt C s (Gate.inputBits .registered)

/-- The budgeted units of a census: every budgeted class, one unit a gate. -/
def Census.budgeted (c : Census) : ℕ := c.rounding + c.compareSelect + c.gather + c.product + c.fold + c.neutral

/-- The budgeted gates of a circuit, one unit each. -/
def budgeted (C : Dag ι) : ℕ := (C.map fun g => if g.op.budget.isSome then 1 else 0).sum

/-- The run inputs' declared bits. -/
def runBits (C : Dag ι) : ℕ := (C.map (Gate.inputBits .run)).sum

/-- The property's parameters: each budgeted class's cap per certified byte read, `R_in` and `L`. -/
structure Caps where
  rounding : ℕ
  compareSelect : ℕ
  gather : ℕ
  product : ℕ
  fold : ℕ
  /-- `R_in`: run-input bytes a window -/
  runBytes : ℕ
  /-- `L`: the engine steps a value may be read after the step that produced it -/
  life : ℕ

/-- The budgeted units the caps allow per certified byte read. -/
def Caps.total (k : Caps) : ℕ := k.rounding + k.compareSelect + k.gather + k.product + k.fold

/-- The audit's windows: `windows` of them, the window `win t` of each engine step `t`, and the bytes the audit
certified `W` read in each (`reads w`). -/
structure Audit where
  windows : ℕ
  win : ℕ → ℕ
  reads : ℕ → ℕ

/-- **The stacking property `P(c, R_in, L)`** of a declared circuit `W`, on its census and the audit's certified reads:
every gate is in a window; in each window each budgeted class is within its cap per certified byte read, no gate is
neutral (the registry classes every primitive) and the run inputs are within `R_in` bytes; and no gate reads a value
produced more than `L` engine steps before it. -/
def Property (k : Caps) (A : Audit) (W : Dag ι) : Prop :=
  (∀ g ∈ W, A.win g.step < A.windows) ∧
    (∀ w < A.windows,
      let c := census W (fun t => A.win t = w)
      c.rounding ≤ k.rounding * A.reads w ∧ c.compareSelect ≤ k.compareSelect * A.reads w ∧
        c.gather ≤ k.gather * A.reads w ∧ c.product ≤ k.product * A.reads w ∧ c.fold ≤ k.fold * A.reads w ∧
          c.neutral = 0 ∧ c.runBits ≤ 8 * k.runBytes) ∧
    ∀ i (h : i < W.length), ∀ j ∈ W[i].args, ∀ g ∈ W[j]?, W[i].step ≤ g.step + k.life

/-- `good` holds more than the fraction `ε = εnum/εden` of `S`: `ε·|S| < |good|`. -/
def MoreThan {κ : Type} (S good : Finset κ) (εnum εden : ℕ) : Prop :=
  εnum * S.card < εden * good.card

end Accounting.Gates
