import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Data.Nat.Log
import Mathlib.Data.Finset.Card

/-!
# The circuit model: a declared circuit's gates, their classes, its window and its census (trusted)

One model of what a declared circuit `W` computes, for the lower bounds every resource audit proves over it: the
stacking property's carry and encode bounds, the requirement `A1` reads, and compute accounting's work statement and
fan-in floor. It follows the IR (`verity/ml/circuits/PROTOCOL.md`) gate for gate, so that census/v1 and property/v1
(`verity/ml/circuits/census.py`, `stacking.py`) can be held to it by difftest.

* **Values.** Every value is an integer (`ℤ`): a `Value<w>` leaf read as the integer its Definition gives it, two's
  complement for a signed word, and a float's encoding read signed at its width as property/v1 reads it (an f32 as
  s32, a bf16 as s16, `signed`), so the exact widening `Bf16ToF32` is `×2^16`. A gate's declared interval `[lo, hi]`
  is its type's range, or the narrower one an interval analysis derives.
* **Gates.** A circuit (`IntDag`) is a list of gates in the IR's canonical, topological order (§6): gate `i` reads the
  values of the gates `args` names, each earlier than `i` (`Topological`). A gate is one primitive: one `Value` out.
  Its values on an input are `values` (`val` for one gate's).
* **Provenance.** Each gate carries its primitive's id (`prim`), which names the device form the bridge to the actual
  run prices (`BlackwellS8ImmaDot32_v1` is sm_120's m16n8k32 IMMA, so `IntOp.mma` has no m and n), and where it sits in the
  Definition that made it, as the IR binds it (`place`: step `s` of an `S8Dot_v1` body, with the statics `K`, `L` and
  `TAIL`; or word `w` of an H1 + fx1 tile's CP-W receipt, with the tile's dot kind, `K` and operand rows). Its role
  (`IntGate.role`: a served product's step toward committed word `T_t`, a tail group's step, a limb split's carry or limb
  read, a receipt fold's step) is a function of the two (`roleOf`), never a field, so no gate carries a role its
  primitive and statics don't give it. A gate is committed (`commit`) when its value is one of the circuit's committed
  outputs (`committed`): S8-PORT's COMMIT layout, the final `T_1` or the prefix `T_1 … T_{K/L}`, or an H1 + fx1 tile's
  receipt. Of an H1 + fx1 tile only the receipt's words after its `K`-segment checkpoints have the `served` role:
  `2·K/32` of them, `256` at `K = 4096`, each checkpoint's two halves (`Place.parts`), never its `z` or a lane's own
  value.
* **Inputs.** An input gate reads slot `i` of the input assignment `x : ι → ℤ`, by its binding time (circuits' input
  origins, `verity/ml/circuits/ports.py`). A `registered` input (the weights, the store's payload, a codec's public
  parameters) is bound before the audit's coins; a `coin` input is a live verifier coin (a noise or fold key), passed
  through untouched; a `run` input (a request's tokens, anything the prover sends a running circuit) is bound after
  the coins it follows. Compose's live commit order (`LayerCommitted`, a premise today's coins don't meet) puts it
  under the root of its first reader's round, before that round's coin, so it may be any function of the coins
  released before that round's and never of its reader's: one chosen after its reader's coin could match the reader's
  noise. Only run inputs are charged, and only they count against `R_in`.
* **Classes.** Free (no unit in any cap): an input, a constant (an operation's own bits), an exact integer affine map
  with constant coefficients (`I32Add`, `I32Sub`, and `Bf16ToF32`, `×2^16` on the readings), proven within s32, and
  an s8-family step (`mma`: the exact int8 dot of `verity/ml/circuits/silicon/integer.py`, `D = C + Σ a_k·b_k` with
  s8 or u8 operands and s32 accumulation, clamped by `.satfinite` or wrapped mod 2^32 outside the s32 range).
  Budgeted: a rounding (`ρ_b(x) = ⌊x/2^b⌋`, and every cast, quantizer, requantizer, FP op, integer floor, mod,
  division, product reduced mod `2^w`, bit op on words, field extraction, limb split and word reinterpretation, but
  the widening `Bf16ToF32`), a comparison, selection or 1-bit logic op, a gather, a receipt
  fold's step, and a neutral gate (a lane instruction whose class is its own, in no cap). A budgeted gate is any
  function of its operands (`budgeted`), the most a cheater could ask of a primitive, except a gather, which selects a
  word of its column by its index, or its family's out-of-range constant (its `fill`, the evaluator's NaN or `−1`)
  for an index outside the column, negative or past it, or a table read's word by its index's low bits (`gather`).
  The registry's primitives map to exactly one class each (`Gates/Registry.lean`); an unlisted primitive is refused,
  so it has no `IntOp`.
* **The window.** On an input, every gate's exact value lies in its declared interval, and every s8-family step's
  operands lie in their formats and its exact sum in s32, so no accumulator clamps or wraps (`InWindow`). The property
  proves it for the free gates and the gathers by interval analysis over the whole DAG from the declared intervals of
  inputs and budgeted gates but gathers (`Proven`), never from a free gate's or a gather's declaration (a gather's
  value is a word of its column or its fill, so its interval is the hull of its column's and its fill: a gather at a
  fixed index can't re-declare a wide value narrow), and proves each affine map's declared interval within s32, where
  `I32Add`, `I32Sub` and `Bf16ToF32` are exact; so on an input on which the inputs and those budgeted gates hold their
  declared intervals (`DeclaredHold`), the circuit is in its window, where an s8-family step's clamp and wrap agree
  (`IntGate.Reclamped` relates the two arms).
* **Units** (property/v1's count, `units`). A budgeted gate is one unit when it reads a value that depends on a run or
  coin input (`dynamic`), and none otherwise: a gate of constants and registered values computes the same value at
  registration. A gather whose index depends on one is the index bits no earlier gather of that index and kind paid
  (fresh index bits, `⌈log₂ (V + 1)⌉` for a column of `V` words and the fill outside it, so a one-word column's
  `[i = 0]` pays as a comparison does, and `n` for a table read of the index's low `n` bits; property/v1's `range` and
  `low` kinds), and one more when its column holds such a value too: a selection among run-time values is a
  `V`-way select. Without that unit a tree of gathers that share indices, over constant leaves, would compute any
  function of `b` index bits for `b` units. A gather at a fixed index is wiring.
* **Data products** (`dataProducts`). The only free operations are linear maps whose coefficients are fixed before
  the run's coins: constants, registered values (the pre-committed weights), additions and wiring. So an s8-family
  lane is free when one of its multiplicands is a constant or registered value, and one unit when both depend on a
  run or coin input (`dynamic`), whatever their width or range: an exact product of two data values is a universal
  gate (an AND of two bits is one, and the bitwise codec computes any function with them,
  `Accounting.SecurityProofs.Gates.EncodeChainRefuted`). A projection by the weights is free; attention's `QKᵀ` and
  `PV` are data products. Every other nonlinear operation (a rounding, comparison or selection, exp, a division, a
  gather at a data-dependent index) is a budgeted gate and pays its units above. **The nonlinear units**
  (`nonlinear`) are the budgeted units and the data products: the circuit's multiplicative complexity in this model,
  every operation but a fixed linear map charged.
* **Cost.** The census (`census`) counts the units of a set of steps by class, the s8 family's MACs, the limb splits'
  carries, the committed outputs and the inputs' bits. Its MACs are every s8-family step's, read from the gate whatever
  its role (`dotMacs`: an H1 + fx1 tile's segment dots and `z` among them), and apart from those the served and the tail
  ones, by role (`servedMacs`, `tailMacs`), so an H1 + fx1 tile, whose served role is on its receipt's words, has none
  served. The counting bounds charge each budgeted gate and each run input the size of its declared interval
  (`volume`, `chargedBits`).
* **The property.** `Property caps A W`: the stacking property `P(c, R_in, L)` of property/v1 as a predicate on `W`'s
  census and the audit's certified reads: per window, each budgeted class's units and the data products each within
  their cap (a rational) per certified byte read, no neutral unit and the run inputs within the window's own `R_in`
  bound, every gate in a window, and no value read more than `L` engine steps after it is produced. A product by a
  weight is free and in no cap, so no MAC is capped as such.
* **The chain's restriction.** Every budgeted gate reads at most `B` operand bits, the declared bits of its operands
  (`OperandBits`): one gate that reads a word's bits computes a byte of its root.
* **Fan-in.** MULT-LB's lane programs (`ofLane`: inputs, then instructions that each apply an arbitrary function to at
  most `f` earlier values, and `c` outputs) are this model's circuits of neutral gates of fan-in at most `f` (`FanIn`).
-/

namespace Accounting.Gates

/-- When an input is bound, as circuits' input origins: at registration, before the audit's coins; a live verifier
coin; or at run time, after the coins it follows (Compose's `LayerCommitted` puts it under the root of its first
reader's round, before that round's coin). -/
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

/-- A `w`-bit word's reading: its low `w` bits as two's complement, in `[−2^(w−1), 2^(w−1))`, as property/v1 reads a
value (an f32's encoding at `w = 32`, a bf16's at `w = 16`). -/
def signed (w u : ℕ) : ℤ := ((u : ℤ) + 2 ^ (w - 1)) % 2 ^ w - 2 ^ (w - 1)

/-- A budgeted class: what the property caps, in units (`units`). -/
inductive Budget
  /-- `ρ_b(x) = ⌊x/2^b⌋`, and every cast, quantizer, requantizer, FP op, integer floor, mod, division or product
  reduced mod `2^w`, bit op on words, field extraction, limb split and word reinterpretation -/
  | rounding
  /-- a comparison, a selection by a condition, a min, max or clamp, a logic op on 1-bit values -/
  | compareSelect
  /-- a selection of a word of a column by a runtime index -/
  | gather
  /-- a receipt fold's step: a fold's state after absorbing words (`PouwDuoAbsorb_v1`) -/
  | fold
  /-- a lane instruction whose class is its own (MULT-LB's `ofLane`): one unit, in no cap -/
  | neutral
  deriving DecidableEq, Repr

/-- A gate's operation, over input slots `ι`. -/
inductive IntOp (ι : Type)
  /-- the input in slot `i`, bound at `b` -/
  | input (b : Binding) (i : ι)
  /-- a constant: an operation's own bits -/
  | const (c : ℤ)
  /-- `c₀ + Σ coeff_k · v_k`: an integer affine map with constant coefficients -/
  | affine (c₀ : ℤ) (coeff : List ℤ)
  /-- an s8-family step's coordinate: operands `C :: a_1 … a_k ++ b_1 … b_k`, `C + Σ a_i·b_i`, then clamped
  (`satfinite`) or wrapped to s32 -/
  | mma (a b : Int8Fmt) (k : ℕ) (satfinite : Bool)
  /-- a gather: operands `idx :: column`, the column's word at `idx` (`fill` outside it, below 0 or past its end), or
  for a table read (`low = some n`) at `idx`'s low `n` bits -/
  | gather (low : Option ℕ) (fill : ℤ)
  /-- a budgeted primitive of class `cls`: any function of its operands -/
  | budgeted (cls : Budget) (f : List ℤ → ℤ)

/-- The dot of the first `k` operands with the next `k`. -/
def dot (k : ℕ) (v : List ℤ) : ℤ := (List.zipWith (· * ·) (v.take k) (v.drop k)).sum

namespace IntOp

variable {ι : Type}

/-- `ρ_b(x) = ⌊x/2^b⌋`, the rounding. -/
def round (b : ℕ) : IntOp ι := budgeted .rounding fun v => v.headD 0 / 2 ^ b

/-- `[a ≤ b]`, a comparison. -/
def le : IntOp ι := budgeted .compareSelect fun v => if v.headD 0 ≤ v.tail.headD 0 then 1 else 0

/-- `c ≠ 0 ? a : b`, a selection. -/
def select : IntOp ι := budgeted .compareSelect fun v => if v.headD 0 ≠ 0 then v.tail.headD 0 else v.tail.tail.headD 0

/-- The position a gather reads in a column of `V` words at index `z`: `z` itself, or `V`, past the column, for a
negative `z` (the evaluator reads its token as s32 and takes the fill outside `[0, V)`); or for a table read of `n`
bits its low `n` bits (two's complement's, `z mod 2^n`). -/
def gatherAt : Option ℕ → ℕ → ℤ → ℕ
  | none, V, z => if z < 0 then V else z.toNat
  | some n, _, z => (z % 2 ^ n).toNat

/-- The exact value of an operation on operands `v`, before an s8-family step's clamp or wrap. -/
def raw : IntOp ι → (ι → ℤ) → List ℤ → ℤ
  | input _ i, x, _ => x i
  | const c, _, _ => c
  | affine c₀ cs, _, v => c₀ + (List.zipWith (· * ·) cs v).sum
  | mma _ _ k _, _, v => v.headD 0 + dot k v.tail
  | gather low d, _, v => v.tail.getD (gatherAt low v.tail.length (v.headD 0)) d
  | budgeted _ f, _, v => f v

/-- The value an operation produces from its exact value: an s8-family step clamps or wraps it, every other passes
it. -/
def out : IntOp ι → ℤ → ℤ
  | mma _ _ _ true, z => clampS32 z
  | mma _ _ _ false, z => wrapS32 z
  | _, z => z

/-- An operation whose value the counting bounds charge: a run input or a budgeted gate. -/
def charged : IntOp ι → Bool
  | input .run _ => true
  | gather _ _ => true
  | budgeted _ _ => true
  | _ => false

/-- Its budgeted class, if any. -/
def budget : IntOp ι → Option Budget
  | gather _ _ => some .gather
  | budgeted cls _ => some cls
  | _ => none

/-- A gather's kind, property/v1's kind of its index bits: `true` for a table read of its index's low bits (`low`),
`false` for a read at its index (`range`); `none` for any other operation. -/
def gatherKind : IntOp ι → Option Bool
  | gather low _ => some low.isSome
  | _ => none

/-- An s8-family step's operands are `C` and `2k` words in their formats; other operations ask nothing. -/
def OperandsOk : IntOp ι → List ℤ → Prop
  | mma a b k _, v =>
      v.length = 2 * k + 1 ∧ (∀ z ∈ v.tail.take k, a.lo ≤ z ∧ z ≤ a.hi) ∧
        ∀ z ∈ v.tail.drop k, b.lo ≤ z ∧ z ≤ b.hi
  | _, _ => True

/-- An s8-family step. -/
def IsMma : IntOp ι → Prop
  | mma .. => True
  | _ => False

/-- An operation whose interval the property derives from its operands', never its declaration: the free ones (a
constant, an affine map, an s8-family step) and a gather, whose value is a word of its column or its fill. -/
def IsDerived : IntOp ι → Prop
  | const _ | affine .. | mma .. | gather .. => True
  | _ => False

end IntOp

/-- A gate's role in the Definition that made it (S8-PORT's `ops/s8.py`, and H1 + fx1's CP-W receipt). -/
inductive Role
  /-- a step of a served product's main chain, toward its committed word `T_t` (its first `t·L` products), or an
  H1 + fx1 tile's receipt word, either half of the checkpoint that binds its first `32·t` products -/
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

/-- Where a gate sits in the Definition that made it, as the IR binds it:
* `dot K L tail s`: step `s` of an `S8Dot_v1` coordinate's body (an `S8Matmul_v1` row's among them), whose statics
  are `K` products committed every `L` (`T_t` after `t·L`) and the tail groups' atom counts (`TAIL`);
* `receipt dot K w a b`: word `w` of an H1 + fx1 tile's CP-W receipt (`PouwH1Tile`,
  `verity/core/protocols/pouw/circuit/h1tile.py`): the XOR over the tile's lanes of half `w % 2` of each lane's fx1
  state after set `w / 2` of `[T_0 … T_{KS−1}, O′, D_1 … D_{m−1}]`, `2·(KS + m)` words at contraction `K` (`KS = K/32`
  segments). `dot` is the tile's dot kind (its `EXACT`: `S8Dot_v2` or `GemmAccumulatorS8_v2`), and `a` and `b` are the
  gates of its `A` and `B` operand rows (its call's arguments), from which a word's kind reads whether each operand is
  an activation, so that a kind reading two (H1 + fx1's attention) stays uncredited under seeded coins (Compose's
  `Scheme.SeedPriced`);
* `other`: any other place. Every other gate of an H1 + fx1 tile is here: its `z` (each cell's exact product, `EXACT` on
  its rows, which no noise reaches and PIN's warp check never compares), its noise, its sets and folds, and each lane's
  own `PouwXinFinal` words, which aren't committed: only their XOR over lanes is. -/
inductive Place
  | dot (K L : ℕ) (tail : List ℕ) (s : ℕ)
  | receipt (dot : String) (K w : ℕ) (a b : List ℕ)
  | other
  deriving DecidableEq, Repr

/-- **The committed words one checkpoint is split into**: `2` for an H1 + fx1 tile's receipt word (a checkpoint's two
halves, both of role `served t` at one `t`), `1` at any other place. A floor priced per checkpoint, as a dot kind's is,
counts once across a checkpoint's parts, `1/parts` a word. -/
def Place.parts : Place → ℕ
  | .receipt .. => 2
  | _ => 1

/-- Tail atom `j`'s group, among groups of the given atom counts numbered from `g`. -/
def tailRole : List ℕ → ℕ → ℕ → Role
  | [], _, _ => .other
  | a :: as, j, g => if j < a then .tail g else tailRole as (j - a) (g + 1)

/-- **The role** of a gate of primitive `prim` at `place` (S8-PORT's `ops/s8.py`). Step `s < K/32` of an `S8Dot_v1`
body is on the main chain toward `T_t`, `t = ⌊32·s/L⌋ + 1`, and a later step is tail atom `s − K/32`, in the group
that holds it.

An H1 + fx1 tile's receipt word `w` after checkpoint `T_s` (`w < 2·(K/32)`, `s = w / 2`, either half) has role
`served (s + 1)`, toward its products' `T_{s+1}`, their first `32·(s + 1)`: each checkpoint binds one 32-wide
`K`-segment of the tile's products, whose floor its dot kind prices (GRIND's fold premise, `HMu`'s Lemma B′), and its
two halves share it (`Place.parts`). The last `2·m` words, after `O′ = T_{KS−1} − U·A_0` and `D_d = U·A_d`, are `other`:
they bind the check's cancellation of the columns' keyed word `U` against the rows' digits, which no segment's floor
prices, so they carry no floor. A receipt word's role reads its place alone, not the primitive that computes it (the
lanes' `PouwXor32`).

Off a dot body and a receipt, `S32ByteCarry` and `S32ClampRest` are a limb split's carries (`N − 1` a split, in
either form), `S32LowByte` and `S32ClampS8` its limb reads, and `PouwDuoAbsorb_v1` the receipt fold's step. -/
def roleOf (prim : String) : Place → Role
  | .dot K L tail s => if s < K / 32 then .served (32 * s / L + 1) else tailRole tail (s - K / 32) 0
  | .receipt _ K w _ _ => if w < 2 * (K / 32) then .served (w / 2 + 1) else .other
  | .other =>
      if prim ∈ ["S32ByteCarry_v1", "S32ByteCarry_v2", "S32ClampRest_v1", "S32ClampRest_v2"] then .limbSplit true
      else if prim ∈ ["S32LowByte_v1", "S32LowByte_v2", "S32ClampS8_v1", "S32ClampS8_v2"] then .limbSplit false
      else if prim = "PouwDuoAbsorb_v1" then .fold
      else .other

/-- A gate: its operation, its operands (earlier gates), its declared interval, the engine step that runs it, its
primitive's id, where it sits in its Definition, and whether it is committed. -/
structure IntGate (ι : Type) where
  op : IntOp ι
  args : List ℕ
  lo : ℤ
  hi : ℤ
  /-- the engine step (the serving engine's scheduler step, A1's `step`): not a device tick, nor an IMMA k-step -/
  step : ℕ
  /-- its primitive's id (`classOf`'s key), which gives its device form -/
  prim : String
  /-- where it sits in the Definition that made it, from the IR's binding -/
  place : Place
  /-- its value is one of the circuit's committed outputs -/
  commit : Bool

/-- A circuit of the model: its gates in canonical order. -/
abbrev IntDag (ι : Type) := List (IntGate ι)

namespace IntGate

variable {ι : Type} (g : IntGate ι)

/-- Its role: its primitive's and place's (`roleOf`). -/
def role : Role := roleOf g.prim g.place

/-- A gather's index bits: `⌈log₂ (V + 1)⌉` for its column of `V` words, the bits that tell its `V + 1` outcomes apart:
a word of the column, or its fill for an index outside it. A one-word column is a comparison (`[i = 0]`), one bit. A
table read of its index's low `n` bits reads `n`, which tell its `2^n` positions apart (property/v1's `low` kind). -/
def indexBits : ℕ :=
  match g.op with
  | .gather (some n) _ => n
  | _ => Nat.clog 2 g.args.length

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
def Reclamped (g' : IntGate ι) : Prop :=
  g'.args = g.args ∧ (g'.op = g.op ∨ ∃ a b k s s', g.op = .mma a b k s ∧ g'.op = .mma a b k s')

end IntGate

variable {ι : Type}

/-- The values of a circuit's gates on input `x`, in order. -/
def values (C : IntDag ι) (x : ι → ℤ) : List ℤ := C.foldl (fun vs g => vs ++ [g.value x vs]) []

/-- Gate `i`'s value on input `x`. -/
def val (C : IntDag ι) (x : ι → ℤ) (i : ℕ) : ℤ := (values C x).getD i 0

/-- The committed outputs: the committed gates, in canonical order. -/
def committed (C : IntDag ι) : List ℕ := (List.range C.length).filter fun i => C[i]?.any IntGate.commit

/-- Canonical order: every gate's operands are earlier gates. -/
def Topological (C : IntDag ι) : Prop := ∀ i (h : i < C.length), ∀ j ∈ C[i].args, j < i

/-- Slot `i` is read by an uncharged input of `C`: a registered or coin input. -/
def FreeInput (C : IntDag ι) (i : ι) : Prop := ∃ g ∈ C, ∃ b, b ≠ .run ∧ g.op = .input b i

/-- **The bounded window** on input `x`: every gate's exact value lies in its declared interval, and no s8-family step's
accumulator clamps or wraps. -/
def InWindow (C : IntDag ι) (x : ι → ℤ) : Prop := ∀ i (h : i < C.length), C[i].Ok x (values (C.take i) x)

/-- The product of the charges: the number of joint values the charged gates' declared intervals allow. -/
def volume (C : IntDag ι) : ℕ := (C.map IntGate.charge).prod

/-- The charged bits: the declared bits of the run inputs and budgeted gates. -/
def chargedBits (C : IntDag ι) : ℕ := (C.map IntGate.chargedBits).sum

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

/-- The hull of `[a]` and `[b]`. -/
def Iv.join (a b : Iv) : Iv := (min a.1 b.1, max a.2 b.2)

/-- The interval a gate's value is derived to lie in, from its operands' derived intervals `ivs`: a constant's own
value, an affine map's or an s8-family step's image, a gather's hull of its column's intervals and its fill (its
value outside the column), and every other gate's declared interval. -/
def IntGate.derive (g : IntGate ι) (ivs : List Iv) : Iv :=
  let a := g.args.map fun j => ivs.getD j (0, 0)
  match g.op with
  | .const c => (c, c)
  | .affine c₀ cs => (List.zipWith Iv.scale cs a).foldl Iv.add (c₀, c₀)
  | .mma _ _ k _ =>
      (List.zipWith Iv.mul (a.tail.take k) (a.tail.drop k)).foldl Iv.add (a.headD (0, 0))
  | .gather _ d => a.tail.foldl Iv.join (d, d)
  | _ => (g.lo, g.hi)

/-- The derived intervals of a circuit's gates, in order. -/
def intervals (C : IntDag ι) : List Iv := C.foldl (fun ivs g => ivs ++ [g.derive ivs]) []

/-- **The window, proven** (property/v1's interval analysis over the whole DAG): every constant, affine map,
s8-family step and gather has its derived interval within its declared one, every affine map its declared interval
within s32 (where `I32Add`, `I32Sub` and `Bf16ToF32` don't wrap), and every s8-family step its operands' derived
intervals within their formats and its declared interval within s32. -/
def Proven (C : IntDag ι) : Prop :=
  ∀ i (h : i < C.length),
    let g := C[i]
    let ivs := intervals (C.take i)
    (g.op.IsDerived → (g.derive ivs).Within g.lo g.hi) ∧
      (∀ c₀ cs, g.op = .affine c₀ cs → s32Min ≤ g.lo ∧ g.hi ≤ s32Max) ∧
      ∀ fa fb k sat, g.op = .mma fa fb k sat →
        g.args.length = 2 * k + 1 ∧ s32Min ≤ g.lo ∧ g.hi ≤ s32Max ∧
          (∀ j ∈ g.args.tail.take k, (ivs.getD j (0, 0)).Within fa.lo fa.hi) ∧
          ∀ j ∈ g.args.tail.drop k, (ivs.getD j (0, 0)).Within fb.lo fb.hi

/-- Every input's and budgeted gate's but a gather's value lies in its declared interval on input `x`: what a
primitive's output type guarantees, and a run input's. -/
def DeclaredHold (C : IntDag ι) (x : ι → ℤ) : Prop :=
  ∀ i (h : i < C.length), ¬ C[i].op.IsDerived →
    C[i].lo ≤ C[i].raw x (values (C.take i) x) ∧ C[i].raw x (values (C.take i) x) ≤ C[i].hi

/-! ## Fan-in: MULT-LB's lane programs -/

/-- Every budgeted gate but a gather reads at most `f` operands (a gather reads its index and the one word it
selects). -/
def FanIn (f : ℕ) (C : IntDag ι) : Prop :=
  ∀ g ∈ C, g.op.charged → g.op.budget ≠ some .gather → g.args.length ≤ f

/-- Every value's declared interval lies within `[−2^w, 2^w)`: `w` bits and a sign, so s32 is `Width 31`. -/
def Width (w : ℕ) (C : IntDag ι) : Prop := ∀ g ∈ C, -2 ^ w ≤ g.lo ∧ g.hi < 2 ^ w

/-- **The chain's restriction**: every budgeted gate reads at most `B` operand bits, the declared bits of the operands
it reads. A gather is exempt: its value is one of its operands. -/
def OperandBits (B : ℕ) (C : IntDag ι) : Prop :=
  ∀ g ∈ C, ∀ cls f, g.op = .budgeted cls f → (g.args.map fun j => (C[j]?.map IntGate.bits).getD 0).sum ≤ B

/-- A MULT-LB lane program (a straight-line dataflow, `Accounting.TS.Dataflow`, over its run inputs) as a circuit: run
inputs `0 … N−1`, then each instruction a neutral gate (`Budget.neutral`) that applies an arbitrary function to the
earlier nodes it names, with the nodes `outs` (its `c` outputs) committed. A lane program is abstract, so its gates
name no primitive (`prim = ""`). -/
def ofLane (N : ℕ) (instrs : List (List ℕ × (List ℤ → ℤ))) (outs : List ℕ) (lo hi : ℤ) : IntDag ℕ :=
  (List.range N).map (fun i => ⟨.input .run i, [], lo, hi, 0, "", .other, decide (i ∈ outs)⟩) ++
    instrs.zipIdx.map fun e => ⟨.budgeted .neutral e.1.2, e.1.1, lo, hi, 0, "", .other, decide (N + e.2 ∈ outs)⟩

/-! ## Units, the census and the property -/

/-- Whether each gate's value depends on a run or coin input, in order: an input bound after registration, or a gate
that reads such a value (property/v1's run dependence: circuits bind a coin as a run input of kind coin). -/
def dynamic (C : IntDag ι) : List Bool :=
  C.foldl (fun ds g => ds ++ [match g.op with
    | .input b _ => b != .registered
    | _ => g.args.any fun j => ds.getD j false]) []

/-- **Gate `i`'s units** (property/v1's count). A budgeted gate is one unit when it reads a value that depends on a run
or coin input, and none when it doesn't (it computes the same value at registration). A gather whose index depends on
one is the index bits no earlier gather of that index and kind paid (fresh index bits, per index and kind as
property/v1 keys them), and one more when its column holds such a value: a `V`-way selection among run-time values,
without which a tree of gathers sharing indices would select among `2^b` constants for `b` units. A gather at a fixed
index is wiring, and every other gate is free. -/
def units (C : IntDag ι) (i : ℕ) : ℕ :=
  let dyn := fun j => (dynamic C).getD j false
  match C[i]? with
  | none => 0
  | some g =>
    match g.op, g.args with
    | .budgeted _ _, args => if args.any dyn then 1 else 0
    | .gather low _, j :: col =>
        if dyn j then
          (g.indexBits - (((C.take i).filter fun h =>
              h.op.gatherKind == some low.isSome && h.args.head? == some j).map IntGate.indexBits).foldl max 0) +
            (if col.any dyn then 1 else 0)
        else 0
    | _, _ => 0

/-- **Gate `i`'s data products**: the lanes of an s8-family step whose two multiplicands both depend on a run or coin
input (`dynamic`), one unit each whatever their width or range. A lane with a constant or registered multiplicand is a
linear map of the other with a coefficient fixed before the coins, and is free. -/
def dataProducts (C : IntDag ι) (i : ℕ) : ℕ :=
  let dyn := fun j => (dynamic C).getD j false
  match C[i]? with
  | some g =>
    match g.op with
    | .mma _ _ k _ => ((g.args.tail.take k).zip (g.args.tail.drop k)).countP fun ab => dyn ab.1 && dyn ab.2
    | _ => 0
  | none => 0

/-- A circuit's census over a set of steps: its units by class, the s8 family's MACs, the limb splits' carries, its
committed outputs, and its inputs' bits. -/
structure Census where
  /-- s8-family coordinates -/
  family : ℕ
  /-- their MACs, `k` each, whatever their role or place: every dot product the steps compute -/
  dotMacs : ℕ
  /-- the MACs of the steps whose role is served (S8-PORT's `served_macs`): a served product's main chain toward its
  committed words; none of an H1 + fx1 tile's, whose served role is on its receipt's words, which take no MACs -/
  servedMacs : ℕ
  /-- the MACs of tail groups (S8-PORT's `tail_atoms`: 32 an atom) -/
  tailMacs : ℕ
  /-- affine maps and constants -/
  affine : ℕ
  /-- roundings' units -/
  rounding : ℕ
  /-- the limb splits' carries (S8-PORT's `carries`: `N − 1` a split) -/
  carries : ℕ
  /-- comparisons' and selections' units -/
  compareSelect : ℕ
  /-- gathers' units: fresh index bits, and one a selection among run-time values -/
  gather : ℕ
  /-- receipt folds' units -/
  fold : ℕ
  /-- neutral gates' units -/
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
def countAt (C : IntDag ι) (s : ℕ → Prop) [DecidablePred s] (p : IntGate ι → ℕ) : ℕ :=
  ((C.filter fun g => decide (s g.step)).map p).sum

/-- The units of `C`'s gates of budgeted class `b` at the steps `s` picks. -/
def unitsAt (C : IntDag ι) (s : ℕ → Prop) [DecidablePred s] (b : Budget) : ℕ :=
  ∑ i ∈ Finset.range C.length, if C[i]?.any fun g => decide (s g.step ∧ g.op.budget = some b) then units C i else 0

/-- The data products of `C`'s gates at the steps `s` picks. -/
def productsAt (C : IntDag ι) (s : ℕ → Prop) [DecidablePred s] : ℕ :=
  ∑ i ∈ Finset.range C.length, if C[i]?.any fun g => decide (s g.step) then dataProducts C i else 0

/-- An s8-family step's MACs, `k`; none for another gate. -/
def IntGate.macs (g : IntGate ι) : ℕ := match g.op with | .mma _ _ k _ => k | _ => 0

/-- The input bits of binding `b`. -/
def IntGate.inputBits (b : Binding) (g : IntGate ι) : ℕ :=
  match g.op with
  | .input b' _ => if b' = b then g.bits else 0
  | _ => 0

/-- **The census** of `C`'s gates at the steps `s` picks. -/
def census (C : IntDag ι) (s : ℕ → Prop) [DecidablePred s] : Census where
  family := countAt C s fun g => match g.op with | .mma .. => 1 | _ => 0
  dotMacs := countAt C s IntGate.macs
  servedMacs := countAt C s fun g => if g.role.isServed then g.macs else 0
  tailMacs := countAt C s fun g => if g.role.isTail then g.macs else 0
  affine := countAt C s fun g => match g.op with | .affine .. | .const _ => 1 | _ => 0
  rounding := unitsAt C s .rounding
  carries := countAt C s fun g => if g.role = .limbSplit true then 1 else 0
  compareSelect := unitsAt C s .compareSelect
  gather := unitsAt C s .gather
  fold := unitsAt C s .fold
  neutral := unitsAt C s .neutral
  committed := countAt C s fun g => if g.commit then 1 else 0
  runBits := countAt C s (IntGate.inputBits .run)
  coinBits := countAt C s (IntGate.inputBits .coin)
  registeredBits := countAt C s (IntGate.inputBits .registered)

/-- The budgeted units of a census: every budgeted class's. -/
def Census.budgeted (c : Census) : ℕ := c.rounding + c.compareSelect + c.gather + c.fold + c.neutral

/-- **The budgeted units** of a circuit: its gates' units. -/
def budgeted (C : IntDag ι) : ℕ := ∑ i ∈ Finset.range C.length, units C i

/-- A circuit's data products. -/
def products (C : IntDag ι) : ℕ := ∑ i ∈ Finset.range C.length, dataProducts C i

/-- **The nonlinear units** of a circuit: its budgeted units and its data products, every operation but a linear map
with coefficients fixed before the coins, what the property's caps bound. -/
def nonlinear (C : IntDag ι) : ℕ := budgeted C + products C

/-- The run inputs' declared bits. -/
def runBits (C : IntDag ι) : ℕ := (C.map (IntGate.inputBits .run)).sum

/-- The property's parameters: each budgeted class's cap in units per certified byte read, a rational as property/v1's
`c` is (an honest circuit's rate is a small fraction of a unit a byte), the data products' cap per certified byte read,
`R_in`'s bound on each window and `L`. -/
structure Caps where
  rounding : ℚ
  compareSelect : ℚ
  gather : ℚ
  fold : ℚ
  /-- the data products (`dataProducts`) per certified byte read: attention's `QKᵀ` and `PV` in an honest circuit -/
  products : ℚ
  /-- `R_in` on window `w`: the run-input bytes its steps may read, one bound a window, since windows differ in length -/
  runBytes : ℕ → ℕ
  /-- `L`: the engine steps a value may be read after the step that produced it -/
  life : ℕ

/-- The nonlinear units (budgeted units and data products) the caps allow per certified byte read. -/
def Caps.total (k : Caps) : ℚ := k.rounding + k.compareSelect + k.gather + k.fold + k.products

/-- The audit's windows: `windows` of them, the window `win t` of each engine step `t`, and the bytes the audit
certified `W` read in each (`reads w`). -/
structure Audit where
  windows : ℕ
  win : ℕ → ℕ
  reads : ℕ → ℕ

/-- **The stacking property `P(c, R_in, L)`** of a declared circuit `W`, on its census and the audit's certified reads:
every gate is in a window; in each window each budgeted class's units are within its cap per certified byte read, no
unit is neutral (the registry classes every primitive), the run inputs are within that window's `R_in` bytes and the
data products (`productsAt`) within their cap per certified byte read; and no gate reads a value produced more than
`L` engine steps before it. -/
def Property (k : Caps) (A : Audit) (W : IntDag ι) : Prop :=
  (∀ g ∈ W, A.win g.step < A.windows) ∧
    (∀ w < A.windows,
      let c := census W (fun t => A.win t = w)
      (c.rounding : ℚ) ≤ k.rounding * A.reads w ∧ (c.compareSelect : ℚ) ≤ k.compareSelect * A.reads w ∧
        (c.gather : ℚ) ≤ k.gather * A.reads w ∧ (c.fold : ℚ) ≤ k.fold * A.reads w ∧
          c.neutral = 0 ∧ c.runBits ≤ 8 * k.runBytes w ∧
          ((productsAt W (fun t => A.win t = w) : ℕ) : ℚ) ≤ k.products * A.reads w) ∧
    ∀ i (h : i < W.length), ∀ j ∈ W[i].args, ∀ g ∈ W[j]?, W[i].step ≤ g.step + k.life

/-- `good` holds more than the fraction `ε = εnum/εden` of `S`: `ε·|S| < |good|`. -/
def MoreThan {κ : Type} (S good : Finset κ) (εnum εden : ℕ) : Prop :=
  εnum * S.card < εden * good.card

end Accounting.Gates
