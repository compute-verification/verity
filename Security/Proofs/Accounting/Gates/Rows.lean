import Definitions.Accounting.Gates.Declared
import Definitions.Accounting.Gates.Registry

/-!
# The circuit model against the running Python, row by row

Rows the kernel evaluates the model on, each set recomputed from the Python by
`verity/ml/circuits/tests/test_lean_gates.py`, and the H1 rows by PoUW's `verity/core/protocols/pouw/tests/test_h1_gates.py`
(the tile is PoUW's, which `verity.ml` doesn't import):

* `s8Rows`: `intervals` over an `S8Dot_v1` coordinate (`s8Dot`, `ops/s8.py`'s `_dot_body`) whose operands and incoming
  accumulator come from producers declaring the binding's `A`, `B`, `A_g`, `B_g` and `C`: the derived intervals of its
  words `T_1 … T_{K/L}` and `D_g`, or none when some step leaves its window (`Proven`'s conditions on an s8-family step),
  against `s8.no_wrap`'s `T_t` and `D_g`, or its refusal.
* `censusRows`: `census` of `S8Dot_v1` coordinates (or one `S8Matmul_v1`) and `LimbSplit_v1` splits over their inputs
  (`censusDag`), against census/v1 of the same Program: the s8 family's steps and MACs, served and tail apart in all and
  per engine step, the constants, the limb splits' carries, the inputs' bits, and property/v1's units of the limb steps.
* `h1Rows`: `census` of an H1 + fx1 tile (`h1Dag`, `h1tile.PouwH1Tile`): its dot steps and `dotMacs` against its
  Program's classes in census/v1, none of its MACs served or tail, and its committed and served words against the
  receipt's layout; and every gate of each row's tile is its primitive's (`Classed`, `classed_h1Rows`).
* `classRows`, `h1ClassRows`, `familyRows`, `tableReadRows`: `classOf` on primitive ids (`classTag`), against
  property/v1's `class_of` with its gathers' kinds and its s8 family's formats, and circuits' input ids.
* `unitsRows`, `tableUnitsRows`: `units` by class on small circuits of primitive ids (`dagOf`: each gate an operation of
  its id's `classOf` class), against property/v1's units of the same Program.
-/

namespace Accounting.SecurityProofs.Gates.Rows

open Accounting.Gates

/-! ## An `S8Dot_v1` coordinate -/

/-- The constant 0 a chain or a fresh tail group starts from (`prims.ZERO32`), at engine step `t`. -/
def zero32 (t : ℕ) : IntGate ℕ := ⟨.const 0, [], 0, 0, t, "Const32[0x00000000]_v1", .other, false⟩

/-- Step `s` of an `S8Dot_v1` body with statics `K`, `L` and the tail's atom counts, at engine step `t`, on the
accumulator at gate `acc` and operands `a`, `b`: `BlackwellU8S8ImmaDot32_v1` (u8·s8, wrapping) when `u8`, else
`BlackwellS8ImmaDot32_v1` (s8·s8, clamped). -/
def s8Step (u8 : Bool) (K L : ℕ) (atoms : List ℕ) (t s acc : ℕ) (a b : List ℕ) (commit : Bool) : IntGate ℕ :=
  ⟨.mma (if u8 then .u8 else .s8) .s8 32 (!u8), acc :: (a ++ b), s32Min, s32Max, t,
    if u8 then "BlackwellU8S8ImmaDot32_v1" else "BlackwellS8ImmaDot32_v1", .dot K L atoms s, commit⟩

/-- `n` chained steps appended to `C`, numbered from `s₀`: the first reads the accumulator at gate `acc`, each later one
the step before it, and step `i` reads operands `a i` and `b i`. -/
def chain (u8 : Bool) (K L : ℕ) (atoms : List ℕ) (t s₀ n acc : ℕ) (a b : ℕ → List ℕ) (commit : ℕ → Bool)
    (C : IntDag ℕ) : IntDag ℕ :=
  C ++ (List.range n).map fun i =>
    s8Step u8 K L atoms t (s₀ + i) (if i = 0 then acc else C.length + i - 1) (a i) (b i) (commit i)

/-- **An `S8Dot_v1` coordinate** at engine step `t` (`ops/s8.py`'s `_dot_body`), appended to `pre`: from the incoming
accumulator at gate `acc`, or from a constant 0 (`T0 = "zero"`), `K/32` steps of `DOT` (u8·s8 when `u8`) over `a s` and
`b s`, `T_t` after every `L` products; then each tail group `(atoms, chain)` in order, `atoms` steps of `TDOT` (u8·s8
when `tu8`) over tail atom `j`'s `ta j` and `tb j`, from `T_{K/L}` (`chain`, the first group only) or a constant 0.
Returns the circuit and its words' gates: `T_1 … T_{K/L}`, then each group's last step `D_g`. -/
def s8Dot (t : ℕ) (u8 tu8 : Bool) (K L : ℕ) (acc : Option ℕ) (tail : List (ℕ × Bool)) (a b ta tb : ℕ → List ℕ)
    (pre : IntDag ℕ) : IntDag ℕ × List ℕ :=
  let atoms := tail.map Prod.fst
  let C₀ := if acc.isSome then pre else pre ++ [zero32 t]
  let C₁ := chain u8 K L atoms t 0 (K / 32) (acc.getD pre.length) a b (fun s => 32 * (s + 1) % L == 0) C₀
  let T := ((List.range (K / 32)).filter fun s => 32 * (s + 1) % L == 0).map fun s => C₀.length + s
  let r := tail.foldl (fun (st : IntDag ℕ × ℕ × List ℕ) g =>
    let C := if g.2 then st.1 else st.1 ++ [zero32 t]
    let j := st.2.1
    (chain tu8 K L atoms t (K / 32 + j) g.1 (if g.2 then C.length - 1 else C.length - 1)
        (fun i => ta (j + i)) (fun i => tb (j + i)) (fun _ => false) C,
      j + g.1, st.2.2 ++ [C.length + g.1 - 1])) (C₁, 0, [])
  (r.1, T ++ r.2.2)

/-! ## `intervals` against `s8.no_wrap` -/

/-- A budgeted gate declaring `iv`: an operand's or an incoming accumulator's producer (a quantizer's clamp). -/
def producer (iv : Iv) : IntGate ℕ := plainGate (.budgeted .rounding fun _ => iv.1) [] iv.1 iv.2 ""

/-- Gate `i` is not an s8-family step, or is one in its window by interval analysis (`Proven`'s conditions on it): its
derived interval within s32 and its operands' derived intervals within their formats. -/
def stepOk (C : IntDag ℕ) (ivs : List Iv) (i : ℕ) : Bool :=
  match C[i]? with
  | none => true
  | some g =>
    match g.op with
    | .mma fa fb k _ =>
        let iv := ivs.getD i (0, 0)
        decide (s32Min ≤ iv.1 ∧ iv.2 ≤ s32Max) &&
          (g.args.tail.take k).all (fun j => let a := ivs.getD j (0, 0); decide (fa.lo ≤ a.1 ∧ a.2 ≤ fa.hi)) &&
          (g.args.tail.drop k).all (fun j => let b := ivs.getD j (0, 0); decide (fb.lo ≤ b.1 ∧ b.2 ≤ fb.hi))
    | _ => true

/-- The derived intervals of the gates `ws` when every s8-family step of `C` is in its window, else none. -/
def wordIntervals (C : IntDag ℕ) (ws : List ℕ) : Option (List Iv) :=
  let ivs := intervals C
  if (List.range C.length).all (stepOk C ivs) then some (ws.map fun j => ivs.getD j (0, 0)) else none

/-- Tail atom `j`'s group, among groups of the given atom counts. -/
def groupOf : List ℕ → ℕ → ℕ
  | [], _ => 0
  | a :: as, j => if j < a then 0 else groupOf as (j - a) + 1

/-- **`no_wrap`'s words, in the model**: an `S8Dot_v1` at `K` and `L` (`DOT` u8·s8 when `u8`) whose operands come from
producers declaring `A` and `B`, from an incoming accumulator declaring `C` or from 0, with tail groups
`(atoms, chain, A_g, B_g)` of `TDOT` (u8·s8 when `tu8`): the derived intervals of `T_1 … T_{K/L}` and each `D_g`, or
none when some step leaves its window. -/
def s8Interval (K L : ℕ) (C : Option Iv) (u8 : Bool) (A B : Iv) (tu8 : Bool) (tail : List (ℕ × Bool × Iv × Iv)) :
    Option (List Iv) :=
  let pre : IntDag ℕ := [producer A, producer B] ++ C.toList.map producer ++
    tail.flatMap fun g => [producer g.2.2.1, producer g.2.2.2]
  let base := if C.isSome then 3 else 2
  let atoms := tail.map Prod.fst
  let r := s8Dot 0 u8 tu8 K L (if C.isSome then some 2 else none) (tail.map fun g => (g.1, g.2.1))
    (fun _ => List.replicate 32 0) (fun _ => List.replicate 32 1)
    (fun j => List.replicate 32 (base + 2 * groupOf atoms j))
    (fun j => List.replicate 32 (base + 2 * groupOf atoms j + 1)) pre
  wordIntervals r.1 r.2

/-- Rows `(K, L, C, u8, A, B, tu8, tail, words)`: `s8Interval` is `s8.no_wrap`'s `T_t` and `D_g` intervals in order,
or none where it refuses the binding. They take the plain and the edited arms, an incoming accumulator, u8 operands in
the chain and in a tail, a chained and a fresh tail group, the s32 bound met exactly and missed by one, and refusals in
the chain, in a tail and of an operand outside its format. -/
def s8Rows : List (ℕ × ℕ × Option Iv × Bool × Iv × Iv × Bool × List (ℕ × Bool × Iv × Iv) × Option (List Iv)) := [
  (64, 64, none, false, (-127, 127), (-127, 127), false, [], some [(-1032256, 1032256)]),
  (64, 32, none, false, (-127, 127), (-127, 127), false, [(2, false, (-127, 127), (-64, 64)), (1, false, (-128, 127), (-64, 64))], some [(-516128, 516128), (-1032256, 1032256), (-520192, 520192), (-262144, 262144)]),
  (96, 96, some (-1073741824, 1073741824), true, (0, 255), (-128, 127), false, [], some [(-1076875264, 1076850784)]),
  (128, 32, none, false, (-127, 127), (-127, 127), false, [(2, true, (-128, 127), (-128, 127)), (1, false, (0, 3), (-5, 5))], some [(-516128, 516128), (-1032256, 1032256), (-1548384, 1548384), (-2064512, 2064512), (-3104896, 3113088), (-480, 480)]),
  (32, 32, none, false, (-127, 127), (-127, 127), true, [(1, false, (0, 255), (-128, 127))], some [(-516128, 516128), (-1044480, 1036320)]),
  (96, 32, none, true, (0, 255), (-128, 127), false, [], some [(-1044480, 1036320), (-2088960, 2072640), (-3133440, 3108960)]),
  (64, 64, none, false, (-128, -1), (-128, -1), false, [], some [(64, 1048576)]),
  (32, 32, some (-2146959360, 2146959359), false, (-128, 127), (-128, 127), false, [], some [(-2147479552, 2147483647)]),
  (32, 32, some (-2146959360, 2146959360), false, (-128, 127), (-128, 127), false, [], none),
  (32, 32, some (2147483547, 2147483647), false, (-127, 127), (-127, 127), false, [], none),
  (32, 32, some (0, 2146883647), false, (-127, 127), (-127, 127), false, [(1, true, (-127, 127), (-127, 127))], none),
  (32, 32, none, true, (-1, 255), (-128, 127), false, [], none),
  (32, 32, none, false, (-127, 127), (-127, 127), false, [(1, false, (-127, 127), (-129, 0))], none)]

/-- `intervals` is `s8.no_wrap` on every row. -/
theorem s8Interval_s8Rows :
    s8Rows.all (fun (K, L, C, u8, A, B, tu8, tail, out) => s8Interval K L C u8 A B tu8 tail == out) := by
  decide +kernel

/-! ## `census` against census/v1 -/

/-- An input's id at the widths the rows take: `RegisteredInput{w}_v1` when registered, else `Input{w}_v1` (a coin's
binding is its port's). -/
def inputId : Binding → ℕ → String
  | .registered, 8 => "RegisteredInput8_v1"
  | .registered, 32 => "RegisteredInput32_v1"
  | _, 8 => "Input8_v1"
  | _, 32 => "Input32_v1"
  | _, _ => ""

/-- An input gate of slot `i`, bound at `b`, declared at a signed `w`-bit word's whole range. -/
def inputGate (b : Binding) (w i : ℕ) : IntGate ℕ :=
  ⟨.input b i, [], -2 ^ (w - 1), 2 ^ (w - 1) - 1, 0, inputId b w, .other, false⟩

/-- A limb step of primitive `prim` (class `cls`) reading gate `y`, at engine step `t`. -/
def limbGate (t : ℕ) (prim : String) (cls : Budget) (y : ℕ) (lo hi : ℤ) : IntGate ℕ :=
  ⟨.budgeted cls fun v => v.headD 0, [y], lo, hi, t, prim, .other, false⟩

/-- `LimbSplit_v1{FORM = "bytes"}`'s steps from the word at gate `r` with `n` carries to go, appended to `C`: its low
byte and its carry, and from the carry on; the last limb a low byte. -/
def limbBytes (t : ℕ) : ℕ → ℕ → IntDag ℕ → IntDag ℕ
  | 0, r, C => C ++ [limbGate t "S32LowByte_v1" .rounding r (-128) 127]
  | n + 1, r, C =>
      limbBytes t n (C.length + 1)
        (C ++ [limbGate t "S32LowByte_v1" .rounding r (-128) 127, limbGate t "S32ByteCarry_v1" .rounding r s32Min s32Max])

/-- **`LimbSplit_v1{FORM, N}`** of the word at gate `y`, at engine step `t`, appended to `C`: `"byteclamp"` (`clamp`,
`N = 3`) is `h = S32ByteCarry(y)`, `S32LowByte(y)`, `S32ClampS8(h)` and `S32ClampRest(h)`, and `"bytes"` `limbBytes`. -/
def limbSplit (t : ℕ) (clamp : Bool) (N y : ℕ) (C : IntDag ℕ) : IntDag ℕ :=
  if clamp then
    C ++ [limbGate t "S32ByteCarry_v1" .rounding y s32Min s32Max, limbGate t "S32LowByte_v1" .rounding y (-128) 127,
      limbGate t "S32ClampS8_v1" .compareSelect C.length (-128) 127,
      limbGate t "S32ClampRest_v1" .rounding C.length (-128) 127]
  else limbBytes t (N - 1) y C

/-- **A census row's circuit**: the inputs, then `N` `S8Dot_v1` coordinates (each its own engine step) or one
`S8Matmul_v1` of `N` rows (engine step 0), then a `LimbSplit_v1` `(clamp, N)` of each `limbs` word (the next engine
step). The inputs are, in order: the incoming accumulators (`acc`: `N` s32 words), `a` (`K` s8 codes), `b` (`N` rows of
`K`, registered when `regB`), `ta` (32 codes a tail atom), `tb` (`N` rows of those, registered when `regB`) and the limb
splits' words (s32), every other one a run input. -/
def censusDag (matmul : Bool) (N K L : ℕ) (acc u8 tu8 : Bool) (tail : List (ℕ × Bool)) (regB : Bool)
    (limbs : List (Bool × ℕ)) : IntDag ℕ :=
  let m := 32 * (tail.map Prod.fst).sum
  let rb : Binding := if regB then .registered else .run
  let nAcc := if acc then N else 0
  let aB := nAcc
  let bB := aB + K
  let taB := bB + N * K
  let tbB := taB + m
  let yB := tbB + N * m
  let ins : IntDag ℕ :=
    (List.range nAcc).map (fun i => inputGate .run 32 i) ++
      (List.range K).map (fun i => inputGate .run 8 (aB + i)) ++
      (List.range (N * K)).map (fun i => inputGate rb 8 (bB + i)) ++
      (List.range m).map (fun i => inputGate .run 8 (taB + i)) ++
      (List.range (N * m)).map (fun i => inputGate rb 8 (tbB + i)) ++
      (List.range limbs.length).map (fun i => inputGate .run 32 (yB + i))
  let dots := (List.range N).foldl (fun C n =>
    (s8Dot (if matmul then 0 else n) u8 tu8 K L (if acc then some n else none) tail
      (fun s => (List.range 32).map fun q => aB + 32 * s + q)
      (fun s => (List.range 32).map fun q => bB + n * K + 32 * s + q)
      (fun j => (List.range 32).map fun q => taB + 32 * j + q)
      (fun j => (List.range 32).map fun q => tbB + n * m + 32 * j + q) C).1) ins
  limbs.zipIdx.foldl (fun C e => limbSplit (if matmul then 1 else N) e.1.1 e.1.2 (yB + e.2) C) dots

/-- A census row's counts: the s8 family's steps, their MACs, served and tail MACs, constants and affine maps, carries,
roundings' and compare-selects' units, run and registered input bits, and each engine step's served and tail MACs. -/
abbrev CensusOut := ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × List (ℕ × ℕ)

/-- `census` of `censusDag`, read as census/v1 and property/v1 report it. -/
def censusOf (matmul : Bool) (N K L : ℕ) (acc u8 tu8 : Bool) (tail : List (ℕ × Bool)) (regB : Bool)
    (limbs : List (Bool × ℕ)) : CensusOut :=
  let C := censusDag matmul N K L acc u8 tu8 tail regB limbs
  let c := census C fun _ => True
  let steps := (if matmul then 1 else N) + if limbs.isEmpty then 0 else 1
  (c.family, c.dotMacs, c.servedMacs, c.tailMacs, c.affine, c.carries, c.rounding, c.compareSelect, c.runBits,
    c.registeredBits, (List.range steps).map fun s => let d := census C (· = s); (d.servedMacs, d.tailMacs))

/-- Rows `(matmul, N, K, L, acc, u8, tu8, tail, regB, limbs, counts)`: `censusOf` is census/v1's count of the same
Program (`test_lean_gates.census_program`) and property/v1's units of its limb steps. They take the plain arm, an
incoming accumulator, chained and fresh tail groups on two coordinates, an `S8Matmul_v1` of u8 steps with registered
rows, and both forms of limb split. -/
def censusRows : List (Bool × ℕ × ℕ × ℕ × Bool × Bool × Bool × List (ℕ × Bool) × Bool × List (Bool × ℕ) × CensusOut) := [
  (false, 1, 64, 64, false, false, false, [], true, [], (2, 64, 64, 0, 1, 0, 0, 0, 512, 512, [(64, 0)])),
  (false, 2, 32, 32, true, false, false, [(1, true), (1, false)], false, [(false, 2)], (6, 192, 64, 128, 2, 1, 3, 0, 2400, 0, [(32, 64), (32, 64), (0, 0)])),
  (true, 2, 32, 32, false, true, true, [(2, false)], true, [(true, 3), (false, 3)], (6, 192, 64, 128, 4, 4, 8, 1, 832, 1536, [(64, 128), (0, 0)])),
  (false, 1, 32, 32, false, false, false, [], false, [(false, 4)], (1, 32, 32, 0, 1, 3, 7, 0, 544, 0, [(32, 0), (0, 0)]))]

/-- `census` is census/v1 (and its units property/v1's) on every row. -/
theorem censusOf_censusRows :
    censusRows.all (fun (matmul, N, K, L, acc, u8, tu8, tail, regB, limbs, out) =>
      censusOf matmul N K L acc u8 tu8 tail regB limbs == out) := by
  decide +kernel

/-! ## `census` of an H1 + fx1 tile -/

/-- H1's digit count `m` at `KS` segments (`segment.n_digits`): the least `m` with `128·KS ≤ 128^m` and
`⌊127·KS / 128^(m−1)⌋ ≤ 127`. -/
def nDigits (ks : ℕ) : ℕ :=
  ((List.range (ks + 1)).find? fun m =>
    decide (128 * ks ≤ 128 ^ (m + 1)) && decide (127 * ks / 128 ^ m ≤ 127)).getD 0 + 1

/-- Entry `q`'s row slot in its lane (`segment.row_slot`). -/
def rowSlot (q : ℕ) : ℕ := 2 * (q / 4 / 4) + q % 4 / 2

/-- Entry `q`'s column slot in its lane (`segment.col_slot`). -/
def colSlot (q : ℕ) : ℕ := 2 * (q / 4 % 4) + q % 4 % 2

/-- Lane `ln`'s row at row slot `s` of an `r × c` tile (`h1tile.Layout.rows`). -/
def laneRow (r c ln s : ℕ) : ℕ := ln / (c / 8) + r / 4 * s

/-- Lane `ln`'s column at column slot `s` of a tile of `c` columns (`h1tile.Layout.cols`). -/
def laneCol (c ln s : ℕ) : ℕ := 2 * (ln % (c / 8)) + 2 * (c / 8) * (s / 2) + s % 2

/-- A budgeted gate of class `cls` and primitive `prim` reading `args`, at engine step 0. -/
def h1Gate (prim : String) (cls : Budget) (args : List ℕ) (place : Place := .other) (commit : Bool := false) : IntGate ℕ :=
  ⟨.budgeted cls fun v => v.headD 0, args, s32Min, s32Max, 0, prim, place, commit⟩

/-- A lane's segment dot of primitive `prim` (`PouwSegDot_v2` from the accumulator at gate `acc`, or `PouwSegDot0_v2`
from the zero there): `acc + Σ a_k·b_k` over 32 s8 pairs, wrapping. -/
def segDot (prim : String) (acc : ℕ) (a b : List ℕ) : IntGate ℕ :=
  ⟨.mma .s8 .s8 32 false, acc :: (a ++ b), s32Min, s32Max, 0, prim, .other, false⟩

/-- A step of a cell's `z`: `BlackwellS8ImmaDot32_v2` (s8·s8, clamped) on the accumulator at gate `acc`. -/
def immaStep (acc : ℕ) (a b : List ℕ) : IntGate ℕ :=
  ⟨.mma .s8 .s8 32 true, acc :: (a ++ b), s32Min, s32Max, 0, "BlackwellS8ImmaDot32_v2", .other, false⟩

/-- **An H1 + fx1 tile's circuit** (`h1tile.PouwH1Tile` with `EXACT` the s8 accumulator `GemmAccumulatorS8_v2{K}` over
`BlackwellS8ImmaDot32_v2`, on the full `r × c` grid), at engine step 0: `A`'s `r` rows of `K` codes (run inputs), `B`'s
`c` rows (registered), the salt (a coin) and a zero; `X = A + b` (`r·K`), `U` (32 a column), `−U`, `W′ = B + c + U`
(`c·K`) and each row's `m` digit words `A_d` (32 a word); then each lane's dots set by set (`T_0 … T_{KS−1}`,
`O′ = T_{KS−1} − U·A_0` and `D_d = U·A_d`, one a lane entry, 32 a set), and per set its fx1 fold and its two halves'
`PouwXinFinal`; the receipt's `2·(KS + m)` words, each the XOR over lanes of one half after one set, at its receipt place
and committed; and `z`, each cell's `K/32` steps from the zero. Its noise and fold keys read the salt, and its digits read
`X` directly: the counts it is held to read neither. -/
def h1Dag (r c K : ℕ) : IntDag ℕ :=
  let ks := K / 32
  let m := nDigits ks
  let sets := ks + m
  let lanes := r / 4 * (c / 8)
  let bB := r * K
  let sB := bB + c * K
  let zB := sB + 1
  let xB := zB + 1
  let uB := xB + r * K
  let nB := uB + 32 * c
  let wB := nB + 32 * c
  let dB := wB + c * K
  let lB := dB + r * m * 32
  let per := 32 * sets + 3 * sets
  let zzB := lB + lanes * per + 2 * sets
  let row := fun (ln e : ℕ) => laneRow r c ln (rowSlot e)
  let col := fun (ln e : ℕ) => laneCol c ln (colSlot e)
  let span := fun (base n : ℕ) => (List.range n).map (base + ·)
  let dotOf := fun (ln s e : ℕ) =>
    if s = 0 then segDot "PouwSegDot0_v2" zB (span (xB + row ln e * K) 32) (span (wB + col ln e * K) 32)
    else if s < ks then
      segDot "PouwSegDot_v2" (lB + ln * per + 32 * (s - 1) + e) (span (xB + row ln e * K + 32 * s) 32)
        (span (wB + col ln e * K + 32 * s) 32)
    else if s = ks then
      segDot "PouwSegDot_v2" (lB + ln * per + 32 * (s - 1) + e) (span (nB + 32 * col ln e) 32)
        (span (dB + 32 * (row ln e * m)) 32)
    else segDot "PouwSegDot0_v2" zB (span (uB + 32 * col ln e) 32) (span (dB + 32 * (row ln e * m + (s - ks))) 32)
  let lane := fun (ln : ℕ) =>
    (List.range (32 * sets)).map (fun t => dotOf ln (t / 32) (t % 32)) ++
      (List.range sets).flatMap fun s =>
        let f := lB + ln * per + 32 * sets + 3 * s
        [h1Gate "PouwFx1Absorb_v2" .fold ((if s = 0 then sB else f - 3) :: span (lB + ln * per + 32 * s) 32),
          h1Gate "PouwXinFinal_v2" .rounding [f], h1Gate "PouwXinFinal_v2" .rounding [f]]
  (List.range (r * K)).map (fun i => inputGate .run 8 i) ++
    (List.range (c * K)).map (fun i => inputGate .registered 8 (bB + i)) ++
    [inputGate .coin 32 sB, zero32 0] ++
    (List.range (r * K)).map (fun i => h1Gate "PouwI8AddBit_v2" .rounding [i, sB]) ++
    (List.range (32 * c)).map (fun _ => h1Gate "PouwU129_v2" .rounding [sB]) ++
    (List.range (32 * c)).map (fun i => h1Gate "PouwI8Neg_v2" .rounding [uB + i]) ++
    (List.range (c * K)).map (fun i => h1Gate "PouwI8Add_v2" .rounding [bB + i, sB, uB + 32 * (i / K) + i % K % 32]) ++
    (List.range (r * m * 32)).map (fun i => h1Gate "PouwDigitLo_v2" .rounding [xB + K * (i / (32 * m)) + i % 32]) ++
    (List.range lanes).flatMap lane ++
    (List.range (2 * sets)).map (fun w =>
      h1Gate "PouwXor32_v2" .rounding
        ((List.range lanes).map fun ln => lB + ln * per + 32 * sets + 3 * (w / 2) + 1 + w % 2)
        (.receipt "GemmAccumulatorS8_v2" K w (span 0 (r * K)) (span bB (c * K))) true) ++
    (List.range (r * c * ks)).map fun t =>
      immaStep (if t % ks = 0 then zB else zzB + t - 1) (span (t / ks / c * K + 32 * (t % ks)) 32)
        (span (bB + t / ks % c * K + 32 * (t % ks)) 32)

/-- An H1 row's counts: the s8 family's steps, their MACs (`dotMacs`), the served and the tail MACs, the committed
words, and those of them whose role is served. -/
abbrev H1Out := ℕ × ℕ × ℕ × ℕ × ℕ × ℕ

/-- `census` of `h1Dag`, with its committed words of the served role. -/
def h1Of (r c K : ℕ) : H1Out :=
  let C := h1Dag r c K
  let d := census C fun _ => True
  (d.family, d.dotMacs, d.servedMacs, d.tailMacs, d.committed,
    countAt C (fun _ => True) fun g => if g.commit && g.role.isServed then 1 else 0)

/-- Rows `(r, c, K, counts)`: `h1Of` is the H1 tile's Program as PoUW's `test_h1_gates.py` counts it. Its dot steps and
`dotMacs` are its classes' instances in census/v1 (the lanes' `PouwSegDot_v2` and `PouwSegDot0_v2` and `z`'s
`BlackwellS8ImmaDot32_v2`, 32 MACs each); it has no served and no tail MACs, and its committed and served words are the
receipt's `2·(KS + m)` and `2·KS`. They take one lane, two lanes (a receipt word an XOR of two) and `KS = 3`. -/
def h1Rows : List (ℕ × ℕ × ℕ × H1Out) := [
  (4, 8, 64, (192, 6144, 0, 0, 8, 4)),
  (4, 16, 64, (384, 12288, 0, 0, 8, 4)),
  (4, 8, 96, (256, 8192, 0, 0, 10, 6))]

/-- `census` of an H1 + fx1 tile is its Program's count on every row. -/
theorem h1Of_h1Rows : h1Rows.all (fun (r, c, K, out) => h1Of r c K == out) := by
  decide +kernel

/-! ## Every gate of an H1 + fx1 tile is its primitive's -/

/-- A gate at a dot step is an s8-family step. -/
def mmaAtDot (g : IntGate ℕ) : Bool :=
  match g.place, g.op with
  | .dot .., .mma .. => true
  | .dot .., _ => false
  | _, _ => true

/-- `Classed` through a list `S` of primitives and classes, as the kernel decides it: each of `S` is listed in its class
(`Class.listed`), and each gate's primitive and class is one of `S`, an s8-family step when it is at a dot step. -/
def classedIn (S : List (String × Class)) (C : IntDag ℕ) : Bool :=
  (S.all fun p => classOf p.1 == some p.2.listed) && C.all fun g => decide ((g.prim, g.op.cls) ∈ S) && mmaAtDot g

theorem classed_of_classedIn {S : List (String × Class)} {C : IntDag ℕ} (h : classedIn S C = true) : Classed C := by
  simp only [classedIn, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  intro g hg
  obtain ⟨hmem, hm⟩ := h.2 g hg
  refine ⟨by simpa using h.1 _ hmem, fun K L tail s hs => ?_⟩
  unfold mmaAtDot at hm
  rw [hs] at hm
  cases hop : g.op <;> simp_all [IntOp.IsMma]

/-- An H1 + fx1 tile's primitives with their gates' classes (`h1Dag`): its inputs (the salt a coin, by its port), its
zero, its noise and digit words, its lanes' segment dots, folds and finals, its receipt words and `z`'s IMMA steps. -/
def h1Classes : List (String × Class) := [
  ("Input8_v1", .input .run),
  ("RegisteredInput8_v1", .input .registered),
  ("Input32_v1", .input .coin),
  ("Const32[0x00000000]_v1", .const),
  ("PouwI8AddBit_v2", .budgeted .rounding),
  ("PouwU129_v2", .budgeted .rounding),
  ("PouwI8Neg_v2", .budgeted .rounding),
  ("PouwI8Add_v2", .budgeted .rounding),
  ("PouwDigitLo_v2", .budgeted .rounding),
  ("PouwSegDot0_v2", .family .s8 .s8 32 false),
  ("PouwSegDot_v2", .family .s8 .s8 32 false),
  ("PouwFx1Absorb_v2", .budgeted .fold),
  ("PouwXinFinal_v2", .budgeted .rounding),
  ("PouwXor32_v2", .budgeted .rounding),
  ("BlackwellS8ImmaDot32_v2", .family .s8 .s8 32 true)]

/-- **Every gate of each H1 row's tile is its primitive's** (`classOf` classes it, in its operation's class). -/
theorem classed_h1Rows : ∀ e ∈ h1Rows, Classed (h1Dag e.1 e.2.1 e.2.2.1) := by
  have h : h1Rows.all (fun e => classedIn h1Classes (h1Dag e.1 e.2.1 e.2.2.1)) = true := by decide +kernel
  exact fun e he => classed_of_classedIn (List.all_eq_true.1 h e he)

/-! ## `classOf` against property/v1's table -/

/-- A class as property/v1 names it: an input's binding, `const`, `affine`, an s8-family step's formats with its `k`,
a gather's index-bit kind (with its `n` for a table read), a budgeted class, or `unlisted`. -/
def classTag : Option Class → String × ℕ
  | none => ("unlisted", 0)
  | some (.input .run) => ("run", 0)
  | some (.input .registered) => ("registration", 0)
  | some (.input .coin) => ("coin", 0)
  | some .const => ("const", 0)
  | some .affine => ("affine", 0)
  | some (.family .s8 .s8 k _) => ("family s8", k)
  | some (.family .u8 .s8 k _) => ("family u8", k)
  | some (.family _ _ k _) => ("family", k)
  | some (.gather (some n) _) => ("low", n)
  | some (.gather none _) => ("range", 0)
  | some (.budgeted .rounding) => ("roundings", 0)
  | some (.budgeted .compareSelect) => ("compare_select", 0)
  | some (.budgeted .gather) => ("gathers", 0)
  | some (.budgeted .fold) => ("folds", 0)
  | some (.budgeted .neutral) => ("neutral", 0)

/-- Rows `(id, class, k)` of registry ids, each class at least once. -/
def classRows : List (String × String × ℕ) := [
  ("BlackwellS8ImmaDot32_v1", "family s8", 32),
  ("BlackwellU8S8ImmaDot32_v2", "family u8", 32),
  ("PouwSegDot_v1", "family s8", 32),
  ("PouwDot16U8S8_v1", "family u8", 16),
  ("I32Add_v1", "affine", 0),
  ("I32Sub_v1", "affine", 0),
  ("Bf16ToF32_v1", "affine", 0),
  ("Const32[0x00000000]_v1", "const", 0),
  ("Const1[0x1]_v1", "const", 0),
  ("I32Mul_v1", "roundings", 0),
  ("F32Add_v1", "roundings", 0),
  ("E4m3ToF32_v1", "roundings", 0),
  ("NcpDot16I7_v1", "roundings", 0),
  ("BlackwellE4m3QmmaDot32_v1", "roundings", 0),
  ("PouwWindowWord_v1", "roundings", 0),
  ("S32LowByte_v1", "roundings", 0),
  ("S32ByteCarry_v1", "roundings", 0),
  ("S32ClampRest_v2", "roundings", 0),
  ("S32ClampS8_v1", "compare_select", 0),
  ("F32Sat_v1", "compare_select", 0),
  ("And2_v1", "compare_select", 0),
  ("I32Le_v1", "compare_select", 0),
  ("SelectI32_v1", "compare_select", 0),
  ("PouwSel32_v1", "compare_select", 0),
  ("PouwDuoAbsorb_v1", "folds", 0)]

/-- Rows `(id, class, k)` of the H1 + fx1 tile's primitives, each as its word Definition and on bits: the segment dots
s8-family steps, the fold's steps folds, the wrap and digit tests and v4's dead-tile mask compare-selects, and the rest
(J10-R's `PouwCallSeeds` and `PouwAdd64` among them) roundings. -/
def h1ClassRows : List (String × String × ℕ) := [
  ("PouwSegDot_v2", "family s8", 32),
  ("PouwSegDot0_v1", "family s8", 32),
  ("PouwSegDot0_v2", "family s8", 32),
  ("PouwFx1Absorb_v1", "folds", 0),
  ("PouwFx1Absorb_v2", "folds", 0),
  ("PouwXinFinal_v1", "roundings", 0),
  ("PouwXinFinal_v2", "roundings", 0),
  ("PouwXor32_v2", "roundings", 0),
  ("PouwU129_v1", "roundings", 0),
  ("PouwU129_v2", "roundings", 0),
  ("PouwI8AddBit_v1", "roundings", 0),
  ("PouwI8AddBit_v2", "roundings", 0),
  ("PouwI8Add_v1", "roundings", 0),
  ("PouwI8Add_v2", "roundings", 0),
  ("PouwI8Neg_v1", "roundings", 0),
  ("PouwI8Neg_v2", "roundings", 0),
  ("PouwUnitSeeds_v1", "roundings", 0),
  ("PouwUnitSeeds_v2", "roundings", 0),
  ("PouwNoiseBlock_v1", "roundings", 0),
  ("PouwNoiseBlock_v2", "roundings", 0),
  ("PouwDigitLo_v1", "roundings", 0),
  ("PouwDigitLo_v2", "roundings", 0),
  ("PouwDigitHi_v1", "roundings", 0),
  ("PouwDigitHi_v2", "roundings", 0),
  ("PouwDigitZ_v1", "compare_select", 0),
  ("PouwDigitZ_v2", "compare_select", 0),
  ("PouwLow8_v1", "roundings", 0),
  ("PouwLow8_v2", "roundings", 0),
  ("PouwSx8_v1", "roundings", 0),
  ("PouwSx8_v2", "roundings", 0),
  ("PouwResAdd_v1", "roundings", 0),
  ("PouwResAdd_v2", "roundings", 0),
  ("PouwOneStream2_v1", "roundings", 0),
  ("PouwOneStream2_v2", "roundings", 0),
  ("PouwOneStream3_v1", "roundings", 0),
  ("PouwOneStream3_v2", "roundings", 0),
  ("PouwWrapBits_v1", "compare_select", 0),
  ("PouwWrapBits_v2", "compare_select", 0),
  ("PouwWrapSeg_v1", "roundings", 0),
  ("PouwWrapSeg_v2", "roundings", 0),
  ("PouwXWrap_v1", "compare_select", 0),
  ("PouwXWrap_v2", "compare_select", 0),
  ("PouwPeelSeg_v1", "roundings", 0),
  ("PouwPeelSeg_v2", "roundings", 0),
  ("PouwOrBit_v1", "compare_select", 0),
  ("PouwOrBit_v2", "compare_select", 0),
  ("PouwMask32_v1", "compare_select", 0),
  ("PouwMask32_v2", "compare_select", 0),
  ("PouwCallSeeds_v1", "roundings", 0),
  ("PouwCallSeeds_v2", "roundings", 0),
  ("PouwAdd64_v1", "roundings", 0),
  ("PouwAdd64_v2", "roundings", 0),
  ("GemmAccumulatorS8_v2", "unlisted", 0)]

/-- Rows `(id, class, k)` of ids `classOf` classes by their family's full pattern, and ids it refuses: another
version, a numeral with a leading `0`, the numeral `0` or none, and a gather with no column size. -/
def familyRows : List (String × String × ℕ) := [
  ("Input32_v1", "run", 0),
  ("Input8_v1", "run", 0),
  ("RegisteredInput8_v1", "registration", 0),
  ("GatherBf16x16_v1", "range", 0),
  ("GatherF32x128_v1", "range", 0),
  ("GatherI32x64_v1", "range", 0),
  ("GatherE4m3x16_v1", "range", 0),
  ("GatherE2m1x16_v1", "range", 0),
  ("GatherE8m0x16_v1", "range", 0),
  ("BitAtx32_v1", "roundings", 0),
  ("TopPMaskWordx151936_v1", "compare_select", 0),
  ("NoSuchPrimitive_v1", "unlisted", 0),
  ("Zero_v1", "unlisted", 0),
  ("I32Add_v2", "unlisted", 0),
  ("GatherBf16x16_v2", "unlisted", 0),
  ("Input8_v2", "unlisted", 0),
  ("RegisteredInput8_v2", "unlisted", 0),
  ("BitAtx32_v2", "unlisted", 0),
  ("GatherF32xQ_v1", "unlisted", 0),
  ("Input08_v1", "unlisted", 0),
  ("Input_v1", "unlisted", 0),
  ("GatherI32x064_v1", "unlisted", 0),
  ("BitAtx0_v1", "unlisted", 0),
  ("TopPMaskWordx_v1", "unlisted", 0)]

/-- Rows `(id, class, k)` of table reads: `n` index bits, and refused: a table read of no bits, another version, a
numeral with a leading `0`, and no width. -/
def tableReadRows : List (String × String × ℕ) := [
  ("TableRead16x4_v1", "low", 16),
  ("TableRead4x16_v1", "low", 4),
  ("TableRead8x32_v1", "low", 8),
  ("TableRead0x4_v1", "unlisted", 0),
  ("TableRead16x4_v2", "unlisted", 0),
  ("TableRead016x4_v1", "unlisted", 0),
  ("TableRead16x04_v1", "unlisted", 0),
  ("TableRead16_v1", "unlisted", 0)]

/-- `classOf` is property/v1's `class_of` on every registry row. -/
theorem classOf_classRows : classRows.all (fun (id, tag, k) => classTag (classOf id) == (tag, k)) := by
  decide +kernel

/-- `classOf` on every H1 row. -/
theorem classOf_h1ClassRows : h1ClassRows.all (fun (id, tag, k) => classTag (classOf id) == (tag, k)) := by
  decide +kernel

/-- `classOf` on every family row. -/
theorem classOf_familyRows : familyRows.all (fun (id, tag, k) => classTag (classOf id) == (tag, k)) := by
  decide +kernel

/-- `classOf` on every table-read row. -/
theorem classOf_tableReadRows : tableReadRows.all (fun (id, tag, k) => classTag (classOf id) == (tag, k)) := by
  decide +kernel

/-! ## `units` against property/v1 -/

/-- The operation of `cl id`'s class at gate `i` (an input reads slot `i`): `units` reads only the class, a gather's
index-bit kind and its operands, so an affine map is the empty sum and a budgeted primitive the constant 0. -/
def opOf (cl : String → Option Class) (id : String) (i : ℕ) : Option (IntOp ℕ) :=
  (cl id).map fun
    | .input b => .input b i
    | .const => .const 0
    | .affine => .affine 0 []
    | .family a b k sat => .mma a b k sat
    | .gather low fill => .gather low fill
    | .budgeted cls => .budgeted cls fun _ => 0

/-- The circuit of gates `(id, operands)`, each the operation `opOf cl` gives it; none when an id is unlisted. -/
def dagOf (cl : String → Option Class) (gs : List (String × List ℕ)) : Option (IntDag ℕ) :=
  gs.zipIdx.mapM fun e => (opOf cl e.1.1 e.2).map fun op => ⟨op, e.1.2, s32Min, s32Max, 0, e.1.1, .other, false⟩

/-- A circuit's units by budgeted class, its ids classed by `cl`: roundings, compare-selects, gathers and folds. -/
def unitsOf (cl : String → Option Class) (gs : List (String × List ℕ)) : Option (ℕ × ℕ × ℕ × ℕ) :=
  (dagOf cl gs).map fun C => let c := census C fun _ => True; (c.rounding, c.compareSelect, c.gather, c.fold)

/-- Rows `(gates, units)`: `unitsOf` is property/v1's units of the same Program (`test_lean_gates.units_program`), or
none where it refuses an unlisted primitive. They take budgeted gates of run, registered and constant operands; gathers
that share an index (fresh index bits, a column of run-time words, a narrower and then a wider column at one index); a
gather at a constant or registered index; the free s8 steps, sum and widening; and field extractions and the top-p mask. -/
def unitsRows : List (List (String × List ℕ) × Option (ℕ × ℕ × ℕ × ℕ)) := [
  ([("Input32_v1", []), ("RegisteredInput32_v1", []), ("F32Add_v1", [0, 1]), ("F32Add_v1", [1, 1]), ("I32Le_v1", [0, 1]),
    ("SelectI32_v1", [4, 1, 1]), ("PouwDuoAbsorb_v1", [5, 1]), ("Const32[0x00000000]_v1", []), ("F32Add_v1", [7, 1])],
    some (1, 2, 0, 1)),
  ([("Input32_v1", []), ("Input32_v1", []), ("Input32_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("Input16_v1", []),
    ("GatherBf16x16_v1", [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    ("GatherBf16x16_v1", [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    ("GatherBf16x16_v1", [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    ("GatherBf16x16_v1", [0, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    ("GatherBf16x1_v1", [1, 19]),
    ("GatherBf16x1_v1", [2, 3]),
    ("GatherBf16x16_v1", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])],
    some (0, 0, 17, 0)),
  ([("Const32[0x00000000]_v1", []),
    ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []),
    ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []),
    ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []), ("Input16_v1", []),
    ("RegisteredInput32_v1", []),
    ("GatherBf16x16_v1", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    ("GatherBf16x16_v1", [17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    ("Bf16ToF32_v1", [18]), ("F32Add_v1", [20, 20])],
    some (1, 0, 0, 0)),
  ([("Const32[0x00000000]_v1", []),
    ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []),
    ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []),
    ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []),
    ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []),
    ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []), ("Input8_v1", []),
    ("Input8_v1", []), ("Input8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []), ("RegisteredInput8_v1", []),
    ("BlackwellS8ImmaDot32_v1", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]),
    ("BlackwellS8ImmaDot32_v1", [65, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]),
    ("I32Add_v1", [65, 66]), ("Input16_v1", []), ("Bf16ToF32_v1", [68]), ("F32Add_v1", [67, 69])],
    some (1, 0, 0, 0)),
  ([("Input32_v1", []), ("Input32_v1", []), ("BitAtx32_v1", [0, 1]), ("Const32[0x00000000]_v1", []), ("BitAtx32_v1", [0, 3]),
    ("RegisteredInput32_v1", []), ("BitAtx32_v1", [5, 3]), ("TopPMaskWordx2_v1", [0, 1])],
    some (2, 1, 0, 0)),
  ([("Input32_v1", []), ("Zero_v1", [0])], none)]

/-- Rows `(gates, units)` with table reads: a table read and a gather at one index pay apart, each kind's bits once. -/
def tableUnitsRows : List (List (String × List ℕ) × Option (ℕ × ℕ × ℕ × ℕ)) := [
  ([("Input32_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []), ("RegisteredInput16_v1", []),
    ("TableRead4x16_v1", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    ("GatherBf16x16_v1", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    ("TableRead4x16_v1", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])],
    some (0, 0, 9, 0))]

/-- `units` is property/v1's on every row. -/
theorem unitsOf_unitsRows : unitsRows.all (fun (gs, out) => unitsOf classOf gs == out) := by
  decide +kernel

/-- `units` is property/v1's on every table-read row. -/
theorem unitsOf_tableUnitsRows : tableUnitsRows.all (fun (gs, out) => unitsOf classOf gs == out) := by
  decide +kernel

end Accounting.SecurityProofs.Gates.Rows
