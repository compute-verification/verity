import Definitions.Accounting.Gates.Declared
import Definitions.Accounting.Gates.Registry

/-!
# The circuit model against the running Python, row by row

Rows the kernel evaluates the model on, each set recomputed from the Python by
`verity/ml/circuits/tests/test_lean_gates.py`:

* `s8Rows`: `intervals` over an `S8Dot_v1` coordinate (`s8Dot`, `ops/s8.py`'s `_dot_body`) whose operands and incoming
  accumulator come from producers declaring the binding's `A`, `B`, `A_g`, `B_g` and `C`: the derived intervals of its
  words `T_1 … T_{K/L}` and `D_g`, or none when some step leaves its window (`Proven`'s conditions on an s8-family step),
  against `s8.no_wrap`'s `T_t` and `D_g`, or its refusal.
* `censusRows`: `census` of `S8Dot_v1` coordinates (or one `S8Matmul_v1`) and `LimbSplit_v1` splits over their inputs
  (`censusDag`), against census/v1 of the same Program: the s8 family's steps and MACs, served and tail apart in all and
  per engine step, the constants, the limb splits' carries, the inputs' bits, and property/v1's units of the limb steps.
* `classRows`, `familyRows`, `tableReadRows`: `classOf` on primitive ids (`classTag`), against property/v1's `class_of`
  with its gathers' kinds and its s8 family's formats, and circuits' input ids.
* `unitsRows`, `tableUnitsRows`: `units` by class on small circuits of primitive ids (`dagOf`: each gate an operation of
  its id's `classOf` class), against property/v1's units of the same Program.
-/

namespace Accounting.SecurityProofs.Gates.Rows

open Accounting.Gates

/-! ## An `S8Dot_v1` coordinate -/

/-- The constant 0 a chain or a fresh tail group starts from (`prims.ZERO32`), at engine step `t`. -/
def zero32 (t : ℕ) : Gate ℕ := ⟨.const 0, [], 0, 0, t, "Const32[0x00000000]_v1", .other, false⟩

/-- Step `s` of an `S8Dot_v1` body with statics `K`, `L` and the tail's atom counts, at engine step `t`, on the
accumulator at gate `acc` and operands `a`, `b`: `BlackwellU8S8ImmaDot32_v1` (u8·s8, wrapping) when `u8`, else
`BlackwellS8ImmaDot32_v1` (s8·s8, clamped). -/
def s8Step (u8 : Bool) (K L : ℕ) (atoms : List ℕ) (t s acc : ℕ) (a b : List ℕ) (commit : Bool) : Gate ℕ :=
  ⟨.mma (if u8 then .u8 else .s8) .s8 32 (!u8), acc :: (a ++ b), s32Min, s32Max, t,
    if u8 then "BlackwellU8S8ImmaDot32_v1" else "BlackwellS8ImmaDot32_v1", .dot K L atoms s, commit⟩

/-- `n` chained steps appended to `C`, numbered from `s₀`: the first reads the accumulator at gate `acc`, each later one
the step before it, and step `i` reads operands `a i` and `b i`. -/
def chain (u8 : Bool) (K L : ℕ) (atoms : List ℕ) (t s₀ n acc : ℕ) (a b : ℕ → List ℕ) (commit : ℕ → Bool)
    (C : Dag ℕ) : Dag ℕ :=
  C ++ (List.range n).map fun i =>
    s8Step u8 K L atoms t (s₀ + i) (if i = 0 then acc else C.length + i - 1) (a i) (b i) (commit i)

/-- **An `S8Dot_v1` coordinate** at engine step `t` (`ops/s8.py`'s `_dot_body`), appended to `pre`: from the incoming
accumulator at gate `acc`, or from a constant 0 (`T0 = "zero"`), `K/32` steps of `DOT` (u8·s8 when `u8`) over `a s` and
`b s`, `T_t` after every `L` products; then each tail group `(atoms, chain)` in order, `atoms` steps of `TDOT` (u8·s8
when `tu8`) over tail atom `j`'s `ta j` and `tb j`, from `T_{K/L}` (`chain`, the first group only) or a constant 0.
Returns the circuit and its words' gates: `T_1 … T_{K/L}`, then each group's last step `D_g`. -/
def s8Dot (t : ℕ) (u8 tu8 : Bool) (K L : ℕ) (acc : Option ℕ) (tail : List (ℕ × Bool)) (a b ta tb : ℕ → List ℕ)
    (pre : Dag ℕ) : Dag ℕ × List ℕ :=
  let atoms := tail.map Prod.fst
  let C₀ := if acc.isSome then pre else pre ++ [zero32 t]
  let C₁ := chain u8 K L atoms t 0 (K / 32) (acc.getD pre.length) a b (fun s => 32 * (s + 1) % L == 0) C₀
  let T := ((List.range (K / 32)).filter fun s => 32 * (s + 1) % L == 0).map fun s => C₀.length + s
  let r := tail.foldl (fun (st : Dag ℕ × ℕ × List ℕ) g =>
    let C := if g.2 then st.1 else st.1 ++ [zero32 t]
    let j := st.2.1
    (chain tu8 K L atoms t (K / 32 + j) g.1 (if g.2 then C.length - 1 else C.length - 1)
        (fun i => ta (j + i)) (fun i => tb (j + i)) (fun _ => false) C,
      j + g.1, st.2.2 ++ [C.length + g.1 - 1])) (C₁, 0, [])
  (r.1, T ++ r.2.2)

/-! ## `intervals` against `s8.no_wrap` -/

/-- A budgeted gate declaring `iv`: an operand's or an incoming accumulator's producer (a quantizer's clamp). -/
def producer (iv : Iv) : Gate ℕ := plainGate (.budgeted .rounding fun _ => iv.1) [] iv.1 iv.2 ""

/-- Gate `i` is not an s8-family step, or is one in its window by interval analysis (`Proven`'s conditions on it): its
derived interval within s32 and its operands' derived intervals within their formats. -/
def stepOk (C : Dag ℕ) (ivs : List Iv) (i : ℕ) : Bool :=
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
def wordIntervals (C : Dag ℕ) (ws : List ℕ) : Option (List Iv) :=
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
  let pre : Dag ℕ := [producer A, producer B] ++ C.toList.map producer ++
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

/-- An input gate of slot `i`, bound at `b`, declared at a signed `w`-bit word's whole range. -/
def inputGate (b : Binding) (w i : ℕ) : Gate ℕ :=
  ⟨.input b i, [], -2 ^ (w - 1), 2 ^ (w - 1) - 1, 0, "", .other, false⟩

/-- A limb step of primitive `prim` (class `cls`) reading gate `y`, at engine step `t`. -/
def limbGate (t : ℕ) (prim : String) (cls : Budget) (y : ℕ) (lo hi : ℤ) : Gate ℕ :=
  ⟨.budgeted cls fun v => v.headD 0, [y], lo, hi, t, prim, .other, false⟩

/-- `LimbSplit_v1{FORM = "bytes"}`'s steps from the word at gate `r` with `n` carries to go, appended to `C`: its low
byte and its carry, and from the carry on; the last limb a low byte. -/
def limbBytes (t : ℕ) : ℕ → ℕ → Dag ℕ → Dag ℕ
  | 0, r, C => C ++ [limbGate t "S32LowByte_v1" .rounding r (-128) 127]
  | n + 1, r, C =>
      limbBytes t n (C.length + 1)
        (C ++ [limbGate t "S32LowByte_v1" .rounding r (-128) 127, limbGate t "S32ByteCarry_v1" .rounding r s32Min s32Max])

/-- **`LimbSplit_v1{FORM, N}`** of the word at gate `y`, at engine step `t`, appended to `C`: `"byteclamp"` (`clamp`,
`N = 3`) is `h = S32ByteCarry(y)`, `S32LowByte(y)`, `S32ClampS8(h)` and `S32ClampRest(h)`, and `"bytes"` `limbBytes`. -/
def limbSplit (t : ℕ) (clamp : Bool) (N y : ℕ) (C : Dag ℕ) : Dag ℕ :=
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
    (limbs : List (Bool × ℕ)) : Dag ℕ :=
  let m := 32 * (tail.map Prod.fst).sum
  let rb : Binding := if regB then .registered else .run
  let nAcc := if acc then N else 0
  let aB := nAcc
  let bB := aB + K
  let taB := bB + N * K
  let tbB := taB + m
  let yB := tbB + N * m
  let ins : Dag ℕ :=
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
  (c.family, c.macs, c.servedMacs, c.tailMacs, c.affine, c.carries, c.rounding, c.compareSelect, c.runBits,
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

/-- Rows `(id, class, k)` of ids `classOf` classes by their family's prefix, and ids it refuses. Four of them
(`_v2` ids of the gathers, the inputs and the field extractions, and a gather with no column size) are the ids property/v1
refuses and Lean's prefixes still class (`test_lean_gates.LEAN_LOOSER`). -/
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
  ("GatherBf16x16_v2", "range", 0),
  ("Input8_v2", "run", 0),
  ("BitAtx32_v2", "roundings", 0),
  ("GatherF32xQ_v1", "range", 0)]

/-- Rows `(id, class, k)` of table reads: `n` index bits, and a table read of no bits refused. The `_v2` id is one more
that property/v1 refuses (`test_lean_gates.LEAN_LOOSER`). -/
def tableReadRows : List (String × String × ℕ) := [
  ("TableRead16x4_v1", "low", 16),
  ("TableRead4x16_v1", "low", 4),
  ("TableRead8x32_v1", "low", 8),
  ("TableRead0x4_v1", "unlisted", 0),
  ("TableRead16x4_v2", "low", 16)]

/-- `classOf` is property/v1's `class_of` on every registry row. -/
theorem classOf_classRows : classRows.all (fun (id, tag, k) => classTag (classOf id) == (tag, k)) := by
  decide +kernel

/-- `classOf` on every family row. -/
theorem classOf_familyRows : familyRows.all (fun (id, tag, k) => classTag (classOf id) == (tag, k)) := by
  decide +kernel

/-- `classOf` on every table-read row. -/
theorem classOf_tableReadRows : tableReadRows.all (fun (id, tag, k) => classTag (classOf id) == (tag, k)) := by
  decide +kernel

/-! ## `units` against property/v1 -/

/-- The operation of `classOf id`'s class at gate `i` (an input reads slot `i`): `units` reads only the class, a gather's
index-bit kind and its operands, so an affine map is the empty sum and a budgeted primitive the constant 0. -/
def opOf (id : String) (i : ℕ) : Option (Op ℕ) :=
  (classOf id).map fun
    | .input b => .input b i
    | .const => .const 0
    | .affine => .affine 0 []
    | .family a b k sat => .mma a b k sat
    | .gather low fill => .gather low fill
    | .budgeted cls => .budgeted cls fun _ => 0

/-- The circuit of gates `(id, operands)`, each the operation `opOf` gives it; none when an id is unlisted. -/
def dagOf (gs : List (String × List ℕ)) : Option (Dag ℕ) :=
  gs.zipIdx.mapM fun e => (opOf e.1.1 e.2).map fun op => ⟨op, e.1.2, s32Min, s32Max, 0, e.1.1, .other, false⟩

/-- A circuit's units by budgeted class: roundings, compare-selects, gathers and folds. -/
def unitsOf (gs : List (String × List ℕ)) : Option (ℕ × ℕ × ℕ × ℕ) :=
  (dagOf gs).map fun C => let c := census C fun _ => True; (c.rounding, c.compareSelect, c.gather, c.fold)

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
theorem unitsOf_unitsRows : unitsRows.all (fun (gs, out) => unitsOf gs == out) := by
  decide +kernel

/-- `units` is property/v1's on every table-read row. -/
theorem unitsOf_tableUnitsRows : tableUnitsRows.all (fun (gs, out) => unitsOf gs == out) := by
  decide +kernel

end Accounting.SecurityProofs.Gates.Rows
