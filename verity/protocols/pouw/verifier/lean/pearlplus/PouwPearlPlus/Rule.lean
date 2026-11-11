module

public import PouwPearlPlus.Words
meta import PouwPearlPlus.Words

/-!
# The tile rule on an opened tile (`clear_work.check_opened`, `pearl_c_debit`, `verify_run.check_y`)

After a tile's openings and digest pass, the record decides it, in `check_opened`'s order:

* **Each row**, A's rows in the tile's order then B's columns: a filler row (at or past the call's real rows) is all zero;
  no row has a nonzero word past the kept width |K|; a voluntary row is skipped; a credited row passes the per-row rules and
  an excluded row fails them. The per-row rules (`row_passes`) are TT_OUT's `OperandOK`: the noise floor ρ·α ≥ 1 on the
  exact product of the two FP32 words, and liveness (`live_row`) on the exact statistic.
* **The cap** (`cap_failure`): over the tile's credited rows × credited columns, the replayed debit is at most ρ = 1/400 of
  their cells' credit, and each credited A row's at most ρ of its own cells' (4ρ at a census k, `row_cap_at`). A word's
  debit (`word_debit`) is 40 units per flagged atom (32 MACs and a quarter of a promotion add) and 32 per later promotion
  into a +0 total. P3's forming term (`pearl_c_debit.forming_elements`, `tileDebitDev`'s elems) adds fs = 64 times |cols|/n
  per credited row's element (i, l), l < k, that is salt-dead (`saltDead`: the support's two ends agree) or whose atom
  block every word of the row over the credited columns flags. In integers, with D the words' inert work and E the
  elements, the tile keeps its inert-work bound when 400·(n·D + 64·|cols|·E) ≤ creditOf(1, |K|, n)·|rows|·|cols|.
* **The served y** (`check_y`), on a call that commits one: each real row of the tile opens against the call's y root,
  holds a word for each of the tile's columns (the opening doesn't bind how many words a row has; one that ends before
  the tile's last column is refused, `the opened row holds N words`), and its word at column j is
  `RN_bf16(RN(RN(U · RN(1/α_A)) · RN(1/α_B)))` widened, `pearl_c_u.y`: `unscale`, the copy of the served y's Definition
  (`Pouw.PearlC.unscale`, `PearlCUnscale_v1` at `BIAS` off), where each step stays finite.

A debit replay that leaves the finite range raises in the Python (`P.Reject`, uncaught by `verify_passes`); here it refuses
the tile, `the debit replay: …`.
-/

@[expose] public section

namespace PouwCheck.PearlPlus.Rule

open Evaluate.Fp32 Words

/-- A finite FP32 word's value as (negative, M, e), `(-1)^s · M · 2^e`. -/
def parts (w : Nat) : Bool × Nat × Int :=
  let e := (w >>> 23) &&& 0xFF
  let m := w &&& 0x7FFFFF
  (w >>> 31 == 1, if e == 0 then m else m + 2 ^ 23, if e == 0 then -149 else (e : Int) - 150)

def isNan (w : Nat) : Bool := (w >>> 23) &&& 0xFF == 0xFF && w &&& 0x7FFFFF != 0
def isInf (w : Nat) : Bool := (w >>> 23) &&& 0xFF == 0xFF && w &&& 0x7FFFFF == 0

/-- `f32(ρ) · f32(α) >= 1.0` in double, which is the exact product of the two words. -/
def floorOK (rho alpha : Nat) : Bool :=
  let (sr, mr, er) := parts rho
  let (sa, ma, ea) := parts alpha
  if isNan rho || isNan alpha then false
  else if isInf rho || isInf alpha then sr == sa && mr != 0 && ma != 0
  else
    let p := mr * ma
    if sr != sa || p == 0 then false
    else
      let s := er + ea
      if s ≥ 0 then true else p ≥ 2 ^ (-s).toNat

/-- x² · 2^300 of a finite FP32 word, an integer. -/
def sq300 (w : Nat) : Nat := let (_, m, e) := parts w; m * m * 2 ^ (2 * e + 300).toNat

/-- `pearl_c.live_row`: with S = Σ_{p ≡ 0 (8)} x_p² and k8 = ⌈k/8⌉, at most k/64 entries have x_l² · k8 ≥ 64 · S. -/
def liveRow (x : Array Nat) : Bool :=
  let k := x.size
  let k8 := (k + 7) / 8
  let bound := 64 * (List.range k8).foldl (fun s i => s + sq300 x[8 * i]!) 0
  64 * x.foldl (fun d w => if sq300 w * k8 ≥ bound then d + 1 else d) 0 ≤ k

/-- `clear_work.row_passes` on a formed row: ρ and α are `s5_stats`'s of the same words. -/
def rowPasses (rho alpha : Nat) (x : Array Nat) : Bool := floorOK rho alpha && liveRow x

/-! ## The debit (`pearl_c_debit`) -/

/-- An FP32 word as an integer in value order; ±0 are one key. -/
def key (w : Nat) : Int := if w >>> 31 == 1 then -((w &&& 0x7FFFFFFF : Nat) : Int) else (w : Int)

def wordOf (x : Int) : Nat := if x < 0 then x.natAbs ||| 0x80000000 else x.toNat

def promote (total w : Nat) : Reject Nat := do
  let out := f32AddV2 total w
  if (out >>> 23) &&& 0xFF == 0xFF then throw "a promotion left the finite range"
  pure (if out == 0x80000000 then 0 else out)

def step (w : Nat) (a b : List Nat) : Reject Nat := do
  let out := atom w a b
  if (out >>> 23) &&& 0xFF == 0xFF then throw "an atom left the finite range"
  pure out

/-- The key of RN(word(x) + w); an overflow keys past every finite value. -/
def carried (x : Int) (w : Nat) : Int :=
  let out := f32AddV2 (wordOf x) w
  if out == 0x80000000 then 0 else key out

def TOP : Int := 0x7F7FFFFF

/-- `_least`: the least key in [−TOP, TOP] with `pred` (monotone), or TOP + 1; bisection, floor midpoints. -/
def least (pred : Int → Bool) : Int := Id.run do
  if !pred TOP then return TOP + 1
  let mut lo := -TOP
  let mut hi := TOP
  for _ in [0:64] do
    if lo < hi then
      let mid := (lo + hi).fdiv 2
      if pred mid then hi := mid else lo := mid + 1
  return lo

/-- `_masked`: a group's word from +0 under each mask of skipped atoms (bit j: atom j passes its incoming word through). -/
def masked (atoms : List (List Nat × List Nat)) : Reject (List (Nat × Nat)) :=
  atoms.zipIdx.foldlM (fun words ((a, b), j) => do
    let mut nxt := []
    for (mask, w) in words do
      nxt := nxt ++ [(mask, ← step w a b), (mask ||| (1 <<< j), w)]
    pure nxt) [(0, 0)]

/-- `word_debit(a, b)`: the word's atom flags (`Skip.wordFlags k a b t` for t < k/32) and its later promotions into a +0
total. -/
def wordFlags (a b : Array Nat) : Reject (Array Bool × Nat) := do
  let k := a.size
  if k != b.size || k % (PearlPlus.G * 32) != 0 then throw "a word's rows are whole promotion groups"
  let nG := k / (PearlPlus.G * 32)
  let groups ← (List.range nG).toArray.mapM fun g =>
    masked ((List.range PearlPlus.G).map fun j => let s := 32 * (PearlPlus.G * g + j); (slice a s 32, slice b s 32))
  let base (g : Nat) : Nat := ((groups[g]!.lookup 0).getD 0)
  let mut totals : Array Nat := #[0]
  for g in [0:nG] do
    totals := totals.push (← promote totals[g]! (base g))
  let mut lo := key totals[nG]!
  let mut hi := lo
  let mut bounds : Array (Int × Int) := #[(lo, hi)]
  for i in [0:nG - 1] do
    let g := nG - 1 - i
    let w := base g
    let (lo0, hi0) := (lo, hi)
    lo := least fun x => carried x w ≥ lo0
    hi := least (fun x => carried x w > hi0) - 1
    bounds := bounds.push (lo, hi)
  bounds := bounds.reverse
  let mut flags : Array Bool := #[]
  for g in [0:nG] do
    let (blo, bhi) := bounds[g]!
    let mut keeps : List Nat := []
    for (mask, w) in groups[g]! do
      if mask != 0 then
        let kk := key (← promote totals[g]! w)
        if blo ≤ kk && kk ≤ bhi then keeps := mask :: keeps
    for j in [0:PearlPlus.G] do
      flags := flags.push (keeps.any fun mask => (mask >>> j) &&& 1 == 1)
  let zeros := ((List.range (nG - 1)).filter fun g => totals[g + 1]! == 0).length
  return (flags, zeros)

/-- The word's flagged atoms (`Skip.chainDebit`'s count) and its later promotions into a +0 total. -/
def wordDebit (a b : Array Nat) : Reject (Nat × Nat) := do
  let (flags, zeros) ← wordFlags a b
  return ((flags.filter id).size, zeros)

/-- The salt-dead support's ends (`SaltDead.support`, `Cap.plusEnd` and `minusEnd`): E's flat entry e4m3(44), and F's
e4m3(+11/4) and e4m3(−11/4). Must equal `pearl_c_debit.E_FLAT` and `F_ENDS`. -/
def E_FLAT : Nat := 0x63
def F_PLUS : Nat := 0x43
def F_MINUS : Nat := 0xC3

/-- `pearl_c_debit.flat_code`: c_β, the E′ code of the support's flat E entry at the row's β. -/
def flatCode (beta : Nat) : Reject Nat := do toFp8 (f32MulV2 beta (← ofFp8 E_FLAT))

/-- `pearl_c_debit.ends_agree`: the two ends' codes have one value, equal or both zeros (`Cap.codeVal_eq_iff`). -/
def endsAgree (cPlus cMinus : Nat) : Bool := cPlus == cMinus || (cPlus ||| cMinus) &&& 0x7F == 0

/-- `pearl_c_debit.salt_dead` on one element, `cb` its row's c_β: the codes of the support's two ends, the atom from the
forming's accumulator f32(α·x) over [c_β]³² and [±11/4]³², agree; by `saltDeadP_iff_ends` it is P3's `saltDead`. -/
def saltDead (alpha cb w : Nat) : Reject Bool := do
  let acc ← finite (f32MulV2 alpha w)
  let e := List.replicate 32 cb
  pure (endsAgree (← toFp8 (← step acc e (List.replicate 32 F_PLUS))) (← toFp8 (← step acc e (List.replicate 32 F_MINUS))))

/-- `pearl_c_debit.forming_elements` on one credited row, `flags` its words' over the credited columns: all its k elements
when every word flags every atom block, else each element l whose block l/32 every word flags or that is salt-dead
(every element's `saltDead` evaluated, as the Python's `salt_dead` is). -/
def rowElements (alpha beta : Nat) (x : Array Nat) (flags : List (Array Bool)) : Reject Nat := do
  let blocks := (flags.head?.map (·.size)).getD 0
  let every := (Array.range blocks).map fun t => flags.all (·[t]!)
  if every.all id then return 32 * blocks
  let cb ← flatCode beta
  let dead ← x.mapM (saltDead alpha cb)
  return ((List.range x.size).filter fun l => every[l / 32]! || dead[l]!).length

/-- fs, the salt-dependent part of `FORMING` (the noise atom and the cast), per inert forming element. Must equal
`Prices.h100`'s `forming_credited`. -/
def FS : Nat := 64

/-- n times `debit_units` at the H100's prices over `cc` credited columns of a call of n columns: 40 per flagged
atom-word, 32 per later +0 promotion, and fs·cc/n per forming element. -/
def debitUnits (n cc atoms zeros elems : Nat) : Nat :=
  n * ((32 + PearlPlus.FADD / PearlPlus.G) * atoms + PearlPlus.FADD * zeros) + FS * cc * elems

/-- ρ, the H100 record's cap, as num/den. -/
def CAP_NUM : Nat := 1
def CAP_DEN : Nat := 400

/-- `f"{float(n / d):.0f}"` of a nonnegative rational: nearest, ties to even (exact here: a tie is n/d itself). -/
def roundHalfEven (n d : Nat) : Nat :=
  let q := n / d
  let r := n % d
  if 2 * r > d || (2 * r == d && q % 2 == 1) then q + 1 else q

/-! ## The rule -/

inductive Kind where
  | filler | excluded | voluntary | credited
  deriving BEq

def Kind.name : Kind → String
  | .filler => "filler" | .excluded => "excluded" | .voluntary => "voluntary" | .credited => "credited"

/-- `check_opened`'s rows (`clear_work.rows_failure`) on a tile whose openings and digest passed: `ops` its formed
operands, the rows' words theirs. `none`, or the reason a row fails. -/
def rowsFailure (r : UnitRecord) (rows cols : List Nat) (ops : Ops) : Option String := Id.run do
  let kinds := (rows.zipIdx.map fun (i, p) =>
      ("A", i, ops.a, p, if i ≥ r.realRows then Kind.filler else if r.excludedA.contains i then .excluded
        else if r.voluntaryA.contains i then .voluntary else .credited)) ++
    (cols.zipIdx.map fun (j, p) =>
      ("B", j, ops.b, p, if r.excludedB.contains j then Kind.excluded else if r.voluntaryB.contains j then .voluntary
        else .credited))
  for (side, index, f, p, kind) in kinds do
    let words := f.x[p]!
    if kind == .filler then
      if words.any (· != 0) then return some s!"filler row {index} is not zero"
      continue
    if (words.extract r.kKept words.size).any (· != 0) then
      return some s!"{side} row {index} is not zero past the kept width"
    if kind == .voluntary then continue
    let ok := rowPasses f.rho[p]! f.alpha[p]! words
    if (kind == .credited && !ok) || (kind == .excluded && ok) then
      return some s!"{side} row {index} is {kind.name} but {if kind == .credited then "fails" else "passes"} the per-row rules"
  return none

/-- The inert-work bound (`clear_work.cap_failure`) on the tile's credited rows × columns: `.ok none` within it,
`.ok (some why)` over it (as a tile, or on one credited A row), `.error why` where the debit replay fails. -/
def capCheck (s : Scheme) (r : UnitRecord) (rows cols : List Nat) (ops : Ops) : Except String (Option String) := do
  let (cr, cc) := r.credited rows cols
  if cr.isEmpty || cc.isEmpty then return none
  let posA := fun i => rows.idxOf i
  let posB := fun j => cols.idxOf j
  let replay : Except String (List Nat) := do
    let words ← cr.mapM fun i => cc.mapM fun j => wordFlags ops.a.codes[posA i]! ops.b.codes[posB j]!
    let elems ← (cr.zip words).mapM fun (i, ws) =>
      rowElements ops.a.alpha[posA i]! ops.a.beta[posA i]! ops.a.x[posA i]! (ws.map (·.1))
    pure ((words.zip elems).map fun (ws, e) =>
      debitUnits r.matmul.n cc.length (ws.foldl (fun t w => t + (w.1.filter id).size) 0) (ws.foldl (· + ·.2) 0) e)
  match replay with
  | .error e => throw s!"the debit replay: {e}"
  | .ok rowDebit =>
    let u := r.matmul
    let credit := creditOf 1 r.kKept u.n
    let debit := rowDebit.foldl (· + ·) 0
    let capNum := CAP_NUM * credit * cr.length * cc.length
    if debit * CAP_DEN > capNum then
      return some (s!"the replayed debit {roundHalfEven debit u.n} exceeds the cap {roundHalfEven capNum (CAP_DEN * u.n)} " ++
        "(ρ·the credited cells' credit)")
    let widen := if s.censusKs.contains u.k then 4 else 1
    for (i, d) in cr.zip rowDebit do
      let capNum := CAP_NUM * credit * cc.length * widen
      if d * CAP_DEN > capNum then
        return some (s!"A row {i}'s replayed debit {roundHalfEven d u.n} exceeds its cap " ++
          s!"{roundHalfEven capNum (CAP_DEN * u.n)} (the row cap: " ++
          s!"{if widen == 1 then "" else toString widen}ρ·its credited cells' credit)")
    return none

/-- `cap_failure`: `capCheck`'s reason, over the bound or the replay's. -/
def capFailure (s : Scheme) (r : UnitRecord) (rows cols : List Nat) (ops : Ops) : Option String :=
  match capCheck s r rows cols ops with
  | .ok why => why
  | .error why => some why

/-- `check_opened` on a tile whose openings and digest passed: the rows, then the inert-work bound. `none`, or the reason
it fails. -/
def checkOpened (s : Scheme) (r : UnitRecord) (rows cols : List Nat) (ops : Ops) : Option String :=
  (rowsFailure r rows cols ops).orElse fun _ => capFailure s r rows cols ops

/-- The FP32 word of 1.0. Must equal `Pouw.PearlC.UNSCALE_ONE`. -/
def UNSCALE_ONE : Nat := 0x3F800000

/-- `PearlCUnscale_v1`'s word at `BIAS` off, the kernels' epilogue: RN_bf16(RN_f32(RN_f32(U · RN_f32(1/α_A)) · RN_f32(1/α_B))).
Must equal `Pouw.PearlC.unscale`. -/
def unscale (u alphaA alphaB : Nat) : Nat :=
  f32ToBf16RnBits (f32MulV2 (f32MulV2 u (f32DivV2 UNSCALE_ONE alphaA)) (f32DivV2 UNSCALE_ONE alphaB)) BF16_NAN_F2FP

/-- `pearl_c_u.y`: the served BF16 y from U and its row's and column's α, `unscale`, `none` where a step leaves the finite
range (the pin's `NonFinite`: either FP32 product, or the BF16 word). -/
def yOf (u alphaA alphaB : Nat) : Option Nat :=
  let t := f32MulV2 u (f32DivV2 UNSCALE_ONE alphaA)
  let w := f32MulV2 t (f32DivV2 UNSCALE_ONE alphaB)
  let out := unscale u alphaA alphaB
  if (t >>> 23) &&& 0xFF == 0xFF || (w >>> 23) &&& 0xFF == 0xFF || out &&& 0x7FFF ≥ 0x7F80 then none else some out

/-- Four cells of Pearl+'s y as the H200 stored it (a decode pass's matmul 0 at k = n = 2,048, tile 5), (U, α_A, α_B, y):
`verity/protocols/pouw/verifier/tests/test_pouw_pearl_c_u.py`'s `H200_Y`. -/
def H200_Y : List (Nat × Nat × Nat × Nat) :=
  [(0x478391C0, 0x426D4BA7, 0x448F000E, 0x3F7E), (0xC78DF380, 0x4275FF63, 0x44965964, 0xBF7C),
   (0x47BB95B0, 0x426E5313, 0x448643AF, 0x3FC0), (0xC81016D0, 0x42757555, 0x4495D9AA, 0xC000)]

theorem yOf_h200 : H200_Y.all (fun (u, a, b, y) => yOf u a b == some y) = true := by decide

/-- `check_y`'s word check: row x of the tile's real rows (`words` its opened FP32 words) against the tile's U. A row is
first held to the tile's columns, `width` their end (the last column + 1): the y opening binds a row's value and not how
many words it has, and a read past a row's end is 0, the word of an unscale of +0, so a row that ends before `width` is
refused before any of its words is read. -/
def yReason (rows cols : List Nat) (ops : Ops) (ch : Checked) (opened : List (Array Nat)) : Option String := Id.run do
  let width := cols.foldl (fun w j => max w (j + 1)) 0
  for (words, x) in opened.zipIdx do
    if words.size < width then
      return some s!"y[{rows[x]!}]: the opened row holds {words.size} words, the tile's columns end at {width}"
    for (j, z) in cols.zipIdx do
      match yOf ch.u[x]![z]! ops.a.alpha[x]! ops.b.alpha[z]! with
      | none => return some s!"y[{rows[x]!}, {j}]: the unscale of U leaves the finite range"
      | some want => if words[j]! != want <<< 16 then return some s!"y[{rows[x]!}, {j}] is not the unscale of the tile's U"
  return none

-- Row 7 over columns 2 and 3, U = +0 at α = 1 on both, so each cell's unscale is +0 and its word 0, what a read past a
-- row's end gives: the whole row passes, and a row that ends at column 3, or before the tile, is refused by its length.
#guard
  let side (n : Nat) : Formed := { (default : Formed) with alpha := Array.replicate n UNSCALE_ONE }
  let ops : Ops := { (default : Ops) with a := side 1, b := side 2 }
  let y := yReason [7] [2, 3] ops ⟨#[], #[#[0, 0]]⟩
  yOf 0 UNSCALE_ONE UNSCALE_ONE == some 0 && y [#[5, 5, 0, 0]] == none &&
    y [#[5, 5, 0]] == some "y[7]: the opened row holds 3 words, the tile's columns end at 4" &&
    y [#[]] == some "y[7]: the opened row holds 0 words, the tile's columns end at 4" &&
    y [#[5, 5, 0, 1]] == some "y[7, 3] is not the unscale of the tile's U"

end PouwCheck.PearlPlus.Rule
