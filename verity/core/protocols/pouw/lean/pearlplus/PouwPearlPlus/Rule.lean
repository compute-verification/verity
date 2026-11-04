import PouwPearlPlus.Words

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
  into a +0 total; P3's forming term is `pearl_c_debit.forming_elements`'s stub, 0.
* **The served y** (`check_y`), on a call that commits one: each real row of the tile opens against the call's y root, and
  its word at column j is `RN_bf16(RN(RN(U · RN(1/α_A)) · RN(1/α_B)))` widened, `pearl_c_u.y`: `unscale`, the copy of the
  served y's Definition (`Pouw.PearlC.unscale`, `PearlCUnscale_v1` at `BIAS` off), where each step stays finite.

A debit replay that leaves the finite range raises in the Python (`P.Reject`, uncaught by `verify_passes`); here it refuses
the tile, `the debit replay: …`.
-/

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

/-- `word_debit(a, b)`: the word's flagged atoms (`Skip.wordFlags`) and its later promotions into a +0 total. -/
def wordDebit (a b : Array Nat) : Reject (Nat × Nat) := do
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
  let mut flagged := 0
  for g in [0:nG] do
    let (blo, bhi) := bounds[g]!
    let mut keeps : List Nat := []
    for (mask, w) in groups[g]! do
      if mask != 0 then
        let kk := key (← promote totals[g]! w)
        if blo ≤ kk && kk ≤ bhi then keeps := mask :: keeps
    for j in [0:PearlPlus.G] do
      if keeps.any fun mask => (mask >>> j) &&& 1 == 1 then flagged := flagged + 1
  let zeros := ((List.range (nG - 1)).filter fun g => totals[g + 1]! == 0).length
  return (flagged, zeros)

/-- `debit_units` at the H100's prices: 40 per flagged atom-word, 0 per forming element (the stub), 32 per later +0
promotion. -/
def debitUnits (atoms zeros : Nat) : Nat := (32 + PearlPlus.FADD / PearlPlus.G) * atoms + PearlPlus.FADD * zeros

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
  let words : Except String (List (List (Nat × Nat))) := cr.mapM fun i => cc.mapM fun j =>
    wordDebit ops.a.codes[posA i]! ops.b.codes[posB j]!
  match words with
  | .error e => throw s!"the debit replay: {e}"
  | .ok words =>
    let u := r.matmul
    let credit := creditOf 1 r.kKept u.n
    let den := CAP_DEN * u.n
    let rowDebit := words.map fun ws => debitUnits (ws.foldl (· + ·.1) 0) (ws.foldl (· + ·.2) 0)
    let debit := rowDebit.foldl (· + ·) 0
    let capNum := CAP_NUM * credit * cr.length * cc.length
    if debit * den > capNum then
      return some s!"the replayed debit {debit} exceeds the cap {roundHalfEven capNum den} (ρ·the credited cells' credit)"
    let widen := if s.censusKs.contains u.k then 4 else 1
    for (i, d) in cr.zip rowDebit do
      let capNum := CAP_NUM * credit * cc.length * widen
      if d * den > capNum then
        return some (s!"A row {i}'s replayed debit {d} exceeds its cap {roundHalfEven capNum den} (the row cap: " ++
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

/-- Four cells of Pearl+'s y as the H200 stored it (r20261103-053659-36f1's decode pass, matmul 0 at k = n = 2,048, tile 5),
(U, α_A, α_B, y): `verity/core/protocols/pouw/tests/test_pouw_pearl_c_u.py`'s `H200_Y`. -/
def H200_Y : List (Nat × Nat × Nat × Nat) :=
  [(0x478391C0, 0x426D4BA7, 0x448F000E, 0x3F7E), (0xC78DF380, 0x4275FF63, 0x44965964, 0xBF7C),
   (0x47BB95B0, 0x426E5313, 0x448643AF, 0x3FC0), (0xC81016D0, 0x42757555, 0x4495D9AA, 0xC000)]

theorem yOf_h200 : H200_Y.all (fun (u, a, b, y) => yOf u a b == some y) = true := by decide

/-- `check_y`'s word check: row x of the tile's real rows (`words` its opened FP32 words, the call's whole width) against
the tile's U. -/
def yReason (rows cols : List Nat) (ops : Ops) (ch : Checked) (opened : List (Array Nat)) : Option String := Id.run do
  for (words, x) in opened.zipIdx do
    for (j, z) in cols.zipIdx do
      match yOf ch.u[x]![z]! ops.a.alpha[x]! ops.b.alpha[z]! with
      | none => return some s!"y[{rows[x]!}, {j}]: the unscale of U leaves the finite range"
      | some want => if words[j]! != want <<< 16 then return some s!"y[{rows[x]!}, {j}] is not the unscale of the tile's U"
  return none

end PouwCheck.PearlPlus.Rule
