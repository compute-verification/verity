import RemoteAudit.Property.Root

/-!
# P(c, R_in, L) at B operand bits (`stacking.judge`)

Under the audit's windows (a range of the census's steps, the bytes the audit certified as read in it, its duration):
no gate is refused (`unlisted`), no word is read at another width (`reinterpretation`), the Program is Restricted B
(`operand-bits`), every core step is proven (`s8-window`), every step lies in a window (`uncovered`), each class's units
in a window are at most c[class] per certified byte read, the run-input bytes a window first reads are at most R_in
times its duration (`inputs`), and no dynamic value is held more than L steps (`lifetime`). Every comparison is exact.
-/

namespace RemoteAudit.Property

open Lean (Json toJson)

/-- A rational, its numerator signed and its denominator positive, in lowest terms (`fractions.Fraction`). -/
structure Q where
  num : Int
  den : Nat
  deriving BEq, Inhabited

def Q.make (n : Int) (d : Nat) : Q :=
  let g := Nat.gcd n.natAbs d
  if g == 0 then ⟨0, 1⟩ else ⟨n / (g : Int), d / g⟩

def Q.ofNat (n : Nat) : Q := ⟨n, 1⟩
def Q.mul (a b : Q) : Q := Q.make (a.num * b.num) (a.den * b.den)
def Q.lt (a b : Q) : Bool := a.num * b.den < b.num * a.den

/-- `str(Fraction)`. -/
def Q.str (q : Q) : String := if q.den == 1 then toString q.num else s!"{q.num}/{q.den}"

/-- A rational written as `str(Fraction)` writes one (an integer, or `n/d` with d > 0), or a JSON integer. -/
def Q.ofJson (j : Json) (what : String) : Except String Q := do
  let int := fun (s : String) => match s.toList with
    | '-' :: ds => if !ds.isEmpty && ds.all Char.isDigit then some (-((String.ofList ds).toNat! : Int)) else none
    | ds => if !ds.isEmpty && ds.all Char.isDigit then some ((String.ofList ds).toNat! : Int) else none
  match j with
  | .num _ => match j.getInt? with
    | .ok n => return ⟨n, 1⟩
    | .error _ => throw s!"{what}: not an integer or a fraction"
  | .str s =>
    match s.splitOn "/" with
    | [n] => match int n with
      | some n => return ⟨n, 1⟩
      | none => throw s!"{what}: {s} is not an integer or a fraction"
    | [n, d] => match int n, int d with
      | some n, some d => if d > 0 then return Q.make n d.toNat else throw s!"{what}: {s} has a denominator ≤ 0"
      | _, _ => throw s!"{what}: {s} is not an integer or a fraction"
    | _ => throw s!"{what}: {s} is not an integer or a fraction"
  | _ => throw s!"{what}: not an integer or a fraction"

/-- A window: its steps `[start, stop)`, the bytes certified as read in it, and its duration in seconds. -/
structure Window where
  start : Int
  stop : Int
  reads : Nat
  seconds : Q
  deriving Inhabited

private def clsObj (u : Array Nat) : Json := Json.mkObj ((Array.range 4).toList.map fun k => (BUDGETED[k]!.name, toJson u[k]!))

private def pairJ (r : Int × Int) : Json := .arr #[toJson r.1, toJson r.2]

/-- `judge`: the verdict's JSON (`Verdict.to_json`). `c` is units per certified byte for each budgeted class in
`BUDGETED`'s order. A window outside the census's steps, or a B below 1, is refused. -/
def judge (a : Analysis) (c : Array Q) (windows : Array Window) (R : Q) (L : Option Nat) (B : Nat) : Except String Json := do
  if c.size != 4 then throw "c gives units per certified byte for exactly the four budgeted classes"
  if B < 1 then throw s!"B is an integer >= 1, not {B}"
  let S := a.units.size
  for w in windows do
    if !(0 ≤ w.start && w.start < w.stop && w.stop ≤ S && Q.lt ⟨0, 1⟩ w.seconds) then
      throw s!"a window is a non-empty range of the census's {S} steps with reads >= 0 and seconds > 0, not [{w.start}, {w.stop})"
  let mut reasons : Array Json := #[]
  for code in REFUSED do
    let xs := a.refused.filter (·.1.1 == code)
    if !xs.isEmpty then
      reasons := reasons.push (Json.mkObj [("code", .str code),
        (if code == "reinterpretation" then "definitions" else "primitives",
          Json.mkObj (xs.toList.map fun ((_, f), n) => (f, toJson n)))])
  let budgeted := BUDGETED.map (·.name)
  let wide := a.shapes.filter fun (g, cls, bits) =>
    (match cls with | some c => budgeted.contains c | none => false) && bits > B && !column g
  if !wide.isEmpty then
    reasons := reasons.push (Json.mkObj [("code", .str "operand-bits"), ("B", toJson B),
      ("primitives", Json.mkObj (wide.toList.map fun (g, _, b) => (g, toJson b)))])
  if !a.unproven.isEmpty then
    reasons := reasons.push (Json.mkObj [("code", .str "s8-window"),
      ("steps", Json.mkObj (a.unproven.toList.map fun (f, n) => (f, toJson n))),
      ("sites", .arr (a.sites.map fun ((d, n), acc, sum) =>
        Json.mkObj [("definition", .str d), ("node", toJson n), ("acc", pairJ acc), ("sum", pairJ sum)]))])
  let covered := (Array.range S).map fun (s : Nat) => windows.any fun w => w.start ≤ s && (s : Int) < w.stop
  if covered.any (!·) then
    reasons := reasons.push (Json.mkObj [("code", .str "uncovered"),
      ("steps", .arr ((Array.range S).filter (!covered[·]!) |>.map toJson))])
  let mut rows : Array Json := #[]
  for w in windows do
    let steps := (Array.range S).filter fun (s : Nat) => w.start ≤ s && (s : Int) < w.stop
    let u := (Array.range 4).map fun k => steps.foldl (fun n s => n + a.units[s]![k]!) 0
    let inb := Q.make ((steps.foldl (fun n s => n + a.inputBits[s]!) 0 : Nat) : Int) 8
    let fields : List (String × Json) := [("steps", .arr #[toJson w.start, toJson w.stop]), ("reads", toJson w.reads),
      ("seconds", .str w.seconds.str), ("units", clsObj u), ("input_bytes", .str inb.str)]
    rows := rows.push (Json.mkObj fields)
    for k in [0:4] do
      if Q.lt (c[k]!.mul (Q.ofNat w.reads)) (Q.ofNat u[k]!) then
        reasons := reasons.push (Json.mkObj ([("code", .str BUDGETED[k]!.name)] ++ fields ++ [("c", .str c[k]!.str)]))
    if Q.lt (R.mul w.seconds) inb then
      reasons := reasons.push (Json.mkObj ([("code", .str "inputs")] ++ fields ++ [("R_in", .str R.str)]))
  if let some L := L then
    if a.maxLifetime > L then
      reasons := reasons.push (Json.mkObj [("code", .str "lifetime"), ("max_lifetime", toJson a.maxLifetime), ("L", toJson L)])
  let totals := (Array.range 4).map fun k => a.units.foldl (fun n u => n + u[k]!) 0
  let numbers := Json.mkObj [
    ("c", Json.mkObj ((Array.range 4).toList.map fun k => (BUDGETED[k]!.name, .str c[k]!.str))),
    ("R_in", .str R.str), ("L", match L with | some l => toJson l | none => .null), ("B", toJson B),
    ("windows", .arr rows), ("units", clsObj totals), ("family_steps", toJson a.family), ("macs", toJson a.macs),
    ("affine", toJson a.affine), ("max_lifetime", toJson a.maxLifetime)]
  return Json.mkObj [("schema", .str SCHEMA), ("classes", .str digest), ("ok", .bool reasons.isEmpty),
    ("reasons", .arr reasons), ("numbers", numbers)]

end RemoteAudit.Property
