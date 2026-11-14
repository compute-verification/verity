import Spot.OneStageZk

/-!
`members-parse-time N...`: the per-member form's parse of its members (`OneStageZk.membersOf`, as the checkout compiles
the call) on N made-up members, each of one range of ten units and no unit in two, timed after the members are made;
then the same members with one more, whose range starts inside the last member's, which the parse must refuse. One
`PHASE {json}` line a size and case on stdout; exit 1 if the parse gives back other members than those made, name by
name and unit by unit, or does not refuse the unit in two. It reads no file. `members_parse.sh` builds it against the
checkout's verifier and runs it; it is a measurement, not a verifier.
-/

open Lean Spot Spot.OneStageZk

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

/-- `f`'s value and its milliseconds; `f` reads the clock's reading and `size` reads its value before the clock stops,
so the compiler can't move it out of the timing. -/
def timed {α : Type} (f : Nat → α) (size : α → Nat) : IO (α × Nat) := do
  let t0 ← IO.monoNanosNow
  let a := f t0
  if size a == 1000000007 then IO.eprintln "unreachable"
  let t1 ← IO.monoNanosNow
  return (a, (t1 - t0) / 1000000)

def row (fn : String) (members ms : Nat) (extra : List (String × Json)) : IO Unit := do
  let hwm ← statusKb "VmHWM:"
  IO.println s!"PHASE {(Json.mkObj ([("fn", Json.str fn), ("members", toJson members), ("ms", toJson ms),
    ("hwm_mb", toJson (hwm / 1024))] ++ extra)).compress}"
  (← IO.getStdout).flush

def hex128 : String := "".pushn 'a' 128

/-- Member `i`'s name, of one width, so that the names' order is the members'. -/
def nameOf (i : Nat) : String :=
  let s := toString i
  "m" ++ "".pushn '0' (12 - s.length) ++ s

def oneRange (lo hi : Nat) : Json :=
  Json.mkObj [("circuit", .str hex128), ("public", .str hex128), ("registered", Json.null),
    ("units", .arr #[.arr #[toJson lo, toJson hi]]), ("vus_per_block", toJson (1 : Nat))]

/-- Member `i`'s units, `[10 i, 10 i + 10)`. -/
def unitsOf (i : Nat) : Nat × Nat := (10 * i, 10 * i + 10)

/-- `n` members, and after them `more`. -/
def membersJ (n : Nat) (more : List (String × Json)) : Json :=
  Json.mkObj (((List.range n).map fun i => (nameOf i, oneRange (unitsOf i).1 (unitsOf i).2)) ++ more)

def noFiles : Own :=
  { programJ := Json.null, program := ByteArray.empty, partition := Json.null, circuit := "", pub := "",
    registered := none }

def count (r : Flock.V (Array Member)) : Nat :=
  match r with
  | .ok ms => ms.size
  | .error e => e.length

/-- The parse gave back the `n` members made: their names, each with its one range. -/
def made (n : Nat) (ms : Array Member) : Bool :=
  (ms.map fun m => (m.name, m.units)).qsort (·.1 < ·.1) == (Array.range n).map fun i => (nameOf i, #[unitsOf i])

def main (args : List String) : IO UInt32 := do
  let ns := args.filterMap String.toNat?
  if ns.isEmpty || ns.length != args.length || ns.any (· == 0) then
    IO.eprintln "usage: members-parse-time N..."
    return 2
  for n in ns do
    let j := membersJ n []
    let (r, ms) ← timed (fun t => if t == 0 then .error "unreachable" else membersOf noFiles j) count
    let ok := match r with | .ok got => made n got | .error _ => false
    row "membersOf" n ms [("ok", toJson ok)]
    -- one more member, whose units start inside the last member's
    let j2 := membersJ n [(nameOf n, oneRange ((unitsOf n).1 - 5) ((unitsOf n).1 + 5))]
    let (r2, ms2) ← timed (fun t => if t == 0 then .error "unreachable" else membersOf noFiles j2) count
    let refused := match r2 with
      | .ok _ => false
      | .error e => (e.splitOn "is in two members' units").length == 2
    row "membersOf, a unit in two" (n + 1) ms2 [("refused", toJson refused)]
    -- the members are read once more here, so that neither is freed inside a timing
    if j.isNull || j2.isNull || !ok || !refused then
      IO.eprintln s!"membersOf at {n} members: the members made came back: {ok}; the unit in two was refused: {refused}"
      return 1
  return 0
