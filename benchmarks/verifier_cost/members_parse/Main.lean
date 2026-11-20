import Spot.OneStageZk

/-!
`members-parse-time DIR N...`: the per-member form's parse of its members (`OneStageZk.membersOf`, as the checkout
compiles the call) on N made-up members, member i naming block i of DIR/N.json's partition object (`members_parse.py`:
N blocks of ten units over a real program), timed after the members are made and the file read; then the same members
with one more naming the last block, whose units are then in two members', which the parse must refuse. One
`PHASE {json}` line a size and case on stdout; exit 1 if the parse gives back other members than those made, name by
name and unit by unit, or does not refuse the unit in two. `members_parse.sh` writes DIR, builds it against the
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

def ofBlock (k : Nat) : Json :=
  Json.mkObj [("block", toJson k), ("public", .str hex128), ("registered", Json.null),
    ("vus_per_block", toJson (1 : Nat))]

/-- Member `i`'s units, block `i`'s: `[10 i, 10 i + 10)`. -/
def unitsOf (i : Nat) : Nat × Nat := (10 * i, 10 * i + 10)

/-- `n` members, and after them `more`. -/
def membersJ (n : Nat) (more : List (String × Json)) : Json :=
  Json.mkObj (((List.range n).map fun i => (nameOf i, ofBlock i)) ++ more)

/-- The program and the partition object of DIR/N.json, naming no file. -/
def sharedOf (dir : String) (n : Nat) : IO Own := do
  let j ← IO.ofExcept (Json.parse (← IO.FS.readFile s!"{dir}/{n}.json"))
  let programJ ← IO.ofExcept (j.getObjVal? "program")
  let program ← IO.ofExcept (Flock.canon programJ)
  pure { programJ, program := program.toUTF8, partition := ← IO.ofExcept (j.getObjVal? "partition"), pub := "",
         registered := none }

def count (r : Flock.V (Array Member)) : Nat :=
  match r with
  | .ok ms => ms.size
  | .error e => e.length

/-- The parse gave back the `n` members made: their names, each with its one range. -/
def made (n : Nat) (ms : Array Member) : Bool :=
  (ms.map fun m => (m.name, m.units)).qsort (·.1 < ·.1) == (Array.range n).map fun i => (nameOf i, #[unitsOf i])

def main (args : List String) : IO UInt32 := do
  let (dir, sizes) := (args.headD "", args.drop 1)
  let ns := sizes.filterMap String.toNat?
  if dir.isEmpty || ns.isEmpty || ns.length != sizes.length || ns.any (· == 0) then
    IO.eprintln "usage: members-parse-time DIR N..."
    return 2
  for n in ns do
    let shared ← sharedOf dir n
    let j := membersJ n []
    let (r, ms) ← timed (fun t => if t == 0 then .error "unreachable" else membersOf shared j) count
    let ok := match r with | .ok got => made n got | .error _ => false
    row "membersOf" n ms [("ok", toJson ok)]
    -- one more member, naming the last member's block
    let j2 := membersJ n [(nameOf n, ofBlock (n - 1))]
    let (r2, ms2) ← timed (fun t => if t == 0 then .error "unreachable" else membersOf shared j2) count
    let refused := match r2 with
      | .ok _ => false
      | .error e => (e.splitOn "is in two members' units").length == 2
    row "membersOf, a unit in two" (n + 1) ms2 [("refused", toJson refused)]
    -- the members are read once more here, so that neither is freed inside a timing
    if j.isNull || j2.isNull || !ok || !refused then
      let why := match r with | .error e => s!" ({e})" | .ok _ => ""
      IO.eprintln s!"membersOf at {n} members: the members made came back: {ok}{why}; the unit in two was refused: {refused}"
      return 1
  return 0
