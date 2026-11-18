import Spot.Wires

/-!
`wires-time N [--no-ref]`: on a wiring of `N` wires over eight templates (half of them between templates, half within one,
each reading one unit of its own port), the wires' check (`Spot.Wires.check`), its overlap test (`pairwiseDisjoint`, which
compiled code runs as the sweep) and its order (`orderCode` with `orderedWire`), and, unless `--no-ref`, the overlap test
as a loop over every pair; then the same wiring with its last wire twice, which every overlap test refuses. One
`PHASE {json}` line each on stdout; exit 1 where the check refuses the wiring, an overlap test passes the doubled one, or
the pairwise loop and the sweep disagree. `wires.sh` builds it against the checkout's verifier and runs it; it is a
measurement, not a verifier.
-/

open Lean Spot.Wires

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

def row (fn : String) (n ms : Nat) (extra : List (String × Json)) : IO Unit := do
  let hwm ← statusKb "VmHWM:"
  IO.println s!"PHASE {(Json.mkObj ([("fn", Json.str fn), ("n", toJson n), ("ms", toJson ms),
    ("hwm_mb", toJson (hwm / 1024))] ++ extra)).compress}"
  (← IO.getStdout).flush

/-- The overlap test as it is stated: every pair. A definition of its own, so the sweep's `@[csimp]` doesn't reach it. -/
def pairwiseRef : List Wire → Bool
  | [] => true
  | w :: ws => ws.all (disjoint w) && pairwiseRef ws

def T : Nat := 8

/-- Wire `k`: reader template `r = 1 + k % 7` at unit `j = k / 7`, from template `r - 1`'s unit `j` on port `x` where `j` is
even, else from its own unit `j - 1` on port `s`. -/
def wireAt (k : Nat) : Wire :=
  let r := 1 + k % (T - 1)
  let j := k / (T - 1)
  if j % 2 == 0 then
    { reader := s!"t{r}", port := "x", writer := s!"t{r - 1}", out := "y", start := j, radix := [1], base := j,
      step := [0], cut0 := 0, cuts := 1 }
  else
    { reader := s!"t{r}", port := "s", writer := s!"t{r}", out := "y", start := j, radix := [1], base := j - 1,
      step := [0], cut0 := 0, cuts := 1 }

def wiring (ws : List Wire) (n : Nat) : Wiring :=
  let units := n / (T - 1) + 2
  { templates := (List.range T).map fun t =>
      (s!"t{t}", ({ units, rows := [("x", ⟨1, 1⟩), ("s", ⟨1, 1⟩)], outs := [("y", 1)] } : Template))
    cuts := [], wires := ws }

def bit (b : Bool) : Nat := if b then 1 else 0

def main (args : List String) : IO UInt32 := do
  let (n, ref) ← match args with
    | [s] => pure (s.toNat!, true)
    | [s, "--no-ref"] => pure (s.toNat!, false)
    | _ => do IO.eprintln "usage: wires-time N [--no-ref]"; return 2
  let ws := (List.range n).map wireAt
  let c := wiring ws n
  if ws.length != n then return 2
  let (ok, ms) ← timed (fun t => t != 0 && (check c).toBool) bit
  row "check" n ms [("ok", toJson ok)]
  if !ok then
    IO.eprintln "the wires' check refused the wiring"
    return 1
  let (dis, ms) ← timed (fun t => t != 0 && pairwiseDisjoint c.wires) bit
  row "pairwiseDisjoint" n ms [("ok", toJson dis)]
  let (ord, ms) ← timed (fun t => t != 0 && (orderCode c orderedWire).isNone) bit
  row "order" n ms [("ok", toJson ord)]
  let twice := c.wires ++ c.wires.getLast?.toList
  let (dup, ms) ← timed (fun t => t != 0 && pairwiseDisjoint twice) bit
  row "pairwiseDisjoint-doubled" (n + 1) ms [("ok", toJson dup)]
  if dup then
    IO.eprintln "the sweep passed a wiring with one wire twice"
    return 1
  if ref then
    let (r, ms) ← timed (fun t => t != 0 && pairwiseRef c.wires) bit
    let (rd, _) ← timed (fun t => t != 0 && pairwiseRef twice) bit
    row "pairwiseRef" n ms [("ok", toJson r), ("agree", toJson (r == dis && rd == dup))]
    if r != dis || rd != dup then
      IO.eprintln "the pairwise loop and the sweep disagree"
      return 1
  return 0
