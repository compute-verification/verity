/-
What the compiled wire check (`Spot.Wires.check`) costs, as a count of comparisons, never a clock.

A pure function's comparisons can't be counted from inside its compiled code without changing its term. So the count
has two halves:

* **Which pass runs**: the overlap pass `check`'s compiled code calls, read from its IR (`compiledCalls`), through its own
  auxiliary declarations; a `@[csimp]` lemma changes the IR, not the term.
* **What that pass costs**: a counting twin of each pass, the same recursion returning the number of comparisons it
  makes: `disjoint` calls for the overlap pass, every pair for `pairwiseDisjoint` (`Cost.pairwise`), and for
  `overlapFree` the sort's comparisons (`Cost.sort`, of `List.mergeSort` as defined; the compiler runs it as
  `mergeSortTR₂`, of the same splits and merges) beside the sweep's (`Cost.sweep`).

The instance (`Cost.window m r`) is shaped like a served window's wiring: a reader port cut into `r` rows, one wire a
unit and row, beside an uncut port with one wire a unit, `n = m · (r + 1)` wires that `check` accepts.
-/
module

public import Spot.Wires
meta import Spot.Wires
meta import Lean.Compiler.IR.CompilerM
meta import Lean.Compiler.IR.EmitUtil
meta import Lean.Elab.Command

@[expose] public section

namespace Spot.Wires.Cost

/-- The `disjoint` calls `pairwiseDisjoint ws` makes: each wire against every later one, until the first pair that
meets. -/
def pairwise : List Wire → Nat
  | [] => 0
  | w :: ws => match ws.findIdx? (fun v => !disjoint w v) with
    | some k => k + 1
    | none => ws.length + pairwise ws

/-- The `disjoint` calls `Wires.sweepFrom w l` makes. -/
def sweepFrom (w : Wire) : List Wire → Nat
  | [] => 0
  | v :: vs =>
    if v.reader == w.reader && v.port == w.port && v.start < w.start + w.readerUnits then
      1 + if disjoint w v then sweepFrom w vs else 0
    else 0

/-- The `disjoint` calls `Wires.sweep l` makes. -/
def sweep : List Wire → Nat
  | [] => 0
  | w :: ws => sweepFrom w ws + if Wires.sweepFrom w ws then sweep ws else 0

/-- The comparisons `List.merge xs ys le` makes, with its result. -/
def merge (le : Wire → Wire → Bool) : List Wire → List Wire → Nat × List Wire
  | [], ys => (0, ys)
  | xs, [] => (0, xs)
  | x :: xs, y :: ys =>
    if le x y then let r := merge le xs (y :: ys); (r.1 + 1, x :: r.2)
    else let r := merge le (x :: xs) ys; (r.1 + 1, y :: r.2)

/-- The comparisons `List.mergeSort l le` makes, as defined (halves of `(n + 1) / 2` and `n / 2`), with its result. -/
def sort (le : Wire → Wire → Bool) (l : List Wire) : Nat × List Wire :=
  if _h : l.length < 2 then (0, l)
  else
    let a := sort le (l.take ((l.length + 1) / 2))
    let b := sort le (l.drop ((l.length + 1) / 2))
    let m := merge le a.2 b.2
    (a.1 + b.1 + m.1, m.2)
termination_by l.length
decreasing_by all_goals simp only [List.length_take, List.length_drop]; omega

/-- The comparisons `overlapFree ws` makes: the sort's, then the sweep's. -/
def overlapFree (ws : List Wire) : Nat × Nat :=
  let s := sort sweepLe ws
  (s.1, sweep s.2)

/-- A wire of reader `A`'s port `p`: unit `u`, rows `[c0, c0 + 1)`, read from writer `G`'s unit `v`. -/
def wireAt (p : String) (u v c0 : Nat) : Wire :=
  { reader := "A", port := p, writer := "G", out := "o0", start := u, radix := [1], base := v, step := [0], cut0 := c0,
    cuts := 1 }

/-- `l` reordered by `37 · i mod n` for the wire at index `i`, so the sort is not handed sorted input. -/
def scramble (l : List Wire) : List Wire :=
  ((l.zipIdx.mergeSort fun a b => (a.2 * 37) % l.length ≤ (b.2 * 37) % l.length)).map (·.1)

/-- `m` reader units, port `k` cut into `r` rows of 2 words with a wire a unit and row, port `q` uncut with a wire a
unit: `m · (r + 1)` wires, scrambled. -/
def window (m r : Nat) : Wiring where
  templates := [("G", { units := m * r + m, rows := [("x", ⟨1, 1⟩)], outs := [("o0", 2)] }),
    ("A", { units := m, rows := [("k", ⟨2 * r, r⟩), ("q", ⟨2, 1⟩)], outs := [("o0", 1)] })]
  cuts := [("A", [("k", r)])]
  wires := scramble (((List.range m).flatMap fun u => (List.range r).map fun c => wireAt "k" u (u * r + c) c) ++
    (List.range m).map fun u => wireAt "q" u (m * r + u) 0)

end Spot.Wires.Cost

namespace Spot.Wires

open Lean

/-- The constants among `cands` that `root`'s compiled code calls, through `root`'s own auxiliary declarations. -/
private meta partial def compiledCalls (env : Environment) (root : Name) (cands : List Name) : List Name :=
  let rec go (todo : List Name) (seen : NameSet) : NameSet :=
    match todo with
    | [] => seen
    | n :: rest =>
      if seen.contains n then go rest seen
      else match IR.findEnvDecl env n with
        | some d =>
          let used := (IR.collectUsedDecls env [d]).toList.filter fun c => root.isPrefixOf c || cands.contains c
          go (used ++ rest) (seen.insert n)
        | none => go rest (seen.insert n)
  let seen := go [root] {}
  cands.filter seen.contains

/-- The overlap pass compiled `check` calls, and its comparisons on `Cost.window m 3`. -/
private def overlapLine (pass : List Name) (c : Wiring) : String :=
  let ok := match check c with | .ok _ => "accepts" | .error _ => "refuses"
  let cost := match pass with
    | [``pairwiseDisjoint] => s!"{Cost.pairwise c.wires} wire comparisons"
    | [``overlapFree] =>
      let (s, w) := Cost.overlapFree c.wires
      s!"{s} sort comparisons and {w} wire comparisons"
    | _ => "no count"
  s!"n = {c.wires.length}, check {ok}: compiled check calls {pass}, {cost}"

/--
info: n = 64, check accepts: compiled check calls [Spot.Wires.overlapFree], 312 sort comparisons and 48 wire comparisons
---
info: n = 128, check accepts: compiled check calls [Spot.Wires.overlapFree], 755 sort comparisons and 96 wire comparisons
---
info: n = 256, check accepts: compiled check calls [Spot.Wires.overlapFree], 1716 sort comparisons and 192 wire comparisons
-/
#guard_msgs in
#eval show Elab.Command.CommandElabM Unit from do
  let pass := compiledCalls (← getEnv) ``check [``pairwiseDisjoint, ``overlapFree]
  for m in [16, 32, 64] do
    logInfo (overlapLine pass (Cost.window m 3))

end Spot.Wires
