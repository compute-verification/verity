/-
What the compiled wire check (`Spot.Wires.check`) costs, as a count of comparisons, never a clock.

A pure function's comparisons can't be counted from inside its compiled code without changing its term. So the count
has two halves:

* **Which pass runs**: the overlap pass `check`'s compiled code calls, read from its IR (`compiledCalls`), through its own
  auxiliary declarations; a `@[csimp]` lemma changes the IR, not the term.
* **What that pass costs**: a counting twin of each pass, the same recursion returning the number of comparisons it
  makes: `disjoint` calls for the overlap pass (`Cost.pairwise`).

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

/-- A wire of reader `A`'s port `p`: unit `u`, rows `[c0, c0 + 1)`, read from writer `G`'s unit `v`. -/
def wireAt (p : String) (u v c0 : Nat) : Wire :=
  { reader := "A", port := p, writer := "G", out := "o0", start := u, radix := [1], base := v, step := [0], cut0 := c0,
    cuts := 1 }

/-- `m` reader units, port `k` cut into `r` rows of 2 words with a wire a unit and row, port `q` uncut with a wire a
unit: `m · (r + 1)` wires. -/
def window (m r : Nat) : Wiring where
  templates := [("G", { units := m * r + m, rows := [("x", ⟨1, 1⟩)], outs := [("o0", 2)] }),
    ("A", { units := m, rows := [("k", ⟨2 * r, r⟩), ("q", ⟨2, 1⟩)], outs := [("o0", 1)] })]
  cuts := [("A", [("k", r)])]
  wires := ((List.range m).flatMap fun u => (List.range r).map fun c => wireAt "k" u (u * r + c) c) ++
    (List.range m).map fun u => wireAt "q" u (m * r + u) 0

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

/--
info: n = 64, check accepts: compiled check calls [Spot.Wires.pairwiseDisjoint], 2016 wire comparisons
---
info: n = 128, check accepts: compiled check calls [Spot.Wires.pairwiseDisjoint], 8128 wire comparisons
---
info: n = 256, check accepts: compiled check calls [Spot.Wires.pairwiseDisjoint], 32640 wire comparisons
-/
#guard_msgs in
#eval show Elab.Command.CommandElabM Unit from do
  let pass := compiledCalls (← getEnv) ``check [``pairwiseDisjoint]
  for m in [16, 32, 64] do
    let c := Cost.window m 3
    let ok := match check c with | .ok _ => "accepts" | .error _ => "refuses"
    logInfo s!"n = {c.wires.length}, check {ok}: compiled check calls {pass}, {Cost.pairwise c.wires} wire comparisons"

end Spot.Wires
