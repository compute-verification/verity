/-
What the compiled wire check (`Spot.Wires.check`) costs, as a count of comparisons and list steps, never a clock.

A pure function's comparisons can't be counted from inside its compiled code without changing its term. So the count
has two halves:

* **Which passes run**: the overlap and order passes that `check`'s callers' compiled code reaches, read from the IR
  (`compiledCalls`) of the `wires` piece, a caller compiled after `Spot.Wires`, through the declarations of
  `Spot.Wires` it calls; a `@[csimp]` lemma changes the IR, not the term.
* **What each pass costs**: a counting twin of each pass, the same recursion returning the number of comparisons or list
  steps it makes:
  - the overlap pass: `disjoint` calls, every pair for `pairwiseDisjoint` (`Cost.pairwise`), and for `overlapFree` the
    sort's comparisons (`Cost.sort`, of `List.mergeSort` as defined; the compiler runs it as `mergeSortTR₂`, of the same
    splits and merges) beside the sweep's (`Cost.sweep`);
  - the order pass: for `orderedWire` the ranges it visits, the whole order per wire and again per range the wire meets
    (`Cost.orderedWire`); for `orderedIdx` the index's sort comparisons (`Cost.index`), and per wire the bisections'
    probes and the ranges between their ends (`Cost.orderedIdx`);
  - the lookups by name, the entries `List.lookup` passes over: per wire `shape`'s and `rowWords`' (with `cutsOf`'s),
    and `orderOk`'s, per range and per template (`Cost.lookups`).

Two instances, each one `check` accepts. `Cost.window m r` is shaped like a served window's wiring for the overlap pass: a
reader port cut into `r` rows, one wire a unit and row, beside an uncut port with one wire a unit, `n = m · (r + 1)`
wires. `Cost.chain t m` is shaped like its order: `t` templates of `m` units, each unit read from the same unit of the
template before by a wire of its own, and each unit its own range of the order, `W = (t − 1) · m` wires and `R = t · m`
ranges.
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
def merge {α : Type} (le : α → α → Bool) : List α → List α → Nat × List α
  | [], ys => (0, ys)
  | xs, [] => (0, xs)
  | x :: xs, y :: ys =>
    if le x y then let r := merge le xs (y :: ys); (r.1 + 1, x :: r.2)
    else let r := merge le (x :: xs) ys; (r.1 + 1, y :: r.2)

/-- The comparisons `List.mergeSort l le` makes, as defined (halves of `(n + 1) / 2` and `n / 2`), with its result. -/
def sort {α : Type} (le : α → α → Bool) (l : List α) : Nat × List α :=
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

/-- The ranges `orderedWire o w` visits: the whole order, and again for each range meeting the wire's reader units. -/
def orderedWire (o : List UnitRange) (w : Wire) : Nat :=
  o.length + (o.filter w.readsIn).length * o.length

/-- The probes `Order.firstTrue q lo hi` makes. -/
def firstTrue (q : Nat → Bool) (lo hi : Nat) : Nat :=
  if lo < hi then
    1 + if q ((lo + hi) / 2) then firstTrue q lo ((lo + hi) / 2) else firstTrue q ((lo + hi) / 2 + 1) hi
  else 0
termination_by hi - lo

/-- The steps `Order.allMeeting a t lo hi` takes, both bisections' probes and the ranges between their ends, with the
ranges among those that meet the units. -/
def allMeeting (a : Array Order.Ranged) (t : String) (lo hi : Nat) : Nat × List Order.Ranged :=
  let ql := Order.pAt a (Order.pastLo t lo)
  let qh := Order.pAt a (Order.pastHi t hi)
  let i := Order.firstTrue ql 0 a.size
  let j := Order.firstTrue qh 0 a.size
  (firstTrue ql 0 a.size + firstTrue qh 0 a.size + (j - i),
    ((List.range' i (j - i)).filterMap (a[·]?)).filter (Order.meets t lo hi))

/-- The steps `orderedIdx a w` takes: its reader ranges' `allMeeting`, and its writer ranges' for each. -/
def orderedIdx (a : Array Order.Ranged) (w : Wire) : Nat :=
  let r := allMeeting a w.reader w.start (w.start + w.readerUnits)
  r.1 + (r.2.map fun b =>
    let s := w.box (w.indicesIn b.p0 b.n).1 (w.indicesIn b.p0 b.n).2
    (allMeeting a w.writer s.1.toNat (s.2.toNat + 1)).1).sum

/-- The comparisons `Order.index o` makes: its sort's. -/
def index (o : List UnitRange) : Nat :=
  (sort Order.keyLe (o.zipIdx.map fun ((t, p0, n), k) => (t, p0, n, k))).1

/-- The entries `l.lookup k` passes over. -/
def lookup {β : Type} (l : List (String × β)) (k : String) : Nat :=
  match l.findIdx? (·.1 == k) with
  | some i => i + 1
  | none => l.length

/-- The entries `c.cutsOf t p` passes over. -/
def cutsOf (c : Wiring) (t p : String) : Nat := lookup c.cuts t + lookup ((c.cuts.lookup t).getD []) p

/-- The entries `shape c w` and `rowWords c w` pass over, for a wire whose templates, port and output are the circuit's. -/
def wireLookups (c : Wiring) (w : Wire) : Nat :=
  let rt := (c.templates.lookup w.reader).getD { units := 0, rows := [], outs := [] }
  let wt := (c.templates.lookup w.writer).getD { units := 0, rows := [], outs := [] }
  let shape := lookup c.templates w.reader + lookup c.templates w.writer + lookup rt.rows w.port + lookup wt.outs w.out +
    cutsOf c w.reader w.port
  let rowWords := lookup c.templates w.reader + lookup rt.rows w.port + cutsOf c w.reader w.port
  shape + rowWords

/-- The entries `orderOk c o` passes over: a lookup per range, and per template the whole order and its ranges' sort. -/
def orderOk (c : Wiring) (o : List UnitRange) : Nat :=
  (o.map fun b => lookup c.templates b.1).sum +
    (c.templates.map fun (t, _) =>
      o.length + (sort (fun (a b : Nat × Nat) => a.1 ≤ b.1) ((o.filter (·.1 == t)).map (·.2))).1).sum

/-- The lookups' entries over the wires, and `orderOk`'s. -/
def lookups (c : Wiring) (o : List UnitRange) : Nat × Nat := ((c.wires.map (wireLookups c)).sum, orderOk c o)

/-- A wire of reader `A`'s port `p`: unit `u`, rows `[c0, c0 + 1)`, read from writer `G`'s unit `v`. -/
def wireAt (p : String) (u v c0 : Nat) : Wire :=
  { reader := "A", port := p, writer := "G", out := "o0", start := u, radix := [1], base := v, step := [0], cut0 := c0,
    cuts := 1 }

/-- `l` reordered by `37 · i mod n` for the entry at index `i`, so a sort is not handed sorted input. -/
def scramble {α : Type} (l : List α) : List α :=
  ((l.zipIdx.mergeSort fun a b => (a.2 * 37) % l.length ≤ (b.2 * 37) % l.length)).map (·.1)

/-- `m` reader units, port `k` cut into `r` rows of 2 words with a wire a unit and row, port `q` uncut with a wire a
unit: `m · (r + 1)` wires, scrambled. -/
def window (m r : Nat) : Wiring where
  templates := [("G", { units := m * r + m, rows := [("x", ⟨1, 1⟩)], outs := [("o0", 2)] }),
    ("A", { units := m, rows := [("k", ⟨2 * r, r⟩), ("q", ⟨2, 1⟩)], outs := [("o0", 1)] })]
  cuts := [("A", [("k", r)])]
  wires := scramble (((List.range m).flatMap fun u => (List.range r).map fun c => wireAt "k" u (u * r + c) c) ++
    (List.range m).map fun u => wireAt "q" u (m * r + u) 0)

/-- Template `i` of `chain`. -/
def chainId (i : Nat) : String := s!"T{i}"

/-- `t` templates of `m` units, unit `u` of each after the first reading unit `u` of the one before by a wire of its own,
and each unit its own range of the order, ranked template by template: `(t − 1) · m` wires, `t · m` ranges, both
scrambled. -/
def chain (t m : Nat) : Wiring where
  templates := (List.range t).map fun i => (chainId i, { units := m, rows := [("x", ⟨2, 1⟩)], outs := [("o0", 2)] })
  cuts := []
  wires := scramble ((List.range (t - 1)).flatMap fun i => (List.range m).map fun u =>
    { reader := chainId (i + 1), port := "x", writer := chainId i, out := "o0", start := u, radix := [1], base := u,
      step := [0], cut0 := 0, cuts := 1 })
  order := some ((List.range t).flatMap fun i => (List.range m).map fun u => (chainId i, u, 1))

end Spot.Wires.Cost

namespace Spot.Wires

open Lean

/-- The constants among `cands` that `root`'s compiled code reaches, through the declarations of `Spot.Wires` it calls. -/
private meta partial def compiledCalls (env : Environment) (root : Name) (cands : List Name) : List Name :=
  let rec go (todo : List Name) (seen : NameSet) : NameSet :=
    match todo with
    | [] => seen
    | n :: rest =>
      if seen.contains n then go rest seen
      else match IR.findEnvDecl env n with
        | some d =>
          let used := (IR.collectUsedDecls env [d]).toList.filter fun c => (`Spot.Wires).isPrefixOf c || cands.contains c
          go (used ++ rest) (seen.insert n)
        | none => go rest (seen.insert n)
  let seen := go [root] {}
  cands.filter seen.contains

/-- The overlap pass the compiled `wires` piece reaches, and its comparisons on `c`. -/
private def overlapLine (pass : List Name) (c : Wiring) : String :=
  let ok := match check c with | .ok _ => "accepts" | .error _ => "refuses"
  let cost := match pass with
    | [``pairwiseDisjoint] => s!"{Cost.pairwise c.wires} wire comparisons"
    | [``overlapFree] =>
      let (s, w) := Cost.overlapFree c.wires
      s!"{s} sort comparisons and {w} wire comparisons"
    | _ => "no count"
  s!"n = {c.wires.length}, check {ok}: the compiled wires piece reaches {pass}, {cost}"

/-- The order pass the compiled `wires` piece reaches, its steps on `c` beside `orderedWire`'s, and the lookups'. -/
private def orderLine (pass : List Name) (c : Wiring) : String :=
  let ok := match check c with | .ok _ => "accepts" | .error _ => "refuses"
  let o := c.order.getD []
  let slow := (c.wires.map (Cost.orderedWire o)).sum
  let cost := match pass with
    | [``orderedWire] => s!"{slow} range visits"
    | [``orderedIdx] =>
      let a := Order.index o
      s!"{Cost.index o} index comparisons and {(c.wires.map (Cost.orderedIdx a)).sum} steps ({slow} for orderedWire)"
    | _ => "no count"
  let (wl, ol) := Cost.lookups c o
  s!"t = {c.templates.length}, W = {c.wires.length}, R = {o.length}, check {ok}: the compiled wires piece reaches \
    {pass}, {cost}; lookups {wl} over the wires, {ol} in orderOk"

/--
info: n = 64, check accepts: the compiled wires piece reaches [Spot.Wires.overlapFree], 312 sort comparisons and 48 wire comparisons
---
info: n = 128, check accepts: the compiled wires piece reaches [Spot.Wires.overlapFree], 755 sort comparisons and 96 wire comparisons
---
info: n = 256, check accepts: the compiled wires piece reaches [Spot.Wires.overlapFree], 1716 sort comparisons and 192 wire comparisons
-/
#guard_msgs in
#eval show Elab.Command.CommandElabM Unit from do
  let pass := compiledCalls (← getEnv) ``piece [``pairwiseDisjoint, ``overlapFree]
  for m in [16, 32, 64] do
    logInfo (overlapLine pass (Cost.window m 3))

/--
info: t = 4, W = 48, R = 64, check accepts: the compiled wires piece reaches [Spot.Wires.orderedIdx], 192 index comparisons and 1251 steps (6144 for orderedWire); lookups 528 over the wires, 544 in orderOk
---
info: t = 4, W = 192, R = 256, check accepts: the compiled wires piece reaches [Spot.Wires.orderedIdx], 1024 index comparisons and 6531 steps (98304 for orderedWire); lookups 2112 over the wires, 2432 in orderOk
---
info: t = 16, W = 240, R = 256, check accepts: the compiled wires piece reaches [Spot.Wires.orderedIdx], 1152 index comparisons and 8163 steps (122880 for orderedWire); lookups 6960 over the wires, 6784 in orderOk
-/
#guard_msgs in
#eval show Elab.Command.CommandElabM Unit from do
  let pass := compiledCalls (← getEnv) ``piece [``orderedWire, ``orderedIdx]
  for (t, m) in [(4, 16), (4, 64), (16, 16)] do
    logInfo (orderLine pass (Cost.chain t m))

end Spot.Wires
