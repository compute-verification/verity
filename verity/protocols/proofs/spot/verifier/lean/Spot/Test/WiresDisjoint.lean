module

public import Spot.Wires
meta import Spot.Wires

/-!
`pairwiseDisjoint`'s compiled code, `pairwiseDisjointFast`, held to the pairwise test on random and adversarial lists
(nested, equal starts, touching ends, empty units or rows, other readers and ports, row cuts), and its work at 700k wires:
the live wires it compares, counted by `sweepWork`, at most a small multiple of the wires, where the pairwise test makes
W(W−1)/2 comparisons. `kahn`'s order, the verifier's output, held to the loop it replaced (`kahnRef`: the first
template by index not placed and reading none not placed, found by a scan at each step) on random wirings, acyclic and
cyclic, with repeated template ids, and on a served class's shape: the same order or the same refusal. `check` and
`orderOk`, compiled with their maps (`check_eq_fast`, `orderOk_eq_fast`), held to their bodies as written (`checkOld`,
`orderOkOld`) on random wirings that reach every refusal, with stated orders and repeated ids; and the whole check on
700k wires of the served shape, its order's work counted (`orderWork`: each wire's ranges of its reader, and of its
writer in each reader range it meets).
-/

@[expose] public section

namespace Spot.Test.WiresDisjoint

open Spot.Wires

/-- The pairwise test as written, kept out of `@[csimp]`'s reach as the reference. -/
def pairwiseRef : List Wire → Bool
  | [] => true
  | w :: ws => ws.all (disjoint w) && pairwiseRef ws

/-- `sweep`'s comparisons: the live wires it filters and compares, wire by wire. -/
def sweepWork : List Wire → List Wire → Nat → Nat
  | _, [], n => n
  | live, w :: ws, n =>
    let n := n + live.length
    let live := live.filter fun a => a.reader == w.reader && a.port == w.port && w.start < a.start + a.readerUnits
    if live.all (disjoint w) then sweepWork (w :: live) ws (n + live.length) else n + live.length

/-- A wire of reader `r`'s port `p` over units `[s, s + n)` and rows `[c0, c0 + c)`. -/
def wire (r p : String) (s n c0 c : Nat) : Wire :=
  { reader := r, port := p, writer := "w", out := "o", start := s, radix := if c ≤ 1 then [n] else [n, c], base := 0,
    step := if c ≤ 1 then [1] else [1, 0], cut0 := c0, cuts := c }

#guard (wire "r" "p" 3 4 0 1).readerUnits == 4 && (wire "r" "p" 3 4 2 3).readerUnits == 4

/-- A deterministic stream of naturals (a 64-bit LCG). -/
def lcg (x : Nat) : Nat := (x * 6364136223846793005 + 1442695040888963407) % 2 ^ 64

/-- `n` wires from seed `x`, each field drawn below its bound: few readers and ports, short ranges, so pairs meet. -/
def randomWires (x n units rows : Nat) : List Wire :=
  (List.range n).foldl (fun (acc, x) _ =>
    let x₁ := lcg x; let x₂ := lcg x₁; let x₃ := lcg x₂; let x₄ := lcg x₃; let x₅ := lcg x₄; let x₆ := lcg x₅
    (wire (toString (x₁ / 2 ^ 40 % 2)) (toString (x₂ / 2 ^ 40 % 2)) (x₃ / 2 ^ 40 % units) (x₄ / 2 ^ 40 % 4)
      (x₅ / 2 ^ 40 % rows) (x₆ / 2 ^ 40 % 3) :: acc, x₆)) ([], x) |>.1

def agree (ws : List Wire) : Bool := pairwiseDisjointFast ws == pairwiseRef ws

/-- Adversarial lists, each with its verdict under the pairwise test. -/
def cases : List (List Wire × Bool) := [
  ([], true),
  ([wire "r" "p" 0 4 0 1], true),
  ([wire "r" "p" 0 10 0 1, wire "r" "p" 2 3 0 1], false),
  ([wire "r" "p" 0 10 0 1, wire "r" "p" 1 1 0 1, wire "r" "p" 5 1 0 1], false),
  ([wire "r" "p" 0 100 0 1, wire "r" "p" 10 1 1 1, wire "r" "p" 99 1 0 1], false),
  ([wire "r" "p" 5 2 0 1, wire "r" "p" 5 3 0 1], false),
  ([wire "r" "p" 5 2 0 1, wire "r" "p" 5 3 1 1], true),
  ([wire "r" "p" 0 4 0 1, wire "r" "p" 4 4 0 1, wire "r" "p" 8 4 0 1], true),
  ([wire "r" "p" 8 4 0 1, wire "r" "p" 0 4 0 1, wire "r" "p" 4 4 0 1], true),
  ([wire "r" "p" 0 4 0 2, wire "r" "p" 0 4 2 2, wire "r" "p" 0 4 1 2], false),
  ([wire "r" "p" 0 4 0 2, wire "r" "p" 0 4 2 2, wire "r" "p" 4 4 1 2], true),
  ([wire "r" "p" 0 4 0 1, wire "r" "q" 0 4 0 1, wire "s" "p" 0 4 0 1], true),
  ([wire "r" "p" 0 4 0 1, wire "s" "p" 0 9 0 1, wire "r" "p" 3 1 0 1], false),
  ([wire "r" "p" 0 0 0 1, wire "r" "p" 0 4 0 1], true),
  ([wire "r" "p" 2 0 0 1, wire "r" "p" 0 4 0 1], false),
  ([wire "r" "p" 2 0 0 1, wire "r" "p" 2 0 0 1], true),
  ([wire "r" "p" 0 4 0 0, wire "r" "p" 0 4 0 1], true),
  ([wire "r" "p" 0 1000 0 1] ++ (List.range 50).map (fun i => wire "r" "p" (2000 + i) 1 0 1) ++
    [wire "r" "p" 999 1 0 1], false),
  ((List.range 50).map (fun i => wire "r" "p" (100 - 2 * i) (4 * i + 1) 0 1), false),
  ((List.range 50).map (fun i => wire "r" "p" (2 * i) 2 0 1), true)]

#guard cases.all fun (ws, v) => pairwiseRef ws == v && pairwiseDisjointFast ws == v

/-- Each random list's seed, length, units and rows. -/
def draws : List (Nat × Nat × Nat × Nat) :=
  (List.range 400).map fun i => (i, 2 + i % 12, 4 + i % 40, 1 + i % 4)

#guard draws.all fun (x, n, u, r) => agree (randomWires x n u r)
#guard (draws.filter fun (x, n, u, r) => pairwiseRef (randomWires x n u r)).length ≥ 20
#guard (draws.filter fun (x, n, u, r) => !pairwiseRef (randomWires x n u r)).length ≥ 20

/-- A served class's shape at 700k wires: 1,000 readers of 7 ports, each port cut into two rows and read by 80 wires over
consecutive units, every fourth one's two rows read by a wire each (100 wires a port), the list in reverse of its sorted
order. -/
def served : List Wire :=
  ((List.range 1000).flatMap fun r => (List.range 7).flatMap fun p => (List.range 80).flatMap fun i =>
    let rd := s!"t{r}"; let pt := s!"p{p}"
    if i % 4 == 0 then [wire rd pt (4 * i) 4 0 1, wire rd pt (4 * i) 4 1 1] else [wire rd pt (4 * i) 4 0 2]).reverse

#guard served.length == 700000
#guard pairwiseDisjointFast served
#guard sweepWork [] (served.mergeSort before) 0 ≤ 4 * served.length
#guard !pairwiseDisjointFast (wire "t5" "p3" 41 2 0 1 :: served)

/-! ## `kahn` against the scan it replaced -/

/-- `kahn`'s loop as it was: a scan of every template at each step. -/
def kahnLoopRef (ids : Array String) (ranges : String → List UnitRange) (outs : Array (List Nat))
    (ins : Array (List (Nat × Nat))) : Nat → Array Bool → Array Nat → List UnitRange → Except Refusal (List UnitRange)
  | 0, _, _, acc => .ok acc.reverse
  | fuel + 1, done, indeg, acc =>
    match (List.range ids.size).find? (fun i => !done[i]! && indeg[i]! == 0) with
    | some i => kahnLoopRef ids ranges outs ins fuel (done.set! i true) ((outs[i]!).foldl (fun a r => a.modify r (· - 1)) indeg)
        ((ranges ids[i]!).reverse ++ acc)
    | none =>
      let rest := (List.range ids.size).filter (!done[·]!)
      let t := kahnWalk ins done rest.length (rest.headD 0)
      .error ⟨.cycle, ((ins[t]!).find? (fun (_, wr) => !done[wr]!)).map (·.1)⟩

/-- `kahn` as it was, every template's self-wires and units found by a scan. -/
def kahnRef (c : Wiring) : Except Refusal (List UnitRange) :=
  let ids := ((c.templates.map (·.1)).mergeSort (· ≤ ·)).toArray
  let idx : Std.HashMap String Nat := ids.zipIdx.foldl (fun m (t, i) => m.insert t i) {}
  let es := c.wires.zipIdx.filterMap fun (w, j) =>
    if w.writer == w.reader then none else do pure (j, ← idx[w.writer]?, ← idx[w.reader]?)
  let indeg := es.foldl (fun a (_, _, r) => a.modify r (· + 1)) (Array.replicate ids.size 0)
  let outs := es.foldr (fun (_, wr, r) a => a.modify wr (r :: ·)) (Array.replicate ids.size [])
  let ins := es.foldr (fun (j, wr, r) a => a.modify r ((j, wr) :: ·)) (Array.replicate ids.size [])
  kahnLoopRef ids (fun t => selfRanges c.wires t (((c.templates.lookup t).map (·.units)).getD 0)) outs ins ids.size
    (Array.replicate ids.size false) indeg []

def sameOrder : Except Refusal (List UnitRange) → Except Refusal (List UnitRange) → Bool
  | .ok a, .ok b => a == b
  | .error a, .error b => a == b
  | _, _ => false

def tmpl (units : Nat) : Template := { units, rows := [("p", { words := 4, outer := 1 })], outs := [("o", 4)] }

/-- A wiring of `nt` templates (ids out of order, the last repeating the first when `dup`) and `nw` wires from seed `x`;
`acyclic`: each wire's writer at or before its reader in the list. -/
def randomWiring (x nt nw : Nat) (acyclic dup : Bool) : Wiring :=
  let names := (List.range nt).map fun i => s!"t{(i * 7 + 3) % nt}"
  let ts := names.map fun n => (n, tmpl 8)
  let ts := if dup then ts ++ [(names.headD "t0", tmpl 8)] else ts
  let ws := (List.range nw).foldl (fun (acc, x) _ =>
    let x₁ := lcg x; let x₂ := lcg x₁; let x₃ := lcg x₂; let x₄ := lcg x₃
    let r := x₁ / 2 ^ 40 % nt
    let wr := if acyclic then x₂ / 2 ^ 40 % (r + 1) else x₂ / 2 ^ 40 % nt
    ({ wire (names.getD r "") "p" (x₃ / 2 ^ 40 % 6) (1 + x₄ / 2 ^ 40 % 2) 0 1 with writer := names.getD wr "" } :: acc, x₄))
    ([], x) |>.1
  { templates := ts, cuts := [], wires := ws }

def kahnDraws : List (Nat × Nat × Nat × Bool × Bool) :=
  (List.range 300).map fun i => (i, 1 + i % 7, i % 13, i % 3 != 0, i % 5 == 0)

#guard kahnDraws.all fun (x, nt, nw, acyc, dup) => let c := randomWiring x nt nw acyc dup; sameOrder (kahn c) (kahnRef c)
#guard (kahnDraws.filter fun (x, nt, nw, acyc, dup) => (kahn (randomWiring x nt nw acyc dup)).toBool).length ≥ 50
#guard (kahnDraws.filter fun (x, nt, nw, acyc, dup) => !(kahn (randomWiring x nt nw acyc dup)).toBool).length ≥ 20

/-- `T` templates of `L` layers, two units a root node, three ports cut into 17 rows; port 0 reads its own template's
previous layer, ports 1 and 2 the template before it (template 0: its own previous layer): 51 wires a root node. -/
def servedWiring (T L : Nat) : Wiring :=
  let port : Port := { words := 68, outer := 17 }
  let tm : Template := { units := 2 * L, rows := [("p0", port), ("p1", port), ("p2", port)], outs := [("o", 68)] }
  let tn (j : Nat) : String := s!"t{1000000 + j}"
  let wires := (List.range T).flatMap fun j => (List.range L).flatMap fun l => (List.range 17).flatMap fun r =>
    let mk (p : String) (wr b : Nat) : Wire :=
      { reader := tn j, port := p, writer := tn wr, out := "o", start := 2 * l, radix := [2], base := b, step := [1],
        cut0 := r, cuts := 1, offset := 4 * r }
    let self := if l > 0 then [mk "p0" j (2 * (l - 1))] else []
    let prev := if j > 0 then [mk "p1" (j - 1) (2 * l), mk "p2" (j - 1) (2 * l)]
      else if l > 0 then [mk "p1" j (2 * (l - 1)), mk "p2" j (2 * (l - 1))] else []
    self ++ prev
  { templates := ((List.range T).map fun j => (tn j, tm)).reverse,
    cuts := (List.range T).map fun j => (tn j, [("p0", 17), ("p1", 17), ("p2", 17)]), wires }

#guard let c := servedWiring 40 12; sameOrder (kahn c) (kahnRef c) && (kahn c).toBool

/-! ## `check` and `orderOk` against their bodies as written -/

def orderOkOld (c : Wiring) (o : List UnitRange) : Bool :=
  o.all (fun b => (c.templates.lookup b.1).isSome) &&
    c.templates.all fun (t, tt) =>
      tiles (((o.filter (·.1 == t)).map (·.2)).mergeSort fun a b => a.1 ≤ b.1) 0 tt.units

def orderOfOld (c : Wiring) : Except Refusal (List UnitRange) :=
  match statedOrder c with
  | .ok o => if orderOkOld c o then .ok o else .error ⟨.order, none⟩
  | .error r => .error r

def checkOld (c : Wiring) : Except Refusal Summary :=
  if !cutsOk c then .error ⟨.cuts, none⟩
  else match firstWire c (wireCode c) with
    | some r => .error r
    | none =>
      if !pairwiseRef c.wires then .error ⟨.overlap, none⟩
      else match (match orderOfOld c with
          | .error r => some r
          | .ok o => firstWire c fun w => if orderedWire o w then none else some .order) with
        | some r => .error r
        | none => .ok ⟨c.wires, c.wires.map (rowWords c)⟩

def sameCheck : Except Refusal Summary → Except Refusal Summary → Bool
  | .ok a, .ok b => a.wires.length == b.wires.length && a.rowWords == b.rowWords
  | .error a, .error b => a == b
  | _, _ => false

def code : Except Refusal Summary → String
  | .ok _ => "ok"
  | .error r => r.code.name

/-- A wiring from seed `x`: `nt` templates of ports `p`, `q` (cut in 1 or 2) and output `o`, the first repeated when
`dup`; `nw` wires, mostly well formed, a writer mostly at or before its reader, some naming no template or port, rows or widths past the port's, or units past the
template's; a stated order of the templates' units in random pieces and ranks when `stated`. -/
def checkWiring (x nt nw : Nat) (stated dup : Bool) : Wiring :=
  let names := (List.range nt).map fun i => s!"t{(i * 7 + 3) % nt}"
  let pp : Port := { words := 4, outer := 2 }
  let pq : Port := { words := 2, outer := 2 }
  let tm : Template := { units := 8, rows := [("p", pp), ("q", pq)], outs := [("o", 4)] }
  let ts := names.map fun n => (n, tm)
  let ts := if dup then ts ++ [(names.headD "t0", { tm with units := 4 })] else ts
  let cuts := names.filterMap fun n => if (n.length + x) % 2 == 0 then some (n, [("p", 2)]) else none
  let (ws, x) := (List.range nw).foldl (fun (acc, x) _ =>
    let x₁ := lcg x; let x₂ := lcg x₁; let x₃ := lcg x₂; let x₄ := lcg x₃; let x₅ := lcg x₄; let x₆ := lcg x₅
    let d (y m : Nat) := y / 2 ^ 40 % m
    let r := d x₁ nt; let wr := if d x₆ 5 == 3 then d x₂ nt else d x₂ (r + 1)
    let reader := if d x₆ 53 == 0 then "nope" else names.getD r ""
    let port := if d x₆ 29 == 1 then "z" else if d x₅ 2 == 0 then "p" else "q"
    let n := 1 + d x₄ 3
    ({ reader, port, writer := names.getD wr "", out := if d x₆ 31 == 2 then "z" else "o", start := d x₃ 8,
       radix := [n], base := (d x₅ 6 : Int), step := [1], cut0 := if d x₆ 7 == 0 then 1 else 0, cuts := 1,
       offset := d x₂ 3 } :: acc, x₆)) ([], x)
  let order := if stated then
      some (names.flatMap fun n => let k := (lcg (x + n.length)) / 2 ^ 40 % 3
        if k == 0 then [(n, 0, 8)] else if k == 1 then [(n, 0, 4), (n, 4, 4)] else [(n, 0, 3), (n, 4, 4)])
    else none
  let order := order.map fun o => (o.zipIdx.map fun (b, i) => (lcg (x + i) / 2 ^ 40 % 1000, b)).mergeSort
    (fun a b => a.1 ≤ b.1) |>.map (·.2)
  { templates := ts, cuts, wires := ws, order }

def checkDraws : List (Nat × Nat × Nat × Bool × Bool) :=
  (List.range 600).map fun i => (i, 1 + i % 5, i % 9, i % 2 == 0, i % 7 == 0)

#guard checkDraws.all fun (x, nt, nw, st, dup) => let c := checkWiring x nt nw st dup; sameCheck (check c) (checkOld c)
#guard checkDraws.all fun (x, nt, nw, st, dup) => let c := checkWiring x nt nw st dup
  match c.order with | some o => orderOk c o == orderOkOld c o | none => true
def codes : List String := checkDraws.map fun (x, nt, nw, st, dup) => code (check (checkWiring x nt nw st dup))

#guard ["ok", "template", "port", "cut", "width", "bounds", "overlap", "order"].all fun k =>
  (codes.filter (· == k)).length ≥ 3

/-- Two templates reading each other: `kahn`'s cycle. -/
def cyclic : Wiring :=
  let c := checkWiring 0 2 0 false false
  { c with wires := [{ wire "t1" "p" 0 1 0 1 with writer := "t0" }, { wire "t0" "p" 0 1 0 1 with writer := "t1" }] }

#guard code (check cyclic) == "cycle" && sameCheck (check cyclic) (checkOld cyclic)

/-- `checkFast`'s order work: each wire's ranges of its reader, and of its writer in each reader range it meets. -/
def orderWork (c : Wiring) : Nat :=
  match orderOf c with
  | .error _ => 0
  | .ok o =>
    let g := groupBy (·.1.1) o.zipIdx
    c.wires.foldl (fun n w =>
      let rs := g.getD w.reader []
      n + rs.length + ((rs.filter fun (b, _) => w.readsIn b).length) * (g.getD w.writer []).length) 0

/-- The served shape at 696,966 wires: 100 templates of 137 layers, 13,700 root nodes. The file takes
about seven minutes under the interpreter: a stage quadratic in the wires, templates or ranges would not return, which
no smaller wiring shows. -/
def big : Wiring := servedWiring 100 137

#guard big.wires.length == 696966
#guard (check big).toBool
#guard orderWork big ≤ 300 * big.wires.length
#guard !(check { big with wires := { big.wires.headD (wire "" "" 0 0 0 1) with start := 3 } :: big.wires }).toBool

end Spot.Test.WiresDisjoint
