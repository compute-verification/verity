module

public import PouwPearlPlus.Records

/-!
# The run: its commitment, the verifier's coin record, and the tiles it opens (`verify_passes`)

* **A pass** (`Pass`) is what the verifier holds once a retained pass's manifest is read and held to what it published
  (`Load`): its epoch's salt and weights' roots, its records, and the roots it published (activations, transcript, records,
  each call's y).
* **The run's commitment** (`runCommitment`) binds the retained passes and their records, before the coin:
  `tagged("verity/pouw/pearl-plus/run/v1", per pass: transcript root, records root, salt, each call's y root or empty)`.
* **The coin record** (`CoinRecord`, `verity-pouw/pearl-plus-coin/v2`; v1, which names no draw, read for development) is
  the verifier's record of its coin against that commitment, and of where the coin came from (its `draw`, `Coins`). A
  verdict over passes whose commitment is another, or with another source, is refused (`after-coin`): neither a retention
  nor a give-up can follow the coin, and a retry never redraws.
* **The draw** keys `t` work-weighted draws over every pass's tickets together on the coin
  (`derive(source, "verity/pouw/pearl-plus-run-draw/v1", {run})`, `clear_work.draw`). A tile's ticket is its credited
  cells' share of `wref` at |K| (`clear_work.ticket_cell`, the scheme's own `wref` on the H100), scaled to integers by the
  lcm of the denominators.
* **The exclusion tiles**: each pass opens one tile per excluded row, an A row's in its first column block and a B row's in
  the first row block of its weight's first unit (`clear_work.exclusion_tiles`).
-/

@[expose] public section

namespace PouwCheck.PearlPlus

open PouwCommit

def PROTOCOL : String := "verity/pouw/pearl-plus/v1"
def RUN_TAG : String := "verity/pouw/pearl-plus/run/v1"
def RUN_DRAW_DOMAIN : String := "verity/pouw/pearl-plus-run-draw/v1"
def COIN_SCHEMA : String := "verity-pouw/pearl-plus-coin/v2"
def COIN_SCHEMA_V1 : String := "verity-pouw/pearl-plus-coin/v1"
/-- The named assumption a run of the H100's scheme is credited at (`verify_run.ASSUMPTION_H100`). -/
def ASSUMPTION : String := "Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1"

/-- One retained pass as the verifier holds it once its manifest is read and checked (`Load.load`). -/
structure Pass where
  scheme : Scheme
  salt : ByteArray
  weightRoots : List (Nat × ByteArray)
  weightsRoot : ByteArray
  records : List UnitRecord
  rootsA : List ByteArray
  transcriptRoot : ByteArray
  count : Nat
  recordsRoot : ByteArray
  yRoots : List (Option ByteArray)
  deriving Inhabited

namespace Pass

/-- Each record's first tile id: tiles are numbered over the pass's matmuls in order, each matmul's row-major. -/
def offsets (p : Pass) : List Nat :=
  (p.records.foldl (fun (acc : List Nat × Nat) r => (acc.1 ++ [acc.2], acc.2 + (p.scheme.tiles r.matmul.m r.matmul.n).length))
    ([], 0)).1

/-- The tile with id `tid`: (record position, its rows, its columns). -/
def tile? (p : Pass) (tid : Nat) : Option (Nat × List Nat × List Nat) := Id.run do
  let mut base := 0
  for (r, pos) in p.records.zipIdx do
    let grid := p.scheme.tiles r.matmul.m r.matmul.n
    if tid < base + grid.length then
      let (rows, cols) := grid[tid - base]!
      return some (pos, rows, cols)
    base := base + grid.length
  return none

/-- The number of tiles over the pass's workload (`len(Verifier.refs)`). -/
def tileCount (p : Pass) : Nat := p.records.foldl (fun n r => n + (p.scheme.tiles r.matmul.m r.matmul.n).length) 0

/-- `clear_work.exclusion_tiles` as tile ids, in its order, each once. -/
def exclusionTiles (p : Pass) : List Nat := Id.run do
  let offs := p.offsets
  let mut out : Array Nat := #[]
  let mut first : List (Nat × Nat) := []
  for (r, pos) in p.records.zipIdx do
    let u := r.matmul
    if (first.lookup u.weight).isNone then first := (u.weight, pos) :: first
    let grid := (p.scheme.tiles u.m u.n).zipIdx
    let add (out : Array Nat) (t : Option ((List Nat × List Nat) × Nat)) : Array Nat :=
      match t with
      | some (_, x) => let id := offs[pos]! + x; if out.contains id then out else out.push id
      | none => out
    for i in r.excludedA do
      out := add out (grid.find? fun ((rows, _), _) => rows.head! ≤ i && i ≤ rows.getLast!)
    if first.lookup u.weight == some pos then
      for j in r.excludedB do
        out := add out (grid.find? fun ((_, cols), _) => cols.head! ≤ j && j ≤ cols.getLast!)
  return out.toList

end Pass

/-- `run_commitment`: each retained pass's transcript root, records root and salt, then its calls' y roots (empty where a
call commits none), in order. -/
def runCommitment (passes : List Pass) : ByteArray :=
  tagged RUN_TAG (passes.flatMap fun p =>
    [.bytes p.transcriptRoot, .bytes p.recordsRoot, .bytes p.salt] ++ p.yRoots.map fun y => .bytes (y.getD ByteArray.empty))

/-- The verifier's record of its coin: the run commitment it was drawn on, the source (a challenger draw coin's 64 bytes,
or the node's 32), the record's schema and its `draw` as the record carries it (`Coins.drawReason` reads it). -/
structure CoinRecord where
  commitment : ByteArray
  source : ByteArray
  schema : String := COIN_SCHEMA
  draw : Lean.Json := .null

def AFTER_COIN : String × String :=
  ("after-coin", "the retained passes or their records are not those the verifier's coin was drawn on: a retention change " ++
    "or a give-up after the coin is refused, and a retry never redraws")

/-- A tile's ticket weight as a fraction `num/den` in lowest terms: |credited rows|·|credited columns| cells at the cell's
share of `wref(1, |K|, n)`, over n. -/
def ticketWeight (r : UnitRecord) (rows cols : List Nat) : Nat × Nat :=
  let (cr, cc) := r.credited rows cols
  let x := wref 1 r.kKept r.matmul.n * cr.length * cc.length
  let g := Nat.gcd x r.matmul.n
  (x / g, r.matmul.n / g)

/-- Every tile of every pass, (pass, tile id), with its integer ticket (`clear_work.tickets` over the passes' records
concatenated). -/
def tickets (passes : List Pass) : List ((Nat × Nat) × Nat) :=
  let ws := passes.zipIdx.flatMap fun (p, i) =>
    let offs := p.offsets
    p.records.zipIdx.flatMap fun (r, pos) =>
      (p.scheme.tiles r.matmul.m r.matmul.n).zipIdx.map fun ((rows, cols), x) => ((i, offs[pos]! + x), ticketWeight r rows cols)
  let den := ws.foldl (fun l (_, (_, d)) => Nat.lcm l d) 1
  ws.map fun (id, (n, d)) => (id, n * (den / d))

/-- The run's draw key on the verifier's source. -/
def drawKey (source root : ByteArray) : Except String ByteArray := key? source RUN_DRAW_DOMAIN [("run", .bytes root)]

/-- `t` work-weighted draws over every pass's tickets together, each a (pass, tile id), in the draw's order. -/
def draw (passes : List Pass) (key : ByteArray) (t : Nat) : Except String (List (Nat × Nat)) := do
  let tk := tickets passes
  let picks ← pearlDraw key (tk.map (·.2)) t
  pure (picks.map fun i => (tk[i]!).1)

/-- The tiles the verdict opens, (pass, tile id), in its order: each pass's drawn tiles in the draw's order, then its
exclusion tiles. -/
def todo (passes : List Pass) (drawn : List (Nat × Nat)) : List (Nat × Nat) :=
  passes.zipIdx.flatMap fun (p, i) =>
    (drawn.filter (·.1 == i)) ++ p.exclusionTiles.map fun tid => (i, tid)

end PouwCheck.PearlPlus
