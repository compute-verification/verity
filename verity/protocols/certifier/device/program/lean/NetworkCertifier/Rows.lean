module

/-!
# The certifier program: the active network certifier's records as rows, and their audit

The executable form of `security/Definitions/NetworkCertifier/RecordRows.lean` over the lanes of
`security/Definitions/NetworkCertifier/Warden.lean`, for the records the active network certifier writes
(`verity/protocols/certifier/verifier/warden.py`, whose `audit` is the reference). Each direction's record has a row a
bucket holding the `r` cells of every connection live in it, connection by connection in ordinal order, each cell
naming its connection (`readCell`); the `fails` record holds a failed connection's reason (`readReason`).

`audit` reads a run's records over its `W` windows and accepts only when they are the schedule's. It reads the schedule
the egress record shows (`scheduleOf`: each connection's admission bucket and its lane's cells) and the data its egress
cells carry (`egressOf`), renders the egress rows that schedule and data give (`expectedRows`, `bucketFrames` of
`laneFrames` over the lane `laneOf`), and compares them with the record's, bucket by bucket. Besides, every record is
complete, no connection failed (no `FAIL` cell and no reason) or was refused, the connections are ordinals `1 … N`
admitted in ordinal order, each connection's data and `END` cells are within the cap `egress_cells`, and each
connection's ingress is data and padding, then at most one `END` and padding to its bucket, from its admission on
(`ingressHolds`). Each cell is decoded once and each bucket rendered once: the audit's work is linear in the records'
cells.

It accepts what `warden.audit` accepts, but for three kinds of run the network certifier never writes: a lane of no
egress cell past its delay, unless the run's end cut it in its delay; a connection with ingress cells and no egress
cells; and a connection of more data and `END` cells than the cap, which the network certifier fails closed (`limit`).
The schedule's cells count a lane through its last bucket (`laneCells`), as `WardenSchedule.cells` does, so they pass
the cap by the padding that fills the `END`'s bucket where `r` doesn't divide `egress_cells`.
-/

@[expose] public section

namespace NetworkCertifier.Rows

/-- A cell's kind (`warden.KIND_NAMES`), its first byte: `PAD` 0, `DATA` 1, `END` 0x20, `FAIL` 0x21, `REFUSED` 0x22. -/
inductive Kind where
  | pad
  | data
  | fin
  | fail
  | refused
  deriving DecidableEq, Repr, Inhabited

def Kind.code : Kind → Nat
  | .pad => 0x00
  | .data => 0x01
  | .fin => 0x20
  | .fail => 0x21
  | .refused => 0x22

def Kind.name : Kind → String
  | .pad => "pad"
  | .data => "data"
  | .fin => "end"
  | .fail => "fail"
  | .refused => "refused"

/-- The kind whose code is `n`. -/
def Kind.ofCode? (n : Nat) : Option Kind :=
  [Kind.pad, .data, .fin, .fail, .refused].find? (·.code == n)

/-- A cell, `warden.read_cell`'s triple: its kind, its connection's ordinal and its payload. -/
structure Cell where
  kind : Kind
  conn : Nat
  payload : ByteArray
  deriving DecidableEq, Inhabited

/-- Connection `o`'s padding cell. -/
def Cell.pad (o : Nat) : Cell := ⟨.pad, o, .empty⟩

/-- A cell's header (`active.HEADER`): its kind u8, its connection's ordinal u32 and its payload's length u32. -/
def headerBytes : Nat := 9

/-- `n mod 2^(8w)` as `w` big-endian bytes. -/
def beBytes (w n : Nat) : ByteArray :=
  ⟨(List.range w).toArray.map fun i => (n >>> (8 * (w - 1 - i))).toUInt8⟩

/-- The number `b` holds big-endian in its `w` bytes from `i`, a byte past its end read as 0. -/
def beAt (b : ByteArray) (i w : Nat) : Nat :=
  (List.range w).foldl (fun a k => a * 256 + (b.data.getD (i + k) 0).toNat) 0

/-- `warden.cell`: the cell's `B` bytes (`active.encode_frame`), its header, its payload, then zero fill. -/
def Cell.encode (B : Nat) (c : Cell) : ByteArray :=
  ⟨#[c.kind.code.toUInt8]⟩ ++ beBytes 4 c.conn ++ beBytes 4 c.payload.size ++ c.payload ++
    ⟨Array.replicate (B - headerBytes - c.payload.size) 0⟩

/-- **`warden.read_cell`**: the cell whose `B` bytes are `f`, or why `f` is none: `B` bytes, a kind of `Kind`, a payload
length within the cell, `f` exactly the cell's encoding (its zero fill exact), a `DATA` cell's payload `B − 9` bytes, an
`END`'s at most that, and a `PAD`'s, `FAIL`'s or `REFUSED`'s none. -/
def readCell (B : Nat) (f : ByteArray) : Except String Cell := do
  if f.size ≠ B then throw s!"a cell is {B} bytes, not {f.size}"
  let n := beAt f 5 4
  let some kind := Kind.ofCode? (beAt f 0 1) | throw "not a warden cell"
  if B < headerBytes + n then throw "not a warden cell"
  let c : Cell := ⟨kind, beAt f 1 4, f.extract headerBytes (headerBytes + n)⟩
  if c.encode B != f then throw "not a warden cell"
  match kind with
  | .data => unless n == B - headerBytes do throw s!"a data cell of {n} bytes"
  | .fin => pure ()
  | k => unless n == 0 do throw s!"a {k.name} cell of {n} bytes"
  pure c

/-- Why a connection fails closed (`warden.REASONS`), each a `fails` frame's one payload byte by its index. -/
def reasons : List String := ["late", "limit", "upstream", "ingress", "malformed"]

/-- **`warden.read_reason`**: the connection and the reason of the `fails` frame `f` of `B` bytes, a `FAIL` cell whose
one payload byte is a reason's index; none when `f` isn't one. -/
def readReason (B : Nat) (f : ByteArray) : Option (Nat × String) :=
  let c : Cell := ⟨.fail, beAt f 1 4, f.extract headerBytes (headerBytes + 1)⟩
  match reasons[(c.payload.data.getD 0 0).toNat]? with
  | some why => if f.size == B && c.encode B == f then some (c.conn, why) else none
  | none => none

/-- The schedule's terms the audit reads (`warden.WardenParams`): the cell's bytes `B` (`cellBytes`), a connection's
cells a bucket each way `r` (`slots`), the buckets `D` from its admission to its first egress cell's (`delay`), the
connections served at once (`lanes`), the buckets a record covers `T` (`window`), and the most cells a connection's
egress lane holds (`egressCells`). -/
structure Params where
  cellBytes : Nat
  slots : Nat
  delay : Nat
  lanes : Nat
  window : Nat
  egressCells : Nat

/-- One window's record of one direction as the network certifier published it (`grid.BlockGrid`): its status, none
when complete, and its grid, a row of slots a bucket, each slot a frame or empty. -/
structure BlockGrid where
  status : Option String
  grid : Array (Array (Option ByteArray))

/-- A direction's frames, a row a bucket over the buckets `0 … W·T − 1` of its `W` records (window `w`'s row `t` is
bucket `w·T + t`), refused unless every record is complete and has `T` rows of `S` slots, each left-packed. `what`
names the direction in a refusal. -/
def framesOf (p : Params) (S : Nat) (what : String) (records : Array BlockGrid) :
    Except String (Array (Array ByteArray)) := do
  let windows ← records.toList.zipIdx.mapM fun (r, w) => do
    if let some why := r.status then throw s!"{what} window {w}: {why}"
    unless r.grid.size == p.window do throw s!"{what} window {w}: {r.grid.size} rows, not T = {p.window}"
    r.grid.toList.zipIdx.mapM fun (row, t) => do
      let frames := row.filterMap id
      unless row.size == S && (row.extract 0 frames.size).all Option.isSome do
        throw s!"{what} bucket {w * p.window + t}: not a left-packed row of {S} slots"
      pure frames
  pure windows.flatten.toArray

/-- A direction's cells (`readCell`), a row a bucket. -/
def cellsOf (p : Params) (what : String) (frames : Array (Array ByteArray)) : Except String (Array (Array Cell)) :=
  frames.toList.zipIdx.toArray.mapM fun (row, b) =>
    row.mapM fun f => (readCell p.cellBytes f).mapError (s!"{what} bucket {b}: " ++ ·)

/-- Refused at the first frame of the `fails` records: each is a failed connection's reason. -/
def noFails (p : Params) (frames : Array (Array ByteArray)) : Except String Unit :=
  match frames.toList.zipIdx.findSome? fun (row, b) => row[0]?.map (b, ·) with
  | none => pure ()
  | some (b, f) => throw <| match readReason p.cellBytes f with
    | some (o, why) => s!"connection {o} failed closed ({why}) in bucket {b}"
    | none => s!"fails bucket {b}: not a reason frame"

/-- Bucket `b`'s connections in a direction's row, one for each `r` cells, refused unless the row is whole groups of `r`
cells of one connection, the groups in strictly increasing ordinal order from ordinal 1 (`warden.audit`'s "r cells a
connection in ordinal order"). -/
def connsOf (r : Nat) (what : String) (b : Nat) (row : Array Cell) : Except String (Array Nat) := do
  let conns := (List.range (row.size / r)).toArray.map fun i => (row.getD (i * r) default).conn
  unless 0 < r && row.size % r == 0 && (row.toList.zipIdx.all fun (c, i) => c.conn == conns.getD (i / r) 0) &&
      ((0 :: conns.toList).zip conns.toList).all (fun (a, o) => a < o) do
    throw s!"{what} bucket {b}: not r cells a connection in ordinal order"
  pure conns

/-- The schedule a run's egress record shows (`WardenSchedule`, its `r` and `D` the parameters'): connection `o`'s
admission bucket `adm[o − 1]` and its lane's cells through its last bucket `cells[o − 1]`. -/
structure Schedule where
  adm : Array Nat
  cells : Array Nat

/-- The cells a lane of `n` `DATA` and `END` cells has through its last bucket, `⌈n/r⌉·r`: the padding after the `END`
fills its bucket. -/
def laneCells (r n : Nat) : Nat := (n + r - 1) / r * r

/-- Each of the `N` connections' first bucket in the egress rows, and its data: the payloads of its egress cells but
the padding, in order. -/
def egressOf (N : Nat) (rows : Array (Array Cell)) : Array (Option Nat) × Array (Array ByteArray) :=
  rows.toList.zipIdx.foldl (fun acc (row, b) => row.foldl (fun (first, data) c =>
      (first.modify (c.conn - 1) (· <|> some b),
       if c.kind == .pad then data else data.modify (c.conn - 1) (·.push c.payload))) acc)
    (Array.replicate N none, Array.replicate N #[])

/-- **The schedule the egress record shows**: each connection's admission bucket, its first in the egress rows, and its
lane's cells, through the bucket of its data's last (`laneCells`). -/
def scheduleOf (p : Params) (adm : Array Nat) (data : Array (Array ByteArray)) : Schedule :=
  ⟨adm, data.map fun d => laneCells p.slots d.size⟩

/-- **Connection `o`'s lane** past its delay, as the network certifier sends it (`Warden._egress`): a `DATA` cell for
each payload of its data but the last, an `END` with the last, then padding to the end of the `END`'s bucket; none for
no data. -/
def laneOf (r o : Nat) (data : Array ByteArray) : Array Cell :=
  if data.size = 0 then #[] else
    let live : Array Cell := (data.extract 0 (data.size - 1)).map fun d => { kind := .data, conn := o, payload := d }
    let last : Cell := { kind := .fin, conn := o, payload := data.getD (data.size - 1) .empty }
    live.push last ++ Array.replicate (laneCells r data.size - data.size) (Cell.pad o)

/-- **Connection `o`'s cells in bucket `b`** (`laneFrames`): `r` padding cells in each bucket from its admission `adm`
to `adm + D`, then its lane's cells `j` of bucket `adm + D + ⌊j/r⌋ = b`. -/
def laneFrames (p : Params) (adm o : Nat) (lane : Array Cell) (b : Nat) : Array Cell :=
  if b < adm then #[]
  else if b < adm + p.delay then Array.replicate p.slots (Cell.pad o)
  else lane.extract ((b - adm - p.delay) * p.slots) ((b - adm - p.delay + 1) * p.slots)

/-- **The egress rows** the schedule `s` and the connections' data give over the buckets `0 … H − 1` (`bucketFrames`):
bucket `b`'s row is each connection's cells in it (`laneFrames`), connection by connection in ordinal order. Each
connection adds its cells to the buckets from its admission to its lane's last, `adm + D + cells/r − 1`, so the rows
take one pass over their cells. -/
def expectedRows (p : Params) (s : Schedule) (data : Array (Array ByteArray)) (H : Nat) : Array (Array Cell) :=
  (List.range s.adm.size).foldl (fun rows i =>
      let adm := s.adm.getD i 0
      let lane := laneOf p.slots (i + 1) (data.getD i #[])
      let stop := min H (adm + p.delay + s.cells.getD i 0 / p.slots)
      (List.range' adm (stop - adm)).foldl (fun rows b => rows.modify b (· ++ laneFrames p adm (i + 1) lane b)) rows)
    (Array.replicate H #[])

/-- A connection's ingress cells in one bucket, as kinds, before its `END`: data and padding, then at most one `END` and
padding; `some true` when the `END` is among them, `some false` when it isn't, none when they aren't that. -/
def ingressGroup (kinds : List Kind) : Option Bool :=
  match kinds.dropWhile fun k => k == .data || k == .pad with
  | [] => some false
  | .fin :: rest => if rest.all (· == .pad) then some true else none
  | _ => none

/-- **Each connection's ingress is the schedule's** (`warden.audit`'s ingress): from its admission `adm` on, in
consecutive buckets, its `r` cells a bucket are data and padding, then at most one `END` and padding to the end of its
bucket, and it has no cells in a later one. `rows` are the ingress rows and `conns` each row's connections
(`connsOf`). -/
def ingressHolds (r : Nat) (adm : Array Nat) (rows : Array (Array Cell)) (conns : Array (Array Nat)) :
    Except String Unit := do
  let mut next := adm
  let mut ended := Array.replicate adm.size false
  for (row, b) in rows.toList.zipIdx do
    for (o, g) in (conns.getD b #[]).toList.zipIdx do
      unless o ≤ adm.size do throw s!"connection {o} has ingress cells and no egress cells"
      unless next.getD (o - 1) 0 == b && !ended.getD (o - 1) false do
        throw s!"connection {o}: its ingress isn't in consecutive buckets from its admission to its END's"
      match ingressGroup ((row.extract (g * r) (g * r + r)).toList.map (·.kind)) with
      | none => throw s!"connection {o}: its ingress isn't data and padding, then one END and padding to its bucket"
      | some e =>
        next := next.modify (o - 1) fun _ => b + 1
        ended := ended.modify (o - 1) fun _ => e
  pure ()

/-- What an accepted run's records show: the schedule, and the cells of each direction. -/
structure Summary where
  schedule : Schedule
  ingressCells : Nat
  egressCells : Nat

/-- **The audit of a run's records** (`warden.audit`, stricter as the module docstring says): its `W ≥ 1` windows'
ingress, egress and fails records, window `w`'s at index `w` of each. Accepted when every record is complete with `T`
left-packed rows of `lanes·r` slots (the fails record's of `lanes`), every frame of the two directions a cell and the
fails record empty, no cell a `REFUSED` or `FAIL`, each row whole groups of `r` cells of a connection in ordinal order,
the connections the ordinals `1 … N` admitted in ordinal order, each one's data and `END` cells within the cap and, when
it has none, cut by the run's end in its delay, the egress rows those `expectedRows` gives from the schedule and data
they show, bucket by bucket, and each connection's ingress the schedule's (`ingressHolds`). The refusal names the first
finding. -/
def audit (p : Params) (ingress egress fails : Array BlockGrid) : Except String Summary := do
  let W := egress.size
  unless 0 < W && ingress.size == W && fails.size == W do
    throw s!"a run's records are W ≥ 1 windows' ingress, egress and fails records, not {ingress.size}, {W} and \
      {fails.size}"
  unless 0 < p.slots && 0 < p.delay && 0 < p.lanes && 0 < p.window && headerBytes < p.cellBytes do
    throw "the schedule's r, D, lanes and T are positive, and a cell holds its header and a byte"
  let H := W * p.window
  let S := p.lanes * p.slots
  let eg ← cellsOf p "egress" (← framesOf p S "egress" egress)
  let ing ← cellsOf p "ingress" (← framesOf p S "ingress" ingress)
  noFails p (← framesOf p p.lanes "fails" fails)
  for (rows, what) in [(ing, "ingress"), (eg, "egress")] do
    for (row, b) in rows.toList.zipIdx do
      for c in row do
        if c.kind == .refused then throw s!"{what} bucket {b}: a refused cell"
        if c.kind == .fail then throw s!"connection {c.conn} failed closed in {what} bucket {b}"
  let egConns ← eg.toList.zipIdx.toArray.mapM fun (row, b) => connsOf p.slots "egress" b row
  let ingConns ← ing.toList.zipIdx.toArray.mapM fun (row, b) => connsOf p.slots "ingress" b row
  let N := egConns.foldl (fun n cs => max n (cs.back?.getD 0)) 0
  unless N ≤ egConns.foldl (· + ·.size) 0 do throw s!"the egress record's ordinals aren't 1 … {N}"
  let (first, data) := egressOf N eg
  let some adm := first.mapM (m := Option) fun a => a
    | throw s!"connection {(first.findIdx? Option.isNone).getD 0 + 1} has no egress cells: the ordinals aren't 1 … {N}"
  for i in [1:N] do
    if adm.getD i 0 < adm.getD (i - 1) 0 then
      throw s!"connection {i + 1} admitted at bucket {adm.getD i 0}, before connection {i}"
  let s := scheduleOf p adm data
  for i in [0:N] do
    let n := (data.getD i #[]).size
    if p.egressCells < n then
      throw s!"connection {i + 1}: {n} egress data and END cells, past the cap {p.egressCells}"
    if n == 0 && adm.getD i 0 + p.delay < H then
      throw s!"connection {i + 1}: no egress cell past its delay, and the run didn't end in it"
  let want := expectedRows p s data H
  if let some b := (List.range H).find? fun b => eg.getD b #[] != want.getD b #[] then
    throw s!"egress bucket {b} isn't the schedule's"
  ingressHolds p.slots adm ing ingConns
  pure ⟨s, ing.foldl (· + ·.size) 0, eg.foldl (· + ·.size) 0⟩

end NetworkCertifier.Rows
