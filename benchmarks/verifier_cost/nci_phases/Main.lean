import NciCheck

/-!
`nci-phases INSTANCE [TOP [BIG]]`: `NciCheck.Desc.evaluate`'s steps on one `nci-check --policy` instance, each timed, one
`PHASE {json}` line per step on stdout (milliseconds, resident and peak RSS in MB): the file's read, the instance's JSON
parse, `NciCheck.parse`, C-Flock's decoder (`Flock.Program.decode`, which parses the program's text again and hashes it),
`NciCheck.Widths.reinterpretation`, `Prog.orderViolation`, the layout checks, `walkable`, `analyze` split into each reached
Definition's summary (`summarize`, callees first, the `TOP` slowest named with their body nodes and return pieces; one of
at least `BIG` body nodes, 2^17 unless given, step by step: `summarizeTimed`) and the
root's reads (`nodeReads`), and the verdict (θ, the owners, `intoOf`, `novelWidth`). The last line is the verdict, which
`nci_cost.sh` holds to `nci-check --policy`'s. A measurement, not a verifier: it runs what `NciCheck.Desc.evaluate`
runs, in its order.
-/

open Lean NciCheck.Desc

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

def report (phase : String) (t0 : Nat) (extra : List (String × Json) := []) : IO Unit := do
  let t1 ← IO.monoNanosNow
  let rss ← statusKb "VmRSS:"
  let hwm ← statusKb "VmHWM:"
  let j := Json.mkObj ([("phase", Json.str phase), ("ms", toJson ((t1 - t0) / 1000000)),
    ("rss_mb", toJson (rss / 1024)), ("hwm_mb", toJson (hwm / 1024))] ++ extra)
  IO.println s!"PHASE {j.compress}"
  (← IO.getStdout).flush

def orFail {α : Type} (what : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw (IO.userError s!"{what}: {e}")

def msSince (t : Nat) : IO Nat := return ((← IO.monoNanosNow) - t) / 1000000

/-- `Ivs.norm` with `List.mergeSort` in place of `Array.qsort`: the same intervals in the same order. -/
def normMerge (xs : Array (Nat × Nat)) : Ivs :=
  ((xs.filter fun p => p.1 < p.2).toList.mergeSort fun x y => x.1 ≤ y.1).foldl (init := #[]) fun acc (a, b) =>
    match acc.back? with
    | some (c, e) => if a ≤ e then acc.pop.push (c, max b e) else acc.push (a, b)
    | none => acc.push (a, b)

/-- The most intervals of `xs` sharing one start. -/
def maxMult (xs : Array (Nat × Nat)) : Nat :=
  (xs.foldl (fun (m : Std.HashMap Nat Nat) x => m.insert x.1 (m.getD x.1 0 + 1)) {}).fold (fun a _ v => max a v) 0

/-- `summarize P memo d`, step by step, each timed: the return's pieces (`down`), every node's reads (`nodeReads`, with
the time of each `CHUNK` nodes), and per parameter `ivsOf` and `Ivs.norm`, with `normMerge`'s time beside it; the
summary is `summarize`'s, and `PHASE` gives the steps and the parameters `Ivs.norm` takes longest on. -/
def summarizeTimed (P : Flock.Program.Prog) (memo : Array Summary) (d : Nat) (topN : Nat) : IO Summary := do
  let CHUNK := 2 ^ 17
  let defn := P.defs[d]!
  let body := defn.body
  let t0 ← IO.monoNanosNow
  let t ← IO.monoNanosNow
  let c : Ctx := { P, memo, d, args := body.nodes.map fun n => n.args.map (ofSeq ·.seq) }
  let mut ret := #[]
  for s in (ofSeq body.ret).segs do
    for q in s.ps do ret := ret ++ (← orFail "ret" (down c Flock.Steps.FUEL q))
  let retMs ← msSince t
  let t ← IO.monoNanosNow
  let mut tc ← IO.monoNanosNow
  let mut chunks : Array Nat := #[]
  let mut reads := 0
  let mut acc : Array (Array Piece) := Array.replicate body.params.size #[]
  for k in [0:body.nodes.size] do
    let rs ← orFail s!"node {k}" (nodeReads c k)
    reads := reads + rs.size
    for s in rs do
      if s.reps > 1 then continue
      for q in s.ps do
        if let .ref true i := q.tgt then
          acc := acc.modify i (·.push q)
    if (k + 1) % CHUNK == 0 then
      chunks := chunks.push (← msSince tc)
      tc ← IO.monoNanosNow
  chunks := chunks.push (← msSince tc)
  let readsMs ← msSince t
  let mut used : Array Ivs := #[]
  let mut stats : Array (Nat × Nat × Nat × Nat × Nat × Nat) := #[]
  let mut ivsMs := 0
  let mut normMs := 0
  let mut mergeMs := 0
  let mut same := true
  for i in [0:acc.size] do
    let qs := acc[i]!
    let t ← IO.monoNanosNow
    let iv ← IO.lazyPure fun _ => ivsOf qs
    let a ← IO.monoNanosNow
    let n ← IO.lazyPure fun _ => Ivs.norm iv
    let b ← IO.monoNanosNow
    let m ← IO.lazyPure fun _ => normMerge iv
    let e ← IO.monoNanosNow
    same := same && n == m
    ivsMs := ivsMs + (a - t)
    normMs := normMs + (b - a)
    mergeMs := mergeMs + (e - b)
    used := used.push n
    stats := stats.push ((b - a) / 1000000, (e - b) / 1000000, i, qs.size, iv.size, maxMult iv)
  let top := (stats.qsort fun x y => x.1 > y.1).extract 0 topN |>.map fun (ms, mms, i, nq, ni, mult) =>
    Json.mkObj [("param", toJson i), ("name", Json.str ((body.params[i]?.map (·.1)).getD "")), ("norm_ms", toJson ms),
      ("merge_ms", toJson mms), ("pieces", toJson nq), ("intervals", toJson ni), ("max_same_start", toJson mult)]
  report "summarize_steps" t0 [("key", Json.str (String.ofList (defn.key.toList.take 80))),
    ("body_nodes", toJson body.nodes.size), ("params", toJson body.params.size), ("ret_ms", toJson retMs),
    ("reads_ms", toJson readsMs), ("reads", toJson reads), ("reads_chunk_ms", toJson chunks), ("chunk", toJson CHUNK),
    ("ivsOf_ms", toJson (ivsMs / 1000000)), ("norm_qsort_ms", toJson (normMs / 1000000)),
    ("norm_merge_ms", toJson (mergeMs / 1000000)), ("norm_same", toJson same),
    ("pieces", toJson (acc.foldl (fun n qs => n + qs.size) 0)), ("top", Json.arr top)]
  return { ret := PS.of ret, used }

def main (args : List String) : IO UInt32 := do
  let some (path : String) := args[0]? | IO.eprintln "usage: nci-phases INSTANCE [TOP [BIG]]"; return 2
  let topN := (args[1]? >>= String.toNat?).getD 20
  let big := (args[2]? >>= String.toNat?).getD (2 ^ 17)
  let t ← IO.monoNanosNow
  let s ← IO.FS.readFile path
  report "read" t [("bytes", toJson s.utf8ByteSize)]
  let t ← IO.monoNanosNow
  let j ← orFail "json" (Json.parse s)
  report "json" t
  let t ← IO.monoNanosNow
  let i ← orFail "instance" (NciCheck.parse j)
  report "instance" t [("program_bytes", toJson i.program.utf8ByteSize), ("theta", toJson i.theta.size),
    ("owner", toJson i.owner.size)]
  let t ← IO.monoNanosNow
  let P ← orFail "decode" (Flock.Program.decode i.program.toUTF8)
  let body := (P.defs[P.root]!).body
  let nodes := P.defs.foldl (fun n d => n + d.body.nodes.size) 0
  report "decode" t [("definitions", toJson P.defs.size), ("body_nodes", toJson nodes),
    ("root_nodes", toJson body.nodes.size), ("gates", toJson (P.defs[P.root]!).gates)]
  let t ← IO.monoNanosNow
  let ri := NciCheck.Widths.reinterpretation P
  report "widths" t [("refused", toJson ri.isSome)]
  if ri.isSome then IO.println (NciCheck.Desc.check j).compress; return 0
  let t ← IO.monoNanosNow
  let ov := P.orderViolation
  report "order" t [("refused", toJson ov.isSome)]
  if ov.isSome then IO.println (NciCheck.Desc.check j).compress; return 0
  let t ← IO.monoNanosNow
  let laid := rootLaidOut P && inputsAtRoot P
  report "layout" t [("ok", toJson laid)]
  let t ← IO.monoNanosNow
  let wk := walkable P
  report "walkable" t [("ok", toJson wk)]
  if !(laid && wk) then IO.println (NciCheck.Desc.check j).compress; return 0
  let t0 ← IO.monoNanosNow
  let seen := reached P
  let mut memo : Array Summary := #[]
  let mut times : Array (Nat × Nat) := #[]
  for d in [0:P.defs.size] do
    let t ← IO.monoNanosNow
    let sm ← if !(seen[d]! && d != P.root) then pure default
      else if !(P.defs[d]!).prim && (P.defs[d]!).body.nodes.size ≥ big then summarizeTimed P memo d topN
      else orFail s!"summarize {(P.defs[d]!).key}" (summarize P memo d)
    let t1 ← IO.monoNanosNow
    memo := memo.push sm
    times := times.push (t1 - t, d)
  let slow := (times.qsort fun a b => a.1 > b.1).extract 0 topN
  let top := slow.map fun (ns, d) =>
    let defn := P.defs[d]!
    let sm := memo[d]!
    Json.mkObj [("key", Json.str (String.ofList (defn.key.toList.take 160))), ("ms", toJson (ns / 1000000)),
      ("body_nodes", toJson defn.body.nodes.size), ("gates", toJson defn.gates),
      ("ret_segments", toJson sm.ret.segs.size),
      ("ret_pieces", toJson (sm.ret.segs.foldl (fun n s => n + s.ps.size) 0)),
      ("out_leaves", toJson defn.outLeaves)]
  let reachedN := (List.range P.defs.size).countP fun d => seen[d]!
  report "summaries" t0 [("reached", toJson reachedN), ("top", Json.arr top)]
  let t ← IO.monoNanosNow
  let c : Ctx := { P, memo, d := P.root, args := body.nodes.map fun n => n.args.map (ofSeq ·.seq) }
  let mut reads : Array (Array Seg) := #[]
  for k in [0:body.nodes.size] do
    let rs ← orFail s!"root node {k}" (nodeReads c k)
    reads := reads.push rs
  let segs := reads.foldl (fun n r => n + r.size) 0
  report "root_reads" t [("segments", toJson segs)]
  let t ← IO.monoNanosNow
  let K := body.nodes.size
  let offs := body.offsets
  let inp := body.nodes.map (isInputNode P)
  let reg := body.nodes.map fun n => NciCheck.isRegisteredId (P.defs[n.fn]!).key
  let theta ← orFail "theta" ((List.range i.theta.size).foldlM (fun m t => thetaStep inp reg m i.theta[t]!)
    (Array.replicate K false))
  let own ← orFail "owner" ((List.range K).foldlM
    (fun own k => own.push <$> ownerStep inp[k]! (offs[k + 1]! - offs[k]! > 0) i.owner[k]!) #[])
  let n := NciCheck.iuCount own
  let size := sizes offs own n
  let into := intoOf offs own theta reads n
  let r := (Array.range n).map fun u => (size[u]!, novelWidth into[u]!)
  let mine := NciCheck.verdict (P.defs[P.root]!).gates r (NciCheck.failure i.gamma i.num i.den r)
  report "verdict" t [("ius", toJson n)]
  IO.println mine.compress
  return 0
