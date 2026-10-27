import NciCheck

/-!
`nci-phases INSTANCE [TOP]`: `NciCheck.Desc.evaluate`'s steps on one `nci-check --policy` instance, each timed, one
`PHASE {json}` line per step on stdout (milliseconds, resident and peak RSS in MB): the file's read, the instance's JSON
parse, `NciCheck.parse`, C-Flock's decoder (`Flock.Program.decode`, which parses the program's text again and hashes it),
`NciCheck.Widths.reinterpretation`, `Prog.orderViolation`, the layout checks, `walkable`, `analyze` split into each reached
Definition's summary (`summarize`, callees first, the `TOP` slowest named with their body nodes and return pieces) and the
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

def main (args : List String) : IO UInt32 := do
  let some (path : String) := args[0]? | IO.eprintln "usage: nci-phases INSTANCE [TOP]"; return 2
  let topN := (args[1]? >>= String.toNat?).getD 20
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
    let sm ← if seen[d]! && d != P.root then orFail s!"summarize {(P.defs[d]!).key}" (summarize P memo d) else pure default
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
