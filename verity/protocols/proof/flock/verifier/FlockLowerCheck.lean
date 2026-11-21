module

public import Flock.Compose

/-! The `flock-lower-check` executable, a test's (`tests/verity/protocols/proof/flock/prover/test_lean_lower.py`): a
descriptor's Definition lowered by `Lower.lower` and its template written, every composite Definition's template `CALL`s
but the program root's held to `Lower.callBound` of its lift to bits, and the template's rows, unrolled, solved on 64
instances a lane word; or, under `compose`, the block's circuit file `Compose.compose` writes on that template. -/

@[expose] public section

open Flock

/-- The unrolled template's columns, 64 instances a word: the input bits `ins` (one word a port bit), the constant all
ones, every other written column `(A·z)(B·z)` in column order, an unwritten one 0 (`Flock.Test.Lower.solve`'s, lane-wise). -/
def lowerLanes (t : Tmpl.Tmpl) (ins : Array UInt64) : V (Array UInt64) := do
  let acc ← Tmpl.Tmpl.rows t
  let k := t.constPos
  let sum := fun (z : Array UInt64) (r : Array Nat) =>
    r.foldl (fun b c => b ^^^ (if c == k then ~~~(0 : UInt64) else z.getD c 0)) 0
  return (List.range acc.ra.size).foldl (fun z c =>
    z.push (if c < ins.size then ins[c]! else if c == k then ~~~(0 : UInt64)
      else if acc.set.getD c false then sum z acc.ra[c]! &&& sum z acc.rb[c]! else 0)) #[]

/-- `name:bits,…` as its pairs. -/
def named (s : String) : Array (String × Nat) :=
  ((s.splitOn ",").map fun p => match p.splitOn ":" with
    | [n, b] => (n, b.toNat!)
    | _ => (p, 0)).toArray

/-- `compose DESC KEY NAME INS OUTS LAYOUT PROGRAM PARTITION DIGESTS FILE`: Definition `KEY` lowered with one 16-bit port a
row word, and the block's file written: input rows `INS` (`name:bits`), output ports `OUTS` (`name:words`), META's
`program_digests` the JSON object `DIGESTS`. Its output is no verdict of the door: the door's lowering is
`Spot.OneStageZk.lowerBlock`, under `admitted` and `admittedBlock`, and nothing cites this executable's acceptance of a
block as the door's. -/
def composeMain : List String → IO UInt32
  | [desc, key, name, ins, outs, layout, prog, part, digests, file] => do
    let ins := named ins
    let outs := named outs
    let some (.obj ds) := (JsonParse.parse digests).toOption | do IO.eprintln "flock-lower-check compose: DIGESTS is not a JSON object"; return 2
    let programDigests := ds.toList.filterMap fun (k, v) => match v with | .str s => some (k, s) | _ => none
    let P ← match Program.decode (← IO.FS.readBinFile desc) with
      | .ok p => pure p
      | .error e => do IO.eprintln s!"flock-lower-check compose: decode: {e}"; return 1
    let some d := P.index[key]? | do IO.eprintln s!"flock-lower-check compose: no Definition {key}"; return 1
    let nIn := ins.foldl (fun a p => a + (p.2 + 15) / 16) 0
    let nOut := outs.foldl (· + ·.2) 0
    let out : V ByteArray := do
      let t ← Lower.lower P d (Array.replicate nIn 16) (Array.replicate nOut 16)
      Compose.compose { name, body := .tmpl t layout, ins, outs, programDigests } prog part
    match out with
    | .ok bs => do IO.FS.writeBinFile file bs; IO.println s!"COMPOSED {name} bytes={bs.size}"; return 0
    | .error e => do IO.eprintln s!"flock-lower-check compose: {e}"; return 1
  | _ => do IO.eprintln "usage: flock-lower-check compose DESC KEY NAME INS OUTS LAYOUT PROGRAM PARTITION DIGESTS FILE"; return 2

def main (args : List String) : IO UInt32 := do
  if args.head? == some "compose" then return ← composeMain args.tail
  let [desc, key, ins, outs, inFile, tmplFile, outFile] := args
    | do IO.eprintln "usage: flock-lower-check DESC KEY INS OUTS IN TMPL OUT"; return 2
  let P ← match Program.decode (← IO.FS.readBinFile desc) with
    | .ok p => pure p
    | .error e => do IO.eprintln s!"flock-lower-check: decode: {e}"; return 1
  let nats := fun (s : String) => ((s.splitOn ",").map String.toNat!).toArray
  let Q ← match Lift.closure P ((Array.range P.defs.size).filter fun i => !(P.defs[i]!).prim) with
    | .ok (q, f) => pure (q, f P.root)
    | .error e => do IO.eprintln s!"flock-lower-check: lift: {e}"; return 1
  let (Q, rootQ) := Q
  let mut over := false
  for i in [0:Q.defs.size] do
    let df := Q.defs[i]!
    if df.prim || i == P.root || i == rootQ || (Lift.lift Q i).toOption.map (·.2) != some i then continue
    match Lower.lower Q i #[max 1 df.paramLeaves] #[max 1 df.outLeaves] with
    | .ok t =>
      let calls := (t.defs.getD (t.root - 1) default).calls.size
      IO.println s!"CALLS {df.key} {calls} bound {Lower.callBound df}"
      if calls > Lower.callBound df then over := true
    | .error e => IO.println s!"REFUSED {df.key}: {e}"
  let some d := P.index[key]? | do IO.eprintln s!"flock-lower-check: no Definition {key}"; return 1
  let t ← match Lower.lower P d (nats ins) (nats outs) with
    | .ok t => pure t
    | .error e => do IO.eprintln s!"flock-lower-check: lower: {e}"; return 1
  IO.FS.writeFile tmplFile (Lower.render t)
  let words := ((← IO.FS.readFile inFile).splitOn "\n").toArray.filter (· ≠ "") |>.map (UInt64.ofNat ·.toNat!)
  let n := (nats ins).foldl (· + ·) 0
  if words.size != n then do IO.eprintln s!"flock-lower-check: {words.size} input words, ports of {n} bits"; return 1
  let z ← match lowerLanes t words with
    | .ok z => pure z
    | .error e => do IO.eprintln s!"flock-lower-check: rows: {e}"; return 1
  let m := match Lift.lift P d with
    | .ok (Q, d) => (Q.defs[d]!).outLeaves
    | .error _ => 0
  IO.FS.writeFile outFile ((List.range m).foldl (fun s k => s ++ toString (z.getD (Lower.outBit t k) 0) ++ "\n") "")
  IO.println s!"LOWERED {key} defs={t.defs.size} lines={(Lower.renderLines t).size} rows={z.size}"
  if over then do IO.eprintln "flock-lower-check: a Definition's CALLs past Lower.callBound"; return 1
  return 0
