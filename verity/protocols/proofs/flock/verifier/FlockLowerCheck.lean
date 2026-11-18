module

public import Flock.Lower

/-! The `flock-lower-check` executable, a test's (`tests/verity/protocols/proofs/flock/prover/test_lean_lower.py`): a
descriptor's Definition lowered by `Lower.lower` and its template written, every composite Definition's template `CALL`s
held to `Lower.callBound` of its lift to bits, and the template's rows, unrolled, solved on 64 instances a lane word. -/

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

def main (args : List String) : IO UInt32 := do
  let [desc, key, ins, outs, inFile, tmplFile, outFile] := args
    | do IO.eprintln "usage: flock-lower-check DESC KEY INS OUTS IN TMPL OUT"; return 2
  let P ← match Program.decode (← IO.FS.readBinFile desc) with
    | .ok p => pure p
    | .error e => do IO.eprintln s!"flock-lower-check: decode: {e}"; return 1
  let nats := fun (s : String) => ((s.splitOn ",").map String.toNat!).toArray
  let Q ← match Lift.closure P ((Array.range P.defs.size).filter fun i => !(P.defs[i]!).prim) with
    | .ok (q, _) => pure q
    | .error e => do IO.eprintln s!"flock-lower-check: lift: {e}"; return 1
  let mut over := false
  for i in [0:Q.defs.size] do
    let df := Q.defs[i]!
    if df.prim then continue
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
  let m := (Q.defs[d]!).outLeaves
  IO.FS.writeFile outFile ((List.range m).foldl (fun s k => s ++ toString (z.getD (Lower.outBit t k) 0) ++ "\n") "")
  IO.println s!"LOWERED {key} defs={t.defs.size} lines={(Lower.renderLines t).size} rows={z.size}"
  if over then do IO.eprintln "flock-lower-check: a Definition's CALLs past Lower.callBound"; return 1
  return 0
