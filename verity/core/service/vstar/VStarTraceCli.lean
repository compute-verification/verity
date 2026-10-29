import VStar.Trace
import Flock.ProvedScope

/-!
# `vstar-trace`'s command: the trace assigner's side of its difftest

`vstar-trace CIRCUIT INST` reads a staged V\* part (its `circuit.txt`, parsed as the setup parses it,
`ProvedScope.circuitOf` under `Tags.circuit`, and its `inst.bin` with shared rows, decoded to its port inputs) and prints
`VStar.Trace.digests` under M0's zero mask as one JSON line: `m`, `k_log`, whether every port's row is whole 16-bit
words (`full_rows`, `VStar.Trace.fullRows`, which the port lemma's row width reads), each block's SHA-512 (`blocks`) and each block's
ranges' (`ranges`), hex. `flock-trace` (`verity/ml/flock/live/src/bin/flock-trace.rs`) prints the same of the witness the
deployed prover assigns; `tests/test_vstar_trace_difftest.py` holds the two equal.
-/

namespace VStar.TraceCli

open Lean (Json)
open Flock

/-- A staged `inst.bin` with shared rows (`Instances::load`) as port inputs: the header line, each port's table rows in the
circuit's port order (`2 · words` bytes each, `shared_rows[name]` of them), their salts, their `b ‖ c`, then each instance's
refs (u32 little-endian, instance-major). A file with public words or without shared rows is refused. -/
def partInOf (c : Circuit) (file : ByteArray) : Except String VStar.In.PartIn := do
  let some nl := findByte file 10 | throw "inst.bin: no header line"
  let some text := String.fromUTF8? (file.extract 0 nl) | throw "inst.bin: the header is not UTF-8"
  let head ← Json.parse text
  let n ← head.getObjValAs? Nat "instances"
  let shared ← head.getObjVal? "shared_rows"
  let np := c.ports.size
  let counts ← c.ports.mapM fun p => shared.getObjValAs? Nat p.name
  let total := counts.foldl (· + ·) 0
  let starts := (counts.foldl (fun (acc : Array Nat × Nat) k => (acc.1.push acc.2, acc.2 + k)) (#[], 0)).1
  let tableAt := (Array.range np).foldl (fun (acc : Array Nat × Nat) p =>
    (acc.1.push acc.2, acc.2 + 2 * (c.ports.getD p default).words * counts[p]!)) (#[], nl + 1)
  let saltAt := tableAt.2
  let refsAt := saltAt + (HmRow.SALT_BYTES + HmRow.BC_BYTES) * total
  if refsAt + 4 * n * np != file.size then throw "inst.bin: not the shared-row layout without public words"
  let u32At := fun (at_ : Nat) => (List.range 4).foldl (fun v j => v + (file.get! (at_ + j)).toNat * 256 ^ j) 0
  pure ((Array.range np).map fun p =>
    let port := c.ports.getD p default
    let w := 2 * port.words
    let rows := (Array.range counts[p]!).map fun r =>
      let at_ := tableAt.1[p]! + w * r
      let s := saltAt + HmRow.SALT_BYTES * (starts[p]! + r)
      ({ words := VStar.In.wordsOf (file.extract at_ (at_ + w)), salt := file.extract s (s + HmRow.SALT_BYTES) } : VStar.In.Row)
    (port.name, { rows, refs := (Array.range n).map fun i => u32At (refsAt + 4 * (i * np + p)) }))

def run (circuitPath instPath : String) : IO UInt32 := do
  let cb ← IO.FS.readBinFile circuitPath
  let ib ← IO.FS.readBinFile instPath
  match ProvedScope.circuitOf Tags.circuit cb [] with
  | .error e => IO.eprintln s!"circuit: {e}"; return 2
  | .ok c =>
    match partInOf c ib with
    | .error e => IO.eprintln e; return 2
    | .ok pin =>
      match Trace.digests c pin (fun _ _ => false) with
      | none => IO.eprintln "refused: the circuit is outside what the trace assigner mirrors (VStar.Trace.supported)"; return 3
      | some ds =>
        IO.println (Json.compress (Json.mkObj [
          ("m", (Trace.m c pin : Json)), ("k_log", (c.kLog : Json)), ("instances", (Trace.count pin : Json)),
          ("full_rows", Json.bool (Trace.fullRows c)),
          ("blocks", Json.arr (ds.map fun d => Json.str (hexOf d.1))),
          ("ranges", Json.arr (ds.map fun d => Json.arr (d.2.map fun r => Json.str (hexOf r))))]))
        return 0

/-- `vstar-trace CIRCUIT INST`'s command. -/
def cmd (args : List String) : IO UInt32 :=
  match args with
  | [c, i] => run c i
  | _ => do IO.eprintln "usage: vstar-trace CIRCUIT INST"; return 1

end VStar.TraceCli
