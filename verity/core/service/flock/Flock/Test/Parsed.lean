/-
`RecVStar.checkParsed` on R's opening part over row ports `m0` and `acc0` (V*'s rows `m0`, `acc0`, `v` of 64 words
and `out` of 128): a circuit carrying V*'s META passes; one with two units per VU, a leaf cut, input leaves off V*'s,
output leaves off V*'s (permuted, short or missing), or a row port off V*'s by name, words or bits, a bit row
(`sha512/row/v2`) even of V*'s bits, or one too few or too many, is refused. V*'s own `rec-cap/n128` file states its
output leaves as the check requires them, its output words in order (read off its META, as `HmRow.parse` reads
`leaves_out`).
-/
import Flock.RecVStar

namespace Flock.Test.Parsed

open Lean Flock Flock.RecVStar

def opening : VStar.Stmt := .opening 22 2 0 { ops := [], res := [] } [("m0", 8), ("acc0", 8)] 1

def outWords (s : VStar.Stmt) : Nat := match s.unit with
  | .ok u => u.outWords
  | .error _ => 0

def outW : Nat := outWords opening

#guard outW > 1

def port (nm : String) (words : Nat) : Port := { name := nm, words, chunks := 2 * words / 128, nb := 0 }

def rows : List Port := [port "m0" 64, port "acc0" 64, port "v" 64, port "out" 128]

def circ (ports : List Port := rows) (upv : Nat := 1) (leaves : Array Int := (Array.range 192).map Int.ofNat)
    (cuts : Array (Nat × Nat × Nat) := #[]) (leavesOut : Array Nat := Array.range outW) : Circuit :=
  { (default : Circuit) with
    ports := ports.toArray, upv, leavesIn := #[leaves], leavesOut := #[leavesOut], leafCuts := cuts }

def ok (c : Circuit) : Bool := (checkParsed opening c).isOk

#guard ok (circ)
#guard !ok (circ (upv := 2))
#guard !ok (circ (cuts := #[(0, 3, 0)]))
#guard !ok (circ (leaves := (Array.range 191).map Int.ofNat))
#guard !ok (circ (leaves := ((Array.range 192).map Int.ofNat).set! 5 (-1)))
#guard !ok (circ (leaves := ((Array.range 192).map Int.ofNat).reverse))
#guard !ok (circ (leavesOut := (Array.range outW).reverse))
#guard !ok (circ (leavesOut := ((Array.range outW).swapIfInBounds 0 1)))
#guard !ok (circ (leavesOut := Array.range (outW - 1)))
#guard !ok { circ with leavesOut := #[] }
#guard !ok (circ [port "m0" 64, port "acc1" 64, port "v" 64, port "out" 128])
#guard !ok (circ [port "m0" 64, port "acc0" 64, port "v" 128, port "out" 128])
#guard !ok (circ [port "m0" 64, port "acc0" 64, port "v" 64, { port "out" 128 with bits := 2000, v2 := true }])
#guard !ok (circ [port "m0" 64, port "acc0" 64, port "v" 64, { port "out" 128 with v2 := true }])
#guard !ok (circ [port "m0" 64, port "acc0" 64, port "v" 64])
#guard !ok (circ (rows ++ [port "w" 64]))
#guard !ok { circ with leavesIn := #[] }

def hex128 : String := "".pushn 'a' 128
def bindings : VStar.Bindings :=
  { partition := hex128, program := hex128, programDigests := Json.mkObj [("RecCap_v1", Json.str hex128)] }

def capLeavesOut : Except String (Array (Array Nat)) := do
  let mj ← HmOut.metaOf (← circuitFile (.cap 0 128) bindings)
  mj.getObjValAs? (Array (Array Nat)) "leaves_out"

#guard outWords (.cap 0 128) == 64
#guard capLeavesOut.toOption == some #[Array.range (outWords (.cap 0 128))]

end Flock.Test.Parsed
