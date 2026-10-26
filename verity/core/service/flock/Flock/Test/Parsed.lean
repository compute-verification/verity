/-
`RecVStar.checkParsed` on R's opening part over row ports `m0` and `acc0` (V*'s rows `m0`, `acc0`, `v` of 64 words
and `out` of 128): a circuit carrying V*'s META passes; one with two units per VU, a leaf cut, input leaves off V*'s,
output leaves off V*'s (permuted, short or missing), or a row port off V*'s by name, words or bits, or one too few or too
many, is refused. V*'s own `rec-cap/n128` file, parsed as the verifier parses it (`HmOut.parse`), passes, and with its
output leaves permuted is refused.
-/
import Flock.RecVStar

namespace Flock.Test.Parsed

open Lean Flock Flock.RecVStar

def opening : VStar.Stmt := .opening 22 2 0 { ops := [], res := [] } [("m0", 8), ("acc0", 8)] 1

def outW : Nat := match opening.unit with
  | .ok u => u.outWords
  | .error _ => 0

#guard outW > 1

def port (nm : String) (words : Nat) : Port := { name := nm, words, chunks := 2 * words / 128, nb := 0 }

def rows : List Port := [port "m0" 64, port "acc0" 64, port "v" 64, port "out" 128]

def circ (ports : List Port := rows) (upv : Nat := 1) (leaves : Array Int := (Array.range 192).map Int.ofNat)
    (cuts : Array (Nat × Nat × Nat) := #[]) (leavesOut : Array Nat := Array.range outW) : Circuit :=
  { (default : Circuit) with ports := ports.toArray, upv, leavesIn := #[leaves], leavesOut := #[leavesOut],
    leafCuts := cuts }

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
#guard !ok (circ [port "m0" 64, port "acc0" 64, port "v" 64])
#guard !ok (circ (rows ++ [port "w" 64]))
#guard !ok { circ with leavesIn := #[] }

def hex128 : String := "".pushn 'a' 128
def bindings : VStar.Bindings :=
  { partition := hex128, program := hex128, programDigests := Json.mkObj [("RecCap_v1", Json.str hex128)] }

def capParsed : Except String Circuit := do
  HmOut.parse Tags.circuit (← circuitFile (.cap 0 128) bindings)

def capOk (perm : Array Nat → Array Nat := id) : Bool := match capParsed with
  | .ok c => (checkParsed (.cap 0 128) { c with leavesOut := c.leavesOut.modify 0 perm }).isOk
  | .error _ => false

#guard capParsed.toBool
#guard (capParsed.toOption.map (·.leavesOut)) == some #[Array.range 64]
#guard capOk
#guard !capOk (·.reverse)
#guard !capOk (·.swapIfInBounds 0 1)

end Flock.Test.Parsed
