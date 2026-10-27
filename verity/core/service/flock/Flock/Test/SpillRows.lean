/-
`ProvedScope.v2Row` is exactly the `sha512/row/v2` rows the parser takes: a row of `w` leaves (`16 w` bits) is in scope at
the compressions `HmRow` derives from its length (`⌈(2 w + 17) / 128⌉ − 1`), input row or output row, and at one more or
one fewer it is not. That includes a row whose bytes mod 128 are at least 112, whose last compression's block holds its
bytes' end and the padding's start, the `hm96` block only padding: H1's tile receipt at K = 384 (8 (KS + m) = 8 · 14 =
112 bytes) and its lanes' partials row there. `proved_scope.rs`'s `v2_rows_are_the_parsers` runs the same rows.
-/
import Flock.ProvedScope

namespace Flock.Test.SpillRows

/-- A `sha512/row/v2` row of `w` u16 leaves in `n` compressions. -/
def row (w n : Nat) : Port := { name := "p", words := w, chunks := n, nb := 0, bits := 16 * w, v2 := true }

/-- The compressions the parser derives for a row of `w` leaves. -/
def serial (w : Nat) : Nat := (2 * w + 17 + 127) / 128 - 1

/-- `ports`, the last `outs` of them output rows of one output port each. -/
def circ (ports : Array Port) (outs : Nat) : Circuit := { (default : Circuit) with ports := ports, outRowPorts := Array.replicate outs 1 }

#guard (List.range 2049).all fun w => ProvedScope.v2Row (row w (serial w))
#guard (List.range 2049).all fun w => !ProvedScope.v2Row (row w (serial w + 1))
#guard (List.range 2049).all fun w => serial w == 0 || !ProvedScope.v2Row (row w (serial w - 1))

-- 112 bytes: one compression, holding the bytes and the padding's first 16
#guard serial 56 == 1 && ProvedScope.v2Row (row 56 1) && !ProvedScope.v2Row (row 56 0)
-- a spilled row as the input row, then as the output row
#guard ProvedScope.bitRows (circ #[row 56 1, row 1 0] 1)
#guard ProvedScope.bitRows (circ #[row 1 0, row 56 1] 1)
#guard ProvedScope.bitRows (circ #[row 120 2, row 248 4] 1)
#guard !ProvedScope.bitRows (circ #[row 1 0, row 56 2] 1)

end Flock.Test.SpillRows
