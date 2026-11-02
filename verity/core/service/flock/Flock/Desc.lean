import Flock.HmRow

/-!
# A circuit's descriptor

A circuit file describes its circuit, and the verifier's statements read the circuit through its descriptor: `Desc.parse`
reads the file as a descriptor `D`, and `D.unroll` is the circuit it describes. The circuit's pin is SHA-512 of the
file, so it is SHA-512 of `D`'s bytes, whatever form `D` takes.

Every descriptor is a flat file for now, `single c`: `HmOut.parse` reads it, and it unrolls to `c` itself
(`parse_flat`, `parse_single`).
-/

namespace Flock

/-- A circuit descriptor: a flat circuit file, the circuit itself. -/
inductive Desc where
  | single (c : Circuit)

namespace Desc

/-- The circuit `D` describes. -/
def unroll : Desc → Circuit
  | single c => c

/-- Read the circuit file `bytes` as a descriptor: a flat file, as `HmOut.parse` reads it. -/
def parse (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V Desc :=
  match HmOut.parse tags bytes tables with
  | .ok c => .ok (.single c)
  | .error e => .error e

/-- A descriptor `Desc.parse` reads unrolls to the circuit `HmOut.parse` reads from the same file. -/
theorem parse_flat {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {D : Desc}
    (h : parse tags bytes tables = .ok D) : HmOut.parse tags bytes tables = .ok D.unroll := by
  unfold parse at h
  split at h
  · next c hc => cases h; exact hc
  · cases h

/-- A file `HmOut.parse` reads as `c` is the flat descriptor `single c`. -/
theorem parse_single {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmOut.parse tags bytes tables = .ok c) : parse tags bytes tables = .ok (.single c) := by
  unfold parse
  rw [h]

end Desc

end Flock
