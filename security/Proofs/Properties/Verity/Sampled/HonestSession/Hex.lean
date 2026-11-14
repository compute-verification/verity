module

public import Flock.HmRow

/-!
# A 64-byte string's hex is a SHA-512 digest's

`HmRow.isSha512Hex` holds of `hexOf b` for every 64-byte `b`, so a check of a spliced digest's form (`hexOf (Sha512.hash
m)`) passes without evaluating the hash.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Hex

open Flock

theorem length_hexOf (b : ByteArray) : (hexOf b).length = 2 * b.size := by
  sorry -- piece P6 (hex)

theorem isSha512Hex_hexOf {b : ByteArray} (h : b.size = 64) : HmRow.isSha512Hex (hexOf b) = true := by
  sorry -- piece P6 (hex)

theorem size_hash (m : ByteArray) : (Sha512.hash m).size = 64 := by
  sorry -- piece P6 (hex): `Properties.NetworkCertifier.Bytes.hash_size`

end Verity.Sampled.Instance.HonestSession.Hex
