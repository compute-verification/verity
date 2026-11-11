module

public import NetworkCheck.RecordsGrid
public import Proofs.Properties.NetworkCertifier.Check.Bytes

/-!
# Bytes the committed digest is written in

`k`-byte pieces of whole bits are `k` bytes (`bytesOfBits_size`). The hex lemmas the digest's checks read (`hexOf_inj`,
`flatten_inj`) live in `Proofs.Flock.Bytes`, under this namespace.
-/

@[expose] public section

open NetworkCheck.Check (bytesOfBits bytesOfBitsList)

set_option maxRecDepth 20000

namespace Properties.NetworkRecords.Bytes

theorem bytesOfBitsList_length : ∀ x : List Bool, (bytesOfBitsList x).length = x.length / 8
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest => by
    simp only [bytesOfBitsList, List.length_cons, bytesOfBitsList_length rest]
    omega
  | [] | [_] | [_, _] | [_, _, _] | [_, _, _, _] | [_, _, _, _, _] | [_, _, _, _, _, _] | [_, _, _, _, _, _, _] => by
    simp [bytesOfBitsList]

theorem bytesOfBits_size {x : List Bool} {k : Nat} (h : x.length = 8 * k) : (bytesOfBits x).size = k := by
  show (bytesOfBitsList x).toArray.size = k
  rw [List.size_toArray, bytesOfBitsList_length, h]
  omega

end Properties.NetworkRecords.Bytes
