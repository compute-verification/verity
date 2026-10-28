/-!
# Digests

A digest is 64 bytes: SHA-512's whole output (the one-hash rule), and the width every digest has in the served formats
(the advice tree's leaves and nodes, frame-v3-sha512's ids, leaves and nodes, hm96-sha512's `x`, `b` and `c`,
`sha512/row-seg/v1`'s segment digests), so a concatenation of digests splits back into them. Every area reads this one:
the advice root (`Verity.Advice`) and the hash area, which re-exports it as `Hash.Digest`. The verifier's code has its
own, `Flock.Digest`, as bytes.
-/

namespace Verity

/-- A digest: 64 bytes, SHA-512's whole output. -/
abbrev Digest := Fin 64 → UInt8

/-- A digest's 64 bytes, in order. -/
def Digest.bytes (d : Digest) : List UInt8 := List.ofFn d

end Verity
