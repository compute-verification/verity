module

/-! A digest: 64 bytes, SHA-512's whole output and the width of every digest in the served formats. -/

@[expose] public section

namespace Verity

/-- A digest: 64 bytes, SHA-512's whole output. -/
abbrev Digest := Fin 64 → UInt8

/-- A digest's 64 bytes, in order. -/
def Digest.bytes (d : Digest) : List UInt8 := List.ofFn d

end Verity
