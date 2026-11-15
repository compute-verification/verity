module

public import Proofs.Flock.Bytes

/-!
# A `be64` field determines its value

`be64` writes a 64-bit word as eight big-endian bytes, each byte the word's shifted remainder (`byte_toNat`), so two
words written alike are one (`be64_inj`), and so are two lengths under `2^64` whose length prefixes are one
(`be64_size_inj`). The three are `Properties.Certifier.Bytes`'s, in `Proofs.Flock.Bytes`, which this module re-exports.
-/
