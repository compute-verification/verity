import Properties.Verity.Game
import Definitions.Accounting.WipeLabels

/-!
# The deployed labels against the Python's, on fixed keys and indices

`Accounting.Wipe.deployedLabels` at the compiled SHA-512 (`Verity.sha512Msg`), checked by the kernel on four keys and
indices against the labels `verity/core/protocols/pous/streamed_wipe.py`'s `label` gives there, written in hex.
`verity/core/protocols/pous/check/tests/test_wipe_labels.py` reads the vectors from this file and checks each against
`label`, so the two agree on them.

Hex is read as bits the most significant first (`bitsOfHex`), the order FIPS 180-4 reads a message in, so a vector holds
only if `keyBytes` and `digestBits` read bytes in that order: with either read least significant first, every vector
whose key is not all ones fails. The keys and indices: `00 01 … 1f` at `0` and `1`; one bit at each end of the key
(`80 00 … 00 01`) at an index whose eight bytes differ (`0x0102030405060708`); all ones at the last index, `2^64 − 1`.
-/

namespace Accounting.Wipe.Vectors

/-- The value of a lowercase hex digit. -/
def hexDigit (c : Char) : ℕ := if c.isDigit then c.toNat - 48 else c.toNat - 87

/-- The first `n` bits of a lowercase hex string, the most significant first: bit `t` is bit `3 − t % 4` of digit
`t / 4`. -/
def bitsOfHex (n : ℕ) (s : String) : Pous.Bits n :=
  fun t => (hexDigit (s.toList.getD (t.val / 4) '0')).testBit (3 - t.val % 4)

/-- Label `i` of the deployed labels under the key `K` (64 hex digits), at the compiled SHA-512. -/
def label (K : String) (i : ℕ) : Pous.Bits 512 :=
  deployedLabels Verity.sha512Msg (i + 1) (bitsOfHex 256 K) (Fin.last i)

theorem label_0 :
    label "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f" 0 =
      bitsOfHex 512 ("180e0cbb08b98b2fda0b65b3a7710e0bc589ef92a63ab1d589f4c171b668e9dc" ++
        "908b956472d1cfe6b74d90cc5c40ee6d6155953b996b6a0da8f617ccd2370368") := by
  decide +kernel

theorem label_1 :
    label "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f" 1 =
      bitsOfHex 512 ("45c86c958e76430dfd7a2c9bf9f612b7a319017570d9d50e0caea5c89347256e" ++
        "6fdf65beae4c1a861186549a7832055bc78be16dd99611a724a7cb4cecae9200") := by
  decide +kernel

theorem label_2 :
    label "8000000000000000000000000000000000000000000000000000000000000001" 72623859790382856 =
      bitsOfHex 512 ("927a30b4513b166e2d26490837bac545361920d3db7385856046d32ca92106e2" ++
        "3f1d2a752a16311e51e0049476700707a7bcd6a5c17fb31104597e534a99a234") := by
  decide +kernel

theorem label_3 :
    label "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff" 18446744073709551615 =
      bitsOfHex 512 ("7ed604fd2be4c3774478856c33cf19f75f39e0d59c8df875abecd1b777f21d96" ++
        "1229ff7880101944621f602a0d07132fa7704f518c876382007fdc3aebe3077f") := by
  decide +kernel

end Accounting.Wipe.Vectors
