import Definitions.Accounting.Wipe
import Definitions.Hash.Basic

/-!
# The streamed wipe's deployed labels (trusted)

The labels `verity/wipe/v0`'s memory challenger streams (`verity/core/protocols/pous/streamed_wipe.py`, `label`): label
`i` is SHA-512 of the label tag, the challenger's 32-byte key `K` and `i` in eight big-endian bytes,

  `L_i = SHA-512(LABEL_TAG ‖ K ‖ u64be i)`, `LABEL_TAG = "verity-pous/streamed-wipe/label/v1\0"`,

64 bytes, read as the label's 512 bits (`Accounting.Wipe.Labels`). `deployedLabels H N` is the table of the first `N`
labels under a hash `H`, as a function of the key's 256 bits: the labels `prf K` that `Accounting.Wipe.Strategy.wipe`
takes, at the key the challenger draws and keeps. A statement about the deployed wipe takes `H` to be the compiled
SHA-512.

A byte string is read as bits the way FIPS 180-4 reads a message: bit `8j + t` is bit `7 − t` of byte `j`, the most
significant first (`keyBytes`, `digestBits`).
-/

namespace Accounting.Wipe

/-- The streamed wipe's label tag: `verity-pous/streamed-wipe/label/v1` and its terminating zero byte. -/
def labelTag : Hash.Msg := Hash.cstr "verity-pous/streamed-wipe/label/v1"

/-- The key's 32 bytes: byte `j` holds the key's bits `8j` to `8j + 7`, the first the most significant. -/
def keyBytes (K : Pous.Bits 256) : Hash.Msg :=
  List.ofFn fun j : Fin 32 =>
    (∑ t : Fin 8, if K ⟨8 * j.val + t.val, by have := j.isLt; have := t.isLt; omega⟩ then 2 ^ (7 - t.val) else 0 :
      ℕ).toUInt8

/-- A digest's first 512 bits: bit `t` is bit `7 − (t mod 8)` of byte `⌊t / 8⌋`, a missing byte read as zero. -/
def digestBits (m : Hash.Msg) : Pous.Bits 512 := fun t => (m.getD (t.val / 8) 0).toNat.testBit (7 - t.val % 8)

/-- **The deployed labels** under the hash `H`: label `i` of the key `K` is the 512 bits of
`H(labelTag ‖ keyBytes K ‖ u64be i)`. -/
def deployedLabels (H : Hash.Msg → Hash.Msg) (N : ℕ) (K : Pous.Bits 256) : Labels N :=
  fun i => digestBits (H (labelTag ++ keyBytes K ++ Hash.be 8 i.val))

end Accounting.Wipe
