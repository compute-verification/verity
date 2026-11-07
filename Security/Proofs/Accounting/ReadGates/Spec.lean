/-
What a table read's forms hold at an assignment `z` of its committed bits, in a read unit's port conventions
(`Accounting.ReadUnit`): a number from its bits least significant first (`numV`), bytes (`bytesV`) and a digest from eight
words in leaf order (`digestW`), and SHA-512 as the hash area's `Msg → Digest` (`sha512D`). Words hold bytes (`HoldsBytes`) in
that layout, and `ShaSound` is what the read's SHA-512 chain (`sha512W`) must do: hold `sha512D m` whenever its rows hold
over words holding a message `m` and its padding.
-/
module

public import Definitions.Accounting.ReadUnit
public import Proofs.Flock.Soundness.Discharge.Hm.Compress
public import Proofs.Accounting.ReadGates.Net

@[expose] public section

namespace Accounting.ReadGates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- The number forms hold at `z`, least significant first. -/
def numV (z : Var → Bool) (bs : List Lin) : ℕ := ReadUnit.num (bs.map (ev z))

/-- The byte a byte's forms hold at `z`. -/
def byteV (z : Var → Bool) (b : Byte) : UInt8 := UInt8.ofNat (numV z b)

/-- The bytes byte forms hold at `z`. -/
def bytesV (z : Var → Bool) (bs : List Byte) : Hash.Msg := bs.map (byteV z)

/-- The digest eight words hold at `z`, in leaf order (`ReadUnit.digestOf`): each word's bits least significant first,
its bytes big-endian. -/
def digestW (z : Var → Bool) (ws : List Word) : Verity.Digest := ReadUnit.digestOf (ws.flatten.map (ev z))

/-- A digest's forms: eight 64-bit words. -/
def DigestForms (ws : List Word) : Prop := ws.length = 8 ∧ Words64 ws

/-- SHA-512 as the hash area's `Msg → Digest`: the verifier's `Flock.Sha512.hash`, its 64 bytes. -/
def sha512D (m : Hash.Msg) : Verity.Digest := fun i => (Flock.Sha512.hash ⟨m.toArray⟩).data.toList.getD i.val 0

/-- **Words hold bytes** at `z`: `ws` are 64-bit words, eight bytes each, and bit `p` of word `i` is bit `p % 8` of byte
`8 i + 7 - p / 8` of `bs`: each word's bytes big-endian, each byte least significant bit first (`gates.lms`' layout). -/
def HoldsBytes (z : Var → Bool) (ws : List Word) (bs : List UInt8) : Prop :=
  Words64 ws ∧ 8 * ws.length = bs.length ∧
    ∀ i < ws.length, ∀ p < 64,
      ev z ((ws.getD i []).getD p Lin.zero) = (bs.getD (8 * i + 7 - p / 8) 0).toNat.testBit (p % 8)

/-- **The read's SHA-512 chain computes SHA-512**: whenever its rows hold, `sha512W` over words holding a message `m` and
its padding (`padTail`) returns words holding `sha512D m`. -/
def ShaSound : Prop :=
  ∀ ws : List Word, Sound (sha512W ws) fun z out =>
    ∀ m : Hash.Msg, HoldsBytes z ws (m ++ padTail m.length) → HoldsBytes z out (sha512D m).bytes

end Accounting.ReadGates
