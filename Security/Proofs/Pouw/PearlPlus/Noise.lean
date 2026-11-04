import PouwPearlPlus.Words
import Definitions.Pouw.PearlC.NoiseSha512

/-!
# Pearl+'s bridge, part (b), the noise: the check's -h3 noise is `NoiseSha512`'s

The check hashes with the compiled SHA-512 every tree reads (`Flock.Sha512.hash`) and keys a row's seed by its
`sha512/row-seg/v1` digest (`Flock.HmRow.segDigest`). The deployed semantics (`sha512Sem`) draws the same lines through the
spec's SHA-512 (`Pouw.PearlC.Sha512.digest`) and row digest (`NoiseSha512.rowDigest`). Each statement here says the
check's value is the spec's on the same bytes, and where the check returns a value, the spec's line or seed is it.
-/

namespace Pouw.PearlPlus.Bridge

open PouwCheck.PearlPlus

/-- A byte array's bytes, as the spec's lists of naturals. -/
def bytesNat (b : ByteArray) : List ℕ := b.data.toList.map UInt8.toNat

/-- **The compiled SHA-512 is the spec's.** -/
def HashIsTheSpec : Prop :=
  ∀ m : ByteArray, bytesNat (Flock.Sha512.hash m) = Pouw.PearlC.Sha512.digest (bytesNat m)

/-- **`sha512/row-seg/v1` on whole bytes is the spec's row digest.** -/
def SegDigestIsRowDigest : Prop :=
  ∀ r : ByteArray, bytesNat (Flock.HmRow.segDigest (8 * r.size) r) = Pouw.PearlC.NoiseSha512.rowDigest (bytesNat r)

/-- The roles -h3 hashes under. -/
def roles : List String := ["seed-A", "seed-B", "noise-line", "line"]

/-- **-h3's labelled hash and its seeds are the spec's.** -/
def SeedsAreTheSpec : Prop :=
  (∀ role ∈ roles, ∀ m : ByteArray,
    bytesNat (Words.s512 role m) = Pouw.PearlC.NoiseSha512.s512 role (bytesNat m)) ∧
  (∀ salt rootB : ByteArray,
    bytesNat (Words.seedB salt rootB) = Pouw.PearlC.NoiseSha512.seedB (bytesNat salt) (bytesNat rootB)) ∧
  ∀ (salt rootB row : ByteArray) (u i : ℕ), bytesNat (Words.rowSeed salt rootB u i row) =
    Pouw.PearlC.NoiseSha512.rowSeed (bytesNat salt) (bytesNat row) (bytesNat rootB) u i

/-- **A noise line is the spec's**, at E's norm 256 and F's line norm 16: where the check's line is a value, its 32 codes
are `NoiseSha512.sampleLine`'s. -/
def LineIsTheSpec : Prop :=
  ∀ (seed : ByteArray) (side factor line norm32 : ℕ) (norm : ℚ) (codes : Array ℕ), side < 256 → factor < 256 →
    (norm32 = Words.NORM_E ∧ norm = 256 ∨ norm32 = Words.NORM_F ∧ norm = 16) →
    Words.sampleLine (Words.lineKey seed) side factor line norm32 = .ok codes →
      codes.size = 32 ∧ ∀ t : Fin Pouw.PearlC.Params.pi.r,
        codes[t.val]! = (Pouw.PearlC.NoiseSha512.sampleLine (bytesNat seed) side factor line norm t).val

/-- **The fixed operand's noise is `unitNoise`'s**: where `fixedNoise` is a value, its seed is the spec's `seed_B` and its
k rows of `F_A` and `F_B` are the spec's basis lines. -/
def FixedNoiseIsTheSpec : Prop :=
  ∀ (salt rootB : ByteArray) (k : ℕ) (wn : Words.FixedNoise), Words.fixedNoise salt rootB k = .ok wn →
    bytesNat wn.seedB = Pouw.PearlC.NoiseSha512.seedB (bytesNat salt) (bytesNat rootB) ∧
      wn.fa.size = k ∧ wn.fb.size = k ∧ ∀ l < k, ∀ t : Fin Pouw.PearlC.Params.pi.r,
        wn.fa[l]![t.val]! = (Pouw.PearlC.NoiseSha512.sampleLine (bytesNat wn.seedB) 0 1 l 16 t).val ∧
          wn.fb[l]![t.val]! = (Pouw.PearlC.NoiseSha512.sampleLine (bytesNat wn.seedB) 1 1 l 16 t).val

end Pouw.PearlPlus.Bridge
