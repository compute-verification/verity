module

public import Definitions.Accounting.ReadUnit
public import Properties.Verity.Game
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.BitVec.Basic
import all Init.Data.ByteArray.Basic
import all Init.Data.Fin.Fold
import all Init.Data.List.Basic
import all Init.Data.List.BasicAux
import all Init.Data.List.Control
import all Init.Data.List.OfFn
import all Init.Data.List.Pairwise
import all Init.Data.Nat.Basic
import all Init.Data.Option.Basic
import all Init.Data.Range.Basic
import all Init.Data.String.Basic
import all Init.Data.String.Bootstrap
import all Init.Data.String.Defs
import all Init.Data.String.Extra
import all Init.Data.UInt.Basic
import all Init.Data.UInt.BasicAux

/-!
# What the counterexample read's checks hash has no SHA-512 collision

`TableReadBinding_nontrivial`'s read (`Witness.R₀`) opens its one-row table at index 0. Its own opening holds no row
bits and a salt of ones (`op₀`), its table's opening 16 zero bits and the same salt (`o₀`), both under the row-leaf
verifier of one row at its own rank, the domain id of ones and the served hm96 key (`V₀`). The messages those two checks
hash under SHA-512 (`L₀`, what `Reads.hashedAt` lists there) have pairwise different digests, which the kernel computes
(`digests₀`), so the reads' collision finder finds no collision there (`no_collision₀`). The shared salt is the
expensive part of each hm96 leaf (its mask), so the kernel computes it once.

The `import all` lines let the kernel unfold the core definitions SHA-512 and the key's derivation run on (`ByteArray`,
`Std.Legacy.Range`'s loops, `String.toUTF8`, the `UInt` operations), which a module otherwise sees only by signature.
`digests₀` reads the key as its bytes (`keyK`, by `key_eq`), and each mask bit through `xorFold` (`maskBit_eq`), which
reads a key bit off the key's number (`bit_eq_testBit`) and forces the fold's accumulator at each step.
-/

@[expose] public section

namespace Accounting.ReadUnit.Witness

/-- The domain id the read's gates hold: every bit set. -/
def d₀ : Verity.Digest := digestOf (List.ofFn fun _ : Fin 512 => true)

/-- The read's opening: no row bits, a salt of ones, no siblings. -/
def op₀ : Hash.RowLeaf.Opening := ⟨[], List.replicate 192 255, []⟩

/-- The table's one row: 16 zero bits, a salt of ones, no siblings. -/
def o₀ : Hash.RowLeaf.Opening := ⟨List.replicate 16 false, List.replicate 192 255, []⟩

/-- The read's row-leaf verifier: one row at its own rank, the domain id `d₀`, the served hm96 key. -/
def V₀ : Hash.RowLeaf.Verifier :=
  Hash.EmbeddingRead.verifier ⟨d₀, 1, fun r => r⟩ Flock.Hm96.Default512.key.data.toList

/-- The salt of ones as bytes. -/
theorem salt_ones : bytesOf (List.ofFn fun _ : Fin (8 * Hash.Hm96.SALT_BYTES) => true) = List.replicate 192 255 := by
  decide +kernel

/-- What the read's check and its table's check hash at index 0 (`Reads.hashedAt` with `H = G = SHA-512`). -/
def L₀ : List Hash.Msg :=
  Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 V₀ 0 op₀ ++
      Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 V₀ 0 o₀ ++
    (Hash.RowLeaf.innerHashed Verity.sha512 V₀ op₀ ++ Hash.RowLeaf.innerHashed Verity.sha512 V₀ o₀)

/-- The served hm96 key's bytes: `H(label ‖ be32 i)` for `i < 4`, its top bit cleared. -/
def keyK : List UInt8 := [
  0x27, 0x56, 0x99, 0x39, 0x48, 0xd3, 0xe0, 0xce, 0x33, 0xda, 0xec, 0x76, 0x12, 0xff, 0x71, 0xb8,
  0x61, 0xbd, 0x15, 0x8e, 0x09, 0xb4, 0x18, 0xa1, 0xe5, 0xbd, 0x28, 0x53, 0xa0, 0x7c, 0x75, 0xb2,
  0xf7, 0x3c, 0x93, 0xa6, 0xb3, 0x06, 0x8b, 0xb3, 0x5f, 0x5f, 0x73, 0x24, 0x38, 0xa7, 0xbe, 0x3f,
  0xc7, 0x6a, 0x2f, 0xc9, 0x56, 0x35, 0xee, 0x77, 0xd6, 0x43, 0x59, 0xcf, 0x6f, 0xea, 0x3e, 0x86,
  0xf9, 0x5f, 0x6f, 0x8e, 0x58, 0xec, 0x9b, 0x6f, 0x31, 0x11, 0xac, 0xff, 0x42, 0x9d, 0x68, 0xcc,
  0x2a, 0x6d, 0xa8, 0x98, 0xe4, 0x26, 0x4a, 0x5c, 0x69, 0xa1, 0x7a, 0x17, 0xe1, 0x71, 0xe7, 0xcd,
  0x0c, 0xa8, 0x1e, 0xaa, 0xe7, 0x8a, 0x36, 0x47, 0x9f, 0x4b, 0x6f, 0x8c, 0x21, 0xe7, 0xea, 0x87,
  0x9f, 0x29, 0x85, 0x94, 0x43, 0x98, 0x09, 0x8a, 0xaf, 0x8f, 0x5c, 0x73, 0x7a, 0xe0, 0x92, 0x03,
  0xe6, 0x34, 0x1d, 0x5c, 0x1b, 0xe0, 0xdc, 0x35, 0x93, 0x93, 0xc7, 0x66, 0xdf, 0x01, 0x56, 0x24,
  0x32, 0x5a, 0xd2, 0x43, 0xd0, 0xd8, 0x44, 0xe7, 0x87, 0x37, 0x03, 0x14, 0xb9, 0xfc, 0x2c, 0x4e,
  0x1c, 0xce, 0x67, 0xea, 0x2c, 0x09, 0x44, 0xe9, 0x73, 0x7c, 0x06, 0xeb, 0xec, 0x49, 0xe9, 0x5b,
  0x33, 0xbc, 0x1e, 0xa2, 0x38, 0x2c, 0x5c, 0x10, 0x03, 0xb2, 0x49, 0xf7, 0xa3, 0x3a, 0xb8, 0x96,
  0x63, 0x22, 0xfb, 0xd5, 0xe9, 0x63, 0xa9, 0x8c, 0x39, 0xb8, 0x5d, 0xc0, 0x36, 0x79, 0xe6, 0xd2,
  0x13, 0x33, 0xe9, 0x93, 0xef, 0x91, 0x12, 0xee, 0x58, 0xf3, 0x7e, 0x84, 0x44, 0xc5, 0xfe, 0x12,
  0x01, 0x16, 0x6b, 0xb1, 0xcd, 0xa5, 0x80, 0x1f, 0x64, 0x0d, 0x9b, 0xe2, 0x39, 0xd4, 0xc7, 0x70,
  0x55, 0x7b, 0xe8, 0x25, 0x86, 0x2e, 0x6f, 0x0e, 0x7c, 0x43, 0xd3, 0x29, 0xaa, 0x28, 0x5b, 0x29]

theorem key_eq : Flock.Hm96.Default512.key.data.toList = keyK := by
  decide +kernel

/-- A byte string as a number, byte `k` its base-256 digit `k`. -/
def natOf : List UInt8 → ℕ
  | [] => 0
  | b :: l => 2 ^ 8 * natOf l + b.toNat

theorem beq_one (x : ℕ) : (x == 1) = decide (x = 1) := by
  by_cases h : x = 1 <;> simp [h]

/-- Bit `t` of a byte string is bit `t` of its number. -/
theorem bit_eq_testBit : ∀ (l : List UInt8) (t : ℕ), Hash.Hm96.bit l t = (natOf l).testBit t
  | [], t => by simp [Hash.Hm96.bit, natOf]
  | b :: l, t => by
    rw [natOf, Nat.testBit_two_pow_mul_add _ b.toNat_lt]
    split
    · rename_i ht
      have h0 : t / 8 = 0 := Nat.div_eq_of_lt ht
      have h1 : t % 8 = t := Nat.mod_eq_of_lt ht
      rw [Hash.Hm96.bit, h0, h1, List.getD_cons_zero, Nat.testBit_eq_decide_div_mod_eq, beq_one]
    · rename_i ht
      have h0 : t / 8 = (t - 8) / 8 + 1 := by omega
      have h1 : t % 8 = (t - 8) % 8 := by omega
      rw [← bit_eq_testBit l (t - 8), Hash.Hm96.bit, Hash.Hm96.bit, h0, h1, List.getD_cons_succ]

/-- `Hm96.maskBit`'s fold over the key's number `k`, its accumulator forced at each step: the kernel then reads a key
bit as one shift of `k`, and never builds the xor of 1536 terms that a salt of ones would leave it to evaluate at the
end. -/
def xorFold (k i : ℕ) : List ℕ → Bool → Bool
  | [], acc => acc
  | j :: js, acc =>
    match xor acc (k.testBit (i + j)) with
    | true => xorFold k i js true
    | false => xorFold k i js false

theorem foldl_xorFold (key : Hash.Msg) (i : ℕ) : ∀ (l : List ℕ) (acc : Bool),
    l.foldl (fun acc j => xor acc (Hash.Hm96.bit key (i + j))) acc = xorFold (natOf key) i l acc
  | [], _ => rfl
  | j :: js, acc => by
    rw [List.foldl_cons, xorFold, bit_eq_testBit]
    cases xor acc ((natOf key).testBit (i + j)) <;> exact foldl_xorFold key i js _

theorem maskBit_eq (key y : Hash.Msg) (i : ℕ) :
    Hash.Hm96.maskBit key y i =
      xorFold (natOf key) i ((List.range (8 * Hash.Hm96.SALT_BYTES)).filter (Hash.Hm96.bit y)) false :=
  foldl_xorFold key i _ false

-- The kernel hashes every message of `L₀` and computes each mask bit of its two openings, about three minutes.
set_option maxHeartbeats 8000000 in
theorem digests₀ : (L₀.dedup.map fun m => (Verity.sha512 m).bytes).Nodup := by
  simp only [L₀, V₀, key_eq, Hash.RowLeaf.outerHashed, Hash.RowLeaf.leafMsg, Hash.Hm96.hashed, Hash.Hm96.leaf,
    Hash.Hm96.commitString, Hash.Hm96.mask, maskBit_eq]
  decide +kernel

theorem no_collision₀ : ¬ ∃ x ∈ L₀, ∃ y ∈ L₀, Hash.Collision (fun m => (Verity.sha512 m).bytes) x y := by
  rintro ⟨x, hx, y, hy, hxy, he⟩
  exact hxy (List.inj_on_of_nodup_map digests₀ (List.mem_dedup.2 hx) (List.mem_dedup.2 hy) he)

end Accounting.ReadUnit.Witness
