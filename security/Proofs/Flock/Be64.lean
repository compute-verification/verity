module

public import Proofs.Flock.Bytes

/-!
# A `be64` field determines its value

`be64` writes a 64-bit word as eight big-endian bytes, each byte the word's shifted remainder (`byte_toNat`), so two
words written alike are one (`be64_inj`), and so are two lengths under `2^64` whose length prefixes are one
(`be64_size_inj`).
-/

@[expose] public section

open Flock

set_option maxRecDepth 20000

namespace Properties.NetworkCertifier.Bytes

theorem byte_toNat (x : UInt64) (k : Nat) (hk : k < 64) :
    ((x >>> (UInt64.ofNat k)).toUInt8).toNat = x.toNat / 2 ^ k % 256 := by
  simp [UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow, Nat.mod_eq_of_lt hk]

theorem be64_inj {x y : UInt64} (h : be64 x = be64 y) : x = y := by
  have h' := congrArg bs h
  rw [be64_bs, be64_bs] at h'
  simp only [List.cons.injEq] at h'
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, -⟩ := h'
  have e : ∀ k, k < 64 → (x >>> (UInt64.ofNat k)).toUInt8 = (y >>> (UInt64.ofNat k)).toUInt8 →
      x.toNat / 2 ^ k % 256 = y.toNat / 2 ^ k % 256 := fun k hk he => by
    rw [← byte_toNat x k hk, ← byte_toNat y k hk, he]
  have a0 := e 56 (by omega) h0
  have a1 := e 48 (by omega) h1
  have a2 := e 40 (by omega) h2
  have a3 := e 32 (by omega) h3
  have a4 := e 24 (by omega) h4
  have a5 := e 16 (by omega) h5
  have a6 := e 8 (by omega) h6
  have a7 := e 0 (by omega) h7
  have hx := x.toNat_lt
  have hy := y.toNat_lt
  apply UInt64.toNat_inj.mp
  simp only [Nat.reducePow] at a0 a1 a2 a3 a4 a5 a6 a7 hx hy ⊢
  omega

/-- `field`'s length prefix of a string under `2^64` bytes determines its length. -/
theorem be64_size_inj {n m : Nat} (hn : n < 2 ^ 64) (hm : m < 2 ^ 64) (h : be64 n.toUInt64 = be64 m.toUInt64) :
    n = m := by
  have := congrArg UInt64.toNat (be64_inj h)
  simp only [Nat.toUInt64, UInt64.toNat_ofNat'] at this
  rwa [Nat.mod_eq_of_lt hn, Nat.mod_eq_of_lt hm] at this

end Properties.NetworkCertifier.Bytes
