module

public import Batteries.Data.ByteArray
public import Definitions.Core.CoinBytes
public import Proofs.Flock.Soundness.Assumptions
public import Proofs.Flock.Soundness.ExecDraw
public import Proofs.Properties.NetworkCertifier.DeviceOutput.Frames
public import Properties.Challenger
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Soundness.ZK.Hiding
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# A3 at the game's own bits: the bytes a sampled sample reads are uniform

In the causal game (`Verity.releases`) the challenger releases each coin as `8 * B` uniform bits (`B` is
`Challenger.COIN_BYTES`), and a sample reads the concatenation of the coins it was released as bytes, bit `t` of a coin's
bits being bit `t mod 8` of its byte `⌊t/8⌋` (`Properties.Challenger.bitsOf`). So over the game's bits for `k`
coins, `Ω = Fin k → Fin (8 * B) → Bool`, the sample's byte source is `coinBytes k B`: one stream of `k * B` bytes, coin
`c`'s byte `q` at `c * B + q`. It is a bijection (`coinBytesEquiv`), so it carries the uniform law on the bits to the
uniform law on the bytes, which is `Assumptions.UniformRandomBytes` at it (`uniformRandomBytes_coinBytes`): A3 holds
there as a theorem, with no premise on the operating system.

`bitsAt_coinBytes` and `testBit_coinBytes` say what the bytes are; `bitsOf_streamOf_coinBytes` that their bits, as
`bitsOf` reads them, are the coins' bits in order; and `streamOf_coinBytes_of_bitsOf` that they are the only bytes with
those bits (`bitsOf_injective`), so the bytes a sample reads, when their bits are the game's coins, are
`streamOf (coinBytes k B ω 0)`. `bitsOf_append` splits the bits of a concatenation of coins.

`byteIdx`, `bitsAt` and `coinBytesEquiv`, with the lemmas they are built from, are `Definitions.Core.CoinBytes`'s.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness
open FlockSoundness.Audit.Law (streamOf)
open Properties.Challenger (bitsOf)

variable {k B : ℕ}

variable (k B) in
/-- **The byte source on the game's bits**: the one stream of `k * B` bytes whose bits are the `k` coins' bits, coin
`c`'s byte `q` at `c * B + q`. -/
noncomputable def coinBytes (ω : Fin k → Fin (8 * B) → Bool) : Fin 1 → Fin (k * B) → Fin 256 :=
  (coinBytesEquiv k B).symm ω

variable (k B) in
/-- **A3 at the game's bits**: the bytes `coinBytes` reads off uniform coin bits are uniform bytes. -/
theorem uniformRandomBytes_coinBytes :
    FlockSoundness.Assumptions.UniformRandomBytes (fun _ : Fin 1 => k * B) (coinBytes k B) := fun E =>
  FlockSoundness.ZK.prCoin_congr_equiv (coinBytesEquiv k B).symm E

theorem bitsAt_coinBytes (ω : Fin k → Fin (8 * B) → Bool) : bitsAt k B (coinBytes k B ω) = ω :=
  (coinBytesEquiv k B).apply_symm_apply ω

theorem coinBytes_eq_iff {ω : Fin k → Fin (8 * B) → Bool} {F : Fin 1 → Fin (k * B) → Fin 256} :
    coinBytes k B ω = F ↔ bitsAt k B F = ω :=
  (coinBytesEquiv k B).symm_apply_eq.trans eq_comm

/-- Bit `j < 8` of coin `c`'s byte `q < B` is the coin's bit `8 * q + j`. -/
theorem testBit_coinBytes (ω : Fin k → Fin (8 * B) → Bool) (c : Fin k) (p : Fin (k * B)) {q j : ℕ}
    (hp : p.val = c.val * B + q) (hq : q < B) (hj : j < 8) :
    (coinBytes k B ω 0 p).val.testBit j = ω c ⟨8 * q + j, by omega⟩ := by
  rw [← bitsAt_spec (coinBytes k B ω) c p hp hq hj, bitsAt_coinBytes]

/-! ## The bytes as `bitsOf` reads them -/

/-- A byte's bits, least significant first: `bitsOf`'s per byte. -/
def byteBits (x : UInt8) : List Bool := (List.range 8).map fun i => x.toNat.testBit i

/-- The byte with bits `l`, least significant first. -/
def ofBits (l : List Bool) : ℕ := l.foldr (fun b n => (if b then 1 else 0) + 2 * n) 0

theorem ofBits_bits : ∀ n < 256, ofBits ((List.range 8).map fun i => n.testBit i) = n := by decide +kernel

theorem byteBits_inj {x y : UInt8} (h : byteBits x = byteBits y) : x = y := by
  have hx := ofBits_bits x.toNat x.toNat_lt
  have hy := ofBits_bits y.toNat y.toNat_lt
  change ofBits (byteBits x) = x.toNat at hx
  change ofBits (byteBits y) = y.toNat at hy
  rw [h] at hx
  exact UInt8.toNat_inj.1 (hx.symm.trans hy)

theorem flatMap_byteBits_inj : ∀ {l l' : List UInt8}, l.flatMap byteBits = l'.flatMap byteBits → l = l'
  | [], [], _ => rfl
  | [], y :: l', h => by simp [byteBits] at h
  | x :: l, [], h => by simp [byteBits] at h
  | x :: l, y :: l', h => by
    rw [List.flatMap_cons, List.flatMap_cons] at h
    obtain ⟨h1, h2⟩ := List.append_inj h (by simp [byteBits])
    rw [byteBits_inj h1, flatMap_byteBits_inj h2]

theorem bitsOf_eq (b : ByteArray) : bitsOf b = b.data.toList.flatMap byteBits :=
  congrArg (fun l : List UInt8 => l.flatMap byteBits) (Properties.CertifierNetwork.Proofs.toList_eq b)

theorem byteBits_ofFin (a : Fin 256) :
    byteBits (UInt8.ofFin a) = List.ofFn fun j : Fin 8 => a.val.testBit j.val := by
  refine List.ext_getElem (by simp [byteBits]) fun i h1 h2 => ?_
  simp only [byteBits, List.getElem_map, List.getElem_range, List.getElem_ofFn]
  rfl

/-- `bitsOf` is one to one: a byte string is the only one with its bits. -/
theorem bitsOf_injective : Function.Injective bitsOf := by
  intro a b h
  rw [bitsOf_eq, bitsOf_eq] at h
  exact ByteArray.ext (Array.toList_inj.1 (flatMap_byteBits_inj h))

theorem bitsOf_append (a b : ByteArray) : bitsOf (a ++ b) = bitsOf a ++ bitsOf b := by
  rw [bitsOf_eq, bitsOf_eq, bitsOf_eq, ByteArray.toList_data_append, List.flatMap_append]

/-- **The bytes' bits are the coins' bits, in order**: `bitsOf` of the stream `coinBytes` makes is coin `0`'s `8 * B`
bits, then coin `1`'s, and so on. -/
theorem bitsOf_streamOf_coinBytes (ω : Fin k → Fin (8 * B) → Bool) :
    bitsOf (streamOf (L := k * B) (coinBytes k B ω 0)) = (List.ofFn fun c => List.ofFn (ω c)).flatten := by
  rw [bitsOf_eq]
  show (Array.ofFn fun p => UInt8.ofFin (coinBytes k B ω 0 p)).toList.flatMap byteBits = _
  rw [Array.toList_ofFn, List.flatMap_def, List.map_ofFn, List.ofFn_mul (m := k) (n := B), List.flatten_flatten,
    List.map_ofFn]
  refine congrArg List.flatten (congrArg List.ofFn (funext fun c => ?_))
  simp only [Function.comp_def]
  rw [List.ofFn_mul' (ω c)]
  refine congrArg List.flatten (congrArg List.ofFn (funext fun q => ?_))
  show byteBits _ = _
  rw [byteBits_ofFin]
  exact congrArg List.ofFn (funext fun j =>
    testBit_coinBytes ω c ⟨c.val * B + q.val, idx_lt c q.isLt⟩ rfl q.isLt j.isLt)

/-- **The only bytes with the coins' bits**: bytes whose bits, as `bitsOf` reads them, are the `k` coins' bits in order
are the stream `coinBytes` makes. -/
theorem streamOf_coinBytes_of_bitsOf (ω : Fin k → Fin (8 * B) → Bool) {ξ : ByteArray}
    (h : bitsOf ξ = (List.ofFn fun c => List.ofFn (ω c)).flatten) :
    ξ = streamOf (L := k * B) (coinBytes k B ω 0) :=
  bitsOf_injective (h.trans (bitsOf_streamOf_coinBytes ω).symm)

end Verity.Sampled
