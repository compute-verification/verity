import FlockSoundness.Discharge.ZkTree.Defs

/-!
# The live coin tree's layout (`coin-tree/hm96-sha512/v2`, `live/src/coin_tree.rs`)

`CoinTree::draw_with` reads its randomness in one order: the 256-byte session key, bit 2047 cleared (bit 7 of byte 255),
then for each stream of the spec in leaf order and each of its `2^rounds_log` rounds, the round's block (`16·block`
bytes, `block` F128 words, each its low then its high 64 bits little-endian) and the round's salt (192 bytes). Each
piece is its own read of uniform bytes, so they are independent and uniform, and each decoder below is a bijection:
* `keyBits`: the key's bits `0..2046` (bit `t` is bit `t % 8` of byte `t / 8`, as `Rules::mask` reads the key's
  little-endian words), so the uniform key with bit 2047 cleared is a uniform `Key`. hm96's mask `M(K)·y` reads key bits
  `i + j ≤ 511 + 1535 = 2046` only, so the 2047 bits `Key` keeps are exactly the bits the leaf reads (gap 2);
* `saltBits`: the salt's 1536 bits, bit `j` bit `j % 8` of byte `j / 8` (`Rules::mask`'s reading of `y`);
* `blockAt`: the block's words, a `LiveBlk block`.
`tapeAt` is where round `k` of stream `s` starts. `maskBytes` is `CROnly.hankel` on the decoded key and salt, written out
as `Rules::mask` writes its 64 bytes.

A round of `n ≤ block` coins is the block's first `n` words (`CoinTree::open`), the rest of the block the `block - n` it
does not open: `drLive block n`, through `blockDec`. The session's link points are round 0 of the session stream's first
`2·m` words, point `k` the `k`-th run of `m` (`self.points = c.chunks(m)`): `dlLive block m`. Past `block` coins `open`
refuses, and the decoders are `none`: the round's coins are fresh in `liveTableC`, and the real run refuses there
(`ZkTree.prR`).

`tests/test_lean_coin_layout.py` holds these decoders to the live crate's `CoinTree` on vectors it draws
(`tests/fixtures/coin_tree_rust.json`, `tests/fixtures/coin_tree_lean.json`).
-/

namespace FlockSoundness.Discharge.ZkTree

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128

/-- A block of the live coin tree: `block` coins. -/
abbrev LiveBlk (block : ℕ) : Type := Fin block → 𝔽

/-- **A round's first `n` coins of a block of `block`, and the rest.** -/
def blockDec {block n : ℕ} (h : n ≤ block) : Dec (LiveBlk block) (Fin n → 𝔽) where
  Rest := Fin (block - n) → 𝔽
  e := (Equiv.sumArrowEquivProdArrow (Fin n) (Fin (block - n)) 𝔽).symm.trans
    (Equiv.arrowCongr (finSumFinEquiv.trans (finCongr (Nat.add_sub_cancel' h))) (Equiv.refl 𝔽))

/-- **A round's `n` coins on the live coin tree**: the block's first `n`, when `n ≤ block`. -/
def drLive (block n : ℕ) : Option (Dec (LiveBlk block) (Fin n → 𝔽)) :=
  if h : n ≤ block then some (blockDec h) else none

/-- **The link points on the live coin tree**: `2·m` coins of the session stream's round 0, point `k`'s coordinate `t`
the block's word `k·m + t`, when `2·m ≤ block`. -/
def dlLive (block m : ℕ) : Option (Dec (LiveBlk block) (Fin 2 → Fin m → 𝔽)) :=
  if h : 2 * m ≤ block then
    some { Rest := (blockDec h).Rest
           e := (Equiv.prodCongr ((Equiv.curry (Fin 2) (Fin m) 𝔽).symm.trans
             (Equiv.arrowCongr finProdFinEquiv (Equiv.refl 𝔽))) (Equiv.refl _)).trans (blockDec h).e }
  else none

theorem blockDec_coin {block n : ℕ} (h : n ≤ block) (b : LiveBlk block) (i : Fin n) :
    ((blockDec h).e.symm b).1 i = b (Fin.castLE h i) := by
  simp [blockDec, Equiv.sumArrowEquivProdArrow, Equiv.arrowCongr]
  rfl

theorem drLive_coin {block n : ℕ} (h : n ≤ block) (b : LiveBlk block) (i : Fin n) :
    ∃ d, drLive block n = some d ∧ (d.e.symm b).1 i = b (Fin.castLE h i) :=
  ⟨_, dif_pos h, blockDec_coin h b i⟩

theorem dlLive_point {block m : ℕ} (h : 2 * m ≤ block) (b : LiveBlk block) (k : Fin 2) (t : Fin m) :
    ∃ d, dlLive block m = some d ∧ (d.e.symm b).1 k t = b ⟨k.val * m + t.val, by
      have := Nat.mul_le_mul_right m (Nat.le_of_lt_succ k.isLt); have := t.isLt; omega⟩ := by
  refine ⟨_, dif_pos h, ?_⟩
  show b (finCongr (Nat.add_sub_cancel' h) (finSumFinEquiv (Sum.inl (finProdFinEquiv (k, t))))) = _
  congr 1
  apply Fin.ext
  simp [finProdFinEquiv]
  ring

theorem drLive_none {block n : ℕ} (h : block < n) : drLive block n = none := by
  simp [drLive]; omega

theorem dlLive_none {block m : ℕ} (h : block < 2 * m) : dlLive block m = none := by
  simp [dlLive]; omega

/-! ## The tape -/

/-- Where round `k` of stream `s` starts on the tape: after the key and every earlier (block, salt). -/
def tapeAt (rl block s k : ℕ) : ℕ := 256 + (s * 2 ^ rl + k) * (16 * block + 192)

/-- The little-endian 64-bit word at byte `a`. -/
def u64le (b : ByteArray) (a : ℕ) : UInt64 :=
  (List.range 8).foldl (fun acc j => acc ||| ((b.get! (a + j)).toUInt64 <<< (8 * j).toUInt64)) 0

/-- The F128 word at byte `a`: its low 64 bits, then its high. -/
def wordAt (b : ByteArray) (a : ℕ) : 𝔽 := GF128.of ⟨u64le b a, u64le b (a + 8)⟩

/-- The block of `block` words at byte `a`. -/
def blockAt (b : ByteArray) (a block : ℕ) : LiveBlk block := fun w => wordAt b (a + 16 * w.val)

/-- Bit `t` of the bytes at `a`: bit `t % 8` of byte `t / 8`. -/
def bitAt (b : ByteArray) (a t : ℕ) : ZMod 2 := if (b.get! (a + t / 8)).toNat.testBit (t % 8) then 1 else 0

/-- The session key's bits `0..2046`, from its 256 bytes at `a`. -/
def keyBits (b : ByteArray) (a : ℕ) : Key (ZMod 2) := fun t => bitAt b a t.val

/-- A salt's 1536 bits, from its 192 bytes at `a`. -/
def saltBits (b : ByteArray) (a : ℕ) : Salt (ZMod 2) := fun j => bitAt b a j.val

/-- **hm96's mask `M(K)·y`** on the decoded key and salt, as `Rules::mask`'s 64 bytes. -/
def maskBytes (κ : Key (ZMod 2)) (y : Salt (ZMod 2)) : ByteArray :=
  let v := CROnly.hankel (n := 512) (l := 1536) κ y
  ⟨(Array.range 64).map fun i => ((List.range 8).foldl (fun acc j =>
    acc + if v ⟨(8 * i + j) % 512, Nat.mod_lt _ (by norm_num)⟩ = 1 then 2 ^ j else 0) 0).toUInt8⟩

end FlockSoundness.Discharge.ZkTree
