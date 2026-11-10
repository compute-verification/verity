module

public import Mathlib.Data.Fin.Tuple.Take
public import Mathlib.Logic.Equiv.Fin.Basic
public import Proofs.Flock.Soundness.Discharge.ZkView.Leaf
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Properties.Verity.Sampled.CoinMap
import Proofs.Flock.Soundness.Discharge.Hm.Commit

/-!
# An opening's empty class at one hm96 leaf

An opening of a coin tree's leaf releases the leaf's first coins, which the verifier compares with live coins flipped
after the leaf was committed. The opening's class at those coins (`OpeningAt.classOf`) holds each round with the leaf's
hash that releases them, and `openEmpty` is the probability that it is empty.

* **At an hm96 leaf** (`FiberMisses`, `fiberMisses_le`): a leaf `H(x(ω) + M·y, c(y))` of uniform coins `ω` and salt
  `y`, a release `P ω`, and live coins `u` whose bits `bits u` have the release's law. `fiberMisses_le` bounds by `2·δ₁`
  the probability that no coins and salt with the leaf's hash release `bits u`. That is T1's bound twice. T1 makes the
  leaf ideal, so the event no longer reads the coins, and `bits u` may be swapped for the release `P ω`. T1 on the
  complement then makes the leaf real again, where the leaf's own coins and salt release `P ω`.
* **A coin tree's leaf is an hm96 leaf** (`treeX`, `treeMy`, `treeC`, `treeH`, `leaf_eq_treeH`, `treeMisses_le`): a
  sample's leaf is `H(x(ω) + M·y, c(y))`, with `x(ω)` the round digest of the sample's coins, `M·y` hm96-sha512's mask of its
  salt under the tree's key and `c(y)` the salt's hash, so `fiberMisses_le` bounds the miss at the tree's leaf.
* **The empty class, averaged over the device's sample** (`OpeningAt.AtLeaf`, `OpeningAt.misses_of_emptyAt`,
  `openEmpty_avg_le`): where the play ends in a close that records its log, a node's class is empty at `bits` only where
  no round with the sampled round's leaf is released as `bits` (`OpeningAt.classOf_nonempty`). At a node that every sample
  `d₀` of one leaf (its nonce, key, stream, round and count) describes, `openEmpty` averaged over the sample is at most
  `2·δ₁`.
* **The round's law** (`avg_take`, `opened_eq_roundBits`, `avg_opened_roundBits`): a sample's first `m` coins of a
  block of at least `m` are released as the bits (`roundBits`) of the `m` challenges they encode, so they have the sampling design
  of a round's `m` live challenges as bits.
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

/-- No coins `ω` and salt `y` whose leaf `H(x(ω) + M·y, c(y))` is `ℓ` release `r`: at a leaf whose hash is `ℓ`, the
class of an opening that releases `P ω` is empty at `r`. -/
def FiberMisses {Ω Salt B C D R : Type} [Add B] (My : Salt → B) (c : Salt → C) (H : B × C → D) (x : Ω → B) (P : Ω → R)
    (ℓ : D) (r : R) : Prop :=
  ∀ ω y, H (x ω + My y, c y) = ℓ → P ω ≠ r

/-- **An opening's class is empty with probability at most `2·δ₁`** at one hm96 leaf. Over the leaf's uniform coins `ω`
and salt `y`, and live coins `u` sampled independently whose bits `bits u` have the sampling design of the release `P ω` (`hlaw`),
this is the probability that no coins and salt with the leaf's hash release `bits u`. -/
theorem fiberMisses_le {Ω Q Salt B C D R : Type} [Fintype Ω] [Nonempty Ω] [Fintype Q] [Nonempty Q] [Fintype Salt]
    [Nonempty Salt] [AddCommGroup B] [Fintype B] (My : Salt → B) (c : Salt → C) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (x : Ω → B) (P : Ω → R) (bits : Q → R)
    (hlaw : ∀ g : R → ℝ≥0∞, avg (fun ω => g (P ω)) = avg fun u => g (bits u)) :
    prCoin (fun p : (Ω × Q) × Salt => FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2))
      ≤ 2 * δ₁ := by
  have : Nonempty B := ⟨0⟩
  have h1 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1)
    (fun w ℓ => FiberMisses My c H x P ℓ (bits w.2))
  have h2 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1)
    (fun w ℓ => ¬ FiberMisses My c H x P ℓ (P w.1))
  have hno : ∀ p : (Ω × Q) × Salt, ¬ FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (P p.1.1) :=
    fun p h => h p.1.1 p.2 rfl rfl
  have hown : prCoin (fun p : (Ω × Q) × Salt => ¬ FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (P p.1.1))
      = 1 := by
    unfold prCoin
    calc avg _ = avg (fun _ : (Ω × Q) × Salt => (1 : ℝ≥0∞)) := by
          congr 1
          funext p
          simp [hno p]
      _ = 1 := FlockSoundness.Game.avg_const _
  have hl : avg (fun w : Ω × Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (bits w.2))
      = avg fun q : Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (bits q) :=
    (FlockSoundness.ZK.avg_prod_split _).trans (FlockSoundness.Game.avg_const
      (avg fun q : Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (bits q)))
  have hr : avg (fun w : Ω × Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (P w.1))
      = avg fun ω : Ω => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (P ω) :=
    (FlockSoundness.ZK.avg_prod_split _).trans (congrArg avg (funext fun ω => FlockSoundness.Game.avg_const
      (prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (P ω))))
  have hswap : prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (bits p.1.2))
      = prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1)) := by
    rw [FlockSoundness.ZK.prCoin_prod_split
        (fun (w : Ω × Q) (u : B × Salt) => FiberMisses My c H x P (H (u.1, c u.2)) (bits w.2)),
      FlockSoundness.ZK.prCoin_prod_split
        (fun (w : Ω × Q) (u : B × Salt) => FiberMisses My c H x P (H (u.1, c u.2)) (P w.1))]
    exact hl.trans ((hlaw fun r => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) r).symm.trans
      hr.symm)
  have hideal : prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
      ≤ δ₁ := by
    have a := FlockSoundness.ZK.prCoin_add_not
      (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
    have hfin := FlockSoundness.ZK.prCoin_ne_top
      (fun p : (Ω × Q) × (B × Salt) => ¬ FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
    refine (ENNReal.add_le_add_iff_right hfin).1 ?_
    rw [a, add_comm δ₁]
    exact hown.symm.le.trans h2
  calc prCoin (fun p : (Ω × Q) × Salt => FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2))
      ≤ prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (bits p.1.2)) + δ₁ := h1
    _ = prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1)) + δ₁ := by
        rw [hswap]
    _ ≤ δ₁ + δ₁ := add_le_add hideal le_rfl
    _ = 2 * δ₁ := (two_mul δ₁).symm

/-! ## A coin tree's leaf -/

open FlockSoundness.Discharge.ZkView (Bits bitsOf bytesOf)
open FlockSoundness.Discharge.Hm (bitL eq_of_bitL get!_eq_getD)

/-- A coin tree's salt as its 192 bytes, from the sample's 1536 bits (`TreeDraw.leaf`'s). -/
def treeSalt (y : Fin 1536 → Bool) : ByteArray := bytesOfBits 192 fun t => if h : t < 1536 then y ⟨t, h⟩ else false

/-- **The round digest** of stream `s`'s round `r` under the nonce `ν` at the block of coins `ω`, as bits: the hm96
leaf's `x(ω)` (`treeLeaf`'s). -/
def treeX (ν : ByteArray) (s r : ℕ) {block : ℕ} (ω : Fin block → Fin 128 → Bool) : Bits :=
  bitsOf (treeHash ν (treeTag "round" ++ Flock.be32 s.toUInt32 ++ Flock.be32 r.toUInt32 ++
    (TreeDraw.coins (ω, fun _ => false)).foldr (fun c t => treeHash ν (treeTag "chain" ++ c ++ t))
      (treeHash ν (treeTag "tail"))))

/-- **hm96-sha512's mask of a coin tree's salt** under the key `key`, `M·y`, as bits. -/
def treeMy (key : ByteArray) (y : Fin 1536 → Bool) : Bits := bitsOf (Flock.Hm96.sha512.mask key (treeSalt y))

/-- A coin tree's hm96 instance under the nonce `ν`: hm96-sha512 with the tree's hash. -/
abbrev treeHm (ν : ByteArray) : Flock.Hm96 := { Flock.Hm96.sha512 with hash := treeHash ν }

/-- **A coin tree's salt hash** under the nonce `ν`, `c(y) = H(SALT_PREFIX ‖ y)`. -/
def treeC (ν : ByteArray) (y : Fin 1536 → Bool) : ByteArray := (treeHm ν).saltDigest (treeSalt y)

/-- **A coin tree's leaf from its commit string's halves** under the nonce `ν` and the key `key`:
`H(LEAF_PREFIX(key) ‖ b ‖ c)`. -/
def treeH (ν key : ByteArray) (p : Bits × ByteArray) : ByteArray :=
  treeHash ν (Flock.Hm96.sha512.leafPrefix key ++ (bytesOf (k := 64) p.1 ++ p.2))

/-- **Two strings' first 64 bytes, XORed**, are the bytes of the sum of their bits. -/
theorem xorBytes_eq (x m : ByteArray) :
    ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i) = bytesOf (k := 64) (bitsOf x + bitsOf m) := by
  have hA : ((Array.range 64).map fun i => x.get! i ^^^ m.get! i).toList.length = 64 := by
    rw [Array.toList_map, List.length_map, Array.toList_range, List.length_range]
  have hB : (bytesOf (k := 64) (bitsOf x + bitsOf m)).data.toList.length = 64 := by
    rw [bytesOf, Array.toList_ofFn, List.length_ofFn]
  refine ByteArray.ext (Array.toList_inj.mp (eq_of_bitL (by rw [hA, hB]) fun t ht => ?_))
  rw [hA] at ht
  have ht' : t < 512 := ht
  rw [FlockSoundness.Discharge.ZkView.bitL_bytesOf (k := 64) _ ht', bitL]
  rw [Array.toList_map, Array.toList_range, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (show t / 8 < 64 by omega), Option.map_some, Option.getD_some]
  rw [UInt8.toNat_xor, Nat.testBit_xor, get!_eq_getD, get!_eq_getD]
  simp only [Pi.add_apply, bitsOf, bitL]
  exact FlockSoundness.Discharge.ZkView.xor_ind _ _

/-- hm96's mask reads only the instance's salt and digest sizes. -/
theorem mask_eq_of_sizes {h h' : Flock.Hm96} (hs : h.saltBytes = h'.saltBytes) (hd : h.digestBytes = h'.digestBytes)
    (key y : ByteArray) : h.mask key y = h'.mask key y := by
  unfold Flock.Hm96.mask
  rw [hs, hd]

/-- **A coin tree's commit string** is the bytes of `x + M·y`, then the salt's hash. -/
theorem treeCommitString_eq (ν key x y : ByteArray) :
    (treeHm ν).commitString key x y =
      bytesOf (k := 64) (bitsOf x + bitsOf ((treeHm ν).mask key y)) ++ (treeHm ν).saltDigest y := by
  rw [← xorBytes_eq]
  rfl

/-- **A sample's leaf is an hm96 leaf** `H(x(ω) + M·y, c(y))` of its coins `ω` and salt `y`. -/
theorem leaf_eq_treeH {block : ℕ} (ν key : ByteArray) (s r : ℕ) (ω : Fin block → Fin 128 → Bool)
    (y : Fin 1536 → Bool) :
    TreeDraw.leaf ν key s r (ω, y) = treeH ν key (treeX ν s r ω + treeMy key y, treeC ν y) := by
  dsimp only [TreeDraw.leaf, treeLeaf]
  rw [treeCommitString_eq, mask_eq_of_sizes (h := treeHm ν) (h' := Flock.Hm96.sha512) rfl rfl]
  rfl

/-- **No sample with a coin tree's leaf releases the live bits, with probability at most `2·δ₁`** (`fiberMisses_le` at
`leaf_eq_treeH`). Over a uniform sample `p.1` and live coins `p.2` whose bits `bits p.2` have the sampling design of a sample's first
`n` coins (`hlaw`), this is the probability that no sample `d` with `p.1`'s leaf has its first `n` coins `bits p.2`. -/
theorem treeMisses_le {block : ℕ} (ν key : ByteArray) (s r n : ℕ) {Q : Type} [Fintype Q] [Nonempty Q]
    (bits : Q → List Bool) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding (treeMy key) (treeC ν) δ₁)
    (hlaw : ∀ G : List Bool → ℝ≥0∞,
      avg (fun ω : Fin block → Fin 128 → Bool => G (TreeDraw.opened (ω, fun _ => false) n)) = avg fun u => G (bits u)) :
    prCoin (fun p : TreeDraw block × Q => ∀ d : TreeDraw block,
      TreeDraw.leaf ν key s r d = TreeDraw.leaf ν key s r p.1 → TreeDraw.opened d n ≠ bits p.2) ≤ 2 * δ₁ := by
  let e : ((Fin block → Fin 128 → Bool) × Q) × (Fin 1536 → Bool) ≃ TreeDraw block × Q :=
    ⟨fun q => ((q.1.1, q.2), q.1.2), fun p => ((p.1.1, p.2), p.1.2), fun _ => rfl, fun _ => rfl⟩
  calc prCoin (fun p : TreeDraw block × Q => ∀ d : TreeDraw block,
        TreeDraw.leaf ν key s r d = TreeDraw.leaf ν key s r p.1 → TreeDraw.opened d n ≠ bits p.2)
      = prCoin (fun q : ((Fin block → Fin 128 → Bool) × Q) × (Fin 1536 → Bool) => ∀ d : TreeDraw block,
          TreeDraw.leaf ν key s r d = TreeDraw.leaf ν key s r (q.1.1, q.2) → TreeDraw.opened d n ≠ bits q.1.2) :=
        (FlockSoundness.ZK.prCoin_congr_equiv e fun p : TreeDraw block × Q => ∀ d : TreeDraw block,
          TreeDraw.leaf ν key s r d = TreeDraw.leaf ν key s r p.1 → TreeDraw.opened d n ≠ bits p.2).symm
    _ ≤ prCoin (fun q : ((Fin block → Fin 128 → Bool) × Q) × (Fin 1536 → Bool) =>
          FiberMisses (treeMy key) (treeC ν) (treeH ν key) (treeX ν s r)
            (fun ω => TreeDraw.opened (ω, fun _ => false) n)
            (treeH ν key (treeX ν s r q.1.1 + treeMy key q.2, treeC ν q.2)) (bits q.1.2)) :=
        prCoin_mono fun q hq ω y hl => hq (ω, y) (by rw [leaf_eq_treeH, leaf_eq_treeH]; exact hl)
    _ ≤ 2 * δ₁ := fiberMisses_le (treeMy key) (treeC ν) hT1 (treeH ν key) (treeX ν s r)
          (fun ω => TreeDraw.opened (ω, fun _ => false) n) bits hlaw

/-- **A uniform block's first `m` entries are uniform**: an average over `Fin block → C` of what reads only the first
`m ≤ block` entries is its average over `Fin m → C`. -/
theorem avg_take {C : Type} [Fintype C] [Nonempty C] {m block : ℕ} (hm : m ≤ block) (f : (Fin m → C) → ℝ≥0∞) :
    avg (fun ω : Fin block → C => f (Fin.take m hm ω)) = avg f := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hm
  let e : (Fin m → C) × (Fin k → C) ≃ (Fin (m + k) → C) :=
    (Equiv.sumArrowEquivProdArrow (Fin m) (Fin k) C).symm.trans (Equiv.arrowCongr finSumFinEquiv (Equiv.refl C))
  have he : ∀ p, Fin.take m hm (e p) = p.1 := fun p => funext fun i => by
    show Sum.elim p.1 p.2 (finSumFinEquiv.symm (Fin.castAdd k i)) = p.1 i
    rw [finSumFinEquiv_symm_apply_castAdd]
    rfl
  calc avg (fun ω : Fin (m + k) → C => f (Fin.take m hm ω))
      = avg (fun p : (Fin m → C) × (Fin k → C) => f (Fin.take m hm (e p))) :=
        (avg_equiv e fun ω => f (Fin.take m hm ω)).symm
    _ = avg (fun p : (Fin m → C) × (Fin k → C) => f p.1) := congrArg avg (funext fun p => congrArg f (he p))
    _ = avg f := (avg_prod _).trans (congrArg avg (funext fun a => avg_const (f a)))

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The empty class at an opening's node -/

namespace OpeningAt

variable {β : Type} {E : List Release → Game β} {g' : Game β}

/-- **A description at one leaf**: `o` opens, under the nonce `ν` and the key `key`, stream `s`'s round `r` with the
count `n` in a block of `block` coins, at the device's sample `d₀` of that round. -/
structure AtLeaf (o : OpeningAt E g') (ν key : ByteArray) (s r n : ℕ) {block : ℕ} (d₀ : TreeDraw block) : Prop where
  nonce_eq : o.T.nonce = ν
  key_eq : treeKey o.T.sample.1 = key
  stream_eq : (o.p.1 : ℕ) = s
  round_eq : (o.p.2 : ℕ) = r
  count_eq : o.n = n
  block_eq : o.T.spec.block = block
  draw_eq : HEq (o.T.sample.2 o.p.1 o.p.2) d₀

/-- **The empty class misses at the leaf**: where the play ends in a close that records its log and `o` describes the
node at the sample `d₀`'s leaf, an empty class at `bits` means no sample with `d₀`'s leaf has its first `n` coins `bits`
(`classOf_nonempty`). -/
theorem misses_of_emptyAt {lg : β → List Release} (hE : RecordsLog lg E) (o : OpeningAt E g') {ν key : ByteArray}
    {s r n block : ℕ} {d₀ : TreeDraw block} (h : o.AtLeaf ν key s r n d₀) {bits : List Bool} (he : EmptyAt E g' bits) :
    ∀ d : TreeDraw block, TreeDraw.leaf ν key s r d = TreeDraw.leaf ν key s r d₀ → TreeDraw.opened d n ≠ bits := by
  obtain ⟨o₀, -, hne⟩ := he
  obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, hd⟩ := h
  obtain rfl := eq_of_heq hd
  intro d hl hb
  exact hne (OpeningAt.classOf_nonempty hE o₀ o ⟨d, congrArg ByteArray.data hl⟩ hb)

end OpeningAt

/-- **An opening's empty class has probability at most `2·δ₁`, averaged over the device's sample** at one leaf. Where the
play ends in a close that records its log and, at each sample `d₀` of stream `s`'s round `r`, the node `g' d₀` has a
description at `d₀`'s leaf (`ho`), the probability that its class is empty at live coins whose bits `bits u` have the sampling design
of a sample's first `n` coins (`hlaw`), averaged over the sample, is at most `2·δ₁` (`treeMisses_le`). -/
theorem openEmpty_avg_le {β : Type} {lg : β → List Release} {E : List Release → Game β} (hE : RecordsLog lg E)
    (ν key : ByteArray) (s r n : ℕ) {block : ℕ} (g' : TreeDraw block → Game β)
    (ho : ∀ d₀, ∃ o : OpeningAt E (g' d₀), o.AtLeaf ν key s r n d₀) {Q : Type} [Fintype Q] [Nonempty Q]
    (bits : Q → List Bool) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding (treeMy key) (treeC ν) δ₁)
    (hlaw : ∀ G : List Bool → ℝ≥0∞,
      avg (fun ω : Fin block → Fin 128 → Bool => G (TreeDraw.opened (ω, fun _ => false) n)) = avg fun u => G (bits u)) :
    avg (fun d₀ : TreeDraw block => openEmpty E (g' d₀) bits) ≤ 2 * δ₁ :=
  calc avg (fun d₀ : TreeDraw block => openEmpty E (g' d₀) bits)
      ≤ avg (fun d₀ : TreeDraw block => prCoin fun u : Q => ∀ d : TreeDraw block,
          TreeDraw.leaf ν key s r d = TreeDraw.leaf ν key s r d₀ → TreeDraw.opened d n ≠ bits u) :=
        avg_mono fun d₀ => prCoin_mono fun _u hu => (ho d₀).elim fun o h => o.misses_of_emptyAt hE h hu
    _ = prCoin (fun p : TreeDraw block × Q => ∀ d : TreeDraw block,
          TreeDraw.leaf ν key s r d = TreeDraw.leaf ν key s r p.1 → TreeDraw.opened d n ≠ bits p.2) :=
        (FlockSoundness.ZK.prCoin_prod_split _).symm
    _ ≤ 2 * δ₁ := treeMisses_le ν key s r n bits hT1 hlaw

/-! ## The round's law -/

variable (e : 𝔽 ≃ (Fin 128 → Bool))

/-- **A sample's first `m` coins are a round's release**: in a block of at least `m` coins, the device releases them as
the bits of the `m` challenges they encode. -/
theorem opened_eq_roundBits {block m : ℕ} (hm : m ≤ block) (d : TreeDraw block) :
    TreeDraw.opened d m = roundBits e fun i => e.symm (Fin.take m hm d.1 i) := by
  have hr : roundBits e (fun i => e.symm (Fin.take m hm d.1 i)) =
      (List.ofFn fun i => List.ofFn (Fin.take m hm d.1 i)).flatten :=
    congrArg List.flatten (congrArg List.ofFn (funext fun i => congrArg List.ofFn (e.apply_symm_apply _)))
  rw [hr, TreeDraw.opened, ← Fin.ofFn_take_eq_take_ofFn hm]
  rfl

/-- **A sample's first `m` coins have a round's law**: in a block of at least `m` coins, a uniform sample's first `m` coins
are released as the bits of `m` uniform live challenges (`opened_eq_roundBits`). -/
theorem avg_opened_roundBits {block m : ℕ} (hm : m ≤ block) (G : List Bool → ℝ≥0∞) :
    avg (fun ω : Fin block → Fin 128 → Bool => G (TreeDraw.opened (ω, fun _ => false) m)) =
      avg fun c : Fin m → 𝔽 => G (roundBits e c) :=
  calc avg (fun ω : Fin block → Fin 128 → Bool => G (TreeDraw.opened (ω, fun _ => false) m))
      = avg (fun ω : Fin block → Fin 128 → Bool => G (roundBits e fun i => e.symm (Fin.take m hm ω i))) :=
        congrArg avg (funext fun ω => congrArg G (opened_eq_roundBits e hm (ω, fun _ => false)))
    _ = avg (fun a : Fin m → Fin 128 → Bool => G (roundBits e fun i => e.symm (a i))) :=
        avg_take hm fun a => G (roundBits e fun i => e.symm (a i))
    _ = avg fun c : Fin m → 𝔽 => G (roundBits e c) :=
        avg_equiv (Equiv.piCongrRight fun _ => e.symm) fun c => G (roundBits e c)

end Verity.Sampled
