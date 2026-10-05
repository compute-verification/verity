import FlockProofs.Hash
import Flock.Zk
import FlockLevel3.ArithFacts
import FlockSoundness.Protocol.Binding.Rows
import FlockSoundness.Protocol.StrictCR
import FlockSoundness.Protocol.ExecDrawOS
import Flock.Derive
import Flock.DeriveCheck
import FlockProofs.Registered
import FlockSoundness.Protocol.CROnly.Hm96Defs
import Mathlib.Algebra.BigOperators.Field
import FlockSoundness.Protocol.ZK.SHVZK
import FlockSoundness.Protocol.ZK.RealView
import Mathlib.Data.List.GetD
import FlockSoundness.Protocol.Discharge.Hm.Hash

/-!
# The `--zk` tables at the executable's leaves, and the honest prover's pre-final messages

`ZK.Table` leaves its leaf functions abstract. A `--zk` session's leaves are the executable's (`flock-verify verify --zk`):
a level-0 leaf is `MerkleScheme.hm96Sha512.leaf` of its row (68 lanes: the `2^k₀` witness lanes, then the extra lanes
`R^0_0, R^0_1, R^1_0, R^1_1`), a pads leaf is `verifyInner`'s `Default512.leaf (SHA-512(a ‖ b))` of its column `(a, b)`, and
`τ`'s commitment is `Zk.tauCommitment`; each under a 192-byte salt. Here:

* `bitsOf`, `bytesOf`: a byte string's bits as a vector over `ZMod 2` (`Hm.bitL`'s order), and back.
* `My`, `saltHash`, `leafH`: hm96-sha512's structure at the pinned key, `leaf(x; y) = H(x ⊕ M·y, c(y))` with `M·y` the
  Hankel matrix of the pinned key `κ₀` applied to the salt's bits (`CROnly.hankel`), `c(y) = SHA-512(sp ‖ y)` and
  `H(b, c) = SHA-512(lp ‖ b ‖ c)` (`LeafSplit` states that this is the executable's leaf).
* `Sizes`, `Coined`, `Coined.tab`: a `--zk` table at one coin vector, everything but its leaves given (`Coined`), with the
  executable's leaves (`tab`). The sizes do not depend on the coins: a table's opened sets are counted with their
  padding to `t_pad` level-0 positions and `Q` pads positions per rep, so the real final message is a function of the
  view (it opens fewer leaves); `Coined` places them in the trees (`s0`, `sP`).
* `Coined.viewExe`, `Coined.simExe`: `Hm96.viewR` and `Hm96.simR` with the executable's hidden leaves.
* `LeafPos`, `Leaves`, `Base`, `randOf`, `preWith`, `content`, `exeLeaf`: the honest prover's pre-final messages by tree
  position. `msgsAt T out` are its messages at fixed answers, `out c` of the session's pre-final view cut where the
  session stops at `c`'s coins, with the leaves at each position given; `honest T out hc` is the prover as a strategy
  against an adaptive verifier, `Causal` saying that its message `i` reads the answers before `i` only.
-/

namespace FlockSoundness.Discharge.ZkView

open Flock ZK

/-! ## Bits and hm96-sha512 -/

/-- An hm96-sha512 salt (192 bytes) as its 1536 bits. -/
abbrev Salt := Fin 1536 → ZMod 2

/-- 512 bits: the part `b = x ⊕ M·y` of an hm96-sha512 commit string. -/
abbrev Bits := Fin 512 → ZMod 2

/-- One byte from its eight bits, least significant first. -/
def byteOf (f : Fin 8 → ZMod 2) : UInt8 := ((List.ofFn fun i => decide (f i = 1)).foldr Nat.bit 0).toUInt8

/-- The `k` bytes of `8·k` bits, in `bitsOf`'s order. -/
def bytesOf {k : ℕ} (x : Fin (8 * k) → ZMod 2) : ByteArray :=
  ⟨Array.ofFn fun i : Fin k => byteOf fun b => x ⟨8 * i.val + b.val, by have := i.isLt; have := b.isLt; omega⟩⟩

/-- A salt's 192 bytes. -/
def saltBytes (y : Salt) : ByteArray := bytesOf (k := 192) y

/-- The pinned hm96-sha512 key's 2047 bits (`Hm96.Default512.key`). -/
def κ₀ : Fin (512 + 1536 - 1) → ZMod 2 := fun t => if Hm96.bit Hm96.Default512.key t.val then 1 else 0

/-- hm96-sha512's keyed hash of a salt at the pinned key, `M(κ₀)·y` (`CROnly.hankel`). -/
def My : Salt → Bits := CROnly.hankel (n := 512) (l := 1536) κ₀

/-- The salt's own hash `c(y) = SHA-512(sp ‖ y)`. -/
def saltHash (y : Salt) : ByteArray := Sha512.hash (Hm96.Default512.saltPrefix ++ saltBytes y)

/-- An hm96-sha512 leaf from its commit string's two halves: `H(b, c) = SHA-512(lp ‖ b ‖ c)`. -/
def leafH (p : Bits × ByteArray) : ByteArray := Hm96.Default512.treeLeaf (bytesOf (k := 64) p.1 ++ p.2)

/-! ## The leaves of a `--zk` table -/

/-- A level-0 row: the witness lanes, then the extra lanes `R^0_0, R^0_1, R^1_0, R^1_1`. -/
def row0 {nL : ℕ} (col : (Fin nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽)) : Array F128 :=
  (Array.ofFn col.1 ++ #[col.2 0 0, col.2 0 1, col.2 1 0, col.2 1 1] : Array 𝔽)

/-- A level-0 leaf: `hm96-sha512/v1`'s leaf of the row under the salt (`MerkleScheme.hm96Sha512.leaf`). -/
def leaf0 {nL : ℕ} (col : (Fin nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽)) (y : Salt) : ByteArray :=
  MerkleScheme.hm96Sha512.leaf (row0 col) (saltBytes y)

/-- A pads leaf: `verifyInner`'s leaf of the opened column `(a, b)`. -/
def leafP (p : 𝔽 × 𝔽) (y : Salt) : ByteArray :=
  Hm96.Default512.leaf (Sha512.hash (F128.toBytes p.1 ++ F128.toBytes p.2)) (saltBytes y)

/-- `τ`'s commitment (`Zk.tauCommitment`). -/
def leafT (t : 𝔽) (y : Salt) : ByteArray := Zk.tauCommitment t (saltBytes y)

/-- An ideal leaf of seed `(U, y)`: `H(U, c(y))`. -/
def leafI (s : Bits × Salt) : ByteArray := leafH (s.1, saltHash s.2)

/-- **A `--zk` table's sizes**, the same at every coin vector (`Table`'s size fields). -/
structure Sizes where
  nMask : ℕ
  nC : ℕ
  nR : ℕ
  nL : ℕ
  L : ℕ
  tpad : ℕ
  nS : ℕ
  nU : ℕ
  nE : ℕ
  nQ : ℕ
  nPU : ℕ
  nPad : ℕ

/-- **A `--zk` table at one coin vector, but its leaves**: `Table`'s other fields at the sizes `z`; each claim's rep; the
level-0 code at the positions nobody opens and each rep's pads code there (`Table.Hm96`'s `EU`, `GpU`, `ChU`, `CpU`); and
where each opened and each other position sits in level 0's tree and in each rep's pads tree. -/
structure Coined (K W Hi VL : Type) [Field K] (z : Sizes) where
  embed : 𝔽 →+* K
  eqw : Fin z.nC → Fin z.nMask → 𝔽
  nu : W → Fin z.nC → Fin 128 → 𝔽
  sReg : W → Fin z.nR → Fin 128 → 𝔽
  pubReg : Fin z.nR → Fin 128 → 𝔽
  lanes : W → (Fin z.nMask → Fin 128 → ZMod 2) → Fin z.nL → Fin z.L → 𝔽
  E : Fin z.nS → Fin z.L → 𝔽
  Gp : Fin z.nS → Fin z.tpad → 𝔽
  rbar : Fin 2 → Fin z.nL → K
  ρ : Fin 2 → Fin 2 → K
  vals : Fin 2 → W → (Fin z.nMask → Fin 128 → ZMod 2) → (Fin 2 → Fin 2 → Fin z.L → 𝔽) → Fin z.nE → 𝔽
  ia : Fin z.nE
  ib : Fin z.nE
  ζ : Fin 2 → 𝔽
  β : Fin 2 → 𝔽
  lig : Fin 2 → (Fin z.L → K) → Hi → VL
  lam : Fin 2 → (Fin z.nC → Fin 128 → 𝔽) × (Fin 2 → Fin z.nE → 𝔽) × (Fin 2 → Fin z.L → K) × (Fin 2 → Fin z.tpad → K)
    × (Fin 2 → 𝔽 × 𝔽) × (Fin 2 → VL) → PadIdx z.nE → 𝔽
  theta : Fin 2 → (Fin z.nC → Fin 128 → 𝔽) × (Fin 2 → Fin z.nE → 𝔽) × (Fin 2 → Fin z.L → K) × (Fin 2 → Fin z.tpad → K)
    × (Fin 2 → 𝔽 × 𝔽) × (Fin 2 → VL) → 𝔽
  Ch : Fin 2 → Fin z.nQ → PadIdx z.nE → 𝔽
  Cp : Fin 2 → Fin z.nQ → Fin z.nPad → 𝔽
  rep : Fin z.nC → Fin 2
  EU : Fin z.nU → Fin z.L → 𝔽
  GpU : Fin z.nU → Fin z.tpad → 𝔽
  ChU : Fin 2 → Fin z.nPU → PadIdx z.nE → 𝔽
  CpU : Fin 2 → Fin z.nPU → Fin z.nPad → 𝔽
  /-- Level 0's tree: the opened positions, then the others. -/
  s0 : Fin z.nS ⊕ Fin z.nU ≃ Fin (z.nS + z.nU)
  /-- Rep `r`'s pads tree: its opened positions, then the others. -/
  sP : Fin 2 → (Fin z.nQ ⊕ Fin z.nPU ≃ Fin (z.nQ + z.nPU))

variable {K W Hi VL : Type} [Field K] {z : Sizes}

/-- **The `--zk` table** at one coin vector: `T`'s fields with the executable's leaves (`leaf0`, `leafP`, `leafT`) and
ideal leaves `H(U, c(y))`. -/
def Coined.tab (T : Coined K W Hi VL z) : Table 𝔽 K W Salt (Bits × Salt) ByteArray Hi VL where
  nMask := z.nMask
  nC := z.nC
  nR := z.nR
  nL := z.nL
  L := z.L
  tpad := z.tpad
  nS := z.nS
  nU := z.nU
  nE := z.nE
  nQ := z.nQ
  nPU := z.nPU
  nPad := z.nPad
  embed := T.embed
  eqw := T.eqw
  nu := T.nu
  sReg := T.sReg
  pubReg := T.pubReg
  lanes := T.lanes
  E := T.E
  Gp := T.Gp
  rbar := T.rbar
  ρ := T.ρ
  vals := T.vals
  ia := T.ia
  ib := T.ib
  ζ := T.ζ
  β := T.β
  lig := T.lig
  lam := T.lam
  theta := T.theta
  Ch := T.Ch
  Cp := T.Cp
  leaf0 := leaf0
  leafP := leafP
  leafT := leafT
  ideal := leafI

/-! ## Lemma B's views with the executable's hidden leaves -/

/-- Level 0's column at the `q`-th position nobody opens, all 68 lanes (the content `Table.Hm96.dig` hashes). -/
noncomputable def Coined.hidCol0 (T : Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (q : Fin z.nU) :
    (Fin z.nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽) :=
  (fun l => ∑ c, T.lanes w ω.1 l c * T.EU q c + ∑ j, ω.2.1.1 l j * T.GpU q j,
    fun r i => ∑ c, ω.2.1.2.1 r i c * T.EU q c + ∑ j, ω.2.1.2.2 r i j * T.GpU q j)

/-- Rep `r`'s pads column `(C_h, C_μ)` at its `k`-th position nobody opens. -/
noncomputable def Coined.hidColP (T : Coined K W Hi VL z) (ω : T.tab.Rand) (r : Fin 2) (k : Fin z.nPU) : 𝔽 × 𝔽 :=
  (∑ i, T.ChU r k i * T.tab.hFull (ω.2.2.1 r) i + ∑ j, T.CpU r k j * ω.2.2.2.1 r j,
    ∑ i, T.ChU r k i * ω.2.2.2.2.2.1 r i + ∑ j, T.CpU r k j * ω.2.2.2.2.1 r j)

/-- The prover's hidden leaves: the executable's leaves of its hidden columns, each under its own salt. -/
noncomputable def Coined.hidExe (T : Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (ys : T.tab.Hid → Salt) : T.tab.Hid → ByteArray
  | .inl q => leaf0 (T.hidCol0 w ω q) (ys (.inl q))
  | .inr (r, k) => leafP (T.hidColP ω r k) (ys (.inr (r, k)))

/-- `S_shvzk`'s hidden leaves: the executable's leaves of its dummy run's level-0 columns and of zero pads columns. -/
noncomputable def Coined.hidExeS (T : Coined K W Hi VL z) (w0 : W) (σ : T.tab.SimRand) (ys : T.tab.Hid → Salt) :
    T.tab.Hid → ByteArray
  | .inl q => leaf0 (T.hidCol0 w0 (T.tab.dummyOf σ) q) (ys (.inl q))
  | .inr (r, k) => leafP (0, 0) (ys (.inr (r, k)))

/-- **The real prover's view** of a `--zk` table: `Table.view` with every hidden leaf the executable's leaf of what it
commits, under a fresh salt. -/
noncomputable def Coined.viewExe (T : Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (ys : T.tab.Hid → Salt) :
    T.tab.View :=
  T.tab.viewWith w ω (T.hidExe w ω ys)

/-- **`S_shvzk`'s output** for a `--zk` table, its hidden leaves the executable's. -/
noncomputable def Coined.simExe (T : Coined K W Hi VL z) (w0 : W) (σ : T.tab.SimRand) (ys : T.tab.Hid → Salt) :
    T.tab.View :=
  T.tab.simWith w0 σ (T.hidExeS w0 σ ys)

/-! ## Lemma C's prover, by tree position -/

variable {J n : ℕ} {Cn : Type} {zs : Fin J → Sizes}

end FlockSoundness.Discharge.ZkView
