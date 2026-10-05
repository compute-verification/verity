import FlockSoundness.CROnly
import FlockSoundness.Discharge.Hm.Mask
import FlockProofs.Hash
import Flock.Zk
import FlockLevel3.ArithFacts
import FlockSoundness.Protocol.Discharge.ZkView.Defs
import FlockSoundness.Discharge.Hm.Slot

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

local notation "𝔽" => FlockLevel3.GF128

/-! ## Bits and hm96-sha512 -/

/-- A byte string's bits over `ZMod 2`, bit `t` being bit `t % 8` of byte `t / 8` (`Hm.bitL`). -/
def bitsOf {n : ℕ} (b : ByteArray) : Fin n → ZMod 2 := fun t => if Hm.bitL b.data.toList t then 1 else 0

/-! ## The leaves of a `--zk` table -/

variable {K W Hi VL : Type} [Field K] {z : Sizes}

/-! ## Lemma B's views with the executable's hidden leaves -/

/-! ## Lemma C's prover, by tree position -/

/-- A table's pre-final leaves by position: level 0's tree, each rep's pads tree, each rep's commitment to `τ`. -/
abbrev LeafPos (z : Sizes) := Fin (z.nS + z.nU) ⊕ (Fin 2 × Fin (z.nQ + z.nPU)) ⊕ Fin 2

/-- A session's pre-final leaves. -/
abbrev Leaves {J : ℕ} (z : Fin J → Sizes) := Σ j, LeafPos (z j)

/-- **A table's randomness but its pre-final leaves' seeds**, at the sizes `z`: the mask slot; the lanes' paddings, the
extra lanes and theirs; each rep's free pads; each rep's `pad_h`, `pad_μ`, `μ`, with the salts the final message reveals
(level 0's opened leaves', each rep's opened pads leaves', `y_τ`) and levels ≥ 1's. The same type at every coin vector. -/
abbrev Base (z : Sizes) (Hi : Type) :=
  (Fin z.nMask → Fin 128 → ZMod 2) × ((Fin z.nL → Fin z.tpad → 𝔽) × (Fin 2 → Fin 2 → Fin z.L → 𝔽)
    × (Fin 2 → Fin 2 → Fin z.tpad → 𝔽)) × (Fin 2 → (Fin z.nE → 𝔽) × 𝔽 × 𝔽)
    × ((Fin 2 → Fin z.nPad → 𝔽) × (Fin 2 → Fin z.nPad → 𝔽) × (Fin 2 → PadIdx z.nE → 𝔽)
      × ((Fin z.nS → Salt) × (Fin 2 → Fin z.nQ → Salt) × (Fin 2 → Salt) × (Fin 2 → Hi)))

set_option synthInstance.maxSize 1024 in
instance (z : Sizes) (Hi : Type) [Fintype Hi] : Fintype (Base z Hi) := inferInstance

/-- **The table's randomness** from `b` and each pre-final leaf's seed by position (`T.s0`, `T.sP`). -/
noncomputable def Coined.randOf (T : Coined K W Hi VL z) (b : Base z Hi) (u : LeafPos z → Bits × Salt) : T.tab.Rand :=
  (b.1, b.2.1, b.2.2.1, (b.2.2.2.1, b.2.2.2.2.1, b.2.2.2.2.2.1,
    (b.2.2.2.2.2.2.1, fun q => u (.inl (T.s0 (.inl q))), fun q => u (.inl (T.s0 (.inr q))), b.2.2.2.2.2.2.2.1,
      fun r k => u (.inr (.inl (r, T.sP r (.inl k)))), fun r k => u (.inr (.inl (r, T.sP r (.inr k)))),
      b.2.2.2.2.2.2.2.2.1, fun r => u (.inr (.inr r)), b.2.2.2.2.2.2.2.2.2)))

/-- **The table's pre-final view with its leaves given by position** (`lv`): `Table.preView`, whose leaves are ideal,
with each leaf replaced by `lv` at its position. -/
noncomputable def Coined.preWith (T : Coined K W Hi VL z) (w : W) (b : Base z Hi) (lv : LeafPos z → ByteArray) :
    T.tab.PreView :=
  { T.tab.preView w (T.randOf b 0) with
    leaves0 := (fun q => lv (.inl (T.s0 (.inl q))), fun q => lv (.inl (T.s0 (.inr q))))
    tauCom := fun r => lv (.inr (.inr r))
    leavesP := fun r => (fun k => lv (.inr (.inl (r, T.sP r (.inl k)))), fun k => lv (.inr (.inl (r, T.sP r (.inr k))))) }

/-- Level 0's column at position `p`: the opened rows (`Table.rows`, `Table.xrows`) or a hidden column. -/
noncomputable def Coined.col0At (T : Coined K W Hi VL z) (w : W) (b : Base z Hi) (p : Fin (z.nS + z.nU)) :
    (Fin z.nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽) :=
  match T.s0.symm p with
  | .inl q => (T.tab.rows w b.1 b.2.1.1 q, T.tab.xrows b.2.1.2.1 b.2.1.2.2 q)
  | .inr q => T.hidCol0 w (T.randOf b 0) q

/-- Rep `r`'s pads column at position `p`: an opened column `(C_h[k], C_μ[k])` or a hidden one. -/
noncomputable def Coined.colPAt (T : Coined K W Hi VL z) (w : W) (b : Base z Hi) (r : Fin 2) (p : Fin (z.nQ + z.nPU)) :
    𝔽 × 𝔽 :=
  match (T.sP r).symm p with
  | .inl k => ((T.tab.view w (T.randOf b 0)).ch r k, (T.tab.view w (T.randOf b 0)).cm r k)
  | .inr k => T.hidColP (T.randOf b 0) r k

/-- **What each pre-final leaf commits**, the digest `x` of its commit string `x ⊕ M·y ‖ c(y)`: `SHA-512` of level 0's row,
of a pads column's two elements, of `τ` under `TAU_TAG`. -/
noncomputable def Coined.content (T : Coined K W Hi VL z) (w : W) (b : Base z Hi) : LeafPos z → ByteArray
  | .inl p => Sha512.hash (MerkleScheme.rowBytes (row0 (T.col0At w b p)))
  | .inr (.inl (r, p)) => Sha512.hash (F128.toBytes (T.colPAt w b r p).1 ++ F128.toBytes (T.colPAt w b r p).2)
  | .inr (.inr r) => Sha512.hash (Zk.TAU_TAG ++ F128.toBytes ((T.tab.view w (T.randOf b 0)).tau r))

/-- **The executable's pre-final leaf** at a position, under salt `y`. -/
noncomputable def Coined.exeLeaf (T : Coined K W Hi VL z) (w : W) (b : Base z Hi) (p : LeafPos z) (y : Salt) :
    ByteArray :=
  Hm96.Default512.leaf (T.content w b p) (saltBytes y)

variable {J n : ℕ} {Cn : Type} {zs : Fin J → Sizes}

/-- The digest each pre-final leaf of the session commits, as bits. -/
noncomputable def digAt (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j)) (w : Fin J → W) (ω : ∀ j, Base (zs j) Hi)
    (c : Fin n → Cn) (l : Leaves zs) : Bits :=
  bitsOf ((T c l.1).content (w l.1) (ω l.1) l.2)

/-- **The prover's messages at fixed answers `c`**, with the session's pre-final leaves `lv`: `out c` of every table's
pre-final view, cut where the session stops at `c`'s coins. -/
noncomputable def msgsAt (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j))
    (out : ∀ c, (∀ j, (T c j).tab.Cut) → Fin n → ByteArray) (w : Fin J → W) (ω : ∀ j, Base (zs j) Hi)
    (c : Fin n → Cn) (lv : Leaves zs → ByteArray) : Fin n → ByteArray :=
  out c fun j => (T c j).tab.cut (Session.sent (fun j => (T c j).tab) (fun j => (T c j).rep) j)
    ((T c j).preWith (w j) (ω j) fun p => lv ⟨j, p⟩)

/-- **The round encoding reads the answers before each round only**, whatever each leaf's pair `(M·y, c(y))`: message `i`
at answers `c` is message `i` at any answers agreeing with `c` before `i`. -/
def Causal (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j)) (out : ∀ c, (∀ j, (T c j).tab.Cut) → Fin n → ByteArray) :
    Prop :=
  ∀ (w : Fin J → W) (ω : ∀ j, Base (zs j) Hi) (p : Leaves zs → Bits × ByteArray) (c c' : Fin n → Cn) (i : Fin n),
    (∀ j, j < i → c j = c' j) →
      msgsAt T out w ω c (fun l => leafH (digAt T w ω c l + (p l).1, (p l).2)) i =
        msgsAt T out w ω c' (fun l => leafH (digAt T w ω c' l + (p l).1, (p l).2)) i

/-- The prover on each leaf's pair `(B, c)` in place of `(M·y, c(y))` (Lemma C's `P`). -/
noncomputable def pairProver (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j))
    (out : ∀ c, (∀ j, (T c j).tab.Cut) → Fin n → ByteArray) (hc : Causal T out) (w : Fin J → W) :
    Strategy ((∀ j, Base (zs j) Hi) × (Leaves zs → Bits × ByteArray)) ByteArray Cn n where
  msg x c := msgsAt T out w x.1 c fun l => leafH (digAt T w x.1 c l + (x.2 l).1, (x.2 l).2)
  causal x c c' i h := hc w x.1 x.2 c c' i h

/-- **The honest `--zk` prover** against an adaptive verifier, on its randomness and every pre-final leaf's salt: each
leaf `H(x ⊕ M·y, c(y))` (`HonestMsg`: the executable's leaf). -/
noncomputable def honest (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j))
    (out : ∀ c, (∀ j, (T c j).tab.Cut) → Fin n → ByteArray) (hc : Causal T out) (w : Fin J → W) :
    Strategy ((∀ j, Base (zs j) Hi) × (Leaves zs → Salt)) ByteArray Cn n :=
  (pairProver T out hc w).comap fun x => (x.1, fun l => (My (x.2 l), saltHash (x.2 l)))

end FlockSoundness.Discharge.ZkView
