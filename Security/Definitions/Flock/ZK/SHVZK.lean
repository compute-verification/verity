import Definitions.Flock.ZK.Table
import Mathlib.Tactic.FieldSimp

/-!
# Zero knowledge of one masked table: the view as a function of four masked components

For the model of `Table` (the reference prover `Table.view` and `S_shvzk` as `Table.sim`), at a fixed non-degenerate coin
vector, Lemma B (§4.3) writes the prover's view as a public function of four components, each a function of the earlier
blocks plus a surjective image of a fresh block:

* `Core`: the four components (the masked `ŝ`; the witness lanes' entries at the opened positions with `(y₁'^r, ȳ^r)`;
  each rep's masked values with `ρ_in`, `σ_in`; each rep's `C_h`, `Y₂`, `Y₁` with the salts and ideal-leaf seeds), and
  `psi`, the view computed from them.
* `mask1`, `mask2`, `mask3`: the masks of rows (ii)–(iii), (vi)–(vii) and (viii)–(x). `Wr_add` is `mask1`'s additivity.
* `realCore`, `simCore` (with `simMask3`): the prover's view and `S_shvzk`'s output in that order.
* The `Fintype` and `DecidableEq` instances of the seeds, the components and both randomness spaces.

The lemmas about them (the identity (★), the masks' surjectivity, `table_shvzk` and Lemma A's `table_prefinal_indep`)
are `Proofs/Flock/Soundness/ZK/SHVZK.lean`'s.
-/

namespace FlockSoundness.ZK

open Finset

-- The randomness and the view's components are products of up to fifteen factors, and their `Fintype`, `DecidableEq`
-- and group instances exceed instance search's default size.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

/-! ## The identity (★) -/

theorem Wr_add (r : Fin 2) (a b a' b' : F) : T.Wr r (a + a') (b + b') = T.Wr r a b + T.Wr r a' b' := by
  simp only [Wr, map_add]
  ring

/-! ## The view as a function of four masked components -/

/-- The four components the view is a function of (Lemma B's sampling order): the masked `ŝ`; the witness lanes' entries
at the opened positions with `(y₁'^r, ȳ^r)`; each rep's masked values with `(ρ_in, σ_in)`; each rep's `C_h`, `Y₂`, `Y₁`,
with every salt and ideal-leaf seed. -/
abbrev Core := (Fin T.nC → Fin 128 → F)
  × ((Fin T.nS → Fin T.nL → F) × (Fin 2 → Fin T.L → K) × (Fin 2 → Fin T.tpad → K))
  × (Fin 2 → (Fin T.nE → F) × F × F)
  × ((Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds)

/-- **The view from its components**, by the public computations of Lemma B's determined rows: the extra-lane entries by
(★) (row (iv)), levels ≥ 1 from `y₁'^r` and their salts (v), `τ` and `C_μ` from the replayed constraint (xi), and every
leaf (xii). -/
noncomputable def psi (c : T.Core) : T.View :=
  let sHat := c.1
  let rows := c.2.1.1
  let y1 := c.2.1.2.1
  let yb := c.2.1.2.2
  let masked : Fin 2 → Fin T.nE → F := fun r => (c.2.2.1 r).1
  let inner : Fin 2 → F × F := fun r => (c.2.2.1 r).2
  let ch := c.2.2.2.1
  let Y2 := c.2.2.2.2.1
  let Y1 := c.2.2.2.2.2.1
  let s := c.2.2.2.2.2.2
  let xrows : Fin T.nS → Fin 2 → Fin 2 → F := fun q r =>
    ![(T.Winv r (T.xTarget rows y1 yb q r)).1, (T.Winv r (T.xTarget rows y1 yb q r)).2]
  let lig := fun r => T.lig r (y1 r) (s.2.2.2.2.2.2.2.2 r)
  let pre := (sHat, masked, y1, yb, inner, lig)
  let tau := fun r => (∑ i, T.lam r pre i * Y1 r i - T.theta r pre) / T.β r
  let cm := fun r k => (∑ i, T.Ch r k i * Y1 r i + ∑ j, T.Cp r k j * Y2 r j - ch r k) / T.β r
  { sHat, sReg := T.pubReg, masked, y1, ybar := yb, lig, inner, Y1, Y2
    tau, ytau := s.2.2.2.2.2.2.1, tauCom := fun r => T.leafT (tau r) (s.2.2.2.2.2.2.1 r)
    rows, xrows, salts0 := s.1
    leaves0 := (fun q => T.leaf0 (rows q, xrows q) (s.1 q), fun q => T.ideal (s.2.2.1 q))
    ch, cm, saltsP := s.2.2.2.1
    leavesP := fun r => (fun k => T.leafP (ch r k, cm r k) (s.2.2.2.1 r k), fun k => T.ideal (s.2.2.2.2.2.1 r k)) }

/-! ## The masks -/

/-- Rows (ii) and (iii)'s mask: the paddings at the opened positions, `W^r` of the extra lanes, and `ȳ^r`. -/
def mask1 : ((Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad) →+
    ((Fin T.nS → Fin T.nL → F) × (Fin 2 → Fin T.L → K) × (Fin 2 → Fin T.tpad → K)) where
  toFun b := (fun q l => ∑ j, b.1 l j * T.Gp q j, fun r c => T.Wr r (b.2.1 r 0 c) (b.2.1 r 1 c),
    fun r j => ∑ l, T.rbar r l * T.embed (b.1 l j) + T.Wr r (b.2.2 r 0 j) (b.2.2 r 1 j))
  map_zero' := by
    simp only [Prod.fst_zero, Prod.snd_zero, Pi.zero_apply, zero_mul, Finset.sum_const_zero, map_zero, mul_zero,
      Wr, add_zero]
    rfl
  map_add' b b' := by
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · funext q l
      simp only [Prod.fst_add, Pi.add_apply, add_mul, Finset.sum_add_distrib]
    · funext r c
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Wr_add]
    · funext r j
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Wr_add, map_add, mul_add, Finset.sum_add_distrib]
      ring

/-- Rows (vi) and (vii)'s mask: each exposed pad, and `ρ_in`, `σ_in` through `h_tx`, `h_ty`. -/
def mask2 : (Fin 2 → (Fin T.nE → F) × F × F) →+ (Fin 2 → (Fin T.nE → F) × F × F) where
  toFun h r := ((h r).1, T.ζ r * (h r).1 T.ia + (h r).2.1, (h r).1 T.ib + (h r).2.2)
  map_zero' := by
    funext r
    simp
  map_add' h h' := by
    funext r
    simp only [Pi.add_apply, Prod.fst_add, Prod.snd_add, Prod.mk_add_mk]
    refine Prod.ext rfl (Prod.ext ?_ ?_) <;> simp only [] <;> ring

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

/-- Rows (viii)–(x)'s mask, with the salts: `C_h` through `pad_h`, `Y₂` through `pad_μ`, `Y₁` through `μ`. -/
def mask3 : ((Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds) →+
    ((Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds) where
  toFun b := (fun r k => ∑ j, T.Cp r k j * b.1 r j, fun r j => b.1 r j + T.β r * b.2.1 r j,
    fun r i => T.β r * b.2.2.1 r i, b.2.2.2)
  map_zero' := by
    simp only [Prod.fst_zero, Prod.snd_zero, Pi.zero_apply, mul_zero, Finset.sum_const_zero, add_zero]
    rfl
  map_add' b b' := by
    refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl))
    · funext r k
      simp only [Prod.fst_add, Pi.add_apply, mul_add, Finset.sum_add_distrib]
    · funext r j
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply]
      ring
    · funext r i
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply]
      ring

/-! ## Lemma B: the real view and `S_shvzk` -/

variable [CharP F 2]

/-- **The prover's view in Lemma B's sampling order**: each component is a function of the earlier blocks plus a mask of
its own block. -/
noncomputable def realCore (w : W) (ω : T.Rand) : T.Core :=
  (T.nu w + T.maskHom ω.1,
    (fun q l => ∑ c, T.lanes w ω.1 l c * T.E q c, fun r => T.y1 w ω.1 r, 0) + T.mask1 ω.2.1,
    (fun r => (T.vals r w ω.1 ω.2.1.2.1, 0, 0)) + T.mask2 ω.2.2.1,
    (fun r k => ∑ i, T.Ch r k i * T.hFull (ω.2.2.1 r) i, 0, fun r => T.hFull (ω.2.2.1 r), 0) + T.mask3 ω.2.2.2)

/-- `S_shvzk`'s last block: its fresh `C_h`, `Y₂`, `Y₁`, and the dummy run's salts and seeds. -/
def simMask3 : (((Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds)
      × (Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F)) →+
    ((Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds) where
  toFun b := (b.2.1, b.2.2.1, b.2.2.2, b.1.2.2.2)
  map_zero' := rfl
  map_add' _ _ := rfl

/-- **`S_shvzk`'s output in the same order**: fresh values where the prover's are masked, the dummy run's where they are
kept. -/
noncomputable def simCore (w0 : W) (σ : T.SimRand) : T.Core :=
  (0 + AddMonoidHom.snd _ _ σ.1,
    (fun q l => ∑ c, T.lanes w0 σ.1.1 l c * T.E q c, fun r => T.y1 w0 σ.1.1 r, 0) + T.mask1 σ.2.1,
    0 + AddMonoidHom.snd _ _ σ.2.2.1,
    0 + T.simMask3 σ.2.2.2)

/-! ## Counting -/

variable [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
  [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]

instance instDecidableEqSeeds : DecidableEq T.Seeds := inferInstance

instance instFintypeSeeds : Fintype T.Seeds := inferInstance

instance instDecidableEqCore : DecidableEq T.Core := inferInstance

instance instFintypeCore : Fintype T.Core := inferInstance

instance instDecidableEqRand : DecidableEq T.Rand := inferInstance

instance instFintypeRand : Fintype T.Rand := inferInstance

instance instDecidableEqSimRand : DecidableEq T.SimRand := inferInstance

instance instFintypeSimRand : Fintype T.SimRand := inferInstance

end Table

end FlockSoundness.ZK
