module

public import Proofs.Flock.Claims.ZK.Deployed
public import Proofs.Flock.Claims.ZK.WorkerSeen
public import Proofs.Flock.Soundness.Discharge.ZkView.Fixed
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.Flock
import Proofs.Flock.Soundness.Discharge.ZkView.Leaf
import Proofs.Flock.Soundness.ZK.Complete
import Proofs.Flock.Soundness.ZK.GK.Link
import Proofs.Flock.Soundness.ZK.PadNonvanishing40
import Proofs.Flock.Soundness.ZK.RealLeaves
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# The switched prover at the session's tables

`WorkerSeen` proves Lemma B at values read from what every table's workers see (`Table.Seen`). Here it is at the `J`
tables of a `--zk` session, in the form the session's models read (`Deployed`), with Lemma C before the last coin in the
same order: the values may read the mask slots, level 0's hiding and level 0's salts, at every table.

* `Coined.swExe`: a `--zk` table's switched view at a worker's values `v`, with the executable's hidden leaves
  (`Table.Hm96.swViewR`).
* `Coined.swPreWith`, `Coined.swContent`: its pre-final view with the leaves given by position, and what each leaf
  commits; `swCutAt`, `swDigOf`, `swExeOf` at the session's tables, as `cutAt`, `digOf`, `exeOf` have the honest prover.
* `ZkView.Seen`, `Coined.seenPos`, `tabSeen`, `seenAt`, `seenOfP`: what a worker sees of a table, with level 0's salts
  by position, at the final message's randomness and before the last coin.
* `Coined.fixPB`, `Coined.cut_swPreWith_fixPB`: the pre-final fix map on a base (`Table.fixP`), which keeps what the
  workers see and makes the switched pre-final view, cut, the honest one; level 0's leaves commit the same digests at
  both (`Coined.swContent_inl`, `Coined.content_fixPB_inl`).
* `card_swCutAt_real0`: **Lemma A at the switched prover, at the session's tables**: with the pads trees' and `τ`'s
  leaves ideal and level 0's real under salts the values read, any function of the cut view takes every value as often
  at the switched prover, at values read from what the workers see, as at the honest prover on the same witness.
  `card_cutAt_ideal01` is `card_cutAt_ideal` with the session's leaves split into level 0's and the others (`leaves01`).
* `zkViewAtJ_sw`: at `J` `CoinedSetup`s, at coins each admits and valid witnesses, for any values read from what every
  table's workers see, the joint switched view and the tables' `S_shvzk` on any dummy are within `Σ_j 2·|Hid_j|·δ₁` both
  ways (`fam_sw_shvzk_hm96`), as `ZkHidden.zkViewAtJ_of_ok` has it for the honest prover.
* `Proto.TabModel.SwLink`, `Proto.TabModel.shvzk_sw`: where the prover goes on, a prover whose messages are `lift` of the
  tables' switched views at the workers' values meets `SHVZK` against `simOut`, the honest prover's simulator, within
  `2·n·δ₁`, as `TabModel.shvzk` has it at `TabModel.Link`.
* `Proto.TPreModel.SwLink`, `Proto.TPreModel.runs_close_sw`; `Proto.TOffModel.SwLink`, `Proto.TOffModel.close_sw`:
  before the last coin and where the prover refuses, Lemma C between the switched prover at the workers' values and the
  honest prover on the dummy, within `2·|L|·δ₁` (`runs_close_sw_of_ideal`, `refused_close_sw_of_ideal`): the pads
  trees' and `τ`'s leaves go ideal at the switched prover, which meets the honest prover with level 0's leaves real
  (`card_swCutAt_real0`), whose leaves go ideal and whose witness swaps (`card_cutAt_ideal01`).
* `Proto.TSwLink`, `Proto.withSim_shvzk_sw`, `Flock.zk_deployZk_of_tables_sw`: **`Claims.Flock.ZKAt` at the switched
  prover**, with the honest prover's simulator and the same `ε_zk` and `γ`, with `TMsgLink` replaced by `TSwLink`: the
  prover's messages are the switched prover's at values the workers supply, which read what the workers see and never
  the pads.
-/

@[expose] public section

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (CoinedSetup CoinsZK)
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- instance search over the ZK view's and coins' product types

/-- **A `--zk` table's switched view** at a worker's clear values `v`, with the executable's hidden leaves: `v`'s view
where its rows hold, else the honest prover's on the moved pads (`Table.Hm96.swViewR`). -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.swExe {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (ω : T.tab.Rand)
    (ys : T.tab.Hid → ZkView.Salt) : T.tab.View :=
  T.tab.toBase (T.hm96.onVals.swViewR (w, v) (w, T.tab.vals) ω ys)

/-- **A `--zk` table at any clear values**: its witnesses paired with clear values, which `vals` reads; every other
field the table's. Its table is the table's at any clear values (`Coined.onVals_tab`). -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.onVals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) : ZkView.Coined K (W × T.tab.Vals) Hi VL z where
  embed := T.embed
  eqw := T.eqw
  nu p := T.nu p.1
  sReg p := T.sReg p.1
  pubReg := T.pubReg
  lanes p := T.lanes p.1
  E := T.E
  Gp := T.Gp
  rbar := T.rbar
  ρ := T.ρ
  vals r p := p.2 r p.1
  ia := T.ia
  ib := T.ib
  ζ := T.ζ
  β := T.β
  lig := T.lig
  lam := T.lam
  theta := T.theta
  Ch := T.Ch
  Cp := T.Cp
  rep := T.rep
  EU := T.EU
  GpU := T.GpU
  ChU := T.ChU
  CpU := T.CpU
  s0 := T.s0
  sP := T.sP

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.onVals_tab {K W Hi VL : Type} [Field K] {z : ZkView.Sizes}
    (T : ZkView.Coined K W Hi VL z) : T.onVals.tab = T.tab.onVals :=
  rfl

/-- The base with each rep's free pads moved from the honest values to `v`'s at its mask slot and level 0's hiding
(`Table.shift`). -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.shiftB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi) :
    ZkView.Base z Hi :=
  let s : Fin 2 → (Fin z.nE → FlockLevel3.GF128) × FlockLevel3.GF128 × FlockLevel3.GF128 :=
    T.tab.onVals.padShift (w, v) (w, T.tab.vals) b.1 b.2.1
  (b.1, b.2.1, b.2.2.1 + s, b.2.2.2)

/-- **The switched prover's pre-final view with its leaves given by position** (`lv`): `Table.swPre`, as
`Coined.preWith` has `Table.preView`. -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.swPreWith {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (lv : ZkView.LeafPos z → ByteArray) : T.tab.PreView :=
  { T.tab.swPre w v (T.randOf b 0) with
    leaves0 := (fun q => lv (.inl (T.s0 (.inl q))), fun q => lv (.inl (T.s0 (.inr q))))
    tauCom := fun r => lv (.inr (.inr r))
    leavesP := fun r => (fun k => lv (.inr (.inl (r, T.sP r (.inl k)))), fun k => lv (.inr (.inl (r, T.sP r (.inr k))))) }

open Classical in
/-- **What each of the switched prover's pre-final leaves commits**: `v`'s columns and `τ` where the rows hold, else the
honest prover's on the moved pads. -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.swContent {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi) :
    ZkView.LeafPos z → ByteArray :=
  if T.tab.onVals.RowsAt (w, v) b.1 b.2.1 then T.onVals.content (w, v) b else T.content w (T.shiftB w v b)

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swContent_size {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (p : ZkView.LeafPos z) : (T.swContent w v b p).size = 64 := by
  unfold ZkView.Coined.swContent
  split_ifs
  · exact T.onVals.content_size _ _ _
  · exact T.content_size _ _ _


/-! ## What the workers see, and the pre-final fix map on a base -/

/-- **What a worker sees of a `--zk` table's randomness**, at the sizes `z` (`Table.Seen`): the mask slot, level 0's
hiding, and level 0's salts by position. -/
abbrev _root_.FlockSoundness.Discharge.ZkView.Seen (z : ZkView.Sizes) : Type :=
  (Fin z.nMask → Fin 128 → ZMod 2) × ((Fin z.nL → Fin z.tpad → FlockLevel3.GF128) ×
    (Fin 2 → Fin 2 → Fin z.L → FlockLevel3.GF128) × (Fin 2 → Fin 2 → Fin z.tpad → FlockLevel3.GF128)) ×
    (Fin (z.nS + z.nU) → ZkView.Salt)

/-- What a worker sees of a `--zk` table (`Table.Seen`), with level 0's salts by position (`Coined.s0`). -/
def _root_.FlockSoundness.Discharge.ZkView.Coined.seenPos {K W Hi VL : Type} [Field K] {z : ZkView.Sizes}
    (T : ZkView.Coined K W Hi VL z) (s : T.tab.Seen) : ZkView.Seen z :=
  (s.1, s.2.1, fun p => Sum.elim s.2.2.1 s.2.2.2 (T.s0.symm p))

/-- **The pre-final fix map on a base** at constant values `c` (`Table.fixP`): the free pads moved by `padShift` and `μ`
by `fixAmt`; the mask slot, level 0's hiding, `pad_h`, `pad_μ` and every salt kept. -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.fixPB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (c : Fin 2 → Fin z.nE → FlockLevel3.GF128) :
    ZkView.Base z Hi ≃ ZkView.Base z Hi where
  toFun b :=
    let s : Fin 2 → (Fin z.nE → FlockLevel3.GF128) × FlockLevel3.GF128 × FlockLevel3.GF128 :=
      T.tab.padShiftC w c b.1 b.2.1
    let m : Fin 2 → FlockSoundness.ZK.PadIdx z.nE → FlockLevel3.GF128 := (T.tab.fixAmt w c b.1 b.2.1 b.2.2.1).2.2
    (b.1, b.2.1, b.2.2.1 + s, b.2.2.2.1, b.2.2.2.2.1, b.2.2.2.2.2.1 + m, b.2.2.2.2.2.2)
  invFun b :=
    let s : Fin 2 → (Fin z.nE → FlockLevel3.GF128) × FlockLevel3.GF128 × FlockLevel3.GF128 :=
      T.tab.padShiftC w c b.1 b.2.1
    let m : Fin 2 → FlockSoundness.ZK.PadIdx z.nE → FlockLevel3.GF128 :=
      (T.tab.fixAmt w c b.1 b.2.1 (b.2.2.1 - s)).2.2
    (b.1, b.2.1, b.2.2.1 - s, b.2.2.2.1, b.2.2.2.2.1, b.2.2.2.2.2.1 - m, b.2.2.2.2.2.2)
  left_inv b := by
    have e := (T.tab.fixP w c).symm_apply_apply (T.randOf b 0)
    exact Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.1) e)
      (Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.2.2.2.1) e) rfl)))))
  right_inv b := by
    have e := (T.tab.fixP w c).apply_symm_apply (T.randOf b 0)
    exact Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.1) e)
      (Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.2.2.2.1) e) rfl)))))

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.randOf_fixPB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (c : Fin 2 → Fin z.nE → FlockLevel3.GF128)
    (b : ZkView.Base z Hi) (u : ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) :
    T.randOf (T.fixPB w c b) u = T.tab.fixP w c (T.randOf b u) :=
  rfl

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swPreWith_congr {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v v' : T.tab.Vals) (b : ZkView.Base z Hi)
    (lv : ZkView.LeafPos z → ByteArray) (h : ∀ r, v r w b.1 b.2.1 = v' r w b.1 b.2.1) :
    T.swPreWith w v b lv = T.swPreWith w v' b lv := by
  unfold ZkView.Coined.swPreWith
  rw [T.tab.swPre_congr w v v' (T.randOf b 0) h]

/-- Two pre-final views whose cuts agree still agree, cut, with their leaves replaced by the same leaves. -/
theorem _root_.FlockSoundness.ZK.Table.cut_withLeaves {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    [CharP F 2] (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) (P Q : T.PreView)
    (h : T.cut S P = T.cut S Q) (l0 : (Fin T.nS → D) × (Fin T.nU → D)) (tc : Fin 2 → D)
    (lp : Fin 2 → (Fin T.nQ → D) × (Fin T.nPU → D)) :
    T.cut S { P with leaves0 := l0, tauCom := tc, leavesP := lp } =
      T.cut S { Q with leaves0 := l0, tauCom := tc, leavesP := lp } := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, -, -, -⟩ := FlockSoundness.ZK.Table.Cut.ext_iff.1 h
  exact FlockSoundness.ZK.Table.Cut.ext h1 h2 h3 h4 h5 h6 h7 h8 h9 rfl rfl rfl

/-- **The switched pre-final view, cut, is the honest one at the pre-final fix map** (`Table.cut_swPre_fixP`), at the
values `v` read at the base's mask slot and level 0's hiding, with the same leaves. -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.cut_swPreWith_fixPB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (S : T.tab.Sent) (hS : T.tab.Admits S) (w : W)
    (v : T.tab.Vals) (b : ZkView.Base z Hi) (lv lv' : ZkView.LeafPos z → ByteArray) (hl : ∀ p, lv p = lv' p) :
    T.tab.cut S (T.swPreWith w v b lv) = T.tab.cut S (T.preWith w (T.fixPB w (fun r => v r w b.1 b.2.1) b) lv') := by
  obtain rfl : lv = lv' := funext hl
  rw [T.swPreWith_congr w v (T.tab.constVals fun r => v r w b.1 b.2.1) b lv fun _ => rfl]
  have h : T.tab.cut S (T.tab.swPre w (T.tab.constVals fun r => v r w b.1 b.2.1) (T.randOf b 0)) =
      T.tab.cut S (T.tab.preView w (T.randOf (T.fixPB w (fun r => v r w b.1 b.2.1) b) 0)) :=
    T.tab.cut_swPre_fixP S hS w (fun r => v r w b.1 b.2.1) (T.randOf b 0)
  exact FlockSoundness.ZK.Table.cut_withLeaves T.tab S _ _ h _ _ _

set_option smartUnfolding false in
/-- Level 0's leaves commit the honest prover's digests at the switched prover: they read the mask slot and level 0's
hiding, which `shiftB` keeps, and no clear value. -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swContent_inl {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (p : Fin (z.nS + z.nU)) : T.swContent w v b (.inl p) = T.content w b (.inl p) := by
  unfold ZkView.Coined.swContent
  split_ifs <;> rfl

set_option smartUnfolding false in
/-- The pre-final fix map keeps level 0's leaves' digests. -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.content_fixPB_inl {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (c : Fin 2 → Fin z.nE → FlockLevel3.GF128)
    (b : ZkView.Base z Hi) (p : Fin (z.nS + z.nU)) : T.content w (T.fixPB w c b) (.inl p) = T.content w b (.inl p) :=
  rfl

/-! ## The switched prover before the last coin, at the session's tables -/

/-- **The switched prover's pre-final view at coins `cz`, cut where the session stops** (`famSent`), at table
witnesses `wt`, the workers' values `v`, bases `ω` and the pre-final leaves `lv` by position (`Coined.swPreWith`), as
`cutAt` has the honest prover's. -/
noncomputable def swCutAt {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) :
    ∀ j, ((C j).T (cz j)).tab.Cut :=
  fun j => ((C j).T (cz j)).tab.cut (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j)
    (((C j).T (cz j)).swPreWith (wt j) (v j) (ω j) fun p => lv ⟨j, p⟩)

/-- The digest each of the switched prover's pre-final leaves commits, as bits (`Coined.swContent`). -/
noncomputable def swDigOf {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (l : ZkView.Leaves (fun j => (C j).z)) : ZkView.Bits :=
  ZkView.bitsOf (((C l.1).T (cz l.1)).swContent (wt l.1) (v l.1) (ω l.1) l.2)

/-- **The executable's pre-final leaves at the switched prover**: `hm96-sha512/v1`'s leaf of each `Coined.swContent`
under its own salt, as `exeOf` has the honest prover's. -/
noncomputable def swExeOf {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) :
    ZkView.Leaves (fun j => (C j).z) → ByteArray :=
  fun l => _root_.Flock.Hm96.Default512.leaf (((C l.1).T (cz l.1)).swContent (wt l.1) (v l.1) (ω l.1) l.2) (ZkView.saltBytes (y l))

/-- The switched prover's leaves are `H(dig ⊕ M·y, c(y))` (`ZkView.leafSplit`). -/
theorem swExeOf_eq {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) :
    swExeOf C cz wt v ω y =
      fun l => ZkView.leafH (swDigOf C cz wt v ω l + ZkView.My (y l), ZkView.saltHash (y l)) :=
  funext fun l => by
    rw [swExeOf, ZkView.leafSplit _ _ (((C l.1).T (cz l.1)).swContent_size _ _ _ _)]
    rfl

/-- **What the workers see of the session's tables** at their randomness and hidden leaves' salts `p` (`famSeen`), with
level 0's salts by position. -/
noncomputable def tabSeen {J : ℕ} (C : Fin J → CoinedSetup) (c : ∀ j, CoinsZK (C j).Rest)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) : ∀ j, ZkView.Seen (C j).z :=
  fun j => ((C j).T (c j)).seenPos (FlockSoundness.ZK.famSeen (fun j => ((C j).T (c j)).tab) p j)

/-- A session's level-0 leaves by position. -/
abbrev Leaves0 {J : ℕ} (zs : Fin J → ZkView.Sizes) : Type := Σ j, Fin ((zs j).nS + (zs j).nU)

/-- A session's other pre-final leaves: each rep's pads tree's and its commitment to `τ`. -/
abbrev Leaves1 {J : ℕ} (zs : Fin J → ZkView.Sizes) : Type := Σ j, (Fin 2 × Fin ((zs j).nQ + (zs j).nPU)) ⊕ Fin 2

/-- A session's pre-final leaves, level 0's and the others. -/
def leaves01 {J : ℕ} (zs : Fin J → ZkView.Sizes) : ZkView.Leaves zs ≃ Leaves0 zs ⊕ Leaves1 zs where
  toFun
    | ⟨j, .inl p⟩ => .inl ⟨j, p⟩
    | ⟨j, .inr q⟩ => .inr ⟨j, q⟩
  invFun
    | .inl ⟨j, p⟩ => ⟨j, .inl p⟩
    | .inr ⟨j, q⟩ => ⟨j, .inr q⟩
  left_inv l := by rcases l with ⟨j, p | q⟩ <;> rfl
  right_inv l := by rcases l with ⟨j, p⟩ | ⟨j, q⟩ <;> rfl

/-- **The pre-final leaves' pairs at level 0's salts `y0`**: level 0's `(M·y, c(y))` under `y0`, the other leaves' `p1`
(the ideal-leaf hybrid's). -/
noncomputable def pairs01 {J : ℕ} {zs : Fin J → ZkView.Sizes} (y0 : Leaves0 zs → ZkView.Salt)
    (p1 : Leaves1 zs → ZkView.Bits × ByteArray) : ZkView.Leaves zs → ZkView.Bits × ByteArray := fun l =>
  Sum.elim (fun l0 => (ZkView.My (y0 l0), ZkView.saltHash (y0 l0))) p1 (leaves01 zs l)

theorem leaves01_elim_congr {J : ℕ} {zs : Fin J → ZkView.Sizes} {α γ : Type} (F : (ZkView.Leaves zs → α) → γ)
    (f : ZkView.Leaves zs → α) :
    F (fun l => Sum.elim (fun l0 => f ((leaves01 zs).symm (.inl l0))) (fun l1 => f ((leaves01 zs).symm (.inr l1)))
      (leaves01 zs l)) = F f :=
  congrArg F (funext fun l => by rcases l with ⟨j, l | l⟩ <;> rfl)

theorem leaves01_split_congr {J : ℕ} {zs : Fin J → ZkView.Sizes} {α β γ : Type} (F : (ZkView.Leaves zs → α) → γ)
    (f : ZkView.Leaves zs → β → α) (g : Leaves0 zs → β) (h : Leaves1 zs → β) :
    F (fun l => Sum.elim (fun l0 => f ((leaves01 zs).symm (.inl l0)) (g l0))
      (fun l1 => f ((leaves01 zs).symm (.inr l1)) (h l1)) (leaves01 zs l)) =
      F (fun l => f l (Sum.elim g h (leaves01 zs l))) :=
  congrArg F (funext fun l => by rcases l with ⟨j, l | l⟩ <;> rfl)

theorem leaves01_symm_congr {J : ℕ} {zs : Fin J → ZkView.Sizes} {α γ : Type} (F : (ZkView.Leaves zs → α) → γ)
    (f : ZkView.Leaves zs → α) : F f = F (fun l => f ((leaves01 zs).symm (leaves01 zs l))) :=
  congrArg F (funext fun l => by rw [Equiv.symm_apply_apply])

/-- What the workers see at bases `ω` and level 0's salts `y0` by position. -/
def seenAt {J : ℕ} (C : Fin J → CoinedSetup) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (y0 : Leaves0 (fun j => (C j).z) → ZkView.Salt) : ∀ j, ZkView.Seen (C j).z :=
  fun j => ((ω j).1, (ω j).2.1, fun p => y0 ⟨j, p⟩)

/-- What the workers see at bases and the pre-final leaves' salts. -/
def seenOfP {J : ℕ} (C : Fin J → CoinedSetup)
    (q : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Salt)) :
    ∀ j, ZkView.Seen (C j).z :=
  seenAt C q.1 fun l => q.2 ((leaves01 _).symm (.inl l))

open Finset in
/-- **Lemma A at the switched prover, at the session's tables**: with the pads trees' and `τ`'s leaves ideal and level
0's real under salts `y0` that the values `vs` may read, any function of the cut view takes every value as often at the
switched prover on `wt`, at values read from what the workers see, as at the honest prover on `wt`: on each fiber of
what the workers see the values are constant, and the pre-final fix map, a bijection of each table's base that keeps
what they see and level 0's digests, makes the switched cut view the honest one (`Coined.cut_swPreWith_fixPB`). -/
theorem card_swCutAt_real0 {J : ℕ} {V : Type} [DecidableEq V] (C : Fin J → Composed.CoinedSetup)
    (cz : ∀ j, Composed.CoinsZK (C j).Rest) (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m)
    (vs : (∀ j, ZkView.Seen (C j).z) → ∀ j, ((C j).T (cz j)).tab.Vals) (out : (∀ j, ((C j).T (cz j)).tab.Cut) → V)
    (σ : V) :
    #{u : ((∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt)) ×
          (Leaves1 (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (swCutAt C cz wt (vs (seenAt C u.1.1 u.1.2)) u.1.1 fun l => Sum.elim
          (fun l0 => ZkView.leafH (swDigOf C cz wt (vs (seenAt C u.1.1 u.1.2)) u.1.1 ((leaves01 _).symm (.inl l0)) +
            ZkView.My (u.1.2 l0), ZkView.saltHash (u.1.2 l0)))
          (fun l1 => ZkView.leafH ((u.2 l1).1, ZkView.saltHash (u.2 l1).2)) (leaves01 _ l)) = σ}
      = #{u : ((∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt)) ×
          (Leaves1 (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt u.1.1 fun l => Sum.elim
          (fun l0 => ZkView.leafH (digOf C cz wt u.1.1 ((leaves01 _).symm (.inl l0)) + ZkView.My (u.1.2 l0),
            ZkView.saltHash (u.1.2 l0)))
          (fun l1 => ZkView.leafH ((u.2 l1).1, ZkView.saltHash (u.2 l1).2)) (leaves01 _ l)) = σ} := by
  let cv : (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt) →
      ∀ j, Fin 2 → Fin (C j).z.nE → FlockLevel3.GF128 :=
    fun q j r => vs (seenAt C q.1 q.2) j r (wt j) (q.1 j).1 (q.1 j).2.1
  let fixQ : (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt) ≃
      (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt) :=
    { toFun := fun q => (fun j => ((C j).T (cz j)).fixPB (wt j) (cv q j) (q.1 j), q.2)
      invFun := fun q => (fun j => (((C j).T (cz j)).fixPB (wt j) (cv q j)).symm (q.1 j), q.2)
      left_inv := fun q =>
        Prod.ext (funext fun j => (((C j).T (cz j)).fixPB (wt j) (cv q j)).symm_apply_apply _) rfl
      right_inv := fun q =>
        Prod.ext (funext fun j => (((C j).T (cz j)).fixPB (wt j) (cv q j)).apply_symm_apply _) rfl }
  refine Finset.card_equiv (fixQ.prodCongr (Equiv.refl _)) fun u => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  refine Iff.of_eq (congrArg (fun x => out x = σ) (funext fun j => ?_))
  refine ((C j).T (cz j)).cut_swPreWith_fixPB _
    (famSent_admits (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j) (wt j)
    (vs (seenAt C u.1.1 u.1.2) j) (u.1.1 j) _ _ fun p => ?_
  rcases p with p | p
  · exact congrArg (fun d => ZkView.leafH (ZkView.bitsOf d + ZkView.My (u.1.2 ⟨j, p⟩), ZkView.saltHash (u.1.2 ⟨j, p⟩)))
      ((((C j).T (cz j)).swContent_inl _ _ _ p).trans (((C j).T (cz j)).content_fixPB_inl _ _ _ p).symm)
  · exact congrArg (fun y : ZkView.Bits × ZkView.Salt => ZkView.leafH (y.1, ZkView.saltHash y.2))
      (rfl : u.2 ⟨j, p⟩ = u.2 ⟨j, p⟩)

open Finset in
/-- `card_cutAt_ideal` with the session's leaves split into level 0's and the others (`leaves01`). -/
theorem card_cutAt_ideal01 {J : ℕ} {V : Type} [DecidableEq V] (C : Fin J → Composed.CoinedSetup)
    (cz : ∀ j, Composed.CoinsZK (C j).Rest) (wt wt' : ∀ j, FlockSoundness.Model.Witness (C j).S.m)
    (hreg : ∀ j, ((C j).T (cz j)).tab.sReg (wt j) = ((C j).T (cz j)).tab.pubReg)
    (hreg' : ∀ j, ((C j).T (cz j)).tab.sReg (wt' j) = ((C j).T (cz j)).tab.pubReg)
    (out : (∀ j, ((C j).T (cz j)).tab.Cut) → V) (σ : V) :
    #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) ⊕ Leaves1 (fun j => (C j).z) →
          ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt u.1 fun l => ZkView.leafH ((u.2 (leaves01 _ l)).1, ZkView.saltHash (u.2 (leaves01 _ l)).2)) = σ}
      = #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) ⊕ Leaves1 (fun j => (C j).z) →
          ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt' u.1 fun l => ZkView.leafH ((u.2 (leaves01 _ l)).1, ZkView.saltHash (u.2 (leaves01 _ l)).2))
          = σ} := by
  have h : ∀ wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m,
      #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt u.1 fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) = σ}
      = #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) ⊕ Leaves1 (fun j => (C j).z) →
          ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt u.1 fun l => ZkView.leafH ((u.2 (leaves01 _ l)).1, ZkView.saltHash (u.2 (leaves01 _ l)).2))
          = σ} := fun wt =>
    Finset.card_equiv ((Equiv.refl _).prodCongr ((leaves01 _).arrowCongr (Equiv.refl _))) fun u => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      show _ ↔ out (cutAt C cz wt u.1 fun l => ZkView.leafH ((u.2 ((leaves01 _).symm (leaves01 _ l))).1,
        ZkView.saltHash (u.2 ((leaves01 _).symm (leaves01 _ l))).2)) = σ
      simp only [Equiv.symm_apply_apply]
  rw [← h wt, ← h wt']
  exact card_cutAt_ideal C cz wt wt' hreg hreg' out σ

set_option maxRecDepth 8000 in
-- `prCoin` over the product of `J` tables' randomness and salts, each table's bound composed by `prCoin_pi_close`
set_option maxHeartbeats 4000000 in
/-- **Lemma B at the workers' values, at `J` `CoinedSetup`s**: at any coin vector of each table that the honest prover
doesn't refuse, valid witnesses `w j` and any clear values `vs` read from what every table's workers see (`tabSeen`:
the mask slots, level 0's hiding and level 0's salts), the joint switched view and the simulators' at any `w0 j`, each
table's randomness independent, are within `Σ_j 2·|Hid_j|·δ₁` both ways (`fam_sw_shvzk_hm96`). -/
theorem zkViewAtJ_sw {J : ℕ} (C : Fin J → CoinedSetup)
    (hC : ∀ j, Composed.CoinedOK FlockSoundness.execArith (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C) (C j))
    (hreg : ∀ j c w, (C j).Valid w → ((C j).T c).tab.sReg w = ((C j).T c).tab.pubReg)
    {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (c : ∀ j, CoinsZK (C j).Rest)
    (hc : ∀ j, (C j).Admits (c j)) {Out : Type} (enc : (∀ j, ((C j).T (c j)).tab.View) → Out)
    (w w0 : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (hv : ∀ j, (C j).Valid (w j))
    (vs : (∀ j, ZkView.Seen (C j).z) → ∀ j, ((C j).T (c j)).tab.Vals) (E : Out → Prop) :
    FlockSoundness.Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (enc fun j => ((C j).T (c j)).swExe (w j) (vs (tabSeen C c p) j) (p j).1 (p j).2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ FlockSoundness.Game.prCoin
          (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E (enc fun j => ((C j).T (c j)).swExe (w j) (vs (tabSeen C c p) j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁ := by
  have hreg := fun j => hreg j (c j) (w j) (hv j)
  have hT : ∀ j, ((C j).T (c j)).tab.Rank ∧ ((C j).T (c j)).tab.PadOnto ∧
      (∀ r, Function.Bijective fun p : FlockLevel3.GF128 × FlockLevel3.GF128 => ((C j).T (c j)).tab.Wr r p.1 p.2) ∧
      ((C j).T (c j)).tab.PadsOnto ∧ (∀ r, ((C j).T (c j)).tab.β r ≠ 0) ∧ ((C j).T (c j)).tab.InnerHolds (w j) := by
    intro j
    obtain ⟨hL, ht, hS, hT⟩ := hC j
    obtain ⟨⟨hpos, hlt⟩, -, hrank, hGp, hW, hpads, hβ, hinner⟩ := hT (c j) (hc j)
    exact ⟨hrank, FlockSoundness.ZK.padOnto_M1 ((C j).T (c j)).tab FlockSoundness.execArith
      FlockSoundness.execArith_correct (C j).cols hL ht hS (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C)
      ((C j).pos (c j)) hpos hlt hGp (FlockSoundness.ZK.padNonvanishing40 _ hL), hW, hpads, hβ,
      hinner (w j) (hv j) (hreg j)⟩
  let vc : (∀ j, ((C j).T (c j)).tab.Seen) → ∀ j, Fin 2 → Fin ((C j).T (c j)).tab.nE → FlockLevel3.GF128 :=
    fun s j r => vs (fun k => ((C k).T (c k)).seenPos (s k)) j r (w j) (s j).1 (s j).2.1
  have hfam : ∀ p, FlockSoundness.ZK.famSw (fun j => ((C j).T (c j)).tab) (fun j => ((C j).T (c j)).hm96) w vc p
      = fun j => ((C j).T (c j)).swExe (w j) (vs (tabSeen C c p) j) (p j).1 (p j).2 :=
    fun p => funext fun j => congrArg ((C j).T (c j)).tab.toBase
      (FlockSoundness.ZK.Table.Hm96.swViewR_congr _ (w j) _ _ (p j).1 (p j).2 fun _ => rfl)
  have hsim : ∀ p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt),
      (fun j => ((C j).T (c j)).hm96.simR (w0 j) (p j).1 (p j).2)
        = fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2 :=
    fun p => funext fun j => (((C j).T (c j)).simExe_eq (w0 j) (p j).1 (p j).2).symm
  obtain ⟨h1, h2⟩ := FlockSoundness.ZK.fam_sw_shvzk_hm96 (fun j => ((C j).T (c j)).tab)
    (fun j => ((C j).T (c j)).hm96) ZkView.My ZkView.saltHash ZkView.leafH (fun _ => rfl) (fun _ => rfl)
    (fun _ => rfl) hT1 (fun j => (hT j).1) (fun j => (hT j).2.1) (fun j => (hT j).2.2.1) (fun j => (hT j).2.2.2.1)
    (fun j => (hT j).2.2.2.2.1) w hreg (fun j => (hT j).2.2.2.2.2) vc w0 fun v => E (enc v)
  simp only [hfam, hsim] at h1 h2
  exact ⟨h1, h2⟩

namespace Proto

/-- **Where the prover goes on at `c`, its messages are `lift` of the tables' switched views at the workers' values
`vs`** (`Coined.swExe`), read from what every table's workers see (`tabSeen`), at valid table witnesses, on randomness
read through a map with fibers of one size. -/
def TabModel.SwLink {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) (vs : (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz j)).tab.Vals) :
    Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧ (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    ∀ r, G.realVec H x w ν r c =
      M.lift fun j => ((M.C j).T (M.cz j)).swExe (M.wt w j) (vs (tabSeen M.C M.cz (M.eR r)) j) (M.eR r j).1
        (M.eR r j).2

/-- **`SHVZK` of the switched prover at a coin vector where it goes on** (`zkViewAtJ_sw`, within `2·n·δ₁`): its
messages at `c` and `simOut`, the honest prover's simulator, on any tape read through a map with fibers of one size,
whatever values the workers supply. -/
theorem TabModel.shvzk_sw {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (vs : (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz j)).tab.Vals) (hL : M.SwLink H ν c w vs)
    (hOK : M.OK) {Ω : Type} [Fintype Ω] [Nonempty Ω] (eS : Ω → M.SimAt) {kS : ℕ}
    (heS : ∀ p, Nat.card {s // eS s = p} = kS) {n : ℕ} (hn : M.size ≤ n) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : Ω => E (M.simOut (eS s)))| ≤ 2 * n * δ₁.toReal := by
  obtain ⟨hv, heR, hR⟩ := hL
  obtain ⟨hset, hA⟩ := hOK
  have hb := zkViewAtJ_sw M.C (fun j => (hset j).ok.1) (fun j => (hset j).ok.2) hT1 M.cz hA M.lift (M.wt w) M.wt₀ hv
    vs E
  have : Nonempty M.SimAt := ⟨eS (Classical.arbitrary _)⟩
  have : Nonempty (∀ j, ((M.C j).T (M.cz j)).tab.Rand × (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt)) :=
    ⟨M.eR (Classical.arbitrary _)⟩
  have eReal : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, ((M.C j).T (M.cz j)).tab.Rand ×
          (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt) =>
        E (M.lift fun j => ((M.C j).T (M.cz j)).swExe (M.wt w j) (vs (tabSeen M.C M.cz p) j) (p j).1
          (p j).2)).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber M.eR heR]
    simp only [hR]
  have eSim : Pr (fun s : Ω => E (M.simOut (eS s))) =
      (FlockSoundness.Game.prCoin fun p : M.SimAt =>
        E (M.lift fun j => ((M.C j).T (M.cz j)).simExe (M.wt₀ j) (p j).1 (p j).2)).toReal := by
    simp only [TabModel.simOut]
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eS heS]
  have hle : ∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁ ≤ 2 * n * δ₁ := by
    rw [← Finset.sum_mul, ← Finset.mul_sum]
    gcongr
    exact_mod_cast hn
  have htop : (2 * n * δ₁ : ENNReal) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hε : (∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁) ≠ ⊤ :=
    ne_top_of_le_ne_top htop hle
  have hεr : (∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁).toReal ≤ 2 * n * δ₁.toReal := by
    have := ENNReal.toReal_mono htop hle
    simpa [ENNReal.toReal_mul] using this
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.2
  rw [eReal, eSim, abs_le]
  constructor <;> linarith

/-- The switched prover's messages before the last coin at the answers `a`, on table witnesses `wt` and the workers'
values `vs a s` at what they see `s`, bases `ω` and pre-final leaves `lv`. -/
noncomputable def TPreModel.swMsgs {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x)
    (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m)
    (vs : ∀ a, (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz a j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi) (s : ∀ j, ZkView.Seen (M.C j).z) (a : Fin (G.K x + 1) → Out)
    (lv : ZkView.Leaves (fun j => (M.C j).z) → ByteArray) : Fin (G.K x + 1) → Msg :=
  M.out a (swCutAt M.C (M.cz a) wt (vs a s) ω lv)

/-- **The switched prover's round encoding reads the answers before each round only**, on one tape: at bases `ω` and
level 0's salts `y0`, what the workers see of them (`seenAt`) and level 0's leaves' pairs under `y0`, each other leaf's
pair `p1` arbitrary (`pairs01`), as `TPreModel.Causal` has the honest prover's. -/
def TPreModel.SwCausal {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x)
    (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m)
    (vs : ∀ a, (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz a j)).tab.Vals) : Prop :=
  ∀ (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi) (y0 : Leaves0 (fun j => (M.C j).z) → ZkView.Salt)
    (p1 : Leaves1 (fun j => (M.C j).z) → ZkView.Bits × ByteArray) (a a' : Fin (G.K x + 1) → Out)
    (i : Fin (G.K x + 1)), (∀ j, j < i → a j = a' j) →
    M.swMsgs wt vs ω (seenAt M.C ω y0) a (fun l => ZkView.leafH (swDigOf M.C (M.cz a) wt (vs a (seenAt M.C ω y0)) ω l +
        (pairs01 y0 p1 l).1, (pairs01 y0 p1 l).2)) i =
      M.swMsgs wt vs ω (seenAt M.C ω y0) a' (fun l => ZkView.leafH
        (swDigOf M.C (M.cz a') wt (vs a' (seenAt M.C ω y0)) ω l + (pairs01 y0 p1 l).1, (pairs01 y0 p1 l).2)) i

/-- **Before the last coin, the prover's messages on `w` are the switched prover's at values the workers supply** (`vs`,
read from what every table's workers see before the last coin, `seenOfP`, and never the pads), its round encoding
causal, at the executable's leaves; on the dummy they are the honest prover's (`TPreModel.Link`'s second half). -/
def TPreModel.SwLink {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg) (ν : G.Nonce) (w : W) :
    Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ a j, ((M.C j).T (M.cz a j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz a j)).tab.pubReg) ∧ M.Causal ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∃ vs : ∀ a, (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz a j)).tab.Vals, M.SwCausal (M.wt w) vs ∧
      ∀ r a, (G.preStrat H x ν (G.msg x w ν)).msg r a =
        M.swMsgs (M.wt w) vs (M.eR r).1 (seenOfP M.C (M.eR r)) a
          (swExeOf M.C (M.cz a) (M.wt w) (vs a (seenOfP M.C (M.eR r))) (M.eR r).1 (M.eR r).2)) ∧
    ∀ r a, (G.preStrat H x ν (G.dmsg x ν)).msg r a =
      M.msgs M.wt₀ (M.eR r).1 a (exeOf M.C (M.cz a) M.wt₀ (M.eR r).1 (M.eR r).2)

set_option maxRecDepth 20000 in
/-- **Lemma C's bound on the runs before the last coin at the switched prover** (`runs_close_sw_of_ideal`, its prover
reading level 0's salts the switched prover on `w`, its table witnesses `w`'s and the dummy's), within `2·|L|·δ₁`,
whatever values the workers supply. -/
theorem TPreModel.runs_close_sw {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (w : W)
    (hL : M.SwLink H ν w) (hOK : M.OK) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
    (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, hc, heR, ⟨vs, hsc, hlink⟩, hlink₀⟩ := hL
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  let e := leaves01 fun j => (M.C j).z
  let Φ : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (ZkView.Leaves (fun j => (M.C j).z) → ZkView.Salt) ≃
      (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) ⊕ Leaves1 (fun j => (M.C j).z) →
        ZkView.Salt) :=
    (Equiv.refl _).prodCongr (e.arrowCongr (Equiv.refl _))
  have heR' : ∀ p, Nat.card {r // Φ (M.eR r) = p} = M.kR := fun p => by
    rw [← heR (Φ.symm p)]
    exact Nat.card_congr (Equiv.subtypeEquivRight fun r => (Φ.eq_symm_apply (x := p) (y := M.eR r)).symm)
  let P : (∀ j, FlockSoundness.Model.Witness (M.C j).S.m) → FlockSoundness.ZK.Strategy
      ((∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) ⊕ Leaves1 (fun j => (M.C j).z) →
        ZkView.Bits × ByteArray)) Msg Out (G.K x + 1) := fun wt =>
    { msg := fun q a => M.msgs wt q.1 a fun l =>
        ZkView.leafH (digOf M.C (M.cz a) wt q.1 (e.symm (e l)) + (q.2 (e l)).1, (q.2 (e l)).2)
      causal := fun q a a' i h => by
        simp only [Equiv.symm_apply_apply]
        exact hc wt q.1 (fun l => q.2 (e l)) a a' i h }
  let msgsS : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) → ZkView.Salt) →
      (Fin (G.K x + 1) → Out) → (Leaves1 (fun j => (M.C j).z) → ByteArray) → Fin (G.K x + 1) → Msg :=
    fun q a lv1 => M.swMsgs (M.wt w) vs q.1 (seenAt M.C q.1 q.2) a fun l => Sum.elim
      (fun l0 => ZkView.leafH (swDigOf M.C (M.cz a) (M.wt w) (vs a (seenAt M.C q.1 q.2)) q.1 (e.symm (.inl l0)) +
        ZkView.My (q.2 l0), ZkView.saltHash (q.2 l0))) lv1 (e l)
  let digS : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) → ZkView.Salt) →
      (Fin (G.K x + 1) → Out) → Leaves1 (fun j => (M.C j).z) → ZkView.Bits :=
    fun q a l1 => swDigOf M.C (M.cz a) (M.wt w) (vs a (seenAt M.C q.1 q.2)) q.1 (e.symm (.inr l1))
  have hsS : ∀ q (p : Leaves1 (fun j => (M.C j).z) → ZkView.Bits × ByteArray) (a a' : Fin (G.K x + 1) → Out)
      (i : Fin (G.K x + 1)), (∀ j, j < i → a j = a' j) →
      msgsS q a (fun l1 => ZkView.leafH (digS q a l1 + (p l1).1, (p l1).2)) i =
        msgsS q a' (fun l1 => ZkView.leafH (digS q a' l1 + (p l1).1, (p l1).2)) i := by
    intro q p a a' i h
    have k := fun a : Fin (G.K x + 1) → Out => leaves01_split_congr
      (M.swMsgs (M.wt w) vs q.1 (seenAt M.C q.1 q.2) a)
      (fun l y => ZkView.leafH (swDigOf M.C (M.cz a) (M.wt w) (vs a (seenAt M.C q.1 q.2)) q.1 l + y.1, y.2))
      (fun l0 => (ZkView.My (q.2 l0), ZkView.saltHash (q.2 l0))) p
    exact (congrFun (k a) i).trans ((hsc q.1 q.2 p a a' i h).trans (congrFun (k a') i).symm)
  let S : FlockSoundness.ZK.Strategy (((∀ j, ZkView.Base (M.C j).z (M.C j).Hi) ×
      (Leaves0 (fun j => (M.C j).z) → ZkView.Salt)) × (Leaves1 (fun j => (M.C j).z) → ZkView.Bits × ByteArray))
      Msg Out (G.K x + 1) :=
    { msg := fun qp a => msgsS qp.1 a fun l1 => ZkView.leafH (digS qp.1 a l1 + (qp.2 l1).1, (qp.2 l1).2)
      causal := fun qp a a' i h => hsS qp.1 qp.2 a a' i h }
  have hl : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      S.msg (((Φ (M.eR r)).1, fun l => (Φ (M.eR r)).2 (.inl l)),
        fun l => (ZkView.My ((Φ (M.eR r)).2 (.inr l)), ZkView.saltHash ((Φ (M.eR r)).2 (.inr l)))) :=
    fun r => funext fun a => by
      rw [hlink r a, swExeOf_eq]
      have k := leaves01_elim_congr (M.swMsgs (M.wt w) vs (M.eR r).1 (seenOfP M.C (M.eR r)) a) fun l =>
        ZkView.leafH (swDigOf M.C (M.cz a) (M.wt w) (vs a (seenOfP M.C (M.eR r))) (M.eR r).1 l +
          ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k.symm
  have hl₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P M.wt₀).msg ((Φ (M.eR r)).1, fun l => (ZkView.My ((Φ (M.eR r)).2 l), ZkView.saltHash ((Φ (M.eR r)).2 l))) :=
    fun r => funext fun a => by
      rw [hlink₀ r a, exeOf_eq]
      have k := leaves01_symm_congr (M.msgs M.wt₀ (M.eR r).1 a) fun l =>
        ZkView.leafH (digOf M.C (M.cz a) M.wt₀ (M.eR r).1 l + ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k
  rw [TPreModel.size, Fintype.card_congr e]
  exact G.runs_close_sw_of_ideal H x ν w ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH
    (fun wt ω a lv => M.msgs wt ω a fun l => lv (e l)) (fun wt ω a l => digOf M.C (M.cz a) wt ω (e.symm l))
    P (fun _ _ _ _ => rfl) msgsS digS S (fun _ _ _ => rfl) (M.wt w) M.wt₀
    (fun a σ => card_swCutAt_real0 M.C (M.cz a) (M.wt w) (vs a) (M.out a) σ)
    (fun a σ => card_cutAt_ideal01 M.C (M.cz a) (M.wt w) M.wt₀ (fun j => (hOK j).ok.2 _ _ (hv j)) (hreg₀ a)
      (M.out a) σ)
    (fun r => Φ (M.eR r)) M.kR heR' hl hl₀ V E

/-- **Where the prover refuses at `c`, its messages on `w` are `out` of the switched prover's cut view at values the
workers supply** (read from what every table's workers see before the last coin, `seenOfP`), at the executable's
leaves; on the dummy they are the honest prover's (`TOffModel.Link`'s second half). -/
def TOffModel.SwLink {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ j, ((M.C j).T (M.cz j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz j)).tab.pubReg) ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∃ vs : (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz j)).tab.Vals, ∀ r, G.realVec H x w ν r c =
      M.out (swCutAt M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1
        (swExeOf M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1 (M.eR r).2))) ∧
    ∀ r, G.dVec H x ν r c = M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 (exeOf M.C M.cz M.wt₀ (M.eR r).1 (M.eR r).2))

/-- **Lemma C where the prover refuses, at the switched prover** (`refused_close_sw_of_ideal`, its prover reading level
0's salts the switched prover on `w`, its table witnesses `w`'s and the dummy's): within `2·|L|·δ₁`, whatever values
the workers supply. -/
theorem TOffModel.close_sw {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (hL : M.SwLink H ν c w) (hOK : M.OK) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun r : G.Rand => E (G.dVec H x ν r c))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, heR, ⟨vs, hR⟩, hD⟩ := hL
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  let e := leaves01 fun j => (M.C j).z
  let Φ : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (ZkView.Leaves (fun j => (M.C j).z) → ZkView.Salt) ≃
      (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) ⊕ Leaves1 (fun j => (M.C j).z) →
        ZkView.Salt) :=
    (Equiv.refl _).prodCongr (e.arrowCongr (Equiv.refl _))
  have heR' : ∀ p, Nat.card {r // Φ (M.eR r) = p} = M.kR := fun p => by
    rw [← heR (Φ.symm p)]
    exact Nat.card_congr (Equiv.subtypeEquivRight fun r => (Φ.eq_symm_apply (x := p) (y := M.eR r)).symm)
  rw [TOffModel.size, Fintype.card_congr e]
  exact G.refused_close_sw_of_ideal H x w ν c ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH
    (fun wt ω lv => M.out (cutAt M.C M.cz wt ω fun l => lv (e l))) (fun wt ω l => digOf M.C M.cz wt ω (e.symm l))
    (fun q lv1 => M.out (swCutAt M.C M.cz (M.wt w) (vs (seenAt M.C q.1 q.2)) q.1 fun l => Sum.elim
      (fun l0 => ZkView.leafH (swDigOf M.C M.cz (M.wt w) (vs (seenAt M.C q.1 q.2)) q.1 (e.symm (.inl l0)) +
        ZkView.My (q.2 l0), ZkView.saltHash (q.2 l0))) lv1 (e l)))
    (fun q l1 => swDigOf M.C M.cz (M.wt w) (vs (seenAt M.C q.1 q.2)) q.1 (e.symm (.inr l1)))
    (M.wt w) M.wt₀
    (fun σ => card_swCutAt_real0 M.C M.cz (M.wt w) vs M.out σ)
    (fun σ => card_cutAt_ideal01 M.C M.cz (M.wt w) M.wt₀ (fun j => (hOK j).ok.2 _ _ (hv j)) hreg₀ M.out σ)
    (fun r => Φ (M.eR r)) M.kR heR'
    (fun r => by
      rw [hR r, swExeOf_eq]
      have k := leaves01_elim_congr
        (fun lv => M.out (swCutAt M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1 lv)) fun l =>
          ZkView.leafH (swDigOf M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1 l +
            ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k.symm)
    (fun r => by
      rw [hD r, exeOf_eq]
      have k := leaves01_symm_congr (fun lv => M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 lv)) fun l =>
        ZkView.leafH (digOf M.C M.cz M.wt₀ (M.eR r).1 l + ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k)
    E

/-- The switched prover's message link at a coin vector: `TabModel.SwLink` at some values the workers supply where
the prover goes on; where it refuses, `TOffModel.SwLink` at some model whose tables the session sets up wherever `M`'s
are, with at most `M`'s leaves. The simulator reads nothing of the model where the prover refuses
(`TCoinModel.simAt`), so that model needn't be `M`. -/
def TCoinModel.SwLink {W : Type} {G : Proto W} {x : Msg} :
    G.TCoinModel x → (Msg → Msg) → G.Nonce → (Fin (G.K x) → G.Coin) → W → Prop
  | .inl M => fun H ν c w => ∃ vs, M.SwLink H ν c w vs
  | .inr M => fun H ν c w => ∃ M' : G.TOffModel x, M'.SwLink H ν c w ∧ (M.OK → M'.OK) ∧ M'.size ≤ M.size

theorem TCoinModel.swLink_inr {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) (h : M.SwLink H ν c w) :
    TCoinModel.SwLink (G := G) (x := x) (.inr M) H ν c w :=
  ⟨M, h, id, le_rfl⟩

/-- **`SHVZK` of the switched prover at a coin vector, from its model**, for any simulator that is `simAt` on a tape
read through a map with fibers of one size: the honest prover's simulator, whatever values the workers supply. -/
theorem TCoinModel.shvzk_sw {W : Type} {G : Proto W} {x : Msg} (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) {n : ℕ} (E : List Msg → Prop) {Ω : Type} [Fintype Ω] [Nonempty Ω] (sim : Ω → List Msg) :
    ∀ (M : G.TCoinModel x) (ev : Ω → M.SimAt) (kS : ℕ), (∀ p, Nat.card {s // ev s = p} = kS) →
      (∀ s, sim s = ((M.simAt (ev s) ν c).eval H).1) → M.SwLink H ν c w → M.OK → M.size ≤ n →
      |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : Ω => E (sim s))| ≤ 2 * n * δ₁.toReal
  | .inl M, ev, _, hev, hsim, ⟨vs, hL⟩, hOK, hn => by
      have : sim = fun s => M.simOut (ev s) := funext hsim
      subst this
      exact M.shvzk_sw H hT1 hδ₁ ν c w vs hL hOK ev hev hn E
  | .inr M, ev, _, hev, hsim, ⟨M', hL, hOK', hn'⟩, hOK, hn => by
      have hD : ∀ s, sim s = G.dVec H x ν (ev s) c := fun s => by
        rw [hsim s]
        simp only [TCoinModel.simAt, ocSeq_eval, List.map_map]
        rfl
      have e : Pr (fun s : Ω => E (sim s)) = Pr (fun r : G.Rand => E (G.dVec H x ν r c)) := by
        simp only [hD]
        exact Pr_comp_fiber ev hev fun r => E (G.dVec H x ν r c)
      rw [e]
      have hn'' : (M'.size : ℝ) ≤ n := by exact_mod_cast hn'.trans hn
      calc _ ≤ 2 * M'.size * δ₁.toReal := M'.close_sw H hT1 hδ₁ ν c w hL (hOK' hOK) E
        _ ≤ 2 * n * δ₁.toReal := by gcongr

/-- **`SHVZK` of the simulator the models give, at the switched prover**, within `2·n_B·δ₁`, from the switched message
links and the tables: the honest prover's simulator (`withSim`), whatever values the workers supply. -/
theorem withSim_shvzk_sw {W : Type} (G : Proto W) [Fintype G.Coin] (H : Msg → Msg) (R : Rel W) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nB : Msg → ℕ)
    (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x) (hL : ∀ x w, R x w → ∀ ν c, (Mc x ν c).SwLink H ν c w)
    (hOK : ∀ x ν c, (Mc x ν c).OK) : (G.withSim Mc).SHVZK H R fun x => 2 * nB x * δ₁.toReal := by
  intro x w hR ν c E
  obtain ⟨p₀⟩ := TCoinModel.instNonemptySimAt (Mc x ν c)
  exact TCoinModel.shvzk_sw (G := G) H hT1 hδ₁ ν c w E (Ω := (G.withSim Mc).SimRand x)
    (fun s => (((G.withSim Mc).sim x ν c s).eval H).1) (Mc x ν c) (fun s => s (ν, c)) _
    (fun p => eval_fiber_card (β := fun q : G.Nonce × (Fin (G.K x) → G.Coin) => (Mc x q.1 q.2).SimAt) (ν, c) p p₀)
    (fun _ => rfl) (hL x w hR ν c) (hOK x ν c) (hnB x ν c)

/-- **The switched prover's message link at the session's tables**: before the last coin, and at each coin vector, the
prover's messages are the switched prover's at values the workers supply (`TPreModel.SwLink`, `TCoinModel.SwLink`). -/
def TSwLink {W : Type} (G : Proto W) (H : Msg → Msg) (R : Rel W) (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : Prop :=
  ∀ x w, R x w → ∀ ν, (Mp x ν).SwLink H ν w ∧ ∀ c, (Mc x ν c).SwLink H ν c w

end Proto

/-- **Flock's zero knowledge at the switched prover, at the tables the session sets up**: `Flock.zk_deployZk_of_tables`
with `TMsgLink` replaced by `TSwLink`. At `P_zk = G.prover`, a prover whose messages on each witness are the switched
prover's at whatever values the workers supply (read from every table's mask slot, level 0's hiding and level 0's salts,
never the pads), and the honest prover's simulator, `Claims.Flock.ZKAt` holds at the same `ε_zk` and `γ`. -/
theorem Flock.zk_deployZk_of_tables_sw (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) [Fintype G.Coin]
    (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TSwLink H (Flock.R (FlockDeploy.deployZk rel).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.CoinCheck) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  obtain ⟨_, _, Tr, nb, root, parse, pos, idx, n, depth, dc, hchk, hdec⟩ := hC
  have hA : ∀ x w, Flock.R (FlockDeploy.deployZk rel).parse x w → ∀ (ν : G.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤
        2 * nL x * δ₁.toReal := by
    intro x w hR ν V E
    calc _ ≤ 2 * (Mp x ν).size * δ₁.toReal :=
          (Mp x ν).runs_close_sw H hT1 hδ₁ ν w (hL x w hR ν).1 (hOK.1 x ν) V E
      _ ≤ 2 * nL x * δ₁.toReal := by
          have : ((Mp x ν).size : ℝ) ≤ nL x := by exact_mod_cast hnL x ν
          gcongr
  rw [← G.withSim_prover Mc]
  exact Flock.zkR_gkE_hm96_deployZk H rel (G.withSim Mc) t ht hK δ₁ nL nB hA
    (G.withSim_shvzk_sw H _ hT1 hδ₁ Mc nB hnB (fun x w hR ν c => (hL x w hR ν).2 c) hOK.2) Tr nb root parse pos idx n
    depth dc hchk hdec

end Claims.ZKMal
