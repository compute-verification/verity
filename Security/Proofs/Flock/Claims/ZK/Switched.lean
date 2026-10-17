import Proofs.Flock.Claims.ZK.Deployed
import Proofs.Flock.Claims.ZK.Worker

/-!
# The switched prover at the session's tables

`Worker` proves Lemma B at a worker's clear values for one table. Here it is at the `J` tables of a `--zk` session, in
the form the session's models read (`Deployed`):

* `Coined.swExe`: a `--zk` table's switched view at a worker's values `v`, with the executable's hidden leaves
  (`Table.Hm96.swViewR`).
* `Coined.swPreWith`, `Coined.swContent`: its pre-final view with the leaves given by position, and what each leaf
  commits; `swCutAt`, `swDigOf`, `swExeOf` at the session's tables, as `cutAt`, `digOf`, `exeOf` have the honest prover.
* `famCard_swCut_eq`, `card_swCutAt_ideal`: **Lemma A at the switched prover**, at the session's tables: at ideal
  leaves, any function of the cut view takes every value as often at the switched prover, whatever values the workers
  supply, as at the honest prover on the dummy (`Table.sw_prefinal_cut` at each table).
* `zkViewAtJ_sw`: at `J` `CoinedSetup`s, at coins each admits and valid witnesses, for any values the workers supply, the
  joint switched view and the tables' `S_shvzk` on any dummy are within `Σ_j 2·|Hid_j|·δ₁` both ways, as
  `ZkHidden.zkViewAtJ_of_ok` has it for the honest prover.
* `Proto.TabModel.SwLink`, `Proto.TabModel.shvzk_sw`: where the prover goes on, a prover whose messages are `lift` of the
  tables' switched views at the workers' values meets `SHVZK` against `simOut`, the honest prover's simulator, within
  `2·n·δ₁`, as `TabModel.shvzk` has it at `TabModel.Link`.
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (CoinedSetup CoinsZK)
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

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

/-- The base with each rep's free pads moved from the honest values to `v`'s at its mask slot and extra lanes
(`Table.shift`). -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.shiftB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi) :
    ZkView.Base z Hi :=
  let s : Fin 2 → (Fin z.nE → FlockLevel3.GF128) × FlockLevel3.GF128 × FlockLevel3.GF128 :=
    T.tab.onVals.padShift (w, v) (w, T.tab.vals) b.1 b.2.1.2.1
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

/-- At ideal leaves `H(U, c(y))`, the switched pre-final view with its leaves given by position is `Table.swPre`. -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swPreWith_ideal {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (u : ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) :
    T.swPreWith w v b (fun p => ZkView.leafH ((u p).1, ZkView.saltHash (u p).2)) = T.tab.swPre w v (T.randOf b u) := by
  by_cases hr : T.tab.onVals.RowsAt (w, v) b.1 b.2.1.2.1
  · have h0 : T.tab.swPre w v (T.randOf b 0) = T.tab.toBasePre (T.tab.onVals.preView (w, v) (T.randOf b 0)) := by
      unfold FlockSoundness.ZK.Table.swPre
      exact ite_eq_left hr
    have hu : T.tab.swPre w v (T.randOf b u) = T.tab.toBasePre (T.tab.onVals.preView (w, v) (T.randOf b u)) := by
      unfold FlockSoundness.ZK.Table.swPre
      exact ite_eq_left hr
    rw [ZkView.Coined.swPreWith, h0, hu]
    rfl
  · have h0 : T.tab.swPre w v (T.randOf b 0) =
        T.tab.preView w (T.tab.onVals.shift (w, v) (w, T.tab.vals) (T.randOf b 0)) := by
      unfold FlockSoundness.ZK.Table.swPre
      exact ite_eq_right hr
    have hu : T.tab.swPre w v (T.randOf b u) =
        T.tab.preView w (T.tab.onVals.shift (w, v) (w, T.tab.vals) (T.randOf b u)) := by
      unfold FlockSoundness.ZK.Table.swPre
      exact ite_eq_right hr
    rw [ZkView.Coined.swPreWith, h0, hu]
    rfl

open Classical in
/-- **What each of the switched prover's pre-final leaves commits**: `v`'s columns and `τ` where the rows hold, else the
honest prover's on the moved pads. -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.swContent {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi) :
    ZkView.LeafPos z → ByteArray :=
  if T.tab.onVals.RowsAt (w, v) b.1 b.2.1.2.1 then T.onVals.content (w, v) b else T.content w (T.shiftB w v b)

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swContent_size {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (p : ZkView.LeafPos z) : (T.swContent w v b p).size = 64 := by
  unfold ZkView.Coined.swContent
  split_ifs
  · exact T.onVals.content_size _ _ _
  · exact T.content_size _ _ _

/-! ## The switched prover before the last coin, at the session's tables -/

open Finset in
/-- **Lemma A at the switched prover, for tables of their own types** (`Table.sw_prefinal_cut` at each table): messages
that are a function of the session's cut pre-view, on the tables' randomness read through a map with fibers of one
size, take every value as often at the switched prover on `w`, whatever values `v` the workers supply, as at the honest
prover on `w₀`, both carrying the public region data. -/
theorem famCard_swCut_eq {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
    [DecidableEq Seed] {J : ℕ} {Wf Hi VL : Fin J → Type} [∀ j, AddCommGroup (Hi j)] [∀ j, Fintype (Hi j)]
    [∀ j, DecidableEq (Hi j)] {Ω V : Type} [Fintype Ω] [DecidableEq V]
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (e : Ω → ∀ j, (T j).Rand) {k : ℕ} (he : ∀ r, Nat.card {x // e x = r} = k) (out : (∀ j, (T j).Cut) → V)
    (w w₀ : ∀ j, Wf j) (v : ∀ j, (T j).Vals) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (σ : V) :
    #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).swPre (w j) (v j) (e x j))) = σ}
      = #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).preView (w₀ j) (e x j))) = σ} := by
  classical
  have : Nonempty (∀ j, (T j).Rand) := ⟨fun _ => 0⟩
  rw [FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).swPre (w j) (v j) (r j))),
    FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).preView (w₀ j) (r j)))]
  congr 1
  exact ((FlockSoundness.ZK.SameDist.pi
    (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).swPre (w j) (v j) ω))
    (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).preView (w₀ j) ω))
    fun j => (T j).sw_prefinal_cut (famSent T rep j) (famSent_admits T rep j) (w j) (w₀ j) (hreg j) (hreg₀ j)
      (v j)).comp out).card_eq σ

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

open Finset in
/-- **World `W₀` at the switched prover, at the session's tables**: at ideal leaves `H(U, c(y))`, any function of the
cut view takes every value as often at the switched prover on `wt`, whatever values `v` the workers supply, as at the
honest one on `wt₀`, both carrying the public region data (`famCard_swCut_eq`, `Coined.swPreWith_ideal`). -/
theorem card_swCutAt_ideal {J : ℕ} {V : Type} [DecidableEq V] (C : Fin J → Composed.CoinedSetup)
    (cz : ∀ j, Composed.CoinsZK (C j).Rest) (wt wt₀ : ∀ j, FlockSoundness.Model.Witness (C j).S.m)
    (v : ∀ j, ((C j).T (cz j)).tab.Vals) (hreg : ∀ j, ((C j).T (cz j)).tab.sReg (wt j) = ((C j).T (cz j)).tab.pubReg)
    (hreg₀ : ∀ j, ((C j).T (cz j)).tab.sReg (wt₀ j) = ((C j).T (cz j)).tab.pubReg)
    (out : (∀ j, ((C j).T (cz j)).tab.Cut) → V) (σ : V) :
    #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (swCutAt C cz wt v u.1 fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) = σ}
      = #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt₀ u.1 fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) = σ} := by
  have e : ∀ u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt),
      swCutAt C cz wt v u.1 (fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) =
        fun j => ((C j).T (cz j)).tab.cut
          (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j)
          (((C j).T (cz j)).tab.swPre (wt j) (v j) (famRandAt (fun j => (C j).T (cz j)) u j)) :=
    fun u => funext fun j => congrArg _ (((C j).T (cz j)).swPreWith_ideal (wt j) (v j) (u.1 j) fun p => u.2 ⟨j, p⟩)
  simp only [e]
  exact famCard_swCut_eq (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep)
    (famRandAt fun j => (C j).T (cz j))
    (fun r => ZkView.card_fiber_of_bijective (famRandAt_bijective fun j => (C j).T (cz j)) r) out wt wt₀ v hreg hreg₀ σ

set_option maxRecDepth 8000 in
set_option maxHeartbeats 4000000 in
/-- **Lemma B at the workers' values, at `J` `CoinedSetup`s**: at any coin vector of each table that the honest prover
doesn't refuse, valid witnesses `w j` and any clear values `v j` the workers supply, the joint switched view and the
simulators' at any `w0 j`, each table's randomness independent, are within `Σ_j 2·|Hid_j|·δ₁` both ways
(`Table.worker_shvzk_hm96` at each table, `prCoin_pi_close`). -/
theorem zkViewAtJ_sw {J : ℕ} (C : Fin J → CoinedSetup)
    (hC : ∀ j, Composed.CoinedOK FlockSoundness.execArith (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C) (C j))
    (hreg : ∀ j c w, (C j).Valid w → ((C j).T c).tab.sReg w = ((C j).T c).tab.pubReg)
    {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (c : ∀ j, CoinsZK (C j).Rest)
    (hc : ∀ j, (C j).Admits (c j)) {Out : Type} (enc : (∀ j, ((C j).T (c j)).tab.View) → Out)
    (w w0 : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (hv : ∀ j, (C j).Valid (w j))
    (v : ∀ j, ((C j).T (c j)).tab.Vals) (E : Out → Prop) :
    FlockSoundness.Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (enc fun j => ((C j).T (c j)).swExe (w j) (v j) (p j).1 (p j).2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ FlockSoundness.Game.prCoin
          (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E (enc fun j => ((C j).T (c j)).swExe (w j) (v j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁ := by
  have hreg := fun j => hreg j (c j) (w j) (hv j)
  have one : ∀ j (E' : ((C j).T (c j)).tab.View → Prop),
      FlockSoundness.Game.prCoin (fun p : ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E' (((C j).T (c j)).swExe (w j) (v j) p.1 p.2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E' (((C j).T (c j)).simExe (w0 j) p.1 p.2)) + 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ FlockSoundness.Game.prCoin
            (fun p : ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E' (((C j).T (c j)).simExe (w0 j) p.1 p.2))
        ≤ FlockSoundness.Game.prCoin (fun p : ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E' (((C j).T (c j)).swExe (w j) (v j) p.1 p.2)) + 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁ := by
    intro j E'
    obtain ⟨hL, ht, hS, hT⟩ := hC j
    obtain ⟨⟨hpos, hlt⟩, -, hrank, hGp, hW, hpads, hβ, hinner⟩ := hT (c j) (hc j)
    have hpad : ((C j).T (c j)).tab.PadOnto :=
      FlockSoundness.ZK.padOnto_M1 ((C j).T (c j)).tab FlockSoundness.execArith FlockSoundness.execArith_correct (C j).cols hL ht
        hS (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C) ((C j).pos (c j)) hpos hlt hGp (FlockSoundness.ZK.padNonvanishing40 _ hL)
    simp only [ZkView.Coined.simExe_eq]
    exact ((C j).T (c j)).tab.worker_shvzk_hm96 ((C j).T (c j)).hm96 hT1 hrank hpad hW hpads hβ (w j) (hreg j)
      (hinner (w j) (hv j) (hreg j)) (v j) (w0 j) E'
  have : Nonempty ZkView.Salt := ⟨0⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((ZkHidden.rand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((ZkHidden.simRand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  exact ⟨FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View)
      (fun j p => ((C j).T (c j)).swExe (w j) (v j) p.1 p.2) (fun j p => ((C j).T (c j)).simExe (w0 j) p.1 p.2)
      (fun j => 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁) (fun j E' => (one j E').1) fun v => E (enc v),
    FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View)
      (fun j p => ((C j).T (c j)).simExe (w0 j) p.1 p.2) (fun j p => ((C j).T (c j)).swExe (w j) (v j) p.1 p.2)
      (fun j => 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁) (fun j E' => (one j E').2) fun v => E (enc v)⟩

namespace Proto

/-- **Where the prover goes on at `c`, its messages are `lift` of the tables' switched views at the workers' values
`vs`** (`Coined.swExe`), at valid table witnesses, on randomness read through a map with fibers of one size. -/
def TabModel.SwLink {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) (vs : ∀ j, ((M.C j).T (M.cz j)).tab.Vals) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧ (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    ∀ r, G.realVec H x w ν r c =
      M.lift fun j => ((M.C j).T (M.cz j)).swExe (M.wt w j) (vs j) (M.eR r j).1 (M.eR r j).2

/-- **`SHVZK` of the switched prover at a coin vector where it goes on** (`zkViewAtJ_sw`, within `2·n·δ₁`): its
messages at `c` and `simOut`, the honest prover's simulator, on any tape read through a map with fibers of one size,
whatever values the workers supply. -/
theorem TabModel.shvzk_sw {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (vs : ∀ j, ((M.C j).T (M.cz j)).tab.Vals) (hL : M.SwLink H ν c w vs) (hOK : M.OK) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (eS : Ω → M.SimAt) {kS : ℕ} (heS : ∀ p, Nat.card {s // eS s = p} = kS) {n : ℕ}
    (hn : M.size ≤ n) (E : List Msg → Prop) :
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
        E (M.lift fun j => ((M.C j).T (M.cz j)).swExe (M.wt w j) (vs j) (p j).1 (p j).2)).toReal := by
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

end Proto

end Claims.ZKMal
