module

public import Proofs.Flock.Claims.ZK.Hypotheses
public import Proofs.Flock.Claims.ZK.Prefinal
public import Proofs.Flock.Soundness.Discharge.Composed.RegWords
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
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.Discharge.Composed.Setup
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask
import Proofs.Flock.Soundness.Discharge.ZkView.Leaf
import Proofs.Flock.Soundness.Discharge.ZkView.Prefinal
import Proofs.Flock.Soundness.ZK.GK.Link
import Proofs.Flock.Soundness.ZK.RealLeaves

/-! Flock's zero knowledge at the tables `verify --zk` sets up. -/

@[expose] public section

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- instance search over the ZK view's and coins' product types

/-- **A table the session sets up**: `coinedOf` at a table over a statement at its own regions, the statement a
`MaskSetup` that avoids the mask slot, its regions well formed and of whole words. -/
def SetupTable (C : Composed.CoinedSetup) : Prop :=
  ∃ (mPts : ℕ) (P : Composed.LigProver) (Tz : ZkSession.TabZK FlockLevel3.GF128 mPts) (st : _root_.Flock.Stmt)
    (hwf : FlockSoundness.Refine.StmtWF st),
    FlockSoundness.Refine.RegionsWF st ∧
      Tz.S = FlockSoundness.Refine.stmtOf st hwf (FlockSoundness.Refine.regionsOf st) ∧ Composed.MaskSetup st ∧
      Composed.avoidsMask st = true ∧ Composed.WholeWords st ∧ C = Composed.coinedOf P Tz st

theorem SetupTable.ok {C : Composed.CoinedSetup} (h : SetupTable C) :
    Composed.CoinedOK FlockSoundness.execArith (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C) C ∧
      ∀ c w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg := by
  obtain ⟨_, P, Tz, st, hwf, hr, hS, M, hdm, hW, rfl⟩ := h
  exact ⟨(Composed.coinedOf_setup_of P Tz st hwf hS M hdm).2.2.2.2.2, Composed.coinedOf_reg P Tz st hwf hr hS hW⟩

theorem setupTable_planZHJ {mPts n : ℕ} {Reg : Type} (P : Composed.LigProver) (I : Exec.Inputs)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : ZkHidden.SampleSetupZHJ (Composed.inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) (S : Finset (Fin n))
    (R : Reg) (j : Fin (Composed.nTab I))
    (y : ZkHidden.SampleSetupZHJ (Composed.inputsJ I j) mPts (Composed.partS I (dj S) S j) (Composed.drawJ I (dj S) j)) :
    SetupTable (Composed.coinedOf P ((ZkHidden.planZHJ (Reg := Reg) dj y₀ S R).get
      (Fin.cast (ZkHidden.planZHJ_length dj y₀ S R).symm j)) y.x.st) :=
  ⟨mPts, P, _, y.x.st, y.x.wf.1, y.x.wf.2, by rw [ZkHidden.planZHJ_get_eq dj y₀ S R j y]; rfl,
    ZkHidden.maskSetup_of_setupHidden ((Exec.tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) y.x.h,
    FailClosed.SampleSetupZHJ.avoidsMask y, Composed.setupHidden_whole y.x.h, rfl⟩

noncomputable def famRandAt {K : Type} [Field K] {J : ℕ} {Wf Hi VL : Fin J → Type} {zs : Fin J → ZkView.Sizes}
    (Tc : ∀ j, ZkView.Coined K (Wf j) (Hi j) (VL j) (zs j))
    (x : (∀ j, ZkView.Base (zs j) (Hi j)) × (ZkView.Leaves zs → ZkView.Bits × ZkView.Salt)) :
    ∀ j, (Tc j).tab.Rand :=
  fun j => (Tc j).randOf (x.1 j) fun p => x.2 ⟨j, p⟩

theorem famRandAt_bijective {K : Type} [Field K] {J : ℕ} {Wf Hi VL : Fin J → Type} {zs : Fin J → ZkView.Sizes}
    (Tc : ∀ j, ZkView.Coined K (Wf j) (Hi j) (VL j) (zs j)) : Function.Bijective (famRandAt Tc) := by
  constructor
  · intro x x' h
    have hj : ∀ j, ((x.1 j, fun p => x.2 ⟨j, p⟩) :
        ZkView.Base (zs j) (Hi j) × (ZkView.LeafPos (zs j) → ZkView.Bits × ZkView.Salt))
        = (x'.1 j, fun p => x'.2 ⟨j, p⟩) := fun j =>
      (Tc j).randOf_injective (a₁ := (x.1 j, fun p => x.2 ⟨j, p⟩)) (a₂ := (x'.1 j, fun p => x'.2 ⟨j, p⟩))
        (congrFun h j)
    refine Prod.ext (funext fun j => (Prod.ext_iff.1 (hj j)).1) (funext fun l => ?_)
    exact congrFun (Prod.ext_iff.1 (hj l.1)).2 l.2
  · intro r
    choose x hx using fun j => (Tc j).randOf_surjective (r j)
    exact ⟨(fun j => (x j).1, fun l => (x l.1).2 l.2), funext fun j => hx j⟩

/-- **The session's pre-final view at coins `cz`, cut where the session stops** (`famSent`), at table witnesses `wt`,
bases `ω` and the pre-final leaves `lv` by position (`Coined.preWith`). -/
noncomputable def cutAt {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) : ∀ j, ((C j).T (cz j)).tab.Cut :=
  fun j => ((C j).T (cz j)).tab.cut (famSent (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j)
    (((C j).T (cz j)).preWith (wt j) (ω j) fun p => lv ⟨j, p⟩)

/-- The digest each pre-final leaf commits, as bits (`ZkView.digAt`). -/
noncomputable def digOf {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (l : ZkView.Leaves (fun j => (C j).z)) : ZkView.Bits :=
  ZkView.bitsOf (((C l.1).T (cz l.1)).content (wt l.1) (ω l.1) l.2)

/-- **The executable's pre-final leaves** (`Coined.exeLeaf`), each under its own salt. -/
noncomputable def exeOf {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) : ZkView.Leaves (fun j => (C j).z) → ByteArray :=
  fun l => ((C l.1).T (cz l.1)).exeLeaf (wt l.1) (ω l.1) l.2 (y l)

theorem exeOf_eq {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) :
    exeOf C cz wt ω y = fun l => ZkView.leafH (digOf C cz wt ω l + ZkView.My (y l), ZkView.saltHash (y l)) :=
  funext fun l => by
    rw [exeOf, ZkView.Coined.exeLeaf, ZkView.leafSplit _ _ (((C l.1).T (cz l.1)).content_size _ _ _)]
    rfl

open Finset in
theorem card_cutAt_ideal {J : ℕ} {V : Type} [DecidableEq V] (C : Fin J → Composed.CoinedSetup)
    (cz : ∀ j, Composed.CoinsZK (C j).Rest) (wt wt' : ∀ j, FlockSoundness.Model.Witness (C j).S.m)
    (hreg : ∀ j, ((C j).T (cz j)).tab.sReg (wt j) = ((C j).T (cz j)).tab.pubReg)
    (hreg' : ∀ j, ((C j).T (cz j)).tab.sReg (wt' j) = ((C j).T (cz j)).tab.pubReg)
    (out : (∀ j, ((C j).T (cz j)).tab.Cut) → V) (σ : V) :
    #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt u.1 fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) = σ}
      = #{u : (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt' u.1 fun l => ZkView.leafH ((u.2 l).1, ZkView.saltHash (u.2 l).2)) = σ} :=
  famCard_cut_eq (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) (famRandAt fun j => (C j).T (cz j))
    (fun r => ZkView.card_fiber_of_bijective (famRandAt_bijective fun j => (C j).T (cz j)) r) out wt wt' hreg hreg' σ

namespace Proto

structure TPreModel {W : Type} (G : Proto W) (x : Msg) where
  J : ℕ
  C : Fin J → Composed.CoinedSetup
  cz : (Fin (G.K x + 1) → Out) → ∀ j, Composed.CoinsZK (C j).Rest
  out : ∀ a, (∀ j, ((C j).T (cz a j)).tab.Cut) → Fin (G.K x + 1) → Msg
  wt : W → ∀ j, FlockSoundness.Model.Witness (C j).S.m
  wt₀ : ∀ j, FlockSoundness.Model.Witness (C j).S.m
  eR : G.Rand → (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Salt)
  kR : ℕ

noncomputable def TPreModel.msgs {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x)
    (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m) (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi)
    (a : Fin (G.K x + 1) → Out) (lv : ZkView.Leaves (fun j => (M.C j).z) → ByteArray) : Fin (G.K x + 1) → Msg :=
  M.out a (cutAt M.C (M.cz a) wt ω lv)

def TPreModel.Causal {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) : Prop :=
  ∀ (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m) (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi)
    (p : ZkView.Leaves (fun j => (M.C j).z) → ZkView.Bits × ByteArray) (a a' : Fin (G.K x + 1) → Out)
    (i : Fin (G.K x + 1)), (∀ j, j < i → a j = a' j) →
    M.msgs wt ω a (fun l => ZkView.leafH (digOf M.C (M.cz a) wt ω l + (p l).1, (p l).2)) i =
      M.msgs wt ω a' (fun l => ZkView.leafH (digOf M.C (M.cz a') wt ω l + (p l).1, (p l).2)) i

def TPreModel.Link {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg) (ν : G.Nonce) (w : W) :
    Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ a j, ((M.C j).T (M.cz a j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz a j)).tab.pubReg) ∧ M.Causal ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∀ r a, (G.preStrat H x ν (G.msg x w ν)).msg r a =
      M.msgs (M.wt w) (M.eR r).1 a (exeOf M.C (M.cz a) (M.wt w) (M.eR r).1 (M.eR r).2)) ∧
    ∀ r a, (G.preStrat H x ν (G.dmsg x ν)).msg r a =
      M.msgs M.wt₀ (M.eR r).1 a (exeOf M.C (M.cz a) M.wt₀ (M.eR r).1 (M.eR r).2)

def TPreModel.OK {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) : Prop := ∀ j, SetupTable (M.C j)

def TPreModel.size {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) : ℕ :=
  Fintype.card (ZkView.Leaves (fun j => (M.C j).z))

theorem TPreModel.runs_close {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (w : W) (hL : M.Link H ν w)
    (hOK : M.OK) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, hc, heR, hlink, hlink₀⟩ := hL
  let P : (∀ j, FlockSoundness.Model.Witness (M.C j).S.m) → FlockSoundness.ZK.Strategy
      ((∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (ZkView.Leaves (fun j => (M.C j).z) → ZkView.Bits × ByteArray))
      Msg Out (G.K x + 1) := fun wt =>
    { msg := fun q a => M.msgs wt q.1 a fun l => ZkView.leafH (digOf M.C (M.cz a) wt q.1 l + (q.2 l).1, (q.2 l).2)
      causal := fun q a a' i h => hc wt q.1 q.2 a a' i h }
  have hP : ∀ wt ω p a, (P wt).msg (ω, p) a =
      M.msgs wt ω a fun l => ZkView.leafH (digOf M.C (M.cz a) wt ω l + (p l).1, (p l).2) := fun _ _ _ _ => rfl
  have hl : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      (P (M.wt w)).msg ((M.eR r).1, fun l => (ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))) :=
    fun r => funext fun a => by rw [hP, hlink r a, exeOf_eq]
  have hl₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P M.wt₀).msg ((M.eR r).1, fun l => (ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))) :=
    fun r => funext fun a => by rw [hP, hlink₀ r a, exeOf_eq]
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  exact G.runs_close_of_ideal H x ν w ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH M.msgs
    (fun wt ω a => digOf M.C (M.cz a) wt ω) P hP (M.wt w) M.wt₀
    (fun a σ => card_cutAt_ideal M.C (M.cz a) (M.wt w) M.wt₀ (fun j => (hOK j).ok.2 _ _ (hv j)) (hreg₀ a) (M.out a) σ)
    M.eR M.kR heR hl hl₀ V E

/-- **Lemma B's setting at `J` `--zk` tables, at a coin vector where the prover goes on** (`ZkHidden.ZkViewAtJ`): table
`j` is the `CoinedSetup` `C j` at its coins `cz j`, each with its own types; the table witnesses of each witness and of
the dummy, the map `lift` from the tables' views to the prover's messages, and the map `eR` from `G`'s randomness to the
tables' randomness and their hidden leaves' salts. -/
structure TabModel {W : Type} (G : Proto W) (x : Msg) where
  J : ℕ
  C : Fin J → Composed.CoinedSetup
  cz : ∀ j, Composed.CoinsZK (C j).Rest
  wt : W → ∀ j, FlockSoundness.Model.Witness (C j).S.m
  wt₀ : ∀ j, FlockSoundness.Model.Witness (C j).S.m
  lift : (∀ j, ((C j).T (cz j)).tab.View) → List Msg
  eR : G.Rand → ∀ j, ((C j).T (cz j)).tab.Rand × (((C j).T (cz j)).tab.Hid → ZkView.Salt)
  kR : ℕ

def TabModel.Link {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧ (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    ∀ r, G.realVec H x w ν r c = M.lift fun j => ((M.C j).T (M.cz j)).viewExe (M.wt w j) (M.eR r j).1 (M.eR r j).2

def TabModel.OK {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) : Prop :=
  (∀ j, SetupTable (M.C j)) ∧ ∀ j, (M.C j).Admits (M.cz j)

noncomputable def TabModel.size {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) : ℕ :=
  ∑ j, Fintype.card ((M.C j).T (M.cz j)).tab.Hid

/-- `S_shvzk`'s tape at the tables: each one's simulator randomness and its hidden leaves' salts. -/
abbrev TabModel.SimAt {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) : Type :=
  ∀ j, ((M.C j).T (M.cz j)).tab.SimRand × (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt)

/-- `lift` of the tables' `S_shvzk` outputs on the dummy (`Coined.simExe`). -/
noncomputable def TabModel.simOut {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (σ : M.SimAt) : List Msg :=
  M.lift fun j => ((M.C j).T (M.cz j)).simExe (M.wt₀ j) (σ j).1 (σ j).2

theorem TabModel.shvzk {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (hL : M.Link H ν c w) (hOK : M.OK) {Ω : Type} [Fintype Ω] [Nonempty Ω] (eS : Ω → M.SimAt) {kS : ℕ}
    (heS : ∀ p, Nat.card {s // eS s = p} = kS) {n : ℕ} (hn : M.size ≤ n) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : Ω => E (M.simOut (eS s)))| ≤ 2 * n * δ₁.toReal := by
  obtain ⟨hv, heR, hR⟩ := hL
  obtain ⟨hset, hA⟩ := hOK
  have hb := ZkHidden.zkViewAtJ_of_ok M.C (fun j => (hset j).ok.1) (fun j => (hset j).ok.2) hT1 M.cz hA M.lift
    (M.wt w) M.wt₀ hv E
  have : Nonempty M.SimAt := ⟨eS (Classical.arbitrary _)⟩
  have : Nonempty (∀ j, ((M.C j).T (M.cz j)).tab.Rand × (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt)) :=
    ⟨M.eR (Classical.arbitrary _)⟩
  have eReal : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, ((M.C j).T (M.cz j)).tab.Rand ×
          (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt) =>
        E (M.lift fun j => ((M.C j).T (M.cz j)).viewExe (M.wt w j) (p j).1 (p j).2)).toReal := by
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

/-- **Lemma C's setting at fixed answers, at a coin vector where the prover refuses**: table `j` is `C j` at its coins
`cz j`, each with its own types; the prover's messages are `out` of the session's pre-final view cut where it stops
(`cutAt`); the table witnesses of each witness and of the dummy, and the map `eR` from `G`'s randomness to the tables'
bases and the pre-final leaves' salts. -/
structure TOffModel {W : Type} (G : Proto W) (x : Msg) where
  J : ℕ
  C : Fin J → Composed.CoinedSetup
  cz : ∀ j, Composed.CoinsZK (C j).Rest
  out : (∀ j, ((C j).T (cz j)).tab.Cut) → List Msg
  wt : W → ∀ j, FlockSoundness.Model.Witness (C j).S.m
  wt₀ : ∀ j, FlockSoundness.Model.Witness (C j).S.m
  eR : G.Rand → (∀ j, ZkView.Base (C j).z (C j).Hi) × (ZkView.Leaves (fun j => (C j).z) → ZkView.Salt)
  kR : ℕ

def TOffModel.Link {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ j, ((M.C j).T (M.cz j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz j)).tab.pubReg) ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∀ r, G.realVec H x w ν r c =
      M.out (cutAt M.C M.cz (M.wt w) (M.eR r).1 (exeOf M.C M.cz (M.wt w) (M.eR r).1 (M.eR r).2))) ∧
    ∀ r, G.dVec H x ν r c = M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 (exeOf M.C M.cz M.wt₀ (M.eR r).1 (M.eR r).2))

def TOffModel.OK {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) : Prop := ∀ j, SetupTable (M.C j)

def TOffModel.size {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) : ℕ :=
  Fintype.card (ZkView.Leaves (fun j => (M.C j).z))

theorem TOffModel.close {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (hL : M.Link H ν c w) (hOK : M.OK) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun r : G.Rand => E (G.dVec H x ν r c))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, heR, hR, hD⟩ := hL
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  exact G.refused_close_of_ideal H x w ν c ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH
    (fun wt ω lv => M.out (cutAt M.C M.cz wt ω lv)) (digOf M.C M.cz) (M.wt w) M.wt₀
    (fun σ => card_cutAt_ideal M.C M.cz (M.wt w) M.wt₀ (fun j => (hOK j).ok.2 _ _ (hv j)) hreg₀ M.out σ)
    M.eR M.kR heR
    (fun r => (hR r).trans (congrArg (fun lv => M.out (cutAt M.C M.cz (M.wt w) (M.eR r).1 lv))
      (exeOf_eq M.C M.cz (M.wt w) (M.eR r).1 (M.eR r).2)))
    (fun r => (hD r).trans (congrArg (fun lv => M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 lv))
      (exeOf_eq M.C M.cz M.wt₀ (M.eR r).1 (M.eR r).2))) E

/-- **The model at one coin vector, at the session's tables**: `TabModel` where the prover goes on, `TOffModel` where it
refuses. -/
abbrev TCoinModel {W : Type} (G : Proto W) (x : Msg) : Type 1 := G.TabModel x ⊕ G.TOffModel x

def TCoinModel.Link {W : Type} {G : Proto W} {x : Msg} :
    G.TCoinModel x → (Msg → Msg) → G.Nonce → (Fin (G.K x) → G.Coin) → W → Prop
  | .inl M => M.Link
  | .inr M => M.Link

def TCoinModel.OK {W : Type} {G : Proto W} {x : Msg} : G.TCoinModel x → Prop
  | .inl M => M.OK
  | .inr M => M.OK

noncomputable def TCoinModel.size {W : Type} {G : Proto W} {x : Msg} : G.TCoinModel x → ℕ
  | .inl M => M.size
  | .inr M => M.size

/-- The SHVZK simulator's tape at a coin vector: `S_shvzk`'s where the prover goes on, the prover's where it refuses. -/
abbrev TCoinModel.SimAt {W : Type} {G : Proto W} {x : Msg} : G.TCoinModel x → Type
  | .inl M => M.SimAt
  | .inr _ => G.Rand

noncomputable instance TCoinModel.instFintypeSimAt {W : Type} {G : Proto W} {x : Msg} :
    (M : G.TCoinModel x) → Fintype M.SimAt
  | .inl _ => inferInstance
  | .inr _ => G.finRand

instance TCoinModel.instNonemptySimAt {W : Type} {G : Proto W} {x : Msg} : (M : G.TCoinModel x) → Nonempty M.SimAt
  | .inl M => ⟨fun j => ((ZkHidden.simRand_nonempty ((M.C j).T (M.cz j)).tab).some, fun _ => 0)⟩
  | .inr _ => G.neRand

/-- An oracle program's messages, each run in order. -/
def ocSeq : List (OC Msg) → OC (List Msg)
  | [] => .ret []
  | o :: os => ocBind o fun m => ocBind (ocSeq os) fun ms => .ret (m :: ms)

theorem ocSeq_eval (H : Msg → Msg) : ∀ os : List (OC Msg), ((ocSeq os).eval H).1 = os.map fun o => (o.eval H).1
  | [] => rfl
  | o :: os => by simp only [ocSeq, ocBind_eval, OC.eval, List.map_cons, ocSeq_eval H os]

/-- **The SHVZK simulator's messages at a coin vector, from its tape there**: `simOut` where the prover goes on, the
dummy-witness prover's messages at `c` where it refuses. -/
noncomputable def TCoinModel.simAt {W : Type} {G : Proto W} {x : Msg} :
    (M : G.TCoinModel x) → M.SimAt → G.Nonce → (Fin (G.K x) → G.Coin) → OC (List Msg)
  | .inl M, σ, _, _ => .ret (M.simOut σ)
  | .inr _, r, ν, c => ocSeq ((List.range (G.K x + 1)).map fun j => G.dmsg x ν r ((List.ofFn c).take j))

theorem TCoinModel.shvzk {W : Type} {G : Proto W} {x : Msg} (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) {n : ℕ} (E : List Msg → Prop) {Ω : Type} [Fintype Ω] [Nonempty Ω] (sim : Ω → List Msg) :
    ∀ (M : G.TCoinModel x) (ev : Ω → M.SimAt) (kS : ℕ), (∀ p, Nat.card {s // ev s = p} = kS) →
      (∀ s, sim s = ((M.simAt (ev s) ν c).eval H).1) → M.Link H ν c w → M.OK → M.size ≤ n →
      |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : Ω => E (sim s))| ≤ 2 * n * δ₁.toReal
  | .inl M, ev, _, hev, hsim, hL, hOK, hn => by
      have : sim = fun s => M.simOut (ev s) := funext hsim
      subst this
      exact M.shvzk H hT1 hδ₁ ν c w hL hOK ev hev hn E
  | .inr M, ev, _, hev, hsim, hL, hOK, hn => by
      have hD : ∀ s, sim s = G.dVec H x ν (ev s) c := fun s => by
        rw [hsim s]
        simp only [TCoinModel.simAt, ocSeq_eval, List.map_map]
        rfl
      have e : Pr (fun s : Ω => E (sim s)) = Pr (fun r : G.Rand => E (G.dVec H x ν r c)) := by
        simp only [hD]
        exact Pr_comp_fiber ev hev fun r => E (G.dVec H x ν r c)
      rw [e]
      have hn' : (M.size : ℝ) ≤ n := by exact_mod_cast hn
      calc _ ≤ 2 * M.size * δ₁.toReal := M.close H hT1 hδ₁ ν c w hL hOK E
        _ ≤ 2 * n * δ₁.toReal := by gcongr

theorem eval_fiber_card {ι : Type} {β : ι → Type} (i : ι) (b b' : β i) :
    Nat.card {f : ∀ k, β k // f i = b} = Nat.card {f : ∀ k, β k // f i = b'} := by
  classical
  exact Nat.card_congr
    { toFun := fun f => ⟨Function.update f.1 i b', Function.update_self ..⟩
      invFun := fun f => ⟨Function.update f.1 i b, Function.update_self ..⟩
      left_inv := fun ⟨f, hf⟩ => Subtype.ext (by subst hf; simp)
      right_inv := fun ⟨f, hf⟩ => Subtype.ext (by subst hf; simp) }

open Classical in
/-- **`G` with the SHVZK simulator its models give**: one tape per nonce and coin vector, and at `(ν, c)` the model's
messages from its tape there (`TCoinModel.simAt`). Its prover is `G`'s. -/
noncomputable def withSim {W : Type} (G : Proto W) [Fintype G.Coin]
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : Proto W :=
  { G with
    SimRand := fun x => ∀ q : G.Nonce × (Fin (G.K x) → G.Coin), (Mc x q.1 q.2).SimAt
    finSim := fun _ => inferInstance
    neSim := fun _ => inferInstance
    sim := fun x ν c s => (Mc x ν c).simAt (s (ν, c)) ν c }

theorem withSim_prover {W : Type} (G : Proto W) [Fintype G.Coin]
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : (G.withSim Mc).prover = G.prover := rfl

theorem withSim_shvzk {W : Type} (G : Proto W) [Fintype G.Coin] (H : Msg → Msg) (R : Rel W) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nB : Msg → ℕ)
    (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x) (hL : ∀ x w, R x w → ∀ ν c, (Mc x ν c).Link H ν c w)
    (hOK : ∀ x ν c, (Mc x ν c).OK) : (G.withSim Mc).SHVZK H R fun x => 2 * nB x * δ₁.toReal := by
  intro x w hR ν c E
  obtain ⟨p₀⟩ := TCoinModel.instNonemptySimAt (Mc x ν c)
  exact TCoinModel.shvzk (G := G) H hT1 hδ₁ ν c w E (Ω := (G.withSim Mc).SimRand x)
    (fun s => (((G.withSim Mc).sim x ν c s).eval H).1) (Mc x ν c) (fun s => s (ν, c)) _
    (fun p => eval_fiber_card (β := fun q : G.Nonce × (Fin (G.K x) → G.Coin) => (Mc x q.1 q.2).SimAt) (ν, c) p p₀)
    (fun _ => rfl) (hL x w hR ν c) (hOK x ν c) (hnB x ν c)

def TMsgLink {W : Type} (G : Proto W) (H : Msg → Msg) (R : Rel W) (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : Prop :=
  ∀ x w, R x w → ∀ ν, (Mp x ν).Link H ν w ∧ ∀ c, (Mc x ν c).Link H ν c w

def TablesOK {W : Type} (G : Proto W) (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : Prop :=
  (∀ x ν, (Mp x ν).OK) ∧ ∀ x ν c, (Mc x ν c).OK

end Proto

theorem Flock.zkAt_of_tables_bind {Coin Col T : Type} (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T)
    (G : Proto Flock.Wit) [Fintype G.Coin] (t : ℕ) (ht : 0 < t) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.TPreModel x) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x)
    (nL nB : Msg → ℕ) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Flock.R D.parse) Mp Mc) (hOK : G.TablesOK Mp Mc) (hC : G.Binding H) :
    Claims.Flock.ZKAt H D G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  have hA : ∀ x w, Flock.R D.parse x w → ∀ (ν : G.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤
        2 * nL x * δ₁.toReal := by
    intro x w hR ν V E
    calc _ ≤ 2 * (Mp x ν).size * δ₁.toReal :=
          (Mp x ν).runs_close H hT1 hδ₁ ν w (hL x w hR ν).1 (hOK.1 x ν) V E
      _ ≤ 2 * nL x * δ₁.toReal := by
          have : ((Mp x ν).size : ℝ) ≤ nL x := by exact_mod_cast hnL x ν
          gcongr
  rw [← G.withSim_prover Mc]
  exact Flock.zkR_gkE_hm96_bind H D (G.withSim Mc) t ht hzk hk δ₁ nL nB hA
    (G.withSim_shvzk H _ hT1 hδ₁ Mc nB hnB (fun x w hR ν c => (hL x w hR ν).2 c) hOK.2) hC

theorem Flock.zk_deployZk_of_tables_bind (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) [Fintype G.Coin]
    (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Flock.R (FlockDeploy.deployZk rel).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding H) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zkAt_of_tables_bind H _ G t ht rfl (fun x => (deployZk_proto_k rel G hK x).symm) hT1 hδ₁ Mp Mc nL nB hnL hnB
    hL hOK hC

theorem Flock.zk_deployZk_of_tables (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) [Fintype G.Coin]
    (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Flock.R (FlockDeploy.deployZk rel).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.CoinCheck) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_of_tables_bind H rel G t ht hK hT1 hδ₁ Mp Mc nL nB hnL hnB hL hOK (G.binding_of_coinCheck H hC)

theorem Flock.zk_deployZk_of_tables_gap (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) [Fintype G.Coin]
    (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Flock.R (FlockDeploy.deployZk rel).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.CoinCheck) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_of_tables H rel G t ht hK (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)
    ENNReal.ofReal_ne_top Mp Mc nL nB hnL hnB hL hOK hC

end Claims.ZKMal
