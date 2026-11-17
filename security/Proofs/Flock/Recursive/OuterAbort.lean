module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.View
public import Proofs.Flock.Soundness.Discharge.ZkEveryCoin.Defs
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
import Proofs.Flock.Soundness.Discharge.Composed.Headline
import Proofs.Flock.Soundness.Discharge.Composed.Setup
import Proofs.Flock.Soundness.Discharge.ZkEveryCoin.Headline
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.Refusal
import Proofs.Flock.Soundness.ZK.Session

/-! V*'s outer sessions with the honest prover's self-check and the coins it refuses, at the same zero-knowledge bound. -/

@[expose] public section

namespace FlockSoundness.ZK.Table

section Stops

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- **The honest prover's self-check stops** (`zk_veil.rs`'s `SelfCheck::Stop`): at some rep `r`, its pads miss the
inner proof's batched constraint, `⟨λ, h⟩ ≠ θ`, `λ` and `θ` the verifier's replay on the messages before it. -/
def Stops (w : W) (ω : T.Rand) : Prop :=
  ∃ r, ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i ≠ T.theta r (T.view w ω).pre

theorem not_stops {w : W} (h : T.InnerHolds w) (ω : T.Rand) : ¬ T.Stops w ω := fun ⟨r, hr⟩ => hr (h ω r)

def StopsAt (w : W) (ω : T.Rand) (r : Fin 2) : Prop :=
  ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i ≠ T.theta r (T.view w ω).pre

end Stops

section Leaves

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt (B × Salt) D Hi VL) [CharP F 2]

theorem viewWith_pre (w : W) (ω : T.Rand) (L : T.Hid → D) : (T.viewWith w ω L).pre = (T.view w ω).pre := rfl

end Leaves

end FlockSoundness.ZK.Table

namespace FlockSoundness.Discharge.ZkView

variable {K W Hi VL : Type} [Field K] {z : Sizes}

theorem Coined.viewExe_pre (T : Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (ys : T.tab.Hid → Salt) :
    (T.viewExe w ω ys).pre = (T.tab.view w ω).pre := by
  unfold Coined.viewExe
  exact FlockSoundness.ZK.Table.viewWith_pre _ _ _ _

end FlockSoundness.Discharge.ZkView

namespace FlockSoundness.Discharge.ZkEveryCoin

open Game Audit Exec Refine Flock FlockLevel3 Model ZkSession ZkLink ZkExec DecodeZL Composed ZkView
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- instance search over the ZK view's and coins' product types

open Classical in
theorem coin_close_at (C : CoinedSetup) (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (c : CoinsZK C.Rest) (w w0 : Witness C.S.m) (hv : C.Valid w)
    (hreg : (C.T c).tab.sReg w = (C.T c).tab.pubReg) (E : (C.T c).tab.View ⊕ (C.T c).tab.Cut → Prop) :
    prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E (outExe C c w p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E (simAll C c w0 p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ)
      ∧ prCoin (fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) => E (simAll C c w0 p.1 p.2))
        ≤ prCoin (fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E (outExe C c w p.1 p.2))
          + 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
  by_cases hA : C.Admits c
  · simp only [outExe, simAll, hA, ite_true]
    obtain ⟨h1, h2⟩ := zkViewAtReg_of_ok C hC hKey c hA
      (fun v => (Sum.inl (v 0) : (C.T c).tab.View ⊕ (C.T c).tab.Cut)) (fun _ => w) (fun _ => w0) (fun _ => hv)
      (fun _ => hreg) E
    have eR := ZK.prCoin_congr_equiv (Equiv.funUnique (Fin 1) ((C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt)))
      (fun p => E (.inl ((C.T c).viewExe w p.1 p.2)))
    have eS := ZK.prCoin_congr_equiv (Equiv.funUnique (Fin 1) ((C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt)))
      (fun p => E (.inl ((C.T c).simExe w0 p.1 p.2)))
    have hs : ∑ _j : Fin 1, 2 * (Fintype.card (C.T c).tab.Hid : ℝ≥0∞) * (2 : ℝ≥0∞) ^ (-193 : ℤ)
        ≤ 2 * Fintype.card (LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
      rw [Fin.sum_univ_one]
      gcongr
      exact_mod_cast card_hid_le C c
    exact ⟨(eR.symm.trans_le h1).trans (add_le_add (le_of_eq eS) hs),
      (eS.symm.trans_le h2).trans (add_le_add (le_of_eq eR) hs)⟩
  · simp only [outExe, simAll, hA, ite_false, ZkView.Coined.viewExe_eq]
    have hT := leafT_hm96 C c
    have hS : (C.T c).tab.Admits (sentAt C c) :=
      ZK.Session.sent_admits (fun _ : Fin 1 => (C.T c).tab) (fun _ => (C.T c).rep) 0
    have h1 : Claims.Hm96Hiding (C.T c).hm96.My (C.T c).hm96.cs ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
      rw [ZkView.Coined.hm96At_My (C.T c) ZkView.My rfl, ZkView.Coined.hm96At_cs (C.T c) ZkView.My rfl]
      exact hKey
    have h := ZK.Table.table_abort_hm96 (C.T c).tab (C.T c).hm96 _ hT h1 (sentAt C c) hS w hreg w0
      (fun v => E (.inr v))
    rw [card_pre] at h
    exact h

end FlockSoundness.Discharge.ZkEveryCoin

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxHeartbeats 1000000 -- instance search over the `J` tables' product randomness
set_option maxHeartbeats 4000000 -- the `J`-table session lemmas below; which of them needs it is unmeasured

open Classical in
noncomputable def sendsJ {J : ℕ} (C : Fin J → CoinedSetup) (c : ∀ j, CoinsZK (C j).Rest) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View) → Out) (w : ∀ j, Witness (C j).S.m)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) : Out :=
  if ∃ j, ((C j).T (c j)).tab.Stops (w j) (p j).1 then stopped p
  else enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2

theorem stopsJ_never {J : ℕ} {C : Fin J → CoinedSetup} (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    {c : ∀ j, CoinsZK (C j).Rest} (hc : ∀ j, (C j).Admits (c j)) {w : ∀ j, Witness (C j).S.m}
    (hv : ∀ j, (C j).Valid (w j)) (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    ¬ ∃ j, ((C j).T (c j)).tab.Stops (w j) (p j).1 := by
  rintro ⟨j, hj⟩
  obtain ⟨-, -, -, hT⟩ := hC j
  obtain ⟨-, -, -, -, -, -, -, hinner⟩ := hT (c j) (hc j)
  exact ((C j).T (c j)).tab.not_stops (hinner (w j) (hv j) (hreg j)) (p j).1 hj

theorem prCoin_stopsJ {J : ℕ} {C : Fin J → CoinedSetup} (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    {c : ∀ j, CoinsZK (C j).Rest} (hc : ∀ j, (C j).Admits (c j)) {w : ∀ j, Witness (C j).S.m}
    (hv : ∀ j, (C j).Valid (w j)) (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg) :
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
      ∃ j, ((C j).T (c j)).tab.Stops (w j) (p j).1) = 0 :=
  le_zero_iff.1 ((Game.prCoin_mono (q := fun _ => False) fun p hp => stopsJ_never hC hc hv hreg p hp).trans
    Game.prCoin_false.le)

theorem sendsJ_honest {J : ℕ} {C : Fin J → CoinedSetup} (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    {c : ∀ j, CoinsZK (C j).Rest} (hc : ∀ j, (C j).Admits (c j)) {w : ∀ j, Witness (C j).S.m}
    (hv : ∀ j, (C j).Valid (w j)) (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
    {Out : Type} (enc : (∀ j, ((C j).T (c j)).tab.View) → Out)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    sendsJ C c enc w stopped p = enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2 := by
  have hn := stopsJ_never hC hc hv hreg p
  unfold sendsJ
  split <;> first | contradiction | rfl

def ZkViewAtJS {J : ℕ} (C : Fin J → CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : ∀ j, CoinsZK (C j).Rest), (∀ j, (C j).Admits (c j)) → ∀ {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View) → Out) (w w0 : ∀ j, Witness (C j).S.m), (∀ j, (C j).Valid (w j)) →
    ∀ (_hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
      (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out) (E : Out → Prop),
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (sendsJ C c enc w stopped p))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (sendsJ C c enc w stopped p))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁

theorem zkViewAtJS_of {J : ℕ} {C : Fin J → CoinedSetup} {δ₁ : ℝ≥0∞}
    (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j)) (h : ZkViewAtJ C δ₁) : ZkViewAtJS C δ₁ := by
  intro c hc Out enc w w0 hv hreg stopped E
  have hs : ∀ p, sendsJ C c enc w stopped p = enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2 :=
    sendsJ_honest hC hc hv hreg enc stopped
  simp only [hs]
  exact h c hc enc w w0 hv E

open Classical in
/-- The session stops at its first refused coin and sends no proof after, so what it sent is a function of the tables'
cuts up to there; a rep's self-check runs only after its outer coins open, where a `1` stops the session first. -/
noncomputable def sendsAllJ {J : ℕ} (C : Fin J → CoinedSetup) (c : ∀ j, CoinsZK (C j).Rest) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut) → Out) (w : ∀ j, Witness (C j).S.m)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) : Out :=
  if ∃ j r, (∀ x ∈ (C j).rOuter (c j) r, x ≠ 1) ∧ ((C j).T (c j)).tab.StopsAt (w j) (p j).1 r then stopped p
  else enc fun j => ZkEveryCoin.outExe (C j) (c j) (w j) (p j).1 (p j).2

def SelfCheckAtReps (C : CoinedSetup) : Prop :=
  ∀ (c : CoinsZK C.Rest) (r : Fin 2), (∀ x ∈ C.rOuter c r, x ≠ 1) → ∀ w, C.Valid w →
    (C.T c).tab.sReg w = (C.T c).tab.pubReg → ∀ ω, ¬ (C.T c).tab.StopsAt w ω r

theorem stopsAllJ_never {J : ℕ} {C : Fin J → CoinedSetup} (hR : ∀ j, SelfCheckAtReps (C j))
    {c : ∀ j, CoinsZK (C j).Rest} {w : ∀ j, Witness (C j).S.m} (hv : ∀ j, (C j).Valid (w j))
    (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    ¬ ∃ j r, (∀ x ∈ (C j).rOuter (c j) r, x ≠ 1) ∧ ((C j).T (c j)).tab.StopsAt (w j) (p j).1 r := by
  rintro ⟨j, r, hr, hj⟩
  exact hR j (c j) r hr (w j) (hv j) (hreg j) (p j).1 hj

theorem sendsAllJ_honest {J : ℕ} {C : Fin J → CoinedSetup} (hR : ∀ j, SelfCheckAtReps (C j))
    {c : ∀ j, CoinsZK (C j).Rest} {w : ∀ j, Witness (C j).S.m} (hv : ∀ j, (C j).Valid (w j))
    (hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut) → Out)
    (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out)
    (p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :
    sendsAllJ C c enc w stopped p = enc fun j => ZkEveryCoin.outExe (C j) (c j) (w j) (p j).1 (p j).2 := by
  have hn := stopsAllJ_never hR hv hreg p
  unfold sendsAllJ
  split <;> first | contradiction | rfl

/-- The refusal reads only the coins, so the simulator refuses exactly where the prover does. -/
def ZkViewAllJ {J : ℕ} (C : Fin J → CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : ∀ j, CoinsZK (C j).Rest) {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut) → Out) (w w0 : ∀ j, Witness (C j).S.m),
    (∀ j, (C j).Valid (w j)) → ∀ (_hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg)
      (stopped : (∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) → Out) (E : Out → Prop),
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (sendsAllJ C c enc w stopped p))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ZkEveryCoin.simAll (C j) (c j) (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card (ZkView.LeafPos (C j).z) * δ₁
      ∧ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E (enc fun j => ZkEveryCoin.simAll (C j) (c j) (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (sendsAllJ C c enc w stopped p))
          + ∑ j, 2 * Fintype.card (ZkView.LeafPos (C j).z) * δ₁

set_option maxRecDepth 8000 in
theorem zkViewAllJ_of_ok {J : ℕ} (C : Fin J → CoinedSetup) (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    (hR : ∀ j, SelfCheckAtReps (C j)) (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ZkViewAllJ C ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  intro c Out enc w w0 hv hreg stopped E
  have hs : ∀ p, sendsAllJ C c enc w stopped p = enc fun j => ZkEveryCoin.outExe (C j) (c j) (w j) (p j).1 (p j).2 :=
    sendsAllJ_honest hR hv hreg enc stopped
  simp only [hs]
  have : Nonempty ZkView.Salt := ⟨0⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((rand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((simRand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  have one := fun j => ZkEveryCoin.coin_close_at (C j) (hC j) hKey (c j) (w j) (w0 j) (hv j) (hreg j)
  exact ⟨FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut)
      (fun j p => ZkEveryCoin.outExe (C j) (c j) (w j) p.1 p.2)
      (fun j p => ZkEveryCoin.simAll (C j) (c j) (w0 j) p.1 p.2)
      (fun j => 2 * Fintype.card (ZkView.LeafPos (C j).z) * (2 : ℝ≥0∞) ^ (-193 : ℤ)) (fun j E' => (one j E').1)
      fun v => E (enc v),
    FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View ⊕ ((C j).T (c j)).tab.Cut)
      (fun j p => ZkEveryCoin.simAll (C j) (c j) (w0 j) p.1 p.2)
      (fun j p => ZkEveryCoin.outExe (C j) (c j) (w j) p.1 p.2)
      (fun j => 2 * Fintype.card (ZkView.LeafPos (C j).z) * (2 : ℝ≥0∞) ^ (-193 : ℤ)) (fun j E' => (one j E').2)
      fun v => E (enc v)⟩

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

theorem zk_session_viewHJS (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : SampleSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), SampleSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJS (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  fun S R ys => zkViewAtJS_of (fun j => (coinedOf_setup_of P _ (ys j).x.st (ys j).x.wf.1
    (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
    (maskSetup_of_setupHidden ((tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) (ys j).x.h)
    (hdm S j (ys j))).2.2.2.2.2) (zk_session_viewHJ P I dj y₀ ht hdm hKey S R ys)

/-- This is honest-verifier zero knowledge at each fixed coin vector; a verifier that chooses its coins is covered by the
coin commitment at `Hello` under `keyed-cr`, which no statement here models. -/
theorem zk_session_view_allHJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : SampleSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), SampleSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAllJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  fun S R ys => zkViewAllJ_of_ok _ (fun j => (coinedOf_setup_of P _ (ys j).x.st (ys j).x.wf.1
    (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
    (maskSetup_of_setupHidden ((tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) (ys j).x.h)
    (hdm S j (ys j))).2.2.2.2.2)
    (fun j c r hr w hv hreg ω hs => hs (innerOf_holds_rep_of P _ (ys j).x.st (ys j).x.wf.1
      (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
      (maskSetup_of_setupHidden ((tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) (ys j).x.h)
      (hdm S j (ys j)) c r hr w hv hreg ω)) hKey

end FlockSoundness.Discharge.ZkHidden
