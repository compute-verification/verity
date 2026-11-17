module

public import Proofs.Flock.Claims.ZK.Masked
public import Proofs.Flock.Claims.Complete.ZkTables.MaskFree

/-! The deployed masked session's mask tape. -/

@[expose] public section

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Masked

/-- **The word of 128 bits**: `Σ_v b_v·x^v` (`FlockLevel3.pack`) as the executable's value. -/
noncomputable def wordOfBits (b : Fin 128 → ZMod 2) : _root_.Flock.F128 := (FlockLevel3.pack b).val

theorem toZ_wordOfBits (b : Fin 128 → ZMod 2) (v : Fin 128) :
    _root_.Flock.Readable.toZ ((wordOfBits b).bit v.val) = b v := by
  have h := FlockSoundness.Refine.unpack_fOf (wordOfBits b) v
  rw [show FlockSoundness.Refine.fOf (wordOfBits b) = FlockLevel3.pack b from rfl, FlockLevel3.unpack_pack] at h
  rw [h]
  rfl

noncomputable def maskOf {C : Tabs} (x₀ : Msg) (r : Rand C x₀) (c k : ℕ) : _root_.Flock.F128 :=
  if hc : c < jOf x₀ then
    if hk : k < (C x₀ ⟨c, hc⟩).z.nMask then wordOfBits ((r.1 ⟨c, hc⟩).1 ⟨k, hk⟩) else 0
  else 0

noncomputable def TapePlan.atMask {C : Tabs} (x₀ : Msg) (z : Rand C x₀ → FlockComplete.TapeZk) : TapePlan C :=
  TapePlan.at x₀ fun r => { z r with mask := maskOf x₀ r }

theorem zOf_maskOf (P : Composed.LigProver) (x₀ : Msg) (r : Rand (tabC P) x₀) (w : Claims.Flock.Wit)
    (j : Fin (jOf x₀)) :
    _root_.Flock.Readable.zOf (stsOf x₀).size (FlockComplete.ZkTables.maskWOf x₀ w (maskOf x₀ r)) j.val
        (tabC P x₀ j).S.m =
      Composed.zWith (stAt x₀ j) (wtOf P x₀ w j) (r.1 j).1 := by
  have hj := lt_of_jOf j
  obtain ⟨s, spec, hs⟩ := setupOf_of_pos (x := x₀) (by omega)
  have e : FlockComplete.ZkTables.maskWOf x₀ w (maskOf x₀ r) =
      FlockComplete.ZkTables.maskW (stsOf x₀) w (maskOf x₀ r) := by
    unfold FlockComplete.ZkTables.maskWOf
    rw [hs]
  rw [e]
  refine (FlockComplete.ZkTables.zOf_maskW (stsOf x₀) w (maskOf x₀ r) hj).trans ?_
  have hw : wtOf P x₀ w j = _root_.Flock.Readable.zOf (stsOf x₀).size w j.val (tabC P x₀ j).S.m :=
    _root_.Flock.Readable.zOf_tableWit hj w _
  rw [hw]
  congr 1
  funext k v
  have hk : k.val < (tabC P x₀ j).z.nMask := k.isLt
  simp only [maskOf, j.isLt, hk, ↓reduceDIte]
  exact toZ_wordOfBits _ v

theorem honestPR_atMask {C : Tabs} (w : Claims.Flock.Wit) (x x₀ : Msg) (z : Rand C x₀ → FlockComplete.TapeZk)
    (r : Rand C x₀) (hist : List (Msg × Msg))
    (h : FlockComplete.ZkTables.outerHit x (hist.map Prod.snd) = false) :
    (FlockComplete.ZkTables.honestPR w).next x hist ((TapePlan.atMask x₀ z).tapeZk r) =
      (FlockComplete.ZkTables.honestP (FlockComplete.ZkTables.maskWOf x w (maskOf x₀ r))).next x hist
        ((TapePlan.atMask x₀ z).tapeZk r) :=
  FlockComplete.ZkTables.honestPR_next h

end Masked

open Masked

open Classical in
theorem Flock.zk_deployZk_masked_atMask_gap (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (Q : RoundPlan (tabC P)) (x₀ : Msg) (z : Rand (tabC P) x₀ → FlockComplete.TapeZk) (t : ℕ) (ht : 0 < t)
    (hc : Q.Causal)
    (hpre : (session nr (tableData P Q (TapePlan.atMask x₀ z))).HonestPreOn (fun x => x = x₀ ∧ Live x) RZk H
      coinOfD fun ω => (TapePlan.atMask x₀ z).tapeZk ω.2)
    (hfin : (session nr (tableData P Q (TapePlan.atMask x₀ z))).HonestFinOn (fun x => x = x₀ ∧ Live x) RZk H
      coinOfD fun ω => (TapePlan.atMask x₀ z).tapeZk ω.2)
    (hoff : (session nr (tableData P Q (TapePlan.atMask x₀ z))).HonestOffOn (fun x => x = x₀ ∧ Live x) RZk H
      coinOfD fun ω => (TapePlan.atMask x₀ z).tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr (tableData P Q (TapePlan.atMask x₀ z)) t)
      (fun x => if x = x₀ ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_at_gap H nr P Q x₀ _ t ht hc hpre hfin hoff

open Classical in
theorem Flock.zk_deployZk_masked_atMask_stop_gap (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg)
    (fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg)
    (x₀ : Msg) (z : Rand (tabC P) x₀ → FlockComplete.TapeZk) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal)
    (hpre : (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x₀ z))).HonestPreOn
      (fun x => x = x₀ ∧ Live x) RZk H coinOfD fun ω => (TapePlan.atMask x₀ z).tapeZk ω.2)
    (hfin : (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x₀ z))).HonestFinOn
      (fun x => x = x₀ ∧ Live x) RZk H coinOfD fun ω => (TapePlan.atMask x₀ z).tapeZk ω.2)
    (hoff : ∀ x, x = x₀ ∧ Live x → ∀ w, RZk x w → ∀ r (ops : List (Msg × Msg)), ops.length = refK x →
      ¬ (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x₀ z))).Admits x
        ((session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x₀ z))).pad
          (ops.map fun p => rd p.2)) →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.atMask x₀ z).tapeZk r)).eval H).1 = []) :
    Claims.Flock.ZK H
      (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P (RoundPlan.stop cz enc fin) (TapePlan.atMask x₀ z)) t)
      (fun x => if x = x₀ ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_atMask_gap H nr P (RoundPlan.stop cz enc fin) x₀ z t ht hc hpre hfin
    ((honestOffOn_stop_iff (TapePlan.atMask x₀ z) nr _ RZk H).2 hoff)

end Claims.ZKMal
