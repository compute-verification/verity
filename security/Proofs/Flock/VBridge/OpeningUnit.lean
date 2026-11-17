module

public import Proofs.Flock.VBridge.ResUnit
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
import Proofs.Flock.VBridge.PartS

/-! The verifier's `rec-opening/…` unit is the net it builds for a part of the opening's algebra, a `rec-residuals` part's, so `ResUnit`'s theorems read it unchanged (`decode_opening`). -/

@[expose] public section

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

theorem check_opening {c : Flock.Circuit} (h : RecOpening.check c = .ok ()) {m cl j : ℕ}
    (hp : RecOpening.params c.unit.name = some (m, cl, j)) {sh : Flock.Shape}
    (hsh : RecOpening.shape? m cl = some sh) :
    ∃ S ns nv, RecOpening.part sh j = .ok (S, ns, nv) ∧
      netOf (RecOpening.unitName m cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit := by
  unfold RecOpening.check at h
  rw [hp] at h
  dsimp only at h
  rw [hsh] at h
  dsimp only at h
  split at h
  · cases h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i S ns nv hpart
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i g hg
          split at h
          · rename_i hcg
            exact ⟨S, ns, nv, hpart, by rw [hcg]; exact hg⟩
          · cases h

theorem upv_of_check {c : Flock.Circuit} (h : RecOpening.check c = .ok ()) {m cl j : ℕ}
    (hp : RecOpening.params c.unit.name = some (m, cl, j)) : c.upv = 1 := by
  unfold RecOpening.check at h
  rw [hp] at h
  dsimp only at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hv
      simpa using hv

theorem opening_small {sh : Flock.Shape} {j : ℕ} {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h : RecOpening.part sh j = .ok (S, ns, nv)) : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS := by
  unfold RecOpening.part at h
  obtain ⟨S0, -, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs, hg, h⟩ := FlockVBridge.PartS.bind_ok h
    split at h
    · rename_i R P V hj
      obtain ⟨S', hS', h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -, -⟩ := h
      obtain ⟨hn, -⟩ := FlockVBridge.PartS.res_partS hS' (w := fun _ => 0) (v := fun _ => 0) (w' := fun _ => 0)
        (v' := fun _ => 0) (fun _ _ => rfl) (fun _ _ => rfl)
      rw [hn]
      exact groups_small hg _ (Array.mem_of_getElem? hj)
    · cases h

section Decode

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

theorem decode_opening {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecOpening.check c = .ok ()) {m cl p : ℕ} (hp : RecOpening.params c.unit.name = some (m, cl, p))
    {sh : Flock.Shape} (hsh : RecOpening.shape? m cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecOpening.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (w vv : ℕ → Flock.F128) (wI V : Array ℕ) (hwI : wI.size ≤ ns.sum) (hV : V.size ≤ nv)
    (hw : ∀ q < 128 * ns.sum, RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub wI w) q))
    (hv : ∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub V vv) q)) :
    ∃ w' v' : ℕ → Flock.F128, (∀ k (hk : k < wI.size), w' k = w wI[k]) ∧
      (∀ k (hk : k < V.size), v' k = vv V[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_opening hc hp hsh
  rw [hpart] at hpart'
  cases hpart'
  exact ⟨sub wI w, sub V vv, sub_lt wI w, sub_lt V vv,
    decode_resUnit zo hA u hj hone S ns nv (opening_small hpart) hn hu _ _ (sub_ge w hwI) (sub_ge vv hV) hw hv⟩

end Decode

end FlockVBridge.OfVStar
