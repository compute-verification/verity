import Proofs.Flock.VBridge.ResUnit

/-!
# The verifier's `rec-opening/…` unit

`RecOpening.check` accepts a `rec-opening/m<m>-c<claims>-p<j>` unit only if it is the net the verifier builds itself
(`RecResiduals.unitLaid S ns nv`) for part `j` of the opening's algebra (`RecOpening.part`), which `RecResiduals`'
grouping and renumbering cut from the opening's S (`RecOpening.structureOf`). The net and its layout are a
`rec-residuals` part's, so `ResUnit`'s theorems read it unchanged.

* `check_opening`: what `check` accepted, the verifier's net for part `j`.
* `opening_small`: that part has at most 16 residuals (`groups_small`), as `laid_part` asks.
* `decode_opening`: in a layer agreeing with the registrant, a copy of the unit whose row ports are registered bits of
  `w` at the part's indices and whose `v` port registered bits of `vv` at `V` holds the part's residuals at part ports
  that read them (`decode_resUnit`).
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-- **What `RecOpening.check` accepted**: a unit named `rec-opening/m<m>-c<claims>-p<j>` at a shape the verifier accepts
is the net the verifier builds for part `j` of the opening's algebra at that shape. -/
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

/-- **The verifier's opening part has at most 16 residuals** (`groups_small`): `partS` gives it one residual for each
of its group's (`PartS.res_partS`). -/
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
  {Lv : Law nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

/-- **An opening part the verifier pinned computes its residuals at the whole ports, in a layer agreeing with the
registrant** (`decode_resUnit` at `check_opening`'s net). Let `RecOpening.check` accept `c`, its unit named for part `p`
at a shape `sh` the verifier accepts, part `p` being `S` with row ports of `ns` elements and a `v` of `nv`. When copy `j`
of unit `u`'s row ports are registered bits of the whole `w` at the part's indices `wI`, and its `v` port registered bits
of `vv` at `V`, its output columns in `bits` hold the part's residuals at part ports `w'`, `v'` that read `w` at `wI` and
`vv` at `V`. -/
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
