import Proofs.Flock.VBridge.AlgHolds
import Proofs.Flock.VBridge.OpeningUnit

/-!
# n4's session (a): the opening's parts' zero columns are its residuals'

The verifier cuts R's opening's S (`RecOpening.structureOf`) into parts as it cuts V*'s algebra (`RecOpening.part`:
`RecResiduals.groups`, then `partS`), and pins each part's `rec-opening/…` unit (`RecOpening.check`). As for
`rec-residuals` (`AlgHolds`):

* `part_eqOpening`: the verifier's opening part `p` is `partS` at the `p`-th group, with row ports of the group's sizes
  and a `v` of its `V`.
* `OpeningZeroAt`: one copy of one unit is opening part `p`, its input columns registered bits of `w` and `v` at the
  part's indices and its outputs registered zero. `partCols_of_openingZeroAt`: its outputs hold the part's residuals
  (`decode_opening`), and they are zero.
* `residuals_zero_of_cols`: when every part's outputs hold its residuals and are zero, every residual of S is
  (`residuals_zero_of_parts`).
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open Flock.RecResiduals (groups partS)
open Flock.RecAlgebra (el)
open FlockVBridge.PartS (wIdx)

/-- **The verifier's opening part `p` is its `p`-th group's** (`part_eq` at `RecOpening.part`). -/
theorem part_eqOpening {sh : Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecOpening.structureOf sh = .ok S) (hg : groups S (RecOpening.portSizes sh) = .ok gs) {p : ℕ}
    (hp : p < gs.size) {S' : RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (h : RecOpening.part sh p = .ok (S', ns, nv)) :
    partS S (RecOpening.portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S' ∧
      ns = gs[p].2.1.toList.map ((RecOpening.portSizes sh)[·]!) ∧ nv = gs[p].2.2.size := by
  unfold RecOpening.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hS] at hS0
  cases hS0
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs0, hg0, h⟩ := FlockVBridge.PartS.bind_ok h
    rw [hg] at hg0
    cases hg0
    split at h
    · rename_i R P V hj
      obtain ⟨S'', hS'', h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      rw [Array.getElem?_eq_getElem hp, Option.some.injEq] at hj
      rw [hj]
      exact ⟨hS'', rfl, rfl⟩
    · cases h

/-- **Every residual of S is zero when every part's outputs hold its residuals and are zero** (`residuals_zero_of_parts`,
as `algebraHolds_of_parts` reads it): at S's parts, each part's columns `col` from `base` hold its residuals
(`PartCols`) at ports reading `w`, `v` where the part's index says, and those columns are zero. -/
theorem residuals_zero_of_cols {S : RecAlgebra.GfS} (hS : S.BitsOk) {sizes : Array ℕ}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S sizes = .ok gs) {w v : Array F128}
    (hw : w.size = S.nw) (hv : v.size = S.nv)
    (hparts : ∀ j (hj : j < gs.size), ∃ S', partS S sizes gs[j].1 gs[j].2.1 gs[j].2.2 = .ok S' ∧
      ∃ (w' v' : ℕ → F128) (col : ℕ → Bool) (base : ℕ),
        (∀ k (hk : k < (wIdx sizes gs[j].2.1).size), w' k = el w (wIdx sizes gs[j].2.1)[k]) ∧
        (∀ k (hk : k < gs[j].2.2.size), v' k = el v gs[j].2.2[k]) ∧
        PartCols S' w' v' col base ∧ ∀ t < 128 * S'.res.length, col (base + t) = false) :
    ∃ rs, RecAlgebra.residuals S w v = .ok rs ∧ ∀ x ∈ rs, x = 0 :=
  FlockVBridge.Parts.residuals_zero_of_parts hS hg hw hv fun j hj => by
    obtain ⟨S', hS', w', v', col, base, hw', hv', ⟨hr, -⟩, h0⟩ := hparts j hj
    refine ⟨S', hS', fun k hk => ?_⟩
    have hk0 : resVal (gfsOf S') (termVal w' v') (resOf S'.res[k]) = 0 :=
      AccChain.eq_of_bitOf fun t ht => by rw [← hr k hk t ht, h0 (128 * k + t) (by omega), bitOf_zero]
    obtain ⟨_, e1⟩ := FlockVBridge.PartS.res_partS hS' (w := el w) (v := el v) hw' hv'
    obtain ⟨_, e2⟩ := FlockVBridge.PartS.res_partS hS' (w := el w) (v := el v)
      (w' := fun k => el w ((wIdx sizes gs[j].2.1).getD k 0)) (v' := fun k => el v (gs[j].2.2.getD k 0))
      (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl)
      (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl)
    exact (e2 k hk).trans ((e1 k hk).symm.trans hk0)

section Copy

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode
open FlockVBridge.AccChain (RegOut copyVal_regOut)

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

/-- **Copy `j` of unit `u` is opening part `p` of the groups `gs`, its outputs registered zero** (`ResZeroAt` at
`RecOpening`): the unit `RecOpening.check` accepted for part `p` at `sh`, its row ports registered bits of `w` at
`wIdx` of group `p`'s ports, its `v` port registered bits of `v` at the group's `V`, and every output column of the part
(`16 · RES_OUT_WORDS` bits) a registered 0 (`RegOut`). -/
def OpeningZeroAt (bits : Fin (Layout.keyN c hU nU) → Bool) (u : Fin nU) (j : ℕ) (sh : Shape)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (p : ℕ) (w v : ℕ → F128) : Prop :=
  ∃ (hpg : p < gs.size) (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true ∧
    RecOpening.check c = .ok () ∧ c.unit.unitLog ≤ 32 ∧
    ∃ m cl, RecOpening.params c.unit.name = some (m, cl, p) ∧ RecOpening.shape? m cl = some sh ∧
    ∃ S ns nv, RecOpening.part sh p = .ok (S, ns, nv) ∧
      (∀ q < 128 * ns.sum,
        RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (RecOpening.portSizes sh) gs[p].2.1) w) q)) ∧
      (∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 v) q)) ∧
      ∀ t < 16 * RecResiduals.RES_OUT_WORDS, RegOut zo u hj (partBase S ns nv + t) false

/-- **An opening part's residuals, zero, from its unit's copy in a good layer** (`decode_opening`, and
`copyVal_regOut` at the output columns). -/
theorem partCols_of_openingZeroAt {bits : Fin (Layout.keyN c hU nU) → Bool} (hG : zo.Good bits) {u : Fin nU}
    {j : ℕ} {sh : Shape} {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecOpening.structureOf sh = .ok S0) (hg : groups S0 (RecOpening.portSizes sh) = .ok gs) {p : ℕ}
    {w v : ℕ → F128} (h : OpeningZeroAt zo bits u j sh gs p w v) :
    ∃ hpg : p < gs.size, ∃ S', partS S0 (RecOpening.portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S' ∧
      ∃ (w' v' : ℕ → F128) (col : ℕ → Bool) (base : ℕ),
        (∀ k (hk : k < (wIdx (RecOpening.portSizes sh) gs[p].2.1).size),
          w' k = w (wIdx (RecOpening.portSizes sh) gs[p].2.1)[k]) ∧
        (∀ k (hk : k < gs[p].2.2.size), v' k = v gs[p].2.2[k]) ∧
        PartCols S' w' v' col base ∧ ∀ t < 128 * S'.res.length, col (base + t) = false := by
  obtain ⟨hpg, hj, hone, hc, hu, m, cl, hp, hsh, S, ns, nv, hpart, hw, hv, hz⟩ := h
  obtain ⟨hS', hns, hnv⟩ := part_eqOpening hS hg hpg hpart
  obtain ⟨w', v', hw', hv', hcols⟩ := decode_opening zo hG.2.1 u hj hone hc hp hsh hpart hu w v _ _
    (Nat.le_of_eq (by rw [wIdx_size, hns])) (Nat.le_of_eq hnv.symm) hw hv
  obtain ⟨S1, ns1, nv1, hpart1, hn⟩ := check_opening hc hp hsh
  rw [hpart] at hpart1
  cases hpart1
  have hbase : (Rows.ofNet hU).nIn ≤ partBase S ns nv := by
    rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
    exact inRows_le_outStart _ _ _
  have hsmall := opening_small hpart
  exact ⟨hpg, S, hS', w', v', _, _, hw', hv', hcols, fun t ht =>
    copyVal_regOut zo hG.2.1 (keyProg_correct hU nU hG.1 u) hj (by omega) (hz t (by omega))⟩

end Copy

end FlockVBridge.OfVStar
