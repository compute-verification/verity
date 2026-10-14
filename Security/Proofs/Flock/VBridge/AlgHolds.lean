import Proofs.Flock.VBridge.ResUnit
import Proofs.Flock.VBridge.PartsCover
import Proofs.Flock.VBridge.AccUnits

/-!
# VBridge G items 1–3: `AlgebraHolds` from the parts' units

`AlgebraHolds sh d` asks that `GfResiduals` on the shape's algebra S return residuals that are all zero at the rep's
values. The verifier cuts S into parts (`RecResiduals.groups`) and pins each part's unit (`RecResiduals.check`); V*
registers every part's outputs zero (`rec-residuals-zero`). This file puts those together:

* `part_eq`: the verifier's part `p` is `partS` at the `p`-th group, with row ports of the group's sizes and a `v` of
  its `V`, so that `decode_check`'s `hwI` and `hV` hold at the group's indices (`decode_part`).
* `algebraHolds_of_parts`: when every part's outputs hold its residuals (`PartCols`) at ports reading the rep's `w` and
  the verifier's coefficients where the part's index says, and those outputs are zero, `AlgebraHolds` holds.
* `ResZeroAt`: one copy of one unit is part `p`, its input columns registered bits of the rep's values at the part's
  indices and its outputs registered zero. `algebraHolds_of_decode` and `algebraDecodes_of_decode`: `AlgebraHolds` from
  one such copy per part, in any session, and `AlgebraDecodes` at every accepted outer proof.
* `decode_part_holo`, `ResZeroAtH`, `partCols_of_resZeroAtH`, `algebraHolds_of_decodeH`: the same on a holo statement
  (META `holo: true`), at the holo form of the shape the verifier accepts, whose parts are the hidden units'.
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open Flock.RecResiduals (portSizes groups partS)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.PartS (wIdx)
open FlockVBridge.Compiled (RepData AlgebraHolds)

/-! ## The verifier's part at its group -/

/-- **The verifier's part `p` is its `p`-th group's**: at S (`structureOf`) and its groups, part `p` is `partS` of
group `p`, its row ports the group's ports' sizes and its `v` the group's `V`. -/
theorem part_eq {sh : Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf sh = .ok S) (hg : groups S (portSizes sh) = .ok gs) {p : ℕ} (hp : p < gs.size)
    {S' : RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (h : RecResiduals.part sh p = .ok (S', ns, nv)) :
    partS S (portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S' ∧
      ns = gs[p].2.1.toList.map ((portSizes sh)[·]!) ∧ nv = gs[p].2.2.size := by
  unfold RecResiduals.part at h
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

/-- A part's `w` index has one element per element of its row ports. -/
theorem wIdx_size (sizes P : Array ℕ) : (wIdx sizes P).size = (P.toList.map (sizes[·]!)).sum := by
  unfold wIdx
  rw [← Array.length_toList, Array.toList_flatMap, List.length_flatMap]
  simp

/-- **`decode_check`'s sizes from `part`**: at the `p`-th group's indices, `wIdx` of its ports and its `V`, the part's
row ports and `v` have exactly their elements. -/
theorem part_sizes {sh : Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf sh = .ok S) (hg : groups S (portSizes sh) = .ok gs) {p : ℕ} (hp : p < gs.size)
    {S' : RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (h : RecResiduals.part sh p = .ok (S', ns, nv)) :
    (wIdx (portSizes sh) gs[p].2.1).size = ns.sum ∧ gs[p].2.2.size = nv := by
  obtain ⟨-, hns, hnv⟩ := part_eq hS hg hp h
  exact ⟨by rw [wIdx_size, hns], hnv.symm⟩

section Decode

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

/-- **A part the verifier pinned computes its residuals at its group's indices** (`decode_check`, its `hwI` and `hV`
from `part_sizes`): copy `j`'s row ports registered bits of `w` at `wIdx` of group `p`'s ports and its `v` port of the
verifier's coefficients at the group's `V`. -/
theorem decode_part {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ} (hp : RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh) {S0 : RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf sh = .ok S0) (hg : groups S0 (portSizes sh) = .ok gs)
    (hpg : p < gs.size) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (hh : RecResiduals.holoOf c = .ok false)
    (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128) (w : ℕ → Flock.F128)
    (hw : ∀ q < 128 * ns.sum,
      RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (portSizes sh) gs[p].2.1) w) q))
    (hv : ∀ q < 128 * nv,
      RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 (vOf sh rounds extras comb)) q)) :
    partS S0 (portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S ∧
    ∃ w' v' : ℕ → Flock.F128,
      (∀ k (hk : k < (wIdx (portSizes sh) gs[p].2.1).size), w' k = w (wIdx (portSizes sh) gs[p].2.1)[k]) ∧
      (∀ k (hk : k < gs[p].2.2.size), v' k = vOf sh rounds extras comb gs[p].2.2[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  obtain ⟨hS', -, -⟩ := part_eq hS hg hpg hpart
  obtain ⟨hwI, hV⟩ := part_sizes hS hg hpg hpart
  exact ⟨hS', decode_check zo hA u hj hone hc hp hsh hpart hu hh rounds extras comb w _ _ hwI.le hV.le hw hv⟩

/-- **`decode_part` on a holo statement** (`decode_check_holo`): `sh` the holo form of the shape the verifier
accepts. -/
theorem decode_part_holo {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ} (hp : RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh0 sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh0) (hho : sh = { sh0 with holo := true })
    {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf sh = .ok S0)
    (hg : groups S0 (portSizes sh) = .ok gs) (hpg : p < gs.size) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (hh : RecResiduals.holoOf c = .ok true)
    (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128) (w : ℕ → Flock.F128)
    (hw : ∀ q < 128 * ns.sum,
      RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (portSizes sh) gs[p].2.1) w) q))
    (hv : ∀ q < 128 * nv,
      RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 (vOf sh rounds extras comb)) q)) :
    partS S0 (portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S ∧
    ∃ w' v' : ℕ → Flock.F128,
      (∀ k (hk : k < (wIdx (portSizes sh) gs[p].2.1).size), w' k = w (wIdx (portSizes sh) gs[p].2.1)[k]) ∧
      (∀ k (hk : k < gs[p].2.2.size), v' k = vOf sh rounds extras comb gs[p].2.2[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  obtain ⟨hS', -, -⟩ := part_eq hS hg hpg hpart
  obtain ⟨hwI, hV⟩ := part_sizes hS hg hpg hpart
  exact ⟨hS', decode_check_holo zo hA u hj hone hc hp hsh hho hpart hu hh rounds extras comb w _ _ hwI.le hV.le hw hv⟩

end Decode

/-! ## `AlgebraHolds` from the parts -/

/-- **The algebra holds when every part's outputs hold its residuals and are zero.** At S (`structureOf`) with its
wire bits in range (`BitsOk`), the verifier's coefficients `v` at the rep's coins (`verifierRows`), the sizes, and S's
groups: if for every group `j` the part's S (`partS`) has its residuals in columns `col` from `base` (`PartCols`) at
ports reading the rep's `w` at `wIdx` of the group's ports and `v` at its `V`, and those columns are zero, every
residual of S is zero (`residuals_zero_of_parts`). -/
theorem algebraHolds_of_parts {sh : Shape} {d : RepData} {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S)
    (hbits : S.BitsOk) {v : Array F128} {info kappa}
    (hvr : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa))
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64) (hv : v.size = S.nv)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (hparts : ∀ j (hj : j < gs.size), ∃ S', partS S (portSizes sh) gs[j].1 gs[j].2.1 gs[j].2.2 = .ok S' ∧
      ∃ (w' v' : ℕ → F128) (col : ℕ → Bool) (base : ℕ),
        (∀ k (hk : k < (wIdx (portSizes sh) gs[j].2.1).size), w' k = el d.w (wIdx (portSizes sh) gs[j].2.1)[k]) ∧
        (∀ k (hk : k < gs[j].2.2.size), v' k = el v gs[j].2.2[k]) ∧
        PartCols S' w' v' col base ∧ ∀ t < 128 * S'.res.length, col (base + t) = false) :
    AlgebraHolds sh d := by
  obtain ⟨rs, hrs, hz⟩ := FlockVBridge.Parts.residuals_zero_of_parts hbits hg hw hv fun j hj => by
    obtain ⟨S', hS', w', v', col, base, hw', hv', ⟨hr, -⟩, h0⟩ := hparts j hj
    refine ⟨S', hS', fun k hk => ?_⟩
    have hk0 : resVal (gfsOf S') (termVal w' v') (resOf S'.res[k]) = 0 :=
      AccChain.eq_of_bitOf fun t ht => by rw [← hr k hk t ht, h0 (128 * k + t) (by omega), bitOf_zero]
    obtain ⟨_, e1⟩ := FlockVBridge.PartS.res_partS hS' (w := el d.w) (v := el v) hw' hv'
    obtain ⟨_, e2⟩ := FlockVBridge.PartS.res_partS hS' (w := el d.w) (v := el v)
      (w' := fun k => el d.w ((wIdx (portSizes sh) gs[j].2.1).getD k 0)) (v' := fun k => el v (gs[j].2.2.getD k 0))
      (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl)
      (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl)
    exact (e2 k hk).trans ((e1 k hk).symm.trans hk0)
  exact ⟨v, info, kappa, S, rs, hvr, hS, hw, hc, hrs, hz⟩

/-! ## From one unit's copy per part -/

section Copy

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode
open FlockVBridge.AccChain (RegOut copyVal_regOut)

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

/-- **Copy `j` of unit `u` is part `p` of the groups `gs`, its outputs registered zero**: `decode_part`'s facts about
one copy, with every output column of the part (`16 · RES_OUT_WORDS` bits) a registered 0 (`RegOut`), as V* registers
`rec-residuals-zero`. Its row ports are registered bits of `w` at `wIdx` of group `p`'s ports and its `v` port of the
verifier's coefficients at the group's `V`. The circuit's parts are plain (`holoOf`). -/
def ResZeroAt (bits : Fin (Layout.keyN c hU nU) → Bool) (u : Fin nU) (j : ℕ) (sh : Shape)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (p : ℕ) (w : ℕ → F128) (rounds : Array (Array F128))
    (extras : Array Claim) (comb : Array F128) : Prop :=
  ∃ (hpg : p < gs.size) (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true ∧
    RecResiduals.check c = .ok () ∧ RecResiduals.holoOf c = .ok false ∧ c.unit.unitLog ≤ 32 ∧
    ∃ m k cl, RecResiduals.params c.unit.name = some (m, k, cl, p) ∧ RecResiduals.shape? m k cl = some sh ∧
    ∃ S ns nv, RecResiduals.part sh p = .ok (S, ns, nv) ∧
      (∀ q < 128 * ns.sum,
        RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (portSizes sh) gs[p].2.1) w) q)) ∧
      (∀ q < 128 * nv,
        RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 (vOf sh rounds extras comb)) q)) ∧
      ∀ t < 16 * RecResiduals.RES_OUT_WORDS, RegOut zo u hj (partBase S ns nv + t) false

/-- **A part's residuals, zero, from its unit's copy in a good layer** (`decode_part`, and `copyVal_regOut` at the
output columns). -/
theorem partCols_of_resZeroAt {bits : Fin (Layout.keyN c hU nU) → Bool} (hG : zo.Good bits) {u : Fin nU} {j : ℕ}
    {sh : Shape} {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf sh = .ok S0)
    (hg : groups S0 (portSizes sh) = .ok gs) {p : ℕ} {w : ℕ → F128} {rounds : Array (Array F128)}
    {extras : Array Claim} {comb : Array F128} (h : ResZeroAt zo bits u j sh gs p w rounds extras comb) :
    ∃ hpg : p < gs.size, ∃ S', partS S0 (portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S' ∧
      ∃ (w' v' : ℕ → F128) (col : ℕ → Bool) (base : ℕ),
        (∀ k (hk : k < (wIdx (portSizes sh) gs[p].2.1).size), w' k = w (wIdx (portSizes sh) gs[p].2.1)[k]) ∧
        (∀ k (hk : k < gs[p].2.2.size), v' k = vOf sh rounds extras comb gs[p].2.2[k]) ∧
        PartCols S' w' v' col base ∧ ∀ t < 128 * S'.res.length, col (base + t) = false := by
  obtain ⟨hpg, hj, hone, hc, hh, hu, m, k, cl, hp, hsh, S, ns, nv, hpart, hw, hv, hz⟩ := h
  obtain ⟨hS', w', v', hw', hv', hcols⟩ :=
    decode_part zo hG.2.1 u hj hone hc hp hsh hS hg hpg hpart hu hh rounds extras comb w hw hv
  obtain ⟨S1, ns1, nv1, hpart1, hn⟩ := check_part hc hp hsh hh
  rw [hpart] at hpart1
  cases hpart1
  have hbase : (Rows.ofNet hU).nIn ≤ partBase S ns nv := by
    rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
    exact inRows_le_outStart _ _ _
  have hsmall := part_small hpart
  exact ⟨hpg, S, hS', w', v', _, _, hw', hv', hcols, fun t ht =>
    copyVal_regOut zo hG.2.1 (keyProg_correct hU nU hG.1 u) hj (by omega) (hz t (by omega))⟩

/-- **`ResZeroAt` on a holo statement**: copy `j` of unit `u` is part `p` of the groups `gs` of `sh`, the holo form of
the shape the verifier accepts (META `holo: true`), its outputs registered zero. -/
def ResZeroAtH (bits : Fin (Layout.keyN c hU nU) → Bool) (u : Fin nU) (j : ℕ) (sh : Shape)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (p : ℕ) (w : ℕ → F128) (rounds : Array (Array F128))
    (extras : Array Claim) (comb : Array F128) : Prop :=
  ∃ (hpg : p < gs.size) (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true ∧
    RecResiduals.check c = .ok () ∧ RecResiduals.holoOf c = .ok true ∧ c.unit.unitLog ≤ 32 ∧
    ∃ m k cl, RecResiduals.params c.unit.name = some (m, k, cl, p) ∧
    ∃ sh0, RecResiduals.shape? m k cl = some sh0 ∧ sh = { sh0 with holo := true } ∧
    ∃ S ns nv, RecResiduals.part sh p = .ok (S, ns, nv) ∧
      (∀ q < 128 * ns.sum,
        RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (portSizes sh) gs[p].2.1) w) q)) ∧
      (∀ q < 128 * nv,
        RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 (vOf sh rounds extras comb)) q)) ∧
      ∀ t < 16 * RecResiduals.RES_OUT_WORDS, RegOut zo u hj (partBase S ns nv + t) false

/-- **`partCols_of_resZeroAt` on a holo statement** (`decode_part_holo`). -/
theorem partCols_of_resZeroAtH {bits : Fin (Layout.keyN c hU nU) → Bool} (hG : zo.Good bits) {u : Fin nU} {j : ℕ}
    {sh : Shape} {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf sh = .ok S0)
    (hg : groups S0 (portSizes sh) = .ok gs) {p : ℕ} {w : ℕ → F128} {rounds : Array (Array F128)}
    {extras : Array Claim} {comb : Array F128} (h : ResZeroAtH zo bits u j sh gs p w rounds extras comb) :
    ∃ hpg : p < gs.size, ∃ S', partS S0 (portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S' ∧
      ∃ (w' v' : ℕ → F128) (col : ℕ → Bool) (base : ℕ),
        (∀ k (hk : k < (wIdx (portSizes sh) gs[p].2.1).size), w' k = w (wIdx (portSizes sh) gs[p].2.1)[k]) ∧
        (∀ k (hk : k < gs[p].2.2.size), v' k = vOf sh rounds extras comb gs[p].2.2[k]) ∧
        PartCols S' w' v' col base ∧ ∀ t < 128 * S'.res.length, col (base + t) = false := by
  obtain ⟨hpg, hj, hone, hc, hh, hu, m, k, cl, hp, sh0, hsh, hho, S, ns, nv, hpart, hw, hv, hz⟩ := h
  obtain ⟨hS', w', v', hw', hv', hcols⟩ :=
    decode_part_holo zo hG.2.1 u hj hone hc hp hsh hho hS hg hpg hpart hu hh rounds extras comb w hw hv
  obtain ⟨S1, ns1, nv1, hpart1, hn⟩ := check_part_holo hc hp hsh hh
  rw [← hho, hpart] at hpart1
  cases hpart1
  have hbase : (Rows.ofNet hU).nIn ≤ partBase S ns nv := by
    rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
    exact inRows_le_outStart _ _ _
  have hsmall := part_small hpart
  exact ⟨hpg, S, hS', w', v', _, _, hw', hv', hcols, fun t ht =>
    copyVal_regOut zo hG.2.1 (keyProg_correct hU nU hG.1 u) hj (by omega) (hz t (by omega))⟩

end Copy

section Sessions

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode

variable {T : ℕ} {Vs : Fin T → VStmt} {Reg' : Type}

/-- **`AlgebraHolds` from each part's unit.** Let the sessions `zs` (one per V* statement) have `Good` layers `L`, and
let part `p` of S's groups be copy `cp p` of unit `un p` in session `sess p`, its row ports the rep's `w` and its `v`
the verifier's coefficients at the rep's coins, its outputs registered zero (`ResZeroAt`). With S's wire bits in range
(`BitsOk`), the verifier's coefficients `v` at the rep's coins (`verifierRows`) and the sizes, the algebra holds. -/
theorem algebraHolds_of_decode (zs : (s : Fin T) → (Vs s).Outer Reg') (L : (s : Fin T) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (L s)) {sh : Shape} {d : RepData} {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S)
    (hbits : S.BitsOk) {v : Array F128} {info kappa}
    (hvr : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa))
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64) (hv : v.size = S.nv)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (sess : ℕ → Fin T) (un : (p : ℕ) → Fin (Vs (sess p)).nv) (cp : ℕ → ℕ)
    (h : ∀ p < gs.size,
      ResZeroAt (zs (sess p)) (L (sess p)) (un p) (cp p) sh gs p (el d.w) d.rounds d.extras d.comb) :
    AlgebraHolds sh d := by
  refine algebraHolds_of_parts hS hbits hvr hw hc hv hg fun p hp => ?_
  obtain ⟨_, S', hS', w', v', col, base, hw', hv', hcols, h0⟩ :=
    partCols_of_resZeroAt (zs (sess p)) (hG (sess p)) hS hg (h p hp)
  rw [vOf_eq hvr] at hv'
  exact ⟨S', hS', w', v', col, base, hw', hv', hcols, h0⟩

/-- **`AlgebraHolds` from each part's unit, at a holo shape**: `algebraHolds_of_decode` at `ResZeroAtH`. -/
theorem algebraHolds_of_decodeH (zs : (s : Fin T) → (Vs s).Outer Reg') (L : (s : Fin T) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (L s)) {sh : Shape} {d : RepData} {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S)
    (hbits : S.BitsOk) {v : Array F128} {info kappa}
    (hvr : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa))
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64) (hv : v.size = S.nv)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (sess : ℕ → Fin T) (un : (p : ℕ) → Fin (Vs (sess p)).nv) (cp : ℕ → ℕ)
    (h : ∀ p < gs.size,
      ResZeroAtH (zs (sess p)) (L (sess p)) (un p) (cp p) sh gs p (el d.w) d.rounds d.extras d.comb) :
    AlgebraHolds sh d := by
  refine algebraHolds_of_parts hS hbits hvr hw hc hv hg fun p hp => ?_
  obtain ⟨_, S', hS', w', v', col, base, hw', hv', hcols, h0⟩ :=
    partCols_of_resZeroAtH (zs (sess p)) (hG (sess p)) hS hg (h p hp)
  rw [vOf_eq hvr] at hv'
  exact ⟨S', hS', w', v', col, base, hw', hv', hcols, h0⟩

section outer

open FlockVBridge.Compiled (Accepted layers AlgebraDecodes)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {Reg₂ : Type}

/-- **G items 1–3 from the parts' units**: `algebraHolds_of_decode` at every accepted outer proof, each rep's part `p`
in session `sess p`, with the verifier's coefficients and the sizes at the rep's values. -/
theorem algebraDecodes_of_decode (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (s : Fin T) → (Vs s).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape) (rep : ((s : Fin T) → (Vs s).Bits) → Fin 2 → RepData)
    {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S) (hbits : S.BitsOk)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (sess : ℕ → Fin T) (un : (p : ℕ) → Fin (Vs (sess p)).nv) (cp : ℕ → ℕ)
    (hvr : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
      verifierRows sh (rep (layers zs dg N k Rw R ω t R₂ τ) r).rounds (rep (layers zs dg N k Rw R ω t R₂ τ) r).extras
        (rep (layers zs dg N k Rw R ω t R₂ τ) r).comb = .ok (v, info, kappa) ∧ v.size = S.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      (rep (layers zs dg N k Rw R ω t R₂ τ) r).w.size = S.nw ∧ (rep (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (h : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
      ResZeroAt (zs R ω t R₂ (sess p)) (layers zs dg N k Rw R ω t R₂ τ (sess p)) (un p) (cp p) sh gs p
        (el (rep (layers zs dg N k Rw R ω t R₂ τ) r).w) (rep (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep (layers zs dg N k Rw R ω t R₂ τ) r).comb) :
    AlgebraDecodes zs dg N k Rw sh fun _ _ _ => rep := by
  intro R ω t R₂ τ o hacc r
  obtain ⟨v, info, kappa, hv, hvs⟩ := hvr R ω t R₂ τ o hacc r
  obtain ⟨hw, hc⟩ := hsz R ω t R₂ τ o hacc r
  exact algebraHolds_of_decode (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (fun s => (hacc.1 s).2) hS hbits hv hw hc
    hvs hg sess un cp (h R ω t R₂ τ o hacc r)

end outer

end Sessions

end FlockVBridge.OfVStar
