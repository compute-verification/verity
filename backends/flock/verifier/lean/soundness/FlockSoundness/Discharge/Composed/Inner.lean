import FlockSoundness.Discharge.Composed.Base
import FlockSoundness.Instance
import FlockSoundness.ZerocheckPhase
import FlockSoundness.LincheckPhase

/-!
# The inner proof's fields of a draw's `--zk` table

`coinedOf`'s tables take the fields the inner proof's constraint `⟨λ, h⟩ = θ` reads (`Table.InnerHolds`) from `innerOf`:
the level-0 lanes, the clear values, Ligerito's value lists and the verifier's replay `λ`, `θ`, at a coin vector
`c : CoinsZK (RestOf Tz st)`. `innerOf_holds` (`Composed/Inner/Holds.lean`) is its obligation.

* **The pads by layout index** (`padEquiv`, `padNat`): a rep's pads are `Zk.PadLayout`'s `len = nE + 4` slots, the
  exposed values first and then `hab, tx, ty, tw`; `PadIdx`'s tail is `h_tx, h_ty, h_ab, h_tw` (`Table.hFull`).
* **The replay** (`rowsOf`, `lamOf`, `thetaOf`): `Zk.verifyRep`'s ten constraints in its order, each an affine form in the
  pads (`LinF`, `ZkSession`'s forms): zerocheck c_eval, zerocheck final a·b and lincheck final (`Zk.replay`), ring switch
  ab and c (the claim checks on `ŝ`), level 0's lane phase c0 and c1 (`Zk.ligeritoConstraints`, on the restart `T'` that
  `lig` carries), and the triple's three (`Zk.triple`); `λ`, `θ` are their batch at the coins `γ` (`Zk.verifyInner`:
  `⟨l.a, h⟩ = l.c`). Every coefficient is `execArith`'s, the model's arithmetic of the executable.
* **The honest prover's clear values** (`honestAt`, by layout index): the zerocheck's round-1 values on `Λ`
  (`PabT`, `PcT`), its residual rounds `(G(1), G(∞))` (`resPoly`), the finals; the lincheck's rounds (`roundPoly`) and
  `z_partial`; level 0's out-of-domain value, lane rounds and the extra lanes' sums `e_i`, on `z = (w, u)` (`zWith`).
* **The lanes** (`lanesOf`): packed word `y = l + 2^k₀·c` of `z` is lane `l`'s column `c`.
* **Levels ≥ 1** (`ligOf`): the restart `T' = Σ_c y₁'(c)·b̄(c)` and levels ≥ 1's bytes.

**Open inputs** (each a named definition, until the coin vector or the table carries it): level 0's out-of-domain point
(`oodPoint`), the region claims' points (`extraPoint`), and levels ≥ 1's bytes (`ligBytes`, which need Ligerito's level
≥ 1 coins). `InnerHolds` reads none of `ligBytes`.
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

noncomputable section

/-- **The inner proof's fields** of a `Coined` table over the witnesses `W` at the sizes `z` (`Coined`'s `lanes`, `vals`,
`lig`, `lam`, `theta`). -/
structure InnerFields (W Hi VL : Type) (z : Sizes) where
  lanes : W → (Fin z.nMask → Fin 128 → ZMod 2) → Fin z.nL → Fin z.L → 𝔽
  vals : Fin 2 → W → (Fin z.nMask → Fin 128 → ZMod 2) → (Fin 2 → Fin 2 → Fin z.L → 𝔽) → Fin z.nE → 𝔽
  lig : Fin 2 → (Fin z.L → 𝕂) → Hi → VL
  lam : Fin 2 → (Fin z.nC → Fin 128 → 𝔽) × (Fin 2 → Fin z.nE → 𝔽) × (Fin 2 → Fin z.L → 𝕂) ×
    (Fin 2 → Fin z.tpad → 𝕂) × (Fin 2 → 𝔽 × 𝔽) × (Fin 2 → VL) → PadIdx z.nE → 𝔽
  theta : Fin 2 → (Fin z.nC → Fin 128 → 𝔽) × (Fin 2 → Fin z.nE → 𝔽) × (Fin 2 → Fin z.L → 𝕂) ×
    (Fin 2 → Fin z.tpad → 𝕂) × (Fin 2 → 𝔽 × 𝔽) × (Fin 2 → VL) → 𝔽

/-- **Levels ≥ 1's value list**: the restart `T'` (`Level0Msgs.tPrime`, a clear `K` message the lane-phase rows read)
and the bytes of levels ≥ 1 after it. -/
abbrev VLOf : Type := 𝕂 × ByteArray

/-- The messages the replay reads (`Table.View.pre`) at the sizes `z`, with `VLOf`'s value lists. -/
abbrev PreOf (z : Sizes) : Type :=
  (Fin z.nC → Fin 128 → 𝔽) × (Fin 2 → Fin z.nE → 𝔽) × (Fin 2 → Fin z.L → 𝕂) × (Fin 2 → Fin z.tpad → 𝕂) ×
    (Fin 2 → 𝔽 × 𝔽) × (Fin 2 → VLOf)

/-! ## The pads by layout index -/

/-- `PadIdx`'s tail `h_tx, h_ty, h_ab, h_tw` to the layout's tail offsets: `tx, ty` are `hab + 1, hab + 2`. -/
def tailPerm : Fin 4 ≃ Fin 4 where
  toFun := ![1, 2, 0, 3]
  invFun := ![2, 0, 1, 3]
  left_inv := by decide
  right_inv := by decide

/-- **A rep's pads at their layout indices**: exposed pad `i` at `i`, `h_tx, h_ty, h_ab, h_tw` at
`nE + 1, nE + 2, nE, nE + 3` (`PadLayout.tx, ty, hab, tw` with `hab = nE`). -/
def padEquiv (nE : ℕ) : PadIdx nE ≃ Fin (nE + 4) :=
  (Equiv.sumCongr (Equiv.refl _) tailPerm).trans finSumFinEquiv

/-- A pad's layout index (`padEquiv`). -/
def padNat {nE : ℕ} (i : PadIdx nE) : ℕ := (padEquiv nE i).val

/-- Layout index `k` as a pad (wrapped; below `nE + 4` at the layout's sizes). -/
def ix (nE k : ℕ) : Fin (nE + 4) := ⟨k % (nE + 4), Nat.mod_lt _ (by omega)⟩

/-- The masked value at layout index `k` (`0` past the exposed values). -/
def msgAt {nE : ℕ} (v : Fin nE → 𝔽) (k : ℕ) : 𝔽 := if h : k < nE then v ⟨k, h⟩ else 0

/-- `Lin.masked`: the masked value at layout index `k`, unmasked by its pad. -/
def mk {nE : ℕ} (v : Fin nE → 𝔽) (k : ℕ) : LinF 𝔽 (nE + 4) := mskF (ix nE k) (msgAt v k)

/-- A finite vector read by index (`0` past its end). -/
def seqOf {α : Type} [Zero α] {n : ℕ} (f : Fin n → α) (j : ℕ) : α := if h : j < n then f ⟨j, h⟩ else 0

/-- The draw's pads layout (`verifyRep`'s `PadLayout.new st.m st.kLog k₀`). -/
abbrev layOf {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) : Zk.PadLayout := Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0

/-- The triple's pads (`Zk.triple`'s `fa, fb, hab, tx, ty, tw`). -/
def trOf (nE : ℕ) (lay : Zk.PadLayout) : Triple (nE + 4) :=
  ⟨ix nE lay.fa, ix nE lay.fb, ix nE lay.hab, ix nE lay.tx, ix nE lay.ty, ix nE lay.tw⟩

/-! ## The coins, as field elements -/

section Coins

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)

/-- The zerocheck's `z`. -/
def zcZ : 𝔽 := fOf (c.rest.zc r)

/-- The zerocheck's eq point `r = pinned ‖ r_outer` (`zcVector`), by index. -/
def zcReq (j : ℕ) : 𝔽 := if h : j < 7 then execArith.pinned ⟨j, h⟩ else fOf (seqOf (c.rest.rOuter r) (j - 7))

/-- The residual rounds' coins `ρ_j`. -/
def zcRho (j : ℕ) : 𝔽 := fOf (seqOf (c.rest.rho r) j)

/-- The lincheck's `α`, `β`, round coins `t_j` and `r′`. -/
def lcAlpha : 𝔽 := fOf (c.rest.lcAlpha r)
def lcBeta : 𝔽 := fOf (c.rest.lcBeta r)
def lcT (j : ℕ) : 𝔽 := fOf (seqOf (c.rest.ts r) j)
def lcR : 𝔽 := fOf (c.rest.rPrime r)

/-- The `ab` claim's point `r_in' ‖ ρ[k_log − 6 …]`, `r_in'` the lincheck's coins reversed (`abcClaims`). -/
def abX (j : ℕ) : 𝔽 :=
  if j < st.c.kLog - 6 then lcT Tz st c r (st.c.kLog - 6 - 1 - j) else zcRho Tz st c r j

/-- Level 0's lane challenges `r̄_j` and extra-lane coins `ρ_i`. -/
def laneCh (j : ℕ) : 𝕂 := FlockLevel3.GF256.of (seqOf (c.rest.lane r) j)
def rhoK (i : Fin 2) : 𝕂 := if i = 0 then FlockLevel3.GF256.of (c.rho r).1 else FlockLevel3.GF256.of (c.rho r).2

/-- The inner proof's sacrifice coin `ε` and its constraints' `γ`. -/
def innerEps : 𝔽 := fOf (c.rest.zeta r)
def innerGamma (k : Fin (7 + 3)) : 𝔽 := fOf (c.rest.gammas r k)

end Coins

/-! ## The replay -/

section Replay

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) {nE : ℕ}
  (v : Fin nE → 𝔽)

/-- `P_c(z)` on the masked round 1 (`computedC`). -/
def pcL : LinF 𝔽 (nE + 4) :=
  ∑ i : Fin 64, execArith.lagΛ (zcZ Tz st c r) i • mk v ((layOf Tz st).r1c i)

/-- The zerocheck's running claim after `j` residual rounds (`residualStepL` on the masked `G(1)`, `G(∞)`). -/
def zcRun : ℕ → LinF 𝔽 (nE + 4)
  | 0 => ∑ i : Fin 64, execArith.combW (zcZ Tz st c r) i •
      (mk v ((layOf Tz st).r1ab i) + mk v ((layOf Tz st).r1c i)) + pcL Tz st c r v
  | j + 1 => residualStepL (zcReq Tz st c r j) (zcRho Tz st c r j) (zcRun j) (mk v ((layOf Tz st).ml j 0))
      (mk v ((layOf Tz st).ml j 1))

/-- **zerocheck c_eval**: `P_c(z) + final_c`. -/
def rowCEval : LinF 𝔽 (nE + 4) := pcL Tz st c r v + mk v (layOf Tz st).fc

/-- **zerocheck final a·b**: the running claim plus `a·b = a'b' + a'·h_fb + b'·h_fa + h_ab`. -/
def rowAB : LinF 𝔽 (nE + 4) :=
  let lay := layOf Tz st
  let a := msgAt v lay.fa
  let b := msgAt v lay.fb
  zcRun Tz st c r v (st.m - 6) + (cF (a * b) + a • pF (ix nE lay.fb) + b • pF (ix nE lay.fa) + pF (ix nE lay.hab))

/-- The lincheck's running claim after `j` rounds, from `α·a + b + β`. -/
def lcRun : ℕ → LinF 𝔽 (nE + 4)
  | 0 => lcAlpha Tz st c r • mk v (layOf Tz st).fa + mk v (layOf Tz st).fb + cF (lcBeta Tz st c r)
  | j + 1 =>
      let q₁ := mk v ((layOf Tz st).lc j 0)
      let q₂ := mk v ((layOf Tz st).lc j 1)
      let e₀ := lcRun j + q₁
      (lcT Tz st c r j ^ 2) • q₂ + lcT Tz st c r j • (e₀ + q₁ + q₂) + e₀

/-- The circuit's folded columns at the lincheck's coins (`comb`, folded by `foldAll`). -/
def lcFolded : ℕ → 𝔽 :=
  foldAll (List.ofFn fun j : Fin (st.c.kLog - 6) => lcT Tz st c r j) (2 ^ Tz.S.kLog)
    (comb execArith Tz.S (zcZ Tz st c r) (lcAlpha Tz st c r) (lcBeta Tz st c r) fun j => zcRho Tz st c r j)

/-- **lincheck final**: the running claim plus `Σ_s folded_s·z_partial_s`. -/
def rowLC : LinF 𝔽 (nE + 4) :=
  lcRun Tz st c r v (st.c.kLog - 6) + ∑ s : Fin 64, lcFolded Tz st c r s • mk v ((layOf Tz st).zp s)

/-- A claim check's weight on the 128 packed positions (`Zk.claimCheck`): `skip[v mod 64]·eq(x₀, ⌊v/64⌋)`. -/
def ccWeight (skip : Fin 64 → 𝔽) (x0 : 𝔽) (v : Fin 128) : 𝔽 :=
  skip ⟨v.val % 64, Nat.mod_lt _ (by norm_num)⟩ * if v.val / 64 = 1 then x0 else 1 + x0

/-- Masked claim `k`'s ring-switch message in `ŝ` (`0` past the claims). -/
def sHatAt {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (k : ℕ) : Fin 128 → 𝔽 := if h : k < nC then sh ⟨k, h⟩ else 0

/-- The `ab` and `c` claim checks on rep `r`'s `ŝ` (claims `2r`, `2r + 1`). -/
def ccAb {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : 𝔽 :=
  ∑ w, ccWeight (execArith.lagS (lcR Tz st c r)) (abX Tz st c r 0) w * sHatAt sh (2 * r.val) w
def ccC {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : 𝔽 :=
  ∑ w, ccWeight (execArith.lagS (zcZ Tz st c r)) (zcReq Tz st c r 0) w * sHatAt sh (2 * r.val + 1) w

/-- **ring switch ab**: the `ab` claim's value `Σ_s L_s(r′)·z_partial_s` plus its claim check. -/
def rowRsAb {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : LinF 𝔽 (nE + 4) :=
  ∑ s : Fin 64, execArith.lagS (lcR Tz st c r) s • mk v ((layOf Tz st).zp s) + cF (ccAb Tz st c r sh)

/-- **ring switch c**: `final_c` plus the `c` claim's check. -/
def rowRsC {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : LinF 𝔽 (nE + 4) :=
  mk v (layOf Tz st).fc + cF (ccC Tz st c r sh)

/-- Claim `k`'s ring-switch message in rep `r`'s opening: the masked `ab`, `c`, then the extras' public values. -/
def swAt {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (k : ℕ) : Fin 128 → 𝔽 :=
  if k = 0 then sHatAt sh (2 * r.val)
  else if k = 1 then sHatAt sh (2 * r.val + 1)
  else fun w => if h : k - 2 < st.regions.size then pubRegOf st ⟨k - 2, h⟩ w else 0

/-- **The opening's target** `T = Σ_k γ_k·β_k`, `β_k = recombine(ŝ_k, r''_k)` (`Opening.ringSwitch`). -/
def openT {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : 𝔽 :=
  ∑ k : Fin (2 + st.regions.size), fOf (c.rest.rsGamma r k) *
    recombine execArith (swAt st r sh k.val) fun i => fOf (c.rest.rpp r k i)

/-- A masked `K` message at layout indices `i₀, i₁` (its coordinates), unmasked by their pads (`l2masked`). -/
def mkK (i₀ i₁ : ℕ) : LinK 𝕂 (nE + 4) :=
  cK (execArith.ofLimbs (msgAt v i₀) (msgAt v i₁)) + padK execArith (ix nE i₀) (ix nE i₁)

/-- Level 0's lane-phase claim after `j` lane rounds, from `T + Σ_i β_i·(y_i' + h)`. -/
def laneRun {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : ℕ → LinK 𝕂 (nE + 4)
  | 0 => cK (execArith.embed (openT Tz st c r sh)) +
      ∑ i : Fin (layOf Tz st).nOod0, execArith.embed (fOf (c.rest.ood r i)) •
        (cK (execArith.embed (msgAt v ((layOf Tz st).lgOod i))) + padF (ix nE ((layOf Tz st).lgOod i)))
  | j + 1 =>
      (fromMsgL (mkK v ((layOf Tz st).lgMsg j 0) ((layOf Tz st).lgMsg j 1),
        mkK v ((layOf Tz st).lgMsg j 2) ((layOf Tz st).lgMsg j 3)) (laneRun sh j)).eval (laneCh Tz st c r j)

/-- **Level 0's lane-phase row** (`K`-valued): the claim after the `k₀` lane rounds plus `Σ_i ρ_i·e_i`, minus `T'`. -/
def laneRow {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) : LinK 𝕂 (nE + 4) :=
  laneRun Tz st c r v sh Tz.sch.k0 +
    ∑ i : Fin 2, rhoK Tz st c r i • mkK v ((layOf Tz st).lgE i 0) ((layOf Tz st).lgE i 1) - cK T'

/-- **Rep `r`'s ten constraints** in `verifyRep`'s order, on its masked values `v`, the masked claims' `ŝ`, the restart
`T'` and the triple's `(ρ, σ)`. -/
def rowsOf {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽) : Fin (7 + 3) → ZkSession.Lin 𝔽 (nE + 4) :=
  Fin.append
    ![(rowCEval Tz st c r v).toLin, (rowAB Tz st c r v).toLin, (rowLC Tz st c r v).toLin,
      (rowRsAb Tz st c r v sh).toLin, (rowRsC Tz st c r v sh).toLin,
      limbRows execArith (laneRow Tz st c r v sh T') 0, limbRows execArith (laneRow Tz st c r v sh T') 1]
    (tripleRows (trOf nE (layOf Tz st)) (innerEps Tz st c r) ρσ.1 ρσ.2)

end Replay

/-- **The replay's batched coefficients** `λ = Σ_k γ_k·coef_k` (`verifyInner`'s `l.a`), at each pad. -/
def lamOf {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (z : Sizes) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
    (pre : PreOf z) (i : PadIdx z.nE) : 𝔽 :=
  batchCoef (innerGamma Tz st c r)
    (rowsOf Tz st c r (pre.2.1 r) pre.1 (pre.2.2.2.2.2 r).1 (pre.2.2.2.2.1 r)) (padEquiv z.nE i)

/-- **The replay's batched target** `θ = −Σ_k γ_k·const_k` (`verifyInner`'s `l.c`, characteristic 2). -/
def thetaOf {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (z : Sizes) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
    (pre : PreOf z) : 𝔽 :=
  batchT (innerGamma Tz st c r) (rowsOf Tz st c r (pre.2.1 r) pre.1 (pre.2.2.2.2.2 r).1 (pre.2.2.2.2.1 r))

/-! ## The honest prover's clear values -/

/-- The mask slot's word indices in the packed witness (`Defs.maskWordsOf`'s definition, until Base holds it). -/
def maskSlotWords (st : Stmt) : List ℕ :=
  match st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => []
  | some r => (List.range (2 ^ st.nbl)).flatMap fun b =>
      (List.range (2 ^ (r.slotLog - 7))).map fun j => b * 2 ^ (st.c.kLog - 7) + r.start / 128 + j

/-- **The prover's table `z = (w, u)`**: the witness with the mask slot's words replaced by `u`. -/
def zWith (st : Stmt) {M nMask : ℕ} (w : Witness M) (u : Fin nMask → Fin 128 → ZMod 2) : Witness M := fun a =>
  match (List.finRange nMask).find? (fun x => (maskSlotWords st).getD x.val 0 = a.val / 128) with
  | some x => u x ⟨a.val % 128, Nat.mod_lt _ (by norm_num)⟩
  | none => w a

/-- Packed word `y` of a table, `f(y) = Σ_v z[128y + v]·x^v` (`Arith.pack`). -/
def wordOf {M : ℕ} (zw : Witness M) (y : ℕ) : 𝔽 :=
  execArith.pack fun v => if h : 128 * y + v.val < 2 ^ M then zw ⟨128 * y + v.val, h⟩ else 0

theorem six_le {mPts : ℕ} (Tz : TabZK 𝔽 mPts) : 6 ≤ Tz.S.m := by have := Tz.hm; omega

/-- **Level 0's out-of-domain point** (open: `RestZK` does not carry it yet; `ZkLigerito.ligerito`'s
`z ← sampleVec n`). -/
def oodPoint {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (_c : CoinsZK (RestOf Tz st)) (_r : Fin 2) (_i : ℕ) :
    Fin (Tz.S.m - 7) → 𝔽 := sorry

/-- **Region claim `k`'s point** (open: the session's link points, which `Tz`, `st` and the coins do not carry;
`Stmt.extraClaims`). -/
def extraPoint {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (_st : Stmt) (_k : ℕ) : Fin (Tz.S.m - 6) → 𝔽 := sorry

/-- **Levels ≥ 1's bytes** after `T'` (open: they read Ligerito's level ≥ 1 coins, which `RestZK` does not carry). -/
def ligBytes {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (_c : CoinsZK (RestOf Tz st)) (_r : Fin 2) {L : ℕ}
    (_y : Fin L → 𝕂) (_hi : HiOf Tz) : ByteArray := sorry

section Honest

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m)

/-- The zerocheck's eq point at the model statement's size. -/
def reqS (j : Fin (Tz.S.m - 6)) : 𝔽 := zcReq Tz st c r j.val

/-- Round 1's clear values on `Λ`: `P_ab(φ₈(64 + i))`, `P_c(φ₈(64 + i))`. -/
def r1abH (i : ℕ) : 𝔽 :=
  seqOf (fun i : Fin 64 => (PabT execArith execArith_correct Tz.S (six_le Tz) zw (reqS Tz st c r)).eval
    (nodeΛ execArith execArith_correct i)) i
def r1cH (i : ℕ) : 𝔽 :=
  seqOf (fun i : Fin 64 => (PcT execArith execArith_correct (six_le Tz) zw (reqS Tz st c r)).eval
    (nodeΛ execArith execArith_correct i)) i

/-- The residual rounds' tables `Â_x(z)`, `B̂_x(z)` over `x`, folded by the first `j` coins (`foldLo`). -/
def resU : ℕ → ℕ → 𝔽
  | 0 => fun x => if h : x < 2 ^ (Tz.S.m - 6) then
      (hatP execArith execArith_correct (six_le Tz) (aVec Tz.S zw) ⟨x, h⟩).eval (zcZ Tz st c r) else 0
  | j + 1 => foldLo (zcRho Tz st c r j) (resU j)
def resV : ℕ → ℕ → 𝔽
  | 0 => fun x => if h : x < 2 ^ (Tz.S.m - 6) then
      (hatP execArith execArith_correct (six_le Tz) (bVec Tz.S zw) ⟨x, h⟩).eval (zcZ Tz st c r) else 0
  | j + 1 => foldLo (zcRho Tz st c r j) (resV j)

/-- Residual round `j`'s quadratic `G_j` (`resPoly` at the eq point's coordinates `j …`). -/
def resG (j : ℕ) : Polynomial 𝔽 :=
  resPoly (n := Tz.S.m - 7 - j) (fun i => zcReq Tz st c r (j + i.val)) (resU Tz st c r zw j) (resV Tz st c r zw j)

/-- The lincheck's tables: the circuit's folded columns and the witness's block-folded columns (`hcol`), folded by the
first `j` round coins (`foldTop`, top bit first). -/
def lcU : ℕ → ℕ → 𝔽
  | 0 => comb execArith Tz.S (zcZ Tz st c r) (lcAlpha Tz st c r) (lcBeta Tz st c r) fun j => zcRho Tz st c r j
  | j + 1 => foldTop (lcT Tz st c r j) (2 ^ (Tz.S.kLog - j - 1)) (lcU j)
def lcV : ℕ → ℕ → 𝔽
  | 0 => hcol Tz.S (ρOutOf Tz.S fun j => zcRho Tz st c r j) zw
  | j + 1 => foldTop (lcT Tz st c r j) (2 ^ (Tz.S.kLog - j - 1)) (lcV j)

/-- Lincheck round `j`'s quadratic (`roundPoly`). -/
def lcQ (j : ℕ) : Polynomial 𝔽 :=
  roundPoly (2 ^ (Tz.S.kLog - j - 1)) (lcU Tz st c r j) (lcV Tz st c r zw j)

/-- Claim `k`'s point in rep `r`'s opening: `ab`, `c`, then the extras. -/
def claimX (k : ℕ) : Fin (Tz.S.m - 6) → 𝔽 :=
  if k = 0 then fun j => abX Tz st c r j.val
  else if k = 1 then fun j => zcReq Tz st c r j.val
  else extraPoint Tz st (k - 2)

/-- **The opening's batched basis** `b(y) = Σ_k γ_k·Φ_k(eq(x_k[1 …], y))` (`Model.opening`). -/
def openB (y : ℕ) : 𝔽 :=
  ∑ k : Fin (2 + st.regions.size), fOf (c.rest.rsGamma r k) *
    phi execArith (fun i => fOf (c.rest.rpp r k i)) (eqAt (fun j : Fin (Tz.S.m - 7) => claimX Tz st c r k.val
      ⟨j.val + 1, by have := j.isLt; omega⟩) y)

/-- Level 0's basis `B(y) = b(y) + Σ_i β_i·eq(z_i, y)`, over `K`. -/
def laneB0 (y : ℕ) : 𝕂 :=
  execArith.embed (openB Tz st c r y) + ∑ i : Fin (layOf Tz st).nOod0,
    execArith.embed (fOf (c.rest.ood r i)) * execArith.embed (eqAt (oodPoint Tz st c r i.val) y)

/-- The lane rounds' tables `f`, `B` over the packed words, folded by the first `j` lane challenges (lowest bit
first). -/
def laneF : ℕ → ℕ → 𝕂
  | 0 => fun y => execArith.embed (wordOf zw y)
  | j + 1 => foldLo (laneCh Tz st c r j) (laneF j)
def laneB : ℕ → ℕ → 𝕂
  | 0 => laneB0 Tz st c r
  | j + 1 => foldLo (laneCh Tz st c r j) (laneB j)

/-- Lane round `j`'s message `(q(0), q₂)` of `q(X) = Σ_y' f(X, y')·B(X, y')`. -/
def laneMsg (j : ℕ) : 𝕂 × 𝕂 :=
  let h := 2 ^ (Tz.S.m - 7 - j - 1)
  (∑ y ∈ Finset.range h, laneF Tz st c r zw j (2 * y) * laneB Tz st c r j (2 * y),
    ∑ y ∈ Finset.range h, (laneF Tz st c r zw j (2 * y + 1) + laneF Tz st c r zw j (2 * y)) *
      (laneB Tz st c r j (2 * y + 1) + laneB Tz st c r j (2 * y)))

/-- `b̄`: the basis folded by every lane challenge, over the columns. -/
def bFold (col : ℕ) : 𝕂 := laneB Tz st c r Tz.sch.k0 col

/-- **The clear value at layout index `k`** of rep `r` (`PadLayout`'s order), on the table `zw` and the extra lanes
`R` (`e_i = Σ_c R_i(c)·b̄(c)`). -/
def honestAt {L : ℕ} (R : Fin 2 → Fin L → 𝔽) (k : ℕ) : 𝔽 :=
  let lay := layOf Tz st
  if k < 64 then r1abH Tz st c r zw k
  else if k < 128 then r1cH Tz st c r zw (k - 64)
  else if k < lay.fa then
    (if (k - 128) % 2 = 0 then (resG Tz st c r zw ((k - 128) / 2)).eval 1
      else (resG Tz st c r zw ((k - 128) / 2)).coeff 2)
  else if k = lay.fa then resU Tz st c r zw (Tz.S.m - 6) 0
  else if k = lay.fb then resV Tz st c r zw (Tz.S.m - 6) 0
  else if k = lay.fc then (PcT execArith execArith_correct (six_le Tz) zw (reqS Tz st c r)).eval (zcZ Tz st c r)
  else if k < lay.zp 0 then
    (if (k - lay.fc - 1) % 2 = 0 then (lcQ Tz st c r zw ((k - lay.fc - 1) / 2)).eval 1
      else (lcQ Tz st c r zw ((k - lay.fc - 1) / 2)).coeff 2)
  else if k < lay.zp 64 then lcV Tz st c r zw (Tz.S.kLog - 6) (k - lay.zp 0)
  else if k < lay.lgMsg 0 0 then
    ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), eqAt (oodPoint Tz st c r (k - lay.lgOod 0)) y * wordOf zw y
  else if k < lay.lgE 0 0 then
    let j := (k - lay.lgMsg 0 0) / 4
    let t := (k - lay.lgMsg 0 0) % 4
    let m := laneMsg Tz st c r zw j
    let x := if t < 2 then m.1 else m.2
    if t % 2 = 0 then (coords execArith x).1 else (coords execArith x).2
  else if k < lay.hab then
    let i := (k - lay.lgE 0 0) / 2
    let e := ∑ col : Fin L, execArith.embed (seqOf R i col) * bFold Tz st c r col.val
    if (k - lay.lgE 0 0) % 2 = 0 then (coords execArith e).1 else (coords execArith e).2
  else 0

end Honest

/-- **The draw's inner-proof fields** at the coin vector `c`: the lanes of `z = (w, u)`, the clear values by layout
index, levels ≥ 1 from the restart `T' = Σ_c y₁'(c)·b̄(c)`, and the replay's batch. -/
def innerOf {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (z : Sizes) (c : CoinsZK (RestOf Tz st)) :
    InnerFields (Witness Tz.S.m) (HiOf Tz) VLOf z where
  lanes w u l col := wordOf (zWith st w u) (l.val + z.nL * col.val)
  vals r w u R i := honestAt Tz st c r (zWith st w u) (R r) i.val
  lig r y hi := (∑ col : Fin z.L, y col * bFold Tz st c r col.val, ligBytes Tz st c r y hi)
  lam := lamOf Tz st z c
  theta := thetaOf Tz st z c

end

end FlockSoundness.Discharge.Composed
