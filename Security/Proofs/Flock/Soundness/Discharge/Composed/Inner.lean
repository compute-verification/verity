import Proofs.Flock.Soundness.Discharge.Composed.Base
import Proofs.Flock.Soundness.Instance
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckMasked
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Lincheck

/-!
# The inner proof's fields of a draw's `--zk` table

`coinedOf`'s tables take the fields the inner proof's constraint `⟨λ, h⟩ = θ` reads (`Table.InnerHolds`) from `innerOf`:
the level-0 lanes, the clear values, Ligerito's value lists and the verifier's replay `λ`, `θ`, at a coin vector
`c : CoinsZK (RestOf Tz st)`. `innerOf_holds` (`Composed/Inner/Holds.lean`) is its obligation.

* **The pads by layout index** (`padEquiv`, `padNat`): a rep's pads are `Zk.PadLayout`'s `len = nE + 4` slots, the
  exposed values first and then `hab, tx, ty, tw`; `PadIdx`'s tail is `h_tx, h_ty, h_ab, h_tw` (`Table.hFull`).
* **The replay** (`rowsOf`, `lamOf`, `thetaOf`): `Zk.verifyRep`'s ten constraints in its order, each an affine form in the
  pads: zerocheck c_eval, zerocheck final a·b and lincheck final (`Zk.replay`'s forms on the masked proof, restated in
  `ZerocheckMasked` and `Lincheck`, at the drawn coins), ring switch ab and c (their values plus `Zk.claimCheck` on `ŝ`),
  read in the field (`ofZk`); level 0's lane phase c0 and c1 (`Zk.ligeritoConstraints`, on the restart `T'` that `lig`
  carries, `execArith`'s forms); and the triple's three (`Zk.triple`). `λ`, `θ` are their batch at the coins `γ`
  (`Zk.verifyInner`: `⟨l.a, h⟩ = l.c`). The lincheck's circuit fold is the statement's (`modelFold`).
* **The honest prover's clear values** (`honestAt`, by layout index): the clear zerocheck `Zerocheck.honest` on the A-side,
  B-side and witness bits (`zcH`), the lincheck's rounds `Lincheck.lcMsg` and `z_partial` (`lcH`, `zpH`); level 0's
  out-of-domain value, lane rounds and the extra lanes' sums `e_i`, on `z = (w, u)` (`zWith`).
* **The lanes** (`lanesOf`): packed word `y = l + 2^k₀·c` of `z` is lane `l`'s column `c`.
* **Levels ≥ 1** (`ligOf`): the restart `T' = Σ_c y₁'(c)·b̄(c)` and levels ≥ 1's bytes.

**Open inputs** (each a named definition, until the coin vector carries it): level 0's out-of-domain point
(`oodPoint`, the drawn coin `RestZK.oodPt`) and levels ≥ 1's bytes (a `LigProver`'s). `InnerHolds` reads none of them.
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

/-- The zerocheck's eq point `r = pinned ‖ r_outer` (`zcVector`), by index. -/
def zcReq (j : ℕ) : 𝔽 := if h : j < 7 then execArith.pinned ⟨j, h⟩ else fOf (seqOf (c.rest.rOuter r) (j - 7))

/-- The residual rounds' coins `ρ_j`. -/
def zcRho (j : ℕ) : 𝔽 := fOf (seqOf (c.rest.rho r) j)

/-- The lincheck's round coins `t_j`. -/
def lcT (j : ℕ) : 𝔽 := fOf (seqOf (c.rest.ts r) j)

/-- The `ab` claim's point `r_in' ‖ ρ[k_log − 6 …]`, `r_in'` the lincheck's coins reversed (`abcClaims`). -/
def abX (j : ℕ) : 𝔽 :=
  if j < st.c.kLog - 6 then lcT Tz st c r (st.c.kLog - 6 - 1 - j) else zcRho Tz st c r j

/-- Level 0's lane challenges `r̄_j` and extra-lane coins `ρ_i`. -/
def laneCh (j : ℕ) : 𝕂 := FlockLevel3.GF256.of (seqOf (c.rest.lane r) j)
def rhoK (i : Fin 2) : 𝕂 := if i = 0 then FlockLevel3.GF256.of (c.rho r).1 else FlockLevel3.GF256.of (c.rho r).2

/-- The inner proof's sacrifice coin `ε` and its constraints' `γ`. -/
def innerEps : 𝔽 := fOf (c.rest.zeta r)
def innerGamma (k : Fin (7 + 3)) : 𝔽 := fOf (c.rest.gammas r k)

/-- The executable's coin arrays: the zerocheck vector `pinned ‖ r_outer`, the residual coins `ρ`, the lincheck's
round coins and `z`. -/
def rF : Array F128 := pinned ++ Array.ofFn (c.rest.rOuter r)
def rhoF : Array F128 := Array.ofFn (c.rest.rho r)
def tsF : Array F128 := Array.ofFn (c.rest.ts r)
def zcF : F128 := c.rest.zc r

end Coins

/-- **The statement's circuit fold, in the model** (`FoldRealizes` by construction): column `col` of
`α·A₀ᵀe + B₀ᵀe`, `e_i = lw[i mod 64]·eq(ρ, ⌊i/64⌋)`; `folded` adds `β` at the pin and halves at `ts`. -/
def modelFold (S : Statement) : CircuitFold where
  kLog := S.kLog
  pin := S.pin.val
  fold α lw ρ := .ok ((Array.range (2 ^ S.kLog)).map fun col => if hc : col < 2 ^ S.kLog then
    (∑ i : Fin (2 ^ S.kLog), fOf lw[i.val % 64]! * eqAt (vecOf (S.kLog - 6) ρ) (i.val / 64) *
      (fOf α * ((S.A₀ i ⟨col, hc⟩).val : 𝔽) + ((S.B₀ i ⟨col, hc⟩).val : 𝔽))).val else 0)
  folded α β lw ρ ts := .ok (ts.foldl (fun acc t => halve t acc)
    (((Array.range (2 ^ S.kLog)).map fun col => if hc : col < 2 ^ S.kLog then
      (∑ i : Fin (2 ^ S.kLog), fOf lw[i.val % 64]! * eqAt (vecOf (S.kLog - 6) ρ) (i.val / 64) *
        (fOf α * ((S.A₀ i ⟨col, hc⟩).val : 𝔽) + ((S.B₀ i ⟨col, hc⟩).val : 𝔽))).val else 0).modify S.pin.val (· + β)))

section Coins'

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)

/-- The circuit's 64 folded entries at the lincheck's coins (`modelFold`'s `folded`, the `combv` the replay reads). -/
def combvOf : Array F128 :=
  match (modelFold Tz.S).folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz st c r))
    ((rhoF Tz st c r).extract 0 (st.c.kLog - 6)) (tsF Tz st c r) with
  | .ok v => v
  | .error _ => #[]

end Coins'

/-! ## The replay -/

section Replay

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) {nE : ℕ}
  (v : Fin nE → 𝔽)

/-- An executable affine form (`Zk.Lin`) in the field, on the rep's pads (`Lincheck.linVal`). -/
def ofZk (l : Zk.Lin) : ZkSession.Lin 𝔽 (nE + 4) := ⟨fun i => fOf l.a[i.val]!, fOf l.c⟩

/-- **The masked zerocheck proof** the replay reads, from the masked values by layout index. -/
def zcOf : ZerocheckProof where
  round1ab := (Array.range 64).map fun i => (msgAt v ((layOf Tz st).r1ab i)).val
  round1c := (Array.range 64).map fun i => (msgAt v ((layOf Tz st).r1c i)).val
  rounds := (Array.range (st.m - 6)).map fun j =>
    ((msgAt v ((layOf Tz st).ml j 0)).val, (msgAt v ((layOf Tz st).ml j 1)).val)
  finalA := (msgAt v (layOf Tz st).fa).val
  finalB := (msgAt v (layOf Tz st).fb).val
  finalC := (msgAt v (layOf Tz st).fc).val

/-- The masked lincheck rounds and `z_partial`. -/
def lcRoundsOf : Array (F128 × F128) := (Array.range (st.c.kLog - 6)).map fun j =>
  ((msgAt v ((layOf Tz st).lc j 0)).val, (msgAt v ((layOf Tz st).lc j 1)).val)
def zPartialOf : Array F128 := (Array.range 64).map fun i => (msgAt v ((layOf Tz st).zp i)).val

/-- **zerocheck c_eval**: `computedC + fc` (`ZerocheckMasked`, `Zk.replay`'s first constraint). -/
def rowCEval : ZkSession.Lin 𝔽 (nE + 4) :=
  ofZk ((ZerocheckMasked.computedC (layOf Tz st) (zcOf Tz st v) (zcF Tz st c r)).add
    (ZerocheckMasked.fcL (layOf Tz st) (zcOf Tz st v)))

/-- **zerocheck final a·b**: the running form after the `m − 6` residual rounds plus `prodL`. -/
def rowAB : ZkSession.Lin 𝔽 (nE + 4) :=
  ofZk ((ZerocheckMasked.running (layOf Tz st) (zcOf Tz st v) (rF Tz st c r) (rhoF Tz st c r) (zcF Tz st c r)
    (st.m - 6)).add (ZerocheckMasked.prodL (layOf Tz st) (zcOf Tz st v)))

/-- **lincheck final** (`Lincheck.lcFinal`) on the circuit's folded entries `combvOf`. -/
def rowLC : ZkSession.Lin 𝔽 (nE + 4) :=
  ofZk (Lincheck.lcFinal (layOf Tz st) (zcOf Tz st v).finalA (zcOf Tz st v).finalB (c.rest.lcAlpha r)
    (c.rest.lcBeta r) (lcRoundsOf Tz st v) (zPartialOf Tz st v) (tsF Tz st c r) (st.c.kLog - 6) (combvOf Tz st c r))

/-- Masked claim `k`'s ring-switch message in `ŝ` (`0` past the claims). -/
def sHatAt {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (k : ℕ) : Fin 128 → 𝔽 := if h : k < nC then sh ⟨k, h⟩ else 0

/-- `ŝ_k` as the executable's array. -/
def shArr {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (k : ℕ) : Array F128 := Array.ofFn fun w => (sHatAt sh k w).val

/-- The `ab` and `c` claims the replay hands the opening (`abcClaims`, values unread). -/
def abClaim : Flock.Claim := Lincheck.lcAbClaim st.m st.c.kLog (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r)
def cClaim : Flock.Claim := { value := 0, skip := skipWeights (zcF Tz st c r), x := rF Tz st c r }

/-- **ring switch ab**: the `ab` claim's value (`lcAbValue`) plus its claim check on `ŝ_{2r}`. -/
def rowRsAb {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : ZkSession.Lin 𝔽 (nE + 4) :=
  ofZk ((Lincheck.lcAbValue (layOf Tz st) (zPartialOf Tz st v) (c.rest.rPrime r)).addConst
    (Zk.claimCheck (abClaim Tz st c r) (shArr sh (2 * r.val))))

/-- **ring switch c**: the masked `final_c` plus the `c` claim's check on `ŝ_{2r+1}`. -/
def rowRsC {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : ZkSession.Lin 𝔽 (nE + 4) :=
  ofZk ((Zk.Lin.masked (zcOf Tz st v).finalC (layOf Tz st).fc (layOf Tz st).len).addConst
    (Zk.claimCheck (cClaim Tz st c r) (shArr sh (2 * r.val + 1))))

/-- Claim `k`'s ring-switch message in rep `r`'s opening: the masked `ab`, `c`, then the extras' public values. -/
def swAt {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (k : ℕ) : Fin 128 → 𝔽 :=
  if k = 0 then sHatAt sh (2 * r.val)
  else if k = 1 then sHatAt sh (2 * r.val + 1)
  else fun w => if h : k - 2 < 2 * st.regions.size then pubRegOf st c.rest.link ⟨k - 2, h⟩ w else 0

/-- **The opening's target** `T = Σ_k γ_k·β_k`, `β_k = recombine(ŝ_k, r''_k)` (`Opening.ringSwitch`). -/
def openT {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) : 𝔽 :=
  ∑ k : Fin (2 + 2 * st.regions.size), fOf (c.rest.rsGamma r k) *
    recombine execArith (swAt Tz st c r sh k.val) fun i => fOf (c.rest.rpp r k i)

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
    ![rowCEval Tz st c r v, rowAB Tz st c r v, rowLC Tz st c r v, rowRsAb Tz st c r v sh, rowRsC Tz st c r v sh,
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

/-- **The prover's table `z = (w, u)`**: the witness with the mask slot's words replaced by `u`. -/
def zWith (st : Stmt) {M nMask : ℕ} (w : Witness M) (u : Fin nMask → Fin 128 → ZMod 2) : Witness M := fun a =>
  match (List.finRange nMask).find? (fun x => (maskWordsOf st).getD x.val 0 = a.val / 128) with
  | some x => u x ⟨a.val % 128, Nat.mod_lt _ (by norm_num)⟩
  | none => w a

/-- Packed word `y` of a table, `f(y) = Σ_v z[128y + v]·x^v` (`Arith.pack`). -/
def wordOf {M : ℕ} (zw : Witness M) (y : ℕ) : 𝔽 :=
  execArith.pack fun v => if h : 128 * y + v.val < 2 ^ M then zw ⟨128 * y + v.val, h⟩ else 0

theorem six_le {mPts : ℕ} (Tz : TabZK 𝔽 mPts) : 6 ≤ Tz.S.m := by have := Tz.hm; omega

/-- **Level 0's out-of-domain point** `i`, read in the model's field: `ZkLigerito.ligerito`'s first `z ← sampleVec n`
(`n = m − 7`), the coin `RestZK.oodPt` (`0` past the layout's `nOod0` samples or the point's dimension). -/
def oodPoint {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) (i : ℕ) :
    Fin (Tz.S.m - 7) → 𝔽 := fun j =>
  if h : i < (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).nOod0 ∧ j.val < st.m - 7 then
    fOf (c.rest.oodPt r ⟨i, h.1⟩ ⟨j.val, h.2⟩)
  else 0

/-- **`oodPoint` is the drawn point**: coordinate `j` of out-of-domain sample `i` is the coin `z ← sampleVec (m − 7)`
that `ZkLigerito.ligerito` draws, embedded by `fOf` (the refinement `Refine.vecOf` reads every executable point by). -/
theorem oodPoint_eq {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
    (i : Fin (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).nOod0) (j : Fin (Tz.S.m - 7)) (hj : j.val < st.m - 7) :
    oodPoint Tz st c r i.val j = fOf (c.rest.oodPt r i ⟨j.val, hj⟩) := by
  simp only [oodPoint, dif_pos (And.intro i.isLt hj)]

/-- **A prover of Ligerito's levels ≥ 1**: rep `r`'s bytes after `T'`, as a function of what the simulator holds (the
draw's table and statement, the public coin vector `c`, whose level ≥ 1 coins `RestZK.ligZ`…`ligB` its sumcheck and
query messages read, and the restart `y₁'`) and the fresh prover coins `Hi`, the salts of levels ≥ 1's salted Merkle
trees, drawn independently of everything else. The zero-knowledge statements quantify over every such function.

Modelling claim (a trusted component, not proved here): the honest Ligerito prover's level-≥ 1 bytes are such a
function. -/
def LigProver : Type :=
  ∀ {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt), CoinsZK (RestOf Tz st) → Fin 2 → (L : ℕ) → (Fin L → 𝕂) → HiOf Tz →
    ByteArray

section Honest

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m)

/-- The zerocheck's tables at address `i`: the A-side, B-side and witness bits (`aVec`, `bVec`, `z`), `0` past `2^m`. -/
def aT (i : ℕ) : 𝔽 := if h : i < 2 ^ Tz.S.m then ((aVec Tz.S zw ⟨i, h⟩).val : 𝔽) else 0
def bT (i : ℕ) : 𝔽 := if h : i < 2 ^ Tz.S.m then ((bVec Tz.S zw ⟨i, h⟩).val : 𝔽) else 0
def cT (i : ℕ) : 𝔽 := if h : i < 2 ^ Tz.S.m then ((zw ⟨i, h⟩).val : 𝔽) else 0

/-- **The honest clear zerocheck** (`Zerocheck.honest` on the tables, at the drawn coins). -/
def zcH : ZerocheckProof :=
  Zerocheck.honest (Tz.S.m - 6) (aT Tz zw) (bT Tz zw) (cT Tz zw) (rF Tz st c r) (rhoF Tz st c r) (zcF Tz st c r)

/-- **The honest lincheck's round messages** (`Lincheck.lcMsg` on the circuit's and the witness's columns). -/
def lcH (j : ℕ) : 𝔽 × 𝔽 :=
  Lincheck.lcMsg (2 ^ Tz.S.kLog) (Lincheck.lcU Tz.S (zcF Tz st c r) (c.rest.lcAlpha r) (c.rest.lcBeta r) (rhoF Tz st c r))
    (Lincheck.lcV Tz.S (rhoF Tz st c r) zw) ((tsF Tz st c r).toList.map fOf) j

/-- **The honest `z_partial`**: the witness's columns folded at the round coins. -/
def zpH (i : ℕ) : 𝔽 := foldAll ((tsF Tz st c r).toList.map fOf) (2 ^ Tz.S.kLog) (Lincheck.lcV Tz.S (rhoF Tz st c r) zw) i

/-- Claim `k`'s word point in rep `r`'s opening, its point but the skip coordinates and `x₀`: `ab`, `c`, then the region
claims (`regionPtOf`). -/
def claimX (k : ℕ) : Fin (Tz.S.m - 7) → 𝔽 :=
  if k = 0 then fun j => abX Tz st c r (j.val + 1)
  else if k = 1 then fun j => zcReq Tz st c r (j.val + 1)
  else fun j => if h : k - 2 < 2 * st.regions.size then
    fOf ((regionPtOf st c.rest.link ⟨k - 2, h⟩).getD j.val 0) else 0

/-- **The opening's batched basis** `b(y) = Σ_k γ_k·Φ_k(eq(x_k[1 …], y))` (`Model.opening`). -/
def openB (y : ℕ) : 𝔽 :=
  ∑ k : Fin (2 + 2 * st.regions.size), fOf (c.rest.rsGamma r k) *
    phi execArith (fun i => fOf (c.rest.rpp r k i)) (eqAt (claimX Tz st c r k.val) y)

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
  let zc := zcH Tz st c r zw
  if k < 64 then fOf zc.round1ab[k]!
  else if k < 128 then fOf zc.round1c[k - 64]!
  else if k < lay.fa then
    (if (k - 128) % 2 = 0 then fOf zc.rounds[(k - 128) / 2]!.1 else fOf zc.rounds[(k - 128) / 2]!.2)
  else if k = lay.fa then fOf zc.finalA
  else if k = lay.fb then fOf zc.finalB
  else if k = lay.fc then fOf zc.finalC
  else if k < lay.zp 0 then
    (if (k - lay.fc - 1) % 2 = 0 then (lcH Tz st c r zw ((k - lay.fc - 1) / 2)).1
      else (lcH Tz st c r zw ((k - lay.fc - 1) / 2)).2)
  else if k < lay.zp 64 then zpH Tz st c r zw (k - lay.zp 0)
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
def innerOf (P : LigProver) {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (z : Sizes) (c : CoinsZK (RestOf Tz st)) :
    InnerFields (Witness Tz.S.m) (HiOf Tz) VLOf z where
  lanes w u l col := wordOf (zWith st w u) (l.val + z.nL * col.val)
  vals r w u R i := honestAt Tz st c r (zWith st w u) (R r) i.val
  lig r y hi := (∑ col : Fin z.L, y col * bFold Tz st c r col.val, P Tz st c r z.L y hi)
  lam := lamOf Tz st z c
  theta := thetaOf Tz st z c

end

end FlockSoundness.Discharge.Composed
