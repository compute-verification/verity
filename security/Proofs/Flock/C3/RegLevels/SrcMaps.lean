module

public import Proofs.Flock.C3.RegLevels.ROpensPR
public import Proofs.Flock.C3.Opening.SrcLayer
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.C3.Opening.ClaimTie
import Proofs.Flock.Level3.Assembly

/-!
# n14: R's levels at the tied source, from V*'s sessions

`levelTie_of_maps` ties R's levels at one run. At the tied source, at R's openings over `readMaps M` (whose rows are
`levelLanes` wide, `row_size_readMaps`), V*'s sessions give most of what it takes:

* `opening_rounds`: `opening_ok` at the coins of `stacked/rep0`, which `RecCoins.opening` runs at.
* `run_of_sessionAt`: at an accepted outer proof, R's run on the messages w holds, at R's claims at w (`claimsO`), from
  round 0 of `stacked/rep0`'s coins (`rep0`), has zero residuals (`openingReadAt_of_sessionAt`'s argument).
* `opsAt`, `MapsLeft`: the openings at R's maps, and what `levelTie_of_maps` takes there beyond the run, the claims'
  and ring switches' sizes, the opening's play and the layers good (`MapsLeft`):
  custody (the coins nonempty, the opening's steps' coins), R's openings at `readMaps`' rows (`OpensAtMaps`), the
  Ligerito steps' weave (`lgSteps_of_lig` makes it a play's tape) and the openings, the cap; and the claim tie, in
  `openingReadAt_src`'s form.
-/

@[expose] public section

namespace FlockC3.Opening

section Rounds

open Flock Flock.RecAlgebra

/-- **`opening_ok` at `stacked/rep0`'s coins**: the run `RecCoins.opening` accepted is from round 0 of them. -/
theorem opening_rounds {c3 : RecCoins.C3} {unit : String} {o : RecCoins.RegisteredOpening}
    (h : RecCoins.opening c3 unit = .ok o) :
    ∃ (c : Array (Array F128)) (a b : AClaim Lin) (lins : Array Lin) (k₁ : Nat) (res : Array Res) (bs : BSt),
      c3.coins unit s!"{RecVStar.STACKED}/rep0" = .ok c ∧
      coinClaim c3 unit o.sh 0 = .ok a ∧ coinClaim c3 unit o.sh 1 = .ok b ∧
      RecOpening.run o.sh (symbolic o.sh) #[a, b] 0 (RecAlgebra.RoundCoins.ofRounds c) = .ok (lins, k₁) ∧
      (RecOpening.buildAll o.sh lins.toList 0).run { kappa := kappa0 o.sh } = .ok (res, bs) ∧
      RecOpening.structureOf o.sh = .ok ⟨RecOpening.nw o.sh, bs.vNames.size, bs.ops, res⟩ ∧
      o.v = bs.vVals ∧ o.kappa = bs.kappa := by
  unfold RecCoins.opening at h
  cases hr : c3.stacked with
  | none => rw [hr] at h; cases h
  | some r =>
    rw [hr] at h
    dsimp only at h
    cases hsh : RecOpening.shape? r.m r.claims with
    | none => rw [hsh] at h; cases h
    | some sh =>
      rw [hsh] at h
      dsimp only at h
      split at h
      · cases h
      · obtain ⟨cs, hcs, h⟩ := FlockVBridge.PartS.bind_ok h
        obtain ⟨a, b, ha, hb, rfl⟩ := mapM_two hcs
        obtain ⟨rounds, hrd, h⟩ := FlockVBridge.PartS.bind_ok h
        have h := mapError_ok h
        obtain ⟨⟨lins, lg⟩, hrun, h⟩ := FlockVBridge.PartS.bind_ok h
        obtain ⟨⟨res, bs⟩, hbuild, h⟩ := FlockVBridge.PartS.bind_ok h
        split at h
        · cases h
        · rename_i hS
          split at h
          · cases h
          cases h
          obtain ⟨k₁, hx⟩ := run'_ok hrun
          exact ⟨rounds, a, b, lins, k₁, res, bs, hrd, ha, hb,
            run_of_inline sh (symbolic sh) #[a, b] 0 _ _ k₁ hx, hbuild, toOption_ok hS, rfl, rfl⟩

end Rounds

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Spot.Sampling Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

/-- The rounds of `stacked/rep0`'s coins at `c3`, where it has them. -/
def rep0 (c3 : RecCoins.C3) (unit : String) : Array (Array F128) :=
  match c3.coins unit s!"{RecVStar.STACKED}/rep0" with
  | .ok c => c
  | .error _ => #[]

theorem rep0_eq {c3 : RecCoins.C3} {unit : String} {c : Array (Array F128)}
    (h : c3.coins unit s!"{RecVStar.STACKED}/rep0" = .ok c) : rep0 c3 unit = c := by
  unfold rep0
  rw [h]

/-- **R's two claims at w**: the claims `coinClaim` builds, their values w's (`claimO`), where it builds them. -/
def claimsO (sh : Flock.Shape) (w : Array F128) (c3 : RecCoins.C3) (unit : String) : Array (RecAlgebra.AClaim F128) :=
  match coinClaim c3 unit sh 0, coinClaim c3 unit sh 1 with
  | .ok a, .ok b => #[claimO sh w ![a, b] 0, claimO sh w ![a, b] 1]
  | _, _ => #[]

theorem claimsO_eq {sh : Flock.Shape} {w : Array F128} {c3 : RecCoins.C3} {unit : String}
    {a b : RecAlgebra.AClaim RecAlgebra.Lin} (ha : coinClaim c3 unit sh 0 = .ok a)
    (hb : coinClaim c3 unit sh 1 = .ok b) : claimsO sh w c3 unit = #[claimO sh w ![a, b] 0, claimO sh w ![a, b] 1] := by
  unfold claimsO
  rw [ha, hb]

theorem decodedO_rs_size {sh : Flock.Shape} (w : Array F128) (cons : ℕ → ℕ → F128) {i : ℕ} (hi : i < sh.claims) :
    ((decodedO sh w cons).rs[i]!).size = 128 := by
  have hrs : (decodedO sh w cons).rs =
      (Array.range sh.claims).map fun k => (Array.range 128).map fun u => valO sh w cons (.rs k u) := by
    simp only [decodedO, RecAlgebra.Messages.map, RecAlgebra.symbolic, Array.map_map, Function.comp_def,
      RecAlgebra.Lin.eval_var]
  have hi' :
      i < ((Array.range sh.claims).map fun k => (Array.range 128).map fun u => valO sh w cons (.rs k u)).size := by
    simp only [Array.size_map, Array.size_range]
    exact hi
  rw [hrs, getElem!_pos _ i hi', Array.getElem_map, Array.getElem_range]
  simp only [Array.size_map, Array.size_range]

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **R's run in V*'s sessions** at one accepted outer proof (`openingReadAt_of_sessionAt`'s argument): on the messages
w holds, at R's claims at w, from round 0 of `stacked/rep0`'s coins, its residuals zero. -/
theorem run_of_sessionAt
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hacc : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) {sh : Flock.Shape}
    (hsh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) {unit : String} {c3 : RecCoins.C3}
    {w : Array F128} {ops : FlockVBridge.AccChain.Openings}
    (hs : SessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh unit c3 w ops) :
    ∃ (a b : RecAlgebra.AClaim RecAlgebra.Lin) (rs : Array F128) (k₁ : ℕ),
      coinClaim c3 unit sh 0 = .ok a ∧ coinClaim c3 unit sh 1 = .ok b ∧
      RecOpening.run sh (decodedO sh w (consO ops)) (claimsO sh w c3 unit) 0
        (RecAlgebra.RoundCoins.ofRounds (rep0 c3 unit)) = .ok (rs, k₁) ∧ ∀ x ∈ rs, x = 0 := by
  obtain ⟨op, hop, hsh', hw, ⟨Sg, gs, sess, un, cp, hS, hg, hparts⟩, ⟨acc, sessA, unA, cpA, hopen, hcoef, hrd⟩⟩ := hs
  obtain ⟨c, a, b, lins, k₁, res, bs, hc, ha, hb, hrun, hbuild, hSeq, hv, hk⟩ := opening_rounds hop
  rw [hsh'] at ha hb hrun hbuild hSeq
  rw [hk] at hcoef
  obtain rfl := Except.ok.inj (hS.symm.trans hSeq)
  have hinit : OpsOkO sh { kappa := kappa0 sh } := ⟨rfl, fun _ _ hk => by cases hk⟩
  have hB : RecOpening.buildAll sh lins.toList 0 { kappa := kappa0 sh } = .ok (res, bs) := hbuild
  have hvs := (buildAll_specO #[] (fun _ _ => 0) lins.toList 0 _ _ _ hinit hB).1.1
  obtain ⟨rs, hres, hzero⟩ := FlockVBridge.OfVStar.residuals_zero_of_cols (structureOf_bitsOkO hS) hg
    (w := w) (v := bs.vVals) hw hvs.symm
    fun p hp => by
      obtain ⟨_, S', hS', w', v', col, base, hw', hv', hcols, h0⟩ :=
        FlockVBridge.OfVStar.partCols_of_openingZeroAtL (zs R ω t R₂ (sess p)) (hacc.1 (sess p)).2 hS hg
          (hparts p hp)
      rw [hv] at hv'
      exact ⟨S', hS', w', v', col, base, hw', hv', hcols, h0⟩
  have hu := FlockVBridge.AccChain.unitSums_of_decode (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
    (fun s => (hacc.1 s).2) sh acc _ sessA unA cpA hopen
  obtain ⟨rs', hop', hres'⟩ := residuals_eq_runO (c := ![a, b]) w (consO ops) (shape_claims hsh)
    (Fin.forall_fin_two.mpr ⟨coinClaim_zero ha, coinClaim_zero hb⟩) hrun hbuild hw (hacc_of_units hbuild hu hcoef hrd)
  obtain rfl := Except.ok.inj (hres'.symm.trans hres)
  refine ⟨a, b, rs', k₁, ha, hb, ?_, hzero⟩
  rw [claimsO_eq ha hb, rep0_eq hc]
  exact hop'

end FlockC3.Opening

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Spot.Sampling Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Refine (capAt)
open Flock (sibsOf)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}
  {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **R's openings at the map `M`**: `StackedMaps.ops` at R's claims at w, values forgotten, from round 0 of
`stacked/rep0`'s coins. -/
def opsAt (M : FlockC3.Reps.StackedMaps stmt Vs) (sh : Flock.Shape) (Lb : (s : Fin S) → (Vs s).Bits)
    (kap : Array (Array (Array Flock.F128))) (c3 : Flock.RecCoins.C3) (unit : String) (w : Array Flock.F128) :
    FlockVBridge.AccChain.Openings :=
  M.ops sh Lb kap ((FlockC3.Opening.claimsO sh w c3 unit).map (Flock.RecAlgebra.AClaim.map fun _ => ())) 0
    (FlockC3.Opening.rep0 c3 unit)

/-- **What `levelTie_of_maps` leaves at one run**, in its form, at R's openings `opsAt` over `readMaps M`, R's claims
at w and `stacked/rep0`'s coins from round 0: the coins nonempty, the cap, the opening's steps' coins, R's openings at
the map (`OpensAtMaps`) with Ligerito's steps woven from R's proof's messages and the coins from the ring switches'
`k₂` (the source's definition, `ligOfVals`' form), and the openings. -/
def MapsLeft (M : FlockC3.Reps.StackedMaps stmt Vs) (zs : (s : Fin S) → (Vs s).Outer Reg')
    (Lb : (s : Fin S) → (Vs s).Bits) (sh : Flock.Shape) (unit : String) (c3 : Flock.RecCoins.C3)
    (w : Array Flock.F128) (kap : Array (Array (Array Flock.F128))) (e0 : ℕ → ByteArray) (sb : ℕ → ℕ → ℕ → ByteArray)
    (e : ℕ → ℕ → ByteArray) (cap₀ : Cap Dig) (u : C3Run (holoAt st hp)) : Prop :=
  let ops := opsAt (readMaps M sh.sched) sh Lb kap c3 unit w
  let a := FlockC3.Opening.claimsO sh w c3 unit
  let c := FlockC3.Opening.rep0 c3 unit
  Flock.RecAlgebra.NoEmpty c ∧ capDig (capR sh e0) = cap₀ ∧
    (∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a 0 (Flock.RecAlgebra.RoundCoins.ofRounds c) = .ok ((rsr, rs), k₂) →
      u.op.filterMap Sum.getRight? = ((List.range' 0 k₂).map fun i => (c)[i]!).map (coinArr fOf)) ∧
    (∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts) (k₁ : ℕ),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a 0 (Flock.RecAlgebra.RoundCoins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) rs.T rs.sws k₂ (Flock.RecAlgebra.RoundCoins.ofRounds c) =
        .ok ((lgr, lg), k₁) →
      OpensAtMaps (readMaps M sh.sched) zs Lb sh (sibsR sh ops sb e) (capR sh e0) (pR sh e0 w ops sb e) lg ∧
      u.lg = FlockSoundness.Discharge.Exec.weave (ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch)
        (ligMsgs (pR sh e0 w ops sb e) capDig sh.sched)
        (((List.range' k₂ (c.size - k₂)).map (fun i => (c)[i]!)).map (coinArr fOf))) ∧
    ∀ ℓ j, u.os.1 2 ℓ j = openRow (pR sh e0 w ops sb e) sh.sched ℓ j

end FlockC3.RegLevels
