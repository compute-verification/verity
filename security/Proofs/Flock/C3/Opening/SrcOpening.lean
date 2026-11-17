module

public import Proofs.Flock.C3.Opening.Session
public import Proofs.Flock.C3.Compose.Decoder
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

/-! `OpeningReadAt` at the source, from V*'s sessions. -/

@[expose] public section

namespace FlockC3.Opening

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

theorem afterSparse_runSteps {e : ExecHolo} (u : C3Run e) :
    afterSparse u (runSteps u) = u.op ++ (u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩]) := by
  rw [afterSparse, runSteps, FlockC3.PubRows.holoSteps_eq, ← FlockC3.PubRows.firstSteps_length]
  simp only [List.drop_succ_cons, List.drop_zero, FlockC3.PubRows.stagesOf, List.append_assoc, ← List.drop_drop,
    List.drop_left]

/-- The source's opening is on w's ring switches; `SrcOK` does not give it: it fixes the source's kinds, coins
and types, not its messages' values. -/
def OpenTie (sh : Flock.Shape) (w : Array F128) (ops : FlockVBridge.AccChain.Openings) {e : ExecHolo} (u : C3Run e) :
    Prop :=
  ∃ (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
    u.op = openSteps (fun i => FlockSoundness.Refine.vecOf 128 (decodedO sh w (consO ops)).rs[i.val]!) ρ γ

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

theorem openingReadAt_of_sessionAt
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hacc : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) {sh : Flock.Shape}
    (hsh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) {unit : String} {c3 : RecCoins.C3}
    {w : Array F128} {ops : FlockVBridge.AccChain.Openings}
    (hs : SessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh unit c3 w ops)
    {l : List Step} {u : C3Run (holoAt st hp)}
    (hclaim : ∀ a b, coinClaim c3 unit sh 0 = .ok a → coinClaim c3 unit sh 1 = .ok b →
      ∀ i, FlockSoundness.Refine.ClaimRel (holoAt st hp).cls.shape.mReg (claimO sh w ![a, b] i).toClaim
        (u.claim i).toClaim)
    (hpre : ∃ (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
      openSteps (fun i => FlockSoundness.Refine.vecOf 128 (decodedO sh w (consO ops)).rs[i.val]!) ρ γ <+:
        afterSparse u l) :
    OpeningReadAt l u := by
  obtain ⟨op, hop, hsh', hw, ⟨Sg, gs, sess, un, cp, hS, hg, hparts⟩, ⟨acc, sessA, unA, cpA, hopen, hcoef, hrd⟩⟩ := hs
  obtain ⟨_, a, b, src, lins, k₁, res, bs, _, _, ha, hb, hrun, hbuild, hSeq, hv, hk⟩ := opening_ok hop
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
  have e : rs' = rs := Except.ok.inj (hres'.symm.trans hres)
  rw [e] at hop'
  obtain ⟨ρ, γ, hpre⟩ := hpre
  exact ⟨sh, decodedO sh w (consO ops), claimO sh w ![a, b], src, 0, k₁, rs, ρ, γ, hsh, hop', hzero,
    hclaim a b ha hb, hpre⟩

theorem openingReadAt_src
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} (hsh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) (unit : String)
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockVBridge.AccChain.Openings)
    (hsess : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      SessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh unit (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (hclaim : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ∀ a b, coinClaim (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit sh 0 = .ok a →
        coinClaim (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit sh 1 = .ok b →
        ∀ i, FlockSoundness.Refine.ClaimRel (holoAt st hp).cls.shape.mReg
          (claimO sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) ![a, b] i).toClaim
          ((src R ω t (layers zs dg N k Rw R ω t R₂ τ)).claim i).toClaim)
    (htie : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      OpenTie sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ))) :
    AtSrc zs dg N k Rw src fun _ u => OpeningReadAt (runSteps u) u := by
  intro R ω t R₂ τ o hacc
  obtain ⟨ρ, γ, he⟩ := htie R ω t R₂ τ o hacc
  refine openingReadAt_of_sessionAt hacc hsh (hsess R ω t R₂ τ o hacc) (hclaim R ω t R₂ τ o hacc) ⟨ρ, γ, ?_⟩
  rw [afterSparse_runSteps, he]
  exact List.prefix_append _ _

end FlockC3.Opening
