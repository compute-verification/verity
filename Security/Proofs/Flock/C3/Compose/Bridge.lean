import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.Sparse.Session
import Proofs.Flock.C3.Opening.Algebra
import Proofs.Flock.C3.RegLevels.Chain
import Proofs.Flock.C3.PubRows.Prefix

/-!
# Piece (b): VBridge at candidate 3's table

`vbridge_c3`, `vbridge_of_family`'s analogue: the stages' obligations, the tape (`HTape`) and the commitments' and
decoding's (as G's `HCommits`, `HDecodes`, `VFullR`) give VBridge at `holoInnerAt`. Its acceptance part is
`holoTableCL_plays` on `HTape`'s steps; the other three are the hypotheses. `HPubRows` enters where the stages'
obligations are discharged at the honest decoder (the gap list's `C3Family.honest`), as `pubCommits_of_pubRows` does in
G, not here.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.HoloValues FlockC3.Sparse FlockC3.Opening FlockC3.RegLevels FlockC3.PubRows

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(b) VBridge at candidate 3's table**, from its family's stage obligations. -/
theorem vbridge_c3 (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (hreg : xR.Opens H512 enc512 leafPlain (holoAt st hp).rsch.sch.level₀)
    (htape : HTape F dg N k Rw) (hreps : HReps F dg N k Rw) (hsp : HSparse F dg N k Rw)
    (hop : HOpening F dg N k Rw) (hlv : HRegLevels F dg N k Rw)
    (hcm : VCommits F.zs F.Sk F.cmt) (hdec : VDecodesR F.wmsg (fun _ _ a b => a = b) F.Sk F.msg)
    (hfull : VFullR (holoInnerAt T CT st hp).r F.Sk) :
    Flock.Assumptions.VBridge (I := holoInnerAt T CT st hp) F.zs dg N k Rw (fun _ _ => Subtype.mk xR hreg) F.wmsg F.Sk
      F.cmt F.msg := by
  refine ⟨fun R ω t R₂ hc τ o h₀ => ?_, hcm, hdec, hfull⟩
  have ht := htape R ω t R₂ τ o ⟨h₀, hc⟩
  obtain ⟨hr₀, hv₀, hk₀⟩ := hreps R ω t R₂ τ o ⟨h₀, hc⟩ 0
  obtain ⟨hr₁, hv₁, hk₁⟩ := hreps R ω t R₂ τ o ⟨h₀, hc⟩ 1
  have hs₀ := hsp R ω t R₂ τ o ⟨h₀, hc⟩ 0
  have hs₁ := hsp R ω t R₂ τ o ⟨h₀, hc⟩ 1
  obtain ⟨hR, ho⟩ := hop R ω t R₂ τ o ⟨h₀, hc⟩
  obtain ⟨hl, hvR, hkR⟩ := hlv R ω t R₂ τ o ⟨h₀, hc⟩
  obtain ⟨out, hpl, hacc⟩ := holoTableCL_plays (capReg := xR.cap) (hreg := hR) (h₀ := hr₀) (h₁ := hr₁)
    (hs₀ := hs₀) (hs₁ := hs₁) (ho := ho) (hl := hl) (hv := ⟨hv₀, hv₁, hvR⟩) (hok := ⟨hk₀, hk₁, hkR⟩)
  exact ⟨out, (congrArg (fun l => PlaysT _ l out) ht).mpr hpl, hacc⟩

end FlockC3.Compose
