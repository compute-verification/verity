module

public import Proofs.Flock.C3.Compose.LastRead
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
import Proofs.Flock.C3.Reps.OpensRead

/-! R's openings, read from the layers' read wires (`OpensReadAt 2`). -/

@[expose] public section

namespace FlockC3.RegLevels

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Family (maskOf)

local notation "𝔽" => FlockLevel3.GF128

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

def ROpensOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig)
    (prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof) (sch : Flock.Schedule) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    ∀ ℓ < sch.levels.size, ∀ q < (sch.levels.getD ℓ default).q,
      openRow (prR (layers zs dg N k Rw R ω t R₂ τ)) sch ℓ q = rd ℓ q (maskOf Vs W 2 (layers zs dg N k Rw R ω t R₂ τ))

/-- Rows past the ranges are pinned by the in-range ones, not constant: a row past the ends reads `openRow`'s
defaults, and a level past the schedule's reads the last level's (`openAt`). -/
def RRowsPinned (prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof) (sch : Flock.Schedule) : Prop :=
  ∀ Lb Lb', (∀ ℓ < sch.levels.size, ∀ q < (sch.levels.getD ℓ default).q,
      openRow (prR Lb) sch ℓ q = openRow (prR Lb') sch ℓ q) →
    ∀ ℓ q, openRow (prR Lb) sch ℓ q = openRow (prR Lb') sch ℓ q

theorem opensReadAt_two_of_layers
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} {sch : Flock.Schedule}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : ROpensOfLayers zs dg N k Rw W rd prR sch) (hpin : RRowsPinned prR sch)
    (hsrc : ∀ R ω t Lb ℓ q, (src R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) sch ℓ q) :
    OpensReadAt zs dg N k Rw src 2 := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  funext ℓ q
  rw [hsrc, hsrc]
  refine hpin _ _ (fun l hl j hj => ?_) ℓ q
  rw [hv R ω t R₂ τ o hA l hl j hj, hv R' ω' t' R₂' τ' o' hA' l hl j hj, FlockC3.Reps.maskOf_congr W 2 hX]

theorem opensReadAt_two_norm
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} {sch : Flock.Schedule}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : ROpensOfLayers zs dg N k Rw W rd prR sch) (hpin : RRowsPinned prR sch)
    (hsrc : ∀ R ω t Lb ℓ q, (stp R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) sch ℓ q) :
    OpensReadAt zs dg N k Rw (tiedSrc (normSrc stp)) 2 :=
  opensReadAt_two_of_layers hv hpin fun R ω t Lb ℓ q =>
    (congrFun (congrFun (normOs_os_of_lt (u := stp R ω t Lb) (by decide : 2 < 3)) ℓ) q).trans (hsrc R ω t Lb ℓ q)

end

end FlockC3.RegLevels
