module

public import Proofs.Flock.C3.Reps.StepsTie
public import Proofs.Flock.C3.Reps.Vals
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

/-! The reps' openings, read from the layers' read wires. -/

@[expose] public section

namespace FlockC3.Reps

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Family (maskOf)

local notation "𝔽" => FlockLevel3.GF128

section mask

variable {S : ℕ} {Vs : Fin S → VStmt}

theorem maskOf_congr (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) (i : ℕ)
    {Xs Xs' : (j : Fin S) → (Vs j).Bits} (h : ∀ j g, ReadW Vs j g → Xs j g = Xs' j g) :
    maskOf Vs Sk i Xs = maskOf Vs Sk i Xs' :=
  funext fun j => funext fun g => by
    simp only [maskOf]
    split_ifs with hg
    · exact h j g hg.2
    · rfl

theorem readAgree_of_skLeaf {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {r i : ℕ}
    (hi : ¬ i + 1 < r) {Xs Xs' : (j : Fin S) → (Vs j).Bits}
    (h : maskOf Vs (skLeaf Vs own sel r) i Xs = maskOf Vs (skLeaf Vs own sel r) i Xs') :
    ∀ j g, ReadW Vs j g → Xs j g = Xs' j g := fun j g hg => by
  simpa [maskOf, skLeaf, hi, hg] using congrFun (congrFun h j) g

end mask

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

def OpensOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ (i : Fin 2) ℓ q,
    openRow (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) i).p.ligerito
        (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) i).sch ℓ q =
      rd i ℓ q (maskOf Vs W i.val (layers zs dg N k Rw R ω t R₂ τ))

def SrcOpensOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig)
    (b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ (i : Fin 2) ℓ q,
    openRow (b (layers zs dg N k Rw R ω t R₂ τ) i).lig (b (layers zs dg N k Rw R ω t R₂ τ) i).sch ℓ q =
      rd i ℓ q (maskOf Vs W i.val (layers zs dg N k Rw R ω t R₂ τ))

theorem openRow_vals (b : ValsSrc) (sh : Flock.Shape) (d : RepData) (ℓ q : ℕ) :
    openRow (b.vals sh d).p.ligerito (b.vals sh d).sch ℓ q = openRow b.lig b.sch ℓ q :=
  rfl

theorem opensOfLayers_vals {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig} {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (h : SrcOpensOfLayers zs dg N k Rw W rd b) :
    OpensOfLayers zs dg N k Rw W rd fun R ω t Lb i => (b Lb i).vals sh (rep R ω t Lb i) :=
  fun R ω t R₂ τ o hA i ℓ q => (openRow_vals (b _ i) sh (rep _ _ _ _ i) ℓ q).trans (h R ω t R₂ τ o hA i ℓ q)

theorem opensReadAt_of_layers {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : OpensOfLayers zs dg N k Rw W rd vals)
    (hsrc : ∀ R ω t Lb (i : Fin 2) ℓ q,
      (src R ω t Lb).os.1 i ℓ q = openRow (vals R ω t Lb i).p.ligerito (vals R ω t Lb i).sch ℓ q)
    (i : Fin 2) : OpensReadAt zs dg N k Rw src i := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  funext ℓ q
  rw [hsrc, hsrc, hv R ω t R₂ τ o hA i ℓ q, hv R' ω' t' R₂' τ' o' hA' i ℓ q, maskOf_congr W i.val hX]

theorem opensReadAt_repsOfVals {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {nm : C3Run (holoAt st hp) → C3Run (holoAt st hp)}
    (hnm : ∀ u (i : Fin 2), (nm u).os.1 i = u.os.1 i) (hv : OpensOfLayers zs dg N k Rw W rd vals) (i : Fin 2) :
    OpensReadAt zs dg N k Rw (tiedSrc fun R ω t Lb => nm (repsOfVals stp sh rep vals R ω t Lb)) i :=
  opensReadAt_of_layers hv (fun R ω t Lb i ℓ q =>
    (congrFun (congrFun (hnm (repsOfVals stp sh rep vals R ω t Lb) i) ℓ) q).trans (dite_eq_left i.isLt)) i

theorem opensReadAt_src {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {nm : C3Run (holoAt st hp) → C3Run (holoAt st hp)} (hnm : ∀ u (i : Fin 2), (nm u).os.1 i = u.os.1 i)
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} (hb : SrcOpensOfLayers zs dg N k Rw W rd b) (i : Fin 2) :
    OpensReadAt zs dg N k Rw
      (tiedSrc fun R ω t Lb => nm (repsOfVals stp sh rep (fun R ω t Lb i => (b Lb i).vals sh (rep R ω t Lb i)) R ω t
        Lb)) i :=
  opensReadAt_repsOfVals hnm (opensOfLayers_vals hb) i

end

end FlockC3.Reps
