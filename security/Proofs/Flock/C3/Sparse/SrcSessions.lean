module

public import Proofs.Flock.C3.Compose.Tied
public import Proofs.Flock.C3.Sparse.SessionAt
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

/-! The sparse tie at the tied source, from V*'s sessions. -/

@[expose] public section

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Sparse (SCoins)

def SparseRepTie (e : ExecHolo) (u : C3Run e) (i : Fin 2) (w : Array Flock.F128) (c : SCoins) : Prop :=
  FlockVBridge.Sparse.pubOf e.cls.shape c =
      holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2 ∧
    ∀ s : Fin 64, fOf w[s.val]! = (u.rep i).2.h s

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

theorem srcSparseTie_of_sessions
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3) (unit : String)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins)
    (h : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
          (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) ∧
        SparseSrcAt (holoAt st hp) (src R ω t (layers zs dg N k Rw R ω t R₂ τ)) i
          (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i)) :
    SrcSparseTie zs dg N k Rw src := by
  intro R ω t R₂ τ o hA i
  obtain ⟨hs, hsrc⟩ := h R ω t R₂ τ o hA i
  exact sparseTie_of_session (fun s => (hA.1 s).2) hs hsrc

def sparseOf (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp) := fun R ω t Lb =>
  { stp R ω t Lb with sp := fun i => stepsOf (holoAt st hp).cls.shape (w R ω t Lb i) (c R ω t Lb i).cs }

theorem srcSparseTie_sparseOf
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3) (unit : String)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins)
    (h : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
          (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) ∧
        SparseRepTie (holoAt st hp) (tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)) i
          (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i)) :
    SrcSparseTie zs dg N k Rw (tiedSrc (sparseOf stp w c)) := by
  refine srcSparseTie_of_sessions c3 unit w c fun R ω t R₂ τ o hA i => ?_
  obtain ⟨hs, hpub, hh⟩ := h R ω t R₂ τ o hA i
  exact ⟨hs, hpub, hh, rfl⟩

end

end FlockC3.Sparse
