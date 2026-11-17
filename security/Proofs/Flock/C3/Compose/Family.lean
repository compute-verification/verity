module

public import Proofs.Flock.VBridge.Compiled
public import Proofs.Flock.Recursive.HoloExec
public import Proofs.Flock.Recursive.VStarCheck
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

/-! Candidate 3's family, `VStarFamily`'s analogue at candidate 3's table (`holoInnerAt`). -/

@[expose] public section

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled

structure C3Run (e : ExecHolo) where
  cap₀ : Cap Dig
  pts : Fin 2 → Fin e.cls.mPts → GF128
  ts : Fin 2 → List Step
  rep : Fin 2 → ZcOut GF128 (e.cls.pubStatement e.regs).m ×
    HoloLcOut GF128 (e.cls.pubStatement e.regs).m ((e.cls.pubStatement e.regs).kLog - 6)
  pend : Fin 2 → PendL GF128 Dig
  sp : Fin 2 → List Step
  claim : Fin 2 → RegClaim GF128 e.cls.shape
  op : List Step
  opened : OpenOut GF128
  lg : List Step
  pendR : PendL GF128 Dig
  os : Opens GF128 Dig × Salts (List UInt8)

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

structure C3Family (st : Flock.Stmt) (hp : HoloParams) (xR : HoloReg GF128 Dig) {n : ℕ} (L : SamplingDesign n)
    (Reg Cm Reg' Reg₂ : Type) where
  ish : RecVStar.Inner
  stmts : Array VStar.Stmt
  hstmts : RecVStar.statements ish = .ok stmts
  bind : Fin (vstarCount ish) → VStar.Bindings
  Vs : Fin (vstarCount ish) → VStmt
  hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs
  zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).Outer Reg'
  wmsg : ((j : Fin (vstarCount ish)) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg
  Sk : ℕ → (j : Fin (vstarCount ish)) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)
  cmt : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg
  msg : ℕ → ((j : Fin (vstarCount ish)) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
    (holoInnerAt T CT st hp).Msg
  run : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)
  row : String → ℕ → List (Cm × (holoInnerAt T CT st hp).Coin) → Option (List UInt8)

variable {T CT} {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n}
  {Reg Cm Reg' Reg₂ : Type}

namespace C3Family

variable (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)

def AtAccepted (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (P : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      Prop) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o →
    P R ω t (layers F.zs dg N k Rw R ω t R₂ τ)

end C3Family

end FlockC3.Compose
