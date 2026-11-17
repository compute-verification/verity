module

public import Proofs.Flock.Soundness.Assumptions.Recursive
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

/-! `Flock.Assumptions.VBridge` at the family's arguments (`VBridgeRead`), read at the wires some unit reads because
V*'s message rows pad each port past the `128 n` bits it reads. -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}

abbrev VBridgeRead (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg

end FlockVBridge.Family
