import Proofs.Flock.Soundness.Assumptions.Recursive
import Proofs.Flock.Recursive.Statements

/-!
# `v-bridge` at the wires V* reads

`Flock.Assumptions.VBridge` reads `VDecodes` and `VFull` at the wires some unit reads (`VDecodesR`, `VFullR`, `ReadW`,
`Proofs.Flock.Recursive.Stage`), and `VCommits` at sessions that accept a transcript they write (`ZkOuter.Writes`):
V*'s message rows pad a port of `n` elements to `64 ⌈8 n / 64⌉` words and read its first `128 n` bits
(`InnerRepCheck.lean:24,39-44`), and a session that accepts nothing it writes may be another history's. `VBridgeRead` is that assumption at the family's arguments, so the
lanes' theorems state what `Flock.SecurityProofs.RecursiveRanRegistered` reads.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}

/-- **`v-bridge` at the read wires**: `Flock.Assumptions.VBridge`. -/
abbrev VBridgeRead (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg

end FlockVBridge.Family
