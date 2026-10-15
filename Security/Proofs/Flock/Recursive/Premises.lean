import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Soundness.Assumptions.Recursive
import Proofs.Flock.Soundness.Assumptions.Zk
import Proofs.Flock.Soundness.Assumptions.Inner
import Proofs.Flock.Soundness.Assumptions.Registrant

/-!
# The premises the recursive audit's property takes

The named premises a property of the recursive audit may take (the ruling of 8 Oct, 4:30 PM): `shape.premises` in
`Security/lean-audit.json` names this module, so the shape check reads a hypothesis whose head is one of these as a
premise and refuses every other. Each but `CompiledForkSound`, a proved lemma's conclusion, is the named assumption it
abbreviates, unchanged.

* `VBridge` (`Flock.Assumptions.VBridge`): V*'s circuits compute C-Flock's inner verifier. On the list until VBridge G
  (`FlockVBridge.Compiled.vbridge_compiled`'s open hypotheses) is proved.
* `VBridgeC` (`Flock.Assumptions.VBridgeC`): `VBridge` at the committed rounds, for an inner protocol whose last round
  is not committed; `VBridge` gives it (`Flock.Assumptions.vbridgeC_of_vbridge`).
* `VBridgeCOff` (`Flock.Assumptions.VBridgeCOff`): `VBridgeC` off one more bad event; `VBridgeC` gives it at every
  bad event (`Flock.Assumptions.vbridgeCOff_of_vbridgeC`).
* `CompiledForkSound` (the compile lemma in fork form): C-Flock's compiled table at the executable's arithmetic and the
  inner statement's Merkle scheme is sound in fork form at `tableError`, `Recursion.flock_inner_sound_compiled`'s
  conclusion. That lemma is proved, and discharges this premise at a class whose schedule is `fast100`'s and whose
  layout holds (`Recursion.compiledForkSound_of_fast100`).
* `UniformRandomBytes` (A3, `uniform/os-random`): the bytes the verifier's operating system returns are uniform, at
  each of V*'s laws (one premise over `Vs`, so the shape check reads its head).
* `HashDerivedKeyHm96` (`hash-derived-key`): hm96's salt hash keys its hiding at `2^-193`.
* `RecordCustody` (`live-verifier`, `Assumptions.Zk.RecordCustodyZKJ`): the record and the draw file each of V*'s
  sessions reads are the challenger's of the live session of its statement's `J` tables and the verifier's own draw, at every history (one premise
  over V*'s sessions as the prover supplies them, `ZkOuterR`, so the shape check reads its head; the property builds the
  typed sessions from them and it).
* `RegistrantCovers` (`registrant-holds-openings`, `Assumptions.RegistrantCovers`): the prover holds a row and salt
  under every leaf it registered at a row a session's reads file names, so at each typing of each of V*'s sessions as
  the prover supplies them every such row is a registered read of its record (one premise over `ZkOuterR`, as
  `RecordCustody`); a prover that registers a hash it can't open is outside the guarantee.
* `InnerRecordCustody` (`live-verifier` at the inner session, `Assumptions.InnerRecordCustody`): the inner transcript
  and session record the coin server writes at each history hold that history's commitments and coins where a map of
  the rounds says they go (one premise over the writer `wr`, as the verifier holds what it writes).

The list belongs in `Security/Definitions/` (the same ruling), whose modules import only Mathlib and the spec; C-Flock's
definitions, which these read, are still in the proofs package, so the list is here until they are extracted.
-/

namespace Flock.Premises

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

/-- **VBridge**: `Flock.Assumptions.VBridge`. -/
abbrev VBridge {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg

/-- **VBridge at the committed rounds**: `Flock.Assumptions.VBridgeC`. -/
abbrev VBridgeC {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  Flock.Assumptions.VBridgeC zs dg N k Rw x wmsg Sk cmt msg

/-- **VBridge at the committed rounds, off `Bad`**: `Flock.Assumptions.VBridgeCOff`. -/
abbrev VBridgeCOff {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  Flock.Assumptions.VBridgeCOff Bad zs dg N k Rw x wmsg Sk cmt msg

/-- **The compile lemma, in fork form**: `Recursion.flock_inner_sound_compiled`'s conclusion at the executable's
arithmetic (`execArith`) and the inner statement's Merkle scheme, SHA-512 with plain leaves (`Refine.enc512`,
`Refine.leafPlain` under `Refine.H512`), on the class `cls`, its regions and the schedule `sch`. -/
def CompiledForkSound {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    (cls : HClass) (regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog)))
    (sch : Accounting.Schedule) : Prop :=
  InnerSoundForkOn Refine.H512 (flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain cls regs sch)
    (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain cls regs sch)
    (ENNReal.ofReal (Accounting.tableError sch ⟨cls.kLog, regs.length, cls.mPts⟩))

/-- **A3** at every one of V*'s laws: `Assumptions.UniformRandomBytes` at each law's lengths and source. -/
abbrev UniformRandomBytes {S : ℕ} (Vs : Fin S → VStmt) : Prop :=
  ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src

/-- **hash-derived-key**: `Assumptions.Zk.HashDerivedKeyHm96`. -/
abbrev HashDerivedKeyHm96 : Prop := FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96

/-- **live-verifier** at every one of V*'s sessions: `Assumptions.Zk.RecordCustodyZKJ` at each session's inputs, draw
files and record writer of the live session of its statement's `J` tables, under the full draw of its units `[0, nv)`
(`--draw subset:<units>`). -/
abbrev RecordCustody {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') : Prop :=
  ∀ R ω t R₂ j, FlockSoundness.Assumptions.Zk.RecordCustodyZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj
    (fun _ _ => (Vs j).J) (zr R ω t R₂ j).wr (Vs j).nv

/-- **The registrant's record covers the rows its reads file names**, at every one of V*'s sessions:
`Assumptions.RegistrantCovers` at each session as the prover supplies it. -/
abbrev RegistrantCovers {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') : Prop :=
  ∀ R ω t R₂ j, Flock.Assumptions.RegistrantCovers (zr R ω t R₂ j)

/-- **live-verifier** at the inner session: `Assumptions.InnerRecordCustody`. -/
abbrev InnerRecordCustody {Cm Coin : Type} (wr : List (Cm × Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits) (sparse : Option Flock.RecSparse.Shape)
    (m : FlockSoundness.Assumptions.InnerRecordMap Cm Coin) : Prop :=
  FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m

end Flock.Premises
