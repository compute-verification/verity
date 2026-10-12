import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.VBridge.Compiled
import Proofs.Flock.C3.Compose.Tape

/-!
# Piece (b): candidate 3's family

`VStarFamily`'s analogue at candidate 3's table (`holoInnerAt`), sharing its layout-free files (`Accepted`, `layers`,
the sessions' `Outer` and their value layers): V*'s statements over the verifier's own list (`RecVStar.statements`, gap
4(c)), their sessions, R's registration, and the inner run as V*'s layers hold it, stage by stage (`C3Run`, a decoder
as `VStarFamily.files` and `rep₀` are). Each stage's obligation is its piece's statement, in its piece's directory:
`HPubRows` and `HTape` (g), `HReps` (c), `HSparse` (d), `HOpening` (e), `HRegLevels` (f); `vbridge_c3`
(`Compose/Bridge.lean`) composes them into VBridge.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled

/-- **Candidate 3's inner run, stage by stage**, at the parameters `e`: the level-0 cap, the link points, each rep's
steps and output, each rep's sparse steps and registered claim, R's opening's steps and output, Ligerito on R's steps
and result, and the openings with their salts. -/
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

/-- **Candidate 3's family** at the inner statement `st`, the sparse parameters `hp` and R's registration `xR`: V*'s
statements over the verifier's list at the inner shape `ish`, their bindings, circuits and sessions (each session's
file is its statement's, `hfile`, with one instance per unit, `hnv`), the messages V*'s layers hold and the rounds'
commit strings, the inner run each history's layers decode to (`run`), and the `b ‖ c` the inner transcript holds at
each registered value's row (`row`, by `RecVStar.fixedRow`'s prefixes: a rep's round, R's stream's round, a node of
R's level top, a sparse stream's round). -/
structure C3Family (st : Flock.Stmt) (hp : HoloParams) (xR : HoloReg GF128 Dig) {n : ℕ} (L : Law n)
    (Reg Cm Reg' Reg₂ : Type) where
  ish : RecVStar.Inner
  stmts : Array VStar.Stmt
  hstmts : RecVStar.statements ish = .ok stmts
  bind : Fin (vstarCount ish) → VStar.Bindings
  Vs : Fin (vstarCount ish) → VStmt
  zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).Outer Reg'
  hfile : ∀ R ω t R₂ (j : Fin (vstarCount ish)) s, stmts[(j : ℕ)]? = some s →
    RecVStar.circuitFile s (bind j) = .ok (zs R ω t R₂ j).I.circuitFile
  hnv : ∀ (j : Fin (vstarCount ish)) s, stmts[(j : ℕ)]? = some s → (Vs j).nv = s.instances
  wmsg : ((j : Fin (vstarCount ish)) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg
  Sk : ℕ → (j : Fin (vstarCount ish)) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)
  cmt : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg
  msg : ℕ → ((j : Fin (vstarCount ish)) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
    (holoInnerAt T CT st hp).Msg
  run : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)
  row : String → ℕ → List (Cm × (holoInnerAt T CT st hp).Coin) → Option (List UInt8)

variable {T CT} {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n}
  {Reg Cm Reg' Reg₂ : Type}

namespace C3Family

variable (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)

/-- **At every accepted outcome**: `P` holds at each history whose sessions V*'s verifier accepts with good layers and
no registrant collision (`Accepted`), at the sessions' layers. -/
def AtAccepted (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (P : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      Prop) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o →
    P R ω t (layers F.zs dg N k Rw R ω t R₂ τ)

end C3Family

end FlockC3.Compose
