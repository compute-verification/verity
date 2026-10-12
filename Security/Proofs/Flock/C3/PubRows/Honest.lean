import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.Sparse.Session
import Proofs.Flock.C3.Opening.Algebra
import Proofs.Flock.C3.RegLevels.Chain
import Proofs.Flock.C3.Compose.Honest

/-!
# n1: the public rows, the tape and the commitments at the honest family

* `hPubRows_of_check` (stub): `HPubRows` at `C3Family.honest`, from the verifier's check of each accepted session's
  public file (`ChecksPubRows`: `RecVStar.checkPubRows` on each statement the session's record sets up, as
  `FlockVerify`'s `--rec-inner` path runs it after setup). `AcceptsZKR` does not make that check, and no premise of the
  guarantee says it was made (gap n1b).
* `hTape_honest` (stub): `HTape` at the honest family, from the stages' obligations there: where every stage plays on
  the prefix `decodeRun` takes for it, the cap, the link points, the stages and the openings are the whole tape.
* `hCommits_honest`: `VCommits` at the honest family, from `HPubRows`, the sessions' reads being `own`'s (`OwnIs`) and
  the rows the verifier holds being the history's commitments (`HeldIs`, the record–history link, gap n1c): G's
  `hCommits_of` after `pubCommits_of_pubRows`.
-/

namespace FlockC3.PubRows

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockVBridge.Family (skOf cmtOf mem_skOf_inl inr_not_mem_skOf vb_cs_inl)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **The verifier's check of a session's public rows**: `RecVStar.checkPubRows` passes on each statement the session's
record sets up, against the reads `own`, the inner transcript `cs` the verifier holds and its RecSparse shape. -/
def PubRowsChecked (cs : Option Flock.RecVStar.InnerCommits) (sparse : Option Flock.RecSparse.Shape) (I : Inputs)
    (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
    sts.forM (fun s => Flock.RecVStar.checkPubRows s (some own) cs sparse) = .ok ()

/-- **The verifier checks every accepted session's public rows** against its hold of the inner transcript
(`innerCommitsAt` at its own `tr`, `record` and `reg`). -/
def ChecksPubRows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (tr : Lean.Json) (record : String)
    (reg : Lean.Json) : Prop :=
  ∀ R ω t R₂ j tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr' →
    PubRowsChecked (innerCommitsAt tr record st reg).toOption F.ish.sparse (F.zs R ω t R₂ j).I
      (F.zs R ω t R₂ j).own tr'

/-- **Each accepted session's reads are `own`'s.** -/
def OwnIs (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (own : Fin (vstarCount F.ish) → Array Flock.Registered.Port) :
    Prop :=
  ∀ R ω t R₂ j, (∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr') → (F.zs R ω t R₂ j).own = own j

/-- **The rows the verifier holds are the history's commitments**: at every history `t` where session `j` accepts and
port `p` of `own j` registers at row `x` a value whose row holds round `i`'s message (`sel`), the row `F.row` holds there
is the `b ‖ c` (`bc`) of `t`'s round-`i` commitment. -/
def HeldIs (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (own : Fin (vstarCount F.ish) → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount F.ish)) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8) : Prop :=
  ∀ i R ω t (e : Cm × (holoInnerAt T CT st hp).Coin), t[i]? = some e → ∀ R₂ j,
    (∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr') →
    ∀ (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size), ∀ p ∈ own j, p.name = (F.Vs j).c.ports[q].name →
      ∀ x, p.reads[u.val]? = some (some x) → sel i p.value x → F.row p.value x t = some (bc j e.1 (.inl (u, q)))

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParses Flock.Tags.circuit ish bind Vs}
  {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).OuterR Reg'}
  {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {tr : Lean.Json} {record : String} {reg : Lean.Json}

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg

/-- **(n1) The public rows at the honest family**, from the verifier's check of them. A stub. -/
theorem hPubRows_of_check (hchk : ChecksPubRows 𝓗 tr record reg) : HPubRows 𝓗 := by
  sorry

/-- **(n1) The tape at the honest family**, from the stages' obligations there. A stub. -/
theorem hTape_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hrows : HPubRows 𝓗)
    (hreps : FlockC3.HoloValues.HReps 𝓗 dg N k Rw) (hsp : FlockC3.Sparse.HSparse 𝓗 dg N k Rw)
    (hop : FlockC3.Opening.HOpening 𝓗 dg N k Rw) (hlv : FlockC3.RegLevels.HRegLevels 𝓗 dg N k Rw) :
    HTape 𝓗 dg N k Rw := by
  sorry

/-- **The commitments at the honest family**: each accepted session's row of a stage's commit string holds, by
`HPubRows`, the row the verifier holds there, which is the history's commitment (`HeldIs`). -/
theorem hCommits_honest (hrows : HPubRows 𝓗) (hown : OwnIs 𝓗 own) (hheld : HeldIs 𝓗 own sel bc) :
    VCommits (𝓗).zs (𝓗).Sk (𝓗).cmt := by
  intro i R ω t e he R₂ j hacc pos hpos R'
  rcases pos with ⟨u, q⟩ | g
  · have hpos' : Sum.inl (u, q) ∈ skOf Vs own sel i j := hpos
    obtain ⟨p, hp, hname, x, hx, hsel⟩ := mem_skOf_inl.1 hpos'
    have hd := hrows R ω t R₂ j hacc p (by rw [hown R ω t R₂ j hacc]; exact hp) u q x hname.symm hx _
      (hheld i R ω t e he R₂ j hacc u q p hp hname x hx hsel)
    show _ = cmtOf Vs bc j e.1 (.inl (u, q))
    rw [vb_cs_inl, hd]
    simp [cmtOf]
    rfl
  · have hpos' : Sum.inr g ∈ skOf Vs own sel i j := hpos
    exact absurd hpos' (inr_not_mem_skOf g)

end FlockC3.PubRows
