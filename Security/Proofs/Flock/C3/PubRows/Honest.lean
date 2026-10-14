import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.Sparse.Session
import Proofs.Flock.C3.Opening.Algebra
import Proofs.Flock.C3.RegLevels.Chain
import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.C3.PubRows.Frame
import Proofs.Flock.C3.PubRows.Check

/-!
# n1: the public rows, the tape and the commitments at the honest family

* `hPubRows_of_check'`: `HPubRows` at `C3Family.honest`, from the verifier's check of each accepted session's public
  file (`ChecksPubRows`: `RecVStar.checkPubRows` on each statement the session's record sets up, as `FlockVerify`'s
  `--rec-inner` path runs it after setup; `AcceptsZKR` does not make it, gap n1b) where the check reads every row
  `HPubRows` does: the drawn units (`RowsHeld`), the first port of a name (`PortsNamed`) and a rec-sparse unit's rows
  at its own shape (`SparseAtShape`, `PubRows/Check.lean`).
* `hTape_of_frame`: `HTape` at the honest family from `HFrame`. The stages' obligations read only the decode, and no
  step after the openings changes it, so they don't give `HTape`: at the honest family it is exactly `HFrame`
  (`hTape_iff_frame`, `PubRows/Frame.lean`), the tape's cap and link points first and the openings alone after the
  stages, an obligation on the decoder `dec`.
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
(`innerCommitsAt` at the records `wr t` written at the history and its own `reg`). -/
def ChecksPubRows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json) : Prop :=
  ∀ R ω t R₂ j tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr' →
    PubRowsChecked (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption F.ish.sparse (F.zs R ω t R₂ j).I
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
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zs own sel bc dec wr reg

/-- **(n1) The tape at the honest family is its frame**: `decodeRun` takes each stage on a prefix of what is left
(`decodeRun_split`), so the tape is `holoSteps` of its decode exactly where the cap, the link points and the openings
frame the stages (`holoSteps_iff_framed`). -/
theorem hTape_iff_frame {dg : ByteArray → ByteArray} {N k Rw : ℕ} : HTape 𝓗 dg N k Rw ↔ HFrame 𝓗 dg N k Rw :=
  ⟨fun h R ω t R₂ τ o hacc => framed_of_holoSteps (e := holoAt st hp) (h R ω t R₂ τ o hacc),
    fun h R ω t R₂ τ o hacc => holoSteps_of_framed (e := holoAt st hp) (h R ω t R₂ τ o hacc)⟩

/-- **(n1) The tape at the honest family, from its frame**, an obligation on the decoder that n6 states at a fixed one. -/
theorem hTape_of_frame {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hfr : HFrame 𝓗 dg N k Rw) : HTape 𝓗 dg N k Rw :=
  hTape_iff_frame.2 hfr

/-- **(n1) The public rows at the honest family, from the verifier's check**, where it reads every row `HPubRows`
does: the accepted sessions draw and label the units reading a value `fixedRow` knows (`RowsHeld`), each circuit
names each port once (`PortsNamed`) and V*'s rec-sparse units are at the verifier's shape (`SparseAtShape`). -/
theorem hPubRows_of_check' (hchk : ChecksPubRows 𝓗 wr reg) (hdrawn : RowsHeld 𝓗) (hnames : PortsNamed 𝓗)
    (hsp : SparseAtShape 𝓗) : HPubRows 𝓗 := by
  intro R ω t R₂ j ⟨tr', hacc⟩ rg hrg u q x hname hread b hrow
  obtain ⟨cs, w, c, hcs, hc, hb⟩ :=
    heldRow_some (show heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) ish.sparse rg.value x = some b from hrow)
  have hfix : (RecVStar.fixedRow { reps := #[] } rg.value 0).isSome = true := by
    rw [fixedRow_isSome { reps := #[] } cs rg.value 0 x none ish.sparse]
    simp [hc]
  obtain ⟨d, D, dp, idx, hd, hD, hdp, hidx, hu, hlab⟩ := hdrawn R ω t R₂ j tr' hacc rg hrg hfix u x hread
  have hfind := port_find_of_unique (hnames j) q
  rw [hname] at hfind
  have hchk' := hchk R ω t R₂ j tr' hacc
  rw [hcs] at hchk'
  obtain ⟨w', c', hc', hdig⟩ := pubRow_of_checked ((𝓗).zs R ω t R₂ j) hacc (cs := cs) hchk' hrg hfix hfind hd hD
    hdp hidx hu (by rw [hlab]; exact hread)
  have he := (fixedRow_or (hsp j) hc).symm.trans hc'
  simp only [Option.some.injEq, Except.ok.injEq, Prod.mk.injEq] at he
  exact hdig.trans (he.2.symm.trans hb.symm)

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
