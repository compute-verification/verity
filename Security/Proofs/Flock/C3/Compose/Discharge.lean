import Proofs.Flock.C3.Compose.Bridge
import Proofs.Flock.C3.PubRows.Honest
import Proofs.Flock.C3.HoloValues.Honest
import Proofs.Flock.C3.Sparse.Honest
import Proofs.Flock.C3.Opening.Honest
import Proofs.Flock.C3.RegLevels.Honest

/-!
# The guarantee's `_hbr` at the honest family

`hbr_honest`: the guarantee's `_hbr` at G's commit strings, digests and messages (`skOf`, `cmtOf`, `wmsgOf`, `msgOf`):
VBridge at every typed session, by `vbridge_c3` at `C3Family.honest`, from n1 (a stub, `C3/PubRows/`), the stages at
the honest family, `hCommits_honest`, `vDecodesR_wmsgOf` and `vFullR_of_read`. Its `Vs` and `hob` are the guarantee's:
at the verifier's list they are `vstarParses_of_list`'s (`VStarList/Statements.lean`, modulo `circuitFile_parses`),
which is `_hob`. The decoder `dec` is free, and a stage's messages are `dec`'s, so no stage follows from the public rows
alone (n2–n5 as stated in their pieces' files are false at some `dec`): the stages are hypotheses, each from its read
fact (n2–n5, `hX_of_read`) at a decoder that reads the sessions' values (n6). What it takes besides is the verifier's:
its list builds (`hss`), R's registered top is `xR`'s cap (`_htop`), it checks the accepted sessions' public rows
(`hchk`), their reads are its own (`hown`), and the rows it holds are the history's commitments (`hheld`).
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf cmtOf wmsgOf msgOf maskOf layerOf)
open FlockC3.HoloValues FlockC3.Sparse FlockC3.Opening FlockC3.RegLevels FlockC3.PubRows

/-- **`VDecodesR` at a decoder**: messages `wmsgOf`'s and `msgOf`'s at one decoder `dec` decode each stage's message
from the read wires of its commit strings, whatever they are (G's `vDecodesRead_of`, at any inner protocol). -/
theorem vDecodesR_wmsgOf {S : ℕ} {Vs : Fin S → VStmt} {I : Recursion.Inner}
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (dec : ℕ → ((j : Fin S) → (Vs j).Bits) → I.Msg) :
    VDecodesR (wmsgOf Vs Sk I.r dec) (fun _ _ a b => a = b) Sk (msgOf Vs Sk dec) := by
  intro i hi _ Xt v hX
  simp only [wmsgOf, List.getElem?_map, List.getElem?_range hi, Option.map_some]
  refine Option.Rel.some ?_
  show dec i (maskOf Vs Sk i Xt) = dec i (layerOf Vs Sk i v)
  refine congrArg (dec i) (funext fun j => funext fun g => ?_)
  simp only [maskOf, layerOf]
  split_ifs with h
  · exact hX j g h.1 h.2
  · rfl

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(step 7) The guarantee's `_hbr` at the honest family**: VBridge at every typed session of `zr`, at G's commit
strings, digests and messages, from n1, the verifier's checks and the stages at the honest family (`hreps`, `hsp`,
`hop`, `hlv`). The decoder `dec` is free, so the stages are hypotheses: each holds where `dec`'s messages on its rounds
are its prover's for a witness its verifier row accepts (its read fact, `hX_of_read`), which a fixed decoder and the
sessions' acceptance give (n6). R's top (`_htop`) is what n5's read fact reads. -/
theorem hbr_honest (ish : RecVStar.Inner) {ss : Array VStar.Stmt} (hss : RecVStar.statements ish = .ok ss)
    (bind : Fin (vstarCount ish) → VStar.Bindings) {Vs : Fin (vstarCount ish) → VStmt}
    (hob : VStarParses Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr)
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (tr : Lean.Json) (record : String) (reg : Lean.Json)
    (hreg : xR.Opens H512 enc512 leafPlain (holoAt st hp).rsch.sch.level₀)
    (_htop : ∃ lv, StackedTop lv st reg xR.cap) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (hreps : ∀ h : Typed zr,
      HReps (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hsp : ∀ h : Typed zr,
      HSparse (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hop : ∀ h : Typed zr,
      HOpening (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hlv : ∀ h : Typed zr,
      HRegLevels (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hchk : ∀ h : Typed zr, ChecksPubRows
      (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) tr record reg)
    (hown : ∀ h : Typed zr, OwnIs
      (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) own)
    (hheld : ∀ h : Typed zr, HeldIs
      (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) own sel bc) :
    ∀ h : Typed zr, Flock.Premises.VBridge (Flock.Guarantees.custodied zr hCust h) dg N k Rw
      (fun _ _ => Subtype.mk xR hreg) (wmsgOf Vs (skOf Vs own sel) (holoInnerAt T CT st hp).r dec) (skOf Vs own sel)
      (cmtOf Vs bc) (msgOf Vs (skOf Vs own sel) dec) := by
  intro h
  have hrows := hPubRows_of_check (hchk h)
  exact vbridge_c3 (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw hreg
    (hTape_honest hrows (hreps h) (hsp h) (hop h) (hlv h)) (hreps h) (hsp h) (hop h) (hlv h)
    (hCommits_honest hrows (hown h) (hheld h))
    (vDecodesR_wmsgOf (I := holoInnerAt T CT st hp) (skOf Vs own sel) dec) (vFullR_of_read _ _)

end FlockC3.Compose
