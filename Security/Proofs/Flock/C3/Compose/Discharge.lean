import Proofs.Flock.C3.Compose.Bridge
import Proofs.Flock.C3.PubRows.Honest
import Proofs.Flock.C3.Sparse.Read
import Proofs.Flock.C3.Opening.Read
import Proofs.Flock.C3.RegLevels.Read
import Proofs.Flock.C3.HoloValues.Read

/-!
# The guarantee's `_hbr` at the honest family

`hbr_honest`: the guarantee's `_hbr` at G's commit strings, digests and messages (`skOf`, `cmtOf`, `wmsgOf`, `msgOf`):
VBridge at every typed session, by `vbridge_c3` at `C3Family.honest`, from n1 (`hPubRows_of_check'`, `hTape_of_frame`),
the stages at the honest family, `hCommits_honest`, `vDecodesR_wmsgOf` and `vFullR_of_read`. Its `Vs` and `hob` are the
guarantee's: at the verifier's list they are `vstarParses_of_list`'s (`VStarList/Statements.lean`, modulo
`circuitFile_parses`), which is `_hob`. The decoder `dec` is free, and a stage's messages are `dec`'s, so no stage
follows from the public rows alone: each stage is its read fact (`RepsRead`, `SparseRead`, `OpeningRead`,
`RegLevelsRead`) and the tape its frame (`HFrame`), obligations on `dec` that n6 states at a
fixed decoder (`C3/Compose/Decoder.lean`). What it takes besides is the verifier's: its list builds (`hss`), R's
registered top is `xR`'s cap (`_htop`, which n6's `RegLevelsRead` reads), it checks the accepted sessions' public rows
(`hchk`) where they read every row `HPubRows` does (`hdrawn`, `hnames`, `hspsh`), their reads are its own (`hown`), and
the rows it holds are the history's commitments (`hheld`).
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
strings, digests and messages, from the verifier's checks, the stages' read facts and the tape's frame at the honest
family. The decoder `dec` is free, so the read facts are hypotheses: each says `dec`'s messages on its stage's rounds
are its prover's for a witness its verifier row accepts, which n6 states at a fixed decoder. R's top (`_htop`) is what
n6's `RegLevelsRead` reads. -/
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
    (hrread : ∀ h : Typed zr,
      RepsRead (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hsread : ∀ h : Typed zr,
      SparseRead (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (horead : ∀ h : Typed zr,
      OpeningRead (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hlread : ∀ h : Typed zr,
      RegLevelsRead (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hfr : ∀ h : Typed zr,
      HFrame (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw)
    (hchk : ∀ h : Typed zr, ChecksPubRows
      (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) tr record reg)
    (hdrawn : ∀ h : Typed zr,
      RowsHeld (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg))
    (hnames : ∀ h : Typed zr,
      PortsNamed (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg))
    (hspsh : ∀ h : Typed zr,
      SparseAtShape (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg))
    (hown : ∀ h : Typed zr, OwnIs
      (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) own)
    (hheld : ∀ h : Typed zr, HeldIs
      (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) own sel bc) :
    ∀ h : Typed zr, Flock.Premises.VBridge (Flock.Guarantees.custodied zr hCust h) dg N k Rw
      (fun _ _ => Subtype.mk xR hreg) (wmsgOf Vs (skOf Vs own sel) (holoInnerAt T CT st hp).r dec) (skOf Vs own sel)
      (cmtOf Vs bc) (msgOf Vs (skOf Vs own sel) dec) := by
  intro h
  have hrows := hPubRows_of_check' (hchk h) (hdrawn h) (hnames h) (hspsh h)
  exact vbridge_c3 (C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg) dg N k Rw hreg
    (hTape_of_frame (hfr h)) (hReps_honest_of_read (hrread h)) (hSparse_honest_of_read (hsread h))
    (hOpening_of_read (horead h)) (hRegLevels_honest_of_read (hlread h)) (hCommits_honest hrows (hown h) (hheld h))
    (vDecodesR_wmsgOf (I := holoInnerAt T CT st hp) (skOf Vs own sel) dec) (vFullR_of_read _ _)

end FlockC3.Compose
