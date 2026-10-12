import Proofs.Flock.C3.Compose.Bridge
import Proofs.Flock.C3.PubRows.Honest
import Proofs.Flock.C3.Sparse.Read
import Proofs.Flock.C3.Opening.Read
import Proofs.Flock.C3.RegLevels.Read
import Proofs.Flock.C3.HoloValues.Read
import Proofs.Flock.C3.PubRows.Shape

/-!
# The guarantee's `_hbr` at the honest family

`hbr_honest`: the guarantee's `_hbr` at G's commit strings, digests and messages (`skOf`, `cmtOf`, `wmsgOf`, `msgOf`,
the decoder at `skLeaf`): `VBridgeC` at V*'s checked sessions (`checkedC3`), by `vbridgeC_c3` at `C3Family.checked`,
from n1 (`hPubRows_of_checked` at the checks the checked sessions keep, `checksAccepted_checked`; `hTape_of_frame`),
the stages at the honest family, `hCommits_honest` at the committed rounds (`vCommits_masked`) and
`vDecodesR_wmsgLeaf`. Its `Vs` and `hob` are the guarantee's: at the verifier's
list they are `vstarParsesScoped_of_list`'s (`VStarList/Statements.lean`), which is `_hob`. The decoder `dec` is free,
and a stage's messages are `dec`'s, so no stage follows from the public rows alone: each stage is its read fact
(`RepsRead`, `SparseRead`, `OpeningRead`, `RegLevelsRead`) and the tape its frame (`HFrame`), obligations on `dec` that
n6 states at a fixed decoder (`C3/Compose/Decoder.lean`). What it takes besides is the verifier's: its list builds
(`hss`), R's registered top is `xR`'s cap (`_htop`, which n6's `RegLevelsRead` reads), the accepted sessions read every
row `HPubRows` does (`hdrawn`, `hnames`), their reads are its own (`hown`), and the rows it holds are the history's
commitments (`hheld`).
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

/-- **(step 7) The guarantee's `_hbr` at the honest family**: VBridge at the committed rounds (`VBridgeC`) at V*'s
checked sessions (`checkedC3`, the guarantee's), at G's commit strings and digests and the messages the decoder reads
(the last round's from the layers in full, `skLeaf`), from the stages' read facts, the tape's frame and the verifier's
facts at the checked family (`C3Family.checked`). A checked session keeps only what the verifier checks
(`checksAccepted_checked`), so the public rows are `hPubRows_of_checked`'s, at the shape every checked rec-sparse unit
has (`sparseAtShapeChecked`). The decoder `dec` is free, so the read facts are hypotheses: each says `dec`'s messages on
its stage's rounds are its prover's for a witness its verifier row accepts, which n6 states at a fixed decoder. R's top
(`_htop`) is what n6's `RegLevelsRead` reads. -/
theorem hbr_honest (ish : RecVStar.Inner) {ss : Array VStar.Stmt} (hss : RecVStar.statements ish = .ok ss)
    (bind : Fin (vstarCount ish) → VStar.Bindings) {Vs : Fin (vstarCount ish) → VStmt}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr) (coins : Option Flock.RecCoins.Source)
    (rows : Option (Array Flock.Registered.Port))
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (tr : Lean.Json) (record : String) (reg : Lean.Json)
    (hreg : xR.Opens H512 enc512 leafPlain (holoAt st hp).rsch.sch.level₀)
    (_htop : ∃ lv, StackedTop lv st reg xR.cap) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (hrread : ∀ h : Typed zr, RepsRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) dg N k Rw)
    (hsread : ∀ h : Typed zr, SparseRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) dg N k Rw)
    (horead : ∀ h : Typed zr, OpeningRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) dg N k Rw)
    (hlread : ∀ h : Typed zr, RegLevelsRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) dg N k Rw)
    (hfr : ∀ h : Typed zr, HFrame
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) dg N k Rw)
    (hdrawn : ∀ h : Typed zr, RowsHeld
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg))
    (hnames : ∀ h : Typed zr, PortsNamed
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg))
    (hown : ∀ h : Typed zr, OwnIs
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) own)
    (hheld : ∀ h : Typed zr, HeldIs
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) own sel bc) :
    ∀ h : Typed zr, Flock.Premises.VBridgeC
      (Flock.Guarantees.checkedC3 ish (fun _ => (innerCommitsAt tr record st reg).toOption) (fun _ => coins) bind rows
        zr hCust h) dg N k Rw
      (fun _ _ => Subtype.mk xR hreg)
      (wmsgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) (holoInnerAt T CT st hp).r dec) (skOf Vs own sel)
      (cmtOf Vs bc) (msgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) dec) := by
  intro h
  have hrows : HPubRows
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg) :=
    hPubRows_of_checked (checksAccepted_checked (coins := coins) (rows := rows) (zr := zr) (hCust := hCust) (h := h))
      (hdrawn h) (hnames h) (sparseAtShapeChecked _)
  exact vbridgeC_c3 (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec tr record reg)
    dg N k Rw hreg (hTape_of_frame (hfr h)) (hReps_honest_of_read (hrread h)) (hSparse_honest_of_read (hsread h))
    (hOpening_of_read (horead h)) (hRegLevels_honest_of_read (hlread h))
    (Flock.Assumptions.vCommits_masked (holoInnerAt T CT st hp).r (hCommits_honest hrows (hown h) (hheld h)))
    (vDecodesR_wmsgLeaf (I := holoInnerAt T CT st hp) (own := own) (sel := sel) dec)

end FlockC3.Compose
