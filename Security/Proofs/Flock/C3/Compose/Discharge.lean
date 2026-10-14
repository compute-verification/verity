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
from n1 (`hPubRows_of_checked` at the checks of a transcript each session writes, `hreach`; `hTape_of_frame`),
the stages at the honest family, `hCommits_honest` at the committed rounds (`vCommits_masked`) and
`vDecodesR_wmsgLeaf`. Its `Vs` and `hob` are the guarantee's: at the verifier's
list they are `vstarParsesScoped_of_list`'s (`VStarList/Statements.lean`), which is `_hob`. The decoder `dec` is free,
and a stage's messages are `dec`'s, so no stage follows from the public rows alone: each stage is its read fact
(`RepsRead`, `SparseRead`, `OpeningRead`, `RegLevelsRead`) and the tape its frame (`HFrame`), obligations on `dec` that
n6 states at a fixed decoder (`C3/Compose/Decoder.lean`). What it takes besides is the verifier's: its list builds
(`hss`), R's registered top is `xR`'s cap (`_htop`, which n6's `RegLevelsRead` reads), a session accepting a
transcript it writes passes its checks at one it writes (`hreach`, `ChecksReached`, where every unit is drawn,
`rowsHeld`, so the check reads every row `HPubRows` does; each circuit names a port once, `PortsNamed`, from the
statement check's refusal, `names_of_checkStmt`), their reads hold its own (`hown`, `OwnIn`; at
`own := fun j => ownAt ss j` it is `ownIn_checked`, from `checkOwn`), and the rows it holds, at the records written at
the history (`wr`), are the history's commitments (`hheld`).

`hbr_honest` is at a fixed word of R's cap (`fun _ _ => xR`); `hbr_honest_reg` is `_hbr` in its form, at the word the
transported prover's runs open (`regStmt`), off `ValueColl` at the verifier's rows: candidate 3's table reads only the
cap (`holoAccepts_of_cap`), so VBridgeC moves between statements of one cap (`vbridgeC_of_cap`), and VBridgeC gives
VBridgeCOff at every bad event.
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
facts at the checked family (`C3Family.checked`). The public rows are `hPubRows_of_checked`'s at the checks of a
transcript each session writes (`hreach`), at the shape every checked rec-sparse unit has (`sparseAtShapeChecked`). A
checked session keeps its list where some transcript it accepts passes the checks, written or not
(`checksAccepted_checked`), so `hreach` is not yet the checked family's. The decoder `dec` is free, so the read facts
are hypotheses: each says `dec`'s messages on its stage's rounds are its prover's for a witness its verifier row
accepts, which n6 states at a fixed decoder. R's top (`_htop`) is what n6's `RegLevelsRead` reads. -/
theorem hbr_honest (ish : RecVStar.Inner) {ss : Array VStar.Stmt} (hss : RecVStar.statements ish = .ok ss)
    (bind : Fin (vstarCount ish) → VStar.Bindings) {Vs : Fin (vstarCount ish) → VStmt}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr) (coins : String → Option Flock.RecCoins.Source)
    (rows : Option (Array Flock.Registered.Port))
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json)
    (bcs' : Array ByteArray) (_htop : ∃ lv salt, StackedTopAt lv st reg xR.cap salt bcs')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (hrread : ∀ h : Typed zr, RepsRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hsread : ∀ h : Typed zr, SparseRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (horead : ∀ h : Typed zr, OpeningRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hlread : ∀ h : Typed zr, RegLevelsRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hfr : ∀ h : Typed zr, HFrame
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hreach : ∀ h : Typed zr, ChecksReached
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) rows wr reg)
    (hown : ∀ h : Typed zr, OwnIn
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) own)
    (hheld : ∀ h : Typed zr, HeldIs
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) own sel bc) :
    ∀ h : Typed zr, Flock.Premises.VBridgeC
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows
        zr hCust h) dg N k Rw
      (fun _ _ => xR)
      (wmsgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) (holoInnerAt T CT st hp).r dec) (skOf Vs own sel)
      (cmtOf Vs bc) (msgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) dec) := by
  intro h
  have hrows : HPubRows
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) :=
    hPubRows_of_checked (hreach h) (sparseAtShapeChecked _)
  exact vbridgeC_c3 (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
    dg N k Rw (hTape_of_frame (hfr h)) (hReps_honest_of_read (hrread h)) (hSparse_honest_of_read (hsread h))
    (hOpening_of_read (horead h)) (hRegLevels_honest_of_read (hlread h))
    (Flock.Assumptions.vCommits_masked (holoInnerAt T CT st hp).r (hCommits_honest hrows (hown h) (hheld h)))
    (vDecodesR_wmsgLeaf (I := holoInnerAt T CT st hp) (own := own) (sel := sel) dec)

/-- Candidate 3's inner protocol reads its statement only through the cap (`holoInnerCL`'s table), so it accepts at
one statement what it accepts at another of the same cap. -/
theorem holoAccepts_of_cap {x x' : HoloReg GF128 Dig} (hx : x.cap = x'.cap)
    {tr : List ((holoInnerAt T CT st hp).Msg × (holoInnerAt T CT st hp).Coin)}
    (ha : (holoInnerAt T CT st hp).accepts x tr) : (holoInnerAt T CT st hp).accepts x' tr := by
  obtain ⟨w, c⟩ := x
  obtain ⟨w', c'⟩ := x'
  obtain rfl : c = c' := hx
  exact ha

/-- **`VBridgeC` moves between statements of one cap** at candidate 3's inner protocol: it reads the statement only
through the inner protocol's acceptance (`VBridgeOff`), which reads only the cap (`holoAccepts_of_cap`). -/
theorem vbridgeC_of_cap {S : ℕ} {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {x x' : Reg → L.Ω → HoloReg GF128 Dig}
    (hx : ∀ R ω, (x R ω).cap = (x' R ω).cap)
    {wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (holoInnerAt T CT st hp).Msg}
    (hb : Flock.Premises.VBridgeC zs dg N k Rw x wmsg Sk cmt msg) :
    Flock.Premises.VBridgeC zs dg N k Rw x' wmsg Sk cmt msg := by
  obtain ⟨hoff, hcm, hdec, hfull⟩ := hb
  exact ⟨fun R ω t R₂ hbad τ o ho => holoAccepts_of_cap (hx R ω) (hoff R ω t R₂ hbad τ o ho), hcm, hdec, hfull⟩

/-- **(step 7) The guarantee's `_hbr` at the honest family, in its form**: `hbr_honest` at the word the transported
prover's runs open at R's level 0 against `cap` (`regStmt`), whose cap is `cap`, which is `xR`'s (`hcap`), off the
verifier's values' collisions (`ValueColl` at `valueSrcC3`), which VBridgeC gives (`vbridgeCOff_of_vbridgeC`). -/
theorem hbr_honest_reg (ish : RecVStar.Inner) {ss : Array VStar.Stmt} (hss : RecVStar.statements ish = .ok ss)
    (bind : Fin (vstarCount ish) → VStar.Bindings) {Vs : Fin (vstarCount ish) → VStmt}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    (zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg')
    (hCust : Flock.Premises.RecordCustody zr) (coins : String → Option Flock.RecCoins.Source)
    (rows : Option (Array Flock.Registered.Port))
    (own : Fin (vstarCount ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json)
    (cap : Cap Dig) (hcap : xR.cap = cap)
    (bcs' : Array ByteArray) (htop : ∃ lv salt, StackedTopAt lv st reg xR.cap salt bcs')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerVR (zr R ω t) dg N))
    (hrread : ∀ h : Typed zr, RepsRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hsread : ∀ h : Typed zr, SparseRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (horead : ∀ h : Typed zr, OpeningRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hlread : ∀ h : Typed zr, RegLevelsRead
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hfr : ∀ h : Typed zr, HFrame
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw)
    (hreach : ∀ h : Typed zr, ChecksReached
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) rows wr reg)
    (hown : ∀ h : Typed zr, OwnIn
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) own)
    (hheld : ∀ h : Typed zr, HeldIs
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) own sel bc) :
    ∀ h : Typed zr, Flock.Premises.VBridgeCOff
      (ValueColl (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind
        rows zr hCust h) (valueSrcC3 fun t => Flock.Guarantees.c3Of ish (Flock.Guarantees.innerHeld wr st reg t)))
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows
        zr hCust h) dg N k Rw
      (Flock.Guarantees.regStmt T CT st hp
        (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr
          hCust h) dg N k Rw
        (wmsgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) (holoInnerAt T CT st hp).r dec)
        (Flock.Guarantees.checkedT ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr
          hCust h σ) cap)
      (wmsgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) (holoInnerAt T CT st hp).r dec) (skOf Vs own sel)
      (cmtOf Vs bc) (msgOf Vs (skLeaf Vs own sel (holoInnerAt T CT st hp).r) dec) := by
  intro h
  refine Flock.Assumptions.vbridgeCOff_of_vbridgeC _ (vbridgeC_of_cap (x := fun _ _ => xR) (fun _ _ => ?_)
    (hbr_honest ish hss bind hob zr hCust coins rows own sel bc dec wr reg bcs' htop dg N k Rw hrread hsread horead hlread
      hfr hreach hown hheld h))
  exact hcap

end FlockC3.Compose
