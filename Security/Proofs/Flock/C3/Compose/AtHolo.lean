import Proofs.Flock.C3.Compose.LastRead
import Proofs.Flock.C3.Compose.HoloTys
import Proofs.Flock.C3.Compose.RsFacts
import Proofs.Flock.C3.RegLevels.LigWhole

/-!
# `SrcOK` at the guarantee's message types and the checked sessions

`srcOK_holo`: `srcOK_tied_norm` at `T := holoTys st hp` (`Recursive/AuditC3.lean`) and at the checked sessions
(`checkedC3`), with the hypotheses those fix discharged: `hcap`, `hbool` and `hty` are `holoTys`'s types 0, 1 and 2,
`hT` is `shapeTys_holo`, `hleaf` is `hLeafC3_checked` (V*'s list is not empty), `hplay` is `stagesPlay_of_ties` at
the stages' ties (the reps', the sparse phases', R's opening's with its read, and R's levels'), and `hw` is
`ligWhole_of_tie` at R's levels' tie.

`srcOK_holo_rs`: `srcOK_holo` at the checked family's sessions (`C3Family.checked`), with R's opening's committed
rounds (`hop`) from the verifier's checks (`openMsgsRead_checked`): the sparse phases' part ends where R's opening
starts, R's opening is tied to V*'s layers at the tied source, and its ring switches are read at their rounds.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (RepsMsgsRead SrcRepTie)

variable {r : ℕ} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt (Flock.Guarantees.holoTys st hp) CT st hp

/-- **`SrcOK` at the normalized tied source, at `holoTys` and the checked sessions** (`zs`, `hzs`): `srcOK_tied_norm`
with its message types, its leaf, its stages' plays and R's Ligerito's last coin discharged. -/
theorem srcOK_holo {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {ish : RecVStar.Inner} {cs : List (Cm × (ℐ).Coin) → Option Flock.RecVStar.InnerCommits}
    {coins : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr}
    (hzs : zs = Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h) (hS : 0 < S)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {cap₀ : Cap Dig}
    (hcoins : AtSrc zs dg N k Rw (tiedSrc stp) fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    (hrp : RepsMsgsRead zs dg N k Rw (tiedSrc stp) own sel) (hsp : SparseMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (hop : OpenMsgsRead zs dg N k Rw (tiedSrc stp) own sel) (hlg : LigMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw (tiedSrc stp) s)
    (hrt : SrcRepTie zs dg N k Rw (tiedSrc stp)) (hst : FlockC3.Sparse.SrcSparseTie zs dg N k Rw (tiedSrc stp))
    (hot : AtSrc zs dg N k Rw (tiedSrc stp) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hlt : FlockC3.RegLevels.SrcLevelTie zs dg N k Rw (tiedSrc stp) cap₀) :
    SrcOK zs dg N k Rw (tiedSrc (normSrc stp)) own sel := by
  have hleaf : HLeafC3 zs dg N k Rw := by
    subst hzs
    exact hLeafC3_checked hS dg N k Rw
  refine srcOK_tied_norm hleaf hcoins hrp hsp hop hlg hos (FlockC3.RegLevels.ligWhole_of_tie hlt) ⟨0, rfl⟩ ⟨1, rfl⟩
    ⟨2, rfl⟩ (fun _ _ _ _ _ _ _ => shapeTys_holo st hp _) fun R ω t R₂ τ o hA => ?_
  obtain ⟨sh, w, ops, ho, hor⟩ := hot R ω t R₂ τ o hA
  exact stagesPlay_of_ties (hrt R ω t R₂ τ o hA) (hst R ω t R₂ τ o hA) ho hor (hlt R ω t R₂ τ o hA)

/-- **`SrcOK` at the checked family**, at `holoTys`: `srcOK_holo` at `C3Family.checked`'s sessions (`hzs`), its `hop`
from the verifier's checks (`openMsgsRead_checked`: the sparse phases' part ends at `c₀` coins, R's opening is tied to
V*'s layers at the tied source, and its ring switches are read at their rounds). -/
theorem srcOK_holo_rs {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).Outer Reg'}
    (hzs : zs = (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (xR := xR) ish hss bind hob zr hCust h coins
      rows own sel bc dec wr reg).zs)
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) (hS : 0 < vstarCount ish)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128} {c₀ : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {cap₀ : Cap Dig}
    (hcoins : AtSrc zs dg N k Rw (tiedSrc stp) fun t u => StageCoins (CT := CT) (t.map Prod.snd) u)
    (hrp : RepsMsgsRead zs dg N k Rw (tiedSrc stp) own sel) (hsp : SparseMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (he : PartEnds sparsePart zs dg N k Rw (tiedSrc stp) c₀)
    (htie : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      FlockC3.Opening.OpenTieL (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (hall : OpenAll zs dg N k Rw sh w) (hk : RsKeys zs dg N k Rw sh w)
    (hr : RoundReads zs dg N k Rw sh w own sel c₀) (hlg : LigMsgsRead zs dg N k Rw (tiedSrc stp) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw (tiedSrc stp) s)
    (hrt : SrcRepTie zs dg N k Rw (tiedSrc stp)) (hst : FlockC3.Sparse.SrcSparseTie zs dg N k Rw (tiedSrc stp))
    (hot : AtSrc zs dg N k Rw (tiedSrc stp) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hlt : FlockC3.RegLevels.SrcLevelTie zs dg N k Rw (tiedSrc stp) cap₀) :
    SrcOK zs dg N k Rw (tiedSrc (normSrc stp)) own sel := by
  have hop : OpenMsgsRead zs dg N k Rw (tiedSrc stp) own sel := by
    subst hzs
    exact openMsgsRead_checked hF he htie hall hk hr
  exact srcOK_holo (ish := ish) (cs := Flock.Guarantees.innerHeld wr st reg) (coins := fun t => coins (wr t).2)
    (bind := bind) (rows := rows) (zr := zr) (hC := hCust) (h := h) hzs hS hcoins hrp hsp hop hlg hos hrt hst hot
    hlt

end

end FlockC3.Compose

#print axioms FlockC3.Compose.srcOK_holo_rs
