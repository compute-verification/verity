import Proofs.Flock.C3.Compose.Bridge
import Proofs.Flock.C3.Compose.Decoder

/-!
# Piece (b) off a bad event of the history

`vbridgeOff_c3` with R's opening's and levels' read facts held only at the histories and registrations where a bad
event `Bad` fails (`C3Family.AtAcceptedOff`): the guarantee's `_hbr` is off a registrant's row off a value the
verifier registers itself (`ValueColl`) and off a read's leaf off the table's (`LeafColl`), and those read facts are
where the value ties enter (`value_tie_coll` at `coef`, `dirs`, `v`), which hold only off `ValueColl`.

* `C3Family.AtAcceptedOff`, `OpeningReadOff`, `RegLevelsReadOff`: `AtAccepted`, `OpeningRead` and `RegLevelsRead` off
  `Bad`;
* `opening_of_readAt`: `hOpening_of_decode` at one tape and run;
* `vbridgeOff_c3_off`: `vbridgeOff_c3` from those, giving `VBridgeOff` off `RegColl ∨ Bad`, the first field of
  `Flock.Assumptions.VBridgeCOff Bad`.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.HoloValues FlockC3.Sparse FlockC3.Opening FlockC3.RegLevels FlockC3.PubRows

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **At every accepted outcome off `Bad`**: `AtAccepted` at the histories and registrations where `Bad` fails. -/
def C3Family.AtAcceptedOff (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (Bad : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → Prop) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ)
    (P : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      Prop) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ →
    P R ω t (layers F.zs dg N k Rw R ω t R₂ τ)

/-- **R's opening's read fact off `Bad`** (`OpeningRead`, at `OpeningReadAt`). -/
def OpeningReadOff (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (Bad : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → Prop) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) : Prop :=
  F.AtAcceptedOff Bad dg N k Rw fun R ω t Lb =>
    OpeningReadAt (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

/-- **R's levels' read fact off `Bad`** (`RegLevelsRead`). -/
def RegLevelsReadOff (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (Bad : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → Prop) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) : Prop :=
  F.AtAcceptedOff Bad dg N k Rw fun R ω t Lb =>
    RegTape xR.cap (holoAt st hp).rsch.sch (F.run R ω t Lb).opened
      (tapeR (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb))

/-- **R's opening plays from its read fact at one tape `l`**, where the run is `l`'s decode. -/
theorem opening_of_readAt {l : List Step} {u : C3Run (holoAt st hp)} (hu : u = decodeRun (holoAt st hp) l)
    (h : OpeningReadAt l u) :
    ∃ hreg : 7 ≤ (holoAt st hp).cls.shape.mReg,
      PlaysT (opening execArith hreg [(u.claim 0).toClaim, (u.claim 1).toClaim]) u.op (some u.opened) := by
  obtain ⟨sh, M, a, src, k₀, k₁, res, ρ, γ, -, hop, hz, hrel, hpre⟩ := h
  subst hu
  exact ⟨seven_le_mReg _, opening_of_steps _ _ (fun i => FlockSoundness.Refine.vecOf 128 M.rs[i.val]!) ρ γ hpre
    fun i => packed_check (seven_le_mReg _) (hrel i) (checks_of_run hop hz i)⟩

/-- **(b) VBridge's acceptance part off `Bad`**: `vbridgeOff_c3` with R's opening's and levels' read facts off `Bad`, at
a family whose run decodes the tape (`hrun`) and R's Ligerito and openings as `decodeRun` does (`hst`). -/
theorem vbridgeOff_c3_off (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (Bad : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → Prop) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (hrun : ∀ R ω t Lb,
      F.run R ω t Lb = decodeRun (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))))
    (htape : HTape F dg N k Rw) (hreps : HReps F dg N k Rw) (hsp : HSparse F dg N k Rw)
    (hst : HLgStages F dg N k Rw) (hop : OpeningReadOff F Bad dg N k Rw) (hlv : RegLevelsReadOff F Bad dg N k Rw) :
    VBridgeOff (I := holoInnerAt T CT st hp) (fun R ω t R₂ => RegColl F.zs R ω t R₂ ∨ Bad R ω t R₂) F.zs dg N k Rw
      (fun _ _ => xR) F.wmsg := by
  intro R ω t R₂ hc τ o h₀
  have ha : Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o := ⟨h₀, fun h => hc (Or.inl h)⟩
  have hb : ¬ Bad R ω t R₂ := fun h => hc (Or.inr h)
  have ht := htape R ω t R₂ τ o ha
  obtain ⟨hr₀, hv₀, hk₀⟩ := hreps R ω t R₂ τ o ha 0
  obtain ⟨hr₁, hv₁, hk₁⟩ := hreps R ω t R₂ τ o ha 1
  have hs₀ := hsp R ω t R₂ τ o ha 0
  have hs₁ := hsp R ω t R₂ τ o ha 1
  obtain ⟨hR, ho⟩ := opening_of_readAt (hrun R ω t (layers F.zs dg N k Rw R ω t R₂ τ))
    (hop R ω t R₂ τ o ha hb)
  obtain ⟨hl, hvR, hkR⟩ := regLevels_of_readAt (hst R ω t R₂ τ o ha) (hlv R ω t R₂ τ o ha hb)
  obtain ⟨out, hpl, hacc⟩ := holoTableCL_plays (capReg := xR.cap) (hreg := hR) (h₀ := hr₀) (h₁ := hr₁)
    (hs₀ := hs₀) (hs₁ := hs₁) (ho := ho) (hl := hl) (hv := ⟨hv₀, hv₁, hvR⟩) (hok := ⟨hk₀, hk₁, hkR⟩)
  exact ⟨out, (congrArg (fun l => PlaysT _ l out) ht).mpr hpl, hacc⟩

end FlockC3.Compose

#print axioms FlockC3.Compose.vbridgeOff_c3_off
