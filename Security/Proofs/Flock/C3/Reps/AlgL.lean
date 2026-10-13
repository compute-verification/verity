import Proofs.Flock.VBridge.AlgHolds
import Proofs.Flock.VBridge.ResLayer
import Proofs.Flock.VBridge.AccChain
import Proofs.Flock.VBridge.Family.Openings

/-!
# G's items 1–4 at the layer-read forms, at the family level

`algebraDecodes_of_decode` and `accHolds_of_units` with the parts' and openings' outputs read in the layers, each
rep's units its own (`un p r`, `unO ℓ r j`), as `vbridgeRead_of_layers` composes them:

* `algebraDecodes_of_decodeLH`, `algebraDecodes_of_decodeL`: `AlgebraDecodes` from one copy per part per rep at
  `ResZeroAtLH` (a holo shape) or `ResZeroAtL`, through `algebraHolds_of_decodeLH` and `algebraHolds_of_decodeL`.
* `accChains_of_decodeL`: `AccChains` from each opening's unit at `OpensAtL`, through `unitSums_of_opensAtL`.
* `hStruct_of_checkH`: the structure and groups at the holo form of the shape a passing holo part names, as
  `VStarFamily.hStruct_of_check` at `check_part_holo`.
-/

namespace FlockC3.Reps

open Flock FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.Compiled (RepData Accepted layers AlgebraDecodes AccChains)
open FlockVBridge.OfVStar (ResZeroAtL ResZeroAtLH)
open FlockVBridge.AccChain (Openings CoefsAre ConsAre ReadsEnds levelLanes)
open FlockVBridge.Family (OpensAtL)

section outer

variable {T : ℕ} {Vs : Fin T → VStmt} {Reg' : Type}
variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {Reg₂ : Type}

/-- **G items 1–3 at a holo shape, the parts' outputs read in the layers**: `algebraDecodes_of_decode` at
`ResZeroAtLH`, rep `r`'s part `p` copy `cp p r` of unit `un p r`. -/
theorem algebraDecodes_of_decodeLH (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (s : Fin T) → (Vs s).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape)
    (rep : Reg → L.Ω → List (Cm × I.Coin) → ((s : Fin T) → (Vs s).Bits) → Fin 2 → RepData)
    {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S) (hbits : S.BitsOk)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (sess : ℕ → Fin T) (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) (cp : ℕ → Fin 2 → ℕ)
    (hvr : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
      verifierRows sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb =
          .ok (v, info, kappa) ∧ v.size = S.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w.size = S.nw ∧
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (h : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
      ResZeroAtLH (zs R ω t R₂ (sess p)) (layers zs dg N k Rw R ω t R₂ τ (sess p)) (un p r) (cp p r) sh gs p
        (el (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w) (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb) :
    AlgebraDecodes zs dg N k Rw sh rep := by
  intro R ω t R₂ τ o hacc r
  obtain ⟨v, info, kappa, hv, hvs⟩ := hvr R ω t R₂ τ o hacc r
  obtain ⟨hw, hc⟩ := hsz R ω t R₂ τ o hacc r
  exact FlockVBridge.OfVStar.algebraHolds_of_decodeLH (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
    (fun s => (hacc.1 s).2) hS hbits hv hw hc hvs hg sess (fun p => un p r) (fun p => cp p r) (h R ω t R₂ τ o hacc r)

/-- **G items 1–3, the parts' outputs read in the layers**: `algebraDecodes_of_decodeLH` at `ResZeroAtL`, through
`algebraHolds_of_decodeL`. -/
theorem algebraDecodes_of_decodeL (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (s : Fin T) → (Vs s).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape)
    (rep : Reg → L.Ω → List (Cm × I.Coin) → ((s : Fin T) → (Vs s).Bits) → Fin 2 → RepData)
    {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S) (hbits : S.BitsOk)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (sess : ℕ → Fin T) (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) (cp : ℕ → Fin 2 → ℕ)
    (hvr : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
      verifierRows sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb =
          .ok (v, info, kappa) ∧ v.size = S.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w.size = S.nw ∧
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (h : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
      ResZeroAtL (zs R ω t R₂ (sess p)) (layers zs dg N k Rw R ω t R₂ τ (sess p)) (un p r) (cp p r) sh gs p
        (el (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w) (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb) :
    AlgebraDecodes zs dg N k Rw sh rep := by
  intro R ω t R₂ τ o hacc r
  obtain ⟨v, info, kappa, hv, hvs⟩ := hvr R ω t R₂ τ o hacc r
  obtain ⟨hw, hc⟩ := hsz R ω t R₂ τ o hacc r
  exact FlockVBridge.OfVStar.algebraHolds_of_decodeL (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
    (fun s => (hacc.1 s).2) hS hbits hv hw hc hvs hg sess (fun p => un p r) (fun p => cp p r) (h R ω t R₂ τ o hacc r)

/-- **G item 4, the openings' outputs read in the layers**: `accHolds_of_units` at each accepted outer proof, each
rep's `UnitSums` from its openings' units at `OpensAtL` (`unitSums_of_opensAtL`), rep `r`'s opening `j` of level `ℓ`
copy `cpO ℓ r j` of unit `unO ℓ r j` in session `sessO ℓ`. -/
theorem accChains_of_decodeL (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (s : Fin T) → (Vs s).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape)
    (rep : Reg → L.Ω → List (Cm × I.Coin) → ((s : Fin T) → (Vs s).Bits) → Fin 2 → RepData)
    (acc : ((s : Fin T) → (Vs s).Bits) → ℕ → Array F128)
    (ops : Reg → L.Ω → List (Cm × I.Coin) → ((s : Fin T) → (Vs s).Bits) → Fin 2 → Openings)
    (sessO : ℕ → Fin T) (unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (sessO ℓ)).nv) (cpO : ℕ → Fin 2 → ℕ → ℕ)
    (hu : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < sh.sched.levels.size,
      ∀ j < (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).q ℓ,
      OpensAtL (Vs (sessO ℓ)) (layers zs dg N k Rw R ω t R₂ τ (sessO ℓ)) (unO ℓ r j) (cpO ℓ r j) (levelLanes sh ℓ)
        ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rows ℓ)[j]!
        #[(acc (layers zs dg N k Rw R ω t R₂ τ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[0]!,
          (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[1]!]
        ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).coef ℓ j)
        (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1)))
    (hc : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      CoefsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (hcons : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ConsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (hrd : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ReadsEnds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (acc (layers zs dg N k Rw R ω t R₂ τ))
        (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r)) :
    AccChains zs dg N k Rw sh rep := fun R ω t R₂ τ o h r =>
  FlockVBridge.AccChain.accHolds_of_units
    (FlockVBridge.Family.unitSums_of_opensAtL (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh
      (acc (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r) sessO
      (fun ℓ j => unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
    (hc R ω t R₂ τ o h r) (hcons R ω t R₂ τ o h r) (hrd R ω t R₂ τ o h r)

end outer

/-- **`VStarFamily.hStruct_of_check` on a holo part**: a circuit `RecResiduals.check` passes, its unit named part `p` at
a shape `RecResiduals.shape?` gives as `sh0`, its META `holo: true`; the structure and the groups return at the holo
form `{ sh0 with holo := true }` (`check_part_holo`; `RecResiduals.part` computes both). -/
theorem hStruct_of_checkH {c : Flock.Circuit} (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, p)) {sh0 : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh0) (hh : RecResiduals.holoOf c = .ok true) :
    ∃ GS gs, structureOf { sh0 with holo := true } = .ok GS ∧
      groups GS (portSizes { sh0 with holo := true }) = .ok gs := by
  obtain ⟨S, ns, nv, hpart, -⟩ := FlockVBridge.OfVStar.check_part_holo hc hp hsh hh
  unfold RecResiduals.part at hpart
  obtain ⟨GS, hGS, hpart⟩ := FlockVBridge.PartS.bind_ok hpart
  dsimp only at hpart
  split at hpart
  · cases hpart
  · obtain ⟨gs, hgs, -⟩ := FlockVBridge.PartS.bind_ok hpart
    exact ⟨GS, gs, hGS, hgs⟩

end FlockC3.Reps

#print axioms FlockC3.Reps.algebraDecodes_of_decodeLH
#print axioms FlockC3.Reps.algebraDecodes_of_decodeL
#print axioms FlockC3.Reps.accChains_of_decodeL
#print axioms FlockC3.Reps.hStruct_of_checkH
