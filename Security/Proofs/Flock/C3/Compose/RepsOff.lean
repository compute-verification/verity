import Proofs.Flock.C3.Compose.BridgeOff
import Proofs.Flock.C3.Reps.Tie
import Proofs.Flock.C3.Reps.AlgL

/-!
# The reps' read fact off a bad event, at the fixed decoder

`RepsReadOff` (`hbr_honest_reg`'s `hrread`) at the honest family at `c3DecOf`, which is the checked family at the
checked sessions: n6's decoder route (`at_c3DecOf`) and n7's reps' tie (`srcRepTie_of_alg`) with G's items 1–4 held
only where the bad event fails. Those items read the parts' and openings' outputs in the layers, which V*'s values are
only off `ValueColl` (`RegLevels.vRowsOff_of`, `zeroOff_of`), so they enter off `Bad`; `RepUnitOfAlg`, the rest of each
unit at its values, takes them as premises and stays as it is.

* `AtSrcOff`, `AlgebraDecodesOff`, `AccChainsOff`, `SrcRepTieOff`: `AtSrc`, `AlgebraDecodes`, `AccChains` and
  `SrcRepTie` off `Bad`;
* `srcRepTieOff_of_alg`: the tie off `Bad` from G's items 1–4 off `Bad` and `RepUnitOfAlg`;
* `algebraDecodesOff_of_decodeLH`, `accChainsOff_of_decodeL`: items 1–3 at `ResZeroAtLH` and item 4 at `OpensAtL`
  off `Bad`, as `algebraDecodes_of_decodeLH` and `accChains_of_decodeL`;
* `at_c3DecOfOff`, `repsReadOff_of_tie`, `repsReadOff_of_alg`: `RepsReadOff` at the fixed decoder.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (wmsgOf OpensAtL)
open FlockVBridge.OfVStar (ResZeroAtLH)
open FlockVBridge.AccChain (Openings CoefsAre ConsAre ReadsEnds levelLanes)
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockC3.Reps

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section source

variable {S : ℕ} {Vs : Fin S → VStmt}

/-- **At every accepted outer proof off `Bad`**, at the history and the source's run there. -/
def AtSrcOff (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (P : List (Cm × (ℐ).Coin) → C3Run (holoAt st hp) → Prop) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ →
    P t (src R ω t (layers zs dg N k Rw R ω t R₂ τ))

/-- **G items 1–3 off `Bad`.** -/
def AlgebraDecodesOff (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop)
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) :
    Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i,
    AlgebraHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i)

/-- **G item 4 off `Bad`.** -/
def AccChainsOff (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop)
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) :
    Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i,
    AccHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i)

/-- **The source tied to V*'s rep units off `Bad`** (`SrcRepTie` off `Bad`). -/
def SrcRepTieOff (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) : Prop :=
  AtSrcOff zs dg N k Rw Bad src fun _ u => ∀ i : Fin 2, RepTie (holoAt st hp) u i

/-- **The reps' tie off `Bad`** from G's items 1–4 off `Bad` and the rest of each unit at its values
(`RepUnitOfAlg`), as `srcRepTie_of_alg`. -/
theorem srcRepTieOff_of_alg {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    (halg : AlgebraDecodesOff zs dg N k Rw Bad sh rep) (hacc : AccChainsOff zs dg N k Rw Bad sh rep)
    (h : RepUnitOfAlg zs dg N k Rw src sh rep) : SrcRepTieOff zs dg N k Rw Bad src := by
  intro R ω t R₂ τ o hA hb i
  obtain ⟨U, -, -⟩ := h R ω t R₂ τ o hA i (halg R ω t R₂ τ o hA hb i) (hacc R ω t R₂ τ o hA hb i)
  exact ⟨U⟩

/-- **G items 1–3 off `Bad` at a holo shape**, the parts' outputs read in the layers: `algebraDecodes_of_decodeLH`
with its premises held off `Bad`. -/
theorem algebraDecodesOff_of_decodeLH
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop)
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    {GS : Flock.RecAlgebra.GfS} (hS : structureOf sh = .ok GS) (hbits : GS.BitsOk)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups GS (portSizes sh) = .ok gs)
    (sess : ℕ → Fin S) (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) (cp : ℕ → Fin 2 → ℕ)
    (hvr : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i, ∃ v info kappa,
      verifierRows sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).comb =
          .ok (v, info, kappa) ∧ v.size = GS.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i,
      (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).w.size = GS.nw ∧
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).comb.size = 64)
    (h : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i, ∀ p < gs.size,
      ResZeroAtLH (zs R ω t R₂ (sess p)) (layers zs dg N k Rw R ω t R₂ τ (sess p)) (un p i) (cp p i) sh gs p
        (el (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).w) (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i).comb) :
    AlgebraDecodesOff zs dg N k Rw Bad sh rep := by
  intro R ω t R₂ τ o hacc hb i
  obtain ⟨v, info, kappa, hv, hvs⟩ := hvr R ω t R₂ τ o hacc hb i
  obtain ⟨hw, hc⟩ := hsz R ω t R₂ τ o hacc hb i
  exact FlockVBridge.OfVStar.algebraHolds_of_decodeLH (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
    (fun s => (hacc.1 s).2) hS hbits hv hw hc hvs hg sess (fun p => un p i) (fun p => cp p i)
    (h R ω t R₂ τ o hacc hb i)

/-- **G item 4 off `Bad`**, the openings' outputs read in the layers: `accChains_of_decodeL` with its premises held
off `Bad`. -/
theorem accChainsOff_of_decodeL (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop)
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (acc : ((j : Fin S) → (Vs j).Bits) → ℕ → Array F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Openings)
    (sessO : ℕ → Fin S) (unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (sessO ℓ)).nv) (cpO : ℕ → Fin 2 → ℕ → ℕ)
    (hu : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i,
      ∀ ℓ < sh.sched.levels.size, ∀ j < (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i).q ℓ,
      OpensAtL (Vs (sessO ℓ)) (layers zs dg N k Rw R ω t R₂ τ (sessO ℓ)) (unO ℓ i j) (cpO ℓ i j) (levelLanes sh ℓ)
        ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i).rows ℓ)[j]!
        #[(acc (layers zs dg N k Rw R ω t R₂ τ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i).start ℓ + j))[0]!,
          (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i).start ℓ + j))[1]!]
        ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i).coef ℓ j)
        (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i).start ℓ + j + 1)))
    (hc : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i,
      CoefsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i))
    (hcons : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i,
      ConsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i))
    (hrd : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ¬ Bad R ω t R₂ → ∀ i,
      ReadsEnds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (acc (layers zs dg N k Rw R ω t R₂ τ))
        (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i)) :
    AccChainsOff zs dg N k Rw Bad sh rep := fun R ω t R₂ τ o h hb i =>
  FlockVBridge.AccChain.accHolds_of_units
    (FlockVBridge.Family.unitSums_of_opensAtL (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh
      (acc (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) i) sessO
      (fun ℓ j => unO ℓ i j) (fun ℓ j => cpO ℓ i j) (hu R ω t R₂ τ o h hb i))
    (hc R ω t R₂ τ o h hb i) (hcons R ω t R₂ τ o h hb i) (hrd R ω t R₂ τ o h hb i)

end source

section honest

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)}
  {Bad : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → Prop}

local notation "𝒵" => zs
local notation "𝓓" => C3Family.honest (xR := xR) ish hss bind hob 𝒵 own sel bc
  (c3DecOf 𝒵 dg N k Rw src own sel) wr reg

/-- **A fact of the tape and the run off `Bad`, at the fixed decoder**, from that fact of the source's run on its own
steps off `Bad`: `at_c3DecOf`, whose tape and run equations hold at every accepted outer proof. -/
theorem at_c3DecOfOff {P : List Step → C3Run (holoAt st hp) → Prop} (hs : SrcOK 𝒵 dg N k Rw src own sel)
    (hP : AtSrcOff 𝒵 dg N k Rw Bad src fun _ u => P (runSteps u) u) :
    (𝓓).AtAcceptedOff Bad dg N k Rw fun R ω t Lb =>
      P (schedSteps T CT 0 (((𝓓).wmsg Lb).zip (t.map Prod.snd))) ((𝓓).run R ω t Lb) := by
  intro R ω t R₂ τ o hacc hb
  have e := tape_c3DecOf hs hacc
  obtain ⟨-, -, -, -, hdec⟩ := hs
  have hd : decodeRun (holoAt st hp) (runSteps (src R ω t (layers 𝒵 dg N k Rw R ω t R₂ τ))) =
      src R ω t (layers 𝒵 dg N k Rw R ω t R₂ τ) := hdec R ω t R₂ τ o hacc
  show P (schedSteps T CT 0 ((wmsgOf Vs (skLeaf Vs own sel (ℐ).r) (ℐ).r (c3DecOf 𝒵 dg N k Rw src own sel)
      (layers 𝒵 dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (decodeRun (holoAt st hp) (schedSteps T CT 0 ((wmsgOf Vs (skLeaf Vs own sel (ℐ).r) (ℐ).r
      (c3DecOf 𝒵 dg N k Rw src own sel) (layers 𝒵 dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))))
  rw [e, hd]
  exact hP R ω t R₂ τ o hacc hb

/-- **`RepsReadOff` at the fixed decoder, from the reps' tie off `Bad`.** -/
theorem repsReadOff_of_tie (hs : SrcOK 𝒵 dg N k Rw src own sel) (ht : SrcRepTieOff 𝒵 dg N k Rw Bad src) :
    RepsReadOff 𝓓 Bad dg N k Rw :=
  at_c3DecOfOff (P := FlockC3.HoloValues.RepsReadAt (holoAt st hp)) hs fun R ω t R₂ τ o hA hb =>
    repsReadAt_of_tie (ht R ω t R₂ τ o hA hb 0) (ht R ω t R₂ τ o hA hb 1)

/-- **`RepsReadOff` at the fixed decoder from G's items 1–4 off `Bad`** and the rest of each rep unit at its values
(`RepUnitOfAlg`): `hbr_honest_reg`'s `hrread` at `dec := c3DecOf …`, at the checked sessions. -/
theorem repsReadOff_of_alg (hs : SrcOK 𝒵 dg N k Rw src own sel) {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → RepData}
    (halg : AlgebraDecodesOff 𝒵 dg N k Rw Bad sh rep) (hacc : AccChainsOff 𝒵 dg N k Rw Bad sh rep)
    (h : RepUnitOfAlg 𝒵 dg N k Rw src sh rep) : RepsReadOff 𝓓 Bad dg N k Rw :=
  repsReadOff_of_tie hs (srcRepTieOff_of_alg halg hacc h)

end honest

end

end FlockC3.Compose

#print axioms FlockC3.Compose.repsReadOff_of_alg
#print axioms FlockC3.Compose.algebraDecodesOff_of_decodeLH
#print axioms FlockC3.Compose.accChainsOff_of_decodeL
