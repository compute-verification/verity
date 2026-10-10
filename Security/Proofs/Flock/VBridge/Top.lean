import Proofs.Flock.VBridge.Stages
import Proofs.Flock.VBridge.AlgHolds
import Proofs.Flock.VBridge.AccUnits
import Proofs.Flock.VBridge.Algebra.BitsOk

/-!
# VBridge G from V*'s facts

`vbridge_of_vstar` is VBridge G at the compiled inner verifier, with every closed piece applied: the algebra from V*'s
part residuals (`algebraDecodes_of_decode`, its `BitsOk` from `structureOf_bitsOk`), the running sums from V*'s opening
units (`accChains_of_decode`), the inner verifier's acceptance from one table's files and both reps' runs
(`stagesAccept_of_runs`), and `VFullR`, which V*'s full draw gives (`vFullR_of_read`). Its hypotheses are the pieces
still open, each a fact about V*'s statements, registrations or files.
-/

namespace FlockVBridge.Top

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.OfVStar (ResZeroAt)
open FlockVBridge.AccChain (Openings OpensAt CoefsAre ConsAre ReadsEnds levelLanes)

section outer

variable {n : ℕ} {L : Law n} {Reg : Type} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Flock.Schedule)

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

/-- **VBridge G from V*'s facts.** -/
theorem vbridge_of_vstar
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (ℐ).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (ℐ).Msg)
    (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    -- the algebra: the shape's structure and groups, and V*'s part residuals at each rep
    {GS : Flock.RecAlgebra.GfS} (hS : structureOf sh = .ok GS)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups GS (portSizes sh) = .ok gs)
    (sess : ℕ → Fin S) (un : (p : ℕ) → Fin (Vs (sess p)).nv) (cp : ℕ → ℕ)
    (hvr : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
      verifierRows sh (rep (layers zs dg N k Rw R ω t R₂ τ) r).rounds (rep (layers zs dg N k Rw R ω t R₂ τ) r).extras
        (rep (layers zs dg N k Rw R ω t R₂ τ) r).comb = .ok (v, info, kappa) ∧ v.size = GS.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      (rep (layers zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw ∧ (rep (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (hres : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
      ResZeroAt (zs R ω t R₂ (sess p)) (layers zs dg N k Rw R ω t R₂ τ (sess p)) (un p) (cp p) sh gs p
        (el (rep (layers zs dg N k Rw R ω t R₂ τ) r).w) (rep (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep (layers zs dg N k Rw R ω t R₂ τ) r).comb)
    -- the running sums: V*'s opening units at each rep's queries
    (acc : ((j : Fin S) → (Vs j).Bits) → ℕ → Array F128) (ops : ((j : Fin S) → (Vs j).Bits) → Fin 2 → Openings)
    (sessO : ℕ → Fin S) (unO : (ℓ : ℕ) → ℕ → Fin (Vs (sessO ℓ)).nv) (cpO : ℕ → ℕ → ℕ)
    (hu : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < sh.sched.levels.size,
      ∀ j < (ops (layers zs dg N k Rw R ω t R₂ τ) r).q ℓ,
        OpensAt (zs R ω t R₂ (sessO ℓ)) (layers zs dg N k Rw R ω t R₂ τ (sessO ℓ)) (unO ℓ j) (cpO ℓ j)
          (levelLanes sh ℓ) ((ops (layers zs dg N k Rw R ω t R₂ τ) r).rows ℓ)[j]!
          #[(acc (layers zs dg N k Rw R ω t R₂ τ) ((ops (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[0]!,
            (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[1]!]
          ((ops (layers zs dg N k Rw R ω t R₂ τ) r).coef ℓ j)
          (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1)))
    (hcoef : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      CoefsAre sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    (hcons : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ConsAre sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    (hrd : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ReadsEnds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (acc (layers zs dg N k Rw R ω t R₂ τ))
        (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    -- the inner verifier: one table's files and both reps' runs
    (files : ((j : Fin S) → (Vs j).Bits) → TableFiles cls) (hsch : ∀ l, (files l).sch = sch)
    (hfiles : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (files (layers zs dg N k Rw R ω t R₂ τ)).Valid regs (x R ω) T CT
        ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (hruns : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (∀ r, AlgebraHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) ∧
        AccHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r)) →
      (files (layers zs dg N k Rw R ω t R₂ τ)).Runs)
    -- V*'s public files
    (hcm : VCommits zs Sk cmt) (hdec : VDecodesR wmsg (fun _ _ a b => a = b) Sk msg) :
    Flock.Assumptions.VBridge (I := ℐ) zs dg N k Rw x wmsg Sk cmt msg :=
  vbridge_compiled T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch) zs dg N k Rw x wmsg Sk cmt msg sh rep
    (FlockVBridge.OfVStar.algebraDecodes_of_decode zs dg N k Rw sh rep hS (Flock.RecAlgebra.structureOf_bitsOk hS) hg
      sess un cp hvr hsz hres)
    (FlockVBridge.AccChain.accChains_of_decode zs dg N k Rw sh rep acc ops sessO unO cpO hu hcoef hcons hrd)
    (stagesAccept_of_runs T CT cls regs sch zs dg N k Rw sh rep x wmsg files hsch hfiles hruns) hcm hdec
    (vFullR_of_read _ Sk)

end outer

end FlockVBridge.Top
