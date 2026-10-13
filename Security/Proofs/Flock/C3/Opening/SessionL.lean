import Proofs.Flock.C3.Opening.Session
import Proofs.Flock.C3.Opening.SessionPartsL

/-!
# R's opening in V*'s sessions, its parts' row ports registered where they are read: `SessionAtL`

`SessionAt` (a) asks each part's copy for a registered bit of w at every row-port column (`OpeningZeroAt`), but some
elements a part holds no computed row of its unit reads, and nothing registers them. `SessionAtL` is `SessionAt` with
`OpeningZeroAtL` in (a): each part's row ports registered where some computed row reads them, as `ResZeroAtLH` asks of
`rec-residuals`. `sessionAtL_of` weakens `SessionAt` to it.
-/

namespace FlockC3.Opening

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

/-- **(n4 items 1 and 2) R's opening in V*'s sessions, at the columns the parts' rows read**: `SessionAt` with each part
`p` of the groups `gs` of the shape's S copy `cp p` of unit `un p` in session `sess p`, its row ports w where a computed
row of the unit reads them and its `v` port `o`'s `v`, its outputs registered zero (`OpeningZeroAtL`). -/
def SessionAtL (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (sh : Flock.Shape)
    (unit : String) (c3 : Flock.RecCoins.C3) (w : Array F128) (ops : FlockVBridge.AccChain.Openings) : Prop :=
  ∃ o : Flock.RecCoins.Opening, Flock.RecCoins.opening c3 unit = .ok o ∧ o.sh = sh ∧
    w.size = Flock.RecOpening.nw sh ∧
    (∃ (S : Flock.RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (sess : ℕ → Fin nS)
        (un : (p : ℕ) → Fin (Vs (sess p)).nv) (cp : ℕ → ℕ),
      Flock.RecOpening.structureOf sh = .ok S ∧ Flock.RecResiduals.groups S (Flock.RecOpening.portSizes sh) = .ok gs ∧
      ∀ p < gs.size, FlockVBridge.OfVStar.OpeningZeroAtL (zs (sess p)) (Lb (sess p)) (un p) (cp p) sh gs p
        (Flock.RecAlgebra.el w) (Flock.RecAlgebra.el o.v)) ∧
    (∃ (acc : ℕ → Array F128) (sess : ℕ → Fin nS) (un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv) (cp : ℕ → ℕ → ℕ),
      (∀ ℓ < sh.sched.levels.size, ∀ j < ops.q ℓ,
        FlockVBridge.AccChain.OpensAt (zs (sess ℓ)) (Lb (sess ℓ)) (un ℓ j) (cp ℓ j)
          (FlockVBridge.AccChain.levelLanes sh ℓ) (ops.rows ℓ)[j]!
          #[(acc (ops.start ℓ + j))[0]!, (acc (ops.start ℓ + j))[1]!] (ops.coef ℓ j) (acc (ops.start ℓ + j + 1))) ∧
      (∀ ℓ < sh.sched.levels.size, ∀ j < ops.q ℓ,
        ops.coef ℓ j = Flock.RecCoins.coefAt (o.kappa[ℓ]!) (FlockVBridge.AccChain.levelLanes sh ℓ)
          (Flock.eqTable (ops.alpha ℓ))[j]!) ∧
      ∀ r < 2, valO sh w (consO ops) (.acc0 r) = (acc ops.p0)[r]! ∧
        valO sh w (consO ops) (.acc1 r) = (acc (ops.start sh.sched.levels.size))[r]!)

/-- `SessionAt` asks more: each part's row ports registered at every column. -/
theorem sessionAtL_of {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits} {sh : Flock.Shape}
    {unit : String} {c3 : Flock.RecCoins.C3} {w : Array F128} {ops : FlockVBridge.AccChain.Openings}
    (h : SessionAt zs Lb sh unit c3 w ops) : SessionAtL zs Lb sh unit c3 w ops :=
  let ⟨o, ho, hsh, hw, ⟨S, gs, sess, un, cp, hS, hg, hp⟩, hc⟩ := h
  ⟨o, ho, hsh, hw, ⟨S, gs, sess, un, cp, hS, hg, fun p hpg => FlockVBridge.OfVStar.openingZeroAtL_of _ (hp p hpg)⟩, hc⟩

end FlockC3.Opening
