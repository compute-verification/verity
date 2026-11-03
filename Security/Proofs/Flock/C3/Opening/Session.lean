import Proofs.Flock.C3.Opening.SessionCoins
import Proofs.Flock.C3.Opening.SessionPartsL
import Proofs.Flock.C3.Opening.SessionAcc

/-!
# n4 from V*'s sessions: `OpeningRead` but the claim tie and the tape prefix

`openingRead_of_rows` asks five things of each accepted outcome. Three of them are V*'s sessions' and the verifier's
coins', with no decoder (`SessionAt`):

* item 2, the coins: `RecCoins.opening` accepted R's opening at its shape (`opening_ok`): its two claims of value zero,
  `RecOpening.run` at its coins, the passes from `kappa` zero;
* item 1 (a) and (b): every part of the shape's S is a copy of its `rec-opening` unit whose row ports are w where its
  computed rows read them and whose `v` port is the opening's `v`, its outputs registered zero (`OpeningZeroAtL`), so
  all of S's residuals are zero (`residuals_zero_of_cols`), S being the passes' S (`opening_ok`'s "structure moved"
  check);
* item 1 (c): each opening of R's levels is a copy of its `rec-open-stacked` unit chaining `rec-acc` (`OpensAt`), at the
  `coef` the verifier registers at the opening's `kappa` (`RecCoins.coefAt`), and w's `acc0`, `acc1` are the chain's
  rows at R's ends, so `hacc` holds at the opened rows' consistency sums (`hacc_of_units`).

`openingRead_of_session` keeps the other two as hypotheses: the claim tie (item 3, at the claims `RecCoins.opening`
builds, `coinClaim`) and the tape prefix (item 4, at the messages w holds with those consistency sums).
-/

namespace FlockC3.Opening

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

section Sessions

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

/-- **(n4 items 1 and 2) R's opening in V*'s sessions**, at the sessions `zs` and their layers `Lb`, R's shape `sh`,
the verifier's coins `c3`, w and R's openings `ops`. `RecCoins.opening` accepts R's opening `o` at `sh`, and w has its
elements. (a) Each part `p` of the groups `gs` of the shape's S is copy `cp p` of unit `un p` in session `sess p`, its
row ports w where a computed row of the unit reads them and its `v` port `o`'s `v`, its outputs registered zero
(`OpeningZeroAtL`). (c) Opening `j` of level `ℓ` is copy `cp ℓ j` of unit `un ℓ j` in session `sess ℓ`, opening its row
at its `coef` with `acc`'s rows at its chain position before and after (`OpensAt`); that `coef` is the verifier's
(`RecCoins.coefAt` at `o`'s `kappa`), and w's `acc0`, `acc1` are `acc`'s rows at R's ends. -/
def SessionAt (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (sh : Flock.Shape)
    (unit : String) (c3 : Flock.RecCoins.C3) (w : Array F128) (ops : FlockVBridge.AccChain.Openings) : Prop :=
  ∃ o : Flock.RecCoins.RegisteredOpening, Flock.RecCoins.opening c3 unit = .ok o ∧ o.sh = sh ∧
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

end Sessions

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n4's read fact from V*'s sessions) `OpeningRead`**, at R's shape `sh` (`m = mReg`, two claims), the coins
`c3`, w and R's openings `ops` read off each history and its layers: at every accepted outcome R's opening is in V*'s
sessions (`SessionAt`), the claims `RecCoins.opening` builds at w's `claim` values are the run's two registered claims
(item 3), and the tape past both sparse phases begins with the opening's steps on the ring switches w holds, at the
opened rows' consistency sums (item 4). -/
theorem openingRead_of_session (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {sh : Flock.Shape} (hsh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh)
    (unit : String)
    (c3 : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      RecCoins.C3)
    (w : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      Array F128)
    (ops : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) →
      FlockVBridge.AccChain.Openings)
    (hsess : ∀ R ω t R₂ τ o,
      FlockVBridge.Compiled.Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o →
        SessionAt (F.zs R ω t R₂) (FlockVBridge.Compiled.layers F.zs dg N k Rw R ω t R₂ τ) sh unit
          (c3 R ω t (FlockVBridge.Compiled.layers F.zs dg N k Rw R ω t R₂ τ))
          (w R ω t (FlockVBridge.Compiled.layers F.zs dg N k Rw R ω t R₂ τ))
          (ops R ω t (FlockVBridge.Compiled.layers F.zs dg N k Rw R ω t R₂ τ)))
    (hclaim : F.AtAccepted dg N k Rw fun R ω t Lb =>
      ∀ a b, coinClaim (c3 R ω t Lb) unit sh 0 = .ok a → coinClaim (c3 R ω t Lb) unit sh 1 = .ok b →
        ∀ i, FlockSoundness.Refine.ClaimRel (holoAt st hp).cls.shape.mReg (claimO sh (w R ω t Lb) ![a, b] i).toClaim
          ((F.run R ω t Lb).claim i).toClaim)
    (hpre : F.AtAccepted dg N k Rw fun R ω t Lb => ∃ (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
      openSteps (fun i => FlockSoundness.Refine.vecOf 128
          (decodedO sh (w R ω t Lb) (consO (ops R ω t Lb))).rs[i.val]!) ρ γ <+:
        afterSparse (F.run R ω t Lb) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd)))) :
    OpeningRead F dg N k Rw := by
  apply openingRead_of_rows F
  intro R ω t R₂ τ o hacc
  obtain ⟨op, hop, hsh', hw, ⟨S, gs, sess, un, cp, hS, hg, hparts⟩, ⟨acc, sessA, unA, cpA, hopen, hcoef, hrd⟩⟩ :=
    hsess R ω t R₂ τ o hacc
  obtain ⟨_, a, b, src, lins, k₁, res, bs, _, _, ha, hb, hrun, hbuild, hSeq, hv, hk⟩ := opening_ok hop
  rw [hsh'] at ha hb hrun hbuild hSeq
  rw [hk] at hcoef
  obtain rfl := Except.ok.inj (hS.symm.trans hSeq)
  have hinit : OpsOkO sh { kappa := kappa0 sh } := ⟨rfl, fun _ _ hk => by cases hk⟩
  have hB : RecOpening.buildAll sh lins.toList 0 { kappa := kappa0 sh } = .ok (res, bs) := hbuild
  have hvs := (buildAll_specO #[] (fun _ _ => 0) lins.toList 0 _ _ _ hinit hB).1.1
  obtain ⟨rs, hres, hzero⟩ := FlockVBridge.OfVStar.residuals_zero_of_cols (structureOf_bitsOkO hS) hg
    (w := w R ω t (FlockVBridge.Compiled.layers F.zs dg N k Rw R ω t R₂ τ)) (v := bs.vVals) hw hvs.symm
    fun p hp => by
      obtain ⟨_, S', hS', w', v', col, base, hw', hv', hcols, h0⟩ :=
        FlockVBridge.OfVStar.partCols_of_openingZeroAtL (F.zs R ω t R₂ (sess p)) (hacc.1 (sess p)).2 hS hg
          (hparts p hp)
      rw [hv] at hv'
      exact ⟨S', hS', w', v', col, base, hw', hv', hcols, h0⟩
  have hu := FlockVBridge.AccChain.unitSums_of_decode (F.zs R ω t R₂)
    (FlockVBridge.Compiled.layers F.zs dg N k Rw R ω t R₂ τ) (fun s => (hacc.1 s).2) sh acc _ sessA unA cpA hopen
  obtain ⟨ρ, γ, hpre⟩ := hpre R ω t R₂ τ o hacc
  exact ⟨sh, _, _, ![a, b], src, lins, k₁, res, bs, rs, ρ, γ, hsh,
    Fin.forall_fin_two.mpr ⟨coinClaim_zero ha, coinClaim_zero hb⟩, hrun, hbuild, hw,
    hacc_of_units hbuild hu hcoef hrd, hres, hzero, hclaim R ω t R₂ τ o hacc a b ha hb, hpre⟩

end FlockC3.Opening
