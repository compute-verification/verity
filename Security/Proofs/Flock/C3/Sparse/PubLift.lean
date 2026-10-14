import Proofs.Flock.C3.Sparse.PubTie
import Proofs.Flock.C3.Sparse.SrcSessions
import Proofs.Flock.C3.Reps.StepsTie

/-!
# `SparseRepTie`'s pub conjunct at the tied source

`pub_tie`: at a rep's unit (`RepUnit`) on a run's rep steps, where check 5's values at the unit's rounds are the coins
`c`'s (`SparsePubRead`: `verifierRows`' points at the statement's `m` and `k_log`, and `layoutFactors` at the unit
range for `U`), the tied run's rep output (`tie`, the rep's play's, `playOut_eq`) has `pubOf c` for `holoPub`'s values
(`pub_of_unit`, `layoutFactors_U` at the points' sizes).
`srcSparsePub_of_alg`: the conjunct at `tiedSrc stp` for every accepted outer proof, from `RepUnitOfAlg`'s units and
the slots' units the unit range's (`UnitsOf`, `units_holoAt`); `srcSparsePub_repsOfVals` at `repsOfVals`, whose units
are built from the reps' parts at the record's rounds (`repUnitOfAlg_of_parts`). `rep_tiedSrc_sparseOf`: the tied reps
of `sparseOf stp w c` are `tiedSrc stp`'s, which `srcSparseTie_sparseOf` reads.
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose FlockC3.Reps
open FlockVBridge.Sparse (SCoins pubOf)
open Flock.RecAlgebra (verifierRows)
open FlockC3.HoloValues (repAt)

/-- **Hypothesis: check 5's values at a rep's rounds are the coins `c`'s** (`RecCoins.checkSparse` at rep `i`, whose
`Inner.rep` runs `verifierRows` at the record's rounds `inn.rounds i`, here `d.rounds`): `verifierRows` accepts `d`'s
rounds at a shape of the statement's `m` and `k_log`, `c`'s `z`, `alpha`, `x_rest` and `y_rest` are the points it
returns (`rec_sparse.inner_points`), and `c`'s `U` is the layout factors at those points and the unit range `rng`. -/
def SparsePubRead (e : ExecHolo) (rng : Flock.Range) (d : RepData) (c : SCoins) : Prop :=
  ∃ (ash : Flock.Shape) (xs : Array Flock.Claim) (cb v : Array Flock.F128) (info : RecAlgebra.Info RecAlgebra.Lin)
    (kappa : Array (Array (Array Flock.F128))),
    verifierRows ash d.rounds xs cb = .ok (v, info, kappa) ∧ ash.m = e.cls.m ∧ ash.kLog = e.cls.kLog ∧
      c.z = info.zc.z ∧ c.alpha = info.lc.fold.alpha ∧ c.xRest = info.lc.fold.rhoIn ∧ c.yRest = info.lc.rIn ∧
      Flock.RecCoins.layoutFactors (unitShape e) c.xRest c.yRest rng = .ok c.U

/-- **The tied rep's public values are `pubOf c`**: the rep's unit plays to outputs whose `holoPub` is `pubOf c`
(`pub_of_unit`), and the tied run's rep is that play's (`playOut_eq`); the later stages keep it. -/
theorem pub_tie {e : ExecHolo} {u : C3Run e} {i : Fin 2}
    (U : FlockC3.HoloValues.RepUnit (e.cls.pubStatement e.regs) e.sch
      (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts) u.cap₀ u.os.1 i (u.ts i))
    {rng : Flock.Range} (hunits : ∀ σ : Fin (2 ^ e.cls.sLog), e.cls.units σ =
      (if σ.val < rng.count then ({rng.pos0 + σ.val} : Finset ℕ) else ∅).filter
        (· < 2 ^ (e.cls.kLog - e.cls.slotLog)))
    {c : SCoins} (h : SparsePubRead e rng U.d c) :
    pubOf e.cls.shape c =
      holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units ((tie u).rep i).1 ((tie u).rep i).2 := by
  obtain ⟨ash, xs, cb, v, info, kappa, hrows, ham, hak, hcz, hca, hcx, hcy, hl⟩ := h
  obtain ⟨zc, lc, pend, hpl, -, -, hpub⟩ := pub_of_unit (sh := e.cls.shape) e.cls.hm U hrows ham hak e.cls.hs6
    e.cls.hsk e.cls.units hcz hca hcx hcy
    (fun hx hy => layoutFactors_U hl hx hy e.cls.hs6 e.cls.hsk e.cls.units hunits)
  have e₁ : playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i) = (some (zc, lc), pend) := playOut_eq _ hpl
  show _ = holoPub _ _ _ (((playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)).1).getD (u.rep i)).1
    (((playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)).1).getD (u.rep i)).2
  rw [e₁]
  exact hpub

/-- **The class's slots' units are the unit range's below the block** where they are the range's (`UnitsOf`). -/
theorem units_holoAt {st : Flock.Stmt} {hp : HoloParams} {rng : Flock.Range} (hu : UnitsOf st hp)
    (hrng : unitRangeOf st = .ok rng) (σ : Fin (2 ^ (holoAt st hp).cls.sLog)) :
    (holoAt st hp).cls.units σ = (if σ.val < rng.count then ({rng.pos0 + σ.val} : Finset ℕ) else ∅).filter
      (· < 2 ^ ((holoAt st hp).cls.kLog - (holoAt st hp).cls.slotLog)) := by
  obtain ⟨rng', h', hu⟩ := hu
  rw [hrng] at h'
  cases h'
  show (hp.units σ).filter _ = _
  rw [hu σ]
  rfl

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) →
    C3Run (holoAt st hp)}
  {sh : Flock.Shape}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {c : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The tied reps of `sparseOf stp w c` are `tiedSrc stp`'s**: `sparseOf` sets only the sparse stages, which no rep
reads. -/
theorem rep_tiedSrc_sparseOf
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) (i : Fin 2) :
    (tiedSrc (sparseOf stp w c) R ω t Lb).rep i = (tiedSrc stp R ω t Lb).rep i := rfl

/-- **`SparseRepTie`'s pub conjunct at `tiedSrc stp`**, at every accepted outer proof: each rep's unit at its values
`rep R ω t Lb i` (`RepUnitOfAlg` at the tied source, whose steps are `stp`'s, where E's preconditions hold: `halg`, `hacc`), check 5's values at their rounds the
coins' (`SparsePubRead`) at the inner circuit's unit range, which the slots' units are (`UnitsOf`). -/
theorem srcSparsePub_of_alg {rng : Flock.Range} (hu : UnitsOf st hp) (hrng : unitRangeOf st = .ok rng)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hU : RepUnitOfAlg zs dg N k Rw (tiedSrc stp) sh rep)
    (hr : AtRep zs dg N k Rw sh rep fun R ω t Lb i => SparsePubRead (holoAt st hp) rng (rep R ω t Lb i) (c R ω t Lb i))
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (τ) (o)
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) (i : Fin 2) :
    pubOf (holoAt st hp).cls.shape (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) =
      holoPub ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.shape (holoAt st hp).cls.units
        ((tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).1
        ((tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).2 := by
  have ha := halg R ω t R₂ τ o hA i
  have hc := hacc R ω t R₂ τ o hA i
  obtain ⟨U, -, hd⟩ := hU R ω t R₂ τ o hA i ha hc
  have hrd : SparsePubRead (holoAt st hp) rng U.d (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) := by
    rw [hd]; exact hr R ω t R₂ τ o hA i ha hc
  exact pub_tie (u := stp R ω t (layers zs dg N k Rw R ω t R₂ τ)) U (units_holoAt hu hrng) hrd

/-- **The pub conjunct at `repsOfVals`**: its units are built from the reps' parts at the values, the record's rounds
being the values' `rep R ω t Lb i` (`repUnitOfAlg_of_parts`, the steps' part by `stepsOfVals_repsOfVals`). -/
theorem srcSparsePub_repsOfVals {rng : Flock.Range}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → FlockC3.Reps.UnitVals}
    (hu : UnitsOf st hp) (hrng : unitRangeOf st = .ok rng)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hs : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i => capDig (vals R ω t Lb i).p.cap = (stp R ω t Lb).cap₀)
    (hr : AtRep zs dg N k Rw sh rep fun R ω t Lb i => SparsePubRead (holoAt st hp) rng (rep R ω t Lb i) (c R ω t Lb i))
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (τ) (o)
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) (i : Fin 2) :
    pubOf (holoAt st hp).cls.shape (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) =
      holoPub ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.shape (holoAt st hp).cls.units
        ((tiedSrc (repsOfVals stp sh rep vals) R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).1
        ((tiedSrc (repsOfVals stp sh rep vals) R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).2 :=
  srcSparsePub_of_alg hu hrng halg hacc
    (repUnitOfAlg_of_parts hpr hs hf (stepsOfVals_repsOfVals hpr hs hf hcap)) hr R ω t R₂ τ o hA i

end

end FlockC3.Sparse

#print axioms FlockC3.Sparse.pub_of_unit
#print axioms FlockC3.Sparse.pub_tie
#print axioms FlockC3.Sparse.units_holoAt
#print axioms FlockC3.Sparse.rep_tiedSrc_sparseOf
#print axioms FlockC3.Sparse.srcSparsePub_of_alg
#print axioms FlockC3.Sparse.srcSparsePub_repsOfVals
