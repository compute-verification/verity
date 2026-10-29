import Proofs.Flock.C3.Sparse.PubLift
import Proofs.Flock.C3.VStarList.Range

/-!
# `SparsePubRead` at the coins of a rep's own values (`scOf`)

`scOf`: the sparse coins whose points are `verifierRows`' at a rep's values (`z`, `alpha`, `x_rest`, `y_rest`) and whose
`U` is the layout factors at those points and the unit range, with phase coins `cs`.
`sparsePubRead_scOf`: at a rep's unit (`RepUnit`), `scOf` reads `SparsePubRead` wherever the layout factors pass at zero
points of the class's `k_log − 6` coordinates (`SparseRangeOk`'s verdict): `layoutFactors` refuses only on the points'
sizes (`layoutFactors_ok_of_size`), which the unit's rounds fix (`points_size_of_unit`, as `pub_of_unit` reads them).
`srcSparsePub_scOf`: the same at every accepted outer proof, from `RepUnitOfAlg`.
-/

namespace FlockC3.Sparse

noncomputable section

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Model
open FlockSoundness.Discharge.Recursion (ExecHolo)
open Flock.RecAlgebra (verifierRows)
open FlockVBridge.Sparse (SCoins)
open FlockVBridge.Runs (rhoOf tsOf)
open FlockVBridge.Compiled (RepData)
open FlockC3.HoloValues (RepUnit holoRows_coins)

/-- **The layout factors refuse only on the points' sizes**: they pass at points the size of points they pass at. -/
theorem layoutFactors_ok_of_size {sh : RecSparse.Shape} {xs ys xs' ys' : Array Flock.F128} {rng : Flock.Range}
    (h : (RecCoins.layoutFactors sh xs' ys' rng).isOk = true) (hx : xs.size = xs'.size) (hy : ys.size = ys'.size) :
    ∃ U, RecCoins.layoutFactors sh xs ys rng = .ok U := by
  obtain ⟨a, h⟩ := FlockC3.VStarList.ok_of_isOk h
  unfold RecCoins.layoutFactors at h ⊢
  dsimp only at h ⊢
  split at h
  · cases h
  rename_i hc
  split
  · rename_i hc'
    exact (hc (by simpa only [FlockLevel3.eqTable_size, Array.size_extract, hx, hy] using hc')).elim
  · exact ⟨_, rfl⟩

/-- **A rep's unit fixes its points' sizes**: `verifierRows` at the unit's rounds, at a shape of the statement's `m` and
`k_log`, returns `x_rest` and `y_rest` of `k_log − 6` coordinates (`holoRows_coins`). -/
theorem points_size_of_unit {S : Model.Statement} {sch₀ : Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
    {cap₀ : Model.Cap Refine.Dig} {os : Model.Opens 𝔽 Refine.Dig} {r : ℕ} {ts : List Game.Step}
    (U : RepUnit S sch₀ extra cap₀ os r ts) {ash : Flock.Shape} {xs : Array Flock.Claim} {cb v : Array Flock.F128}
    {info : RecAlgebra.Info RecAlgebra.Lin} {kappa : Array (Array (Array Flock.F128))}
    (hrows : verifierRows ash U.d.rounds xs cb = .ok (v, info, kappa)) (ham : ash.m = S.m) (hak : ash.kLog = S.kLog) :
    info.lc.fold.rhoIn.size = S.kLog - 6 ∧ info.lc.rIn.size = S.kLog - 6 := by
  have h22 := Refine.exec_fast100_ge U.fast
  have hmS := U.mS
  have hSk := S.hk
  obtain ⟨-, -, hrho, hts', -, hrin, hrIn⟩ := holoRows_coins hrows (by omega) U.k0 U.coins
    (by rw [ham, hak]; exact hpos_of_unit U)
  rw [ham] at hrho hts'
  rw [hak] at hrin hts'
  have hT : (tsOf U.t (U.t.k + 3 + (S.m - 6)) S.kLog).size = S.kLog - 6 := by simp [tsOf]
  exact ⟨by rw [hrin, hrho]; simp [rhoOf]; omega, by rw [hrIn, hts', Array.size_reverse, hT]⟩

/-- **The sparse coins at a rep's values `d`**: `verifierRows`' points at `d`'s rounds, extra claims and comb at the
shape `sh` (`rec_sparse.inner_points`), the layout factors at those points and the unit range `rng` for `U`, and `cs`
for the phase's own coins. -/
def scOf (e : ExecHolo) (rng : Flock.Range) (sh : Flock.Shape) (cs : Array Flock.F128) (d : RepData) : SCoins :=
  match verifierRows sh d.rounds d.extras d.comb with
  | .ok (_, info, _) => ⟨info.lc.fold.alpha, info.zc.z, info.lc.fold.rhoIn, info.lc.rIn,
      ((RecCoins.layoutFactors (unitShape e) info.lc.fold.rhoIn info.lc.rIn rng).toOption).getD #[], cs⟩
  | .error _ => ⟨0, 0, #[], #[], #[], cs⟩

theorem scOf_cs (e : ExecHolo) (rng : Flock.Range) (sh : Flock.Shape) (cs : Array Flock.F128) (d : RepData) :
    (scOf e rng sh cs d).cs = cs := by
  unfold scOf
  split <;> rfl

/-- **`SparsePubRead` at `scOf`**: at a rep's unit, `verifierRows` accepts its values at its shape (`AlgebraHolds`),
at the statement's `m` and `k_log`, and its points have `k_log − 6` coordinates, so the layout factors pass there where
they pass at zero points of that size (`hok`). -/
theorem sparsePubRead_scOf {e : ExecHolo} {sch₀ : Accounting.Schedule}
    {extra : List (Model.Claim 𝔽 (e.cls.pubStatement e.regs).m)} {cap₀ : Model.Cap Refine.Dig}
    {os : Model.Opens 𝔽 Refine.Dig} {r : ℕ} {ts : List Game.Step}
    (U : RepUnit (e.cls.pubStatement e.regs) sch₀ extra cap₀ os r ts) {rng : Flock.Range}
    (hok : (RecCoins.layoutFactors (unitShape e) (Array.replicate (e.cls.kLog - 6) (0 : Flock.F128))
      (Array.replicate (e.cls.kLog - 6) (0 : Flock.F128)) rng).isOk = true) (cs : Array Flock.F128) :
    SparsePubRead e rng U.d (scOf e rng U.sh cs U.d) := by
  obtain ⟨v, info, kappa, _, _, hrows, _⟩ := U.alg
  have ham : U.sh.m = e.cls.m := U.m.symm.trans U.mS
  have hak : U.sh.kLog = e.cls.kLog := U.kL.symm.trans U.fold.kLog
  obtain ⟨hx, hy⟩ := points_size_of_unit U hrows ham hak
  obtain ⟨Uf, hU⟩ := layoutFactors_ok_of_size hok (by rw [hx, Array.size_replicate]; rfl)
    (by rw [hy, Array.size_replicate]; rfl)
  have hsc : scOf e rng U.sh cs U.d = ⟨info.lc.fold.alpha, info.zc.z, info.lc.fold.rhoIn, info.lc.rIn, Uf, cs⟩ := by
    unfold scOf
    rw [hrows]
    dsimp only
    rw [hU]
    rfl
  rw [hsc]
  exact ⟨U.sh, U.d.extras, U.d.comb, v, info, kappa, hrows, ham, hak, rfl, rfl, rfl, rfl, hU⟩

section src

open FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose FlockC3.Reps

/-- **`SparsePubRead` at `scOf`, at every accepted outer proof**: each rep's unit at its values (`RepUnitOfAlg`, where
E's preconditions hold), at any phase coins `cs`. -/
theorem srcSparsePub_scOf {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ}
    {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) →
      C3Run (holoAt st hp)}
    {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    (hU : RepUnitOfAlg zs dg N k Rw src sh rep) {rng : Flock.Range}
    (hok : (Flock.RecCoins.layoutFactors (FlockC3.Sparse.unitShape (holoAt st hp))
      (Array.replicate ((holoAt st hp).cls.kLog - 6) (0 : Flock.F128))
      (Array.replicate ((holoAt st hp).cls.kLog - 6) (0 : Flock.F128)) rng).isOk = true)
    (cs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 →
      Array Flock.F128) :
    AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      SparsePubRead (holoAt st hp) rng (rep R ω t Lb i)
        (scOf (holoAt st hp) rng sh (cs R ω t Lb i) (rep R ω t Lb i)) := by
  intro R ω t R₂ τ o hA i ha hc
  obtain ⟨U, hsh, hd⟩ := hU R ω t R₂ τ o hA i ha hc
  have h := sparsePubRead_scOf U hok (cs R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
  rw [hd, hsh] at h
  exact h

end src

end

end FlockC3.Sparse
