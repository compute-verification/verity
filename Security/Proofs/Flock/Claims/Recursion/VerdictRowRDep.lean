import Proofs.Flock.Claims.Recursion.DecompBDeployed

/-!
# Row R's leaf at the assembly's pieces

`Top.TargetRDep rd (PD resP key elemAt vvAt side posAt) (accR rd chainAt)`, the row R leaf of
`recKSDecompL_of_leaves`, from one fact about the assembly's `rec-residuals` pieces (`targetRDep_of_cover`): at every
decision `At` admits, each rep `r` and group `p` of S has a piece of `resP x` whose group is the verifier's part `p`,
whose elements at its key are the rep's `w` at the group's row ports, and whose verifier values at its key are
`verifierRows`' `v` at the group's `V`. The four accumulators are `accR_size`, and the pieces hold because `PD` says so
(`res_holds`). The leaf's `keyAt r p` is that piece's key.

The cover needs the key to carry the group: two groups' pieces at one key read the same elements and verifier values,
so where both cover rep `r` the groups' row ports would read the same `w` elements.
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open FlockVBridge.PartS (wIdx)
open Flock.RecAlgebra (verifierRows structureOf el)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt)

variable {resP : Msg → List ResPiece} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ}
  {elemAt : Msg → List Msg → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128}
  {vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128}
  {side : Msg → List Msg → OpenPiece → OpenSide} {posAt : Msg → List Coin → OpenPiece → ℕ}

/-- **The residual pieces hold where `PD` does.** -/
theorem res_holds {x : Msg} {ms : List Msg} {cs : List Coin} (h : PD resP key elemAt vvAt side posAt x ms cs)
    {q : ResPiece} (hq : q ∈ resP x) : resHolds elemAt vvAt q x ms cs :=
  h (.res q) (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl (List.mem_map.2 ⟨q, hq, rfl⟩)))))

/-- **Row R's leaf from the pieces' cover**: `TargetRDep` at the assembly's pieces, where at every decision `At`
admits, each rep and group of S has a piece of `resP x` whose group is the verifier's part, whose elements at its key
are `w` at the group's row ports and whose verifier values at its key are `verifierRows`' `v` at the group's `V`. -/
theorem targetRDep_of_cover (rd : Msg → Option FinalP) (chainAt : Msg → List Msg → ℕ → ℕ → Array _root_.Flock.F128)
    (hcov : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), Top.At rd (PD resP key elemAt vvAt side posAt) x ms cs →
      ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ), i < D.sts.size →
      ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
        structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
        ∀ (r : Fin 2) p (hp : p < gs.size), ∃ q ∈ resP x,
          (∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv)) ∧
          (∀ k (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size),
            elemAt x ms q.key k =
              el (D.rep cls (accR rd chainAt x ms cs) i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k]) ∧
          ∀ v info kappa, verifierRows (D.sh cls i) (D.rep cls (accR rd chainAt x ms cs) i r).rounds
              (D.rep cls (accR rd chainAt x ms cs) i r).extras (D.rep cls (accR rd chainAt x ms cs) i r).comb =
              .ok (v, info, kappa) →
            ∀ k (hk : k < gs[p].2.2.size), vvAt x cs q.key k = el v gs[p].2.2[k]) :
    Top.TargetRDep rd (PD resP key elemAt vvAt side posAt) (accR rd chainAt) := by
  classical
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, hP⟩ := id hA
  obtain ⟨S, gs, hS, hg⟩ := D.structure_ok cls hst hf hx hi D.points_eq
  have hc := hcov x ms cs hA D cls i hi S gs hS hg
  refine ⟨fun r => accR_size rd chainAt x ms cs _, resP x, elemAt, vvAt,
    fun r p => if hp : p < gs.size then (Classical.choose (hc r p hp)).key else (0, 0, 0, 0),
    fun q hq => res_holds hP hq, ?_, ?_, ?_⟩
  · intro S' gs' hS' hg'
    rw [hS] at hS'
    cases hS'
    rw [hg] at hg'
    cases hg'
    intro r p hp
    obtain ⟨hq, hpart, -, -⟩ := Classical.choose_spec (hc r p hp)
    exact ⟨_, hq, by simp only [dif_pos hp], hpart⟩
  · intro S' gs' hS' hg'
    rw [hS] at hS'
    cases hS'
    rw [hg] at hg'
    cases hg'
    intro r p hp k hk
    simp only [dif_pos hp]
    exact (Classical.choose_spec (hc r p hp)).2.2.1 k hk
  · intro S' gs' hS' hg'
    rw [hS] at hS'
    cases hS'
    rw [hg] at hg'
    cases hg'
    intro r v info kappa hv p hp k hk
    simp only [dif_pos hp]
    exact (Classical.choose_spec (hc r p hp)).2.2.2 v info kappa hv k hk

end RecDeploy.DecompBR
