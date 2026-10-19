import Proofs.Flock.Claims.KS.ZkXferJ

/-!
# The masked level-0 rows' extra lanes

Level 0's committed rows under `--zk` are `width₀ = 2^k0 + 2e` lanes wide, the witness lanes and both reps' extra lanes
in one tree (`OpensOKZ`), and the masked session's level-0 term compares an opened row with the oracle on all of them
(`Off₀Z`). `Flock.KS`'s columns at the masked record are the witness lanes alone (`zkRepColsOf`, `wideJ`, the claim's
`cols`, `zkEncD`), and `oracleOf` reads `0` past a column, so a table `C` of them, as an oracle, has zero extra lanes.
The honest prover's extra lanes are uniform (`verity/ml/flock/live/PROTOCOL.md`, "Four uniform lanes"), and rep `r`'s
level-0 row check reads its pair of them.

* `off₀Z_oracleOf_of_extra`, `offNoClashZ_oracleOf_of_extra`: a verifying level-0 opening with a nonzero extra lane is
  off every table of witness-lane columns, and is `OffNoClashZ` there unless it self-clashes. So `AccOffJ` at
  `oracleOf C` fails on every live outcome whose decoded table `j` plays to such an opening, and the deployed prover's
  own runs, which accept with every column on `C`, are not runs that accept with no opening off `oracleOf C`: the
  mixture `ZkXferJ` asks for is not the prover's live image.
* `committed_congr_lanes`: whether a table is committed to an oracle reads only its witness lanes (`CommittedPad` reads
  rep `0`'s list with no extra lane), so the masked session's bound (`session_sound_of_table_accZ_sc`) bounds the
  uncommitted acceptance at `oracleOf C` by its term off any oracle that agrees with `oracleOf C` on the witness lanes.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

/-- **A verifying level-0 opening with a nonzero extra lane is off every table of witness-lane columns**: where a
masked output opens position `p` to the row `v` with a verifying path and `v`'s lane `i ≥ 2^k0` is nonzero, it opens a
level-0 position off `oracleOf C` for every word `C` of columns no wider than the witness lanes. -/
theorem off₀Z_oracleOf_of_extra {D Sl : Type} {mPts N : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (cap₀ : Model.Cap D) (C : Fin N → List 𝔽) (hC : ∀ q, (C q).length ≤ 2 ^ T.sch.k0) (o : OutZC 𝔽 D Sl)
    (p : Fin (2 ^ (lvOf T.sch 0).logLen)) (v : Fin (width₀ T) → 𝔽)
    (hv : op₀Z Hs.H Hs.E Hs.Lf T cap₀ o p = some v) (i : Fin (width₀ T)) (hi : 2 ^ T.sch.k0 ≤ i.val)
    (hne : v i ≠ 0) : Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) o := by
  refine ⟨p, by rw [hv]; rfl, fun h => hne ?_⟩
  rw [hv] at h
  have h1 := congrFun (Option.some.inj h) i
  rw [h1]
  show Claims.Flock.oracleOf C i.val p.val = 0
  unfold Claims.Flock.oracleOf
  split
  · exact List.getD_eq_default _ _ ((hC _).trans hi)
  · rfl

/-- **Without a self-clash, such an opening is `OffNoClashZ`** at every table of witness-lane columns. -/
theorem offNoClashZ_oracleOf_of_extra {D Sl : Type} {mPts N : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (cap₀ : Model.Cap D) (C : Fin N → List 𝔽) (hC : ∀ q, (C q).length ≤ 2 ^ T.sch.k0) (o : OutZC 𝔽 D Sl)
    (p : Fin (2 ^ (lvOf T.sch 0).logLen)) (v : Fin (width₀ T) → 𝔽)
    (hv : op₀Z Hs.H Hs.E Hs.Lf T cap₀ o p = some v) (i : Fin (width₀ T)) (hi : 2 ^ T.sch.k0 ≤ i.val)
    (hne : v i ≠ 0) (hsc : ¬ SelfClashZ Hs T o) : OffNoClashZ Hs T cap₀ (Claims.Flock.oracleOf C) o :=
  ⟨off₀Z_oracleOf_of_extra Hs T cap₀ C hC o p v hv i hi hne, hsc⟩

/-- **Whether a padded table is committed to an oracle reads only its witness lanes**: two oracles that agree on the
lanes below `2^k0` are committed alike. -/
theorem committed_congr_lanes {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] {mPts : ℕ} (A : Model.Arith F K) (T : TabZK F mPts) {C C' : Model.Oracle F}
    (h : ∀ ℓ < 2 ^ T.sch.k0, ∀ q, C ℓ q = C' ℓ q) : T.Committed A C ↔ T.Committed A C' := by
  have hr : padRows T.sch.k0 0 0 T.sch.level₀.logLen C = padRows T.sch.k0 0 0 T.sch.level₀.logLen C' := by
    funext ℓ q
    refine Fin.addCases (fun ℓ => ?_) (fun i => i.elim0) ℓ
    simp only [padRows, padLane, Fin.addCases_left]
    exact h ℓ.val ℓ.isLt q.val
  unfold TabZK.Committed CommittedPad padWits padList
  rw [hr]

end FlockDeploy.KS
