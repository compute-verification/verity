import Proofs.Flock.Bridge.LeafMap

/-!
# rel_deploy, proved

At a code whose layout is `commitDep`'s (`LayoutIs`), the deployed commitment (`SamplingDeploy.commit`) opens what
`commitDep` opens: at a `RowsV2` file every row's message is its `sha512/row/v2` input (`rowDep_v2`), and the two differ
only in their published leaf layer, which the relation never reads (`relS_congr_opens`). So a witness outside `RelS` at
`commitDep` is outside the record's `R` wherever the code parses the bytes to that statement and partitions it to that
partition (`not_R_of_opens`), whatever the witness:

* `relDeploy : RelDeployTarget`;
* `relDeployL`: the same at a leaf map (`LayoutIsL`, `commitDepL`), V5's layout.
-/

namespace Flock.Bridge

open Claims Claims.Sampling

/-- **The relation reads a commitment only through its opening check**: two schemes with one `opens` have one `RelS`. -/
theorem relS_congr_opens {H : Msg → Msg} {Cm Cm' : CommitmentScheme Bool} (h : Cm.opens = Cm'.opens) {δ : ℝ}
    {s : AuditStatement} {P : Partition} {w : Wit} : RelS H Cm δ s P w ↔ RelS H Cm' δ s P w := by
  obtain ⟨o, l⟩ := Cm
  obtain ⟨o', l'⟩ := Cm'
  cases h
  exact Iff.rfl

/-- **Outside `RelS` at a scheme the deployed commitment agrees with is outside the record's `R`**, at bytes the code
parses to the statement and partitions to the partition. -/
theorem not_R_of_opens {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (cd : SamplingDeploy.Code) {Cm : CommitmentScheme Bool} (h : (SamplingDeploy.commit cd).opens = Cm.opens)
    {H : Msg → Msg} {δ : ℝ} {x : Msg} {s : AuditStatement} {P : Partition} {w : Wit} (hp : cd.parse x = some s)
    (hP : cd.partition s = some P) (hS : ¬ RelS H Cm δ s P w) :
    ¬ R H (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x w := by
  rintro ⟨s', P', hs', hP', hR⟩
  rw [hp, Option.some.injEq] at hs'
  subst hs'
  rw [hP, Option.some.injEq] at hP'
  subst hP'
  exact hS ((relS_congr_opens h).1 hR)

/-- **At a code whose layout is `commitDep`'s, the deployed commitment opens what `commitDep` opens.** -/
theorem commit_opens_of_layoutIs {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt}
    {c : Flock.Circuit} {nl : ℕ} {roots : List Msg} {cd : SamplingDeploy.Code} (h : LayoutIs C rows c nl roots cd) :
    (SamplingDeploy.commit cd).opens = (commitDep C rows c nl roots).opens := by
  obtain ⟨hv2, hat, hpl, hbits, hsalt, hπ, hown⟩ := h
  unfold commitDep
  rw [rowDep_v2 C rows hv2]
  simp only [SamplingDeploy.commit, pathCommit, hat, hpl, hbits, hsalt, hπ, hown]

/-- **At a code whose layout is `commitDepL`'s, the deployed commitment opens what `commitDepL` opens.** -/
theorem commit_opens_of_layoutIsL {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt}
    {c : Flock.Circuit} {L : LeafMap} {roots : List Msg} {cd : SamplingDeploy.Code}
    (h : LayoutIsL C rows c L roots cd) : (SamplingDeploy.commit cd).opens = (commitDepL C rows c L roots).opens := by
  obtain ⟨hv2, hat, hpl, hbits, hsalt, hπ, hown⟩ := h
  rw [commitDepL_v2 C rows hv2]
  simp only [SamplingDeploy.commit, pathCommit, hat, hpl, hbits, hsalt, hπ, hown]

/-- **rel_deploy** (`RelDeployTarget`), proved. -/
theorem relDeploy : RelDeployTarget :=
  fun F cd _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hL hp hP hS => not_R_of_opens F cd (commit_opens_of_layoutIs hL) hp hP hS

/-- **rel_deploy at a leaf map**: at the deployed sampling record of a verifier's code whose layout is `commitDepL`'s
(`LayoutIsL`, V5's), and which parses `x` to `stmtOf` and partitions it to `partOf`, a `witDep` outside `RelS` at
SHA-512 and `commitDepL` is outside the record's `R`. -/
theorem relDeployL {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (cd : SamplingDeploy.Code) {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (L : LeafMap) (ns : List ℕ) (r : OneStage)
    (κ : Fin C.N → Option Bool) (desc : Msg) (q : Query) (roots : List Msg) (Xr : Fin C.N → Bool) (δ : ℝ) (x : Msg)
    (hL : LayoutIsL C rows c L roots cd) (hp : cd.parse x = some (stmtOf P κ r desc q roots))
    (hP : cd.partition (stmtOf P κ r desc q roots) = some (partOf P κ ns))
    (hS : ¬ RelS sha512Msg (commitDepL C rows c L roots) δ (stmtOf P κ r desc q roots) (partOf P κ ns)
      (witDep C (commitDepL C rows c L roots) sha512Msg roots Xr)) :
    ¬ R sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep C (commitDepL C rows c L roots) sha512Msg roots Xr) :=
  not_R_of_opens F cd (commit_opens_of_layoutIsL hL) hp hP hS

end Flock.Bridge
