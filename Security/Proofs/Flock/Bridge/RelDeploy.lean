import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.LeafMap

/-!
# rel_deploy, proved (`RelDeployTarget`, and `RelDeployTargetL` at a leaf map)

The sampling relation reads a commitment only through its opening check (`faulty_eq_of_opens`, `relS_iff_of_opens`):
`layer`, the published leaf layer, is read by ZK leakage alone. A sampled verifier's code whose layout is `commitDep`'s
(`LayoutIs`) has the opening check of `commitDep` (`opens_commit_of_layoutIs`): at a `RowsV2` file every row is a
`sha512/row/v2` row in both (`rowDep_v2`), and the other fields are the layout's. So at a statement the code parses and
partitions, `R` at the code's commitment is `RelS` at `commitDep` (`R_iff_relS`), at every witness:

* `relDeployTarget`: `RelDeployTarget`.
* `RelDeployTargetL`, `relDeployTargetL`: the same at a leaf map (`LayoutIsL`, `commitDepL`), V5's layout.
-/

namespace Flock.Bridge

open Claims Claims.Sampling

/-- **The faulty set reads the commitment only through its opening check.** -/
theorem faulty_eq_of_opens {H : Msg → Msg} {Cm Cm' : CommitmentScheme Bool} (h : Cm.opens = Cm'.opens)
    (s : AuditStatement) (Pt : Partition) (w : Wit) : faulty H Cm s Pt w = faulty H Cm' s Pt w := by
  have hv : ∀ s' w', Wit.val H Cm s' w' = Wit.val H Cm' s' w' := by
    intro s' w'
    funext g
    simp only [Wit.val, CommitmentScheme.Opens, h]
  have hF : Faulty H Cm s Pt w = Faulty H Cm' s Pt w := by
    funext v
    simp only [Faulty, hv]
  unfold faulty
  rw [hF]

theorem relS_iff_of_opens {H : Msg → Msg} {Cm Cm' : CommitmentScheme Bool} (h : Cm.opens = Cm'.opens) {δ : ℝ}
    {s : AuditStatement} {Pt : Partition} {w : Wit} : RelS H Cm δ s Pt w ↔ RelS H Cm' δ s Pt w := by
  unfold RelS
  rw [faulty_eq_of_opens h]

/-- **`R` at a code's commitment is `RelS` at any commitment with its opening check**, at a statement the code parses
and partitions. -/
theorem R_iff_relS {H : Msg → Msg} {cd : SamplingDeploy.Code} {Cm : CommitmentScheme Bool}
    (hop : (SamplingDeploy.commit cd).opens = Cm.opens) {δ : ℝ} {x : Msg} {s : AuditStatement} {Pt : Partition}
    {w : Wit} (hs : cd.parse x = some s) (hP : cd.partition s = some Pt) :
    R H (SamplingDeploy.commit cd) cd.parse cd.partition δ x w ↔ RelS H Cm δ s Pt w := by
  constructor
  · rintro ⟨s', Pt', hs', hP', hR⟩
    obtain rfl := Option.some.inj (hs.symm.trans hs')
    obtain rfl := Option.some.inj (hP.symm.trans hP')
    exact (relS_iff_of_opens hop).1 hR
  · exact fun hR => ⟨s, Pt, hs, hP, (relS_iff_of_opens hop).2 hR⟩

/-- **A code whose layout is `commitDep`'s has `commitDep`'s opening check.** -/
theorem opens_commit_of_layoutIs {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt}
    {c : Flock.Circuit} {nl : ℕ} {roots : List Msg} {cd : SamplingDeploy.Code} (h : LayoutIs C rows c nl roots cd) :
    (SamplingDeploy.commit cd).opens = (commitDep C rows c nl roots).opens := by
  obtain ⟨hv2, hat, hplace, hbits, hsalt, hπ, hown⟩ := h
  rw [commitDep, rowDep_v2 C rows hv2, ← hat, ← hplace, ← hbits, ← hsalt, ← hπ, ← hown]
  rfl

/-- **A code whose layout is `commitDepL`'s has `commitDepL`'s opening check.** -/
theorem opens_commit_of_layoutIsL {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt}
    {c : Flock.Circuit} {L : LeafMap} {roots : List Msg} {cd : SamplingDeploy.Code}
    (h : LayoutIsL C rows c L roots cd) :
    (SamplingDeploy.commit cd).opens = (commitDepL C rows c L roots).opens := by
  obtain ⟨hv2, hat, hplace, hbits, hsalt, hπ, hown⟩ := h
  rw [commitDepL_v2 C rows hv2, ← hat, ← hplace, ← hbits, ← hsalt, ← hπ, ← hown]
  rfl

/-- **rel_deploy** (`RelDeployTarget`): the code's commitment has `commitDep`'s opening check, so outside `RelS` at
`commitDep` is outside the record's `R`. -/
theorem relDeployTarget : RelDeployTarget := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hL hx hP hrel hR
  exact hrel ((R_iff_relS (opens_commit_of_layoutIs hL) hx hP).1 hR)

/-- **rel_deploy at a leaf map**: at the deployed sampling record of a verifier's code whose layout is `commitDepL`'s
(`LayoutIsL`), and which parses `x` to `stmtOf` and partitions it to `partOf`, a `witDep` outside `RelS` at SHA-512 and
`commitDepL` is outside the record's `R`. -/
def RelDeployTargetL : Prop :=
  ∀ {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (cd : SamplingDeploy.Code)
    {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (L : LeafMap) (ns : List ℕ) (r : OneStage)
    (κ : Fin C.N → Option Bool) (desc : Msg) (q : Query) (roots : List Msg) (Xr : Fin C.N → Bool) (δ : ℝ) (x : Msg),
    LayoutIsL C rows c L roots cd →
    cd.parse x = some (stmtOf P κ r desc q roots) → cd.partition (stmtOf P κ r desc q roots) = some (partOf P κ ns) →
    ¬ RelS sha512Msg (commitDepL C rows c L roots) δ (stmtOf P κ r desc q roots) (partOf P κ ns)
      (witDep C (commitDepL C rows c L roots) sha512Msg roots Xr) →
    ¬ R sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep C (commitDepL C rows c L roots) sha512Msg roots Xr)

/-- **rel_deploy at a leaf map** (`RelDeployTargetL`), as `relDeployTarget`. -/
theorem relDeployTargetL : RelDeployTargetL := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hL hx hP hrel hR
  exact hrel ((R_iff_relS (opens_commit_of_layoutIsL hL) hx hP).1 hR)

end Flock.Bridge
