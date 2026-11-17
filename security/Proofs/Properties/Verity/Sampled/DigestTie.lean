module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP

/-! SPOT's digest tie: the record's round digest is the live session's. -/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Discharge

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem statement_hm96Rows {t : Flock.Tags}
    (h : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some t) : t.hm96Rows = true := by
  have htg : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  obtain rfl := Option.some.inj (htg.symm.trans h)
  rfl

theorem setup_sha512Digests {I : Exec.Inputs} {own : Option ByteArray} {record : String} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (hh : I.tags.hiddenOutputs = true) (hr : I.tags.hm96Rows = true) (hc : I.count = 1)
    (hs : Exec.setupOfP (ZkExec.zkInputs I) own record = .ok (spec, sts)) : spec.sha512Digests = true := by
  unfold Exec.setupOfP at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs I).trans hh) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · obtain ⟨_, hP⟩ := Hidden.setupHidden_specP hst
    show st.tags.hm96Rows = true
    rw [hP.tagsEq]
    exact (ZkExec.zkTags_hm96Rows I).trans hr
  · exact absurd h1 (by show ¬ 1 < I.count; omega)

theorem SetsUp.spec_dg {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) : spec.dg = A.dg := by
  have h := setup_sha512Digests A.hTags.1 (statement_hm96Rows hA.tags) hA.count hs
  funext b
  simp only [Flock.SessionSpec.dg, h, ↓reduceIte, Live.dg]

end Verity.Sampled
