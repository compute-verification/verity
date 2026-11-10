module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP

/-!
# Sampled proofs' digest tie: the record's round digest is the live session's

At the session v2 sets up on the parameters and the committed files (`SetsUp`), the session `flock-verify --zk` sets up
at its inputs for any record (`Exec.setupOfP` at `ZkExec.zkInputs`) digests a round's bytes as the live session does,
SHA-512 (`SetsUp.spec_dg`, `Live.dg`). So the digests a record's rounds hold, as the session decodes them
(`Flock.Record.decode`), are `A.dg` of their bytes, wherever v2's check reads the record.

* **v2's statement has hm96-sha512 rows** (`statement_hm96Rows`): v2 hands flock-verify the statement
  `Spot.OneStageZk.STATEMENT`, and the tags registered under it have them.
* **A one-table session of such a statement digests in SHA-512** (`setup_sha512Digests`): flock-verify's one-table
  session keeps the statement's tags, and its record's digests are SHA-512 exactly where they have hm96-sha512 rows
  (`Flock.Stmt.spec`).
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Discharge

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **v2's statement has hm96-sha512 rows**: the tags registered under `Spot.OneStageZk.STATEMENT`. -/
theorem statement_hm96Rows {t : Flock.Tags}
    (h : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some t) : t.hm96Rows = true := by
  have htg : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  obtain rfl := Option.some.inj (htg.symm.trans h)
  rfl

/-- **A one-table `--zk` session of a statement with hidden outputs and hm96-sha512 rows digests its record in
SHA-512**: whatever record it reads, the session flock-verify sets up at `I`'s `--zk` inputs has `sha512Digests`. -/
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

/-- **The record's round digest is the live session's**: at the session v2 sets up on the parameters and the committed
files, the session flock-verify `--zk` sets up at its inputs for any record digests a round's bytes as `A.dg` does. -/
theorem SetsUp.spec_dg {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) : spec.dg = A.dg := by
  have h := setup_sha512Digests A.hTags.1 (statement_hm96Rows hA.tags) hA.count hs
  funext b
  simp only [Flock.SessionSpec.dg, h, ↓reduceIte, Live.dg]

end Verity.Sampled
