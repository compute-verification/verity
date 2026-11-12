module

public import Proofs.Properties.Verity.Sampled.Record.Inner
public import Proofs.Properties.Verity.Sampled.Record.Law
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.Claims.KS.ZkExtend
public import Proofs.Flock.Bridge.LeafMap

/-!
# Knowledge soundness at one-stage v2's sampling record (`ks_record`)

`Claims.Sampling.KS_of` at the record, at SHA-512, each premise discharged at the deployed record:

* binding: the record's trace commitment is `A`'s at a circuit file of `sha512/row/v2` rows (`record_commit`), and that
  binds since no port of such a file is segmented (`segs_of_rowsV2`, `Flock.Bridge.commitDepL_binds`);
* `SessionKS`: Flock's masked KS (`FlockDeploy.KS.zk_KS_untrim`), with the draw fixed before the session
  (`drawFixed`), every checked unit's session the record's one (`sessionRuns`) and the inner numbers Flock's
  (`innerNumbers`, where a reference draw file exists);
* `SessionOpens` at a session v2 sets up (`sessionOpens`), the interior roots the statement's (`interiorFixed`), the
  draw law (`drawLaw`) and the scope refusal (`refuses_of_scopeOk`).

Every premise is a fact about the statement and the owner's files, never about a prover: the reference draw file
exists wherever v2 accepts a session (`refOk_of_accepts`).
-/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling

/-- **A circuit file of `sha512/row/v2` rows has no segmented port.** -/
theorem segs_of_rowsV2 {c} (h : Flock.Bridge.RowsV2 c) : c.ports.all (fun p => !p.segs.isSome) = true := by
  refine Array.all_eq_true.mpr fun i hi => ?_
  have hp := (Array.all_eq_true.mp h) i hi
  simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at hp
  simp [hp.2]

/-- **The record's trace commitment binds at SHA-512**, at a circuit file of `sha512/row/v2` rows. -/
theorem record_binds (A : Live) (o : Own) (e : Ext) (hv2 : Flock.Bridge.RowsV2 A.D.unroll) :
    (record A o e).commit.Binds sha512Msg := by
  rw [record_commit A o e hv2]
  exact Flock.Bridge.commitDepL_binds sha512Msg A.Cc A.rows A.D.unroll (segs_of_rowsV2 hv2) A.Lm A.roots

set_option synthInstance.maxSize 1024 in
/-- **Knowledge soundness at the record**, at SHA-512: at a session v2 sets up (`SetsUp`, `Names` and v2's reads, as
`sessionOpens` takes them), a circuit file of `sha512/row/v2` rows, the statement in scope (`ScopeOk`) and a reference
draw file (`RefOk`), the record's sampled verifier is knowledge sound (`Claims.Sampling.KS_of`). -/
theorem ks_record (A : Live) (o : Own) (e : Ext) {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]
    {V : Vocab dv α} {θ : ByteArray} {F : Array ByteArray} (hsu : SetsUp V θ F A)
    {p : _root_.Spot.OneStageZk.Params} {bs : Core.Blobs}
    (hn : Names A o p bs) (hpl : A.I.tags.plainLeaves = false) {rd : _root_.Spot.OneStageZk.Reads}
    (hrd : _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd)
    (hpart : A.I.partition = some p.own.partition)
    (hidx : _root_.Spot.FilePart.indicesOf A.pub.header = .ok (Array.range A.n))
    (he : e.reveal = revealRec A) (hv2 : Flock.Bridge.RowsV2 A.D.unroll) (hscope : ScopeOk A)
    (href : ∃ d, RefOk A o d) : Claims.Sampling.KS sha512Msg (record A o e) :=
  Claims.Sampling.KS_of (fun _ _ _ => A.roots) (record_binds A o e hv2)
    (Claims.Sampling.SessionKS_of FlockDeploy.KS.zk_KS_untrim (drawFixed A o e) (sessionRuns A o e)
      (innerNumbers e href) (SamplingDeploy.deployOf_εtable_nonneg _ (code A o e)))
    (sessionOpens A o e hsu hn hpl hrd hpart hidx (record_commit A o e hv2) he _)
    (interiorFixed A o e) (drawLaw A o e) (refuses_of_scopeOk A o e sha512Msg hscope)
    (SamplingDeploy.deployOf_εtable_nonneg _ (code A o e))

end Verity.Sampled.Record
