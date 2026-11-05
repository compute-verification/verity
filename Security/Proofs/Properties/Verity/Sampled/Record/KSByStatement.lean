module

public import Proofs.Properties.Verity.Sampled.Record.Inner
public import Proofs.Properties.Verity.Sampled.Record.Law
public import Proofs.Flock.Claims.Sampling.KS

/-!
# Sampling's KS at a check on each proof block, at one-stage v2's deployed sampled verifier (`KSByDeployed`)

`Claims.Sampling.KSBy` at the record (`Verity.Sampled.Record.record A o e`), the sampled verifier one-stage v2 deploys,
at SHA-512 and the statement's interior roots, for every check `Ok` on a proof block. A protocol that asks more of a
drawn proof block than that it is correct states its check and proves that each checked unit's Flock witness reveals
openings passing it (`SessionOpensBy`); this gives it KS at the incorrect set at that check. At `correctOk` it is `KS`.

It reads three facts of the deployment, each about the circuit file or the owner's files and never about a prover: the
circuit file's rows are `sha512/row/v2` (`Flock.Bridge.RowsV2`), the statement is in scope at v2's slots over the
lowered program (`ScopeSlotsOk`), and a reference draw file exists (`RefOk`).
-/

@[expose] public section

namespace Claims.Sampling

/-- **Sampling: KS at a check on each proof block, at one-stage v2's deployed sampled verifier.** For every live session
`A` whose circuit file has `sha512/row/v2` rows, whose statement is in scope and which has a reference draw file, every
owner's files `o`, every extractor's reveal `e` and every check `Ok` on a proof block: where each checked unit's Flock
witness reveals openings that pass `Ok` (`SessionOpensBy` at the statement's interior roots), the record's sampled
verifier is knowledge sound at the incorrect set at `Ok` (`KSBy`), at SHA-512. -/
def KSByDeployed : Prop :=
  ∀ (A : Verity.Sampled.Live) (o : Verity.Sampled.Record.Own) (e : Verity.Sampled.Record.Ext),
    Flock.Bridge.RowsV2 A.D.unroll → Verity.Sampled.Record.ScopeSlotsOk A → (∃ d, Verity.Sampled.Record.RefOk A o d) →
    ∀ Ok : BlockCheck, SessionOpensBy sha512Msg (Verity.Sampled.Record.record A o e) (fun _ _ _ => A.roots) Ok →
      KSBy sha512Msg (Verity.Sampled.Record.record A o e) Ok

end Claims.Sampling
