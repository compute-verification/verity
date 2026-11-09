module

public import Proofs.Flock.Claims.Sampling.Assembled
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Sampling.Committed
public import Proofs.Flock.Claims.Sampling.Count
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.Down
public import Proofs.Flock.Claims.Sampling.Draws
public import Proofs.Flock.Claims.Sampling.EndToEndDeploy
public import Proofs.Flock.Claims.Sampling.EndToEndDraw
public import Proofs.Flock.Claims.Sampling.Escape
public import Proofs.Flock.Claims.Sampling.ExecBernoulli
public import Proofs.Flock.Soundness.Audit.Pooled
public import Proofs.Flock.Claims.Sampling.InputSlot
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Path
public import Proofs.Flock.Claims.KS.Refit
public import Proofs.Flock.Claims.Sampling.Prefix
public import Proofs.Flock.Claims.Sampling.Reference
public import Proofs.Flock.Claims.Sampling.Select
public import Proofs.Flock.Claims.Sampling.Seq
public import Proofs.Flock.Claims.Sampling.Runs
public import Proofs.Flock.Claims.Sampling.RunsExp
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Tables
public import Proofs.Flock.Claims.Sampling.Universal
public import Proofs.Flock.Claims.Sampling.Wait
public import Proofs.Flock.Claims.Sampling.ZK

/-! The sampling compiler's lemmas: each registered sampling design's miss bound in `Claims.Pr` (`Sampling.Draws`), the
executable Bernoulli sample's bound (`Sampling.ExecBernoulli`); each design's acceptance probability as its model sample's
miss probability, the nested escape and the classes' product (`Sampling.Escape`), and the deployed count bound at the
claims' designs (`Sampling.Count`); `P_a` antitone and `M_δ` closed downward at a one-class uniform subset design
(`Sampling.Down`); the compiled extractor's rerun programs (`Sampling.Extract`, `Sampling.Cost`), the steps of its
rerun and own-call bounds (`Sampling.Runs`) and of its expected reruns (`Sampling.RunsExp`); the commitment's binding at
the openings the extractor checks (`Sampling.Binding`), at the registered path check (`Sampling.Path`) and at its hm96
leaf (`Sampling.Leaf`); `B_ext` and bits from a linear KS bound (`Sampling.Bext`); the waits' truncation and union bound
(`Sampling.Wait`); the deployed sampled verifier's named facts, which the compiled verifier's KS reads at a frozen
record (`Sampling.KS`, and at a session `Sampling.Session`); `SessionRuns` at a verifier that runs a prefix of rounds,
then one Flock session at the statement the prefix fixes (`Sampling.Prefix`); the deployed sample's miss bound and
`DrawLaw` from it (`Sampling.Select`); the universal unit's gates running its private circuit (`Sampling.Universal`);
the block's claims assembled, with what remains as hypotheses (`Sampling.Assembled`, at any Flock record), and at the
deployed record over any Flock record (`Sampling.Deployed`); its zero knowledge by
sequential composition of the masked sessions' at the masked deployed record, the route of record (`Sampling.Seq`), and,
the earlier route, from a committed-coin protocol given the leaf layer (`Sampling.ZK`), at the sampled masked protocol
with every coin committed at `Hello` (`Sampling.Committed`), and at the sessions' tables, with the simulator its models
give (`Sampling.Tables`); and the deployed record's reference prover and its completeness from Flock's
(`Sampling.Reference`). -/

@[expose] public section
