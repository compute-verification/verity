import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Sampling.AssembledPlain
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Committed
import Proofs.Flock.Claims.Sampling.Count
import Proofs.Flock.Claims.Sampling.Deployed
import Proofs.Flock.Claims.Sampling.Down
import Proofs.Flock.Claims.Sampling.DeployedSha512
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.Sampling.EndToEndDeploy
import Proofs.Flock.Claims.Sampling.EndToEndDraw
import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Claims.Sampling.ExecBernoulli
import Proofs.Flock.Claims.Sampling.PooledPr
import Proofs.Flock.Claims.Sampling.InputSlot
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Sampling.Leaf
import Proofs.Flock.Claims.Sampling.Live
import Proofs.Flock.Claims.Sampling.Path
import Proofs.Flock.Claims.Sampling.Raise
import Proofs.Flock.Claims.Sampling.Reference
import Proofs.Flock.Claims.Sampling.Select
import Proofs.Flock.Claims.Sampling.Seq
import Proofs.Flock.Claims.Sampling.Runs
import Proofs.Flock.Claims.Sampling.RunsExp
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.Claims.Sampling.Tables
import Proofs.Flock.Claims.Sampling.Universal
import Proofs.Flock.Claims.Sampling.Wait
import Proofs.Flock.Claims.Sampling.ZK

/-! The sampling compiler's lemmas: each registered draw law's miss bound in `Claims.Pr` (`Sampling.Draws`), the
executable Bernoulli draw's bound (`Sampling.ExecBernoulli`), the audit layer's pooled escape (`Sampling.PooledPr`) and every law's live draw's (`Sampling.Live`); each
design's acceptance probability as its model draw's miss probability, the nested escape and the classes' product
(`Sampling.Escape`), and the deployed count bound at the claims' designs (`Sampling.Count`); `P_a` antitone and `M_δ`
closed downward at a one-class uniform subset design (`Sampling.Down`); the extractor's rerun
programs (`Sampling.Extract`, `Sampling.Cost`), its rerun and own-call bounds (`Sampling.Runs`) and its expected reruns
(`Sampling.RunsExp`); the commitment's binding at the openings the extractor checks (`Sampling.Binding`), at the
registered path check (`Sampling.Path`) and at its hm96 leaf (`Sampling.Leaf`); `B_ext` and the block's bits from its
KS (`Sampling.Bext`); the waits' truncation and union bound (`Sampling.Wait`); the block's KS, its count form and its
bits from Flock's KS at the checked units' sessions and the deployed sampled verifier's named facts (`Sampling.KS`);
Flock's KS at those sessions from `Claims.Flock.KS` (`Sampling.Session`), and at a raised `N₀` with the weight scaled
by `k` (`Sampling.Raise`); the deployed draw's miss bound and `DrawLaw` from it (`Sampling.Select`); the universal
unit's gates running its private circuit (`Sampling.Universal`); the block's claims assembled, with what remains as
hypotheses (`Sampling.Assembled`, at any Flock record; at SHA-512 at the plain one, `Sampling.AssembledPlain`), and at the
deployed record over any Flock record (`Sampling.Deployed`); its zero knowledge by sequential composition of the
masked sessions' at the masked deployed record, the route of record (`Sampling.Seq`), and, the earlier route,
from a committed-coin protocol given the leaf layer (`Sampling.ZK`), at the sampled masked protocol with every coin
committed at `Hello` (`Sampling.Committed`), and at the sessions' tables, with the simulator its models give
(`Sampling.Tables`); and the
deployed record's reference prover and its completeness from Flock's (`Sampling.Reference`). -/
