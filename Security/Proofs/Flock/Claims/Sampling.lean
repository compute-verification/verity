import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Count
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Claims.Sampling.ExecBernoulli
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Sampling.Leaf
import Proofs.Flock.Claims.Sampling.Live
import Proofs.Flock.Claims.Sampling.Path
import Proofs.Flock.Claims.Sampling.Runs
import Proofs.Flock.Claims.Sampling.RunsExp
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.Claims.Sampling.Universal
import Proofs.Flock.Claims.Sampling.Wait

/-! The sampling compiler's lemmas: each registered draw law's miss bound in `Claims.Pr` (`Sampling.Draws`), the
executable Bernoulli draw's bound (`Sampling.ExecBernoulli`) and every law's live draw's (`Sampling.Live`); each
design's acceptance probability as its model draw's miss probability, the nested escape and the classes' product
(`Sampling.Escape`), and the deployed count bound at the claims' designs (`Sampling.Count`); the extractor's rerun
programs (`Sampling.Extract`, `Sampling.Cost`), its rerun and own-call bounds (`Sampling.Runs`) and its expected reruns
(`Sampling.RunsExp`); the commitment's binding at the openings the extractor checks (`Sampling.Binding`), at the
registered path check (`Sampling.Path`) and at its hm96 leaf (`Sampling.Leaf`); `B_ext` and the block's bits from its KS (`Sampling.Bext`); the waits' truncation
and union bound (`Sampling.Wait`); the block's KS, its count form and its bits from Flock's KS at the checked units'
sessions and the deployed sampled verifier's named facts (`Sampling.KS`); Flock's KS at those sessions from
`Claims.Flock.KS` (`Sampling.Session`); the universal unit's gates computing its netlist (`Sampling.Universal`); and
the block's claims assembled, with what remains as hypotheses (`Sampling.Assembled`). -/
