import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.Sampling.ExecBernoulli
import Proofs.Flock.Claims.Sampling.Live
import Proofs.Flock.Claims.Sampling.Path
import Proofs.Flock.Claims.Sampling.Wait

/-! The sampling compiler's lemmas: each registered draw law's miss bound in `Claims.Pr` (`Sampling.Draws`), the
executable Bernoulli draw's bound (`Sampling.ExecBernoulli`) and every law's live draw's (`Sampling.Live`); the
extractor's rerun programs (`Sampling.Extract`, `Sampling.Cost`); the commitment's binding at the openings the extractor
checks (`Sampling.Binding`) and at the registered path check (`Sampling.Path`); `B_ext` and the block's bits from its KS
(`Sampling.Bext`); and the waits' truncation and union bound (`Sampling.Wait`). -/
