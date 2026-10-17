import Proofs.Flock.Claims.Sampling.Reference.Prover
import Proofs.Flock.Claims.Sampling.Reference.Complete
import Proofs.Flock.Claims.Sampling.Reference.Deployed
import Proofs.Flock.Claims.Sampling.Reference.DeployedCode

/-! The sampled reference prover and its completeness: the prover, which commits every unit's values under hm96
leaves, waits for the selection coin and runs Flock's masked reference prover in each checked unit's session on what it
opens (`Reference.Prover`); `Claims.Sampling.Complete` from Flock's completeness session by session, by the union bound
over the checked slots (`Reference.Complete`); at the deployed record, with what remains as hypotheses
(`Reference.Deployed`); and at the code whose checked units are the Lean sampled verifier's draw of record, one session
per registration (`Reference.DeployedCode`). -/
