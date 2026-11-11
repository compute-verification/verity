module

public import Proofs.Flock.Claims.Sampling.Select.Seq
public import Proofs.Flock.Claims.Sampling.Select.Take
public import Proofs.Flock.Claims.Sampling.Select.Layout
public import Proofs.Flock.Claims.Sampling.Select.Steps
public import Proofs.Flock.Claims.Sampling.Select.Bounds
public import Proofs.Flock.Claims.Sampling.Select.Misses
public import Proofs.Flock.Claims.Sampling.Select.DrawLaw
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.Claims.Sampling.Select.Scope
public import Proofs.Flock.Claims.Sampling.Select.Refusals
public import Proofs.Flock.Claims.Sampling.Select.Code
public import Proofs.Flock.Claims.Sampling.Select.OS
public import Proofs.Flock.Claims.Sampling.Select.Second
public import Proofs.Flock.Claims.Sampling.Select.SecondWitness
public import Proofs.Flock.Claims.Sampling.Select.SecondUnits
public import Proofs.Flock.Claims.Sampling.Select.SecondLaw
public import Proofs.Flock.Claims.Sampling.Select.SecondSample
public import Proofs.Flock.Claims.Sampling.Select.SecondFormed
public import Proofs.Flock.Claims.Sampling.Select.SecondFirst

/-! The deployed sampled verifier's sample (`Spot.Slots.select`) against the claims' `P_a`: a sequential sample's
stopping-time bound (`Select.Seq`); the sample's coins read in order and its sampling design's stages (`Select.Take`); a subset sample
as the live sample `Law.runOS` on the same bytes, and its bound under A3 (`Select.OS`); the
verifier's partition as the claims' slots (`Select.Layout`); `Select.select`'s steps and the design it samples
(`Select.Steps`); each sampling design's miss bound on its coins (`Select.Bounds`); `drawMisses`, which proves
`Properties.Spot.DrawMisses` (`Select.Misses`); `DrawLaw` at a deployment whose sample is `Slots.select`'s
(`Select.DrawLaw`); `designWellFormed`, which proves `Properties.Spot.DesignWellFormed` (`Select.Design`); and
`ScopeMatches` and `Refuses` from it, where the scope check is the readable scope's program part (`Select.Scope`); and that
program part as five refusals of the verifier's evaluation, each needed (`Select.Refusals`); and `secondMisses`, which
proves `Properties.Spot.SecondMisses`, the two-stage rule's second stage (`Select.Second`), with its honest
and nontrivial lemmas (`Select.SecondWitness`); and the units a stratified or work sample returns: ascending, in its
strata, each stratum's count of them (`Select.SecondUnits`); the second stage's sampling design on its cells, with the
chance its units are not sampled on a stream holding the bytes its strata read first (`Select.SecondLaw`); the sample
object the second stage builds from its units passing `Draw.checkDraw` under a well-formed sampling design
(`Select.SecondSample`); that sampling design well formed wherever `secondProblem` passes, so the second stage refuses
with at most that chance (`Select.SecondFormed`); and a uniform subset sample as a replay sample `secondProblem`'s first
test passes (`Select.SecondFirst`). -/

@[expose] public section
