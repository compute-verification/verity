import Proofs.Flock.Claims.Sampling.Select.Seq
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Layout
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Claims.Sampling.Select.Bounds
import Proofs.Flock.Claims.Sampling.Select.Misses
import Proofs.Flock.Claims.Sampling.Select.DrawLaw
import Proofs.Flock.Claims.Sampling.Select.Design
import Proofs.Flock.Claims.Sampling.Select.Scope
import Proofs.Flock.Claims.Sampling.Select.Refusals
import Proofs.Flock.Claims.Sampling.Select.Code
import Proofs.Flock.Claims.Sampling.Select.OS

/-! The deployed sampled verifier's draw (`SampledProofs.Slots.select`) against the claims' `P_a`: a sequential draw's
stopping-time bound (`Select.Seq`); the draw's coins read in order and its law's stages (`Select.Take`); a subset draw
as the live draw `Law.runOS` on the same bytes, and its bound under A3 (`Select.OS`); the
verifier's partition as the claims' slots (`Select.Layout`); `Select.select`'s steps and the design it draws
(`Select.Steps`); each law's miss bound on its coins (`Select.Bounds`); `drawMisses`, which proves
`Properties.SampledProofs.DrawMisses` (`Select.Misses`); `DrawLaw` at a deployment whose draw is `Slots.select`'s
(`Select.DrawLaw`); `designWellFormed`, which proves `Properties.SampledProofs.DesignWellFormed` (`Select.Design`); and
`ScopeMatches` and `Refuses` from it, where the scope check is the readable scope's program part (`Select.Scope`); and that
program part as five refusals of the verifier's evaluation, each needed (`Select.Refusals`). -/
