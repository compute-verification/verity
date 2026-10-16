import Proofs.Flock.Claims.Sampling.Select.Seq
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Layout
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Claims.Sampling.Select.Bounds
import Proofs.Flock.Claims.Sampling.Select.Misses
import Proofs.Flock.Claims.Sampling.Select.DrawLaw

/-! The deployed sampled verifier's draw (`SampledProofs.Slots.select`) against the claims' `P_a`: a sequential draw's
stopping-time bound (`Select.Seq`); the draw's coins read in order and its law's stages (`Select.Take`); the
verifier's partition as the claims' slots (`Select.Layout`); `Select.select`'s steps and the design it draws
(`Select.Steps`); each law's miss bound on its coins (`Select.Bounds`); `drawMisses`, which proves
`Properties.SampledProofs.DrawMisses` (`Select.Misses`); and `DrawLaw` at a deployment whose draw is
`Slots.select`'s (`Select.DrawLaw`). -/
