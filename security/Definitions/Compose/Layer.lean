module

public import Definitions.Core.Scheme

/-!
# Compose: the composite theorem's shape over the layers and the resource schemes

Compose is the top layer: one verifier runs a composite's subprotocols (`Verity.Registration.Composite`, compose's two
lists `remote` and `onsite`, and remote's plain sibling) and accepts only when each accepts.

**What it takes.** For each subprotocol of the composite, its target at that composite, the claim it makes with its
error. The proofs' subprotocol, SPOT, meets its target through the sampling layer (`Spot.Sampling.Sound`) over the
argument system, or the recursion layer above it; each resource accounting protocol supplies a `Verity.Scheme`
(`Definitions/Core/Scheme.lean`): what it credits on a run the proofs accept, which units it reads, and `δscheme`.

**What it gives.** The probability that the composite's compiled verifier accepts committed files while one of its
subprotocols' claims fails is at most the subprotocols' errors at that strategy summed. A scheme gets back
`Verity.Scheme.GetsBack`: an accepted run's credit plus the sample's exposure at `δ` and `other` covers what it used,
except with probability `δ + δscheme`.

**Its error, in the layers below's.** `Σ_a (δ_a + c_a · Pr[SHA-512 collision])` over the subprotocols `a`. SPOT's `δ` is
the sampling layer's: the sample's miss probability plus `ε_ks + ε_bind`, each the argument system's (or the recursion
layer's `Comp.err`).

**Statements**, in the properties tree (`Properties/Verity/Soundness.lean`), each proved from its targets, which are
its hypotheses (no target is discharged by the sampling layer's theorem yet):
* `Verity.SoundnessOf`, generic in the composite: `soundness_at_of_targets`.
* `Verity.Soundness` (onsite, seven targets): `soundness_of_subprotocols`; `Verity.RemoteSoundness`:
  `remote_soundness_of_subprotocols`; `Verity.PlainSoundness`: `plain_soundness_of_subprotocols`.
* `Verity.Scheme.GetsBack`: stated; no scheme proves it yet.

**Duplicates, for next week.** SPOT's target (`Vocab.SpotSound`, `Properties/Verity/Game.lean`) is its own statement,
not `Spot.Sampling.Sound` at the composite's design; `Verity.Scheme.δscheme` and each resource target's `δ` name one
error twice.
-/
