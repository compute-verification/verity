module

public import Definitions.Core.Game
public import Definitions.Flock.ZK.AnyValues

/-!
# Zero knowledge at any values

What the ZK server releases is simulated per outcome (`FlockSoundness.ZK.PerOutcome`) for any worker and any trace,
valid or not, at any values carrying the statement's public region data, which the ZK server's splice of the public
regions makes true of every worker. The simulator reads the public inputs, the auditor's coins and samples, a dummy
carrying the public region data, and the release's public outcome (each verdict and where the release stopped). It reads nothing of the
worker's values: `Table.sim` reads none of the table's `vals`, `nu` or `sReg`, so at the table whose witnesses carry a
worker's clear values it is one simulator at every value (`FlockSoundness.ZK.Table.sim_onVals`). An outcome that
correlates with the values costs a factor of the number of outcomes `|St|`, `log₂ |St|` bits.

Scope:

* The released bytes and their order, not their times: when each frame leaves is the worker's, and its timing is the
  network certifier's (`trusted-host`).
* The registered commitments are public input, made outside the ZK server; a worker that chooses their salts is outside
  this guarantee.
* Both statements are over the table model (`FlockSoundness.ZK.Table`), at one statement and fixed coins, not yet at the
  session loop's release. There the one zero-knowledge definition, `ZKOut` in the claims core, is stated against every
  verifier, and these statements are what its proof reads.

* `TableReleaseAny`: one table, two outcomes. Where the ZK server's check passes, the view leaves, and the check passes
  only where the values pass the inner constraint at the drawn pads (`Table.Gated`); where it fails, only the pre-final
  view leaves. The released bytes are simulated per outcome, exactly, by `S_shvzk(x, e)` on a dummy `w₀` and the
  dummy's pre-final view.
* `TablesReleaseAny`: a release of `J` tables, the outcome being where it stopped: the tables whose files left show
  their views, each behind its check (`Release.Gated`), and the others their pre-final views. The released bytes are
  simulated per outcome, exactly, by the simulator at that outcome (`Release.simAt`).
-/

@[expose] public section

namespace Flock.Guarantees

open Verity.Game FlockSoundness.ZK
open scoped ENNReal

-- As in `SHVZK`: a table's randomness is a product of up to fifteen factors.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **One table's release behind the ZK server's check, at any values**: at coins non-degenerate for the table
(`Table.Simulable`), for a witness `w` and a dummy `w₀` carrying the public region data (`Table.RegPublic`), and a check
`go` on the server's randomness that passes only where `w`'s values pass the inner constraint at the drawn pads
(`Table.Gated`), and with no premise that they pass it anywhere, the released bytes (`Table.release`: the view where
`go` passes, the pre-final view where it doesn't) are simulated per outcome, exactly, by `S_shvzk(x, e)` on `w₀` where
the check passed and the dummy's pre-final view where it failed. -/
def TableReleaseAny : Prop :=
  ∀ {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
    [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL)
    (go : T.Rand → Bool) (w w₀ : W),
    T.Simulable → T.RegPublic w → T.RegPublic w₀ → T.Gated go w →
    FlockSoundness.ZK.PerOutcome go (T.release go w) (FlockSoundness.ZK.gatedSim (T.sim w₀) (T.preView w₀))
      fun _ => 0

/-- **A release of `J` tables behind the ZK server's checks, at any values**: at coins non-degenerate for each of its
tables, for witnesses `w` and dummies `w₀` carrying the public region data, an outcome `out ω` (where the release
stopped, a function of the server's randomness) and the tables whose files leave at each outcome (`full`), where a
table's file leaves only if its values pass the inner constraint at its drawn pads (`Release.Gated`), and with no
premise that they pass it anywhere, the released bytes (`Release.bytesAt`) are simulated per outcome, exactly, by the
simulator at that outcome (`Release.simAt`). -/
def TablesReleaseAny : Prop :=
  ∀ {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
    [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] {J : ℕ}
    (T : Fin J → FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) {St : Type}
    (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool) (w w₀ : Fin J → W),
    (∀ j, (T j).Simulable) → (∀ j, (T j).RegPublic (w j)) → (∀ j, (T j).RegPublic (w₀ j)) →
    FlockSoundness.ZK.Release.Gated T out full w →
    FlockSoundness.ZK.PerOutcome out (fun ω => FlockSoundness.ZK.Release.bytesAt T full w (out ω) ω)
      (FlockSoundness.ZK.Release.simAt T full w₀) fun _ => 0

end Flock.Guarantees
