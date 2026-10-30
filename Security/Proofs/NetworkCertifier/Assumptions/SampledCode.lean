import Proofs.Flock.Claims.Sampling.Deployed

/-!
# The sampled verifier's code

`SampledCodeRuns`, a named assumption: the sampled verifier the service runs (`verity/core/protocols/sampled_proofs`) is
the deployed sampling record at some `SamplingDeploy.Code` whose run meets the facts `Claims.Sampling.KS` takes of a
record's run: its draw precedes the sessions (`DrawFixed`), each checked unit's session is a Flock run (`SessionRuns`)
in Flock's masked scope (`SamplingDeploy.SessionsIn`), its slots are the draw law's (`DrawSlots`), each session opens
under interior roots the run fixes (`SessionOpens`, `InteriorFixed`), and its scope checks are the program's and the
checks' (`ProgramScope`, `ChecksScope`); and some statement is in its scope. The code is Python, of which the tree has
no Lean port, so these are facts about it no proof here can reach. The record's commitment (`SamplingDeploy.commit`:
hm96 leaves under the registered path check) and its sessions' Flock record (`FlockDeploy.deployZk`) are `deploy`'s, not
assumed.
-/

namespace NetTiming.Assumptions

open Claims Claims.Sampling

set_option synthInstance.maxSize 1024 in
/-- **The sampled verifier the service runs meets its record's facts**: it is `SamplingDeploy.deploy` at the deployed
relation and some code `c` with a statement in scope, whose run at SHA-512 meets the facts `Claims.Sampling.KS` takes
of a record's run. -/
def SampledCodeRuns : Prop :=
  ∃ c : SamplingDeploy.Code,
    (∃ x, Scope (SamplingDeploy.deploy _root_.Flock.Readable.relation c) x) ∧
      DrawFixed (SamplingDeploy.deploy _root_.Flock.Readable.relation c) ∧
      SessionRuns FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation c) ∧
      SamplingDeploy.SessionsIn c ∧ DrawSlots (SamplingDeploy.deploy _root_.Flock.Readable.relation c) ∧
      (∃ iroots : Msg → Seen → ℕ → List Msg,
        SessionOpens FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation c) iroots ∧
          InteriorFixed FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation c) iroots) ∧
      ProgramScope (SamplingDeploy.deploy _root_.Flock.Readable.relation c) ∧
      ChecksScope FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation c)

end NetTiming.Assumptions
