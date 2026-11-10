module

public import Proofs.Flock.RecGeneric.ZKSessions
public import Proofs.Flock.SamplingGeneric.FlockZk

/-!
# The recursion's zero knowledge with Flock's ZK server as each session

Flock's ZK server is zero knowledge per outcome (`Claims.SamplingGeneric.FlockZk.server_zkOut`): on its tape
`Tape T × Chk`, against every verifier of its rounds (`FlockZk.rounds`: the masked verifier's and the `Hello`), on every
witness in the relation with the live path's premises (`FlockZk.LiveRel`), at its simulator given the outcome
(`FlockZk.simOut`) and within `FlockZk.errOut`. Its simulator's coins are its statement's, so its session system is
`ZkSess.ofZKOut` at finite sets `Xs x` of the session statements each compiled statement admits (`flockServerSess`).

`GenericRecursiveZKAtFlockServer` (Z1-F): at hm96 leaves (`Leaves.hm96`, any encoding of a commit string's halves)
hiding within `δ₁` at the pinned key, the server's premises as `server_zkOut` takes them, every first part (`FirstPart`)
and every layout of the server's sessions that makes up the outer phase's rounds (`Sessions.Shaped`), the release
(`ZkRelease.ofSessions`) is zero knowledge per outcome at every witness, within `ZkRelease.err` at the sessions' summed
`errOut` (`Sessions.err`) plus `δ₁` per leaf of the first part, at factor `1`. It is Z1 at sessions
(`GenericRecursiveSessionsZK`), Z1-L (`GenericSessionsOfZKOut`) at `server_zkOut`, and Z2 (`Hm96Hides`).

The sessions are those of Flock's masked deployment (`FlockDeploy.deployZk rel`), whose parser reads no recursive
statement (`RecDeploy.RecRaw.encode`'s format is not `FlockDeploy.Raw`'s): at a layout of V\* statements
(`RecDeploy.deployRec`'s) no session witness is in the relation, and the release refuses there. Z1-F at `deployRec`'s
V\* sessions is the same server at the recursive outer record (`Flock.Guarantees.GenericRecursiveZKAtRecServer`,
`Proofs.Flock.RecGeneric.ZKAtRec`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.SamplingGeneric

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Flock's ZK server as a session system** at the finite sets `Xs x` of session statements: `ZkSess.ofZKOut` at
`server_zkOut`'s prover (`proverD` on `Tape T × Chk`), rounds, relation (`LiveRel`), outcome (`outD`) and simulator
given the outcome (`simOut` at the attempt budget `t`). -/
noncomputable abbrev flockServerSess (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData)
    (vs : (FlockZk.Sess T).SwVals) (L : FlockZk.Release T) (wk : L.toSrv.Worker) (pc : FlockZk.PC T)
    (fc : FlockZk.FC T) (t : ℕ) (Xs : Msg → Finset Msg) : ZkSess :=
  ZkSess.ofZKOut (FlockZk.LiveRel rel T vs L wk pc fc) (FlockZk.rounds rel) (FlockZk.Tape T) L.toSrv.Chk
    (fun w => L.toSrv.proverD L.fillOf (FlockZk.rounds rel) wk w) (ZKServer.outD L.toSrv.stop (FlockZk.rounds rel))
    (FlockZk.simOut rel T L pc fc t) Xs

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric Claims.SamplingGeneric FlockSoundness.Discharge
open scoped NNReal ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Z1-F: the recursion is zero knowledge per outcome with Flock's ZK server as each session.** At an attempt budget
`t ≥ 1`, wherever hm96 leaves hide within `δ₁` at the pinned key, and at the server's premises as `server_zkOut` takes
them (the coin layer's session facts and the release `L` the session loop builds): for every finite set `Xs x` of
session statements, encoding `enc` of a commit string's halves, first part `F` and layout `Ls` of the server's sessions
that makes up the outer phase's rounds, the release of `F` and `Ls` is zero knowledge per outcome at every witness, at
its simulator given the released outcome, within `ZkRelease.err` at `Sessions.err` of the server's `errOut`: each
session's `errOut` at `V*` after the real prefix, averaged over the first part where the server releases and over the
earlier sessions' tapes, plus `δ₁` per leaf of the first part. -/
def GenericRecursiveZKAtFlockServer : Prop :=
  ∀ (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData) (vs : (FlockZk.Sess T).SwVals) (t : ℕ) (δ₁ : ℝ≥0),
    0 < t → Claims.Hm96Hiding ZkView.My ZkView.saltHash (δ₁ : ℝ≥0∞) →
    (FlockZk.Sess T).Tables → (FlockZk.Sess T).Complete (Claims.Flock.R (FlockDeploy.deployZk rel).parse) →
    (FlockZk.Sess T).DummyReg → (FlockZk.Sess T).Causal → (FlockZk.Sess T).SwCausal vs → (FlockZk.Sess T).TapeOK →
    ∀ (L₀ L : FlockZk.Release T), L = ZKServer.LeanRelease.loop ZKServer.Loop.loopNr T L₀ →
    ∀ (wk : L.toSrv.Worker) (pc : FlockZk.PC T) (fc : FlockZk.FC T) (Xs : Msg → Finset Msg)
      (enc : ZkView.Bits × ByteArray → Msg) {W : Type}
      (F : FirstPart W (Leaves.hm96 enc) (ℕ → (flockServerSess rel T vs L wk pc fc t Xs).WS))
      (Ls : Sessions (flockServerSess rel T vs L wk pc fc t Xs)),
      Ls.Shaped →
        ZKOut (ZkRelease.ofSessions F Ls).N (ZkRelease.ofSessions F Ls).Ω FlockZk.sha (ZkRelease.ofSessions F Ls).k
          (ZkRelease.ofSessions F Ls).prover (fun _ _ => True) (ZkRelease.ofSessions F Ls).outcome
          (ZkRelease.ofSessions F Ls).sim
          ((ZkRelease.ofSessions F Ls).err FlockZk.sha
            (Ls.err FlockZk.sha fun _ => FlockZk.errOut T vs L pc fc t (δ₁ : ℝ≥0∞)) (δ₁ : ℝ))

end Flock.Guarantees
