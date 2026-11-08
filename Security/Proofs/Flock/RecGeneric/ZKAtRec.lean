module

public import Proofs.Flock.RecGeneric.ZKSessions
public import Proofs.Flock.RecGeneric.ZKServerRec
public import Proofs.Flock.Claims.Recursion.Deploy

/-!
# The recursion's zero knowledge with Flock's ZK server as each V\* session

The compiled verifier's outer phase (`RecDeploy.deployRec`) is its V\* sessions: at the compiled statement `x`, its
`nS x` parts in turn, part `s` at its V\* statement (`RecDeploy.vstarAt`) off the rounds before the sessions, each over
the outer verifier's rounds (`RecDeploy.VO`, the masked session at the recursive setup). Flock's ZK server is zero
knowledge per outcome at the recursive outer record (`server_zkOutRec`), so its session system (`recServerSess`,
`ZkSess.ofZKOut` at finite sets `Xs x` of session statements) runs those sessions. `recLayout` lays them out: part `s`
at `vstarAt` of the rounds `msOf x_O` the outer statement fixes, and `Σ_s ((nums s).rounds + 1)` rounds in all, since
every V\* statement's rounds are its part's numbers' (`RecDeploy.numsOf_vstarAt`) and the server adds the `Hello`.

`GenericRecursiveZKAtRecServer` (Z1-F at `deployRec`): at hm96 leaves hiding within `δ₁` at the pinned key and the
server's premises as `server_zkOutRec` takes them, for every reading `msOf` of the rounds before the sessions off the
outer statement and every first part (`FirstPart`), the release of the first part and `deployRec`'s V\* sessions
(`ZkRelease.ofSessions`) is zero knowledge per outcome at every witness, within `ZkRelease.err` at the sessions' summed
`errOut` plus `δ₁` per leaf of the first part, at factor `1`.

A session statement outside `Xs x` has no witness in the session relation, so where a run's V\* statement is not in
`Xs x` the release refuses: zero knowledge holds at every `Xs`, and the release is a run's outer view only where `Xs x`
holds its V\* statements.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.SamplingGeneric

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Flock's ZK server at the recursive outer record as a session system** at the finite sets `Xs x` of session
statements: `ZkSess.ofZKOut` at `server_zkOutRec`'s prover, rounds, relation, outcome and simulator given the outcome. -/
noncomputable abbrev recServerSess (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData)
    (vs : (FlockZk.Sess T).SwVals) (L : FlockZk.Release T) (wk : L.toSrv.Worker) (pc : FlockZk.PC T)
    (fc : FlockZk.FC T) (t : ℕ) (Xs : Msg → Finset Msg) : ZkSess :=
  ZkSess.ofZKOut (FlockZk.LiveRelAt (RecDeploy.deployZkRec rel) T vs L wk pc fc)
    (FlockZk.roundsAt (RecDeploy.deployZkRec rel)) (FlockZk.Tape T) L.toSrv.Chk
    (fun w => L.toSrv.proverD L.fillOf (FlockZk.roundsAt (RecDeploy.deployZkRec rel)) wk w)
    (ZKServer.outD L.toSrv.stop (FlockZk.roundsAt (RecDeploy.deployZkRec rel)))
    (FlockZk.simOutAt (RecDeploy.deployZkRec rel) T L pc fc t) Xs

/-- **`deployRec`'s V\* sessions as a layout of the session system `S`**, at what the deployment brings (`b`) and a
reading `msOf` of the rounds before the sessions off the outer statement: at the compiled statement `x`, its `nS x`
parts, part `s` at its V\* statement off those rounds (`RecDeploy.vstarAt`), and `Σ_s ((nums s).rounds + 1)` rounds in
all, each part's numbers' rounds and the `Hello`. -/
noncomputable def recLayout (S : ZkSess) (b : RecDeploy.Brings) (msOf : Msg → List (Msg × Out)) : Sessions S where
  n := RecDeploy.nS
  xs x xO s := RecDeploy.vstarAt b.st x (msOf xO) s
  ko x := ∑ s ∈ Finset.range (RecDeploy.nS x), ((RecDeploy.nums b.st x s).rounds + 1)

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric Claims.SamplingGeneric FlockSoundness.Discharge
open scoped NNReal ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Z1-F at `deployRec`: the recursion is zero knowledge per outcome with Flock's ZK server as each V\* session.** At
an attempt budget `t ≥ 1`, wherever hm96 leaves hide within `δ₁` at the pinned key, and at the server's premises as
`server_zkOutRec` takes them (the coin layer's session facts, its completeness on the recursive outer record's relation,
and the release `L` the session loop builds): for every finite set `Xs x` of session statements, deployment's brings
`b`, reading `msOf` of the rounds before the sessions off the outer statement, encoding `enc` of a commit string's halves
and first part `F`, the release of `F` and `deployRec`'s V\* sessions (`recLayout`) is zero knowledge per outcome at
every witness, at its simulator given the released outcome, within `ZkRelease.err` at `Sessions.err` of the server's
`errOut`: each session's `errOut` at `V*` after the real prefix, averaged over the first part where the server releases
and over the earlier sessions' tapes, plus `δ₁` per leaf of the first part. -/
def GenericRecursiveZKAtRecServer : Prop :=
  ∀ (rel : FlockZk.Parser) (T : ZKMal.CoinLayer.TableData) (vs : (FlockZk.Sess T).SwVals) (t : ℕ) (δ₁ : ℝ≥0),
    0 < t → Claims.Hm96Hiding ZkView.My ZkView.saltHash (δ₁ : ℝ≥0∞) →
    (FlockZk.Sess T).Tables → (FlockZk.Sess T).Complete (Claims.Flock.R (RecDeploy.deployZkRec rel).parse) →
    (FlockZk.Sess T).DummyReg → (FlockZk.Sess T).Causal → (FlockZk.Sess T).SwCausal vs → (FlockZk.Sess T).TapeOK →
    ∀ (L₀ L : FlockZk.Release T), L = ZKServer.LeanRelease.loop ZKServer.Loop.loopNr T L₀ →
    ∀ (wk : L.toSrv.Worker) (pc : FlockZk.PC T) (fc : FlockZk.FC T) (Xs : Msg → Finset Msg) (b : RecDeploy.Brings)
      (msOf : Msg → List (Msg × Out)) (enc : ZkView.Bits × ByteArray → Msg) {W : Type}
      (F : FirstPart W (Leaves.hm96 enc) (ℕ → (recServerSess rel T vs L wk pc fc t Xs).WS)),
      ZKOut (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).N
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).Ω FlockZk.sha
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).k
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).prover (fun _ _ => True)
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).outcome
        (ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).sim
        ((ZkRelease.ofSessions F (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf)).err FlockZk.sha
          ((recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf).err FlockZk.sha
            fun _ => FlockZk.errOut T vs L pc fc t (δ₁ : ℝ≥0∞)) (δ₁ : ℝ))

end Flock.Guarantees
