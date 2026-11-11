module

public import Proofs.Flock.SessionLoop.Statement

/-!
# `LoopInnerRound0Own`'s fixture: a deployed session loop reaches a rep's inner proof

`LoopInnerRound0` speaks only where the history stands at a rep stream's first round past its scheduled rounds: where
the statement sets up a session, the rep's table and salts are the loop's own (`Loop.realOf`), and its schedule so far
(`Loop.schedSoFar`) ends one round before. Wherever the history reaches no such round, `LoopInnerRound0` holds at every
release, so `LoopInnerRound0Own`'s witnesses need one. What the kernel can't build is the reading: `Loop.realOf` commits
a table's level 0 (`Finish.tabOfLevel0`) with SHA-512 trees over the rows of a deployed table, and its source `src` is
the host's packed witness and the operating system's bytes, which no Lean term is. So the named assumption
`Loop.ReachesInner` says only that one deployed run reaches such a round.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- the session's tape types, as `LoopInnerRound0`'s in `Statement`

/-- **A deployed session loop reaches a rep's inner proof**: at some level-0 source `src`, statement `x`, witness `w`,
tape `ρ` and history `h`, `LoopInnerRound0`'s premises hold at the deployed tables (`Masked.dataE`) and the loop's own
tables (`Loop.realOf src`): `x` sets up a session (`FlockDeploy.setupOf true`), `h` stands at rep round `q`, rep stream
`k`'s round `j + 1` (`CoinLayer.locate`), the rep's table and salts are the loop's (`Loop.realOf`), and its scheduled
rounds so far (`Loop.schedSoFar`: the algebraic phase's, then the opening's) are `j + 1`, so the round is the inner
proof's round 0.

Assumption record. No claim: evaluations of deterministic deployed code on one run's bytes, which say nothing about the
prover or the world. It is witnessed outside Lean by a run of `verity/ml/flock/verifier/firewall_finish_agree.sh` at
5a7e015281, at its default statement (`hidden:256`, one instance, sampled): one firewalled CPU `prove --zk` session
(`FC_FIREWALL=1`, `FC_HOST_WITNESS=1`) through that tree's Lean session loop, in which every round in a device round's
place is the loop's own (`Skeleton.rounds`, then `Drive.openStep`, which `Loop.schedSoFar` reads) and the finish holds
each rep's opening's and inner proof's rounds itself, on its table's level 0 from the loop (`Finish.tabOfLevel0`, which
`Loop.realOf` reads). The session had one table and two reps. The table's root of record and its finish's were one, each
rep left its proof file, the session's live verifier accepted with both reps passing, and `flock-verify verify --zk`
passed every check, refusing only for want of the verifier's Program (`ProvedScope.NO_PROGRAM`, its last check), so its
setup of the statement passed. A rep's inner proof leaves only past its scheduled rounds, so the loop's history passed
each rep stream's inner proof's round 0. The run is weaker than the Prop in two ways: no Lean term `x`, `w`, `ρ`, `h` or
`src` is formed, so the readings are the compiled loop's and verifier's on the run's files rather than `setupOf`,
`locate`, `realOf` and `schedSoFar` evaluated at them; and `src` is the run's host witness and operating-system bytes,
which the tape models (`Loop.Src`), not a function Lean evaluates. -/
def Loop.ReachesInner : Prop :=
  ∃ (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (h : List (Msg × Msg)) (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (q k j : ℕ)
    (tr : _root_.Flock.Firewall.Finish.Tab) (hr : Array ByteArray) (rsched : Array (ByteArray × ℕ)),
    FlockDeploy.setupOf true x = some (s0, spec, sts) ∧ h.length = q + 2 ∧
    CoinLayer.locate (Loop.roundsOf spec sts) 0 q = (k, j + 1) ∧ Loop.realOf src x w ρ k = .ok (tr, hr) ∧
    Loop.schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) 0))) hr
      ((Array.range (j + 1)).map
        fun i => Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched ∧
    rsched.size = j + 1

end Claims.ZKServer
