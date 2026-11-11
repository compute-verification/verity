module

public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.ZkAny.Table
public import Proofs.Flock.ZkAny.Unswitched
public import Proofs.Flock.ZkAny.Deferred
public import Proofs.Flock.ZkAny.UnswitchedZk
public import Proofs.Flock.ZkAny.RealLeaves
public import Proofs.Flock.ZkAny.Tables
public import Proofs.Flock.ZkAny.Gate
public import Proofs.Flock.ZkAny.Refused
public import Proofs.Flock.ZkAny.Relay
public import Proofs.Flock.ZkAny.RelayFrames
public import Proofs.Flock.ZkAny.RelayOut
public import Proofs.Flock.ZkAny.RelayEnv
public import Proofs.Flock.ZkAny.RelayReg
public import Proofs.Flock.ZkAny.RelayStay
public import Proofs.Flock.ZkAny.RelayRound
public import Proofs.Flock.ZkAny.RelayFold
public import Proofs.Flock.ZkAny.RelayMask
public import Proofs.Flock.ZkAny.RelayStop
public import Proofs.Flock.ZkAny.RelayRelease
public import Proofs.Flock.ZkAny.RelayCoins
public import Proofs.Flock.ZkAny.RelayFirst
public import Proofs.Flock.ZkAny.RelayLive
public import Proofs.Flock.ZkAny.RelaySession
public import Proofs.Flock.ZkAny.RelayParams
public import Proofs.Flock.ZkAny.RelaySched
public import Proofs.Flock.ZkAny.RelaySlot
public import Proofs.Flock.ZkAny.RelayRun
public import Proofs.Flock.ZkAny.RelayMark
public import Proofs.Flock.ZkAny.RelayInv
public import Proofs.Flock.ZkAny.RelayNext
public import Proofs.Flock.ZkAny.RelayCarry
public import Proofs.Flock.ZkAny.RelayFollow
public import Proofs.Flock.ZkAny.RelaySwitch
public import Proofs.Flock.ZkAny.RelayProg
public import Proofs.Flock.ZkAny.RelayPlace
public import Proofs.Flock.ZkAny.RelayWords
public import Proofs.Flock.ZkAny.RelayCount
public import Proofs.Flock.ZkAny.RelaySame
public import Proofs.Flock.ZkAny.RelayPrior
public import Proofs.Flock.ZkAny.RelayLast
public import Proofs.Flock.ZkAny.RelayGap
public import Proofs.Flock.ZkAny.RelayAlg
public import Proofs.Flock.ZkAny.RelayFresh
public import Proofs.Flock.ZkAny.RelayLoop
public import Proofs.Flock.ZkAny.RelayLoopCommit
public import Proofs.Flock.ZkAny.RelayLoopFinish
public import Proofs.Flock.ZkAny.RelayLoopRound
public import Proofs.Flock.ZkAny.RelayLoopLink
public import Proofs.Flock.ZkAny.RelayLoopJ
public import Proofs.Flock.ZkAny.RelayLoopKept
public import Proofs.Flock.ZkAny.RelayReps
public import Proofs.Flock.ZkAny.RelayRoundFirst
public import Proofs.Flock.ZkAny.RelayLoopSalts
public import Proofs.Flock.ZkAny.RelayLoopFacts
public import Proofs.Flock.ZkAny.RelayLoopCur
public import Proofs.Flock.ZkAny.RelayLoopFirst
public import Proofs.Flock.ZkAny.RelayLoopOwn
public import Proofs.Flock.ZkAny.RelayLoopNext
public import Proofs.Flock.ZkAny.RelayLoopRep
public import Proofs.Flock.ZkAny.RelayLoopL0
public import Proofs.Flock.ZkAny.RelayLoopDeployed
public import Proofs.Flock.ZkAny.RelayLoopPre
public import Proofs.Flock.ZkAny.RelayLoopInner
public import Proofs.Flock.ZkAny.RelayLoopOwnAt
public import Proofs.Flock.ZkAny.RelayTook
public import Proofs.Flock.ZkAny.RelayRank
public import Proofs.Flock.ZkAny.RelayTaken
public import Proofs.Flock.ZkAny.RelayRoundTook
public import Proofs.Flock.ZkAny.RelayLoopStart
public import Proofs.Flock.ZkAny.RelayServes
public import Proofs.Flock.ZkAny.RelayRankStop
public import Proofs.Flock.ZkAny.RelayOpenFirst
public import Proofs.Flock.ZkAny.RelayRankSched
public import Proofs.Flock.ZkAny.RelayFreshStart
public import Proofs.Flock.ZkAny.RelayWorker
public import Proofs.Flock.ZkAny.RelayAuditor
public import Proofs.Flock.ZkAny.Finish
public import Proofs.Flock.ZkAny.SampledAudit
public import Proofs.Flock.ZkAny.Stream
public import Proofs.Flock.ZkAny.RecordHiding
public import Proofs.Flock.ZkAny.SampledSessions
public import Proofs.Flock.ZkAny.SampledAuditTwo
public import Proofs.Flock.ZkAny.SampledSessionsTwo
public import Proofs.Flock.ZkAny.SampledArgTwo
public import Proofs.Flock.ZkAny.SampledChosenTwo

/-! Zero knowledge at any values: the generic forms, a table's and a release of `J` tables behind the ZK server's
checks, the ZK server's check as the gate a table's file leaves behind, a release at coins it refuses, the release
before the `Finish` held to the unswitched prover, the deferred release per outcome at its `Finish` verdict, the
unswitched prover's zero knowledge at any values, the go prover's one-sided zero knowledge and the release split at
the `Finish` on both outcomes, the release with real hm96 leaves on both branches, the deployed relay's release held
to the session's schedule, its
model frames held to the model's messages, what it reads at the model's `Hello` and `Commit` as the model's, its
release per outcome, read through `ZKOut`, and the loop's frames between a session's requests: each rep's frame
written by its own rounds and coins alone, its session's entry and salts kept, each round at `placeOf`'s place, the
frame its run on the loop's requests to it, its masking state the model's but for `stopped`, and the session's stop,
which the other sessions' requests, the frames port's and a round's coins keep; and a round the frame releases at the
model's place as the model's release, stopping the session where the model's does, after its stream's domain frame at
the stream's first round in the schedule: where the session's stop is whether an earlier round failed, the deferred
model's message; and that stop from one request to the next, the model's at the session's first frame round and
carried by a round and its coins to its next request; and a frame round's coin count, its frame's configuration and
its scheduled round, the model's; and so what the relay read for a frame round at the model's place, the deferred
model's message, with the session's stop at its next request the model's one round on; both at a frame round as the
loop ran it, its reads and puts of the loop's frames with only the updates between the session's requests between
them; and the loop's mark of a stream's first round, the schedule's; and `EnvPre` from any invariant of the relay's
requests that each slot's step keeps; and a rep's frame at the session's next round, its run one round on, at the
model's masking state for that round but for `stopped`, which the rep's next round reads, not its stream's first;
and at a rep switch the model's state the next rep's start, rep 0's frame starting there; and from one round of a
rep past its first to the same rep's next, at the model's configuration and state for the round, the frame the
session's next round finds at the model's state for that round; and past a stream's first round the readied
frame `placeOf`'s, its round in the place its schedule so far gives and the frame after it keeping that schedule;
and a rep's frame holding the history's words for its rounds so far from one round of the rep to the next; and its
configuration kept, and a round's count its frame's schedule's where it is the placeholder's; and so, from one
round of a rep to the same rep's next, the rep's frame at the model's configuration, masking state and words, keeping
its schedule so far, its round in that schedule's place; and at a rep 1 stream's first round the readied frame its run
on rep 0's algebraic-phase words, at the model's state for that round; and after a rep's round, whichever rep's
round comes next, its frame its run, holding the history's words for its rounds so far and keeping its schedule; and
after a stream's last round, until the next stream's first, the rep's frame kept, holding the history's words for all
the stream's rounds, as many as its count; and on a frame with no scheduled rounds yet, its algebraic phase as many
rounds as the placeholder's; and at a stream's first round on a frame as its start pushed it, its round in the place
its schedule so far gives and the frame after it keeping that schedule; and `EnvPre` from the relay's invariant at
the deployed loop against the model's run, the model's messages the slots took one more than the openings the
auditor answered, given its step at the `Commit`, each frame round and the `Finish` and an invariant of the loop kept
at every other slot, the step at the `Hello` and at every other slot proved; and the step at the `Commit`, where the
ZK server's own link is at the level 0 the loop drew for the session, and at the `Finish`, where the session's stop is
whether an earlier round of the model's run failed; and a frame round at its slot before a stop, which the session
lets through, and the step at it, given what the loop's frames and the model hold there and the invariant kept; and
the session's reps and labels the setup's rep streams' once its `Hello` left, so a round's rep its stream's own and no
other rep stream's; and after a stream's first round the rep's frame its run, holding the history's words for its one
round and keeping its schedule so far, which a stream of one round hands to the next stream's first, and at the model's
state where the next round is the same rep's; and the rep's salts for levels ≥ 1 in the frames a round read wherever
`placeOf` reads them, so a frame keeping `SameInv` keeps it at those salts; and that invariant at the deployed loop, the
session's stop and each seen rep's frame, entry and level 0 against the model's history, its plan the setup's and the
streams seen those before the model's next round, kept at every slot that reads no model message and at the `Hello`,
and at a frame round given what the loop's frames and the model hold there and the round's own rep one round on; and
what the loop's frames and the model hold at a frame round, from that invariant on the deployed loop's model, given
the model's answer there and what it holds at the round's place, past a stream's first round the model's scheduled
round where the model's tables and salts are the loop's; and past the `Commit` the session's entry linked to the
coins answering the model's link, at the setup's link points; and past a rep's schedule so far, at the inner proof's
rounds, the rep's frame at the session's next round keeping `SameInv` one round more of the rep; and at the first
round of a stream of one round, the rep's frame at the session's next round holding the history's words for it; and so,
the setup's sizes within its frames' from its `Hello` on, at every frame round of the deployed loop the round's own rep
one round on; and the invariant kept at a frame round of the deployed loop at its own tables, given what the loop's
frames and the model hold there; and so `EnvPre` from that invariant at the deployed loop's own tables, given what the
loop's frames and the model hold at each frame round; and past a rep's schedule so far the masking state's `ε` and `β`
the first words of the coins of the inner proof's rounds 0 and 2, all `Frame.innerSched` reads of it; and so, at the
deployed model with the session loop's own rounds of the inner proof, where `placeOf` places a round of a stream seen
at the inner proof's own, the round's place the model's scheduled round, given each round of the inner proof the
loop's finish computed so far the model's finish's; and so `EnvPre` from that invariant at the deployed loop with the
session loop's own rounds of the inner proof, where the release takes the run's coins and the crossing passes below
the `Finish`; and what the loop took of the coins so far against the model's release: the transport's verdict, the
frame's taking of each earlier round's coins and its coin step on the last; and the model's taking of the coins
before its rank stop, which a frame releasing an opening round passes; and so the model's taking of the coins after a
frame round's history, from what the loop took, a rep's first-round state and the rank stop at the opening round the
frame releases; and that first-round state a masking state at the deployed loop on a `Live` statement, where the rep's
frame starts at its table's statement, the placeholder's table of the shape of the loop's and its algebraic phase one
as the loop's; and the deployed worker, a device's rounds at the frame rounds and the ZK server's own message at every
other round, the honest workers among them, answering the link and the `Finish` with the ZK server's own message; and
Rust's checks the worker's, so `CrossPasses` by definition at the release without them beside the frame; and the
relay's auditor as `V*`, which answers as `V*` does at the `Hello`, the `Commit` and each frame round; and so, from the
session's first request, at the release without Rust's checks and the model's run of the deferred server at the
deployed worker against the relay's auditor, `EnvPre` at the relay's requests, and the relay's model frames the
model's messages, encoded; a sampled audit's view, the registration's record with the sampled sessions' releases;
sessions in sequence per outcome; the registration's ideal record hiding its values at hm96 commitments; the view at
sampled sessions of one protocol that meets `ZKOut`; a two-stage audit's view, a stream of registrations whose
values follow the first-stage sample, with its ideal stream's hiding and its sessions one protocol's or a proof
system for proof blocks'; and that view where the auditor chooses the first-stage sample from the boundary's
record, the interior a stream under a key of its own. -/

@[expose] public section
