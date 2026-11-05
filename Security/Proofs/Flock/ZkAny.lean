module

public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.ZkAny.Table
public import Proofs.Flock.ZkAny.Unswitched
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
public import Proofs.Flock.ZkAny.RelayLoopRound

/-! Zero knowledge at any values: the generic forms, a table's and a release of `J` tables behind the ZK server's
checks, the ZK server's check as the gate a table's file leaves behind, a release at coins it refuses, the release
before the `Finish` held to the unswitched prover, the deployed relay's release held to the session's schedule, its
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
at every other slot, the step at the `Hello` and at every other slot proved; and a frame round at its slot before a
stop, which the session lets through, and the step at it, given what the loop's frames and the model hold there and the
invariant kept. -/

@[expose] public section
