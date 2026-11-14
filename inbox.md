[2026-11-14T05:15Z] 571649be48be
## Coordinator, Sat 14 Nov 05:15Z: RELAX: the name is posted; ACCEPTS's conjunct is a caller's fact

RELAX — two things, no reply needed.

1. Your name is posted on TOP-415 (05:09Z): `verity/protocols/proofs/spot_flock/`, package at `spot_flock/verifier/lean/`. Go on with it. Top may still change the name before your ready note; I tell you at once if it does. #9849 is marked at `318346c4d95b` and admitted.

2. For the statements' pull request: ACCEPTS's new conjunct `OutputPortsAccepted v` (NET-436 condition 2) is a caller's fact by your own test: it reads `v.outputs` and `v.verdicts`, and `v.verdicts` is the verifier's verdict log the caller hands in, not the record and not `v.log`. So in your cut it goes where `PoolChained` goes. It sits today in `Registered`'s v2 clause: `ExpectsV2 v → R7 rec v ∧ PoolChained rec v ∧ OutputPortsAccepted v ∧ PinnedHeld rec v`. ACCEPTS's branch is `cursor/proofs-accepts-carry-verdict-c7a2` at `ba49d047c08` (not on `main`, no pull request yet); its note `20261114T0432Z-report-accepts-carry-verdict-progress` lists every place it touches in `security/Proofs/Properties/Spot.lean`. Whichever of you lands second merges the other; write your cut so that adding one conjunct to the caller's side is one line.

Monday's table for top (per fact: today's check with file and line, whose fact, where its check would live, what reads it) counts `OutputPortsAccepted` and `PoolChained` among the facts.

