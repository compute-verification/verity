# The sampled leg over a window of m members

Scratch notes for PRF-490, on `cursor/soundness-multileg-8fb2` only. This file never goes in the PR.

Base: `cursor/soundness-remote-deployed-8fb2` (#8569, `remote_soundness_deployedReady`, the one-member form). Everything
is in namespace `Verity.Sampled`, package `security_proofs`, under
`Security/Proofs/Properties/Verity/Sampled/` (`Window.lean` and `Window/*.lean`), registered in
`Security/Proofs/lakefile.toml` and `Security/Proofs/lean-audit.json`. One worker (LEG).

## The statement

`Window/Instance.lean`:

```lean
theorem remote_soundness_window (I : WindowInstance d α) (K₀ : WindowConstructions I.base)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hC : I.ClaimTransfers (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest))
    (hRg : I.RegisteredFiles) (hWa : I.WindowAt .remote) (hWR : I.WindowReads .remote ev filesDec)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest)).SoundnessAt .remote
```

Same shape as the one-member `remote_soundness_deployedAt` (`Sampled/Instance.lean`): `RemoteSoundness`
(`remote_soundness_of_subprotocols`) with its sampled leg `WindowInstance.spotSound`, which is
`windowSpotSound_sessions` at the deployed vocabulary. The sampled leg's error at a window is `Window.delta`:
`W.drawMiss + (W.joint t).eps` (`JointShareV5`: the one sample's own miss and `Σ_k (ε_ks,k + ε_bind,k)` at member
`k`'s part of the one sample), one two-continuation term per round of every member's session
(`W.rounds = Σ_m m.rounds`, each member's closing round among them: 1026 a member at the deployed table), the
record's, the bound digests', the hiding term, and the read finder's bound.

`WindowInstance` (the deployment over windows, before SPOT's names): `base : Vocab d α`, `deployment`,
`signer` (the base's signer is the deployed device's), `costs : CostsRead`, `window : ByteArray → Option Window`,
`SetsUp : ByteArray → Array ByteArray → Window → Prop` (the files the verifier sets that window's sessions up on),
`lb` (a set-up window reads `MAX_COINS` frames' bytes), `members` (`MembersAt W W.table`: each member has the one
table, hm96-sha512 rows).

How it is put together, each layer proved once:

| Layer | Theorem | File |
|---|---|---|
| the bound: `JointShareV5`'s, applied at `Window.joint`, not proved again | `Window.ks`, `spotSound_of_window` | `Window.lean` |
| the projection through m sessions in order, under its joint law | `prob_seqP`, `seqOff_le_prob`, `windowLogged`, `windowCouplingTarget_of_logged` | `Window/Walk.lean` |
| the run-of-record walk a member at a time | `SeqReach.chain`, `SeqChain.forall₂`, `windowDecodes_of_members` | `Window/Reach.lean` |
| what a member's trace keeps and logs | `MemberReach.stopsAt`, `MemberReach.logged`, `MemberSteps`, `memberDecodes_of_stops` | `Window/Logged.lean` |
| the draw to member subsets and its marginals; the coin map's premise over the window | `Member.mem_part`, `Window.sum_part_card`, `Window.avg_design`, `Window.eps_extend`, `windowCouples_of_close`, `windowSampleMap_fresh`, `windowSampleMap_start` | `Window/Draw.lean` |
| the stops' term within the hiding term at every strategy | `seqErr_le`, `windowErr_sampleMap_le_at`, `windowCouplingTarget_of_sample_at` | `Window/Charge.lean` |
| the composition | `windowSpotSound_of_sample` | `Window/Leg.lean` |
| (a), (a') and the sample's coins | `windowSpotSound_assembled` | `Window/Assembled.lean` |
| v2's acceptance as m verdicts in order | `Verdicts`, `run_verdicts_at`, `Verdicts.getElem` | `Window/Accepts.lean` |
| the sessions held one after another, in the device's log | `OpensInOrder`, `EvidenceRecorded`, `SessionsInOrder` | `Window/Order.lean` |
| a session's verdict is `--zk`'s acceptance of what the member's writer writes | `SessionVerdict.accepts_member`, `StatesMember` | `Window/Writes.lean` |
| the members' answered stops, in order in the play's log | `Walked`, `Walked.next` | `Window/Answered.lean` |
| the readings, a session at a time (the `u`-th tree opened is the `u`-th session's) | `openedTrees`, `sessionRound`, `closeSlot` | `Window/Sessions.lean` |
| the rounds a session's trace leaves on one stream; which session each answered release is of | `session_stream_logs`, `SessionPicks`, `OpensLast` | `Window/Streams.lean`, `Window/Trees.lean` |
| a session's moves are its record's (roots, link points, rep rounds, the closing round's evidence) | `SessionRead`, `sessionReveal`, `MemberMovesOf` clauses | `Window/Record.lean` |
| a stop's frame on any tape through it, session counts | `stop_frame_through`, `session_card` | `Window/Counts.lean` |
| a session's read with its closing opening dropped | `DropPre.reads`, `SessionOpens.drop_iff`, `SessionRead.of_dropped` | `Window/Closing.lean` |
| **the members' steps from each session's read** | `memberSteps_of_sessions` (premises `MembersAt`, `WindowSessions`) | `Window/Steps.lean` |
| **the sessions of the window's one sample, at the sample-coupled coin map** | `windowSessions_firstStop`, `memberSteps_sampleMap`, `windowSpotSound_sessions` (premise `WindowRead`) | `Window/Read.lean` |
| **the deployed vocabulary over windows** | `WindowInstance.deployed`, `.covers`, `.spotSound`, `remote_soundness_window` | `Window/Instance.lean` |

## What each premise is

Discharged inside the statement:

| What | By | Rests on |
|---|---|---|
| the sample's coins are uniform | `Window.avg_srcCoinsAt`, from `Window.hA3` | uniform randomness (A3 at the device's bytes) |
| (a) the log v2 read is the device's and the pinned tables are the registered files | `Binding.logged_tables_le`, `Binding.binds_of_prints` | SHA512CR (the read finder), LmsUnforgeable (the forger; remote only), the device's log (`Device.printsLog_dev`, `Device.signsPlay_dev`) |
| (a') v2 samples at the strategy's `afterAt` | `prob_not_sampledDigest_le` at `windowBindFinder` | SHA512CR (a two-continuation finder) |
| the coin map's premise at the sample-coupled map; the stops' term within the hiding term | `windowSampleMap_couples_at`, `windowErr_sampleMap_le_at` | nothing further |
| each round's reading picks only openings | `windowReadingAt_picks` | nothing further |
| **`MemberSteps`** (each member's step: `--zk` accepts what the member's writer writes for its session's outcome) | `memberSteps_of_sessions`, `memberSteps_sampleMap` at the invariant `StepInv` | `WindowReads` below |
| `WindowDeployed.finder`, `.forger`, `.loss`, `.finderCR`, `.delta` | `WindowInstance.deployed` (`Window.delta_extend`) | SHA512CR per finder, in `error`'s condition (`JointFinderCR`, the strict birthday bound at the two-continuation finders) |
| `WindowCovers` | `WindowInstance.covers` | `RegisteredFiles`, `WindowAt` below; SHA512CR at the two file finders |
| the bound | `Window.ks` (`JointShareV5`) | SHA512CR per finder |

Left as hypotheses of `remote_soundness_window`:

| Premise | What it is | Whose |
|---|---|---|
| `hWR : I.WindowReads .remote ev filesDec` | at each play the sampled verdict accepts on files the verifier sets one of the windows up on (and (a), (a') hold): `WindowRead`, i.e. every release of the sample's family precedes every opening, and where the play's coins of that family are the window's bytes at `ω`, `WindowSessions` at the window's sample at `ω`: the log opens at most as many trees as drawn members, each naming its streams once, an opening of a tree opened before is of the last tree opened, and each drawn member has a `SessionVerdict` at a statement that is its own (`StatesMember`) whose record carries the member's part of the sample, is the record sent for its session, decodes to what the verifier's read holds the log to (`SessionRead`), on a tree the log opens, with proof files decoded from the evidence its closing round reveals | the per-member verifier's (the port, PRF-371): what `membersRun`'s acceptance gives of a play. Not provable on this base: v2 here is `oneSession` |
| `hWa : I.WindowAt .remote` | committed files that name the registered circuit and public files and that some play's sampled verdict accepts are files the verifier sets the window up on (the one-member `SessionAt`'s shape) | the port's, with `SetsUp` |
| `hRg : I.RegisteredFiles` | the registered files pass the file check wherever some files do | the registration's (the one-member form takes the same; `registeredFiles_of` proves it at `registeredOf`) |
| `hC : I.ClaimTransfers …` | `WindowDeployed.claim`: where the extracted transcript fails the claim on the committed files at the published share, more than the window's `ε` share of its population's proof blocks is incorrect at the projected strategy (`JointAuditV5.wrongSet`) | **open, needs soundness's decision** (below) |
| `hW`, `hI` | PoUW's and the inference-only policy's targets at the base | the other legs' (as in the one-member form) |

Trust base when those close: SHA512CR, uniform randomness, the honest verifier server's log premises;
LmsUnforgeable remote only. Nothing else is introduced.

## What is left

1. **The claim's transfer (`ClaimTransfers`).** `V.read θ F` is one statement's files: `FileRead` holds one circuit
   file and one public file, `programDigest`/`publicDigest` are single digests, and `partsOf`, `Opens`, `extract`
   are over `(V.read θ F).circuit`. A window's members each have their own circuit file and public file
   (`Member.I`). So `¬ IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) ε` cannot be carried to
   `JointAuditV5.wrongSet` (`⋃_k (memberWrong τ k).image emb`) for several members with different circuits without
   the vocabulary's reading of several members' files. Two shapes: (i) the vocabulary reads the window's files as
   one circuit, the members' summed, with `partsOf` the members' parts through `emb`; (ii) the claim is stated a
   member at a time. Either is a statement in `Security/Properties/Verity/Game.lean`, soundness's. Then the proof
   needs the several-member analog of the one-member `Live.ks` route (`unbound_subset_faulty_part`,
   `AssemblyTargetV5`'s `faulty ⊆ wrongRegZc`, `OpensUnder`), which Flock has only at one member. Asked on PRF-490.
2. **`WindowReads` and `WindowAt` from the per-member verifier** (the port, PRF-371; done-when 2). On this base v2
   is `Spot.OneStageZk.oneSession`; the port's `membersRun` (`heldStep`, `laidHeld`, `membersTreesHeld`,
   `sessionEvidence`) still refuses (`checkMembersAt_refuses`, `MULTI_REFUSED`). The gate is lifted there, for the
   per-member form only, with `WindowReads` proved of its acceptance. `WindowSessions`' shape matches the port's
   form at `ad6e765e05`: one session per drawn member, in the members' order, the closing round's evidence the
   proof files' bytes. The Python mirror and difftest change with the gate, there.
3. **The window's own class** (`row-table` roots and `registered` values; soundness's 03:35Z ruling, my answer of
   03:53Z): `MemberSteps` holds at such a window but says nothing of the class. Covering it takes proofs'
   several-member `EndToEndRegisteredRows`, then a clause on `StatesMember` and the step's conclusion. Soundness
   chooses (a) accept with the open premise named, or (b) refuse until then; I build for (b).
4. **Records.** The fast path for Security and Security/Proofs at the final head.
5. `Window.hcls` stays a field of `Window` (`JointAuditV5.hcls_of_bounds` discharges it from bounds).

## Rulings in force (PRF-490)

- 19:35Z Nov 10 (soundness): the sessions are held one after another, ordered by the honest verifier server's log,
  the nonce chain only as a fallback; (B), a member at a time, is out.
- 00:48Z Nov 11 (soundness): the closing round is in the member's own coin tree (1026 rounds a member).
- 03:35Z Nov 11 (soundness): member m's roots are the `roots` entries whose `port` is `<m's key>/<port>`; a window
  with row tables is refused in the per-member form until the statement covers that class.
- The guarantee enters the lock with the port onto main's v2 (PRF-371), not from this branch.

## Runs

`verity lean try FILE --on auto`, each `lake build +MODULE:deps` then `lake env lean FILE` with the axiom probe.

| Run | File | Tree | rc | Probe |
|---|---|---|---|---|
| `r20261110-163654-4fe7` | `Window/Assembled.lean` | `a8013ac42d` | 0 | 3 theorems, standard axioms |
| `r20261111-021917-5b7e` | `Window/Record.lean` | `66aa92551e` | 0 | standard axioms |
| `r20261111-022013-9626` | `Window/Counts.lean` | `9e8f2ea265` | 0 | standard axioms |
| `r20261111-025914-f994` | `Window/Closing.lean` | `66a4360c93` | 0 | standard axioms |
| `r20261111-033352-9065` | `Window/Steps.lean` | `c8e650a71b` | 1 | 3 rewrite errors in `memberSteps_of_sessions`, fixed in `c2bda01148` |

Earlier rows (the layers up to `Window/Assembled.lean`, and the failed tries fixed forward) are in the git history of
this file.

## Branch name

The cloud instructions name `cursor/verity-soundness-8fb2`; the brief names `cursor/soundness-multileg-8fb2`. This
work is on the brief's.
