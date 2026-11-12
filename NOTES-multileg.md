# The sampled leg over a window of m members

Scratch notes for PRF-490, on `cursor/soundness-multileg-8fb2` only. This file never goes in the PR.

Base: `cursor/soundness-remote-deployed-8fb2` (#8569, `remote_soundness_deployedReady`, the one-member form). Everything
is in namespace `Verity.Sampled`, package `security_proofs`, under
`security/Proofs/Properties/Verity/Sampled/` (`Window.lean` and `Window/*.lean`), registered in
`security/Proofs/lakefile.toml` and `security/Proofs/lean-audit.json`. One worker (LEG).

## The statement

`Window/Instance.lean`:

```lean
theorem remote_soundness_window (I : WindowInstance d α) (K₀ : WindowConstructions I.base)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hC : I.ClaimReads (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest))
    (hRg : I.RegisteredFiles) (hWa : I.WindowAt .remote) (hWR : I.WindowReads .remote ev filesDec)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest)).SoundnessAt .remote
```

Same shape as the one-member `remote_soundness_deployedAt` (`Sampled/Instance.lean`): `RemoteSoundness`
(`remote_soundness_of_subprotocols`) with its sampled leg `WindowInstance.spotSound`, which is
`windowSpotSound_sessions` at the deployed vocabulary. The sampled leg's error at a window is `Window.delta`:
`W.drawMiss + (W.joint t).eps` (`JointShareV5`: the one sample's own miss and `Σ_k (ε_ks,k + ε_bind,k)` at member
`k`'s part of the one sample), each member's roots term (`Window.rootsErr`: the calls of the member's
`rootsFinderL` over `2^256`, one a member, drawn or not, its finder listed under `ecr/sha-512`), one two-continuation term per round of every member's session
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
| `hC : I.ClaimReads …` | what the claim's transfer reads of the window: `laid` (`Window.Laid`, the members in unit order), `share` (`(certOf θ).ε = W.ε`), `units` (`units θ F = W.nJ`), and, under `Window.RootsFix`, `reads` (`JointAuditV5.ReadsAt`: the vocabulary's read is the members' files in order, each member's files pass the files check, the openings hold where a member's commitment opens, the extracted transcript is each member's at `memberX`) | **open**: the per-member verifier's, with `SetsUp`. `ClaimTransfers` itself is proved from it (`WindowInstance.claimTransfers`, on `UnboundWrong.card_le`) |
| `hW`, `hI` | PoUW's and the inference-only policy's targets at the base | the other legs' (as in the one-member form) |

Trust base when those close: SHA512CR, uniform randomness, the honest verifier server's log premises;
LmsUnforgeable remote only. Nothing else is introduced.

## What is left

1. **`ClaimReads` from the per-member verifier.** `laid` from `laidHeld`; `share` and `units` from the parameters
   and the committed files; `reads` from the files check on every member, drawn or not. That check is not in this
   branch's verifier: `runMembers` checks only the members a statement names. It goes in
   `Spot.OneStageZk.membersRun`, after `rootsOf` and before `runMembers`, over every member of `ms` (`m.statement`,
   then its `ksFiles` and
   `population`), public files only, O(descriptor). The δ already pays its roots clause (`Window.rootsErr`).
   The check and its property exist on proofs' branches, not here (`membersRun_files`: every member has
   `Member.FilesHeld`; `Member.FilesChecked → MemberV5.Checked`). They give `ReadAt.checked` inside `reads` for
   every member: `parsed` and `distinct` from `FilesHeld`'s ports and roots check and population check, `noPublic` by
   definition, `fixes` being `reads`' own hypothesis. Missing: the lemma from the verifier's `FilesHeld` to the
   window member's `FilesChecked.loads` and `.distinct` (the member's `I`, `D`, `pub`, `n` are the statement's), which
   is `SetsUp`'s to give. `read`, `opens`, `consts` and `trace` of `ReadAt`, and `laid`, `share`, `units`, are not
   theirs.
2. **`WindowReads` and `WindowAt` from the per-member verifier** (the port). `WindowSessions` now takes
   `members : MemberForm.Accepted x.log` (the log is the read of lines `membersTreeReason` accepts) in place of
   `order` and `specs`, which are theorems (`Accepted.opensLast`, `Accepted.specs`). The several-member path still
   refuses (`checkMembersAt_refuses`, `MULTI_REFUSED`).
3. **Every line of a member is ordered, at the log's lines.** `membersOrder` reads a tree's `coin-tree` line and its
   `tree-open` lines (`memberLine`), and refuses one of member k+1 before member k's closing round
   (`membersOrder_before_closing`, `memberLine_hello`). `OpensInOrder`, over the read's frames, still speaks of
   openings only: the same order on the read's `hello` frames is not stated.
4. **The roots term's form.** Each member's roots finder is a further finder of the vocabulary (`rootsSpec`,
   `rootsSpecs`, under `ecr/sha-512` at its calls), its premise `Member.RootsCR` in `error`'s condition and its term
   its calls over `2^256` (`Member.rootsErr`), the form `finderPr_le_crBound` gives the read finder.
   `Verity.sha512Msg = Claims.Sampling.sha512Msg` is `sha512Msg_eq`, by definition; the two definitions stay
   (the vocabulary's in `Properties/Verity/Game.lean`, the claims' in `Proofs/Flock/Claims/Sampling/Path.lean`). The
   one-statement leg's `Live.rootsErr` is still `ecrBound` at the claims' one.
5. **The window's own class** (`row-table` roots and `registered` values): refused in the per-member form; the
   rows link is proofs'.
6. `Window.hcls` stays a field of `Window` (`JointAuditV5.hcls_of_bounds` discharges it from bounds).
7. Bookkeeping: `tests/test_lean_vocabulary.py` warns that `Verity.Sampled.Window` declares `Window` again.

## Rulings in force (PRF-490)

- 19:35Z Nov 10 (soundness): the sessions are held one after another, ordered by the honest verifier server's log,
  the nonce chain only as a fallback; (B), a member at a time, is out.
- 00:48Z Nov 11 (soundness): the closing round is in the member's own coin tree (1026 rounds a member).
- 03:35Z Nov 11 (soundness): member m's roots are the `roots` entries whose `port` is `<m's key>/<port>`; a window
  with row tables is refused in the per-member form until the statement covers that class.
- 04:41Z Nov 11 (proofs): (a) for the window's class; the rows link is named open, proofs' piece (1).
- 04:43Z Nov 11 (soundness): shape (i), one circuit for the window; LEG writes the read's statement in the
  properties tree, keeps `ClaimTransfers` as the named hypothesis until proofs' assembly (i-a) lands, and posts the
  statement before the shared vocabulary (`Game.lean`'s `FileRead`) is touched. (ii), a member at a time, is out.
- 04:13Z Nov 11 (soundness): write to Linear only with `research msg linear comment ... --worker LEG`.
- The guarantee enters the lock with the port onto main's v2 (PRF-371), not from this branch.
- Nov 12 (soundness, the WINDOW brief): `Circuit.beside` from `Circuit.append`; the place at a window is each
  member's `placeOfRead`, members' roots only; the files check runs on every member, drawn or not, and the window's
  δ pays a roots finder a member; the member-order refusal is the check's, and its acceptance gives `OpensInOrder`.

## Runs

`verity lean try FILE --on auto`, each `lake build +MODULE:deps` then `lake env lean FILE` with the axiom probe.

| Run | File | Tree | rc | Probe |
|---|---|---|---|---|
| `r20261110-163654-4fe7` | `Window/Assembled.lean` | `a8013ac42d` | 0 | 3 theorems, standard axioms |
| `r20261111-021917-5b7e` | `Window/Record.lean` | `66aa92551e` | 0 | standard axioms |
| `r20261111-022013-9626` | `Window/Counts.lean` | `9e8f2ea265` | 0 | standard axioms |
| `r20261111-025914-f994` | `Window/Closing.lean` | `66a4360c93` | 0 | standard axioms |
| `r20261111-033352-9065` | `Window/Steps.lean` | `c8e650a71b` | 1 | 3 rewrite errors in `memberSteps_of_sessions`, fixed in `c2bda01148` |
| `r20261111-040437-c918` | `Window/Steps.lean` | `c2bda01148` | 0 | 26 theorems, standard axioms |
| `r20261111-040939-3980` | `Window/Read.lean` | `e5aad38d65` | 2 | infra: no dependencies on the node, nothing ran |
| `r20261111-041449-343d` | `Window/Read.lean` | `e5aad38d65` | 0 | 7 theorems, standard axioms |
| `r20261111-043407-3b49` | `Window/Instance.lean` | `38b78daff7` | 0 | 14 theorems, standard axioms (`remote_soundness_window` among them) |
| `r20261111-051805-5fe9` | `Properties/Verity/WindowFiles.lean` | `41e2d5a6bb` | cancelled | superseded before its check started (`beside` made reducible) |
| `r20261111-052756-d9a8` | `Properties/Verity/WindowFiles.lean` | `55b900da74` | 0 | 4 theorems, standard axioms |
| `r20261111-052756-c92d` | `Proofs/Properties/Verity/WindowFiles.lean` | `55b900da74` | 2 | infra: no dependencies on the node, nothing ran |
| `r20261111-053108-072a` | `Proofs/Properties/Verity/WindowFiles.lean` | `a1f057c487` | 0 | 34 theorems, standard axioms |
| `r20261112-095016-237c` | `Window/Instance.lean` | head `7c5571d97e` less six reflowed lines | 0 | 21 theorems, standard axioms (`remote_soundness_window` among them) |
| `r20261112-095017-87ae` | `Sampled/OpensAt.lean` | the same | 0 | 15 theorems, standard axioms |

Records (`lean_changed.py --records security/Properties --update`, the fast path: no kernel replay):

| Run | Tree | rc | Per package |
|---|---|---|---|
| `r20261110-191204-f7c8` | `ece623c082` | 1 | `security/Proofs` pass; the base's own failures elsewhere |
| `r20261111-053739-8e18` | `a1f057c487` | 1 | `security/Proofs` **pass** (193583 declarations in 3823 modules, standard axioms, the package's roots built: no name clash). `security/Properties` fail: its build passes; the 4 findings are `trees:` on the base's `remote_soundness_deployedReady` and `_honest`, whose statements read `Proofs.Properties.Verity.Sampled*` definitions. `security` fail: the base's `Flock.SecurityProofs.SampledKSLive*` records (`guarantees:` restated, `witnesses:`). No finding names a module of this branch; against the base branch this branch changes only `Properties/Verity/WindowFiles.lean` and the two root lists outside `security/Proofs`. No record was kept, so there is none to commit. |
| `r20261112-102444-bf1e` | `7c5571d97e` | 1 | `verity/protocols/spot/verifier/lean` **pass** (5629 declarations in 50 modules); its requirers failed on the lock (next rows) |
| `r20261112-102444-788a` | `7c5571d97e` | 1 | `security/Proofs` **pass** (197575 declarations in 3930 modules). `security/Properties` fail: `layers` (Game imports WindowFiles, WindowFiles imports `Definitions.Core.Compiler`). `security` fail: `reads` (WindowFiles outside `meaning`) and record differences |
| `r20261112-111216-22a7` | `503ea26f8a` (the lock's layer and meaning) | 0 | `security/Proofs` pass; `security/Properties` pass, record regenerated; `security` record regenerated (every failure a record difference). Committed in `ae16d2211c` |
| `r20261112-120421-3642` | `349f7d16b3` (no `--update`) | 0 | `security/Properties`, `security` and `security/Proofs` pass on the committed records |

Suites (`suites.py --changed BASE --quick`), at `7c5571d97e`; the last row at `349f7d16b3`:

| Run | Base | Verdict |
|---|---|---|
| `r20261112-102450-4e71` | `1a3eadc9c5` (the merge base with main) | 73 suites: 73 passed, 0 failed |
| `r20261112-103631-4015` | `e67c8c956f` (the branch before this work) | 68 suites: 68 passed, 0 failed |
| `r20261112-120426-2e1e` | `7c5571d97e` (the lock's commits after it) | 9 suites: 9 passed, 0 failed |

Earlier rows (the layers up to `Window/Assembled.lean`, and the failed tries fixed forward) are in the git history of
this file.

## Branch name

The cloud instructions name `cursor/verity-soundness-8fb2`; the brief names `cursor/soundness-multileg-8fb2`. This
work is on the brief's.
