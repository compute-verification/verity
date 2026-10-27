# E2E chains per protocol

The detail behind `examples/prove_path/README.md`'s "What an accepted plain audit means": lean's CHAINMAP status
for every protocol's end-to-end chain, and the premises locked statements take from `Proofs/`. The page keeps 2a's
and 2b's chains; this file keeps the rest, which changes with each status update.

**CHAINMAP's status per protocol.** lean's CHAINMAP maps each protocol's e2e chain layer by layer: the session's KS,
then sampling, then the audit, then the accounting bound. It reads the lock's 899 guarantees and a dependency walk
(note:lean/20261025T0420Z-report-e2e-chains, run r20261025-032006-79c2, at #3627's head 85bd0f9e26). The status column
is CHAINMAP's as of that note. The last column gives the refactor item that changes it, with the owner and date proofs
posted on 25 Oct at 06:17Z, and what PRs have done since.

| Protocol | Config | Status (CHAINMAP) | Item, owner, date; open PRs |
|---|---|---|---|
| Sampled proofs / C-Flock | `--zk` | Proved. `sampling_assembled` is not a lock guarantee | item 1, FINTAU, landed: #4037 makes `sampling_assembled` the `deployZk` instance of `sampling_assembled_of`, and #4075 gives one deployed record over any Flock record (`sampling_deployed_of`) |
| | plain | Proved in the game form. No sampling theorem at plain in claims form | item 1, FINTAU, landed: #4037's `sampling_assembled_plain`, at the trimmed record. Flock's KS at the untrimmed record is on main (#4081), without a sampling instance there yet. The game-to-claims link is the broken shared row: E2EMAP's plan, 22:00Z 25 Oct |
| | recursive | Effectively none: proved, but near-vacuous. No KS-to-sampling for recursion | item 12, REC100 and RESLOW on the RecursiveKS track, Fri 30 Oct, after Daniel's vcalls ruling (default A on 30 Oct). Item 1's recursive instance follows it |
| PoUW | `--zk` | Chain broken at KS → A: `SessionSoundAt` needs `AcceptedLive`, which no served path meets yet | item 9, proofs (ZKFIN, after FinishIs): `SessionSoundAt` to `ProofSoundAt` (δs + δ), with the TS headlines its instances, Tue 27 Oct. `AcceptedLive` on a served path needs FinishIs (Mon 26 Oct) and a served window whose lanes all verify (PRF-131's first pass verified 9 of 32): Wed 28 Oct |
| | plain, recursive | none | none posted |
| PoUS | all | none (ideal model only) | item 11, E2EMAP, over item 1 as a hypothesis: Thu 29 Oct |
| Bandwidth (over PoUS's store) | all | none with a session layer | item 11, E2EMAP: Thu 29 Oct |
| Remote bandwidth | all | none with a session layer | item 11, E2EMAP: Thu 29 Oct |
| Network certifier | all | none: `SampledAuditAccepted` stated, unproved | item 10, FINTAU: proved on main (#4054), and listed in the lock since #4090, which landed #4083's lock entry. The leak step is proved and listed on main (#4055, #4044), at the registered inputs pinned since #4090 (option A, NET-68's P1a): c = #(`freeInputs` \ `reg`). P1b, that the trace's values at `reg` are the registration's, is on proofs' branches, with no PR yet |
| NCI | all | none with a session layer; it rests on the assumption `PouwSound`, which no PoUW guarantee instantiates | item 11's NCI part: `PouwSound` replaced by item 9's e2e, Fri 30 Oct, after item 9 |
| Accounting (top layer) | — | Already modular: the pattern every lower layer should copy | — |

Items 1 and 9–12 are the statement-changing steps of the note's refactor sketch: KS to sampling once (1), PoUW's headline
from `SessionSoundAt` (9), the certifier's property (10), PoUS, bandwidth, remote bandwidth and NCI over sampling (11),
and recursion's knowledge form (12). The rest (2–6, 8 and 13) are E2EMAP's; item 2 is #4043. E2EMAP turns the map into
a refactor plan by 22:00Z 25 Oct, with the first refactor PR marked Mon 26 Oct.

**Premises in `Proofs/`.** `trees.except` (`Security/lean-audit.json`) exempts 33 module prefixes from the tree rule
until C-Flock's spec is extracted (RECKSLOCK, Wed 4 Nov). 31 are C-Flock's; the other two, `Proofs.Accounting.TS.Session`
and `Proofs.Pouw.PearlC.FlockTile`, are on PoUW's chain. None is PoUS's, bandwidth's, remote bandwidth's, the
certifier's, NCI's or accounting's, so this is a C-Flock question, plus one PoUW definition.
- **lean's list** (run r20261025-031300-0fcc, in CHAINMAP's note): 40 `Prop`-valued definitions under `trees.except`
  that a locked statement takes without a lock entry. 39 are C-Flock's, and one, `SoundAtBridge.RowsSession.CanonRead`,
  is PoUW's. CHAINMAP's list was due by 12:00Z 25 Oct and came out at 04:20Z.
- **REROWS's first sort** (note:proofs/20261025T0353Z-report-premise-audit) gives each of the 40 a verdict: 10 premises
  and 30 conclusions.
  - The premises are C-Flock's:
    - the plain custody (`DecodeZL.RecordsLive`, `DecodeZL.DrawOk`, `HiddenDec.DrawOkH`);
    - the `--zk` custody (`ZkLink.RecordsLiveZK`, `ZkLink.DrawOkZK`, `ZkHidden.DrawOkZKH`, `Composed.RecordsLiveZKJ`);
    - plain's finder hash bounds (`Exec.LinkCRL`, `Exec.SessCRL`);
    - recursion's `RecDeploy.PlainLeavesAdmitted`.
  - The conclusions are acceptance events and the verifier's checks, events a bound measures, antecedents inside a
    conclusion, and fields of setups built from the verifier's own. PoUW's `CanonRead` is one of them, a selector.
  - It also found premises the scan doesn't print: recursion's draw custody (`ZkOuterD.hDraw`, `ZkOuterR.hDraw`), and
    `--zk`'s `TableCRZ` and `LinkCRZC`.
- **Form conditions**, top's third class: `StmtWF` and `RegionsWF` (26 guarantees each), `PadOK` (29), and perhaps
  `isRow` and `isRead`. The first sort counts them as conclusions: the first three are fields of setups that
  `drawSetup*_of` builds from flock-verify's setup, and the last two are selectors. The classification is due by
  15:00Z 25 Oct (proofs). REROWS's four-class sort (conclusion, premise, form condition, well-formedness), which says
  whether the compiled verifier checks each form condition, is due by 22:00Z. A form condition the verifier doesn't
  check comes to this page as a gap.
- **Moved and listed:** every premise is due moved and listed by 22:00Z 25 Oct (proofs).
  - On main: the plain custody (`RecordCustodyExec`, `…ExecH`, #4048) and the `--zk` custody
    (`Zk.RecordCustodyZKLaw` and its forms, #4067).
  - #4107 (open) names the rest: `TableCRAt`, `LinkCRAt`, `TableCRZAt`, `Zk.RecordsCustodyZKJ`, `Zk.DrawCustodyZKHJ`,
    and `PlainLeavesAdmitted` moved.
