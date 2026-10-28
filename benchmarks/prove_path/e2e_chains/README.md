# E2E chains per protocol

The detail behind `benchmarks/prove_path/README.md`'s "What an accepted plain audit means": the status of every
protocol's end-to-end chain, and the premises locked statements take from `Proofs/`. The page keeps 2a's and 2b's
chains; this file keeps the rest, which changes with each status update.

**Status per protocol.** Each protocol's e2e chain, layer by layer: the session's KS, then sampling, then the audit,
then the accounting bound, as the lock's guarantees and a walk of their dependencies give it. The last column gives the
refactor item that changes it, and what has landed since.

| Protocol | Config | Status | Refactor item; what has landed |
|---|---|---|---|
| Sampled proofs / C-Flock | `--zk` | Proved. `sampling_assembled` is not a lock guarantee | item 1, landed: `sampling_assembled` is the `deployZk` instance of `sampling_assembled_of`, and `sampling_deployed_of` gives one deployed record over any Flock record |
| | plain | Proved in the game form. No sampling theorem at plain in claims form | item 1, FINTAU, landed: #4037's `sampling_assembled_plain`, at the trimmed record. Flock's KS at the untrimmed record is on main (#4081), without a sampling instance there yet. The game-to-claims link is the broken shared row: E2EMAP's plan, 22:00Z 25 Oct |
| | recursive | Effectively none: proved, but near-vacuous. No KS-to-sampling for recursion | item 12, REC100 and RESLOW on the RecursiveKS track, Fri 30 Oct, after Daniel's vcalls ruling (default A on 30 Oct). Item 1's recursive instance follows it |
| PoUW | `--zk` | Chain broken at KS → A: `SessionSoundAt` needs `AcceptedLive`, which no served path meets yet | item 9, proofs (ZKFIN, after FinishIs): `SessionSoundAt` to `ProofSoundAt` (δs + δ), with the TS headlines its instances, Tue 27 Oct. `AcceptedLive` on a served path needs FinishIs (Mon 26 Oct) and a served window whose lanes all verify (PRF-131's first pass verified 9 of 32): Wed 28 Oct |
| | plain, recursive | none | none posted |
| PoUS | all | none (ideal model only) | item 11, over item 1 as a hypothesis |
| Bandwidth (over PoUS's store) | all | none with a session layer | item 11 |
| PoUB (over a node's registered store) | all | none with a session layer | item 11 |
| Network certifier | all | none: `SampledAuditAccepted` is proved and listed in the lock, but it takes `Claims.Sampling.KS` as a hypothesis, and only `sampling_assembled`, not a lock guarantee, gives it | item 10: proved on main and listed in the lock. The leak step is proved and listed on main, at the registered inputs: c = #(`freeInputs` \ `reg`). That the trace's values at `reg` are the registration's is not proved on main |
| NCI | all | none with a session layer; it rests on the assumption `PouwSound`, which no PoUW guarantee instantiates | item 11's NCI part: `PouwSound` replaced by item 9's e2e, after item 9 |
| Remote (composite: sampled proofs and PoUW) | `--zk` | none: its composed theorem, `Verity.Soundness`, is not in Lean; its sampled-proofs half reads one-stage v1, which takes the sessions' verdicts as evidence, and its PoUW half is broken at the proof layer (above) | `Verity.Soundness`, after the Lean dedupe (owner soundness) |
| Accounting (top layer) | — | Already modular: the pattern every lower layer should copy | — |

Items 1 and 9–12 are the refactor's statement-changing steps: KS to sampling once (1), PoUW's headline from
`SessionSoundAt` (9), the certifier's property (10), PoUS, bandwidth, PoUB and NCI over sampling (11), and
recursion's knowledge form (12). The rest (2–6, 8 and 13) are the refactor plan's other steps.

**Premises in `Proofs/`.** `trees.except` (`Security/lean-audit.json`) exempts 33 module prefixes from the tree rule
until C-Flock's spec is extracted (RECKSLOCK, Wed 4 Nov). 31 are C-Flock's; the other two, `Proofs.Accounting.TS.Session`
and `Proofs.Pouw.PearlC.FlockTile`, are on PoUW's chain. None is PoUS's, bandwidth's, remote bandwidth's, the
certifier's, NCI's or accounting's, so this is a C-Flock question, plus one PoUW definition.
- **lean's list** (run r20261025-031300-0fcc, in CHAINMAP's note): 40 `Prop`-valued definitions under `trees.except`
  that a locked statement takes without a lock entry. 39 are C-Flock's, and one, `SoundAtBridge.RowsSession.CanonRead`,
  is PoUW's.
- **A first sort** gives each of the 40 a verdict: 10 premises and 30 conclusions.
  - The premises are C-Flock's:
    - the plain custody (`DecodeZL.RecordsLive`, `DecodeZL.DrawOk`, `HiddenDec.DrawOkH`);
    - the `--zk` custody (`ZkLink.RecordsLiveZK`, `ZkLink.DrawOkZK`, `ZkHidden.DrawOkZKH`, `Composed.RecordsLiveZKJ`);
    - plain's finder hash bounds (`Exec.LinkCRL`, `Exec.SessCRL`);
    - recursion's `RecDeploy.PlainLeavesAdmitted`.
  - The conclusions are acceptance events and the verifier's checks, events a bound measures, antecedents inside a
    conclusion, and fields of setups built from the verifier's own. PoUW's `CanonRead` is one of them, a selector.
  - It also found premises the scan doesn't print: recursion's sample custody (`ZkOuterD.hDraw`, `ZkOuterR.hDraw`), and
    `--zk`'s `TableCRZ` and `LinkCRZC`.
- **Form conditions**, a third class: `StmtWF` and `RegionsWF` (26 guarantees each), `PadOK` (29), and perhaps
  `isRow` and `isRead`. The first sort counts them as conclusions: the first three are fields of setups that
  `drawSetup*_of` builds from the Lean verifier's setup, and the last two are selectors. A four-class sort (conclusion,
  premise, form condition, well-formedness) says whether the compiled verifier checks each form condition. A form
  condition the verifier doesn't check comes to this page as a gap.
- **Moved and listed:** on main, these premises are named assumptions listed in the lock: the plain custody
  (`RecordCustodyExec`, `…ExecH`), the `--zk` custody (`Zk.RecordCustodyZKLaw` and its forms), `TableCRAt`,
  `LinkCRAt`, `TableCRZAt`, `Zk.RecordsCustodyZKJ`, `Zk.DrawCustodyZKHJ` and `RecDeploy.PlainLeavesAdmitted`.
